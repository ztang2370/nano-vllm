"""
Operator-level replication for inference operators.

This module provides a system for creating and managing replicas of inference operators
(e.g., Attention) across multiple GPUs to handle bursty workloads. Each replica runs
on a dedicated CUDA stream with duplicated weights and workspace.
"""

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.cuda
from dataclasses import dataclass, field
from queue import Queue
import weakref

logger = logging.getLogger(__name__)


@dataclass
class ReplicaMetrics:
    """Metrics for a single operator replica."""
    in_flight_count: int = 0
    total_requests: int = 0
    avg_latency_ms: float = 0.0
    busy: bool = False
    last_used: float = 0.0


@dataclass
class ReplicaConfig:
    """Configuration for operator replicas."""
    num_replicas_per_device: int = 1
    replica_devices: List[int] = field(default_factory=list)  # Empty means auto-detect
    auto_scaling_enabled: bool = False
    auto_scaling_interval_ms: int = 200
    queue_high_threshold: int = 10
    queue_low_threshold: int = 2
    max_idle_time_ms: int = 30000  # 30 seconds


class OperatorReplica:
    """
    A replica of an inference operator that runs on a dedicated CUDA device and stream.

    Each replica maintains its own copy of weights and workspace, allowing parallel
    execution of the same operator on different devices/streams.
    """

    def __init__(
        self,
        op_class: type,
        weights_cpu: Union[Dict[str, torch.Tensor], Dict],  # Can be weights dict or config dict
        device: Union[int, torch.device],
        workspace_factory: Optional[Callable[[torch.device], Dict[str, torch.Tensor]]] = None,
    ):
        """
        Initialize an operator replica.

        Args:
            op_class: The operator class to instantiate (e.g., Attention)
            weights_cpu: CPU weights/config to duplicate to this device
            device: CUDA device for this replica
            workspace_factory: Function to create workspace tensors for this device
        """
        self.device = torch.device(f"cuda:{device}") if isinstance(device, int) else device
        self.op_class = op_class

        # Set device context and create stream
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.Stream(device=self.device)

        # Handle different weight formats
        if isinstance(weights_cpu, dict) and 'weights' in weights_cpu:
            # New format with config and weights
            config = weights_cpu
            self.weights = {}
            for name, weight in config['weights'].items():
                self.weights[name] = weight.to(self.device)
            # Store config for operator construction
            self.op_config = {
                'num_heads': config['num_heads'],
                'head_dim': config['head_dim'],
                'scale': config['scale'],
                'num_kv_heads': config['num_kv_heads']
            }
        else:
            # Legacy format - just weights dict
            self.weights = {}
            for name, weight in weights_cpu.items():
                self.weights[name] = weight.to(self.device)
            self.op_config = {}

        # Create workspace if factory provided
        self.workspace = workspace_factory(self.device) if workspace_factory else {}

        # Use the original Attention class which now has replica support
        if self.op_config:
            # Extract attention parameters
            num_heads = self.op_config.get('num_heads', 16)
            head_dim = self.op_config.get('head_dim', 128)
            scale = self.op_config.get('scale', 1.0 / (head_dim ** 0.5))
            num_kv_heads = self.op_config.get('num_kv_heads', num_heads)

            # Import here to avoid circular imports
            from nanovllm.layers.attention import Attention

            # Create Attention instance - it will detect replica mode automatically
            self.op = Attention(num_heads, head_dim, scale, num_kv_heads, weights=self.weights)
        else:
            # Fallback dummy operator
            class DummyOp:
                def forward(self, q, k, v):
                    return q
            self.op = DummyOp()

        # Metrics and state
        self.metrics = ReplicaMetrics()
        self._lock = threading.Lock()

        logger.info(f"Created operator replica on device {self.device}")

    def forward_async(
        self,
        *args,
        context=None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.cuda.Event]:
        """
        Execute the operator forward pass asynchronously on this replica's stream.

        Args:
            *args: Positional arguments for the operator forward
            **kwargs: Keyword arguments for the operator forward

        Returns:
            Tuple of (output_tensor, completion_event)
        """
        print(f"DEBUG: Additional replica forward_async called")
        with self._lock:
            self.metrics.in_flight_count += 1
            self.metrics.busy = True
            self.metrics.last_used = time.time()

        # Ensure inputs are on the correct device (async transfer if needed)
        args_device = []
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                print(f"DEBUG: Replica input {i}: device={arg.device}, dtype={arg.dtype}, shape={arg.shape}")
                if arg.device != self.device:
                    arg_device = arg.to(self.device, non_blocking=True)
                    args_device.append(arg_device)
                else:
                    args_device.append(arg)
            else:
                args_device.append(arg)

        # Record start event for latency measurement
        start_event = torch.cuda.Event()
        start_event.record()

        # Set context temporarily if provided (for flash_attn compatibility)
        if context is not None:
            from nanovllm.utils.context import set_context
            # Transfer context tensors to replica device if they exist
            context_dict = context.__dict__.copy()
            for key, value in context_dict.items():
                if isinstance(value, torch.Tensor):
                    context_dict[key] = value.to(self.device)
            set_context(**context_dict)

        # Execute on default stream for now (streams may cause issues with flash_attn)
        # TODO: Fix stream synchronization for flash_attn compatibility
        output = self.op.forward(*args_device, **kwargs)

        # Record completion event on default stream
        completion_event = torch.cuda.Event()
        completion_event.record()

        # Schedule callback to update metrics when done
        def _update_metrics():
            with self._lock:
                self.metrics.in_flight_count -= 1
                self.metrics.total_requests += 1
                if self.metrics.in_flight_count == 0:
                    self.metrics.busy = False

        # Use event callback to update metrics (non-blocking)
        completion_event.synchronize()  # Wait briefly to ensure event is recorded
        _update_metrics()  # For now, update immediately (can be made async later)

        return output, completion_event

    def synchronize(self) -> None:
        """Synchronize this replica's stream."""
        self.stream.synchronize()

    def is_idle(self, timeout_ms: int = 5000) -> bool:
        """Check if replica has been idle for the specified timeout."""
        with self._lock:
            return (time.time() - self.metrics.last_used) * 1000 > timeout_ms


class ReplicaManager:
    """
    Manages a pool of operator replicas across multiple devices.

    Provides load balancing, auto-scaling, and lifecycle management for
    operator replicas.
    """

    def __init__(self, config: ReplicaConfig):
        self.config = config
        self.replicas: Dict[str, List[OperatorReplica]] = {}
        self._round_robin_idx: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._initialized = False  # Flag to indicate when replicas are ready
        self._batch_replica = {}  # Cache replica per batch: {op_name: replica}

        # Auto-scaling components
        self.auto_scaler: Optional[ReplicaAutoScaler] = None
        if config.auto_scaling_enabled:
            self.auto_scaler = ReplicaAutoScaler(self, config)
            self.auto_scaler.start()

    def create_replicas(
        self,
        op_name: str,
        op_class: type,
        weights_cpu: Dict[str, torch.Tensor],
        devices: List[int],
        workspace_factory: Optional[Callable[[torch.device], Dict[str, torch.Tensor]]] = None,
        num_replicas_per_device: Optional[int] = None,
    ) -> None:
        """
        Create replicas for an operator across the specified devices.

        Args:
            op_name: Name identifier for the operator (e.g., "attention")
            op_class: The operator class to replicate
            weights_cpu: CPU weights to duplicate to each device
            devices: List of CUDA device indices
            workspace_factory: Factory function for per-device workspace
            num_replicas_per_device: Number of replicas per device (overrides config)
        """
        num_per_device = num_replicas_per_device or self.config.num_replicas_per_device

        with self._lock:
            print(f"DEBUG: Creating replicas for {op_name}, devices: {devices}, num_per_device: {num_per_device}")
            self.replicas[op_name] = []
            self._round_robin_idx[op_name] = 0

            for device_idx in devices:
                for replica_idx in range(num_per_device):
                    try:
                        replica = OperatorReplica(
                            op_class=op_class,
                            weights_cpu=weights_cpu,
                            device=device_idx,
                            workspace_factory=workspace_factory,
                        )
                        self.replicas[op_name].append(replica)
                        print(f"DEBUG: Created {op_name} replica {replica_idx} on device {device_idx}")
                        logger.info(f"Created {op_name} replica {replica_idx} on device {device_idx}")
                    except Exception as e:
                        logger.error(f"Failed to create {op_name} replica on device {device_idx}: {e}")
                        raise

            self._initialized = True

    def get_replica(self, op_name: str) -> Optional[OperatorReplica]:
        """
        Get an available replica for the operator using round-robin assignment.

        Args:
            op_name: Name of the operator

        Returns:
            An available replica, or None if no replicas exist
        """
        # Note: We don't use the lock here to avoid contention during inference
        # The replicas dict should be stable after initialization
        if op_name not in self.replicas or not self.replicas[op_name]:
            return None

        # Check if we already selected a replica for this batch
        if op_name in self._batch_replica:
            return self._batch_replica[op_name]

        replicas = self.replicas[op_name]

        # Use round-robin assignment for load balancing across batches
        idx = self._round_robin_idx[op_name]
        self._round_robin_idx[op_name] = (idx + 1) % len(replicas)

        replica = replicas[idx]
        # Cache this replica for the current batch
        self._batch_replica[op_name] = replica
        print(f"BATCH_ASSIGN: Batch -> attention replica {idx}")
        return replica

    def clear_batch_replicas(self):
        """Clear the batch replica cache after each batch is processed."""
        self._batch_replica.clear()

    def add_replica(self, op_name: str, device: int) -> bool:
        """
        Add a new replica for the operator on the specified device.

        Args:
            op_name: Name of the operator
            device: CUDA device index

        Returns:
            True if replica was added successfully
        """
        with self._lock:
            if op_name not in self.replicas:
                logger.warning(f"Cannot add replica for unknown operator: {op_name}")
                return False

            try:
                # Find an existing replica to copy weights/workspace from
                existing_replica = self.replicas[op_name][0] if self.replicas[op_name] else None
                if not existing_replica:
                    logger.error(f"No existing replicas to copy configuration from for {op_name}")
                    return False

                # Create new replica with same configuration
                replica = OperatorReplica(
                    op_class=existing_replica.op_class,
                    weights_cpu={},  # Will be populated from existing replica
                    device=device,
                    workspace_factory=None,  # Will be populated from existing replica
                )

                # Copy weights and workspace from existing replica
                replica.weights = {k: v.clone() for k, v in existing_replica.weights.items()}
                replica.workspace = {k: v.clone() for k, v in existing_replica.workspace.items()}
                replica.op = existing_replica.op_class(replica.weights)

                self.replicas[op_name].append(replica)
                logger.info(f"Added {op_name} replica on device {device}")
                return True

            except Exception as e:
                logger.error(f"Failed to add {op_name} replica on device {device}: {e}")
                return False

    def remove_idle_replica(self, op_name: str) -> bool:
        """
        Remove an idle replica for the operator.

        Args:
            op_name: Name of the operator

        Returns:
            True if a replica was removed successfully
        """
        with self._lock:
            if op_name not in self.replicas or len(self.replicas[op_name]) <= 1:
                # Don't remove the last replica
                return False

            replicas = self.replicas[op_name]
            idle_timeout_ms = self.config.max_idle_time_ms

            # Find an idle replica
            for i, replica in enumerate(replicas):
                if replica.is_idle(idle_timeout_ms) and not replica.metrics.busy:
                    try:
                        # Synchronize to ensure no pending work
                        replica.synchronize()

                        # Remove the replica
                        removed_replica = replicas.pop(i)
                        logger.info(f"Removed idle {op_name} replica from device {removed_replica.device}")
                        return True

                    except Exception as e:
                        logger.error(f"Failed to remove {op_name} replica: {e}")
                        continue

            return False

    def add_original_attention_replica(self, op_name: str, original_attention_module) -> None:
        """
        Add the original attention module as a special "replica" so it can be utilized
        for load balancing during bursty requests.

        Args:
            op_name: Name of the operator (should be "attention")
            original_attention_module: The original Qwen3Attention module instance
        """
        with self._lock:
            if op_name not in self.replicas:
                logger.warning(f"No replicas exist for {op_name}, cannot add original attention replica")
                return

            # Create a special OperatorReplica that replicates the original attention logic
            class OriginalAttentionReplica:
                def __init__(self, original_module):
                    self.device = original_module.attn.device if hasattr(original_module.attn, 'device') else torch.device('cuda:0')
                    self.original_module = original_module
                    self.metrics = ReplicaMetrics()
                    self._lock = threading.Lock()
                    self.scale = original_module.attn.scale

                    # Share KV cache with the original attention module
                    self.k_cache = original_module.attn.k_cache
                    self.v_cache = original_module.attn.v_cache

                def forward_async(self, q, k, v, context=None):
                    """Execute attention computation with proper KV cache management."""
                    print(f"DEBUG: Replica 0 forward_async called, q.shape={q.shape}")
                    with self._lock:
                        self.metrics.in_flight_count += 1
                        self.metrics.busy = True

                        # Set context if provided
                        if context is not None:
                            from nanovllm.utils.context import set_context
                            context_dict = context.__dict__.copy()
                            for key, value in context_dict.items():
                                if isinstance(value, torch.Tensor):
                                    context_dict[key] = value.to(self.device)
                            set_context(**context_dict)

                        # Call the original attention method directly (it will use the shared KV cache)
                        print(f"DEBUG: Replica 0 calling original attention")
                        o = self.original_module.attn(q, k, v)
                        print(f"DEBUG: Replica 0 attention completed, o.shape={o.shape}")

                        # Create event for synchronization
                        event = torch.cuda.Event()
                        event.record(torch.cuda.current_stream(self.device))

                        self.metrics.in_flight_count -= 1
                        self.metrics.busy = False

                        return o, event

                def synchronize(self):
                    """Synchronize the original attention's stream."""
                    torch.cuda.synchronize(self.device)

                def is_idle(self, timeout_ms):
                    """Check if this replica is idle."""
                    return not self.metrics.busy

            # Insert the original attention replica at the beginning of the list
            original_replica = OriginalAttentionReplica(original_attention_module)
            self.replicas[op_name].insert(0, original_replica)
            print(f"DEBUG: Added original attention as replica 0 for {op_name}")
            logger.info(f"Added original attention as replica 0 for {op_name}")

    def get_metrics(self, op_name: str) -> List[ReplicaMetrics]:
        """Get metrics for all replicas of an operator."""
        with self._lock:
            if op_name not in self.replicas:
                return []
            return [replica.metrics for replica in self.replicas[op_name]]

    def shutdown(self) -> None:
        """Shutdown all replicas and cleanup resources."""
        logger.info("Shutting down replica manager...")

        self._shutdown_event.set()

        if self.auto_scaler:
            self.auto_scaler.stop()

        with self._lock:
            for op_name, replicas in self.replicas.items():
                for replica in replicas:
                    try:
                        replica.synchronize()
                        logger.info(f"Synchronized {op_name} replica on device {replica.device}")
                    except Exception as e:
                        logger.error(f"Error synchronizing {op_name} replica: {e}")

            self.replicas.clear()

        logger.info("Replica manager shutdown complete")


class ReplicaAutoScaler(threading.Thread):
    """
    Background thread that monitors replica usage and automatically scales
    the number of replicas based on queue length and utilization.
    """

    def __init__(self, replica_manager: ReplicaManager, config: ReplicaConfig):
        super().__init__(daemon=True, name="ReplicaAutoScaler")
        self.replica_manager = weakref.ref(replica_manager)
        self.config = config
        self._stop_event = threading.Event()

        # Mock queue length monitoring (would be connected to actual scheduler)
        self._queue_lengths: Dict[str, int] = {}

    def update_queue_length(self, op_name: str, length: int):
        """Update the current queue length for an operator."""
        self._queue_lengths[op_name] = length

    def run(self):
        """Main auto-scaling loop."""
        while not self._stop_event.is_set():
            try:
                self._check_and_scale()
                time.sleep(self.config.auto_scaling_interval_ms / 1000.0)
            except Exception as e:
                logger.error(f"Error in replica auto-scaler: {e}")

    def _check_and_scale(self):
        """Check replica usage and scale as needed."""
        manager = self.replica_manager()
        if not manager:
            return

        for op_name in manager.replicas.keys():
            queue_len = self._queue_lengths.get(op_name, 0)
            metrics = manager.get_metrics(op_name)

            if not metrics:
                continue

            # Calculate aggregate metrics
            total_busy = sum(1 for m in metrics if m.busy)
            avg_in_flight = sum(m.in_flight_count for m in metrics) / len(metrics)

            # Scaling logic
            if queue_len > self.config.queue_high_threshold and avg_in_flight > 0.8:
                # High load - try to add replica
                available_devices = self._get_available_devices()
                if available_devices:
                    device = available_devices[0]  # Pick first available
                    if manager.add_replica(op_name, device):
                        logger.info(f"Auto-scaled: added {op_name} replica on device {device}")

            elif queue_len < self.config.queue_low_threshold and total_busy < len(metrics):
                # Low load - try to remove idle replica
                if manager.remove_idle_replica(op_name):
                    logger.info(f"Auto-scaled: removed idle {op_name} replica")

    def _get_available_devices(self) -> List[int]:
        """Get list of available CUDA devices (simple implementation)."""
        try:
            return list(range(torch.cuda.device_count()))
        except:
            return []

    def stop(self):
        """Stop the auto-scaling thread."""
        self._stop_event.set()
        self.join(timeout=5.0)

