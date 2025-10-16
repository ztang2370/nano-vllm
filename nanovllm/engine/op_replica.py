"""
Operator-level replication for inference operators.

This module provides a system for creating and managing replicas of inference operators
across multiple GPUs to handle bursty workloads. Each replica runs on a dedicated
CUDA stream with duplicated weights.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.cuda

from nanovllm.layers.linear import RowParallelLinear
from nanovllm.utils.context import set_context

logger = logging.getLogger(__name__)


@dataclass
class ReplicaMetrics:
    """Metrics for a single operator replica."""
    in_flight_count: int = 0
    total_requests: int = 0
    busy: bool = False
    last_used: float = 0.0


@dataclass
class ReplicaConfig:
    """Configuration for operator replicas."""
    num_replicas_per_device: int = 1
    replica_devices: List[int] = field(default_factory=list)  # Empty means auto-detect


class OperatorReplica:
    """
    A replica of an inference operator that runs on a dedicated CUDA device and stream.

    Each replica maintains its own copy of weights, allowing parallel
    execution of the same operator on different devices/streams.
    """

    def __init__(
        self,
        op_class: type,
        weight_config: Union[Dict[str, torch.Tensor], Dict],  # Can be weights dict or config dict
        device: Union[int, torch.device],
    ):
        """
        Initialize an operator replica.

        Args:
            op_class: The operator class to instantiate
            weight_config: Weights/config to duplicate to this device
            device: CUDA device for this replica
        """
        self.device = torch.device(f"cuda:{device}") if isinstance(device, int) else device
        self.op_class = op_class

        # Set device context and create stream
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.Stream(device=self.device)

        # Handle different weight formats and operator types
        if isinstance(weight_config, dict) and 'weights' in weight_config:
            # New format with config and weights
            config = weight_config
            self.weights = {}
            for name, weight in config['weights'].items():
                self.weights[name] = weight.to(self.device)
            # Store config for operator construction
            self.op_config = {k: v for k, v in config.items() if k != 'weights' and k != 'module'}
        else:
            # Legacy format - just weights dict
            self.weights = {}
            for name, weight in weight_config.items():
                self.weights[name] = weight.to(self.device)
            self.op_config = {}

        # Create the appropriate operator based on the class
        if str(self.op_class.__name__) == 'RowParallelLinear':
            # Extract linear parameters
            input_size = self.op_config.get('input_size', 1024)
            output_size = self.op_config.get('output_size', 1024)
            bias = self.op_config.get('bias', False)

            # Create RowParallelLinear instance
            self.op = RowParallelLinear(input_size, output_size, bias=bias)
            # Load the weights by directly assigning them and converting dtype
            with torch.no_grad():
                for name, param in self.op.named_parameters():
                    if name in self.weights:
                        # Copy the weight data and convert to the weight's dtype
                        param.data = self.weights[name].clone()
            # Ensure the operator is on the correct device
            self.op = self.op.to(self.device)
        else:
            # Fallback dummy operator
            class DummyOp:
                def forward(self, *args, **kwargs):
                    return args[0] if args else None
            self.op = DummyOp()

        # Use the current stream (same as main execution) to avoid stream issues
        self.stream = torch.cuda.current_stream(self.device)

        # Metrics and state
        self.metrics = ReplicaMetrics()
        self._lock = threading.Lock()

        logger.info(f"Created operator replica on device {self.device} with dedicated stream")

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
            # Transfer context tensors to replica device if they exist
            context_dict = context.__dict__.copy()
            for key, value in context_dict.items():
                if isinstance(value, torch.Tensor):
                    context_dict[key] = value.to(self.device)
            set_context(**context_dict)

        # Execute on dedicated replica stream for true parallelism
        with torch.cuda.stream(self.stream):
            output = self.op.forward(*args_device, **kwargs)

        # Record completion event on replica stream
        completion_event = torch.cuda.Event()
        completion_event.record(self.stream)

        # Schedule callback to update metrics when done
        def _update_metrics():
            with self._lock:
                self.metrics.in_flight_count -= 1
                self.metrics.total_requests += 1
                self.metrics.busy = self.metrics.in_flight_count > 0

        # Use event callback to update metrics (non-blocking)
        completion_event.synchronize()  # Wait briefly to ensure event is recorded
        _update_metrics()  # For now, update immediately (can be made async later)

        return output, completion_event

    def synchronize(self) -> None:
        """Synchronize this replica's stream."""
        self.stream.synchronize()



class ReplicaManager:
    """
    Manages a pool of operator replicas across multiple devices.

    Provides load balancing and lifecycle management for operator replicas.
    """

    def __init__(self, config: ReplicaConfig):
        self.config = config
        self.replicas: Dict[str, List[OperatorReplica]] = {}
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._initialized = False  # Flag to indicate when replicas are ready

    def create_replicas(
        self,
        op_name: str,
        op_class: type,
        weight_config: Union[Dict[str, torch.Tensor], Dict],
        devices: List[int],
        num_replicas_per_device: Optional[int] = None,
    ) -> None:
        """
        Create replicas for an operator across the specified devices.

        Args:
            op_name: Name identifier for the operator (e.g., "attention")
            op_class: The operator class to replicate
            weight_config: Weights/config to duplicate to each device
            devices: List of CUDA device indices
            num_replicas_per_device: Number of replicas per device (overrides config)
        """
        num_per_device = num_replicas_per_device or self.config.num_replicas_per_device

        with self._lock:
            self.replicas[op_name] = []

            for device_idx in devices:
                for replica_idx in range(num_per_device):
                    try:
                        replica = OperatorReplica(
                            op_class=op_class,
                            weight_config=weight_config,
                            device=device_idx,
                        )
                        self.replicas[op_name].append(replica)
                        logger.info(f"Created {op_name} replica {replica_idx} on device {device_idx}")
                    except Exception as e:
                        logger.error(f"Failed to create {op_name} replica on device {device_idx}: {e}")
                        raise

            self._initialized = True

    def get_replicas_for_batch_split(self, op_name: str, layer_id: Optional[int] = None, batch_size: int = 1) -> List[OperatorReplica]:
        """
        Get all available replicas for batch splitting - distribute batch across all replicas.

        Args:
            op_name: Name of the operator
            layer_id: Optional layer ID for layer-specific replicas
            batch_size: Size of the batch to split

        Returns:
            List of replicas to use for parallel processing, or empty list if none available
        """
        # Try layer-specific replica first, then fall back to general replica
        candidate_names = []
        if layer_id is not None:
            candidate_names.append(f"{op_name}_layer_{layer_id}")
        candidate_names.append(op_name)

        selected_op_name = None
        replicas = None
        for candidate in candidate_names:
            if candidate in self.replicas and self.replicas[candidate]:
                selected_op_name = candidate
                replicas = self.replicas[candidate]
                break

        if not replicas:
            return []

        # For batch splitting, use all available replicas
        print(f"BATCH_SPLIT: Splitting batch of size {batch_size} across {len(replicas)} {selected_op_name} replicas")
        return replicas

    def add_original_operator_replica(self, op_name: str, original_module) -> None:
        """
        Add the original operator module as a special "replica" so it can be utilized
        for load balancing during bursty requests. For stateless operators like linear layers.

        Args:
            op_name: Name of the operator
            original_module: The original operator module instance
        """
        with self._lock:
            if op_name not in self.replicas:
                logger.warning(f"No replicas exist for {op_name}, cannot add original operator replica")
                return

            # Create a simple wrapper for the original stateless operator
            class OriginalOperatorReplica:
                def __init__(self, original_module):
                    self.device = original_module.weight.device if hasattr(original_module, 'weight') else torch.device('cuda:0')
                    self.original_module = original_module
                    self.metrics = ReplicaMetrics()
                    self._lock = threading.Lock()
                    # Use the current stream (same as main execution) to avoid stream issues
                    self.stream = torch.cuda.current_stream(self.device)

                def forward_async(self, *args, **kwargs):
                    """Execute the original operator."""
                    with self._lock:
                        self.metrics.in_flight_count += 1
                        self.metrics.busy = self.metrics.in_flight_count > 0
                        self.metrics.last_used = time.time()

                    # Execute on dedicated replica stream for true parallelism
                    with torch.cuda.stream(self.stream):
                        output = self.original_module(*args, **kwargs)

                    # Create event for synchronization on replica stream
                    event = torch.cuda.Event()
                    event.record(self.stream)

                    # Update metrics after completion
                    with self._lock:
                        self.metrics.in_flight_count -= 1
                        self.metrics.total_requests += 1
                        self.metrics.busy = self.metrics.in_flight_count > 0

                    return output, event

                def synchronize(self):
                    """Synchronize the operator's stream."""
                    torch.cuda.synchronize(self.device)

            # Add the original operator replica to the pool
            original_replica = OriginalOperatorReplica(original_module)
            self.replicas[op_name].append(original_replica)
            logger.info(f"Added original operator module as replica for {op_name}")


    def shutdown(self) -> None:
        """Shutdown all replicas and cleanup resources."""
        logger.info("Shutting down replica manager...")

        self._shutdown_event.set()

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
