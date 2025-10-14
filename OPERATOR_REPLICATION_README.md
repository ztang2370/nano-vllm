# Operator-Level Replication in nano-vLLM

This document describes the operator-level replication feature implemented for nano-vLLM, which allows multiple replicas of inference operators (currently Attention) to run on different GPUs to handle bursty workloads.

## Overview

The operator replication system provides:

- **Multiple operator replicas** across GPUs with dedicated CUDA streams
- **Load balancing** with round-robin scheduling
- **Asynchronous execution** with CUDA streams and events
- **Auto-scaling** based on queue length and utilization
- **Metrics and logging** for monitoring performance

## Architecture

### Core Components

1. **`OperatorReplica`**: Represents a single replica of an operator on a specific GPU
   - Maintains duplicated weights and workspace
   - Uses a dedicated CUDA stream for execution
   - Tracks metrics (latency, utilization, request count)

2. **`ReplicaManager`**: Manages a pool of operator replicas
   - Creates and destroys replicas across devices
   - Provides load balancing (round-robin scheduling)
   - Handles lifecycle management and cleanup

3. **`ReplicaAutoScaler`**: Background thread for dynamic scaling
   - Monitors queue lengths and GPU utilization
   - Adds/removes replicas based on load thresholds
   - Configurable scaling parameters

### Files Modified/Added

- `nanovllm/engine/op_replica.py` - Core replica implementation
- `nanovllm/config.py` - Configuration options
- `nanovllm/engine/llm_engine.py` - Integration with inference engine
- `nanovllm/models/qwen3.py` - Attention layer integration
- `bench/op_replica_burst_test.py` - Stress testing script
- `nanovllm/tests/test_op_replica.py` - Unit tests

## Configuration

### CLI Options

```bash
# Enable attention replicas (2 per GPU, auto-detect GPUs)
--op-replica attention:2

# Specify GPU devices for replicas
--replica-devices 0,1,2

# Enable auto-scaling
--enable-op-replica-auto-scaling
```

### Programmatic Configuration

```python
from nanovllm import LLM

llm = LLM(
    model="path/to/model",
    op_replica_configs={"attention": 2},  # 2 replicas per GPU
    replica_devices=[0, 1],               # Use GPUs 0 and 1
    enable_op_replica_auto_scaling=True,
)
```

## Usage Examples

### Basic Usage with Replicas

```python
from nanovllm import LLM, SamplingParams

# Initialize LLM with attention replicas
llm = LLM(
    model="path/to/model",
    op_replica_configs={"attention": 2},  # 2 attention replicas per GPU
    replica_devices=[0, 1],               # Use GPUs 0 and 1
)

# Generate as usual - replicas are used automatically
outputs = llm.generate(prompts, sampling_params)
```

### Stress Testing

```bash
# Run burst test to compare performance with/without replicas
python bench/op_replica_burst_test.py --model path/to/model --output results.json

# Quick test
python bench/op_replica_burst_test.py --model path/to/model --quick
```

### Unit Testing

```bash
python -m pytest nanovllm/tests/test_op_replica.py -v
```

## Performance Characteristics

### Benefits

- **Reduced tail latency** during bursty workloads
- **Better GPU utilization** by distributing load across devices
- **Scalable throughput** as more GPUs are added

### Trade-offs

- **Memory overhead** from weight duplication (first iteration)
- **Initialization time** for creating replicas
- **Complexity** in synchronization and stream management

## Implementation Details

### Weight Duplication

Currently, weights are duplicated to each GPU replica. Future optimizations may include:
- CUDA IPC for sharing read-only weights
- NCCL broadcast for efficient weight distribution

### Synchronization

- Uses CUDA streams for asynchronous execution
- CUDA events for synchronization between operations
- Automatic device transfer with `non_blocking=True`

### Load Balancing

- Round-robin scheduling across available replicas
- Future enhancements may include load-aware scheduling

### Auto-scaling

- Monitors queue length and replica utilization
- Adds replicas when queue length > 10 and utilization > 80%
- Removes idle replicas after 30 seconds

## Limitations & Future Work

### Current Limitations

1. **Single operator type**: Only Attention operator replication implemented
2. **Weight duplication**: No memory sharing between replicas
3. **Single machine**: No cross-host replication
4. **Synchronous waits**: Some operations still wait for completion

### Future Enhancements

1. **Multi-operator support**: Extend to MLP, embedding layers
2. **Memory optimization**: CUDA IPC, memory pooling
3. **Cross-host replication**: RPC-based remote replicas
4. **Advanced scheduling**: Load-aware, priority-based scheduling
5. **Async pipeline**: Full asynchronous execution pipeline

## Testing

### Correctness Tests

The implementation includes comprehensive unit tests covering:
- Replica creation and lifecycle
- Asynchronous execution
- Load balancing
- Auto-scaling logic
- Deadlock prevention

### Performance Tests

The burst test script measures:
- Throughput improvement with replicas
- Latency reduction under load
- Scalability across multiple GPUs

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `num_replicas_per_device`
2. **Slow initialization**: Expected due to weight duplication
3. **No performance gain**: Check if workload is compute-bound

### Debugging

Enable logging to see replica operations:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

Metrics are available through `ReplicaManager.get_metrics(op_name)`.

## API Reference

### ReplicaManager

```python
class ReplicaManager:
    def create_replicas(self, op_name, op_class, weights_cpu, devices, workspace_factory=None, num_replicas_per_device=1)
    def get_replica(self, op_name) -> OperatorReplica
    def add_replica(self, op_name, device) -> bool
    def remove_idle_replica(self, op_name) -> bool
    def get_metrics(self, op_name) -> List[ReplicaMetrics]
    def shutdown()
```

### OperatorReplica

```python
class OperatorReplica:
    def forward_async(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.cuda.Event]
    def synchronize()
    def is_idle(self, timeout_ms=5000) -> bool
```

This implementation provides a solid foundation for operator-level replication in nano-vLLM, with room for future optimizations and extensions.


