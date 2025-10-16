# Operator-Level Replication in nano-vLLM

This document describes the operator-level replication feature implemented for nano-vLLM, which provides a framework for parallel processing of stateless operators (currently `down_proj` linear layers) across multiple GPUs, though currently limited by CUDA stream synchronization issues.

## Overview

The operator replication system provides:

- **Multiple operator replicas** across GPUs (currently serial execution due to stream issues)
- **Batch splitting** framework for potential parallel processing
- **Per-layer replication** to handle model-specific weight configurations
- **Load balancing** with original operators integrated into replica pools

## Architecture

### Core Components

1. **`OperatorReplica`**: Represents a single replica of an operator on a specific GPU
   - Maintains duplicated weights and configuration
   - Uses a dedicated CUDA stream for asynchronous execution
   - Tracks metrics for monitoring and debugging

2. **`OriginalOperatorReplica`**: Wrapper for original model operators
   - Allows original operators to participate in load balancing
   - Maintains the same interface as replicated operators

3. **`ReplicaManager`**: Manages a pool of operator replicas
   - Creates and destroys replicas across devices
   - Provides batch splitting for parallel processing
   - Integrates original operators into replica pools
   - Handles lifecycle management and cleanup

### Files Modified/Added

- `nanovllm/engine/op_replica.py` - Core replica implementation
- `nanovllm/config.py` - Configuration options
- `nanovllm/engine/llm_engine.py` - Integration with inference engine and replica initialization
- `nanovllm/models/qwen3.py` - MLP layer integration with batch splitting

## Configuration

### CLI Options

```bash
# Enable down_proj replicas (2 per GPU, auto-detect GPUs)
--op-replica down_proj:2

# Specify GPU devices for replicas (optional, auto-detects if not specified)
--replica-devices 0,1,2
```

### Programmatic Configuration

```python
from nanovllm import LLM

llm = LLM(
    model="path/to/model",
    op_replica_configs={"down_proj": 2},  # 2 replicas per GPU per layer
    replica_devices=[0, 1],               # Use GPUs 0 and 1 (optional)
)
```

## Usage Examples

### Basic Usage with Replicas

```python
from nanovllm import LLM, SamplingParams

# Initialize LLM with down_proj replicas
llm = LLM(
    model="path/to/model",
    op_replica_configs={"down_proj": 2},  # 2 down_proj replicas per GPU per layer
    replica_devices=[0, 1],               # Use GPUs 0 and 1 (optional)
)

# Generate as usual - batch splitting happens automatically during inference
outputs = llm.generate(prompts, sampling_params)
```

## Performance Characteristics

### Benefits

- **Framework for parallel processing** (not currently achieved due to stream issues)
- **Load distribution framework** across multiple GPUs
- **Per-layer specialization** ensuring correct weights for each model layer
- **Scalable architecture** ready for parallel execution once stream issues resolved

### Trade-offs

- **Memory overhead** from weight duplication across GPUs
- **Initialization time** for creating and loading replicas
- **Coordination overhead** for batch splitting and result aggregation
- **Limited to stateless operators currently** (linear layers, not attention)

## Implementation Details

### Per-Layer Replication

Each model layer gets its own set of replicas with the correct weights:
- `down_proj_layer_0`, `down_proj_layer_1`, etc.
- Ensures each layer uses its specific weight configuration
- Original operators integrated into replica pools for load balancing

### Batch Splitting

Input batches are automatically split across available replicas:
- **Current Issue**: Serial execution on default stream, not simultaneous across GPUs
- Framework in place for parallel execution once stream issues are resolved
- Aggregates results back into correct order
- Balances load when batch size ≥ number of replicas

### Synchronization

- **Current Limitation**: Dedicated CUDA streams cause garbled output, so replicas currently use the default stream
- **Impact**: No true parallelism achieved - replicas execute serially despite multiple GPUs
- CUDA events for coordination between operations
- Thread-safe metrics tracking with proper locking

## Limitations & Future Work

### Current Limitations

1. **🚨 No True Parallelism**: Dedicated CUDA streams cause garbled output, forcing serial execution on default stream
2. **Single operator type**: Only `down_proj` (MLP linear layer) replication implemented
3. **Stateless operators only**: Cannot replicate stateful operators like attention

### Future Enhancements

1. **🚨 Fix Stream Parallelism**: Resolve CUDA stream issues to enable true parallel execution
2. **Multi-operator support**: Extend to other linear layers (gate_proj, up_proj)
3. **Advanced scheduling**: Load-aware balancing, priority-based routing
4. **Cross-host replication**: Distributed replica pools across machines
5. **Activation function replication**: Support for stateless activation layers

This implementation provides a foundation for stateless operator replication in nano-vLLM, with batch splitting and per-layer weight management in place, but currently limited by CUDA stream issues that prevent true parallel execution.
