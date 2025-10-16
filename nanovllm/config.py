import argparse
import os
from dataclasses import dataclass, field, fields

from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    # Operator replication options
    op_replica_configs: dict = field(default_factory=dict)  # e.g., {"down_proj": 2}
    replica_devices: list[int] = field(default_factory=list)  # Empty means auto-detect available GPUs

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len

        # Auto-detect replica devices if not specified
        if not self.replica_devices:
            try:
                import torch
                self.replica_devices = list(range(torch.cuda.device_count()))
            except Exception:
                self.replica_devices = [0]  # Fallback to device 0

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        """Create Config from parsed command line arguments."""
        # Parse op_replica arguments (format: "down_proj:2")
        op_replica_configs = {}
        if hasattr(args, 'op_replica') and args.op_replica:
            for config_str in args.op_replica:
                if ':' in config_str:
                    op_name, count_str = config_str.split(':', 1)
                    try:
                        op_replica_configs[op_name.strip()] = int(count_str.strip())
                    except ValueError:
                        raise ValueError(f"Invalid op_replica format: {config_str}. Expected 'op_name:count'")

        # Parse replica_devices (format: "0,1,2")
        replica_devices = []
        if hasattr(args, 'replica_devices') and args.replica_devices:
            try:
                replica_devices = [int(x.strip()) for x in args.replica_devices.split(',')]
            except ValueError:
                raise ValueError(f"Invalid replica_devices format: {args.replica_devices}. Expected comma-separated integers")

        # Get Config field names
        config_fields = {field.name for field in fields(cls)}

        # Build kwargs for Config constructor - only include valid Config fields
        config_kwargs = {}
        for field_name in config_fields:
            if hasattr(args, field_name):
                config_kwargs[field_name] = getattr(args, field_name)

        # Add parsed replica configs
        config_kwargs['op_replica_configs'] = op_replica_configs
        config_kwargs['replica_devices'] = replica_devices

        return cls(**config_kwargs)

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """Add operator replication CLI arguments to the parser."""
        parser.add_argument(
            '--op-replica',
            action='append',
            help='Operator replication configuration in format "op_name:num_replicas". '
                 'Currently supports "down_proj". Example: --op-replica down_proj:2'
        )
        parser.add_argument(
            '--replica-devices',
            type=str,
            help='Comma-separated list of CUDA device indices to use for replicas. '
                 'Default: auto-detect all available GPUs'
        )
