import atexit
import logging
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.op_replica import ReplicaManager, ReplicaConfig

logger = logging.getLogger(__name__)


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # Initialize operator replica manager
        replica_config = ReplicaConfig(
            num_replicas_per_device=1,  # Default, will be overridden by op configs
            replica_devices=config.replica_devices,
            auto_scaling_enabled=config.enable_op_replica_auto_scaling,
        )
        self.replica_manager = ReplicaManager(replica_config)

        # Set replica manager in the model for attention layers to use BEFORE creating ModelRunner
        from nanovllm.models.qwen3 import set_attention_replica_manager
        set_attention_replica_manager(self.replica_manager)

        self.model_runner = ModelRunner(config, 0, self.events)

        # Initialize attention replicas if configured (needs to happen after model is loaded)
        self._init_attention_replicas(config)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)

        atexit.register(self.exit)

    def _extract_attention_weights(self) -> dict:
        """Extract attention configuration and weights from the model for replication."""
        # Find Qwen3Attention layers in the model to extract config and weights
        attention_config = {}
        for name, module in self.model_runner.model.named_modules():
            if module.__class__.__name__ == 'Qwen3Attention':
                # Store the Qwen3Attention instance for later use
                attention_config = {
                    'module': module,
                    'num_heads': module.num_heads,
                    'head_dim': module.head_dim,
                    'scale': module.scaling,
                    'num_kv_heads': module.num_kv_heads,
                    'weights': dict(module.state_dict())
                }
                break
        return attention_config

    def _init_attention_replicas(self, config: Config) -> None:
        """Initialize attention operator replicas if configured."""
        if "attention" in config.op_replica_configs:
            num_replicas = config.op_replica_configs["attention"]

            # Get attention configuration and weights from the model
            attention_config = self._extract_attention_weights()

            if not attention_config:
                logger.warning("Could not find Qwen3Attention module for replication")
                return

            # Create workspace factory for attention
            def attention_workspace_factory(device):
                # For now, return empty workspace - attention doesn't need extra workspace
                # beyond what's already in the Attention class
                return {}

            # Create replicas across configured devices
            self.replica_manager.create_replicas(
                op_name="attention",
                op_class=self._get_attention_class(),
                weights_cpu=attention_config,  # Pass full config including constructor params
                devices=config.replica_devices,
                workspace_factory=attention_workspace_factory,
                num_replicas_per_device=num_replicas,
            )

            # Add the original attention as a special "replica 0"
            # This allows the original attention operator to also be utilized for load balancing
            original_attention_module = attention_config['module']
            self.replica_manager.add_original_attention_replica("attention", original_attention_module)

    def _get_attention_class(self):
        """Get the Attention class for replication."""
        from nanovllm.layers.attention import Attention
        return Attention

    def exit(self):
        # Shutdown replica manager first
        if hasattr(self, 'replica_manager'):
            self.replica_manager.shutdown()

        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)


        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)

        # Clear batch replica cache only if all sequences in batch are finished
        # This allows continuing sequences to keep their replica assignment
        if hasattr(self, 'replica_manager') and self.replica_manager:
            all_finished = all(seq.is_finished for seq in seqs)
            if all_finished:
                self.replica_manager.clear_batch_replicas()

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
