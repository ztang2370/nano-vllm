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
        from nanovllm.models.qwen3 import set_attention_replica_manager, set_linear_replica_manager
        set_attention_replica_manager(self.replica_manager)
        set_linear_replica_manager(self.replica_manager)

        self.model_runner = ModelRunner(config, 0, self.events)

        # Initialize attention replicas if configured (needs to happen after model is loaded)
        self._init_attention_replicas(config)

        # Initialize linear replicas if configured (needs to happen after model is loaded)
        self._init_linear_replicas(config)

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

    def _extract_attention_weights_for_layer(self, layer_id: int) -> dict:
        """Extract attention configuration and weights for a specific layer."""
        # Find the specific layer's attention module
        attention_config = {}
        for name, module in self.model_runner.model.named_modules():
            if f'layers.{layer_id}.self_attn' in name and hasattr(module, 'attn'):
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

            # Create replicas for each layer
            num_layers = config.hf_config.num_hidden_layers
            for layer_id in range(num_layers):
                # Get attention configuration and weights for this specific layer
                layer_attention_config = self._extract_attention_weights_for_layer(layer_id)

                if not layer_attention_config:
                    logger.warning(f"Could not find Qwen3Attention module for layer {layer_id}")
                    continue

                # Create layer-specific replicas
                op_name = f"attention_layer_{layer_id}"
                self.replica_manager.create_replicas(
                    op_name=op_name,
                    op_class=self._get_attention_class(),
                    weights_cpu=layer_attention_config,  # Pass full config including constructor params
                    devices=config.replica_devices,
                    workspace_factory=attention_workspace_factory,
                    num_replicas_per_device=num_replicas,
                )

                # Add the original attention as a special "replica 0" for this layer
                # Only when we actually have replicas configured
                if num_replicas > 0:
                    original_attention_module = layer_attention_config['module']
                    self.replica_manager.add_original_attention_replica(op_name, original_attention_module)

            logger.info(f"Created attention replicas for {num_layers} layers, {num_replicas} replicas per layer across {len(config.replica_devices)} devices")

    def _init_linear_replicas(self, config: Config) -> None:
        """Initialize linear operator replicas if configured."""
        if "down_proj" in config.op_replica_configs:
            num_replicas = config.op_replica_configs["down_proj"]

            # Create replicas for each layer
            num_layers = config.hf_config.num_hidden_layers
            for layer_id in range(num_layers):
                # Get down_proj configuration and weights for this specific layer
                down_proj_config = self._extract_down_proj_weights_for_layer(layer_id)

                if not down_proj_config:
                    logger.warning(f"Could not find down_proj for layer {layer_id}")
                    continue

                # Create workspace factory for down_proj
                def linear_workspace_factory(device):
                    # Linear layers don't need extra workspace
                    return {}

                # Create layer-specific replicas
                op_name = f"down_proj_layer_{layer_id}"
                self.replica_manager.create_replicas(
                    op_name=op_name,
                    op_class=self._get_linear_class(),
                    weights_cpu=down_proj_config,  # Pass full config including constructor params
                    devices=config.replica_devices,
                    workspace_factory=linear_workspace_factory,
                    num_replicas_per_device=num_replicas,
                )

                # Add the original down_proj as a special "replica" so it can participate in load balancing
                # Only when we actually have replicas configured
                if num_replicas > 0:
                    original_down_proj_module = down_proj_config['module']
                    self.replica_manager.add_original_operator_replica(op_name, original_down_proj_module)

                logger.info(f"Created {num_replicas} replicas for {op_name} across {len(config.replica_devices)} devices")

            logger.info(f"Created down_proj replicas for {num_layers} layers, {num_replicas} replicas per layer across {len(config.replica_devices)} devices")

    def _extract_down_proj_weights_for_layer(self, layer_id: int) -> dict:
        """Extract down_proj configuration and weights for a specific layer."""
        # Find the specific layer's down_proj module
        linear_config = {}
        for name, module in self.model_runner.model.named_modules():
            if f'layers.{layer_id}.mlp' in name and hasattr(module, 'down_proj'):
                # Store the RowParallelLinear instance for later use
                down_proj_module = module.down_proj

                # For RowParallelLinear: weight shape is (output_size, input_size // tp_size)
                # Since RowParallelLinear shards the input dimension (tp_dim=1)
                weight_shape = down_proj_module.weight.shape
                original_output_size = weight_shape[0]
                sharded_input_size = weight_shape[1]
                original_input_size = sharded_input_size * down_proj_module.tp_size

                linear_config = {
                    'module': down_proj_module,
                    'input_size': original_input_size,
                    'output_size': original_output_size,
                    'bias': down_proj_module.bias is not None,
                    'tp_dim': down_proj_module.tp_dim,
                    'weights': dict(down_proj_module.state_dict())
                }
                break
        return linear_config

    def _extract_activation_weights_for_layer(self, layer_id: int) -> dict:
        """Extract activation function configuration for a specific layer."""
        # Find the specific layer's activation function
        act_config = {}
        for name, module in self.model_runner.model.named_modules():
            if f'layers.{layer_id}.mlp' in name and hasattr(module, 'act_fn'):
                # Store the activation function instance
                act_fn_module = module.act_fn
                act_config = {
                    'module': act_fn_module,
                    'class_name': act_fn_module.__class__.__name__,
                    # Activation functions are stateless, so no weights to extract
                    'weights': {}
                }
                break
        return act_config

    def _extract_activation_weights(self) -> dict:
        """Extract activation function configuration from the model for replication."""
        # Find activation function in the model
        act_config = {}
        for name, module in self.model_runner.model.named_modules():
            if hasattr(module, 'act_fn') and module.__class__.__name__ == 'Qwen3MLP':
                # Store the activation function instance
                act_fn_module = module.act_fn
                act_config = {
                    'module': act_fn_module,
                    'class_name': act_fn_module.__class__.__name__,
                    # Activation functions are stateless, so no weights to extract
                    'weights': {}
                }
                break
        return act_config

    def _extract_down_proj_weights(self) -> dict:
        """Extract down_proj configuration and weights from the model for replication."""
        # Find down_proj layers in the model to extract config and weights
        linear_config = {}
        for name, module in self.model_runner.model.named_modules():
            if hasattr(module, 'down_proj') and module.__class__.__name__ == 'Qwen3MLP':
                # Store the RowParallelLinear instance for later use
                down_proj_module = module.down_proj

                # For RowParallelLinear: weight shape is (output_size, input_size // tp_size)
                # Since RowParallelLinear shards the input dimension (tp_dim=1)
                weight_shape = down_proj_module.weight.shape
                original_output_size = weight_shape[0]
                sharded_input_size = weight_shape[1]
                original_input_size = sharded_input_size * down_proj_module.tp_size

                linear_config = {
                    'module': down_proj_module,
                    'input_size': original_input_size,
                    'output_size': original_output_size,
                    'bias': down_proj_module.bias is not None,
                    'tp_dim': down_proj_module.tp_dim,
                    'weights': dict(down_proj_module.state_dict())
                }
                break
        return linear_config

    def _get_attention_class(self):
        """Get the Attention class for replication."""
        from nanovllm.layers.attention import Attention
        return Attention

    def _get_activation_class(self):
        """Get the ReplicaSiluAndMul class for replication."""
        from nanovllm.layers.activation import ReplicaSiluAndMul
        return ReplicaSiluAndMul

    def _get_linear_class(self):
        """Get the RowParallelLinear class for replication."""
        from nanovllm.layers.linear import RowParallelLinear
        return RowParallelLinear

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
