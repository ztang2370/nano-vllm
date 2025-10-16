#!/usr/bin/env python3
"""
CLI interface for nano-vLLM with operator replication support.
"""

import argparse
import sys

from nanovllm import LLM, SamplingParams
from nanovllm.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="nano-vLLM CLI with operator replication")

    # Model configuration
    parser.add_argument("model", help="Path to model directory")
    parser.add_argument("--prompt", "-p", help="Single prompt to generate from")
    parser.add_argument("--prompts", help="Multiple prompts (comma-separated)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--top-p", type=float, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, help="Top-k sampling")

    # Performance options
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--enforce-eager", action="store_true", help="Use eager mode instead of CUDA graphs")

    # Operator replication options (from Config.add_cli_args)
    Config.add_cli_args(parser)

    return parser.parse_args()


def main():
    args = parse_args()

    # Create config from args
    config = Config.from_args(args)

    # Force enforce_eager=True when using operator replicas (they don't work with CUDA graphs)
    use_op_replicas = bool(config.op_replica_configs)
    enforce_eager = config.enforce_eager or use_op_replicas

    if use_op_replicas and not config.enforce_eager:
        print("Note: Operator replicas enabled, forcing enforce_eager=True (CUDA graphs not supported)")

    # Convert config back to kwargs for LLM
    llm_kwargs = {
        'model': args.model,  # Use the model path from CLI args
        'max_num_batched_tokens': config.max_num_batched_tokens,
        'max_num_seqs': config.max_num_seqs,
        'max_model_len': config.max_model_len,
        'gpu_memory_utilization': config.gpu_memory_utilization,
        'tensor_parallel_size': config.tensor_parallel_size,
        'enforce_eager': enforce_eager,
        'hf_config': config.hf_config,
        'eos': config.eos,
        'kvcache_block_size': config.kvcache_block_size,
        'num_kvcache_blocks': config.num_kvcache_blocks,
        'op_replica_configs': config.op_replica_configs,
        'replica_devices': config.replica_devices,
    }

    print("nano-vLLM CLI")
    print(f"Model: {args.model}")
    print(f"Op Replica Config: {config.op_replica_configs}")
    print(f"Replica Devices: {config.replica_devices}")
    print("-" * 50)

    try:
        llm = LLM(**llm_kwargs)
        print("Model loaded successfully!")
        print()

        if args.interactive:
            run_interactive(llm, args)
        elif args.prompts:
            run_multiple_prompts(llm, args)
        elif args.prompt:
            run_single_prompt(llm, args)
        else:
            print("Use --prompt, --prompts, or --interactive mode")
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_single_prompt(llm, args):
    """Run a single prompt."""
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        # Note: top_p and top_k not supported by current SamplingParams
    )

    print(f"Prompt: {args.prompt}")
    print("Generating...")

    outputs = llm.generate([args.prompt], sampling_params)

    print("\nCompletion:")
    print(outputs[0])


def run_multiple_prompts(llm, args):
    """Run multiple concurrent prompts to test operator replicas."""
    # Parse comma-separated prompts
    prompts = [p.strip() for p in args.prompts.split(',') if p.strip()]

    if len(prompts) < 2:
        print("Error: --prompts requires at least 2 prompts for concurrent testing")
        return

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        # Note: top_p and top_k not supported by current SamplingParams
    )

    print(f"Running {len(prompts)} concurrent prompts:")
    for i, prompt in enumerate(prompts):
        print(f"  {i+1}: {prompt}")
    print("Generating...")
    print()

    outputs = llm.generate(prompts, sampling_params)

    print("Completions:")
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"{i+1}. Prompt: {prompt}")
        print(f"   Completion: {output['text']}")
        print()


def run_interactive(llm, args):
    """Run in interactive mode."""
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        # Note: top_p and top_k not supported by current SamplingParams
    )

    print("Interactive mode. Type 'quit' or 'exit' to stop.")
    print()

    while True:
        try:
            prompt = input("Prompt> ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            if not prompt:
                continue

            print("Generating...")
            outputs = llm.generate([prompt], sampling_params)
            print(f"Response: {outputs[0]['text']}")
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break


if __name__ == "__main__":
    main()
