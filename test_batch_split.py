#!/usr/bin/env python3
"""
Test script for batch splitting across replicas
"""

import torch
from nanovllm import LLM

def test_batch_splitting():
    # Initialize LLM with down_proj replication
    llm = LLM('/home/ztang23/huggingface/Qwen3-0.6B/', op_replica_configs={'down_proj': 1})

    # Access the model through the engine
    model = llm.model_runner.model.model  # Qwen3ForCausalLM.model is the Qwen3Model

    # Create a batch of inputs (batch size 4)
    prompts = ["Hello", "Hi", "Hey", "Yo"]
    batch_size = len(prompts)

    print(f"Testing batch splitting with batch_size={batch_size}")

    # Create simple token sequences for testing
    # Use token IDs that exist in the vocabulary
    inputs = []
    for prompt in prompts:
        tokens = llm.tokenizer.encode(prompt)[:2]  # Take first 2 tokens
        inputs.append(torch.tensor(tokens, dtype=torch.long))

    # Pad to same length
    max_len = max(len(x) for x in inputs)
    padded_inputs = []
    for inp in inputs:
        if len(inp) < max_len:
            padded = torch.cat([inp, torch.full((max_len - len(inp),), llm.tokenizer.pad_token_id or 0)])
        else:
            padded = inp
        padded_inputs.append(padded)

    # Stack into batch
    batch_input = torch.stack(padded_inputs).to('cuda')

    print(f"Input batch shape: {batch_input.shape}")

    # Run inference
    with torch.no_grad():
        # Get embeddings
        hidden_states = model.embed_tokens(batch_input)

        # Create positions
        positions = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1).to('cuda')

        print(f"Initial hidden_states shape: {hidden_states.shape}")

        # Create a simple MLP test input (batch_size, hidden_size)
        # The MLP expects input of shape (batch_size, hidden_size) = (4, 1024)
        test_input = torch.randn(batch_size, 1024, dtype=torch.bfloat16).to('cuda')
        print(f"Test MLP input shape: {test_input.shape}")

        # Test just the MLP part to demonstrate batch splitting
        mlp = model.layers[0].mlp  # Get the MLP from the first layer
        output = mlp(test_input)

        print(f"MLP output shape: {output.shape}")
        print("Batch splitting test completed successfully!")

if __name__ == "__main__":
    test_batch_splitting()
