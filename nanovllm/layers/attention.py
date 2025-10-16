import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        weights=None,  # For operator replication - weights dict from replica
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

        # If weights provided (for replication), store them
        # Note: The actual weights are managed by the parent Qwen3Attention module
        # This is just for API compatibility with the replica system
        self.replica_weights = weights
        self.is_replica = weights is not None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()

        # Check if this instance has KV cache (original attention) vs replica
        has_kv_cache = hasattr(self, 'k_cache') and self.k_cache is not None and self.k_cache.numel() > 0

        if has_kv_cache:
            # Original attention with KV cache - use flash_attn
            k_cache, v_cache = self.k_cache, self.v_cache
            if k_cache.numel() and v_cache.numel():
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            if context.is_prefill:
                if context.block_tables is not None:    # prefix cache
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            else:    # decode
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True)
                o = o.squeeze(1)  # Remove the extra dimension added by unsqueeze(1)
            return o
        else:
            # Replica mode - use flash_attn without KV cache (stateless)
            if context is not None:
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
                return o
            else:
                # Fallback to simplified attention if no context
                return self._replica_forward(q, k, v)

    def _replica_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Simplified batched attention computation.
        Input: [total_tokens, num_heads, head_dim] where total_tokens may span multiple sequences
        """
        total_tokens, num_q_heads, head_dim = q.shape
        _, num_kv_heads, _ = k.shape

        # Handle GQA: expand k,v to match query heads
        if num_q_heads != num_kv_heads:
            repeat_factor = num_q_heads // num_kv_heads
            k = k.repeat(1, repeat_factor, 1)  # [total_tokens, num_heads, head_dim]
            v = v.repeat(1, repeat_factor, 1)  # [total_tokens, num_heads, head_dim]

        # Use PyTorch's efficient batched attention
        # Reshape to [batch_size=1, seq_len, num_heads, head_dim] for scaled_dot_product_attention
        # But since we have variable sequence lengths, we need to handle this differently

        # For simplicity, treat the entire input as one long sequence with causal masking
        # This approximates the behavior of flash_attn for batched inputs

        # Reshape for attention: [num_heads, total_tokens, head_dim]
        q = q.transpose(0, 1)  # [num_heads, total_tokens, head_dim]
        k = k.transpose(0, 1)  # [num_heads, total_tokens, head_dim]
        v = v.transpose(0, 1)  # [num_heads, total_tokens, head_dim]

        # Use PyTorch's scaled_dot_product_attention with causal masking
        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,  # Let is_causal handle masking
            dropout_p=0.0,
            is_causal=True,  # Enable causal attention automatically
            scale=self.scale
        )

        # Transpose back: [num_heads, total_tokens, head_dim] -> [total_tokens, num_heads, head_dim]
        output = output.transpose(0, 1)
        return output
