"""Tests for INT4 polar-quantized attention kernel.

Compares output against reference bf16 scaled_dot_product_attention
to validate that the Triton kernel produces correct results.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from fish_speech.models.text2semantic.turboquant import (
    TurboQuantKVCache,
    _gaussian_lloyd_max_centroids,
    _gaussian_lloyd_max_boundaries,
    _pack_indices,
)


def _requires_rocm():
    """Skip if no ROCm GPU available."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", "")
    if not arch:
        pytest.skip("Not a ROCm GPU")


def _reference_attention(q, k, v, head_dim):
    """Standard bf16 scaled dot-product attention (single head, no mask)."""
    scale = 1.0 / math.sqrt(head_dim)
    scores = (q.float() @ k.float().T) * scale
    # Causal mask
    num_q, seq_kv = scores.shape
    causal = torch.tril(torch.ones(num_q, seq_kv, device=q.device))
    # For decode (num_q=1), offset the mask so it covers all valid KV positions
    if num_q == 1:
        causal = torch.ones(1, seq_kv, device=q.device)
    scores = scores.masked_fill(causal == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return (attn @ v.float()).to(q.dtype)


class TestInt4AttentionKernel:
    """Test the Triton INT4 attention kernel against reference implementation."""

    @pytest.fixture
    def setup(self):
        _requires_rocm()
        head_dim = 128
        seq_kv = 32
        device = "cuda"
        dtype = torch.bfloat16

        # Generate random K, V
        k_orig = torch.randn(seq_kv, head_dim, device=device, dtype=dtype)
        v_orig = torch.randn(seq_kv, head_dim, device=device, dtype=dtype)

        # Quantize K and V using TurboQuant
        cache = TurboQuantKVCache(
            max_batch_size=1, max_seq_len=64, n_heads=1,
            head_dim=head_dim, dtype=dtype, bits=4,
        ).to(device)

        # Feed through cache to get quantized form
        pos = torch.arange(seq_kv, device=device)
        k_4d = k_orig.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_kv, D)
        v_4d = v_orig.unsqueeze(0).unsqueeze(0)
        k_deq, v_deq = cache.update(pos, k_4d, v_4d)

        return {
            "head_dim": head_dim,
            "seq_kv": seq_kv,
            "device": device,
            "dtype": dtype,
            "k_orig": k_orig,
            "v_orig": v_orig,
            "k_deq": k_deq[0, 0, :seq_kv],  # (seq_kv, D)
            "v_deq": v_deq[0, 0, :seq_kv],
            "cache": cache,
        }

    def test_import(self):
        """Verify the kernel module imports without error."""
        _requires_rocm()
        from fish_speech.kernels.int4_attention import int4_polar_attention
        assert callable(int4_polar_attention)

    def test_quantized_vs_bf16_cosine(self, setup):
        """Dequantized KV should be close to original (baseline check)."""
        k_cos = F.cosine_similarity(
            setup["k_orig"].flatten().float(),
            setup["k_deq"].flatten().float(),
            dim=0,
        )
        assert k_cos > 0.90, f"K cosine similarity too low: {k_cos:.4f}"

    def test_kernel_output_shape(self, setup):
        """Kernel output should match query shape."""
        from fish_speech.kernels.int4_attention import int4_polar_attention

        cache = setup["cache"]
        q = torch.randn(1, setup["head_dim"], device=setup["device"], dtype=setup["dtype"])

        out = int4_polar_attention(
            q=q,
            k_packed=cache.k_packed[0, 0],
            k_mag=cache.k_mag[0, 0, :, 0],
            k_mean=cache.k_mean[0, 0, :, 0],
            v_packed=cache.v_packed[0, 0],
            v_mag=cache.v_mag[0, 0, :, 0],
            v_mean=cache.v_mean[0, 0, :, 0],
            centroids=cache.centroids,
            rotation=cache.rotation,
            seq_kv=setup["seq_kv"],
        )
        assert out.shape == q.shape
        assert out.dtype == q.dtype

    def test_kernel_vs_reference_decode(self, setup):
        """Decode mode (1 query): kernel output should approximate bf16 attention."""
        from fish_speech.kernels.int4_attention import int4_polar_attention

        cache = setup["cache"]
        q = torch.randn(1, setup["head_dim"], device=setup["device"], dtype=setup["dtype"])

        # Reference: bf16 attention on dequantized KV
        ref_out = _reference_attention(q, setup["k_deq"], setup["v_deq"], setup["head_dim"])

        # Kernel: INT4 attention
        kern_out = int4_polar_attention(
            q=q,
            k_packed=cache.k_packed[0, 0],
            k_mag=cache.k_mag[0, 0, :, 0],
            k_mean=cache.k_mean[0, 0, :, 0],
            v_packed=cache.v_packed[0, 0],
            v_mag=cache.v_mag[0, 0, :, 0],
            v_mean=cache.v_mean[0, 0, :, 0],
            centroids=cache.centroids,
            rotation=cache.rotation,
            seq_kv=setup["seq_kv"],
        )

        cos = F.cosine_similarity(ref_out.flatten().float(), kern_out.flatten().float(), dim=0)
        assert cos > 0.85, f"Kernel vs reference cosine similarity too low: {cos:.4f}"
