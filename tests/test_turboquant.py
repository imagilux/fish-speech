"""Tests for TurboQuant KV cache compression."""

import torch
import pytest

from fish_speech.models.text2semantic.turboquant import (
    TurboQuantKVCache,
    _pack_indices,
    _unpack_indices,
    _gaussian_lloyd_max_centroids,
)


class TestPackUnpack:
    def test_roundtrip_4bit(self):
        indices = torch.randint(0, 16, (2, 4, 8, 128), dtype=torch.uint8)
        packed = _pack_indices(indices, bits=4)
        unpacked = _unpack_indices(packed, bits=4, dim=128)
        assert torch.equal(indices, unpacked)

    def test_roundtrip_2bit(self):
        indices = torch.randint(0, 4, (2, 4, 8, 128), dtype=torch.uint8)
        packed = _pack_indices(indices, bits=2)
        unpacked = _unpack_indices(packed, bits=2, dim=128)
        assert torch.equal(indices, unpacked)

    def test_packed_size_4bit(self):
        indices = torch.randint(0, 16, (1, 8, 32, 128), dtype=torch.uint8)
        packed = _pack_indices(indices, bits=4)
        assert packed.shape[-1] == 64  # 128 / 2

    def test_packed_size_2bit(self):
        indices = torch.randint(0, 4, (1, 8, 32, 128), dtype=torch.uint8)
        packed = _pack_indices(indices, bits=2)
        assert packed.shape[-1] == 32  # 128 / 4


class TestCentroids:
    def test_symmetric(self):
        for bits in (2, 3, 4):
            c = _gaussian_lloyd_max_centroids(bits)
            assert len(c) == 2**bits
            # Centroids should be symmetric around 0
            assert torch.allclose(c + c.flip(0), torch.zeros_like(c), atol=1e-3)

    def test_sorted(self):
        for bits in (2, 3, 4):
            c = _gaussian_lloyd_max_centroids(bits)
            assert torch.all(c[1:] > c[:-1])


class TestTurboQuantKVCache:
    @pytest.fixture
    def cache(self):
        return TurboQuantKVCache(
            max_batch_size=1,
            max_seq_len=64,
            n_heads=8,
            head_dim=128,
            dtype=torch.float32,
            bits=4,
        )

    def test_update_returns_correct_shape(self, cache):
        input_pos = torch.arange(0, 4)
        k = torch.randn(1, 8, 4, 128)
        v = torch.randn(1, 8, 4, 128)

        k_out, v_out = cache.update(input_pos, k, v)
        assert k_out.shape == (1, 8, 64, 128)
        assert v_out.shape == (1, 8, 64, 128)

    def test_reconstruction_quality_4bit(self, cache):
        """4-bit should preserve most of the signal."""
        input_pos = torch.arange(0, 16)
        k = torch.randn(1, 8, 16, 128)
        v = torch.randn(1, 8, 16, 128)

        k_out, v_out = cache.update(input_pos, k, v)

        # Extract the positions we wrote
        k_recon = k_out[:, :, :16]
        v_recon = v_out[:, :, :16]

        # Cosine similarity should be high
        k_cos = torch.nn.functional.cosine_similarity(
            k.flatten(), k_recon.flatten(), dim=0
        )
        v_cos = torch.nn.functional.cosine_similarity(
            v.flatten(), v_recon.flatten(), dim=0
        )
        assert k_cos > 0.95, f"K cosine similarity too low: {k_cos:.4f}"
        assert v_cos > 0.95, f"V cosine similarity too low: {v_cos:.4f}"

    def test_reconstruction_quality_2bit(self):
        cache = TurboQuantKVCache(
            max_batch_size=1, max_seq_len=64, n_heads=8,
            head_dim=128, dtype=torch.float32, bits=2,
        )
        input_pos = torch.arange(0, 16)
        k = torch.randn(1, 8, 16, 128)

        k_out, _ = cache.update(input_pos, k, torch.randn_like(k))
        k_recon = k_out[:, :, :16]

        k_cos = torch.nn.functional.cosine_similarity(
            k.flatten(), k_recon.flatten(), dim=0
        )
        # 2-bit is coarser but should still preserve direction
        assert k_cos > 0.85, f"2-bit K cosine similarity too low: {k_cos:.4f}"

    def test_incremental_update(self, cache):
        """Simulate autoregressive generation — one position at a time."""
        all_k = torch.randn(1, 8, 8, 128)

        for i in range(8):
            pos = torch.tensor([i])
            k_i = all_k[:, :, i:i+1]
            v_i = torch.randn(1, 8, 1, 128)
            k_out, v_out = cache.update(pos, k_i, v_i)

        # All 8 positions should be filled
        k_filled = k_out[:, :, :8]
        assert k_filled.abs().sum() > 0

    def test_memory_smaller_than_bf16(self, cache):
        """Quantized cache should use less memory than bf16."""
        bf16_size = 1 * 8 * 64 * 128 * 2  # bf16 K cache
        quant_size = (
            cache.k_packed.numel() * 1  # uint8
            + cache.k_mag.numel() * 4   # float32
            + cache.k_mean.numel() * 4  # float32
        )
        assert quant_size < bf16_size, f"Quantized {quant_size} >= bf16 {bf16_size}"


class TestDropInCompatibility:
    def test_same_interface_as_kvcache(self):
        """TurboQuantKVCache must have the same update() signature."""
        from fish_speech.models.text2semantic.llama import KVCache

        dtype = torch.bfloat16
        kv = KVCache(1, 64, 8, 128, dtype=dtype)
        tq = TurboQuantKVCache(1, 64, 8, 128, dtype=dtype, bits=4)

        pos = torch.arange(4)
        k = torch.randn(1, 8, 4, 128, dtype=dtype)
        v = torch.randn(1, 8, 4, 128, dtype=dtype)

        k1, v1 = kv.update(pos, k, v)
        k2, v2 = tq.update(pos, k, v)

        assert k1.shape == k2.shape
        assert v1.shape == v2.shape
