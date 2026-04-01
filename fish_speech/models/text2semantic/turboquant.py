"""TurboQuant KV Cache — Polar-coordinate quantization for KV cache compression.

Based on:
  TurboQuant (ICLR 2026): https://arxiv.org/abs/2504.19874
  PolarQuant (AISTATS 2026): https://arxiv.org/abs/2502.02617

Core idea:
  1. Apply a random orthogonal rotation to each KV vector — makes coordinates
     approximately independent Gaussian, enabling per-coordinate quantization
     without joint codebook overhead.
  2. Store the vector magnitude (L2 norm) in fp16.
  3. Quantize each rotated coordinate to N bits using precomputed Lloyd-Max
     centroids for the standard Gaussian distribution.
  4. On read: dequantize centroids, apply inverse rotation, scale by magnitude.

This gives ~4x compression at 4-bit and ~6x at 2-bit with minimal accuracy loss.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger


def _gaussian_lloyd_max_centroids(bits: int) -> torch.Tensor:
    """Precomputed Lloyd-Max optimal centroids for N(0,1) distribution.

    These are the optimal reconstruction points that minimize MSE for
    scalar quantization of a standard normal distribution. Values from
    the Lloyd-Max algorithm (iterative k-means on the continuous PDF).
    """
    # Precomputed centroids for common bit widths (symmetric around 0)
    tables = {
        2: torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104]),
        3: torch.tensor([
            -2.1519, -1.3440, -0.7560, -0.2451,
            0.2451, 0.7560, 1.3440, 2.1519,
        ]),
        4: torch.tensor([
            -2.7326, -2.0690, -1.6180, -1.2562,
            -0.9423, -0.6568, -0.3881, -0.1284,
            0.1284, 0.3881, 0.6568, 0.9423,
            1.2562, 1.6180, 2.0690, 2.7326,
        ]),
    }
    if bits not in tables:
        raise ValueError(f"Unsupported bit width {bits}, must be 2, 3, or 4")
    return tables[bits]


def _gaussian_lloyd_max_boundaries(bits: int) -> torch.Tensor:
    """Decision boundaries (midpoints between adjacent centroids)."""
    centroids = _gaussian_lloyd_max_centroids(bits)
    # Boundaries are midpoints between adjacent centroids, plus -inf/+inf
    mids = (centroids[:-1] + centroids[1:]) / 2
    return mids


def _pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack quantization indices into uint8 tensor.

    Args:
        indices: (*, D) tensor of values in [0, 2^bits - 1]
        bits: 2 or 4

    Returns:
        Packed uint8 tensor of shape (*, D // vals_per_byte)
    """
    vals_per_byte = 8 // bits
    *prefix, D = indices.shape
    packed_D = (D + vals_per_byte - 1) // vals_per_byte

    indices = indices.to(torch.uint8)
    # Pad D to multiple of vals_per_byte
    if D % vals_per_byte != 0:
        pad = vals_per_byte - (D % vals_per_byte)
        indices = torch.nn.functional.pad(indices, (0, pad))

    indices = indices.view(*prefix, -1, vals_per_byte)

    packed = torch.zeros(*prefix, indices.shape[-2], dtype=torch.uint8, device=indices.device)
    for i in range(vals_per_byte):
        packed |= indices[..., i] << (i * bits)

    return packed


def _unpack_indices(packed: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    """Unpack uint8 tensor back to quantization indices.

    Args:
        packed: (*, packed_D) uint8 tensor
        bits: 2 or 4
        dim: original unpacked dimension D

    Returns:
        (*, D) tensor of values in [0, 2^bits - 1]
    """
    vals_per_byte = 8 // bits
    mask = (1 << bits) - 1

    *prefix, packed_D = packed.shape
    unpacked = torch.zeros(*prefix, packed_D, vals_per_byte, dtype=torch.uint8, device=packed.device)
    for i in range(vals_per_byte):
        unpacked[..., i] = (packed >> (i * bits)) & mask

    unpacked = unpacked.view(*prefix, -1)
    return unpacked[..., :dim]


class TurboQuantKVCache(nn.Module):
    """Drop-in replacement for KVCache with PolarQuant-style compression.

    Memory comparison for (B=1, H=8, S=4096, D=128):
      bf16 KVCache:      1 * 8 * 4096 * 128 * 2 = 8 MB per K or V
      4-bit TurboQuant:  1 * 8 * 4096 * (64 + 2) = 0.52 MB per K or V  (~4x)
      2-bit TurboQuant:  1 * 8 * 4096 * (32 + 2) = 0.27 MB per K or V  (~7.5x)
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        bits: int = 4,
    ):
        super().__init__()
        self.bits = bits
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        n_levels = 1 << bits  # 2^bits

        # Random orthogonal rotation matrix — makes coordinates independent
        R = torch.randn(head_dim, head_dim, dtype=torch.float32)
        R, _ = torch.linalg.qr(R)
        self.register_buffer("rotation", R.to(dtype))

        # Lloyd-Max centroids and boundaries for Gaussian quantization
        centroids = _gaussian_lloyd_max_centroids(bits)
        boundaries = _gaussian_lloyd_max_boundaries(bits)
        self.register_buffer("centroids", centroids.to(dtype))
        self.register_buffer("boundaries", boundaries.to(dtype))

        # Packed quantized cache
        vals_per_byte = 8 // bits
        packed_dim = (head_dim + vals_per_byte - 1) // vals_per_byte
        cache_shape = (max_batch_size, n_heads, max_seq_len, packed_dim)
        self.register_buffer("k_packed", torch.zeros(cache_shape, dtype=torch.uint8))
        self.register_buffer("v_packed", torch.zeros(cache_shape, dtype=torch.uint8))

        # Per-vector magnitude (L2 norm) and mean, stored in full precision
        norm_shape = (max_batch_size, n_heads, max_seq_len, 1)
        self.register_buffer("k_mag", torch.zeros(norm_shape, dtype=dtype))
        self.register_buffer("v_mag", torch.zeros(norm_shape, dtype=dtype))
        self.register_buffer("k_mean", torch.zeros(norm_shape, dtype=dtype))
        self.register_buffer("v_mean", torch.zeros(norm_shape, dtype=dtype))

        logger.info(
            f"TurboQuantKVCache: {bits}-bit, {head_dim}D, "
            f"~{self._compression_ratio():.1f}x compression"
        )

    def _compression_ratio(self) -> float:
        bf16_bytes = self.head_dim * 2  # bf16 = 2 bytes per element
        quant_bytes = (self.head_dim * self.bits / 8) + 2 + 2  # packed + mag + mean
        return bf16_bytes / quant_bytes

    def _quantize_vector(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize vectors using random rotation + scalar Lloyd-Max.

        Args:
            x: (B, H, S, D) input vectors

        Returns:
            packed: (B, H, S, packed_D) uint8 packed indices
            mag: (B, H, S, 1) L2 norms
            mean: (B, H, S, 1) per-vector means
        """
        B, H, S, D = x.shape

        # Store mean and magnitude for reconstruction
        mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - mean
        mag = torch.norm(x_centered, dim=-1, keepdim=True).clamp(min=1e-8)

        # Normalize to unit variance per coordinate: x_norm ~ N(0, 1/sqrt(D))
        x_norm = x_centered / mag * math.sqrt(D)

        # Apply random rotation — makes coordinates approximately iid N(0, 1/sqrt(D))
        # After rotation and scaling, each coordinate ~ N(0, 1) approximately
        x_rot = x_norm @ self.rotation  # (B, H, S, D)

        # Scalar quantization using Lloyd-Max boundaries
        # boundaries shape: (n_levels - 1,)
        # Result: indices in [0, n_levels - 1]
        indices = torch.bucketize(x_rot, self.boundaries)  # (B, H, S, D)

        # Pack into uint8
        packed = _pack_indices(indices, self.bits)

        return packed, mag, mean

    def _dequantize_vector(
        self, packed: torch.Tensor, mag: torch.Tensor, mean: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize vectors from packed indices + magnitude + mean.

        Args:
            packed: (B, H, S, packed_D) uint8
            mag: (B, H, S, 1)
            mean: (B, H, S, 1)

        Returns:
            x_approx: (B, H, S, D) reconstructed vectors in original dtype
        """
        # Unpack indices
        indices = _unpack_indices(packed, self.bits, self.head_dim)

        # Look up centroids
        x_rot = self.centroids[indices.long()]  # (B, H, S, D)

        # Inverse rotation
        x_norm = x_rot @ self.rotation.T

        # Rescale by magnitude and add mean
        x_approx = x_norm * mag / math.sqrt(self.head_dim) + mean

        return x_approx

    def update(
        self, input_pos: torch.Tensor, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize and store new K/V, return dequantized full cache.

        Drop-in compatible with KVCache.update().
        """
        assert input_pos.shape[0] == k_val.shape[2]

        # Quantize incoming K and V
        k_packed, k_mag, k_mean = self._quantize_vector(k_val)
        v_packed, v_mag, v_mean = self._quantize_vector(v_val)

        # Store compressed form
        self.k_packed[:, :, input_pos] = k_packed
        self.v_packed[:, :, input_pos] = v_packed
        self.k_mag[:, :, input_pos] = k_mag
        self.v_mag[:, :, input_pos] = v_mag
        self.k_mean[:, :, input_pos] = k_mean
        self.v_mean[:, :, input_pos] = v_mean

        # Dequantize full cache for attention
        k_out = self._dequantize_vector(self.k_packed, self.k_mag, self.k_mean)
        v_out = self._dequantize_vector(self.v_packed, self.v_mag, self.v_mean)

        return k_out, v_out
