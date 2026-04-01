"""Micro-benchmark: INT4 kernel vs bf16 SDPA.

Run with: uv run --extra rocm72 python tests/bench_int4_kernel.py
"""

import math
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from fish_speech.models.text2semantic.turboquant import TurboQuantKVCache
from fish_speech.kernels.int4_attention import int4_attention_multihead


def bench_bf16_sdpa(q, k, v, n_reps=100):
    """Benchmark standard bf16 scaled_dot_product_attention."""
    # Warmup
    for _ in range(5):
        with sdpa_kernel(SDPBackend.MATH):
            F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_reps):
        with sdpa_kernel(SDPBackend.MATH):
            F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / n_reps


def bench_int4_kernel(q_orig, cache, rotation, centroids, n_heads, n_reps=100):
    """Benchmark the INT4 Triton kernel."""
    # Warmup
    for _ in range(5):
        int4_attention_multihead(q_orig, cache, n_heads, rotation, centroids)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_reps):
        int4_attention_multihead(q_orig, cache, n_heads, rotation, centroids)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / n_reps


def bench_rotation_only(q, rotation, n_reps=100):
    """Benchmark just the Q@R and O@R^T matmuls."""
    for _ in range(5):
        q_rot = (q.float() @ rotation.float()).to(q.dtype)
        o = (q_rot.float() @ rotation.float().T).to(q.dtype)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_reps):
        q_rot = (q.float() @ rotation.float()).to(q.dtype)
        o = (q_rot.float() @ rotation.float().T).to(q.dtype)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_reps


def main():
    device = "cuda"
    dtype = torch.bfloat16
    B, H_q, H_kv, D = 1, 32, 8, 128
    heads_per_kv = H_q // H_kv

    print("INT4 Attention Kernel Micro-Benchmark")
    print("=" * 60)
    print(f"B={B}, H_q={H_q}, H_kv={H_kv}, D={D}, dtype={dtype}")
    print()

    for seq_kv in [32, 64, 128, 256]:
        print(f"--- seq_kv = {seq_kv} ---")

        # Setup bf16 path
        q = torch.randn(B, H_q, 1, D, device=device, dtype=dtype)
        k = torch.randn(B, H_q, seq_kv, D, device=device, dtype=dtype)
        v = torch.randn(B, H_q, seq_kv, D, device=device, dtype=dtype)

        # Setup INT4 path
        cache = TurboQuantKVCache(B, seq_kv + 64, H_kv, D, dtype=dtype, bits=4).to(device)
        k_raw = torch.randn(B, H_kv, seq_kv, D, device=device, dtype=dtype)
        v_raw = torch.randn(B, H_kv, seq_kv, D, device=device, dtype=dtype)
        pos = torch.arange(seq_kv, device=device)
        cache.store(pos, k_raw, v_raw)

        n_reps = 200

        # Benchmark
        t_bf16 = bench_bf16_sdpa(q, k, v, n_reps) * 1000
        t_int4 = bench_int4_kernel(q, cache, cache.rotation, cache.centroids, H_q, n_reps) * 1000
        t_rot = bench_rotation_only(q, cache.rotation, n_reps) * 1000

        print(f"  bf16 SDPA:     {t_bf16:.3f} ms")
        print(f"  INT4 kernel:   {t_int4:.3f} ms  ({t_int4/t_bf16:.2f}x vs bf16)")
        print(f"    of which Q@R + O@R^T: {t_rot:.3f} ms ({t_rot/t_int4*100:.0f}%)")
        print(f"    kernel only: {t_int4 - t_rot:.3f} ms")

        # Memory comparison
        bf16_kv_mb = B * H_kv * seq_kv * D * 2 * 2 / 1e6  # K + V, bf16
        int4_kv_mb = (
            cache.k_packed[:, :, :seq_kv].numel()
            + cache.v_packed[:, :, :seq_kv].numel()
            + cache.k_mag[:, :, :seq_kv].numel() * 2
            + cache.v_mag[:, :, :seq_kv].numel() * 2
            + cache.k_mean[:, :, :seq_kv].numel() * 2
            + cache.v_mean[:, :, :seq_kv].numel() * 2
        ) / 1e6
        print(f"  KV memory: bf16={bf16_kv_mb:.2f} MB, INT4={int4_kv_mb:.2f} MB ({bf16_kv_mb/int4_kv_mb:.1f}x)")
        print()


if __name__ == "__main__":
    main()
