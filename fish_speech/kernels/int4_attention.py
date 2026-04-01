"""INT4 Quantized KV Cache Attention — Triton kernel for ROCm.

Phase 3 optimized: rotation-free. At 4-bit with D=128, raw coordinates
after mean-subtraction + normalization are already ~Gaussian (CLT), so
the PolarQuant rotation adds no quality benefit but costs 43% of the
kernel time. Dropped entirely.

Quantization (per-vector):
  1. Subtract mean, store mean (fp16)
  2. Normalize by L2 norm, store magnitude (fp16)
  3. Quantize each coordinate to 4-bit via Lloyd-Max Gaussian centroids
  4. Pack 2 values per byte → uint8

Attention kernel:
  score_j = Q · centroids[K_idx_j] * K_mag_j/√D + K_mean_j * sum(Q)
  O = Σ softmax(scores) · (centroids[V_idx_j] * V_mag_j/√D + V_mean_j)
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _int4_attn_fused(
    Q_ptr, K_packed_ptr, K_mag_ptr, K_mean_ptr,
    V_packed_ptr, V_mag_ptr, V_mean_ptr,
    C_ptr, Q_sum_ptr, O_ptr,
    seq_kv,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_kpb, stride_kph, stride_kps, stride_kpd,
    stride_vpb, stride_vph, stride_vps, stride_vpd,
    stride_kmb, stride_kmh, stride_kms,
    stride_vmb, stride_vmh, stride_vms,
    stride_qsb, stride_qsh, stride_qss,
    heads_per_kv: tl.constexpr,
    inv_sqrt_d: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Fused multi-head INT4 attention. Grid: (S_new, H_q, B)."""
    q_idx = tl.program_id(0)
    h_q = tl.program_id(1)
    b = tl.program_id(2)
    h_kv = h_q // heads_per_kv

    d_offs = tl.arange(0, HEAD_DIM)

    # Load query vector (original space — no rotation needed)
    q_base = b * stride_qb + h_q * stride_qh + q_idx * stride_qs
    q_vec = tl.load(Q_ptr + q_base + d_offs * stride_qd).to(tl.float32)
    q_sum = tl.load(Q_sum_ptr + b * stride_qsb + h_q * stride_qsh + q_idx * stride_qss).to(tl.float32)

    kp_base = b * stride_kpb + h_kv * stride_kph
    km_base = b * stride_kmb + h_kv * stride_kmh
    vp_base = b * stride_vpb + h_kv * stride_vph
    vm_base = b * stride_vmb + h_kv * stride_vmh

    m_run = -1e30
    d_run = 0.0
    o_run = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for j in range(seq_kv):
        # Dequantize K_j: unpack → centroid lookup
        k_byte = tl.load(K_packed_ptr + kp_base + j * stride_kps + (d_offs // 2) * stride_kpd).to(tl.int32)
        k_idx = tl.where(d_offs % 2 == 0, k_byte & 0x0F, (k_byte >> 4) & 0x0F)
        k_quant = tl.load(C_ptr + k_idx).to(tl.float32)

        k_mag = tl.load(K_mag_ptr + km_base + j * stride_kms).to(tl.float32)
        k_mean = tl.load(K_mean_ptr + km_base + j * stride_kms).to(tl.float32)

        # Score: Q · K_deq where K_deq = centroid * mag/√D + mean
        # = Q · centroid * mag/√D + Q · mean
        # = sum(Q * centroid) * mag/√D + sum(Q) * mean
        score = tl.sum(q_vec * k_quant) * k_mag * inv_sqrt_d + k_mean * q_sum
        score = score * inv_sqrt_d  # 1/√D attention scaling

        # Online softmax
        m_new = tl.maximum(m_run, score)
        exp_old = tl.exp(m_run - m_new)
        exp_new = tl.exp(score - m_new)
        d_new = d_run * exp_old + exp_new

        # Dequantize V_j
        v_byte = tl.load(V_packed_ptr + vp_base + j * stride_vps + (d_offs // 2) * stride_vpd).to(tl.int32)
        v_idx = tl.where(d_offs % 2 == 0, v_byte & 0x0F, (v_byte >> 4) & 0x0F)
        v_quant = tl.load(C_ptr + v_idx).to(tl.float32)

        v_mag = tl.load(V_mag_ptr + vm_base + j * stride_vms).to(tl.float32)
        v_mean = tl.load(V_mean_ptr + vm_base + j * stride_vms).to(tl.float32)
        v_deq = v_quant * v_mag * inv_sqrt_d + v_mean

        o_run = o_run * exp_old + exp_new * v_deq
        m_run = m_new
        d_run = d_new

    o_run = o_run / d_run

    o_base = b * stride_ob + h_q * stride_oh + q_idx * stride_os
    tl.store(O_ptr + o_base + d_offs * stride_od, o_run.to(tl.bfloat16))


# ── Python wrappers ─────────────────────────────────────────────────────────


def _launch_fused(q, k_packed, k_mag, k_mean, v_packed, v_mag, v_mean,
                  centroids, q_sum, seq_kv, n_heads_q, heads_per_kv, head_dim):
    B, _, S_new, _ = q.shape
    o = torch.empty_like(q)

    grid = (S_new, n_heads_q, B)
    _int4_attn_fused[grid](
        q, k_packed, k_mag, k_mean, v_packed, v_mag, v_mean,
        centroids, q_sum, o,
        seq_kv=seq_kv,
        stride_qb=q.stride(0), stride_qh=q.stride(1),
        stride_qs=q.stride(2), stride_qd=q.stride(3),
        stride_ob=o.stride(0), stride_oh=o.stride(1),
        stride_os=o.stride(2), stride_od=o.stride(3),
        stride_kpb=k_packed.stride(0), stride_kph=k_packed.stride(1),
        stride_kps=k_packed.stride(2), stride_kpd=k_packed.stride(3),
        stride_vpb=v_packed.stride(0), stride_vph=v_packed.stride(1),
        stride_vps=v_packed.stride(2), stride_vpd=v_packed.stride(3),
        stride_kmb=k_mag.stride(0), stride_kmh=k_mag.stride(1),
        stride_kms=k_mag.stride(2),
        stride_vmb=v_mag.stride(0), stride_vmh=v_mag.stride(1),
        stride_vms=v_mag.stride(2),
        stride_qsb=q_sum.stride(0), stride_qsh=q_sum.stride(1),
        stride_qss=q_sum.stride(2),
        heads_per_kv=heads_per_kv,
        inv_sqrt_d=1.0 / math.sqrt(head_dim),
        HEAD_DIM=head_dim,
    )
    return o


def int4_polar_attention(
    q, k_packed, k_mag, k_mean, v_packed, v_mag, v_mean,
    centroids, rotation, seq_kv,
):
    """Single-head INT4 attention (backward-compatible test wrapper).

    rotation arg kept for API compat but ignored (rotation-free).
    """
    num_q, head_dim = q.shape
    q_sum = q.float().sum(dim=-1)

    q_4d = q[None, None]
    qs_3d = q_sum[None, None]
    kp = k_packed[None, None]
    km = k_mag[None, None, :, None]
    kmu = k_mean[None, None, :, None]
    vp = v_packed[None, None]
    vm = v_mag[None, None, :, None]
    vmu = v_mean[None, None, :, None]

    o = _launch_fused(q_4d, kp, km, kmu, vp, vm, vmu,
                      centroids, qs_3d, seq_kv, 1, 1, head_dim)
    return o.squeeze(0).squeeze(0)


def int4_attention_multihead(q, cache, n_heads_q, rotation, centroids):
    """Multi-head INT4 attention — rotation-free, single fused kernel."""
    heads_per_kv = n_heads_q // cache.n_heads
    seq_kv = cache._seq_high_water
    D = q.shape[-1]

    q_sum = q.float().sum(dim=-1)  # (B, H_q, S_new)

    return _launch_fused(
        q, cache.k_packed, cache.k_mag, cache.k_mean,
        cache.v_packed, cache.v_mag, cache.v_mean,
        centroids, q_sum, seq_kv,
        n_heads_q, heads_per_kv, D,
    )
