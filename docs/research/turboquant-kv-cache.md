# TurboQuant KV Cache — Research Notes

## Status: Proof of concept — needs native INT4 attention kernel

The current implementation (`fish_speech/models/text2semantic/turboquant.py`)
compresses KV vectors to 4-bit using PolarQuant-style polar coordinate
quantization, but **decompresses back to bf16 before attention**. This means:

- No actual VRAM savings during attention (the full bf16 tensors are materialized)
- Additional overhead from quantize + dequantize per step
- The compression ratio is theoretical only — never realized in practice

### What we proved

| Test | Result |
|------|--------|
| Pack/unpack roundtrip | Exact (lossless bit packing) |
| 4-bit cosine similarity | > 0.95 (good reconstruction quality) |
| 2-bit cosine similarity | > 0.85 (usable for long-context) |
| VRAM with compress→decompress | **Worse** than bf16 (extra buffers) |
| VRAM with compressed-only storage | 6.5 GB idle (vs 7 GB bf16 baseline) |
| TTS output quality | Subjectively equivalent at 4-bit |

### Why the current approach doesn't work

`F.scaled_dot_product_attention()` requires bf16/fp16/fp32 inputs. There is no
PyTorch API that accepts quantized KV cache natively. The only options are:

1. **Custom attention kernel** that reads INT4 packed data directly and
   dequantizes per-tile inside the kernel (never materializing full bf16)
2. **Triton kernel** that does the same but is portable across GPU backends
3. **Upstream support** — wait for PyTorch/ROCm to add quantized attention

Option 1 (HIP kernel) is the path forward for our ROCm-focused repo.

### Algorithm recap (PolarQuant)

```
WRITE (per KV vector, dim=128):
  1. Subtract mean, store mean (fp16)         — 2 bytes
  2. Compute L2 norm, store magnitude (fp16)  — 2 bytes
  3. Normalize → unit vector
  4. Multiply by random orthogonal matrix R   — coordinates now ~iid N(0,1)
  5. Quantize each coordinate to 4-bit        — 64 bytes (128 * 4bit / 8)
  Total: 68 bytes vs 256 bytes (bf16) = 3.8x compression

READ (inside attention kernel):
  1. Load packed 4-bit indices (64 bytes)
  2. Look up Lloyd-Max centroids (table in shared memory)
  3. Multiply by R^T (inverse rotation)
  4. Scale by magnitude, add mean
  5. Use directly in Q·K^T dot product
```

The key insight: steps 1-4 can be **fused into the attention kernel's inner
loop**, dequantizing one tile of K/V at a time into registers/shared memory.
The full bf16 KV cache is never materialized in global memory.

## Results: Native INT4 Triton Kernel (Phase 1+2 complete)

| Metric | bf16 KVCache | INT4 Kernel |
|--------|-------------|-------------|
| Idle VRAM | 7.0 GB | **5.6 GB** (-1.4 GB) |
| Post-inference VRAM | ~11 GB | **9.1 GB** (-1.9 GB) |
| Kernel accuracy | reference | cosine 0.998 |
| TTS latency | 25.3s | 25.3s (no overhead) |
| TTS output | Valid WAV | Valid WAV |

The Triton kernel reads packed INT4 bytes directly on GPU. No bf16 KV
tensors are materialized in global memory. GQA (32 Q heads → 8 KV heads)
is handled by mapping Q heads to KV heads in the Python wrapper.

## Roadmap: Remaining work

### Phase 1: HIP kernel prototype (target: proof of concept)

**Goal:** A HIP kernel that computes `softmax(Q·K^T/√d)·V` where K and V
are stored in packed INT4 format with per-vector magnitude/mean.

**Scope:**
- Single-head attention (no GQA yet)
- Fixed head_dim=128
- Batch=1, no padding/masking
- Causal attention only

**Files to create:**
```
fish_speech/kernels/
├── __init__.py
├── int4_attention.hip        # HIP kernel source
├── int4_attention_binding.cpp # PyBind11 / torch extension binding
└── setup.py                  # Build extension
```

**Kernel design:**
```
__global__ void int4_polar_attention(
    // Q: (seq_new, head_dim) bf16 — current query
    // K_packed: (seq_all, packed_dim) uint8 — packed 4-bit K
    // K_mag: (seq_all,) fp16 — K magnitudes
    // K_mean: (seq_all,) fp16 — K means
    // V_packed: (seq_all, packed_dim) uint8 — packed 4-bit V
    // V_mag: (seq_all,) fp16
    // V_mean: (seq_all,) fp16
    // R: (head_dim, head_dim) fp16 — rotation matrix (in constant memory)
    // centroids: (16,) fp16 — Lloyd-Max centroids (in shared memory)
    // Output: (seq_new, head_dim) bf16
)
{
    // Per-thread-block: one query position
    // Tile over K positions (e.g., 32 at a time):
    //   1. Load K_packed tile into shared memory
    //   2. Dequantize: unpack → centroid lookup → R^T rotation → scale
    //   3. Compute partial Q·K^T scores
    //   4. Track running softmax (online softmax algorithm)
    //   5. Load V_packed tile, dequantize
    //   6. Accumulate weighted V
    // Final: normalize by softmax denominator, write output
}
```

**Key optimizations for RDNA 4 (gfx1201):**
- 16KB shared memory per CU — fits centroids (32B) + one K tile
- Wave64 execution — 64 threads per wavefront
- DS (LDS) permute for intra-wave reductions
- Use `__builtin_amdgcn_ds_bpermute` for warp shuffles
- Pack 2 INT4 values per byte → 2x memory bandwidth vs INT8

### Phase 2: Integration and correctness

**Goal:** Drop-in replacement for KVCache that uses the HIP kernel.

- Modify `TurboQuantKVCache.update()` to store compressed-only (no bf16 output)
- Replace `F.scaled_dot_product_attention` call in `Attention.forward()` with
  the custom kernel when KV cache is quantized
- Correctness tests: compare output vs bf16 attention (tolerance < 1e-2)
- Handle prefill (multiple query positions) and decode (single position)

### Phase 3: GQA + masking + performance

**Goal:** Production-ready kernel with full feature support.

- Grouped-query attention (n_heads=32, n_kv_heads=8 → 4x repeat)
- Causal masking
- Variable sequence lengths
- Benchmark vs bf16 FlashAttention:
  - Target: >80% of bf16 FLOPS (dequant overhead < 20%)
  - Target: 3.8x less VRAM for KV cache

### Phase 4: 2-bit and Triton port

**Goal:** Maximum compression + portability.

- 2-bit variant (7x compression) for long-context (32k+ tokens)
- Triton kernel for portability across ROCm versions
- Benchmark quality degradation at 2-bit on TTS output

## References

- [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874) — the theory
- [PolarQuant (AISTATS 2026)](https://arxiv.org/abs/2502.02617) — polar decomposition
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) — tiling strategy to adapt
- [KIVI](https://arxiv.org/abs/2402.02750) — prior KV cache quantization (no custom kernel)
- [ROCm composable_kernel](https://github.com/ROCm/composable_kernel) — AMD's attention kernels
- [Triton ROCm backend](https://github.com/triton-lang/triton) — portable GPU kernels
