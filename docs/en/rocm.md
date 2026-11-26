# ROCm / AMD GPU Support (RDNA 4)

This fork adds full ROCm support for AMD RDNA 4 GPUs (RX 9070 XT / gfx1201), achieving performance comparable to NVIDIA CUDA with a subprocess-based architecture that works around HIP driver limitations.

## Quick Start (Docker)

```bash
# Clone and configure
git clone https://github.com/imagilux/fish-speech.git
cd fish-speech
cp .env.rocm_example .env  # Edit: set the checkpoint paths

# Build and run
BACKEND=rocm UV_EXTRA=rocm72 docker compose -f compose.yml -f compose.rocm.yml up -d server

# Test
curl -X POST http://localhost:8080/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "format": "wav"}' \
  -o test.wav
```

## Architecture

Fish Speech on RDNA 4 uses a **two-process architecture** that isolates LLM generation from DAC audio decoding, similar to [vLLM-Omni](https://github.com/vllm-project/vllm-omni)'s disaggregated stage design:

```
Parent Process                          Decoder Subprocess
==========================              ==========================
uvicorn API server                      Own HIP context (isolated)
LLM worker thread (7.2 tok/s)          DAC model + miopen-conv-fix
  generates VQ tokens                   88 conv layers patched
  every 25 tokens → pipe.send    →     Decodes chunk (0.056s)
  (LLM keeps generating)               Returns audio via pipe
  ...                              ←
  next 25 tokens → pipe.send     →     Decodes next chunk
  ...                              ←
Yields audio chunks to HTTP client
```

### Why Two Processes?

On RDNA 4, MIOpen convolution kernels **GPU page-fault** when executed after LLM generation in the same process. The root cause is a HIP driver bug where heavy `hipMalloc`/`hipFree` cycles from LLM token generation leave stale GPU page table entries. Native PyTorch conv (using `ConvDirectNaive` with workspace=0) is unaffected but 34x slower.

The subprocess gets a completely clean HIP context that never touches LLM memory allocations, so MIOpen's optimized GEMM kernels work correctly.

This is auto-detected on gfx1201/gfx1200 via `USE_SUBPROCESS_DECODER=auto`.

## miopen-conv-fix

[miopen-conv-fix](https://github.com/imagilux/miopen-conv-fix) is a C++ extension that fixes PyTorch's workspace=0 bug ([pytorch#150168](https://github.com/pytorch/pytorch/issues/150168)) for MIOpen convolutions.

### How It Works

MIOpen ships **pre-compiled convolution kernels** (GEMM, implicit GEMM, Winograd) optimized by AMD for each GPU architecture. These kernels need temporary workspace memory. PyTorch's bug passes `workspace=0`, forcing MIOpen to use `ConvDirectNaive` — a slow fallback needing no workspace.

miopen-conv-fix calls MIOpen's **Immediate Mode API** directly:

1. **Enumerate** — queries available pre-compiled solutions for each conv shape
2. **Select** — picks the best non-blacklisted solution by heuristic ranking
3. **Compile** — prepares the solution (loads pre-built kernel, not true compilation)
4. **Cache** — stores solution ID + workspace size for instant reuse
5. **Execute** — allocates workspace and dispatches via `miopenConvolutionForwardImmediate()`

First decode takes ~3-5s (solution evaluation), subsequent calls are cached.

### miopen-conv-fix vs torch.compile

These are independent systems:

| | torch.compile (Inductor/Triton) | miopen-conv-fix |
|---|---|---|
| **What** | Generates new GPU kernels from Python model | Uses AMD's pre-built MIOpen kernels |
| **Path** | Python → FX graph → Inductor → Triton → HIP | MIOpen C API → pre-built kernels → HIP |
| **RDNA 4 perf** | **4.6x slower** than eager (immature Triton codegen) | **34x faster** than naive fallback |
| **Recommendation** | `COMPILE=0` on gfx1201 | Always enabled (auto-detected) |

## Streaming Chunk Decode

When `streaming=true`, audio chunks are produced while the LLM is still generating:

- Every 25 new tokens (~1.16s audio), a chunk is sent to the decoder subprocess
- Each chunk includes 25 tokens of left context for the quantizer's windowed transformer (window=128, causal)
- The context audio is trimmed — only new audio is yielded to the client
- Per-chunk decode: **0.056s** (padded to 64 tokens for MIOpen shape caching)

**Time-to-first-audio: ~3.7s** (vs ~26s without streaming)

## Performance

Benchmarked on AMD RX 9070 XT (gfx1201, 16GB) with INT8 quantized model:

| Configuration | Render Time | Time-to-First-Audio | LLM tok/s | VQ Decode |
|---|---|---|---|---|
| ROCm 7.2.0, COMPILE=1, no fix | 85s | 85s | ~2 | 7.98s (naive) |
| ROCm 7.2.1, COMPILE=1, host | 34.9s | 34.9s | ~7.2 | ~0.4s |
| Docker, COMPILE=0, subprocess | **26.5s** | **26.5s** | 7.28 | 0.39s |
| Docker, COMPILE=0, streaming | **25.6s** | **3.7s** | 7.21 | 8x 0.056s |

## Docker Configuration

### compose.rocm.yml

Key settings for RDNA 4:

| Setting | Value | Why |
|---|---|---|
| `COMPILE` | `0` | torch.compile is slower on gfx1201 |
| `VRAM_FRACTION` | `0.95` | Safety cap, prevents GPU hang on OOM |
| `MAX_SEQ_LEN` | `4096` | Limits KV cache VRAM (~0.6GB vs ~4.8GB at 32768) |
| `MIOPEN_FIND_MODE` | `3` | Fast heuristic search |
| `HSA_ENABLE_SDMA` | `0` | Disables async DMA (stability) |
| `GPU_MAX_HW_QUEUES` | `1` | Single GPU queue (stability) |
| `mem_limit` | `40g` | Prevents host freeze on OOM |
| `ulimits.nofile` | `65536` | Triton + MIOpen exhaust default fd limit |

### Cache Volumes

```yaml
volumes:
  - miopen-kernels:/app/.cache/miopen      # MIOpen kernel cache (used)
  - miopen-userdb:/app/.config/miopen      # MIOpen user DB (used)
  - triton-cache:/app/.triton              # Triton cache (unused with COMPILE=0)
```

### Requirements

- ROCm 7.2.1+ (`rocm/dev-ubuntu-24.04:7.2.1` base image)
- `/dev/kfd` and `/dev/dri` device access
- `video` and `render` group membership
- `seccomp=unconfined` (ROCm requires it)

## Known Issues

1. **GPU page fault in same-process decode** — solved by subprocess architecture (auto-detected on RDNA 4)
2. **torch.compile 4.6x slower** — use `COMPILE=0` on gfx1201
3. **Weight norm parametrizations** — baked to plain tensors at load time for stable GPU memory access
4. **PyTorch 2.11 partial fix** — workspace=0 bug fixed for small conv layers only; large layers (>50MB workspace) still need miopen-conv-fix
