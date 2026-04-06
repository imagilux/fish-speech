# Fish Speech — AMD ROCm Fork

**Multilingual text-to-speech on AMD GPUs.** This is the [Imagilux](https://github.com/imagilux) fork of [Fish Audio S2 Pro](https://huggingface.co/fishaudio/s2-pro), optimized for AMD RDNA 3/4 hardware (RX 7000/9000 series) running ROCm 7.2.

> Based on [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) — see their [technical report](https://arxiv.org/abs/2603.08823) for model details.

## What's different from upstream

- **Two-service architecture** — API server (GPU, models) + lightweight Gradio WebUI (HTTP client, 507MB image)
- **ROCm RDNA 3/4 support** — subprocess DAC decoder for HIP context isolation, auto-detection of consumer GPU architectures (gfx1100–gfx1201)
- **[miopen-conv-fix](https://github.com/Imagilux/miopen-conv-fix)** — our companion library that patches PyTorch's `workspace=0` bug for large conv layers via MIOpen Immediate Mode API, giving a 34x VQ decode speedup on ROCm
- **MIOpen / Triton cache persistence** across container restarts
- **Async chunk streaming** — 5x faster time-to-first-audio
- **Chunked VQ decode** — long texts decoded in fixed-size chunks to stay within warmed MIOpen conv shapes, avoiding timeouts and VRAM spikes
- **No CUDA Docker support** — ROCm + CPU only (PyTorch source code is backend-agnostic)

## Quick Start

Requirements: AMD GPU with ROCm 7.2+ kernel driver, Docker with compose plugin.

```bash
git clone https://github.com/imagilux/fish-speech.git
cd fish-speech
```

Download the model (INT8 recommended for 16 GB VRAM):

```bash
# Full precision (11 GB)
huggingface-cli download fishaudio/s2-pro --local-dir checkpoints/s2-pro

# INT8 quantized (6.5 GB) — recommended for RX 7000/9000 series
huggingface-cli download fishaudio/s2-pro-int8 --local-dir checkpoints/fish-speech-s2-pro-int8
```

Copy and edit the example environment file:

```bash
cp .env.rocm_example .env
# Edit .env — set LLAMA_CHECKPOINT_PATH and DECODER_CHECKPOINT_PATH
```

Start both services:

```bash
docker compose -f compose.yml -f compose.rocm.yml up
```

- **WebUI**: http://localhost:7860
- **API**: http://localhost:8080

## Architecture

```
                  +-------------------+
  Browser ------->|  WebUI (Gradio)   |  507MB image, no GPU
  :7860           |  Dockerfile.webui |
                  +--------+----------+
                           | HTTP
                  +--------v----------+
                  |  API Server (kui) |  29GB image, ROCm GPU
  curl :8080 ---->|  Dockerfile       |
                  |  + LLM (INT8)     |
                  |  + DAC subprocess |
                  +-------------------+
```

| Component | Image | GPU | Purpose |
|-----------|-------|-----|---------|
| `server` | `docker/Dockerfile` (29GB) | Yes | Model loading, TTS inference, reference management |
| `webui` | `docker/Dockerfile.webui` (507MB) | No | Gradio UI, calls server API |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/tts` | Text-to-speech (streaming or batch) |
| `GET` | `/v1/references/list` | List saved voice references |
| `POST` | `/v1/references/add` | Add a new voice reference |
| `DELETE` | `/v1/references/delete` | Delete a voice reference |
| `GET` | `/v1/health` | Health check |

## Configuration

Copy `.env.rocm_example` to `.env` and set at minimum the checkpoint paths. All other variables have sensible defaults.

**Runtime variables** (read when the container starts):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_CHECKPOINT_PATH` | `checkpoints/s2-pro` | LLM checkpoint directory |
| `DECODER_CHECKPOINT_PATH` | `checkpoints/s2-pro/codec.pth` | DAC codec weights |
| `MAX_SEQ_LEN` | `32768` | KV cache sequence length (use `4096` to save ~4 GB VRAM) |
| `VRAM_FRACTION` | `0` (no limit) | Fraction of GPU VRAM PyTorch may allocate |
| `RENDER_GID` | `993` | Host render group GID (`stat -c '%g' /dev/kfd`) |
| `COMPILE` | `0` | Enable `torch.compile` (experimental) |

**Build args** (passed to `docker compose build`, defaults work for ROCm):

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND` | `rocm` | `rocm` or `cpu` — selects the PyTorch build |
| `UV_EXTRA` | `rocm72` | uv extras group (`rocm72` or `cpu`) |

## Development

```bash
# Run tests (no GPU required)
uv run --extra rocm72 pytest tests/ -v

# Dev container
docker build -f docker/Dockerfile --target dev -t fish-speech-dev:rocm .
```

## Related Projects

- **[Imagilux/miopen-conv-fix](https://github.com/Imagilux/miopen-conv-fix)** — MIOpen conv layer fix for ROCm, used by the DAC decoder subprocess
- **[fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)** — upstream project

## License

Code: [Fish Audio Research License](LICENSE) (Copyright 39 AI, INC)
Model weights: [Fish Audio Research License](https://huggingface.co/fishaudio/s2-pro/blob/main/LICENSE.md)

Built with [Fish Audio](https://fish.audio/).

## Citation

```bibtex
@misc{liao2026fishaudios2technical,
      title={Fish Audio S2 Technical Report},
      author={Shijia Liao and Yuxuan Wang and Songting Liu and Yifan Cheng and Ruoyi Zhang and Tianyu Li and Shidong Li and Yisheng Zheng and Xingwei Liu and Qingzheng Wang and Zhizhuo Zhou and Jiahua Liu and Xin Chen and Dawei Han},
      year={2026},
      eprint={2603.08823},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2603.08823},
}
```
