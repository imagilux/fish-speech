# Fish Speech — AMD ROCm Fork

**Multilingual text-to-speech on AMD GPUs.** This is the [Imagilux](https://github.com/imagilux) fork of [Fish Audio S2 Pro](https://huggingface.co/fishaudio/s2-pro), optimized for AMD RDNA 4 hardware (RX 9070 XT / gfx1201) running ROCm 7.2.

> Based on [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) — see their [technical report](https://arxiv.org/abs/2603.08823) for model details.

## What's different from upstream

- **Two-service architecture** — API server (GPU, models) + lightweight Gradio WebUI (HTTP client, 507MB image)
- **ROCm RDNA 4 support** — subprocess DAC decoder for HIP context isolation, auto-detection of gfx1201/gfx1200
- **[miopen-conv-fix](https://github.com/Imagilux/miopen-conv-fix)** — our companion library that patches PyTorch's `workspace=0` bug for large conv layers via MIOpen Immediate Mode API, giving a 34x VQ decode speedup on ROCm
- **MIOpen / Triton cache persistence** across container restarts
- **Async chunk streaming** — 5x faster time-to-first-audio
- **No CUDA Docker support** — ROCm + CPU only (PyTorch source code is backend-agnostic)

## Quick Start

Requirements: AMD GPU with ROCm 7.2+ kernel driver, Docker with compose plugin.

```bash
git clone https://github.com/imagilux/fish-speech.git
cd fish-speech
```

Download the model (INT8 recommended for 16GB VRAM):

```bash
# Full precision (11GB)
huggingface-cli download fishaudio/s2-pro --local-dir checkpoints/s2-pro

# INT8 quantized (6.5GB) — recommended for RX 9070 XT
huggingface-cli download fishaudio/s2-pro-int8 --local-dir checkpoints/fish-speech-s2-pro-int8
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

Environment variables (set in `.env` or pass to compose):

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND` | `rocm` | `rocm` or `cpu` |
| `UV_EXTRA` | `rocm72` | PyTorch index (`rocm72` or `cpu`) |
| `LLAMA_CHECKPOINT_PATH` | `checkpoints/s2-pro` | Model checkpoint path |
| `DECODER_CHECKPOINT_PATH` | `checkpoints/s2-pro/codec.pth` | DAC codec path |
| `VRAM_FRACTION` | `0.95` | GPU memory cap (0-1) |
| `MAX_SEQ_LEN` | `4096` | Max sequence length |
| `COMPILE` | `0` | Enable torch.compile |

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
