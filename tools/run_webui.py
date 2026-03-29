import os
import time
from argparse import ArgumentParser
from pathlib import Path

import pyrootutils
import torch
from loguru import logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.dac.modded_dac import DAC
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.gpu import auto_detect_rocm_gfx, check_vram_and_advise
from fish_speech.utils.schema import ServeTTSRequest
from tools.webui import build_app
from tools.webui.inference import get_inference_wrapper

# Make einx happy
os.environ["EINX_FILTER_TRACEBACK"] = "false"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--llama-checkpoint-path",
        type=Path,
        default="checkpoints/s2-pro",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=Path,
        default="checkpoints/s2-pro/codec.pth",
    )
    parser.add_argument("--decoder-config-name", type=str, default="modded_dac_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-gradio-length", type=int, default=0)
    parser.add_argument("--theme", type=str, default="light")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.precision = torch.half if args.half else torch.bfloat16

    # Known issue: PyTorch ROCm passes workspace=0 to MIOpen for conv ops,
    # forcing fallback to ConvDirectNaiveConvFwd (~10-12x slower).
    # Tracked: pytorch/pytorch#150168, ROCm/MIOpen#3650, ROCm/TheRock#3077
    # miopen-conv-fix (Imagilux/miopen-conv-fix) extension exists but causes
    # segfaults when LLM + decoder share VRAM. Disabled until stable.

    # Optional VRAM cap — set VRAM_FRACTION (0.0-1.0) to prevent system freeze
    # on memory-constrained GPUs. Unset or 0 = no cap (default).
    vram_fraction = float(os.environ.get("VRAM_FRACTION", "0"))
    if 0 < vram_fraction <= 1 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(vram_fraction)
        total_mem = torch.cuda.get_device_properties(0).total_memory
        logger.info(
            f"VRAM cap: {vram_fraction:.0%} "
            f"({vram_fraction * total_mem / 1e9:.1f}GB / {total_mem / 1e9:.1f}GB)"
        )

    auto_detect_rocm_gfx()
    check_vram_and_advise(args.llama_checkpoint_path)

    # Check if MPS or CUDA is available
    if torch.backends.mps.is_available():
        args.device = "mps"
        logger.info("mps is available, running on mps.")
    elif torch.xpu.is_available():
        args.device = "xpu"
        logger.info("XPU is available, running on XPU.")
    elif not torch.cuda.is_available():
        logger.info("CUDA is not available, running on CPU.")
        args.device = "cpu"

    logger.info("Loading Llama model...")
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=args.llama_checkpoint_path,
        device=args.device,
        precision=args.precision,
        compile=args.compile,
    )

    logger.info("Loading VQ-GAN model...")
    decoder_model = load_decoder_model(
        config_name=args.decoder_config_name,
        checkpoint_path=args.decoder_checkpoint_path,
        device=args.device,
        precision=args.precision,
    )

    logger.info("Decoder model loaded, warming up...")

    # Create the inference engine
    inference_engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        compile=args.compile,
        precision=args.precision,
    )

    # Dry run to check if the model is loaded correctly and avoid the first-time latency
    list(
        inference_engine.inference(
            ServeTTSRequest(
                text="Hello world.",
                references=[],
                reference_id=None,
                max_new_tokens=1024,
                chunk_length=200,
                top_p=0.7,
                repetition_penalty=1.5,
                temperature=0.7,
                format="wav",
            )
        )
    )

    # Warm up the DAC decoder with the fixed padded shape (DECODE_PAD_TO=512).
    # All decode calls pad to this size, so MIOpen only needs to cache one set
    # of conv kernel shapes. First run triggers exhaustive search (~150s),
    # subsequent runs hit the cache (~4s).
    if isinstance(decoder_model, DAC) and args.device == "cuda":
        from fish_speech.inference_engine.vq_manager import VQManager
        pad_to = VQManager.DECODE_PAD_TO
        logger.info(f"Warming up DAC decoder (fixed shape: {pad_to} tokens)...")
        t0 = time.perf_counter()
        with torch.no_grad():
            synthetic_indices = torch.randint(
                0, 1024, (1, 10, pad_to), device=args.device, dtype=torch.long
            )
            _ = decoder_model.from_indices(synthetic_indices)
            torch.cuda.synchronize()
        del synthetic_indices, _
        torch.cuda.empty_cache()
        logger.info(f"DAC decoder warmup done in {time.perf_counter() - t0:.1f}s")

    logger.info("Warming up done, launching the web UI...")

    # Get the inference function with the immutable arguments
    inference_fct = get_inference_wrapper(inference_engine)

    app = build_app(inference_fct, args.theme)
    app.launch()
