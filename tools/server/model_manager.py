import atexit
import os

import torch
from loguru import logger

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.gpu import apply_vram_fraction, auto_detect_rocm_gfx, check_vram_and_advise
from fish_speech.utils.schema import ServeTTSRequest


def _should_use_subprocess_decoder():
    """Auto-detect RDNA 4 where subprocess decoder is needed.

    On gfx1201/gfx1200, MIOpen conv kernels page-fault after LLM generation
    due to a HIP driver bug. A subprocess with its own HIP context avoids this.
    """
    if not torch.cuda.is_available():
        return False
    env = os.environ.get("USE_SUBPROCESS_DECODER", "auto").lower()
    if env in ("true", "1"):
        return True
    if env in ("false", "0"):
        return False
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", "")
    return "gfx1201" in arch or "gfx1200" in arch


class ModelManager:
    def __init__(
        self,
        mode: str,
        device: str,
        half: bool,
        compile: bool,
        llama_checkpoint_path: str,
        decoder_checkpoint_path: str,
        decoder_config_name: str,
    ) -> None:

        self.mode = mode
        self.device = device
        self.half = half
        self.compile = compile
        self.precision = torch.half if half else torch.bfloat16

        auto_detect_rocm_gfx()
        apply_vram_fraction()
        check_vram_and_advise(llama_checkpoint_path)

        if torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("mps is available, running on mps.")
        elif not torch.cuda.is_available():
            self.device = "cpu"
            logger.info("CUDA is not available, running on CPU.")

        use_subprocess = _should_use_subprocess_decoder()

        # Load models sequentially — offload LLM before loading decoder
        # so they never coexist on GPU simultaneously during startup.
        self.load_llama_model(
            llama_checkpoint_path, self.device, self.precision, self.compile, self.mode
        )
        if self.device == "cuda" and hasattr(self.llama_queue, "offload_to_cpu"):
            self.llama_queue.offload_to_cpu()

        self.load_decoder_model(
            decoder_config_name, decoder_checkpoint_path, self.device, self.precision
        )

        # Move parent DAC to CPU BEFORE launching subprocess — gives the
        # subprocess full VRAM for model load + MIOpen warmup.
        if use_subprocess:
            self.decoder_model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Parent DAC model moved to CPU (decode via subprocess)")

        self.tts_inference_engine = TTSInferenceEngine(
            llama_queue=self.llama_queue,
            decoder_model=self.decoder_model,
            precision=self.precision,
            compile=self.compile,
        )

        if use_subprocess:
            logger.info("RDNA 4 detected — launching decoder subprocess")
            self.tts_inference_engine.launch_decoder_subprocess(
                config_name=decoder_config_name,
                checkpoint_path=decoder_checkpoint_path,
                device=self.device,
                precision=self.precision,
            )
            atexit.register(self.tts_inference_engine.shutdown_decoder_subprocess)
        else:
            logger.info("Using in-process decoder")

    def load_llama_model(
        self, checkpoint_path, device, precision, compile, mode
    ) -> None:
        if mode == "tts":
            self.llama_queue = launch_thread_safe_queue(
                checkpoint_path=checkpoint_path,
                device=device,
                precision=precision,
                compile=compile,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")
        logger.info("LLAMA model loaded.")

    def load_decoder_model(
        self, config_name, checkpoint_path, device, precision=None
    ) -> None:
        self.decoder_model = load_decoder_model(
            config_name=config_name,
            checkpoint_path=checkpoint_path,
            device=device,
            precision=precision,
        )
        logger.info("Decoder model loaded.")
