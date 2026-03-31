import multiprocessing
import time
from typing import Callable, Optional

import torch
from loguru import logger

from fish_speech.models.dac.modded_dac import DAC

# Fixed decode length — pad all inputs to this size so MIOpen only
# ever sees one set of conv shapes (cached after first warmup).
# 512 tokens ~ 23s of audio at 21.5 Hz, covers most TTS outputs.
DECODE_PAD_TO = 512


def _pad_decode_truncate(model, codes, device):
    """Pad codes to consistent shape, decode via DAC, truncate to original length.

    Shared by both in-process and subprocess decode paths.
    """
    seq_len = codes.shape[-1]
    pad_to = max(seq_len, DECODE_PAD_TO)
    pad_to = ((pad_to + DECODE_PAD_TO - 1) // DECODE_PAD_TO) * DECODE_PAD_TO

    if seq_len < pad_to:
        padded = torch.zeros(
            (*codes.shape[:-1], pad_to),
            dtype=codes.dtype, device=device,
        )
        padded[..., :seq_len] = codes
    else:
        padded = codes

    t0 = time.perf_counter()
    with torch.no_grad():
        result = model.from_indices(padded[None])[0].squeeze()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    output_len = int(result.shape[-1] * seq_len / pad_to)
    result = result[..., :output_len]
    elapsed = time.perf_counter() - t0
    return result, elapsed, seq_len, pad_to


def _decoder_subprocess_worker(
    conn: multiprocessing.connection.Connection,
    config_name: str,
    checkpoint_path: str,
    device: str,
    precision_str: str,
):
    """Persistent subprocess that owns the DAC decoder in its own HIP context.

    On RDNA 4 (gfx1201), MIOpen conv kernels page-fault after LLM generation
    corrupts the parent process's HIP page tables. This subprocess gets a clean
    HIP context that never touches LLM allocations.
    """
    import traceback

    try:
        from fish_speech.models.dac.inference import load_model
        from fish_speech.utils.gpu import apply_vram_fraction, auto_detect_rocm_gfx

        auto_detect_rocm_gfx()
        apply_vram_fraction()

        precision = getattr(torch, precision_str, torch.bfloat16)
        logger.info(f"[decoder-subprocess] Loading DAC model on {device}...")
        model = load_model(config_name, checkpoint_path, device=device, precision=precision)

        # Warmup: pre-compile MIOpen solutions on clean GPU
        logger.info("[decoder-subprocess] Warming up MIOpen conv solutions...")
        fake_codes = torch.randint(0, 1024, (10, DECODE_PAD_TO), device=device)
        _pad_decode_truncate(model, fake_codes, device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[decoder-subprocess] Warmup done")

        conn.send({"status": "ready", "sample_rate": model.sample_rate})

        while True:
            msg = conn.recv()
            if msg is None:
                break

            try:
                codes = torch.from_numpy(msg["codes"]).to(device=device)
                result, elapsed, seq_len, pad_to = _pad_decode_truncate(model, codes, device)
                audio_np = result.float().cpu().numpy()
                logger.info(
                    f"[decoder-subprocess] VQ decode: {elapsed:.3f}s "
                    f"(padded {seq_len}->{pad_to} tokens)"
                )
                conn.send({"audio": audio_np})

            except Exception as e:
                logger.error(f"[decoder-subprocess] Decode error: {traceback.format_exc()}")
                conn.send({"error": str(e)})

    except Exception as e:
        logger.error(f"[decoder-subprocess] Init error: {traceback.format_exc()}")
        try:
            conn.send({"status": "error", "error": str(e)})
        except Exception:
            pass
    finally:
        conn.close()
        logger.info("[decoder-subprocess] Exiting")


class VQManager:

    def __init__(self):
        # Make Pylance happy
        self.decoder_model: DAC
        self.load_audio: Callable

        # Subprocess state (None = in-process decode)
        self._decoder_conn: Optional[multiprocessing.connection.Connection] = None
        self._decoder_process: Optional[multiprocessing.Process] = None

    DECODE_PAD_TO = DECODE_PAD_TO

    @property
    def subprocess_active(self):
        return self._decoder_conn is not None

    def launch_decoder_subprocess(self, config_name, checkpoint_path, device, precision):
        """Spawn a persistent decoder subprocess with its own HIP context."""
        parent_conn, child_conn = multiprocessing.Pipe()

        precision_str = {
            torch.float32: "float32",
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
        }.get(precision, "bfloat16")

        proc = multiprocessing.Process(
            target=_decoder_subprocess_worker,
            args=(child_conn, config_name, checkpoint_path, device, precision_str),
            daemon=True,
        )
        proc.start()
        child_conn.close()

        if not parent_conn.poll(timeout=180):
            proc.kill()
            raise RuntimeError("Decoder subprocess timed out during init")

        msg = parent_conn.recv()
        if msg.get("status") != "ready":
            proc.kill()
            raise RuntimeError(f"Decoder subprocess init failed: {msg.get('error', 'unknown')}")

        self._decoder_conn = parent_conn
        self._decoder_process = proc
        logger.info(
            f"Decoder subprocess ready (pid={proc.pid}, "
            f"sample_rate={msg['sample_rate']})"
        )

    def shutdown_decoder_subprocess(self):
        """Gracefully shut down the decoder subprocess."""
        conn = self._decoder_conn
        proc = self._decoder_process
        self._decoder_conn = None
        self._decoder_process = None

        if conn is not None:
            try:
                conn.send(None)
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        if proc is not None:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()

    def decode_vq_tokens(self, codes):
        logger.info(f"VQ features: {codes.shape}")

        if self._decoder_conn is not None:
            return self._decode_via_subprocess(codes)

        # In-process fallback (non-RDNA4 or subprocess not active)
        if isinstance(self.decoder_model, DAC):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            result, elapsed, seq_len, pad_to = _pad_decode_truncate(
                self.decoder_model, codes, codes.device,
            )
            logger.info(
                f"VQ decode: {elapsed:.3f}s "
                f"(padded {seq_len}->{pad_to} tokens)"
            )
            return result

        raise ValueError(f"Unknown model type: {type(self.decoder_model)}")

    def _decode_via_subprocess(self, codes):
        """Send codes to decoder subprocess, receive audio."""
        if not self._decoder_process.is_alive():
            logger.error("Decoder subprocess died, falling back to in-process")
            self.shutdown_decoder_subprocess()
            return self.decode_vq_tokens(codes)

        t0 = time.perf_counter()
        self._decoder_conn.send({"codes": codes.cpu().numpy()})

        if not self._decoder_conn.poll(timeout=120):
            raise RuntimeError("Decoder subprocess timed out")

        response = self._decoder_conn.recv()
        if "error" in response:
            raise RuntimeError(f"Decoder subprocess error: {response['error']}")

        logger.info(f"VQ decode (subprocess): {time.perf_counter() - t0:.3f}s")
        return torch.from_numpy(response["audio"])

    def encode_reference(self, reference_audio, enable_reference_audio):
        if enable_reference_audio and reference_audio is not None:
            if hasattr(self.decoder_model, "spec_transform"):
                sample_rate = self.decoder_model.spec_transform.sample_rate
            else:
                sample_rate = self.decoder_model.sample_rate
            reference_audio_content = self.load_audio(reference_audio, sample_rate)

            audios = torch.from_numpy(reference_audio_content).to(
                device=self.decoder_model.device,
                dtype=next(self.decoder_model.parameters()).dtype,
            )[None, None, :]
            audio_lengths = torch.tensor(
                [audios.shape[2]], device=self.decoder_model.device, dtype=torch.long
            )
            logger.info(
                f"Loaded audio with {audios.shape[2] / sample_rate:.2f} seconds"
            )

            if isinstance(self.decoder_model, DAC):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                prompt_tokens = self.decoder_model.encode(audios, audio_lengths)[0][0]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                logger.info(f"VQ encode: {time.perf_counter() - t0:.3f}s")
                logger.info(f"Encoded prompt: {prompt_tokens.shape}")
            else:
                raise ValueError(f"Unknown model type: {type(self.decoder_model)}")
        else:
            prompt_tokens = None
            logger.info("No reference audio provided")

        return prompt_tokens
