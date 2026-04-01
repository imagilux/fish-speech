import multiprocessing
import time
from typing import Callable, Optional

import numpy as np
import torch
from loguru import logger

from fish_speech.models.dac.modded_dac import DAC

# Fixed decode length — pad all inputs to this size so MIOpen only
# ever sees one set of conv shapes (cached after first warmup).
# 512 tokens ~ 23s of audio at 21.5 Hz, covers most TTS outputs.
DECODE_PAD_TO = 512


CHUNK_DECODE_PAD_TO = 64  # Smaller pad target for streaming chunks


def _pad_decode_truncate(model, codes, device, pad_multiple=DECODE_PAD_TO):
    """Pad codes to consistent shape, decode via DAC, truncate to original length.

    Shared by both in-process and subprocess decode paths.
    pad_multiple: pad to this multiple (512 for batch, 64 for streaming chunks).
    """
    seq_len = codes.shape[-1]
    pad_to = max(seq_len, pad_multiple)
    pad_to = ((pad_to + pad_multiple - 1) // pad_multiple) * pad_multiple

    if seq_len < pad_to:
        padded = torch.zeros(
            (*codes.shape[:-1], pad_to),
            dtype=codes.dtype, device=device,
        )
        padded[..., :seq_len] = codes
    else:
        padded = codes

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        result = model.from_indices(padded[None])[0].squeeze()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    output_len = int(result.shape[-1] * seq_len / pad_to)
    result = result[..., :output_len]
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

        # Warmup: pre-compile MIOpen solutions for both batch and chunk shapes
        logger.info("[decoder-subprocess] Warming up MIOpen conv solutions...")
        fake_codes = torch.randint(0, 1024, (10, DECODE_PAD_TO), device=device)
        _pad_decode_truncate(model, fake_codes, device, pad_multiple=DECODE_PAD_TO)
        fake_chunk = torch.randint(0, 1024, (10, CHUNK_DECODE_PAD_TO), device=device)
        _pad_decode_truncate(model, fake_chunk, device, pad_multiple=CHUNK_DECODE_PAD_TO)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[decoder-subprocess] Warmup done (batch + chunk shapes)")

        conn.send({"status": "ready", "sample_rate": model.sample_rate})

        while True:
            msg = conn.recv()
            if msg is None:
                break

            action = msg.get("action", "decode")

            try:
                if action == "encode":
                    audio_np = msg["audio"]
                    audio_len = msg["audio_length"]
                    audios = torch.from_numpy(audio_np).to(
                        device=device,
                        dtype=next(model.parameters()).dtype,
                    )[None, None, :]
                    audio_lengths = torch.tensor(
                        [audio_len], device=device, dtype=torch.long,
                    )
                    t0 = time.perf_counter()
                    tokens = model.encode(audios, audio_lengths)[0][0]
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0
                    logger.info(
                        f"[decoder-subprocess] VQ encode: {elapsed:.3f}s "
                        f"({tokens.shape} tokens)"
                    )
                    conn.send({"tokens": tokens.cpu().numpy()})

                else:
                    codes = torch.from_numpy(msg["codes"]).to(device=device)
                    pad_mult = msg.get("pad_multiple", DECODE_PAD_TO)
                    result, elapsed, seq_len, pad_to = _pad_decode_truncate(model, codes, device, pad_multiple=pad_mult)
                    audio_np = result.float().cpu().numpy()
                    logger.info(
                        f"[decoder-subprocess] VQ decode: {elapsed:.3f}s "
                        f"(padded {seq_len}->{pad_to} tokens)"
                    )
                    conn.send({"audio": audio_np})

            except Exception as e:
                logger.error(f"[decoder-subprocess] {action} error: {traceback.format_exc()}")
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

        if not parent_conn.poll(timeout=60):
            proc.kill()
            raise RuntimeError("Decoder subprocess timed out during init (60s)")

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

    def shutdown_decoder_subprocess(self) -> None:
        """Gracefully shut down the decoder subprocess."""
        conn = self._decoder_conn
        proc = self._decoder_process
        self._decoder_conn = None
        self._decoder_process = None

        if conn is not None:
            try:
                conn.send(None)
            except (OSError, BrokenPipeError, EOFError):
                logger.debug("Could not send shutdown signal to decoder subprocess")
            try:
                conn.close()
            except OSError:
                logger.debug("Could not close decoder subprocess connection")
        if proc is not None:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
                logger.warning(f"Force-killed decoder subprocess (pid={proc.pid})")

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

    def _decode_via_subprocess(self, codes: torch.Tensor) -> torch.Tensor:
        """Send codes to decoder subprocess, receive audio."""
        if not self._decoder_process.is_alive():
            logger.error("Decoder subprocess died, attempting in-process fallback")
            self.shutdown_decoder_subprocess()
            # Move parent DAC back to GPU for in-process decode
            target_device = codes.device
            try:
                if self.decoder_model.device != target_device:
                    logger.warning(f"Moving DAC model to {target_device} for fallback decode")
                    self.decoder_model.to(target_device)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Decoder subprocess died and fallback model move to "
                    f"{target_device} failed: {e}"
                ) from e
            return self.decode_vq_tokens(codes)

        t0 = time.perf_counter()
        # Use smaller pad for short sequences (streaming chunks)
        pad_mult = CHUNK_DECODE_PAD_TO if codes.shape[-1] <= CHUNK_DECODE_PAD_TO else DECODE_PAD_TO
        self._decoder_conn.send({"codes": codes.cpu().numpy(), "pad_multiple": pad_mult})

        if not self._decoder_conn.poll(timeout=30):
            raise RuntimeError("Decoder subprocess timed out (30s)")

        response = self._decoder_conn.recv()
        if "error" in response:
            raise RuntimeError(f"Decoder subprocess error: {response['error']}")

        logger.info(f"VQ decode (subprocess): {time.perf_counter() - t0:.3f}s")
        return torch.from_numpy(response["audio"])

    def _encode_via_subprocess(
        self, audio_np: np.ndarray, audio_length: int
    ) -> Optional[torch.Tensor]:
        """Send audio to decoder subprocess for encoding, receive tokens."""
        if not self._decoder_process.is_alive():
            logger.error("Decoder subprocess died, falling back to in-process encode")
            self.shutdown_decoder_subprocess()
            # Move parent DAC back to GPU for in-process encode fallback
            try:
                if torch.cuda.is_available() and self.decoder_model.device.type == "cpu":
                    logger.warning("Moving DAC model to cuda for fallback encode")
                    self.decoder_model.to("cuda")
            except RuntimeError as e:
                logger.error(f"Failed to move DAC model to GPU for encode fallback: {e}")
            return None  # caller will fall through to in-process path

        t0 = time.perf_counter()
        self._decoder_conn.send({
            "action": "encode",
            "audio": audio_np,
            "audio_length": audio_length,
        })

        if not self._decoder_conn.poll(timeout=120):
            raise RuntimeError("Decoder subprocess timed out during encode")

        response = self._decoder_conn.recv()
        if "error" in response:
            raise RuntimeError(f"Decoder subprocess encode error: {response['error']}")

        logger.info(f"VQ encode (subprocess): {time.perf_counter() - t0:.3f}s")
        return torch.from_numpy(response["tokens"])

    def encode_reference(self, reference_audio, enable_reference_audio):
        if enable_reference_audio and reference_audio is not None:
            if hasattr(self.decoder_model, "spec_transform"):
                sample_rate = self.decoder_model.spec_transform.sample_rate
            else:
                sample_rate = self.decoder_model.sample_rate
            reference_audio_content = self.load_audio(reference_audio, sample_rate)

            logger.info(
                f"Loaded audio with {len(reference_audio_content) / sample_rate:.2f} seconds"
            )

            # Route through subprocess when active — the subprocess has a
            # clean HIP context with miopen-conv-fix, avoiding page faults
            # from the parent process's corrupted HIP page tables on RDNA 4.
            if self._decoder_conn is not None:
                prompt_tokens = self._encode_via_subprocess(
                    reference_audio_content, len(reference_audio_content),
                )
                if prompt_tokens is not None:
                    logger.info(f"Encoded prompt: {prompt_tokens.shape}")
                    return prompt_tokens
                # Fall through to in-process if subprocess died

            audios = torch.from_numpy(reference_audio_content).to(
                device=self.decoder_model.device,
                dtype=next(self.decoder_model.parameters()).dtype,
            )[None, None, :]
            audio_lengths = torch.tensor(
                [audios.shape[2]], device=self.decoder_model.device, dtype=torch.long
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
