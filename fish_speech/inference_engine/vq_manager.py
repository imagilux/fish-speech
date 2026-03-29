import time
from typing import Callable

import torch
from loguru import logger

from fish_speech.models.dac.modded_dac import DAC


class VQManager:

    def __init__(self):
        # Make Pylance happy (attribut/method not defined...)
        self.decoder_model: DAC
        self.load_audio: Callable

    # Fixed decode length — pad all inputs to this size so MIOpen only
    # ever sees one set of conv shapes (cached after first warmup).
    # 512 tokens ≈ 23s of audio at 21.5 Hz, covers most TTS outputs.
    DECODE_PAD_TO = 512

    def decode_vq_tokens(self, codes):
        logger.info(f"VQ features: {codes.shape}")

        if isinstance(self.decoder_model, DAC):
            seq_len = codes.shape[-1]
            pad_to = max(seq_len, self.DECODE_PAD_TO)
            # Round up to next multiple of DECODE_PAD_TO for consistent shapes
            pad_to = ((pad_to + self.DECODE_PAD_TO - 1) // self.DECODE_PAD_TO) * self.DECODE_PAD_TO

            if seq_len < pad_to:
                # Pad with zeros (silence) on the right
                padded = torch.zeros(
                    (*codes.shape[:-1], pad_to),
                    dtype=codes.dtype, device=codes.device,
                )
                padded[..., :seq_len] = codes
            else:
                padded = codes

            # Free cached VRAM so MIOpen can allocate workspace for optimized
            # conv kernels. Without this, PyTorch's allocator holds fragmented
            # blocks and MIOpen gets workspace=0, falling back to naive convolution.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = self.decoder_model.from_indices(padded[None])[0].squeeze()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Truncate to original length (proportional to token count)
            output_len = int(result.shape[-1] * seq_len / pad_to)
            result = result[..., :output_len]

            logger.info(
                f"VQ decode: {time.perf_counter() - t0:.3f}s "
                f"(padded {seq_len}→{pad_to} tokens)"
            )
            return result

        raise ValueError(f"Unknown model type: {type(self.decoder_model)}")

    def encode_reference(self, reference_audio, enable_reference_audio):
        if enable_reference_audio and reference_audio is not None:
            # Load audios, and prepare basic info here
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

            # VQ Encoder
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
