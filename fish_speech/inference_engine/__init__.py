import queue
from typing import Generator

import numpy as np
import torch
from loguru import logger

from fish_speech.inference_engine.reference_loader import ReferenceLoader
from fish_speech.inference_engine.utils import InferenceResult, wav_chunk_header
from fish_speech.inference_engine.vq_manager import VQManager
from fish_speech.models.dac.modded_dac import DAC
from fish_speech.models.text2semantic.inference import (
    GenerateRequest,
    GenerateResponse,
    WrappedGenerateResponse,
)
from fish_speech.utils import autocast_exclude_mps, set_seed
from fish_speech.utils.schema import ServeTTSRequest


STREAM_CHUNK_SIZE = 25    # New tokens per chunk (~1.16s audio at 21.5 Hz)
STREAM_LEFT_CONTEXT = 25  # Context tokens for windowed transformer quality


class TTSInferenceEngine(ReferenceLoader, VQManager):

    def __init__(
        self,
        llama_queue: queue.Queue,
        decoder_model: DAC,
        precision: torch.dtype,
        compile: bool,
    ) -> None:

        super().__init__()

        self.llama_queue = llama_queue
        self.decoder_model = decoder_model
        self.precision = precision
        self.compile = compile
        self._frame_length = getattr(decoder_model, "frame_length", 2048)

    @torch.inference_mode()
    def inference(self, req: ServeTTSRequest) -> Generator[InferenceResult, None, None]:
        """
        Main inference function:
        - Loads the reference audio and text.
        - Calls the LLAMA model for inference.
        - Decodes the VQ tokens to audio.
        """

        ref_id: str | None = req.reference_id
        prompt_tokens, prompt_texts = [], []
        # Load the reference audio and text based on id or hash
        if ref_id is not None:
            prompt_tokens, prompt_texts = self.load_by_id(ref_id, req.use_memory_cache)

        elif req.references:
            prompt_tokens, prompt_texts = self.load_by_hash(
                req.references, req.use_memory_cache
            )

        # Set the random seed if provided
        if req.seed is not None:
            set_seed(req.seed)
            logger.warning(f"set seed: {req.seed}")

        # Lazy reload: only move LLM to GPU when actually needed
        if hasattr(self.llama_queue, "offloaded") and self.llama_queue.offloaded:
            self.llama_queue.reload_to_gpu()

        # Get the symbolic tokens from the LLAMA model
        response_queue = self.send_Llama_request(req, prompt_tokens, prompt_texts)

        # Get the sample rate from the decoder model
        if hasattr(self.decoder_model, "spec_transform"):
            sample_rate = self.decoder_model.spec_transform.sample_rate
        else:
            sample_rate = self.decoder_model.sample_rate

        # If streaming, send the header
        if req.streaming:
            yield InferenceResult(
                code="header",
                audio=(
                    sample_rate,
                    np.array(wav_chunk_header(sample_rate=sample_rate)),
                ),
                error=None,
            )

        # Process LLM results — either streaming (decode chunks as they
        # arrive) or batch (collect all, then decode).
        segments = []
        tokens_decoded = 0  # Track how many tokens we've yielded audio for
        is_streaming_chunks = req.streaming and self.subprocess_active

        while True:
            try:
                wrapped_result: WrappedGenerateResponse = response_queue.get(timeout=300)
            except queue.Empty:
                yield InferenceResult(
                    code="error",
                    audio=None,
                    error=RuntimeError("LLM inference timed out (300s)"),
                )
                break
            if wrapped_result.status == "error":
                yield InferenceResult(
                    code="error",
                    audio=None,
                    error=(
                        wrapped_result.response
                        if isinstance(wrapped_result.response, Exception)
                        else Exception("Unknown error")
                    ),
                )
                break

            if not isinstance(wrapped_result.response, GenerateResponse):
                raise TypeError(
                    f"Expected GenerateResponse, got {type(wrapped_result.response).__name__}"
                )

            result: GenerateResponse = wrapped_result.response

            if result.action == "chunk" and is_streaming_chunks:
                # Decode this chunk while LLM continues generating
                segment = self._decode_streaming_chunk(
                    result.codes, tokens_decoded, sample_rate,
                )
                if segment is not None and len(segment) > 0:
                    tokens_decoded = result.codes.shape[-1]
                    segments.append(segment)
                    yield InferenceResult(
                        code="segment",
                        audio=(sample_rate, segment),
                        error=None,
                    )

            elif result.action == "sample":
                # Final result — decode any remaining tokens after last chunk
                if is_streaming_chunks and tokens_decoded > 0:
                    tail = self._decode_streaming_chunk(
                        result.codes, tokens_decoded, sample_rate,
                    )
                    if tail is not None and len(tail) > 0:
                        segments.append(tail)
                        yield InferenceResult(
                            code="segment",
                            audio=(sample_rate, tail),
                            error=None,
                        )
                else:
                    # Batch mode: offload LLM, decode full sequence
                    if self.subprocess_active and hasattr(self.llama_queue, "offload_to_cpu"):
                        self.llama_queue.offload_to_cpu()
                    segment = self.get_audio_segment(result)
                    segments.append(segment)
                    if req.streaming:
                        yield InferenceResult(
                            code="segment",
                            audio=(sample_rate, segment),
                            error=None,
                        )

            elif result.action == "next":
                break

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Edge case: no audio generated
        if len(segments) == 0:
            yield InferenceResult(
                code="error",
                audio=None,
                error=RuntimeError("No audio generated, please check the input text."),
            )
        else:
            # Streaming or not, return the final audio
            audio = np.concatenate(segments, axis=0)
            yield InferenceResult(
                code="final",
                audio=(sample_rate, audio),
                error=None,
            )

    def _decode_streaming_chunk(self, all_codes, tokens_decoded, sample_rate):
        """Decode a streaming chunk with left context, return only new audio."""
        total_tokens = all_codes.shape[-1]
        new_start = tokens_decoded
        new_end = total_tokens

        if new_end <= new_start:
            return None

        # Include left context for windowed transformer quality
        context_start = max(0, new_start - STREAM_LEFT_CONTEXT)
        codes_with_context = all_codes[:, context_start:new_end]

        # Decode the chunk (context + new tokens)
        audio = self.decode_vq_tokens(codes_with_context)
        if not isinstance(audio, np.ndarray):
            audio = audio.float().cpu().numpy()

        # Trim context audio — only keep the new tokens' audio
        context_tokens = new_start - context_start
        trim_samples = context_tokens * self._frame_length
        if trim_samples > 0 and trim_samples <= len(audio):
            audio = audio[trim_samples:]

        return audio

    def send_Llama_request(
        self, req: ServeTTSRequest, prompt_tokens: list, prompt_texts: list
    ) -> queue.Queue:
        """
        Send a request to the LLAMA model to generate the symbolic tokens.
        """

        # Prepare the request
        request = dict(
            device=self.llama_queue.device,
            max_new_tokens=req.max_new_tokens,
            text=req.text,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            temperature=req.temperature,
            compile=self.compile,
            iterative_prompt=req.chunk_length > 0,
            chunk_length=req.chunk_length,
            prompt_tokens=prompt_tokens,
            prompt_text=prompt_texts,
            stream_chunk_size=STREAM_CHUNK_SIZE if req.streaming and self.subprocess_active else 0,
        )

        # Create a queue to get the response
        response_queue = queue.Queue()

        # Send the request to the LLAMA model
        self.llama_queue.put(
            GenerateRequest(
                request=request,
                response_queue=response_queue,
            )
        )

        return response_queue

    def get_audio_segment(self, result: GenerateResponse) -> np.ndarray:
        """
        Decode the VQ tokens to audio.
        """

        # Don't use autocast on MPS devices
        with autocast_exclude_mps(
            device_type=self.decoder_model.device.type, dtype=self.precision
        ):
            # Decode the symbolic tokens to audio
            segment = self.decode_vq_tokens(codes=result.codes)

        # Convert the audio to numpy
        return segment.float().cpu().numpy()
