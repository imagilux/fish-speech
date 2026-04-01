"""Tests for tools.webui.inference — HTTP client for API server."""

import io
import struct
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from tools.webui.inference import get_inference_wrapper


def _make_wav_bytes(duration: float = 0.1, sr: int = 44100) -> bytes:
    """Generate minimal valid WAV bytes."""
    samples = np.sin(np.linspace(0, 440 * 2 * np.pi * duration, int(sr * duration)))
    buf = io.BytesIO()
    sf.write(buf, samples.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


class TestInferenceWrapper:
    def test_successful_tts(self):
        wav = _make_wav_bytes()
        inference_fn = get_inference_wrapper("http://fake:8080")

        with patch("tools.webui.inference.httpx.Client") as mock_client_cls:
            mock_resp = mock_client_cls.return_value.__enter__.return_value.post.return_value
            mock_resp.status_code = 200
            mock_resp.content = wav

            audio = inference_fn(
                text="Hello",
                reference_id="",
                reference_audio=None,
                reference_text="",
                max_new_tokens=1024,
                chunk_length=200,
                top_p=0.8,
                repetition_penalty=1.1,
                temperature=0.8,
                seed=0,
                use_memory_cache="off",
            )

        assert audio is not None
        sample_rate, data = audio
        assert sample_rate == 44100
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.float32

    def test_connection_error_raises_gr_error(self):
        import gradio as gr
        import httpx

        inference_fn = get_inference_wrapper("http://fake:8080")

        with patch("tools.webui.inference.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.post.side_effect = (
                httpx.ConnectError("Connection refused")
            )

            with pytest.raises(gr.Error):
                inference_fn(
                    text="Hello", reference_id="", reference_audio=None,
                    reference_text="", max_new_tokens=1024, chunk_length=200,
                    top_p=0.8, repetition_penalty=1.1, temperature=0.8,
                    seed=0, use_memory_cache="off",
                )


class TestErrorHandling:
    def test_api_error_raises_gr_error(self):
        import gradio as gr

        inference_fn = get_inference_wrapper("http://fake:8080")

        with patch("tools.webui.inference.httpx.Client") as mock_client_cls:
            mock_resp = mock_client_cls.return_value.__enter__.return_value.post.return_value
            mock_resp.status_code = 500
            mock_resp.text = "Internal Server Error"

            with pytest.raises(gr.Error):
                inference_fn(
                    text="Hello", reference_id="", reference_audio=None,
                    reference_text="", max_new_tokens=1024, chunk_length=200,
                    top_p=0.8, repetition_penalty=1.1, temperature=0.8,
                    seed=0, use_memory_cache="off",
                )
