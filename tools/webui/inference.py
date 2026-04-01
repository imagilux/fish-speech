import base64
import html
import io
from functools import partial
from typing import Any, Callable

import httpx
import numpy as np
import soundfile as sf
from loguru import logger

from fish_speech.i18n import i18n


def inference_wrapper(
    text,
    reference_id,
    reference_audio,
    reference_text,
    max_new_tokens,
    chunk_length,
    top_p,
    repetition_penalty,
    temperature,
    seed,
    use_memory_cache,
    api_url,
):
    """Call the Fish Speech API server for TTS inference."""

    references = []
    if reference_audio:
        with open(reference_audio, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        references.append({
            "audio": audio_b64,
            "text": reference_text or "",
        })

    payload = {
        "text": text,
        "reference_id": reference_id if reference_id else None,
        "references": references,
        "max_new_tokens": int(max_new_tokens),
        "chunk_length": int(chunk_length),
        "top_p": float(top_p),
        "repetition_penalty": float(repetition_penalty),
        "temperature": float(temperature),
        "seed": int(seed) if seed else None,
        "use_memory_cache": use_memory_cache,
        "format": "wav",
        "streaming": False,
    }

    try:
        with httpx.Client(timeout=300) as client:
            resp = client.post(f"{api_url}/v1/tts", json=payload)

        if resp.status_code != 200:
            error_msg = resp.text or f"API error {resp.status_code}"
            return None, build_html_error_message(error_msg)

        audio_data, sample_rate = sf.read(io.BytesIO(resp.content))
        return (sample_rate, audio_data.astype(np.float32)), None

    except httpx.ConnectError:
        return None, build_html_error_message(
            "Cannot connect to API server. Is the server service running?"
        )
    except Exception as e:
        logger.error(f"TTS request failed: {e}")
        return None, build_html_error_message(str(e))


def build_html_error_message(error: Any) -> str:
    error_str = str(error) if error is not None else "Unknown error"
    return f"""
    <div style="color: red;
    font-weight: bold;">
        {html.escape(error_str)}
    </div>
    """


def list_references(api_url: str) -> list[str]:
    """Fetch available reference voice IDs from the API server."""
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{api_url}/v1/references/list")
        if resp.status_code == 200:
            data = resp.json()
            return data.get("reference_ids", [])
    except Exception as e:
        logger.warning(f"Failed to fetch reference list: {e}")
    return []


def get_inference_wrapper(api_url: str) -> Callable:
    """Return inference function with the API URL baked in."""
    return partial(inference_wrapper, api_url=api_url)
