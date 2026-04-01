import base64
import io
from functools import partial
from typing import Callable, Optional

import gradio as gr
import httpx
import numpy as np
import soundfile as sf
from loguru import logger


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
) -> Optional[tuple[int, np.ndarray]]:
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
        "reference_id": reference_id if reference_id and reference_id != "None" else None,
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
            raise gr.Error(resp.text or f"API error {resp.status_code}")

        audio_data, sample_rate = sf.read(io.BytesIO(resp.content))
        return (sample_rate, audio_data.astype(np.float32))

    except gr.Error:
        raise
    except httpx.ConnectError:
        raise gr.Error("Cannot connect to API server. Is the server service running?")
    except Exception as e:
        logger.error(f"TTS request failed: {e}")
        raise gr.Error(str(e))


def list_references(api_url: str) -> list[str]:
    """Fetch available reference voice IDs from the API server."""
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(
                f"{api_url}/v1/references/list",
                headers={"Accept": "application/json"},
            )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("reference_ids", [])
    except Exception as e:
        logger.warning(f"Failed to fetch reference list: {e}")
    return []


def save_reference(
    name: str, audio_path: str, text: str, api_url: str
) -> None:
    """Save a new reference voice via the API server."""
    if not name or not name.strip():
        raise gr.Error("Please enter a name for the voice")
    if not audio_path:
        raise gr.Error("Please upload an audio file")
    if not text or not text.strip():
        raise gr.Error("Please enter the reference text (transcription of the audio)")

    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{api_url}/v1/references/add",
                data={"id": name.strip(), "text": text.strip()},
                files={"audio": ("reference.wav", audio_bytes, "audio/wav")},
            )

        if resp.status_code == 409:
            raise gr.Error(f"Voice '{name.strip()}' already exists")
        if resp.status_code != 200:
            raise gr.Error(resp.text or f"API error {resp.status_code}")

        gr.Info(f"Voice '{name.strip()}' saved successfully")

    except gr.Error:
        raise
    except httpx.ConnectError:
        raise gr.Error("Cannot connect to API server")
    except Exception as e:
        logger.error(f"Failed to save reference: {e}")
        raise gr.Error(str(e))


def get_inference_wrapper(api_url: str) -> Callable:
    """Return inference function with the API URL baked in."""
    return partial(inference_wrapper, api_url=api_url)
