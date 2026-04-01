"""Tests for fish_speech.inference_engine.reference_loader — path validation."""

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from fish_speech.inference_engine.reference_loader import ReferenceLoader


def _make_wav_file(path: str, duration: float = 0.1, sr: int = 16000) -> None:
    """Write a minimal WAV file."""
    samples = np.zeros(int(sr * duration), dtype=np.float32)
    sf.write(path, samples, sr)


class DummyLoader(ReferenceLoader):
    """Minimal subclass that satisfies the MRO (VQManager.__init__ needs to run)."""
    pass


@pytest.fixture
def loader():
    ldr = DummyLoader()
    # Stub out attributes that ReferenceLoader.__init__ tries to check
    ldr.decoder_model = MagicMock()
    ldr.decoder_model.sample_rate = 16000
    return ldr


class TestLoadAudio:
    def test_bytes_input(self, loader: DummyLoader):
        buf = io.BytesIO()
        sf.write(buf, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
        wav_bytes = buf.getvalue()

        result = loader.load_audio(wav_bytes, sr=16000)
        assert isinstance(result, np.ndarray)
        assert len(result) == 1600

    def test_file_path_input(self, loader: DummyLoader, tmp_path: Path):
        wav = tmp_path / "test.wav"
        _make_wav_file(str(wav))

        result = loader.load_audio(str(wav), sr=16000)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_nonexistent_path_raises(self, loader: DummyLoader):
        with pytest.raises(FileNotFoundError, match="not found"):
            loader.load_audio("/nonexistent/audio.wav", sr=16000)

    def test_resamples_if_needed(self, loader: DummyLoader, tmp_path: Path):
        wav = tmp_path / "test_48k.wav"
        sf.write(str(wav), np.zeros(4800, dtype=np.float32), 48000)

        result = loader.load_audio(str(wav), sr=16000)
        # 4800 samples at 48kHz = 0.1s → 1600 samples at 16kHz
        assert len(result) == 1600


class TestListReferenceIds:
    def test_empty_when_no_references_dir(self, loader: DummyLoader, monkeypatch):
        monkeypatch.chdir(tempfile.mkdtemp())
        assert loader.list_reference_ids() == []

    def test_finds_valid_references(self, loader: DummyLoader, monkeypatch):
        base = Path(tempfile.mkdtemp())
        monkeypatch.chdir(base)

        ref_dir = base / "references" / "voice1"
        ref_dir.mkdir(parents=True)
        _make_wav_file(str(ref_dir / "sample.wav"))
        (ref_dir / "sample.lab").write_text("hello world")

        ids = loader.list_reference_ids()
        assert ids == ["voice1"]

    def test_skips_dir_without_lab(self, loader: DummyLoader, monkeypatch):
        base = Path(tempfile.mkdtemp())
        monkeypatch.chdir(base)

        ref_dir = base / "references" / "no-lab"
        ref_dir.mkdir(parents=True)
        _make_wav_file(str(ref_dir / "sample.wav"))
        # No .lab file

        ids = loader.list_reference_ids()
        assert ids == []


class TestValidateId:
    def test_valid_ids(self):
        for id_ in ("voice1", "my-voice", "test_01", "John Doe"):
            ReferenceLoader._validate_id(id_)

    def test_invalid_ids(self):
        for id_ in ("../etc", "voice;rm -rf", "a" * 256):
            with pytest.raises(ValueError):
                ReferenceLoader._validate_id(id_)
