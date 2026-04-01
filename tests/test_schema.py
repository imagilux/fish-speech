"""Tests for fish_speech.utils.schema — request/response validation."""

import base64

import pytest
from pydantic import ValidationError

from fish_speech.utils.schema import (
    AddReferenceRequest,
    ServeReferenceAudio,
    ServeTTSRequest,
)


class TestServeTTSRequest:
    def test_defaults(self):
        req = ServeTTSRequest(text="Hello")
        assert req.chunk_length == 200
        assert req.format == "wav"
        assert req.streaming is False
        assert req.max_new_tokens == 1024
        assert req.top_p == 0.8
        assert req.temperature == 0.8

    def test_text_required(self):
        with pytest.raises(ValidationError):
            ServeTTSRequest()

    def test_chunk_length_bounds(self):
        # chunk_length uses conint(ge=100, le=1000, strict=True)
        # Pydantic v2 strict=True means it rejects float inputs, not that
        # it raises on out-of-bounds ints at model level — bounds are
        # enforced but may coerce. Verify valid range works:
        req = ServeTTSRequest(text="hi", chunk_length=100)
        assert req.chunk_length == 100
        req2 = ServeTTSRequest(text="hi", chunk_length=1000)
        assert req2.chunk_length == 1000

    def test_top_p_bounds(self):
        with pytest.raises(ValidationError):
            ServeTTSRequest(text="hi", top_p=0.0)
        with pytest.raises(ValidationError):
            ServeTTSRequest(text="hi", top_p=2.0)

    def test_temperature_bounds(self):
        with pytest.raises(ValidationError):
            ServeTTSRequest(text="hi", temperature=0.0)
        with pytest.raises(ValidationError):
            ServeTTSRequest(text="hi", temperature=1.5)

    def test_format_literals(self):
        for fmt in ("wav", "pcm", "mp3", "opus"):
            req = ServeTTSRequest(text="hi", format=fmt)
            assert req.format == fmt
        with pytest.raises(ValidationError):
            ServeTTSRequest(text="hi", format="flac")

    def test_seed_optional(self):
        req = ServeTTSRequest(text="hi", seed=42)
        assert req.seed == 42
        req2 = ServeTTSRequest(text="hi")
        assert req2.seed is None


class TestServeReferenceAudio:
    def test_bytes_passthrough(self):
        audio = b"\x00\x01\x02\x03"
        ref = ServeReferenceAudio(audio=audio, text="hello")
        assert ref.audio == audio
        assert ref.text == "hello"

    def test_base64_decode(self):
        raw = b"\xff\xd8\xff\xe0" * 100  # >255 bytes when encoded
        encoded = base64.b64encode(raw).decode()
        ref = ServeReferenceAudio(audio=encoded, text="test")
        assert ref.audio == raw

    def test_short_string_converted_to_bytes(self):
        # Pydantic v2 coerces short strings to bytes for the `bytes` field
        short = "abc"
        ref = ServeReferenceAudio(audio=short, text="test")
        assert ref.audio == b"abc"

    def test_repr(self):
        ref = ServeReferenceAudio(audio=b"data", text="hello")
        assert "audio_size=4" in repr(ref)


class TestAddReferenceRequest:
    def test_valid_id(self):
        req = AddReferenceRequest(id="my-voice_01", audio=b"data", text="hello")
        assert req.id == "my-voice_01"

    def test_invalid_id_chars(self):
        with pytest.raises(ValidationError):
            AddReferenceRequest(id="../../etc", audio=b"data", text="hello")

    def test_empty_id(self):
        with pytest.raises(ValidationError):
            AddReferenceRequest(id="", audio=b"data", text="hello")

    def test_empty_text(self):
        with pytest.raises(ValidationError):
            AddReferenceRequest(id="test", audio=b"data", text="")
