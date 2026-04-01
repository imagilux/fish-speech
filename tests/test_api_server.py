"""Tests for tools.api_server — args validation and app creation."""

import json
import os
from unittest.mock import patch

import pytest

from tools.api_server import ENV_ARGS_KEY, _ALLOWED_ARGS


class TestAllowedArgs:
    def test_known_args_present(self):
        expected = {
            "mode", "llama_checkpoint_path", "decoder_checkpoint_path",
            "decoder_config_name", "device", "half", "compile",
            "max_text_length", "listen", "workers", "api_key",
        }
        assert _ALLOWED_ARGS == expected

    def test_filters_unknown_keys(self):
        raw = {
            "mode": "tts",
            "device": "cuda",
            "evil_key": "/etc/shadow",
            "__class__": "hack",
        }
        filtered = {k: v for k, v in raw.items() if k in _ALLOWED_ARGS}
        assert "evil_key" not in filtered
        assert "__class__" not in filtered
        assert filtered["mode"] == "tts"
        assert filtered["device"] == "cuda"
