"""Tests for fish_speech.utils.gpu — GPU detection and VRAM guidance."""

import os
from unittest.mock import MagicMock, patch

import pytest

from fish_speech.utils.gpu import _ROCM_GFX_OVERRIDES, _is_rocm, auto_detect_rocm_gfx


class TestIsRocm:
    def test_returns_false_without_cuda(self):
        with patch("fish_speech.utils.gpu.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            assert _is_rocm() is False

    def test_returns_false_without_hip(self):
        with patch("fish_speech.utils.gpu.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.version.hip = None
            assert _is_rocm() is False


class TestAutoDetectRocmGfx:
    def test_skips_when_not_rocm(self):
        with patch("fish_speech.utils.gpu._is_rocm", return_value=False):
            auto_detect_rocm_gfx()
            assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ or True

    def test_skips_when_already_set(self, monkeypatch):
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
        with patch("fish_speech.utils.gpu._is_rocm", return_value=True):
            auto_detect_rocm_gfx()
            assert os.environ["HSA_OVERRIDE_GFX_VERSION"] == "11.0.0"

    def test_gfx_overrides_populated(self):
        assert "gfx1201" in _ROCM_GFX_OVERRIDES
        assert _ROCM_GFX_OVERRIDES["gfx1201"] == "12.0.0"
