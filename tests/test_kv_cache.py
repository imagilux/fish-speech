"""Tests for KV cache sizing and prefix attention in the text2semantic model."""

import pytest
import torch
from loguru import logger

from fish_speech.models.text2semantic import inference
from fish_speech.models.text2semantic.inference import generate, kv_cache_limit
from fish_speech.models.text2semantic.llama import DualARModelArgs, DualARTransformer

NUM_CODEBOOKS = 3
CODEBOOK_SIZE = 16
SEMANTIC_BEGIN = 8
IM_END = 7


class _Tokenizer:
    def get_token_id(self, token):
        return IM_END


def _tiny_model(max_seq_len: int = 64) -> DualARTransformer:
    torch.manual_seed(0)
    config = DualARModelArgs(
        vocab_size=48,
        n_layer=2,
        n_head=4,
        n_local_heads=2,
        dim=32,
        head_dim=8,
        intermediate_size=64,
        max_seq_len=max_seq_len,
        codebook_size=CODEBOOK_SIZE,
        num_codebooks=NUM_CODEBOOKS,
        semantic_begin_id=SEMANTIC_BEGIN,
        semantic_end_id=SEMANTIC_BEGIN + CODEBOOK_SIZE - 1,
        n_fast_layer=1,
    )
    model = DualARTransformer(config).eval()
    model.tokenizer = _Tokenizer()
    return model


def _prompt(length: int) -> torch.Tensor:
    """A text-only prompt: (1 + codebooks, length)."""
    prompt = torch.zeros((1 + NUM_CODEBOOKS, length), dtype=torch.int)
    prompt[0] = 1
    return prompt


def _never_end(logits, temperature, top_p, top_k):
    """Stands in for `sample`: always the same audio token, so generation
    only stops when it runs out of room."""
    is_main_token = logits.shape[-1] > CODEBOOK_SIZE
    token = SEMANTIC_BEGIN + 1 if is_main_token else 1
    return torch.tensor([token], dtype=torch.int), None


def _generate(model, prompt_len: int, max_new_tokens: int) -> torch.Tensor:
    return generate(
        model=model,
        prompt=_prompt(prompt_len),
        max_new_tokens=max_new_tokens,
        audio_masks=None,
        audio_parts=None,
    )


@pytest.fixture(autouse=True)
def _plain_cache(monkeypatch):
    monkeypatch.delenv("MAX_SEQ_LEN", raising=False)
    monkeypatch.delenv("KV_CACHE_BITS", raising=False)
    monkeypatch.setattr(inference, "sample", _never_end)


@pytest.fixture
def warnings():
    """Warning messages logged while the test runs."""
    messages = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    yield messages
    logger.remove(sink)


class TestKvCacheLimit:
    def test_defaults_to_model_limit(self):
        assert kv_cache_limit(_tiny_model(64)) == 64

    def test_env_caps_the_cache(self, monkeypatch):
        monkeypatch.setenv("MAX_SEQ_LEN", "32")
        assert kv_cache_limit(_tiny_model(64)) == 32

    def test_env_cannot_exceed_model_limit(self, monkeypatch):
        monkeypatch.setenv("MAX_SEQ_LEN", "4096")
        assert kv_cache_limit(_tiny_model(64)) == 64

    @pytest.mark.parametrize("value", ["", "0"])
    def test_empty_or_zero_means_model_limit(self, monkeypatch, value):
        monkeypatch.setenv("MAX_SEQ_LEN", value)
        assert kv_cache_limit(_tiny_model(64)) == 64


class TestGenerateStaysInsideCache:
    def test_request_larger_than_cache_is_shortened(self, monkeypatch, warnings):
        # 10 prompt positions + 100 new tokens do not fit 32 slots. This used
        # to write past the cache: an IndexError on CPU, an abort on ROCm.
        monkeypatch.setenv("MAX_SEQ_LEN", "32")
        model = _tiny_model(64)

        seq = _generate(model, prompt_len=10, max_new_tokens=100)

        assert model.max_seq_len == 32
        assert seq.shape == (1 + NUM_CODEBOOKS, 32)
        assert any("MAX_SEQ_LEN" in message for message in warnings)

    def test_unbounded_request_fills_the_cache(self, monkeypatch, warnings):
        monkeypatch.setenv("MAX_SEQ_LEN", "24")

        seq = _generate(_tiny_model(64), prompt_len=8, max_new_tokens=0)

        assert seq.shape[1] == 24
        assert any("MAX_SEQ_LEN" in message for message in warnings)

    def test_request_that_fits_is_not_reported(self, monkeypatch, warnings):
        monkeypatch.setenv("MAX_SEQ_LEN", "32")

        seq = _generate(_tiny_model(64), prompt_len=10, max_new_tokens=5)

        assert seq.shape[1] == 15
        assert warnings == []

    def test_ending_on_the_last_slot_is_not_reported(self, monkeypatch, warnings):
        # The model closes the utterance exactly where the cache ends.
        monkeypatch.setenv("MAX_SEQ_LEN", "12")

        def ends_now(logits, temperature, top_p, top_k):
            return torch.tensor([IM_END], dtype=torch.int), None

        monkeypatch.setattr(inference, "sample", ends_now)

        seq = _generate(_tiny_model(64), prompt_len=11, max_new_tokens=0)

        assert seq.shape[1] == 12
        assert warnings == []

    def test_prompt_larger_than_cache_is_rejected(self, monkeypatch):
        monkeypatch.setenv("MAX_SEQ_LEN", "32")

        with pytest.raises(ValueError, match="MAX_SEQ_LEN"):
            _generate(_tiny_model(64), prompt_len=40, max_new_tokens=10)

    def test_single_free_slot_gives_one_token(self, monkeypatch):
        monkeypatch.setenv("MAX_SEQ_LEN", "16")

        seq = _generate(_tiny_model(64), prompt_len=15, max_new_tokens=10)

        assert seq.shape[1] == 16

    def test_cache_grows_for_a_longer_request(self):
        model = _tiny_model(4096)

        _generate(model, prompt_len=4, max_new_tokens=4)
        assert model.max_seq_len == 1024

        seq = _generate(model, prompt_len=1030, max_new_tokens=4)
        assert model.max_seq_len == 2048
        assert seq.shape[1] == 1034

    def test_cache_is_kept_when_large_enough(self):
        model = _tiny_model(4096)
        _generate(model, prompt_len=4, max_new_tokens=4)
        cache = model.layers[0].attention.kv_cache

        _generate(model, prompt_len=8, max_new_tokens=8)

        assert model.layers[0].attention.kv_cache is cache


class TestPrefixAttention:
    """Attending to the filled prefix must give what masking the whole cache gave."""

    @staticmethod
    def _model_with_dirty_cache() -> DualARTransformer:
        model = _tiny_model(64)
        model.setup_caches(max_batch_size=1, max_seq_len=32, dtype=torch.float32)
        # Leftovers from an earlier, longer request must not leak in.
        for layer in model.layers:
            layer.attention.kv_cache.k_cache.normal_()
            layer.attention.kv_cache.v_cache.normal_()
        return model

    @torch.no_grad()
    def test_prefill_and_decode_match_whole_cache(self):
        prompt = _prompt(6)[None]
        prompt_pos = torch.arange(6)
        step = torch.tensor([[[SEMANTIC_BEGIN + 1], [1], [1], [1]]], dtype=torch.int)
        step_pos = torch.tensor([6])

        model = self._model_with_dirty_cache()
        whole_prefill = model.forward_generate(prompt, prompt_pos)
        whole_step = model.forward_generate(step, step_pos)

        model = self._model_with_dirty_cache()
        prefix_prefill = model.forward_generate(prompt, prompt_pos, kv_len=6)
        prefix_step = model.forward_generate(step, step_pos, kv_len=7)

        for whole, prefix in (
            (whole_prefill, prefix_prefill),
            (whole_step, prefix_step),
        ):
            assert torch.allclose(whole.logits, prefix.logits, atol=1e-5)
            assert torch.allclose(whole.hidden_states, prefix.hidden_states, atol=1e-5)

    @pytest.mark.parametrize("kv_len", [0, 33])
    @torch.no_grad()
    def test_kv_len_outside_cache_is_rejected(self, kv_len):
        model = self._model_with_dirty_cache()

        with pytest.raises(ValueError, match="kv_len"):
            model.forward_generate(_prompt(6)[None], torch.arange(6), kv_len=kv_len)
