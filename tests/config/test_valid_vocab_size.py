# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import ModelConfig, SpeculativeConfig, VllmConfig
from vllm.tokenizers import get_tokenizer_vocab_upper_bound


class _FakeTokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.vocab_size


class _SparseFakeTokenizer:
    def __init__(self, vocab_upper_bound: int):
        self.vocab_upper_bound = vocab_upper_bound

    def __len__(self) -> int:
        return 2

    def get_vocab(self) -> dict[str, int]:
        return {
            "zero": 0,
            "sparse-added-token": self.vocab_upper_bound - 1,
        }


class _UnsupportedGetVocabTokenizer(_FakeTokenizer):
    def get_vocab(self) -> dict[str, int]:
        raise NotImplementedError


class _InvalidGetVocabTokenizer(_FakeTokenizer):
    def get_vocab(self) -> dict[str, int]:
        raise ValueError("malformed tokenizer vocabulary")


class _MetadataTokenizer:
    """A tokenizer whose metadata sources intentionally disagree."""

    vocab_size = 7

    def __len__(self) -> int:
        # ``len`` is a count and does not account for sparse IDs.
        return 6

    def get_vocab(self) -> dict[str, int]:
        return {"zero": 0, "sparse-token": 3}


class _SpecialTokenOmittedTokenizer:
    """A tokenizer whose high-ID special token is absent from get_vocab()."""

    vocab_size = 2

    def __len__(self) -> int:
        return 2

    def get_vocab(self) -> dict[str, int]:
        return {"zero": 0, "one": 1}

    def get_added_vocab(self) -> dict[str, int]:
        return {"<special>": 42}

    @property
    def all_special_ids(self) -> list[int]:
        return [42]


class _LegacyMaxTokenIdTokenizer:
    """Legacy tokenizer exposing only the inclusive max_token_id property."""

    max_token_id = 9


def _make_model_config(
    *,
    valid_vocab_size: int | None = None,
    architecture: str = "Step4ForCausalLM",
    vocab_size: int = 128896,
    skip_tokenizer_init: bool = False,
) -> ModelConfig:
    model_config = object.__new__(ModelConfig)
    model_config._architecture = architecture
    model_config.model_arch_config = SimpleNamespace(vocab_size=vocab_size)
    model_config.valid_vocab_size = valid_vocab_size
    model_config.skip_tokenizer_init = skip_tokenizer_init
    return model_config


def test_valid_vocab_size_defaults_to_tokenizer_length():
    model_config = _make_model_config()

    model_config.set_and_verify_valid_vocab_size(_FakeTokenizer(128815))

    assert model_config.get_valid_vocab_size() == 128815


def test_valid_vocab_size_uses_token_id_upper_bound_for_sparse_vocab():
    model_config = _make_model_config(vocab_size=1024)
    tokenizer = _SparseFakeTokenizer(1001)

    assert len(tokenizer) == 2
    model_config.set_and_verify_valid_vocab_size(tokenizer)

    assert model_config.get_valid_vocab_size() == 1001


def test_tokenizer_vocab_upper_bound_merges_metadata_sources():
    tokenizer = _MetadataTokenizer()

    assert get_tokenizer_vocab_upper_bound(tokenizer) == 7


def test_tokenizer_vocab_upper_bound_includes_omitted_special_tokens():
    tokenizer = _SpecialTokenOmittedTokenizer()

    assert get_tokenizer_vocab_upper_bound(tokenizer) == 43


def test_tokenizer_vocab_upper_bound_supports_legacy_max_token_id():
    tokenizer = _LegacyMaxTokenIdTokenizer()

    assert get_tokenizer_vocab_upper_bound(tokenizer) == 10


def test_valid_vocab_size_falls_back_when_get_vocab_is_not_implemented():
    model_config = _make_model_config()

    model_config.set_and_verify_valid_vocab_size(_UnsupportedGetVocabTokenizer(128815))

    assert model_config.get_valid_vocab_size() == 128815


def test_valid_vocab_size_does_not_swallow_get_vocab_errors():
    model_config = _make_model_config()

    with pytest.raises(ValueError, match="malformed tokenizer vocabulary"):
        model_config.set_and_verify_valid_vocab_size(_InvalidGetVocabTokenizer(128815))


def test_valid_vocab_size_falls_back_without_tokenizer_initialization():
    model_config = _make_model_config()

    assert model_config.get_valid_vocab_size() == 128896


def test_skip_tokenizer_init_requires_explicit_valid_vocab_size():
    model_config = _make_model_config(skip_tokenizer_init=True)

    with pytest.raises(ValueError, match="requires an explicit valid_vocab_size"):
        model_config._verify_valid_vocab_size()


def test_skip_tokenizer_init_accepts_explicit_valid_vocab_size():
    model_config = _make_model_config(
        valid_vocab_size=128815,
        skip_tokenizer_init=True,
    )

    model_config._verify_valid_vocab_size()
    assert model_config.get_valid_vocab_size() == 128815


@pytest.mark.parametrize("valid_vocab_size", [0, -1, 128897])
def test_valid_vocab_size_must_fit_model_vocab(valid_vocab_size):
    model_config = _make_model_config(valid_vocab_size=valid_vocab_size)

    with pytest.raises(ValueError, match="valid_vocab_size"):
        model_config._verify_valid_vocab_size()


def test_valid_vocab_size_must_fit_tokenizer():
    model_config = _make_model_config(valid_vocab_size=128816)

    with pytest.raises(ValueError, match="token-ID upper bound"):
        model_config.set_and_verify_valid_vocab_size(_FakeTokenizer(128815))


def test_valid_vocab_size_rejects_unsupported_architecture():
    model_config = _make_model_config(
        valid_vocab_size=1000,
        architecture="Qwen2ForCausalLM",
    )

    with pytest.raises(ValueError, match="only for Step4"):
        model_config._verify_valid_vocab_size()


def test_resolve_valid_vocab_size_loads_tokenizer_before_engine_start(
    monkeypatch,
):
    model_config = _make_model_config()
    tokenizer = _FakeTokenizer(128815)
    monkeypatch.setattr(
        "vllm.tokenizers.cached_tokenizer_from_config",
        lambda _: tokenizer,
    )

    resolved = model_config.resolve_valid_vocab_size()

    assert resolved is tokenizer
    assert model_config.get_valid_vocab_size() == 128815


def test_resolve_explicit_valid_vocab_size_does_not_load_tokenizer(
    monkeypatch,
):
    model_config = _make_model_config(valid_vocab_size=128815)

    def fail_if_called(_):
        raise AssertionError("tokenizer must not be initialized")

    monkeypatch.setattr(
        "vllm.tokenizers.cached_tokenizer_from_config",
        fail_if_called,
    )

    assert model_config.resolve_valid_vocab_size() is None
    assert model_config.get_valid_vocab_size() == 128815


@pytest.mark.parametrize("method", ["mtp", "draft_model"])
def test_vllm_config_syncs_valid_vocab_size_to_same_tokenizer_draft(method):
    target_config = _make_model_config()
    draft_config = _make_model_config(architecture="Step4MTP")
    vllm_config = SimpleNamespace(
        model_config=target_config,
        speculative_config=SimpleNamespace(
            method=method,
            target_model_config=target_config,
            draft_model_config=draft_config,
            use_heterogeneous_vocab=False,
        ),
    )
    tokenizer = _FakeTokenizer(128815)

    VllmConfig.resolve_valid_vocab_size(vllm_config, tokenizer)

    assert target_config.get_valid_vocab_size() == 128815
    assert draft_config.get_valid_vocab_size() == 128815


def test_vllm_config_revalidates_non_step4_draft_after_target_resolution():
    target_config = _make_model_config()
    draft_config = _make_model_config(
        architecture="Qwen2ForCausalLM",
    )
    speculative_config = SimpleNamespace(
        method="draft_model",
        target_model_config=target_config,
        draft_model_config=draft_config,
        use_heterogeneous_vocab=False,
    )
    vllm_config = SimpleNamespace(
        model_config=target_config,
        speculative_config=speculative_config,
    )

    SpeculativeConfig.verify_equal_vocab_size_if_draft_model(speculative_config)
    with pytest.raises(ValueError, match="use_heterogeneous_vocab=True"):
        VllmConfig.resolve_valid_vocab_size(
            vllm_config,
            _FakeTokenizer(128815),
        )


def test_vllm_config_resolves_heterogeneous_step4_draft_tokenizer(monkeypatch):
    target_config = _make_model_config()
    draft_config = _make_model_config(
        architecture="Step4ForCausalLM",
        vocab_size=65536,
    )
    draft_tokenizer = _FakeTokenizer(64000)
    monkeypatch.setattr(
        "vllm.tokenizers.cached_tokenizer_from_config",
        lambda model_config: (
            draft_tokenizer
            if model_config is draft_config
            else pytest.fail("target tokenizer should be supplied by the caller")
        ),
    )
    vllm_config = SimpleNamespace(
        model_config=target_config,
        speculative_config=SimpleNamespace(
            method="draft_model",
            draft_model_config=draft_config,
            use_heterogeneous_vocab=True,
        ),
    )

    VllmConfig.resolve_valid_vocab_size(
        vllm_config,
        _FakeTokenizer(128815),
    )

    assert target_config.get_valid_vocab_size() == 128815
    assert draft_config.get_valid_vocab_size() == 64000


def test_draft_model_compares_effective_vocab_sizes():
    speculative_config = object.__new__(SpeculativeConfig)
    speculative_config.method = "draft_model"
    speculative_config.target_model_config = _make_model_config(valid_vocab_size=128815)
    speculative_config.draft_model_config = _make_model_config(
        architecture="Qwen2ForCausalLM",
    )

    with pytest.raises(ValueError, match="same effective vocabulary size"):
        speculative_config.verify_equal_vocab_size_if_draft_model()


def test_draft_model_defers_vocab_comparison_until_step4_is_resolved():
    speculative_config = object.__new__(SpeculativeConfig)
    speculative_config.method = "draft_model"
    speculative_config.target_model_config = _make_model_config()
    speculative_config.draft_model_config = _make_model_config(
        architecture="Qwen2ForCausalLM",
        vocab_size=128815,
    )

    speculative_config.verify_equal_vocab_size_if_draft_model()

    speculative_config.target_model_config.set_valid_vocab_size(128815)
    speculative_config.verify_equal_vocab_size_if_draft_model()
