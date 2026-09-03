# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.v1.structured_output.backend_xgrammar as backend_xgrammar_module
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend


@pytest.mark.parametrize(
    ("valid_vocab_size", "expected_vocab_size"),
    [
        pytest.param(5, 5, id="explicit-valid-boundary"),
        pytest.param(None, 8, id="legacy-tokenizer-vocab"),
    ],
)
def test_xgrammar_mistral_respects_valid_vocab_boundary(
    monkeypatch,
    valid_vocab_size,
    expected_vocab_size,
):
    tokenizer_info_calls = []

    class FakeTokenizerInfo:
        def __init__(self, **kwargs):
            tokenizer_info_calls.append(kwargs)

    class FakeGrammarCompiler:
        def __init__(self, *_args, **_kwargs):
            pass

    fake_xgr = SimpleNamespace(
        TokenizerInfo=FakeTokenizerInfo,
        GrammarCompiler=FakeGrammarCompiler,
        VocabType=SimpleNamespace(RAW="raw", BYTE_FALLBACK="byte_fallback"),
    )
    monkeypatch.setattr(backend_xgrammar_module, "xgr", fake_xgr)
    monkeypatch.setattr(
        backend_xgrammar_module,
        "is_mistral_tokenizer",
        lambda _tokenizer: True,
    )
    tokenizer = SimpleNamespace(
        eos_token_id=1,
        vocab=[f"token-{index}" for index in range(8)],
        is_tekken=False,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(valid_vocab_size=valid_vocab_size),
        structured_outputs_config=SimpleNamespace(
            disable_any_whitespace=False,
        ),
        speculative_config=None,
    )

    backend = XgrammarBackend(
        vllm_config,
        tokenizer=tokenizer,
        vocab_size=5,
    )

    assert backend.vocab_size == expected_vocab_size
    assert tokenizer_info_calls[0]["vocab_size"] == expected_vocab_size
