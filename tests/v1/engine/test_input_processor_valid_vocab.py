# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.v1.engine.input_processor import InputProcessor


class _SparseTokenizer:
    vocab_size = 2
    max_token_id = 1

    def __len__(self) -> int:
        return 2

    def get_vocab(self) -> dict[str, int]:
        return {"zero": 0, "sparse-token": 1000}


def test_explicit_valid_vocab_rejects_negative_token_id():
    processor = InputProcessor.__new__(InputProcessor)
    processor.model_config = SimpleNamespace(
        valid_vocab_size=5,
        get_valid_vocab_size=lambda: 5,
        get_vocab_size=lambda: 8,
    )
    processor.renderer = SimpleNamespace(tokenizer=None)
    processor._validate_prompt_len = lambda *_: None

    with pytest.raises(VLLMValidationError, match=r"Token id -1 .* vocabulary"):
        processor._validate_model_input(
            {
                "type": "token",
                "prompt_token_ids": [0, -1],
            },
            prompt_type="decoder",
        )


def test_prompt_validation_uses_sparse_tokenizer_upper_bound():
    processor = InputProcessor.__new__(InputProcessor)
    processor.model_config = SimpleNamespace(
        valid_vocab_size=None,
        get_valid_vocab_size=lambda: 2,
        get_vocab_size=lambda: 2,
    )
    processor.renderer = SimpleNamespace(tokenizer=_SparseTokenizer())
    processor._validate_prompt_len = lambda *_: None

    # The sparse ID is exposed by get_vocab(), even though len() and the
    # legacy max_token_id property only describe the contiguous prefix.
    processor._validate_model_input(
        {
            "type": "token",
            "prompt_token_ids": [1000],
        },
        prompt_type="decoder",
    )


def test_prompt_validation_rejects_negative_token_id_without_explicit_boundary():
    processor = InputProcessor.__new__(InputProcessor)
    processor.model_config = SimpleNamespace(
        valid_vocab_size=None,
        get_valid_vocab_size=lambda: 8,
        get_vocab_size=lambda: 8,
    )
    processor.renderer = SimpleNamespace(tokenizer=_SparseTokenizer())
    processor._validate_prompt_len = lambda *_: None

    with pytest.raises(VLLMValidationError, match=r"Token id -1 .* vocabulary"):
        processor._validate_model_input(
            {
                "type": "token",
                "prompt_token_ids": [-1, 0],
            },
            prompt_type="decoder",
        )
