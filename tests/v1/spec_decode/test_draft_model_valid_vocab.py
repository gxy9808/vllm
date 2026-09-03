# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

import vllm.v1.spec_decode.draft_model as draft_model_module
from vllm.v1.spec_decode.draft_model import DraftModelProposer


def test_heterogeneous_vocab_mapping_uses_effective_vocab_sizes(monkeypatch):
    target_model_config = SimpleNamespace(
        tokenizer="target-tokenizer",
        trust_remote_code=False,
        hf_config=SimpleNamespace(model_type="step4"),
        get_valid_vocab_size=Mock(return_value=128815),
        get_vocab_size=Mock(side_effect=AssertionError("checkpoint vocab used")),
    )
    draft_model_config = SimpleNamespace(
        model="draft-model",
        tokenizer="draft-tokenizer",
        trust_remote_code=True,
        hf_config=SimpleNamespace(model_type="qwen2"),
        get_valid_vocab_size=Mock(return_value=64000),
        get_vocab_size=Mock(side_effect=AssertionError("checkpoint vocab used")),
    )
    speculative_config = SimpleNamespace(
        use_heterogeneous_vocab=True,
        target_model_config=target_model_config,
        draft_model_config=draft_model_config,
    )
    vllm_config = SimpleNamespace(speculative_config=speculative_config)

    def fake_base_init(self, **kwargs):
        self.speculative_config = kwargs["vllm_config"].speculative_config

    monkeypatch.setattr(
        draft_model_module.SpecDecodeBaseProposer,
        "__init__",
        fake_base_init,
    )
    monkeypatch.setattr(
        DraftModelProposer,
        "_raise_if_draft_tp_mismatch",
        lambda self: None,
    )

    tokenizers = {
        id(target_model_config): object(),
        id(draft_model_config): object(),
    }
    get_tokenizer = Mock(side_effect=lambda config: tokenizers[id(config)])
    monkeypatch.setattr(
        draft_model_module,
        "cached_tokenizer_from_config",
        get_tokenizer,
    )

    captured_mapping_kwargs = {}

    def fake_vocab_mapping(**kwargs):
        captured_mapping_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(draft_model_module, "VocabMapping", fake_vocab_mapping)

    DraftModelProposer(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )

    assert captured_mapping_kwargs == {
        "target_tokenizer": tokenizers[id(target_model_config)],
        "draft_tokenizer": tokenizers[id(draft_model_config)],
        "target_vocab_size": 128815,
        "draft_vocab_size": 64000,
        "device": torch.device("cpu"),
    }
    assert [mock_call.args for mock_call in get_tokenizer.call_args_list] == [
        (target_model_config,),
        (draft_model_config,),
    ]
    assert all(not mock_call.kwargs for mock_call in get_tokenizer.call_args_list)
    target_model_config.get_valid_vocab_size.assert_called_once_with()
    draft_model_config.get_valid_vocab_size.assert_called_once_with()
    target_model_config.get_vocab_size.assert_not_called()
    draft_model_config.get_vocab_size.assert_not_called()
