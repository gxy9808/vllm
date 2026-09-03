# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.model_loader.mtp_validation import (
    disable_mtp_completeness_check,
)
from vllm.models.step4 import model as step4_model


@pytest.mark.parametrize(
    ("tie_word_embeddings", "expected_skip_prefixes"),
    [
        (False, ["vision_model."]),
        (
            True,
            ["vision_model.", "lm_head."],
        ),
    ],
)
def test_step4_text_loader_skips_multimodal_weights_and_tied_lm_head(
    monkeypatch,
    tie_word_embeddings,
    expected_skip_prefixes,
):
    observed_skip_prefixes = None

    class FakeLoader:
        def __init__(self, _module, *, skip_prefixes):
            nonlocal observed_skip_prefixes
            observed_skip_prefixes = skip_prefixes

        def load_weights(self, _weights, *, mapper):
            del mapper
            return set()

    monkeypatch.setattr(step4_model, "AutoWeightsLoader", FakeLoader)
    model = SimpleNamespace(
        config=SimpleNamespace(tie_word_embeddings=tie_word_embeddings),
        hf_to_vllm_mapper=object(),
        model=nn.Module(),
    )

    assert step4_model.Step4ForCausalLM.load_weights(model, []) == set()
    assert observed_skip_prefixes == expected_skip_prefixes


class _DummyStep4TextModel(nn.Module):
    hf_to_vllm_mapper = step4_model.Step4ForCausalLM.hf_to_vllm_mapper

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.model = nn.Module()


def test_step4_text_loader_accepts_known_multimodal_checkpoint_weights():
    model = _DummyStep4TextModel()
    weights = [
        ("vision_model.transformer.layers.0.weight", torch.empty(0)),
        ("vit_large_projector.weight", torch.empty(0)),
    ]

    with disable_mtp_completeness_check():
        assert step4_model.Step4ForCausalLM.load_weights(model, weights) == set()


@pytest.mark.parametrize(
    "unexpected_name",
    [
        "vision_model_extra.weight",
        "vit_large_projector.bias",
        "vit_large_projector.weight_scale",
        "unknown.weight",
    ],
)
def test_step4_text_loader_rejects_other_unexpected_weights(unexpected_name):
    model = _DummyStep4TextModel()

    with (
        disable_mtp_completeness_check(),
        pytest.raises(ValueError, match="There is no module or parameter"),
    ):
        step4_model.Step4ForCausalLM.load_weights(
            model, [(unexpected_name, torch.empty(0))]
        )
