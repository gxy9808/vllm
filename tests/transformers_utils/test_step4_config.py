# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest

from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.step4 import Step4Config, Step4MTPConfig


def test_step4_config_defaults_to_independent_lm_head():
    config = Step4Config()

    assert config.tie_word_embeddings is False
    assert Step4Config(**config.to_dict()).tie_word_embeddings is False


def test_step4_config_preserves_explicit_tied_embeddings():
    config = Step4Config(tie_word_embeddings=True)

    assert config.tie_word_embeddings is True
    assert Step4Config(**config.to_dict()).tie_word_embeddings is True


def test_step4_config_roundtrips_dense_and_mtp_layer_types():
    layer_types = ["sliding_attention", "full_attention", "full_attention"]
    config = Step4Config(
        num_hidden_layers=2,
        num_nextn_predict_layers=1,
        layer_types=layer_types,
    )

    assert config.layer_types == layer_types[:2]
    assert config.layer_types_with_mtp == layer_types

    restored = Step4Config(**config.to_dict())
    assert restored.layer_types == layer_types[:2]
    assert restored.layer_types_with_mtp == layer_types


def test_step4_config_rejects_layer_types_missing_mtp_entries():
    with pytest.raises(ValueError, match="expected at least 3 entries"):
        Step4Config(
            num_hidden_layers=2,
            num_nextn_predict_layers=1,
            layer_types=["sliding_attention", "full_attention"],
        )


def test_step4_config_rejects_conflicting_serialized_layer_types():
    with pytest.raises(ValueError, match="must match the dense prefix"):
        Step4Config(
            num_hidden_layers=2,
            num_nextn_predict_layers=1,
            layer_types=["full_attention", "full_attention"],
            layer_types_with_mtp=[
                "sliding_attention",
                "full_attention",
                "full_attention",
            ],
        )


def test_step4_mtp_config_has_independent_model_type():
    config = Step4MTPConfig()

    assert config.model_type == "step4_mtp"
    assert config.to_dict()["model_type"] == "step4_mtp"
    assert config.architectures == ["Step4MTP"]
    assert config.num_nextn_predict_layers == 1
    assert Step4Config.model_type == "step4"


def test_step4_mtp_standalone_config_loads_without_remote_code(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "step4_mtp",
                "vocab_size": 128896,
                "num_hidden_layers": 92,
                "num_nextn_predict_layers": 1,
            }
        ),
        encoding="utf-8",
    )

    config = get_config(tmp_path, trust_remote_code=False)

    assert isinstance(config, Step4MTPConfig)
    assert config.model_type == "step4_mtp"
    assert config.architectures == ["Step4MTP"]
