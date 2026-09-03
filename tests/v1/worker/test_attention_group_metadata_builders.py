# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.worker.utils import AttentionGroup


class _TestMetadataBuilder:
    requires_block_table_width = False

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.ubatch_id = -1


class _TestBackend:
    @classmethod
    def get_builder_cls(cls):
        return _TestMetadataBuilder


def test_attention_group_defaults_metadata_builder_to_ubatch_zero():
    group = AttentionGroup(
        backend=_TestBackend,
        layer_names=[],
        kv_cache_spec=object(),
        kv_cache_group_id=0,
    )

    group.create_metadata_builders(
        vllm_config=object(),
        device=torch.device("cpu"),
    )

    assert len(group.metadata_builders) == 1
    assert group.get_metadata_builder().ubatch_id == 0


def test_attention_group_binds_metadata_builder_ubatch_ids():
    group = AttentionGroup(
        backend=_TestBackend,
        layer_names=[],
        kv_cache_spec=object(),
        kv_cache_group_id=0,
    )

    group.create_metadata_builders(
        vllm_config=object(),
        device=torch.device("cpu"),
        num_metadata_builders=2,
    )

    assert [builder.ubatch_id for builder in group.metadata_builders] == [0, 1]
