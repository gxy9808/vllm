# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module

import pytest
import torch

pytest.importorskip("cutlass")

_paged_logits = import_module(
    "vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.indexer_ops."
    "prefill_paged_logits_sm90_steptron_gqa"
)


def _make_validation_inputs(*, block_table_cols: int):
    q_heads_per_kv = 4
    query_rows = _paged_logits._prefill_block_q_for_heads(q_heads_per_kv)
    # This is a padded selector/logits capacity, not the live KV width. A
    # one-column block table can be valid for a short request in this bucket.
    max_regions = _paged_logits.PREFILL_BLOCK_KV * 16
    index_q = torch.empty(
        (query_rows, 1, q_heads_per_kv, _paged_logits.HEAD_DIM),
        dtype=torch.bfloat16,
    )
    weights = torch.empty(
        (query_rows, 1, q_heads_per_kv),
        dtype=torch.float32,
    )
    summary = torch.empty(
        (1, _paged_logits.PREFILL_TMA_ROWS, 1, _paged_logits.HEAD_DIM),
        dtype=torch.uint8,
    )
    block_table = torch.zeros(
        (1, block_table_cols),
        dtype=torch.int32,
    )
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 1], dtype=torch.int32)
    q_runtime = torch.empty(
        (query_rows * q_heads_per_kv, _paged_logits.HEAD_DIM),
        dtype=torch.float8_e4m3fn,
    )
    kernel_weights = torch.empty(
        (query_rows, q_heads_per_kv),
        dtype=torch.float32,
    )
    out = torch.empty(
        (query_rows, max_regions),
        dtype=torch.float32,
    )
    return (
        index_q,
        weights,
        summary,
        block_table,
        cu_seqlens_q,
        cu_seqlens_k,
        q_runtime,
        kernel_weights,
        out,
    )


def test_padded_max_regions_does_not_require_padded_block_table_width():
    inputs = _make_validation_inputs(block_table_cols=1)

    query_rows, max_regions, q_heads_per_kv, block_q, batch_size = (
        _paged_logits._validate_inputs(*inputs)
    )

    assert query_rows == block_q
    assert max_regions == _paged_logits.PREFILL_BLOCK_KV * 16
    assert q_heads_per_kv == 4
    assert batch_size == 1


def test_prefill_paged_logits_rejects_empty_block_table_width():
    inputs = _make_validation_inputs(block_table_cols=0)

    with pytest.raises(ValueError, match="pages > 0"):
        _paged_logits._validate_inputs(*inputs)
