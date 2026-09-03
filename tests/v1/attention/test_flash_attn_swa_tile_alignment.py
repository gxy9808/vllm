# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import fields

import pytest
import torch

from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionMetadata,
    _build_swa_tile_alignment_metadata,
    _get_swa_tile_alignment_buffers,
    _prepare_swa_tile_alignment_workspace,
    _SWATileAlignmentBuffers,
)
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func


def test_flash_attention_metadata_preserves_legacy_field_order():
    names = [field.name for field in fields(FlashAttentionMetadata)]

    assert names[21:23] == ["causal", "sliding_window"]
    assert names[-5:] == [
        "swa_query_start_loc",
        "swa_seqused_q",
        "swa_query_indices",
        "swa_num_query_rows",
        "swa_max_query_len",
    ]


def test_swa_tile_alignment_preserves_absolute_query_phase():
    query_start_loc = torch.tensor([0, 1, 5, 5, 7], dtype=torch.int32)
    query_start_loc_cpu = query_start_loc.clone()
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    num_computed_tokens = torch.tensor(
        [1773, 1773, 0, 511],
        dtype=torch.int32,
    )
    seq_lens = num_computed_tokens + query_lens
    buffers = _SWATileAlignmentBuffers.allocate(
        torch.device("cpu"),
        max_num_reqs=4,
        max_num_tokens=7,
    )

    padded_starts, seqused_q, query_indices = _build_swa_tile_alignment_metadata(
        query_start_loc,
        query_start_loc_cpu,
        seq_lens,
        num_actual_tokens=7,
        num_query_rows=7 + 31 * 4,
        buffers=buffers,
    )

    torch.testing.assert_close(
        padded_starts,
        torch.tensor([0, 14, 31, 31, 64], dtype=torch.int32),
    )
    torch.testing.assert_close(
        seqused_q,
        torch.tensor([14, 17, 0, 33], dtype=torch.int32),
    )
    torch.testing.assert_close(
        query_indices,
        torch.tensor([13, 27, 28, 29, 30, 62, 63], dtype=torch.int64),
    )


def test_swa_tile_alignment_maps_graph_padding_to_scratch_rows():
    query_start_loc = torch.tensor([0, 1, 5, 5, 7], dtype=torch.int32)
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    seq_lens = torch.tensor([1773, 1773, 0, 511], dtype=torch.int32) + query_lens
    num_actual_tokens = 10
    num_query_rows = num_actual_tokens + 31 * 4
    buffers = _SWATileAlignmentBuffers.allocate(
        torch.device("cpu"),
        max_num_reqs=4,
        max_num_tokens=num_actual_tokens,
    )

    padded_starts, _, query_indices = _build_swa_tile_alignment_metadata(
        query_start_loc,
        query_start_loc.clone(),
        seq_lens,
        num_actual_tokens=num_actual_tokens,
        num_query_rows=num_query_rows,
        buffers=buffers,
    )

    torch.testing.assert_close(
        query_indices,
        torch.tensor(
            [13, 27, 28, 29, 30, 62, 63, 131, 132, 133],
            dtype=torch.int64,
        ),
    )
    num_mapped_tokens = int(query_start_loc[-1])
    real_indices = query_indices[:num_mapped_tokens]
    scratch_indices = query_indices[num_mapped_tokens:]
    assert real_indices.max() < padded_starts[-1]
    assert scratch_indices.min() >= padded_starts[-1]
    assert scratch_indices.max() < num_query_rows


def test_swa_tile_alignment_clears_stale_graph_padding_outputs():
    query_start_loc = torch.tensor([0, 1, 5, 5, 7], dtype=torch.int32)
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    seq_lens = torch.tensor([1773, 1773, 0, 511], dtype=torch.int32) + query_lens
    num_mapped_tokens = int(query_start_loc[-1])
    num_actual_tokens = 10
    num_query_rows = num_actual_tokens + 31 * 4
    buffers = _SWATileAlignmentBuffers.allocate(
        torch.device("cpu"),
        max_num_reqs=4,
        max_num_tokens=num_actual_tokens,
    )
    _, _, query_indices = _build_swa_tile_alignment_metadata(
        query_start_loc,
        query_start_loc.clone(),
        seq_lens,
        num_actual_tokens=num_actual_tokens,
        num_query_rows=num_query_rows,
        buffers=buffers,
    )

    query = torch.arange(num_actual_tokens * 2, dtype=torch.float32).view(-1, 2)
    padded_query = torch.full((num_query_rows, 2), -1.0)
    padded_output = torch.full((num_query_rows, 2), 123.0)
    _prepare_swa_tile_alignment_workspace(
        padded_query,
        padded_output,
        query_indices,
        query,
    )

    real_output = torch.arange(num_mapped_tokens * 2, dtype=torch.float32).view(-1, 2)
    padded_output.index_copy_(
        0,
        query_indices[:num_mapped_tokens],
        real_output,
    )
    gathered_output = padded_output.index_select(0, query_indices)

    torch.testing.assert_close(
        padded_query.index_select(0, query_indices),
        query,
    )
    torch.testing.assert_close(
        gathered_output[:num_mapped_tokens],
        real_output,
    )
    assert torch.count_nonzero(gathered_output[num_mapped_tokens:]) == 0


def test_swa_tile_alignment_rejects_graph_padding_overlap():
    query_start_loc = torch.tensor([0, 1, 5, 5, 7], dtype=torch.int32)
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    seq_lens = torch.tensor([1773, 1773, 0, 511], dtype=torch.int32) + query_lens
    num_actual_tokens = 10
    buffers = _SWATileAlignmentBuffers.allocate(
        torch.device("cpu"),
        max_num_reqs=4,
        max_num_tokens=num_actual_tokens,
    )

    with pytest.raises(ValueError, match="insufficient query rows"):
        _build_swa_tile_alignment_metadata(
            query_start_loc,
            query_start_loc.clone(),
            seq_lens,
            num_actual_tokens=num_actual_tokens,
            num_query_rows=40,
            buffers=buffers,
        )


def test_swa_tile_alignment_shares_groups_but_isolates_ubatches():
    first = _get_swa_tile_alignment_buffers(
        torch.device("cpu"),
        max_num_reqs=4,
        max_num_tokens=8,
        num_ubatches=2,
    )
    second = _get_swa_tile_alignment_buffers(
        torch.device("cpu"),
        max_num_reqs=4,
        max_num_tokens=8,
        num_ubatches=2,
    )

    assert first is second
    assert first[0] is second[0]
    assert first[0] is not first[1]
    for buffer_name in vars(first[0]):
        first_tensor = getattr(first[0], buffer_name)
        second_tensor = getattr(first[1], buffer_name)
        assert first_tensor.data_ptr() != second_tensor.data_ptr()


def test_flash_attn_fa2_rejects_seqused_q():
    q = torch.empty((1, 1, 8), dtype=torch.float16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32)
    seqused_q = torch.tensor([1], dtype=torch.int32)
    seqused_k = torch.tensor([1], dtype=torch.int32)

    with pytest.raises(NotImplementedError, match="FA2 does not support seqused_q"):
        flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=1,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            fa_version=2,
        )


def test_flash_attn_fa4_rejects_seqused_q():
    q = torch.empty((1, 1, 8), dtype=torch.float16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32)
    seqused_q = torch.tensor([1], dtype=torch.int32)
    seqused_k = torch.tensor([1], dtype=torch.int32)

    with pytest.raises(NotImplementedError, match="FA4 does not support seqused_q"):
        flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=1,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            fa_version=4,
        )
