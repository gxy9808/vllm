# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel tests for Step multi-module MTP rejection re-prefill."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.spec_decode.step3p5 import (
    _cache_step_mtp_inputs_kernel,
    _pad_step_mtp_trailing_slots_kernel,
    _prepare_step_mtp_hidden_states_kernel,
    _prepare_step_mtp_input_buffers_kernel,
    _publish_step_mtp_history_lens_kernel,
    _shift_step_mtp_inputs_kernel,
)
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID
from vllm.v1.worker.block_table import BlockTable

pytest.importorskip("triton")
if not current_platform.is_cuda_alike():
    pytest.skip("CUDA required for Step MTP kernel tests", allow_module_level=True)


def _compute_packed_slot_mapping(
    positions,
    block_ids,
    query_start_loc,
    last_token_indices,
    manager_block_size,
    kernel_block_size=None,
):
    kernel_block_size = kernel_block_size or manager_block_size
    blocks_per_kv_block = manager_block_size // kernel_block_size
    block_table = BlockTable(
        block_size=manager_block_size,
        max_num_reqs=block_ids.shape[0],
        max_num_blocks_per_req=block_ids.shape[1] // blocks_per_kv_block,
        max_num_batched_tokens=positions.shape[0],
        pin_memory=False,
        device=positions.device,
        kernel_block_size=kernel_block_size,
        cp_kv_cache_interleave_size=1,
    )
    block_table.block_table.gpu[: block_ids.shape[0], : block_ids.shape[1]].copy_(
        block_ids
    )
    block_table.compute_slot_mapping(
        block_ids.shape[0],
        query_start_loc,
        positions,
    )
    slot_mapping = block_table.slot_mapping.gpu[: positions.shape[0]]
    _pad_step_mtp_trailing_slots_kernel[(block_ids.shape[0],)](
        slot_mapping,
        query_start_loc,
        last_token_indices,
        PADDING_SLOT_ID,
        BLOCK_SIZE=256,
    )
    return slot_mapping


def test_step_mtp_reprefill_packs_and_caches_by_stable_request_slot():
    device = torch.device(current_platform.device_type)
    batch_size = 4
    query_len = 4
    num_tokens = batch_size * query_len
    hidden_size = 4
    history_window = 2

    query_start_loc = torch.arange(
        0,
        num_tokens + 1,
        query_len,
        dtype=torch.int32,
        device=device,
    )
    target_input_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)
    target_positions = (
        torch.arange(query_len, dtype=torch.int64, device=device)
        .repeat(batch_size)
        .add_(
            torch.arange(
                20, 20 + batch_size * query_len, query_len, device=device
            ).repeat_interleave(query_len)
        )
    )
    target_hidden_states = (
        torch.arange(num_tokens * hidden_size, dtype=torch.float32, device=device)
        .view(num_tokens, hidden_size)
        .add_(2000)
    )
    next_token_ids = torch.arange(
        100, 100 + batch_size, dtype=torch.int32, device=device
    )
    num_rejected = torch.arange(batch_size, dtype=torch.int32, device=device)
    history_slot_mapping = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device=device)
    prefill_token_overrides = torch.full(
        (batch_size, 3), -1, dtype=torch.int32, device=device
    )

    cached_input_ids = (
        torch.arange(batch_size * history_window, dtype=torch.int32, device=device)
        .view(batch_size, history_window)
        .add_(500)
    )
    cached_input_embeds = (
        torch.arange(
            batch_size * history_window * hidden_size,
            dtype=torch.float32,
            device=device,
        )
        .view(batch_size, history_window, hidden_size)
        .add_(3000)
    )
    cached_hidden_states = cached_input_embeds + 1000
    cached_history_lens = torch.full(
        (batch_size,), history_window, dtype=torch.int32, device=device
    )

    draft_input_ids = torch.full((num_tokens,), -99, dtype=torch.int32, device=device)
    draft_positions = torch.full((num_tokens,), -99, dtype=torch.int64, device=device)
    seq_lens = torch.full((batch_size,), 40, dtype=torch.int32, device=device)
    last_token_indices = torch.zeros(batch_size, dtype=torch.int64, device=device)

    _prepare_step_mtp_input_buffers_kernel[(batch_size,)](
        last_token_indices,
        draft_input_ids,
        draft_positions,
        seq_lens,
        target_input_ids,
        target_positions,
        cached_input_ids,
        cached_input_ids.stride(0),
        cached_history_lens,
        history_slot_mapping,
        next_token_ids,
        prefill_token_overrides,
        prefill_token_overrides.stride(0),
        num_rejected,
        query_start_loc,
        history_window,
        BLOCK_SIZE=16,
    )

    expected_ids = torch.full_like(draft_input_ids, -99)
    expected_positions = torch.full_like(draft_positions, -99)
    expected_hidden = torch.full((num_tokens, hidden_size), -99.0, device=device)
    expected_embeds = torch.full_like(expected_hidden, -99.0)
    expected_last_indices = []
    expected_seq_lens = []

    for req_index in range(batch_size):
        start = req_index * query_len
        rejected = req_index
        num_reprefill = max(rejected - 1, 0)
        num_current = query_len - rejected
        last = start + num_reprefill + num_current - 1
        expected_last_indices.append(last)
        expected_seq_lens.append(40 - num_reprefill)

        slot = history_slot_mapping[req_index].item()
        for repair_index in range(num_reprefill):
            cache_index = history_window - num_reprefill + repair_index
            out_index = start + repair_index
            expected_ids[out_index] = cached_input_ids[slot, cache_index]
            expected_positions[out_index] = (
                target_positions[start] - num_reprefill + repair_index
            )
            expected_hidden[out_index] = cached_hidden_states[slot, cache_index]
            expected_embeds[out_index] = cached_input_embeds[slot, cache_index]

        for current_index in range(num_current):
            out_index = start + num_reprefill + current_index
            expected_positions[out_index] = target_positions[start + current_index]
            expected_hidden[out_index] = target_hidden_states[start + current_index]
            if current_index + 1 < num_current:
                expected_ids[out_index] = target_input_ids[start + current_index + 1]
            else:
                expected_ids[out_index] = next_token_ids[req_index]
            expected_embeds[out_index] = expected_ids[out_index].float() + torch.arange(
                hidden_size, device=device
            )

    assert torch.equal(
        last_token_indices,
        torch.tensor(expected_last_indices, dtype=torch.int64, device=device),
    )
    assert torch.equal(
        seq_lens,
        torch.tensor(expected_seq_lens, dtype=torch.int32, device=device),
    )
    for req_index, last in enumerate(expected_last_indices):
        start = req_index * query_len
        assert torch.equal(
            draft_input_ids[start : last + 1],
            expected_ids[start : last + 1],
        )
        assert torch.equal(
            draft_positions[start : last + 1],
            expected_positions[start : last + 1],
        )

    draft_input_embeds = draft_input_ids[:, None].float() + torch.arange(
        hidden_size, device=device
    )
    draft_hidden_states = torch.full_like(draft_input_embeds, -99.0)
    _prepare_step_mtp_hidden_states_kernel[(batch_size, 1, 1)](
        draft_hidden_states,
        draft_hidden_states.stride(0),
        target_hidden_states,
        target_hidden_states.stride(0),
        cached_hidden_states,
        cached_hidden_states.stride(0),
        cached_hidden_states.stride(1),
        draft_input_embeds,
        draft_input_embeds.stride(0),
        cached_input_embeds,
        cached_input_embeds.stride(0),
        cached_input_embeds.stride(1),
        cached_history_lens,
        history_slot_mapping,
        num_rejected,
        query_start_loc,
        history_window,
        hidden_size,
        BLOCK_SIZE_Q=16,
        BLOCK_SIZE_H=16,
        USE_INPUT_EMBEDS=True,
    )

    for req_index, last in enumerate(expected_last_indices):
        start = req_index * query_len
        assert torch.equal(
            draft_hidden_states[start : last + 1],
            expected_hidden[start : last + 1],
        )
        assert torch.equal(
            draft_input_embeds[start : last + 1],
            expected_embeds[start : last + 1],
        )

    manager_block_size = 8
    kernel_block_size = 4
    block_table = torch.arange(batch_size * 16, dtype=torch.int32, device=device).view(
        batch_size, 16
    )
    draft_positions.clamp_min_(0)
    slot_mapping = _compute_packed_slot_mapping(
        draft_positions,
        block_table,
        query_start_loc,
        last_token_indices,
        manager_block_size,
        kernel_block_size,
    )
    for req_index, last in enumerate(expected_last_indices):
        start = req_index * query_len
        for token_index in range(start, last + 1):
            position = draft_positions[token_index]
            block_id = block_table[req_index, position // kernel_block_size]
            expected_slot = block_id * kernel_block_size + position % kernel_block_size
            assert slot_mapping[token_index] == expected_slot
        assert torch.all(
            slot_mapping[last + 1 : (req_index + 1) * query_len] == PADDING_SLOT_ID
        )

    _cache_step_mtp_inputs_kernel[(batch_size, 1)](
        draft_input_ids,
        draft_input_embeds,
        draft_input_embeds.stride(0),
        draft_hidden_states,
        draft_hidden_states.stride(0),
        cached_input_ids,
        cached_input_ids.stride(0),
        cached_input_embeds,
        cached_input_embeds.stride(0),
        cached_input_embeds.stride(1),
        cached_hidden_states,
        cached_hidden_states.stride(0),
        cached_hidden_states.stride(1),
        cached_history_lens,
        history_slot_mapping,
        last_token_indices,
        query_start_loc,
        history_window,
        hidden_size,
        BLOCK_SIZE=16,
        USE_INPUT_EMBEDS=True,
    )
    _publish_step_mtp_history_lens_kernel[(batch_size,)](
        cached_history_lens,
        history_slot_mapping,
        last_token_indices,
        query_start_loc,
        history_window,
    )
    for req_index, last in enumerate(expected_last_indices):
        slot = history_slot_mapping[req_index].item()
        expected_slice = slice(last - history_window + 1, last + 1)
        assert cached_history_lens[slot] == history_window
        assert torch.equal(cached_input_ids[slot], draft_input_ids[expected_slice])
        assert torch.equal(
            cached_input_embeds[slot], draft_input_embeds[expected_slice]
        )
        assert torch.equal(
            cached_hidden_states[slot], draft_hidden_states[expected_slice]
        )


def test_step_mtp_history_cache_rolls_across_short_queries():
    device = torch.device(current_platform.device_type)
    hidden_size = 4
    history_window = 3
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    last_token_indices = torch.tensor([0], dtype=torch.int64, device=device)
    history_slot_mapping = torch.tensor([0], dtype=torch.int32, device=device)
    cached_history_lens = torch.tensor([3], dtype=torch.int32, device=device)
    cached_input_ids = torch.tensor([[10, 11, 12]], dtype=torch.int32, device=device)
    cached_input_embeds = cached_input_ids[:, :, None].float() + torch.arange(
        hidden_size, device=device
    )
    cached_hidden_states = cached_input_embeds + 1000

    def append(token_id: int) -> None:
        draft_input_ids = torch.tensor([token_id], dtype=torch.int32, device=device)
        draft_input_embeds = draft_input_ids[:, None].float() + torch.arange(
            hidden_size, device=device
        )
        draft_hidden_states = draft_input_embeds + 1000
        _cache_step_mtp_inputs_kernel[(1, 1)](
            draft_input_ids,
            draft_input_embeds,
            draft_input_embeds.stride(0),
            draft_hidden_states,
            draft_hidden_states.stride(0),
            cached_input_ids,
            cached_input_ids.stride(0),
            cached_input_embeds,
            cached_input_embeds.stride(0),
            cached_input_embeds.stride(1),
            cached_hidden_states,
            cached_hidden_states.stride(0),
            cached_hidden_states.stride(1),
            cached_history_lens,
            history_slot_mapping,
            last_token_indices,
            query_start_loc,
            history_window,
            hidden_size,
            BLOCK_SIZE=16,
            USE_INPUT_EMBEDS=True,
        )
        _publish_step_mtp_history_lens_kernel[(1,)](
            cached_history_lens,
            history_slot_mapping,
            last_token_indices,
            query_start_loc,
            history_window,
        )

    append(20)
    assert cached_history_lens.tolist() == [3]
    assert cached_input_ids.tolist() == [[11, 12, 20]]
    assert torch.equal(
        cached_input_embeds,
        cached_input_ids[:, :, None].float() + torch.arange(hidden_size, device=device),
    )
    assert torch.equal(cached_hidden_states, cached_input_embeds + 1000)

    append(21)
    assert cached_input_ids.tolist() == [[12, 20, 21]]
    assert torch.equal(
        cached_input_embeds,
        cached_input_ids[:, :, None].float() + torch.arange(hidden_size, device=device),
    )
    assert torch.equal(cached_hidden_states, cached_input_embeds + 1000)


def test_step_mtp_k1_uses_zero_length_history_window():
    device = torch.device(current_platform.device_type)
    hidden_size = 4
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    last_token_indices = torch.tensor([0], dtype=torch.int64, device=device)
    history_slot_mapping = torch.tensor([0], dtype=torch.int32, device=device)
    cached_history_lens = torch.tensor([1], dtype=torch.int32, device=device)
    cached_input_ids = torch.tensor([[77]], dtype=torch.int32, device=device)
    cached_input_embeds = torch.full((1, 1, hidden_size), 88.0, device=device)
    cached_hidden_states = torch.full_like(cached_input_embeds, 99.0)
    draft_input_ids = torch.tensor([20], dtype=torch.int32, device=device)
    draft_input_embeds = torch.full((1, hidden_size), 20.0, device=device)
    draft_hidden_states = torch.full((1, hidden_size), 30.0, device=device)

    _cache_step_mtp_inputs_kernel[(1, 1)](
        draft_input_ids,
        draft_input_embeds,
        draft_input_embeds.stride(0),
        draft_hidden_states,
        draft_hidden_states.stride(0),
        cached_input_ids,
        cached_input_ids.stride(0),
        cached_input_embeds,
        cached_input_embeds.stride(0),
        cached_input_embeds.stride(1),
        cached_hidden_states,
        cached_hidden_states.stride(0),
        cached_hidden_states.stride(1),
        cached_history_lens,
        history_slot_mapping,
        last_token_indices,
        query_start_loc,
        0,
        hidden_size,
        BLOCK_SIZE=16,
        USE_INPUT_EMBEDS=True,
    )
    _publish_step_mtp_history_lens_kernel[(1,)](
        cached_history_lens,
        history_slot_mapping,
        last_token_indices,
        query_start_loc,
        0,
    )

    assert cached_history_lens.tolist() == [0]
    assert cached_input_ids.tolist() == [[77]]
    assert torch.all(cached_input_embeds == 88)
    assert torch.all(cached_hidden_states == 99)


def test_step_mtp_text_kernels_do_not_require_cached_embeddings():
    device = torch.device(current_platform.device_type)
    hidden_size = 4
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    last_token_indices = torch.tensor([0], dtype=torch.int64, device=device)
    history_slot_mapping = torch.tensor([0], dtype=torch.int32, device=device)
    cached_history_lens = torch.zeros(1, dtype=torch.int32, device=device)
    num_rejected = torch.zeros(1, dtype=torch.int32, device=device)
    target_hidden_states = torch.arange(
        hidden_size, dtype=torch.float32, device=device
    ).unsqueeze(0)
    draft_hidden_states = torch.full_like(target_hidden_states, -1)
    draft_input_embeds = torch.full_like(target_hidden_states, 7)
    cached_hidden_states = torch.zeros(
        (1, 1, hidden_size), dtype=torch.float32, device=device
    )

    _prepare_step_mtp_hidden_states_kernel[(1, 1, 1)](
        draft_hidden_states,
        draft_hidden_states.stride(0),
        target_hidden_states,
        target_hidden_states.stride(0),
        cached_hidden_states,
        cached_hidden_states.stride(0),
        cached_hidden_states.stride(1),
        draft_input_embeds,
        draft_input_embeds.stride(0),
        None,
        0,
        0,
        cached_history_lens,
        history_slot_mapping,
        num_rejected,
        query_start_loc,
        1,
        hidden_size,
        BLOCK_SIZE_Q=16,
        BLOCK_SIZE_H=16,
        USE_INPUT_EMBEDS=False,
    )
    assert torch.equal(draft_hidden_states, target_hidden_states)

    draft_input_ids = torch.tensor([7], dtype=torch.int32, device=device)
    cached_input_ids = torch.full((1, 1), -1, dtype=torch.int32, device=device)
    _cache_step_mtp_inputs_kernel[(1, 1)](
        draft_input_ids,
        draft_input_embeds,
        draft_input_embeds.stride(0),
        draft_hidden_states,
        draft_hidden_states.stride(0),
        cached_input_ids,
        cached_input_ids.stride(0),
        None,
        0,
        0,
        cached_hidden_states,
        cached_hidden_states.stride(0),
        cached_hidden_states.stride(1),
        cached_history_lens,
        history_slot_mapping,
        last_token_indices,
        query_start_loc,
        1,
        hidden_size,
        BLOCK_SIZE=16,
        USE_INPUT_EMBEDS=False,
    )
    assert cached_input_ids.tolist() == [[7]]
    assert torch.equal(cached_hidden_states[0, 0], target_hidden_states[0])


def test_step_mtp_reprefill_masks_short_history_and_skips_padding_row():
    device = torch.device(current_platform.device_type)
    hidden_size = 4
    history_window = 2
    query_start_loc = torch.tensor([0, 4, 8, 12], dtype=torch.int32, device=device)
    target_input_ids = torch.arange(12, dtype=torch.int32, device=device)
    target_positions = torch.tensor(
        [10, 11, 12, 13, 30, 31, 32, 33, 20, 21, 22, 23],
        dtype=torch.int64,
        device=device,
    )
    target_hidden_states = (
        torch.arange(12 * hidden_size, dtype=torch.float32, device=device)
        .view(12, hidden_size)
        .add_(2000)
    )
    next_token_ids = torch.tensor([100, 101, 102], dtype=torch.int32, device=device)
    num_rejected = torch.tensor([3, 0, 3], dtype=torch.int32, device=device)
    history_slot_mapping = torch.tensor([0, -1, 1], dtype=torch.int32, device=device)
    prefill_token_overrides = torch.full((3, 3), -1, dtype=torch.int32, device=device)

    cached_input_ids = (
        torch.arange(3 * history_window, dtype=torch.int32, device=device)
        .view(3, history_window)
        .add_(500)
    )
    cached_input_embeds = (
        torch.arange(
            3 * history_window * hidden_size,
            dtype=torch.float32,
            device=device,
        )
        .view(3, history_window, hidden_size)
        .add_(3000)
    )
    cached_hidden_states = cached_input_embeds + 1000
    cached_history_lens = torch.tensor([1, 2, 2], dtype=torch.int32, device=device)

    draft_input_ids = torch.full((12,), -99, dtype=torch.int32, device=device)
    draft_positions = torch.full((12,), -99, dtype=torch.int64, device=device)
    seq_lens = torch.tensor([40, 50, 60], dtype=torch.int32, device=device)
    last_token_indices = torch.zeros(3, dtype=torch.int64, device=device)

    _prepare_step_mtp_input_buffers_kernel[(3,)](
        last_token_indices,
        draft_input_ids,
        draft_positions,
        seq_lens,
        target_input_ids,
        target_positions,
        cached_input_ids,
        cached_input_ids.stride(0),
        cached_history_lens,
        history_slot_mapping,
        next_token_ids,
        prefill_token_overrides,
        prefill_token_overrides.stride(0),
        num_rejected,
        query_start_loc,
        history_window,
        BLOCK_SIZE=16,
    )

    assert last_token_indices.tolist() == [1, 3, 10]
    assert seq_lens.tolist() == [39, 0, 58]
    assert draft_input_ids[0] == cached_input_ids[0, 1]
    assert draft_input_ids[1] == next_token_ids[0]
    assert torch.all(draft_input_ids[4:8] == -99)
    assert torch.equal(draft_input_ids[8:10], cached_input_ids[1])
    assert draft_input_ids[10] == next_token_ids[2]
    assert draft_positions[:2].tolist() == [9, 10]
    assert torch.all(draft_positions[4:8] == -99)
    assert draft_positions[8:11].tolist() == [18, 19, 20]

    draft_input_embeds = draft_input_ids[:, None].float() + torch.arange(
        hidden_size, device=device
    )
    draft_hidden_states = torch.full_like(draft_input_embeds, -99.0)
    _prepare_step_mtp_hidden_states_kernel[(3, 1, 1)](
        draft_hidden_states,
        draft_hidden_states.stride(0),
        target_hidden_states,
        target_hidden_states.stride(0),
        cached_hidden_states,
        cached_hidden_states.stride(0),
        cached_hidden_states.stride(1),
        draft_input_embeds,
        draft_input_embeds.stride(0),
        cached_input_embeds,
        cached_input_embeds.stride(0),
        cached_input_embeds.stride(1),
        cached_history_lens,
        history_slot_mapping,
        num_rejected,
        query_start_loc,
        history_window,
        hidden_size,
        BLOCK_SIZE_Q=16,
        BLOCK_SIZE_H=16,
        USE_INPUT_EMBEDS=True,
    )

    assert torch.equal(draft_hidden_states[0], cached_hidden_states[0, 1])
    assert torch.equal(draft_hidden_states[1], target_hidden_states[0])
    assert torch.all(draft_hidden_states[4:8] == -99)
    assert torch.equal(draft_hidden_states[8:10], cached_hidden_states[1])
    assert torch.equal(draft_hidden_states[10], target_hidden_states[8])
    assert torch.equal(draft_input_embeds[0], cached_input_embeds[0, 1])

    block_size = 4
    block_table = torch.arange(3 * 16, dtype=torch.int32, device=device).view(3, 16)
    draft_positions.clamp_min_(0)
    slot_mapping = _compute_packed_slot_mapping(
        draft_positions,
        block_table,
        query_start_loc,
        last_token_indices,
        block_size,
    )
    assert torch.all(slot_mapping[4:8] == PADDING_SLOT_ID)

    _cache_step_mtp_inputs_kernel[(3, 1)](
        draft_input_ids,
        draft_input_embeds,
        draft_input_embeds.stride(0),
        draft_hidden_states,
        draft_hidden_states.stride(0),
        cached_input_ids,
        cached_input_ids.stride(0),
        cached_input_embeds,
        cached_input_embeds.stride(0),
        cached_input_embeds.stride(1),
        cached_hidden_states,
        cached_hidden_states.stride(0),
        cached_hidden_states.stride(1),
        cached_history_lens,
        history_slot_mapping,
        last_token_indices,
        query_start_loc,
        history_window,
        hidden_size,
        BLOCK_SIZE=16,
        USE_INPUT_EMBEDS=True,
    )
    _publish_step_mtp_history_lens_kernel[(3,)](
        cached_history_lens,
        history_slot_mapping,
        last_token_indices,
        query_start_loc,
        history_window,
    )

    assert cached_history_lens.tolist() == [2, 2, 2]
    assert torch.equal(cached_input_ids[0], draft_input_ids[:2])
    assert torch.equal(cached_input_ids[1], draft_input_ids[9:11])

    shifted_input_ids = torch.tensor([700, 701, 702], dtype=torch.int32, device=device)
    shifted_input_embeds = shifted_input_ids[:, None].float() + torch.arange(
        hidden_size, device=device
    )
    _shift_step_mtp_inputs_kernel[(3, 1)](
        draft_input_ids,
        draft_input_embeds,
        draft_input_embeds.stride(0),
        shifted_input_ids,
        shifted_input_embeds,
        shifted_input_embeds.stride(0),
        history_slot_mapping,
        query_start_loc,
        last_token_indices,
        hidden_size,
        BLOCK_SIZE_Q=16,
        BLOCK_SIZE_H=16,
    )

    assert draft_input_ids[0] == next_token_ids[0]
    assert draft_input_ids[1] == shifted_input_ids[0]
    assert torch.all(draft_input_ids[4:8] == -99)
    assert draft_input_ids[8] == cached_input_ids[1, 0]
    assert draft_input_ids[9] == next_token_ids[2]
    assert draft_input_ids[10] == shifted_input_ids[2]
