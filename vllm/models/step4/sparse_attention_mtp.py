# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP-only extension for the Step4 DSA attention path.

The ordinary implementation remains MTP-free. This module owns fixed
transaction storage and CUDA kernels used to save verifier online-softmax
states and install the last accepted state after verification.
"""

from __future__ import annotations

from numbers import Integral

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .sparse_summary_cache import (
    _step4_sparse_csa_compact_clear_completed_kernel,
    _step4_sparse_csa_compact_clear_scratch_kernel,
)

_MTP_ACTION_NONE = 0
_MTP_ACTION_CLEAR = 1
_MTP_ACTION_INSTALL = 2
_MTP_ACTION_INSTALL_PRE = 3
# Compatibility for tests and callers written against the pre-install name.
_MTP_ACTION_RESTORE = _MTP_ACTION_INSTALL


_MTP_FILL_SCRATCH_ALIGNMENT_DYNAMIC_ARGS = (
    "scratch_region_ids",
    "scratch_row_map",
    "scratch_reset_map",
    "flat_slot",
    "reset_slots",
    "token_valid",
    "token_positions",
    "valid_tokens_ptr",
)


@triton.jit(
    do_not_specialize=(
        "num_decode_requests",
        "valid_requests_ptr",
        "valid_tokens_ptr",
    ),
    do_not_specialize_on_alignment=(
        "query_start_loc",
        "layout_valid",
        "q1_valid",
        "transaction_valid",
        "valid_requests_ptr",
        "valid_tokens_ptr",
    ),
)
def _step4_dsa_mtp_partition_decode_validity_kernel(
    query_start_loc,
    layout_valid,
    q1_valid,
    transaction_valid,
    valid_requests_ptr,
    valid_tokens_ptr,
    num_decode_requests,
    max_rows_per_req: tl.constexpr,
) -> None:
    """Partition a decode prefix without assuming q1/q2+ request ordering."""
    request = tl.program_id(0)
    row_offset = tl.program_id(1)
    live_requests = tl.load(valid_requests_ptr).to(tl.int32)
    live_tokens = tl.load(valid_tokens_ptr).to(tl.int32)
    request_valid = request < tl.minimum(num_decode_requests, live_requests)
    request_start = tl.load(
        query_start_loc + request,
        mask=request_valid,
        other=0,
    ).to(tl.int64)
    request_end = tl.load(
        query_start_loc + request + 1,
        mask=request_valid,
        other=0,
    ).to(tl.int64)
    request_rows = request_end - request_start
    source_row = request_start + row_offset
    source_in_range = (
        request_valid
        & (row_offset < request_rows)
        & (source_row >= 0)
        & (source_row < live_tokens)
    )
    safe_source_row = tl.where(source_in_range, source_row, 0)
    valid = source_in_range & (
        tl.load(
            layout_valid + safe_source_row,
            mask=source_in_range,
            other=0,
        )
        != 0
    )
    tl.store(
        q1_valid + safe_source_row,
        valid & (request_rows == 1),
        mask=source_in_range,
    )
    tl.store(
        transaction_valid + safe_source_row,
        valid & (request_rows > 1),
        mask=source_in_range,
    )


@triton.jit(
    do_not_specialize=("valid_tokens_ptr", "scratch_rows"),
    do_not_specialize_on_alignment=(
        "valid_tokens_ptr",
        "scratch_rows",
    )
    + _MTP_FILL_SCRATCH_ALIGNMENT_DYNAMIC_ARGS,
)
def _step4_dsa_mtp_fill_scratch_valid_rows_kernel(
    scratch_region_ids,
    scratch_row_map,
    scratch_reset_map,
    flat_slot,
    reset_slots,
    token_valid,
    token_positions,
    valid_tokens_ptr,
    total_regions: tl.constexpr,
    scratch_rows,
    region_block_size: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    live_rows = tl.load(valid_tokens_ptr).to(tl.int32)
    if row >= live_rows:
        return
    region = tl.load(flat_slot + row)
    valid = tl.load(token_valid + row).to(tl.int1)
    region_ok = (region >= 0) & (region < total_regions)
    if ~(valid & region_ok):
        return

    region_i32 = region.to(tl.int32)
    found_slot = scratch_rows
    start_slot = region_i32 % scratch_rows
    probe = 0
    while (probe < scratch_rows) & (found_slot == scratch_rows):
        slot = (start_slot + probe) % scratch_rows
        old = tl.atomic_cas(
            scratch_region_ids + slot,
            -1,
            region_i32,
            sem="relaxed",
        )
        found_slot = tl.where(
            (old == -1) | (old == region_i32),
            slot,
            found_slot,
        )
        probe += 1

    if found_slot == scratch_rows:
        return

    token_position = tl.load(token_positions + row)
    offset = token_position - (token_position // region_block_size) * region_block_size
    offset_ok = (offset >= 0) & (offset < region_block_size)
    if offset_ok:
        tl.store(scratch_row_map + found_slot * region_block_size + offset, row)
    reset = tl.load(reset_slots + row)
    if reset == region:
        tl.store(scratch_reset_map + found_slot, 1)


_MTP_INVALIDATE_DYNAMIC_ARGS = (
    "num_block_ids",
    "blocks_per_scheduler_block",
)


@triton.jit(
    do_not_specialize=_MTP_INVALIDATE_DYNAMIC_ARGS,
    do_not_specialize_on_alignment=(
        "block_ids",
        "row_regions",
        "row_positions",
        "row_source",
        "row_owner_block",
        "row_owner_block_index",
        "correction_action",
        "num_block_ids",
        "blocks_per_scheduler_block",
    ),
)
def _step4_invalidate_mtp_transaction_for_reset_blocks_kernel(
    block_ids,
    row_regions,
    row_positions,
    row_source,
    row_owner_block,
    row_owner_block_index,
    correction_action,
    num_block_ids,
    summaries_per_page: tl.constexpr,
    blocks_per_scheduler_block,
    ACTION_NONE: tl.constexpr,
    BLOCK_IDS: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    region = tl.load(row_regions + row).to(tl.int64)
    valid = region >= 0
    page = region // summaries_per_page
    scheduler_block = page // blocks_per_scheduler_block
    offsets = chunk * BLOCK_IDS + tl.arange(0, BLOCK_IDS)
    in_range = offsets < num_block_ids
    # Masked loads still form the pointer; clamp offsets before pointer
    # arithmetic and let in_range decide whether the value is used.
    safe_offsets = tl.minimum(offsets, num_block_ids - 1)
    ids = tl.load(block_ids + safe_offsets, mask=in_range, other=-1).to(tl.int64)
    reset = valid & (tl.max((ids == scheduler_block).to(tl.int32), axis=0) != 0)
    if reset:
        tl.store(row_regions + row, -1)
        tl.store(row_positions + row, -1)
        tl.store(row_source + row, -1)
        tl.store(row_owner_block + row, -1)
        tl.store(row_owner_block_index + row, -1)
        tl.store(correction_action + row, ACTION_NONE)


# Ubatch metadata and model outputs can be offset views. Their runtime pointer
# alignment must not create post-seal Triton variants.
_MTP_ORDERED_ALIGNMENT_DYNAMIC_ARGS = (
    "mean_cache",
    "active_region_ids",
    "active_slot_by_region",
    "allocation_success",
    "active_numerator",
    "denominator",
    "max_logits",
    "active_token_k",
    "active_token_z",
    "active_token_valid",
    "scratch_region_ids",
    "scratch_row_map",
    "scratch_reset_map",
    "index_k",
    "index_z",
    "source_to_transaction",
    "transaction_row_source",
    "transaction_row_regions",
    "transaction_numerator",
    "transaction_denominator",
    "transaction_max_logits",
    "transaction_pre_numerator",
    "transaction_pre_denominator",
    "transaction_pre_max_logits",
)


@triton.jit(
    do_not_specialize=("valid_tokens_ptr",),
    do_not_specialize_on_alignment=(
        ("valid_tokens_ptr",) + _MTP_ORDERED_ALIGNMENT_DYNAMIC_ARGS
    ),
)
def _step4_dsa_mtp_csa_compact_update_ordered_kernel(
    mean_cache,
    active_region_ids,
    active_slot_by_region,
    active_numerator,
    denominator,
    max_logits,
    active_token_k,
    active_token_z,
    active_token_valid,
    scratch_region_ids,
    scratch_row_map,
    scratch_reset_map,
    index_k,
    index_z,
    source_to_transaction,
    transaction_row_source,
    transaction_row_regions,
    transaction_numerator,
    transaction_denominator,
    transaction_max_logits,
    transaction_pre_numerator,
    transaction_pre_denominator,
    transaction_pre_max_logits,
    valid_tokens_ptr,
    total_regions: tl.constexpr,
    summaries_per_page: tl.constexpr,
    num_kv_heads: tl.constexpr,
    proxy_dim: tl.constexpr,
    region_block_size: tl.constexpr,
    active_capacity: tl.constexpr,
    mean_stride_page: tl.constexpr,
    denominator_stride_slot: tl.constexpr,
    denominator_stride_head: tl.constexpr,
    numerator_stride_slot: tl.constexpr,
    numerator_stride_head: tl.constexpr,
    max_stride_slot: tl.constexpr,
    stage_k_stride_slot: tl.constexpr,
    stage_k_stride_token: tl.constexpr,
    stage_k_stride_head: tl.constexpr,
    stage_z_stride_slot: tl.constexpr,
    stage_z_stride_token: tl.constexpr,
    stage_z_stride_head: tl.constexpr,
    stage_valid_stride_slot: tl.constexpr,
    use_active_slot_map: tl.constexpr,
    maintain_slot_map: tl.constexpr,
    transaction_num_rows: tl.constexpr,
    PROCESS_COMPLETE: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """MTP-only compact update that records post-token N/D/M states."""
    group_slot = tl.program_id(0)
    head = tl.program_id(1)
    live_rows = tl.load(valid_tokens_ptr).to(tl.int32)
    region_i32 = tl.load(scratch_region_ids + group_slot)
    if region_i32 < 0:
        return
    region = region_i32.to(tl.int64)
    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < proxy_dim

    # A valid scratch region is published only after fill owns a slot and
    # writes at least one row. The ordered launch runs after fill, so scanning
    # the whole row map again just to rediscover has_row is redundant.
    complete_row = tl.load(
        scratch_row_map + group_slot * region_block_size + region_block_size - 1
    )
    complete = complete_row >= 0
    if complete != PROCESS_COMPLETE:
        return

    reset = tl.load(scratch_reset_map + group_slot) != 0
    found_slot = tl.full((), active_capacity, dtype=tl.int64)
    if use_active_slot_map:
        mapped_slot_i32 = tl.load(active_slot_by_region + region)
        mapped_slot = mapped_slot_i32.to(tl.int64)
        mapped_ok = (mapped_slot >= 0) & (mapped_slot < active_capacity)
        # Triton masked load computes the pointer regardless of the mask.
        # mapped_slot can be -1 (region unowned) or a stale value beyond
        # active_capacity; either drops the pointer outside active_region_ids.
        safe_mapped_slot = tl.where(mapped_ok, mapped_slot, 0)
        mapped_region = tl.load(
            active_region_ids + safe_mapped_slot,
            mask=mapped_ok,
            other=-1,
        )
        found_slot = tl.where(
            mapped_region == region,
            mapped_slot,
            active_capacity,
        )
    else:
        for slot in tl.range(0, active_capacity):
            active_region = tl.load(active_region_ids + slot)
            if (active_region == region) & (found_slot == active_capacity):
                found_slot = slot.to(tl.int64)

    page = region // summaries_per_page
    fragment = region - page * summaries_per_page
    old_denominator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    numerator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    old_max = -float("inf")
    if found_slot != active_capacity:
        denominator_offsets = (
            found_slot * denominator_stride_slot
            + head * denominator_stride_head
            + dim_offsets
        )
        numerator_offsets = (
            found_slot * numerator_stride_slot
            + head * numerator_stride_head
            + dim_offsets
        )
        max_offset = found_slot * max_stride_slot + head
        old_denominator = tl.load(
            denominator + denominator_offsets,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        numerator = tl.load(
            active_numerator + numerator_offsets,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        old_max = tl.load(max_logits + max_offset).to(tl.float32)

    old_denominator = tl.where(
        reset,
        tl.zeros((BLOCK_D,), dtype=tl.float32),
        old_denominator,
    )
    numerator = tl.where(
        reset,
        tl.zeros((BLOCK_D,), dtype=tl.float32),
        numerator,
    )
    old_max = tl.where(reset, -float("inf"), old_max)
    denominator_acc = old_denominator
    max_acc = old_max

    stage_slot_present = found_slot < active_capacity
    stage_slot = tl.where(stage_slot_present, found_slot, 0)
    # A reset starts a fresh region state. Do not consume its old staged tail,
    # but still invalidate the tail below before the slot can be reused.
    stage_slot_valid = stage_slot_present & ~reset
    seen_current_row = tl.full((), False, dtype=tl.int1)
    for offset in tl.range(0, region_block_size):
        row = tl.load(scratch_row_map + group_slot * region_block_size + offset)
        valid_row = (row >= 0) & (row < live_rows)
        # Same masked-load pointer-safety rule: clamp row before it enters
        # value_offsets; valid_row still discards the loaded value.
        safe_row = tl.where(valid_row, row, 0)
        value_offsets = (safe_row * num_kv_heads + head) * proxy_dim + dim_offsets
        staged_valid = tl.load(
            active_token_valid + stage_slot * stage_valid_stride_slot + offset,
            mask=stage_slot_valid,
            other=0,
        ).to(tl.int1)
        use_staged = stage_slot_valid & staged_valid & ~seen_current_row & ~valid_row
        staged_offsets = (
            stage_slot * stage_k_stride_slot
            + offset * stage_k_stride_token
            + head * stage_k_stride_head
            + dim_offsets
        )
        staged_z_offsets = (
            stage_slot * stage_z_stride_slot
            + offset * stage_z_stride_token
            + head * stage_z_stride_head
            + dim_offsets
        )
        staged_values = tl.load(
            active_token_k + staged_offsets,
            mask=dim_mask & use_staged,
            other=0.0,
        ).to(tl.float32)
        staged_logits = tl.load(
            active_token_z + staged_z_offsets,
            mask=dim_mask & use_staged,
            other=-float("inf"),
        ).to(tl.float32)
        staged_row_max = tl.max(staged_logits, axis=0)
        staged_new_max = tl.maximum(max_acc, staged_row_max)
        staged_old_scale = tl.where(
            denominator_acc > 0.0,
            tl.exp(max_acc - staged_new_max),
            0.0,
        )
        staged_weights = tl.where(
            use_staged,
            tl.exp(staged_logits - staged_new_max),
            0.0,
        )
        numerator = numerator * staged_old_scale + staged_weights * staged_values
        denominator_acc = denominator_acc * staged_old_scale + staged_weights
        max_acc = tl.where(use_staged, staged_new_max, max_acc)
        values = tl.load(
            index_k + value_offsets,
            mask=dim_mask & valid_row,
            other=0.0,
        ).to(tl.float32)
        logits = tl.load(
            index_z + value_offsets,
            mask=dim_mask & valid_row,
            other=-float("inf"),
        ).to(tl.float32)
        # Resolve the transaction row before mutating the online-softmax
        # state. The pre-token state is identical for every verifier row in
        # this region and is needed when the whole verifier proposal rejects.
        transaction_row_i32 = tl.load(
            source_to_transaction + safe_row,
            mask=valid_row,
            other=-1,
        )
        transaction_row = transaction_row_i32.to(tl.int64)
        transaction_in_range = (transaction_row >= 0) & (
            transaction_row < transaction_num_rows
        )
        safe_transaction_row = tl.where(transaction_in_range, transaction_row, 0)
        transaction_matches = (
            valid_row
            & transaction_in_range
            & (
                tl.load(
                    transaction_row_source + safe_transaction_row,
                    mask=transaction_in_range,
                    other=-1,
                )
                == safe_row
            )
            & (
                tl.load(
                    transaction_row_regions + safe_transaction_row,
                    mask=transaction_in_range,
                    other=-1,
                )
                == region
            )
        )
        transaction_offsets = (
            safe_transaction_row * num_kv_heads + head
        ) * proxy_dim + dim_offsets
        tl.store(
            transaction_pre_numerator + transaction_offsets,
            numerator,
            mask=transaction_matches & dim_mask,
        )
        tl.store(
            transaction_pre_denominator + transaction_offsets,
            denominator_acc,
            mask=transaction_matches & dim_mask,
        )
        tl.store(
            transaction_pre_max_logits + safe_transaction_row * num_kv_heads + head,
            max_acc,
            mask=transaction_matches,
        )

        row_max = tl.max(logits, axis=0)
        new_max = tl.maximum(max_acc, row_max)
        old_scale = tl.where(
            denominator_acc > 0.0,
            tl.exp(max_acc - new_max),
            0.0,
        )
        weights = tl.where(valid_row, tl.exp(logits - new_max), 0.0)
        numerator = numerator * old_scale + weights * values
        denominator_acc = denominator_acc * old_scale + weights
        max_acc = new_max

        # The verifier transaction also records the online-softmax state after
        # each token: S1, S2, ... Sq for accepted-prefix installation.
        tl.store(
            transaction_numerator + transaction_offsets,
            numerator,
            mask=transaction_matches & dim_mask,
        )
        tl.store(
            transaction_denominator + transaction_offsets,
            denominator_acc,
            mask=transaction_matches & dim_mask,
        )
        tl.store(
            transaction_max_logits + safe_transaction_row * num_kv_heads + head,
            max_acc,
            mask=transaction_matches,
        )
        seen_current_row |= valid_row

    stage_clear_offsets = tl.minimum(dim_offsets, region_block_size - 1)
    tl.store(
        active_token_valid + stage_slot * stage_valid_stride_slot + stage_clear_offsets,
        False,
        mask=(stage_slot_present & (head == 0) & (dim_offsets < region_block_size)),
    )

    materialized_out = numerator / tl.maximum(denominator_acc, 1.0e-20)
    materialized_out = tl.where(
        denominator_acc > 0.0,
        materialized_out,
        0.0,
    )
    atom = dim_offsets // 128
    atom_dim = dim_offsets - atom * 128
    chunk = atom_dim // 16
    byte = atom_dim - chunk * 16
    row_in_block = fragment - (fragment // 8) * 8
    block_in_page = fragment // 8
    row_swizzle = region % 8
    mean_offsets = (
        page * mean_stride_page
        + block_in_page * 2048
        + atom * 1024
        + row_in_block * 128
        + ((chunk ^ row_swizzle) * 16)
        + byte
    )
    mean_fp8 = tl.cast(materialized_out, tl.float8e4nv)
    tl.store(
        mean_cache + mean_offsets,
        tl.cast(mean_fp8, tl.uint8, bitcast=True),
        mask=dim_mask,
    )

    if complete:
        if found_slot != active_capacity:
            denominator_offsets = (
                found_slot * denominator_stride_slot
                + head * denominator_stride_head
                + dim_offsets
            )
            numerator_offsets = (
                found_slot * numerator_stride_slot
                + head * numerator_stride_head
                + dim_offsets
            )
            max_offset = found_slot * max_stride_slot + head
            tl.store(
                active_numerator + numerator_offsets,
                tl.zeros((BLOCK_D,), dtype=tl.float32),
                mask=dim_mask,
            )
            tl.store(
                denominator + denominator_offsets,
                tl.zeros((BLOCK_D,), dtype=tl.float32),
                mask=dim_mask,
            )
            tl.store(max_logits + max_offset, -float("inf"))
        # Keep the logical slot occupied while incomplete regions allocate.
        # The MTP driver clears completed owners after both ordered passes so
        # a saved verifier state cannot be silently overwritten in this update.
        return

    slot_to_store = found_slot.to(tl.int64)
    if slot_to_store != active_capacity:
        if maintain_slot_map:
            tl.store(
                active_slot_by_region + region,
                slot_to_store.to(tl.int32),
            )
        denominator_offsets = (
            slot_to_store * denominator_stride_slot
            + head * denominator_stride_head
            + dim_offsets
        )
        numerator_offsets = (
            slot_to_store * numerator_stride_slot
            + head * numerator_stride_head
            + dim_offsets
        )
        max_offset = slot_to_store * max_stride_slot + head
        tl.store(
            active_numerator + numerator_offsets,
            numerator,
            mask=dim_mask,
        )
        tl.store(
            denominator + denominator_offsets,
            denominator_acc,
            mask=dim_mask,
        )
        tl.store(max_logits + max_offset, max_acc)


_MTP_UPDATE_DYNAMIC_ARGS = (
    "num_verifier_requests",
    "source_map_rows",
    "block_table_stride_request",
    "block_table_stride_column",
    "block_table_cols",
    "valid_tokens_ptr",
)
_MTP_UPDATE_ALIGNMENT_DYNAMIC_ARGS = (
    "query_start_loc",
    "seq_lens",
    "block_table",
    "layout_regions",
    "layout_positions",
    "layout_valid",
    "source_to_transaction",
    "row_source",
    "row_regions",
    "row_positions",
    "row_owner_block",
    "row_owner_block_index",
    "correction_action",
    "valid_requests_ptr",
    "valid_tokens_ptr",
)


@triton.jit(
    do_not_specialize=_MTP_UPDATE_DYNAMIC_ARGS,
    do_not_specialize_on_alignment=(
        _MTP_UPDATE_DYNAMIC_ARGS + _MTP_UPDATE_ALIGNMENT_DYNAMIC_ARGS
    ),
)
def _step4_prepare_mtp_update_transaction_kernel(
    query_start_loc,
    seq_lens,
    block_table,
    layout_regions,
    layout_positions,
    layout_valid,
    source_to_transaction,
    row_source,
    row_regions,
    row_positions,
    row_owner_block,
    row_owner_block_index,
    correction_action,
    valid_requests_ptr,
    valid_tokens_ptr,
    num_verifier_requests,
    source_map_rows,
    block_table_stride_request,
    block_table_stride_column,
    block_table_cols,
    page_size: tl.constexpr,
    total_regions: tl.constexpr,
    max_rows_per_req: tl.constexpr,
    transaction_num_rows: tl.constexpr,
    ACTION_NONE: tl.constexpr,
):
    """Publish verifier row ownership before the compact N/D/M update."""
    request = tl.program_id(0)
    row_offset = tl.program_id(1)
    transaction_row = request * max_rows_per_req + row_offset
    transaction_in_range = transaction_row < transaction_num_rows

    # Clear only the rows owned by this verifier prefix. The correction phase
    # invalidates the full fixed-capacity transaction after it consumes the
    # previous step; this local clear also handles q1/padded shape changes.
    tl.store(row_source + transaction_row, -1, mask=transaction_in_range)
    tl.store(row_regions + transaction_row, -1, mask=transaction_in_range)
    tl.store(row_positions + transaction_row, -1, mask=transaction_in_range)
    tl.store(row_owner_block + transaction_row, -1, mask=transaction_in_range)
    tl.store(
        row_owner_block_index + transaction_row,
        -1,
        mask=transaction_in_range,
    )
    tl.store(
        correction_action + transaction_row,
        ACTION_NONE,
        mask=transaction_in_range,
    )

    live_requests = tl.load(valid_requests_ptr).to(tl.int32)
    live_tokens = tl.load(valid_tokens_ptr).to(tl.int32)
    request_valid = request < tl.minimum(num_verifier_requests, live_requests)
    request_start = tl.load(
        query_start_loc + request,
        mask=request_valid,
        other=0,
    ).to(tl.int64)
    request_end = tl.load(
        query_start_loc + request + 1,
        mask=request_valid,
        other=0,
    ).to(tl.int64)
    request_rows = request_end - request_start
    source_row = request_start + row_offset
    source_in_range = (
        request_valid
        & transaction_in_range
        & (request_rows > 1)
        & (row_offset < request_rows)
        & (source_row >= 0)
        & (source_row < live_tokens)
        & (source_row < source_map_rows)
    )
    safe_source_row = tl.where(source_in_range, source_row, 0)
    valid = source_in_range & (
        tl.load(
            layout_valid + safe_source_row,
            mask=source_in_range,
            other=0,
        )
        != 0
    )
    region = tl.load(
        layout_regions + safe_source_row,
        mask=valid,
        other=-1,
    ).to(tl.int64)
    position = tl.load(
        layout_positions + safe_source_row,
        mask=valid,
        other=-1,
    ).to(tl.int64)
    valid &= (region >= 0) & (region < total_regions)

    # The fingerprint is the last already-committed context block. It is past
    # the shared prefix in normal serving batches, so correction can distinguish
    # a recycled request slot without a new host-side metadata field.
    context_end = (
        tl.load(
            seq_lens + request,
            mask=request_valid,
            other=0,
        ).to(tl.int64)
        - request_rows
    )
    owner_index_signed = (context_end - 1) // page_size
    owner_index = tl.maximum(owner_index_signed, 0)
    owner_index_in_range = (owner_index >= 0) & (owner_index < block_table_cols)
    safe_owner_index = tl.where(owner_index_in_range, owner_index, 0)
    owner_block = tl.load(
        block_table
        + request * block_table_stride_request
        + safe_owner_index * block_table_stride_column,
        mask=request_valid & owner_index_in_range,
        other=-1,
    ).to(tl.int64)
    fingerprint_valid = valid & owner_index_in_range & (context_end > 0)

    tl.store(row_source + transaction_row, safe_source_row, mask=valid)
    tl.store(row_regions + transaction_row, region, mask=valid)
    tl.store(row_positions + transaction_row, position, mask=valid)
    tl.store(
        row_owner_block + transaction_row,
        owner_block,
        mask=fingerprint_valid,
    )
    tl.store(
        row_owner_block_index + transaction_row,
        owner_index,
        mask=fingerprint_valid,
    )
    tl.store(
        source_to_transaction + safe_source_row,
        transaction_row.to(tl.int32),
        mask=valid,
    )


def _step4_dsa_mtp_csa_compact_update_impl(
    mean_cache: torch.Tensor,
    active_region_ids: torch.Tensor,
    active_slot_by_region: torch.Tensor,
    allocation_success: torch.Tensor,
    active_numerator: torch.Tensor,
    denominator: torch.Tensor,
    max_logits: torch.Tensor,
    active_token_k: torch.Tensor,
    active_token_z: torch.Tensor,
    active_token_valid: torch.Tensor,
    flat_slot: torch.Tensor,
    reset_slots: torch.Tensor,
    token_valid: torch.Tensor,
    token_positions: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    source_to_transaction: torch.Tensor,
    transaction_row_source: torch.Tensor,
    transaction_row_regions: torch.Tensor,
    transaction_row_positions: torch.Tensor,
    transaction_row_owner_block: torch.Tensor,
    transaction_row_owner_block_index: torch.Tensor,
    transaction_correction_action: torch.Tensor,
    valid_requests: torch.Tensor,
    transaction_numerator: torch.Tensor,
    transaction_denominator: torch.Tensor,
    transaction_max_logits: torch.Tensor,
    transaction_pre_numerator: torch.Tensor,
    transaction_pre_denominator: torch.Tensor,
    transaction_pre_max_logits: torch.Tensor,
    region_block_size: int,
    page_size: int,
    max_rows_per_req: int,
    num_verifier_requests: int,
    scratch_region_ids: torch.Tensor,
    scratch_row_map: torch.Tensor,
    scratch_reset_map: torch.Tensor,
    valid_tokens: torch.Tensor,
    allocation_free_slots: torch.Tensor,
    allocation_free_count: torch.Tensor,
    allocation_cursor: torch.Tensor,
) -> None:
    layout_rows = int(flat_slot.numel())
    region_block_size = int(region_block_size)
    proxy_dim = int(active_numerator.shape[2])
    total_regions = int(mean_cache.shape[0] * mean_cache.shape[1])
    num_kv_heads = int(active_numerator.shape[1])
    active_capacity = int(active_region_ids.numel())
    if (
        allocation_success.dtype != torch.int32
        or allocation_success.device != active_region_ids.device
        or tuple(allocation_success.shape) != (1,)
        or not allocation_success.is_contiguous()
    ):
        raise RuntimeError(
            "Step4 DSA MTP requires contiguous int32 "
            "allocation_success=[1] on the active-state device."
        )
    use_active_slot_map = num_kv_heads == 1
    transaction_num_rows = int(transaction_row_regions.numel())
    _step4_prepare_mtp_update_transaction_kernel[
        (int(num_verifier_requests), int(max_rows_per_req))
    ](
        query_start_loc,
        seq_lens,
        block_table,
        flat_slot,
        token_positions,
        token_valid,
        source_to_transaction,
        transaction_row_source,
        transaction_row_regions,
        transaction_row_positions,
        transaction_row_owner_block,
        transaction_row_owner_block_index,
        transaction_correction_action,
        valid_requests,
        valid_tokens,
        int(num_verifier_requests),
        int(source_to_transaction.numel()),
        int(block_table.stride(0)),
        int(block_table.stride(1)),
        int(block_table.shape[1]),
        page_size=int(page_size),
        total_regions=total_regions,
        max_rows_per_req=int(max_rows_per_req),
        transaction_num_rows=transaction_num_rows,
        ACTION_NONE=_MTP_ACTION_NONE,
    )
    scratch_total = layout_rows * region_block_size
    clear_block = 256
    _step4_sparse_csa_compact_clear_scratch_kernel[
        (triton.cdiv(max(scratch_total, layout_rows), clear_block),)
    ](
        scratch_region_ids,
        scratch_row_map,
        scratch_reset_map,
        layout_rows,
        region_block_size,
        max(scratch_total, layout_rows),
        BLOCK_N=clear_block,
    )
    _step4_dsa_mtp_fill_scratch_valid_rows_kernel[(layout_rows,)](
        scratch_region_ids,
        scratch_row_map,
        scratch_reset_map,
        flat_slot,
        reset_slots,
        token_valid,
        token_positions,
        valid_tokens,
        total_regions,
        layout_rows,
        region_block_size,
    )
    free_slot_block = 256
    _step4_reset_mtp_correction_allocator_kernel[(1,)](
        allocation_free_count,
        allocation_cursor,
    )
    _step4_build_mtp_correction_free_slots_kernel[
        (triton.cdiv(active_capacity, free_slot_block),)
    ](
        active_region_ids,
        active_slot_by_region,
        allocation_free_slots,
        allocation_free_count,
        active_capacity,
        total_regions,
        BLOCK_CAPACITY=free_slot_block,
    )
    _step4_allocate_mtp_update_slots_kernel[(layout_rows,)](
        active_region_ids,
        active_slot_by_region,
        allocation_success,
        active_token_valid,
        scratch_region_ids,
        scratch_row_map,
        scratch_reset_map,
        allocation_free_slots,
        allocation_free_count,
        allocation_cursor,
        region_block_size=region_block_size,
        active_capacity=active_capacity,
        active_token_valid_stride_slot=int(active_token_valid.stride(0)),
    )
    for process_complete in (True, False):
        _step4_dsa_mtp_csa_compact_update_ordered_kernel[(layout_rows, num_kv_heads)](
            mean_cache,
            active_region_ids,
            active_slot_by_region,
            active_numerator,
            denominator,
            max_logits,
            active_token_k,
            active_token_z,
            active_token_valid,
            scratch_region_ids,
            scratch_row_map,
            scratch_reset_map,
            index_k,
            index_z,
            source_to_transaction,
            transaction_row_source,
            transaction_row_regions,
            transaction_numerator,
            transaction_denominator,
            transaction_max_logits,
            transaction_pre_numerator,
            transaction_pre_denominator,
            transaction_pre_max_logits,
            valid_tokens,
            total_regions,
            int(mean_cache.shape[1]),
            num_kv_heads,
            proxy_dim,
            region_block_size,
            active_capacity,
            int(mean_cache.stride(0)),
            int(denominator.stride(0)),
            int(denominator.stride(1)),
            int(active_numerator.stride(0)),
            int(active_numerator.stride(1)),
            int(max_logits.stride(0)),
            int(active_token_k.stride(0)),
            int(active_token_k.stride(1)),
            int(active_token_k.stride(2)),
            int(active_token_z.stride(0)),
            int(active_token_z.stride(1)),
            int(active_token_z.stride(2)),
            int(active_token_valid.stride(0)),
            use_active_slot_map=use_active_slot_map,
            maintain_slot_map=True,
            transaction_num_rows=transaction_num_rows,
            PROCESS_COMPLETE=process_complete,
            BLOCK_D=triton.next_power_of_2(proxy_dim),
        )
    _step4_sparse_csa_compact_clear_completed_kernel[(layout_rows,)](
        active_region_ids,
        active_slot_by_region,
        scratch_region_ids,
        scratch_row_map,
        region_block_size,
        active_capacity,
        USE_SLOT_MAP=use_active_slot_map,
        MAINTAIN_SLOT_MAP=True,
    )


def _step4_dsa_mtp_csa_update_op_fake(
    mean_cache: torch.Tensor,
    active_region_ids: torch.Tensor,
    active_slot_by_region: torch.Tensor,
    allocation_success: torch.Tensor,
    active_numerator: torch.Tensor,
    denominator: torch.Tensor,
    max_logits: torch.Tensor,
    active_token_k: torch.Tensor,
    active_token_z: torch.Tensor,
    active_token_valid: torch.Tensor,
    flat_slot: torch.Tensor,
    reset_slots: torch.Tensor,
    token_valid: torch.Tensor,
    token_positions: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    source_to_transaction: torch.Tensor,
    transaction_row_source: torch.Tensor,
    transaction_row_regions: torch.Tensor,
    transaction_row_positions: torch.Tensor,
    transaction_row_owner_block: torch.Tensor,
    transaction_row_owner_block_index: torch.Tensor,
    transaction_correction_action: torch.Tensor,
    valid_requests: torch.Tensor,
    transaction_numerator: torch.Tensor,
    transaction_denominator: torch.Tensor,
    transaction_max_logits: torch.Tensor,
    transaction_pre_numerator: torch.Tensor,
    transaction_pre_denominator: torch.Tensor,
    transaction_pre_max_logits: torch.Tensor,
    region_block_size: int,
    page_size: int,
    max_rows_per_req: int,
    num_verifier_requests: int,
    scratch_region_ids: torch.Tensor,
    scratch_row_map: torch.Tensor,
    scratch_reset_map: torch.Tensor,
    valid_tokens: torch.Tensor,
    allocation_free_slots: torch.Tensor,
    allocation_free_count: torch.Tensor,
    allocation_cursor: torch.Tensor,
) -> None:
    return None


def _step4_dsa_mtp_csa_update_op_impl(
    mean_cache: torch.Tensor,
    active_region_ids: torch.Tensor,
    active_slot_by_region: torch.Tensor,
    allocation_success: torch.Tensor,
    active_numerator: torch.Tensor,
    denominator: torch.Tensor,
    max_logits: torch.Tensor,
    active_token_k: torch.Tensor,
    active_token_z: torch.Tensor,
    active_token_valid: torch.Tensor,
    flat_slot: torch.Tensor,
    reset_slots: torch.Tensor,
    token_valid: torch.Tensor,
    token_positions: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    source_to_transaction: torch.Tensor,
    transaction_row_source: torch.Tensor,
    transaction_row_regions: torch.Tensor,
    transaction_row_positions: torch.Tensor,
    transaction_row_owner_block: torch.Tensor,
    transaction_row_owner_block_index: torch.Tensor,
    transaction_correction_action: torch.Tensor,
    valid_requests: torch.Tensor,
    transaction_numerator: torch.Tensor,
    transaction_denominator: torch.Tensor,
    transaction_max_logits: torch.Tensor,
    transaction_pre_numerator: torch.Tensor,
    transaction_pre_denominator: torch.Tensor,
    transaction_pre_max_logits: torch.Tensor,
    region_block_size: int,
    page_size: int,
    max_rows_per_req: int,
    num_verifier_requests: int,
    scratch_region_ids: torch.Tensor,
    scratch_row_map: torch.Tensor,
    scratch_reset_map: torch.Tensor,
    valid_tokens: torch.Tensor,
    allocation_free_slots: torch.Tensor,
    allocation_free_count: torch.Tensor,
    allocation_cursor: torch.Tensor,
) -> None:
    _step4_dsa_mtp_csa_compact_update_impl(
        mean_cache,
        active_region_ids,
        active_slot_by_region,
        allocation_success,
        active_numerator,
        denominator,
        max_logits,
        active_token_k,
        active_token_z,
        active_token_valid,
        flat_slot,
        reset_slots,
        token_valid,
        token_positions,
        index_k,
        index_z,
        query_start_loc,
        seq_lens,
        block_table,
        source_to_transaction,
        transaction_row_source,
        transaction_row_regions,
        transaction_row_positions,
        transaction_row_owner_block,
        transaction_row_owner_block_index,
        transaction_correction_action,
        valid_requests,
        transaction_numerator,
        transaction_denominator,
        transaction_max_logits,
        transaction_pre_numerator,
        transaction_pre_denominator,
        transaction_pre_max_logits,
        region_block_size,
        page_size,
        max_rows_per_req,
        num_verifier_requests,
        scratch_region_ids,
        scratch_row_map,
        scratch_reset_map,
        valid_tokens=valid_tokens,
        allocation_free_slots=allocation_free_slots,
        allocation_free_count=allocation_free_count,
        allocation_cursor=allocation_cursor,
    )


direct_register_custom_op(
    op_name="step4_dsa_mtp_csa_update",
    op_func=_step4_dsa_mtp_csa_update_op_impl,
    mutates_args=[
        "mean_cache",
        "active_region_ids",
        "active_slot_by_region",
        "allocation_success",
        "active_numerator",
        "denominator",
        "max_logits",
        "active_token_valid",
        "source_to_transaction",
        "transaction_row_source",
        "transaction_row_regions",
        "transaction_row_positions",
        "transaction_row_owner_block",
        "transaction_row_owner_block_index",
        "transaction_correction_action",
        "transaction_numerator",
        "transaction_denominator",
        "transaction_max_logits",
        "transaction_pre_numerator",
        "transaction_pre_denominator",
        "transaction_pre_max_logits",
        "scratch_region_ids",
        "scratch_row_map",
        "scratch_reset_map",
        "allocation_free_slots",
        "allocation_free_count",
        "allocation_cursor",
    ],
    fake_impl=_step4_dsa_mtp_csa_update_op_fake,
)


_MTP_CORRECTION_DYNAMIC_ARGS = (
    "block_table_cols",
    "block_table_stride_request",
    "block_table_stride_column",
)
_MTP_CORRECTION_ALIGNMENT_DYNAMIC_ARGS = (
    "query_start_loc",
    "seq_lens",
    "block_table",
    "row_regions",
    "row_positions",
    "row_owner_block",
    "row_owner_block_index",
    "correction_action",
    "valid_requests_ptr",
)


@triton.jit(
    do_not_specialize=_MTP_CORRECTION_DYNAMIC_ARGS,
    do_not_specialize_on_alignment=(
        _MTP_CORRECTION_DYNAMIC_ARGS + _MTP_CORRECTION_ALIGNMENT_DYNAMIC_ARGS
    ),
)
def _step4_prepare_mtp_correction_kernel(
    query_start_loc,
    seq_lens,
    block_table,
    row_regions,
    row_positions,
    row_owner_block,
    row_owner_block_index,
    correction_action,
    valid_requests_ptr,
    block_table_cols,
    block_table_stride_request,
    block_table_stride_column,
    page_size: tl.constexpr,
    summaries_per_page: tl.constexpr,
    max_rows_per_req: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    ACTION_NONE: tl.constexpr,
    ACTION_CLEAR: tl.constexpr,
    ACTION_INSTALL: tl.constexpr,
    ACTION_INSTALL_PRE: tl.constexpr,
):
    """Resolve old transaction owners against the current main-attention batch."""
    live_requests = tl.load(valid_requests_ptr).to(tl.int32)
    old_request = tl.program_id(0)
    transaction_base = old_request * max_rows_per_req
    row_offsets = tl.arange(0, BLOCK_ROWS)
    row_mask = row_offsets < max_rows_per_req
    transaction_rows = transaction_base + row_offsets
    regions = tl.load(row_regions + transaction_rows, mask=row_mask, other=-1).to(
        tl.int64
    )
    valid = row_mask & (regions >= 0)
    positions = tl.load(
        row_positions + transaction_rows,
        mask=valid,
        other=-1,
    ).to(tl.int64)
    row_owner = tl.load(
        row_owner_block + transaction_rows,
        mask=valid,
        other=-1,
    ).to(tl.int64)
    row_owner_index = tl.load(
        row_owner_block_index + transaction_rows,
        mask=valid,
        other=-1,
    ).to(tl.int64)
    row_owner_valid = (
        valid & (row_owner_index >= 0) & (row_owner_index < block_table_cols)
    )
    safe_row_owner_index = tl.where(row_owner_valid, row_owner_index, 0)
    has_rows = tl.max(valid.to(tl.int32), axis=0) != 0

    current_owner = tl.full((), -1, tl.int32)
    for candidate in tl.range(0, live_requests):
        # Owner fingerprint at the stored column (in the request's private
        # context region, past any shared prefix). If the slot has been
        # reassigned to a different request, this column holds a different
        # physical block for the new occupant. Kernel reads one column per
        # transaction row so different rows can, in principle, reference
        # different columns; in practice a slot's rows share the same column.
        candidate_owner = tl.load(
            block_table
            + candidate * block_table_stride_request
            + safe_row_owner_index * block_table_stride_column,
            mask=row_owner_valid,
            other=-1,
        ).to(tl.int64)
        block_positions = positions // page_size
        block_in_range = (block_positions >= 0) & (block_positions < block_table_cols)
        # Triton masked load computes the pointer regardless of the mask, so
        # long-sequence positions (100k+ tokens) can push block_positions past
        # block_table_cols and read outside the tensor. Clamp before pointer
        # arithmetic; block_in_range still discards the value.
        safe_block_positions = tl.minimum(
            tl.maximum(block_positions, 0), block_table_cols - 1
        )
        blocks = tl.load(
            block_table
            + candidate * block_table_stride_request
            + safe_block_positions * block_table_stride_column,
            mask=valid & has_rows & block_in_range,
            other=-1,
        ).to(tl.int64)
        # A stale row can claim a new candidate only if BOTH the content page
        # at its captured position AND the ownership fingerprint at its
        # captured private column still match. Prefix-cache sharing can force
        # column 0 to collide, but the private column past the shared prefix
        # will not.
        rows_match = ~valid | (
            block_in_range
            & (blocks == regions // summaries_per_page)
            & row_owner_valid
            & (candidate_owner >= 0)
            & (row_owner == candidate_owner)
        )
        candidate_match = has_rows & (tl.min(rows_match.to(tl.int32), axis=0) != 0)
        current_owner = tl.where(
            (current_owner < 0) & candidate_match,
            candidate,
            current_owner,
        )

    owner_found = current_owner >= 0
    safe_owner = tl.maximum(current_owner, 0)
    request_start = tl.load(
        query_start_loc + safe_owner,
        mask=owner_found,
        other=0,
    ).to(tl.int64)
    request_end = tl.load(
        query_start_loc + safe_owner + 1,
        mask=owner_found,
        other=0,
    ).to(tl.int64)
    context_boundary = tl.load(
        seq_lens + safe_owner,
        mask=owner_found,
        other=0,
    ).to(tl.int64) - (request_end - request_start)

    accepted = owner_found & valid & (positions < context_boundary)
    has_accepted = tl.max(accepted.to(tl.int32), axis=0) != 0
    has_rejected = tl.max((valid & ~accepted).to(tl.int32), axis=0) != 0
    needs_rollback = has_rows & owner_found & has_accepted & has_rejected
    all_rejected = has_rows & owner_found & ~has_accepted & has_rejected

    region_has_accepted = tl.zeros((BLOCK_ROWS,), dtype=tl.int1)
    region_has_rejected = tl.zeros((BLOCK_ROWS,), dtype=tl.int1)
    later_accepted_same_region = tl.zeros((BLOCK_ROWS,), dtype=tl.int1)
    earlier_rejected_same_region = tl.zeros((BLOCK_ROWS,), dtype=tl.int1)
    for other_offset in tl.static_range(0, max_rows_per_req):
        other_row = transaction_base + other_offset
        other_region = tl.load(row_regions + other_row).to(tl.int64)
        other_valid = other_region >= 0
        other_position = tl.load(
            row_positions + other_row,
            mask=other_valid,
            other=-1,
        ).to(tl.int64)
        other_accepted = owner_found & other_valid & (other_position < context_boundary)
        same_region = valid & other_valid & (regions == other_region)
        region_has_accepted |= same_region & other_accepted
        region_has_rejected |= same_region & ~other_accepted
        later_accepted_same_region |= (
            same_region & other_accepted & (other_position > positions)
        )
        earlier_rejected_same_region |= (
            same_region & ~other_accepted & (other_position < positions)
        )

    # CLEAR and INSTALL are two ordered phases. Rejected-only regions are
    # removed first, making their active slots available for the accepted
    # boundary state of a mixed region. A fully rejected verifier installs the
    # state from before its first token; all-accepted requests remain untouched.
    install = (
        needs_rollback
        & valid
        & accepted
        & region_has_rejected
        & ~later_accepted_same_region
    )
    clear = (
        needs_rollback
        & valid
        & ~accepted
        & ~region_has_accepted
        & ~earlier_rejected_same_region
    )
    install_pre = all_rejected & valid & ~earlier_rejected_same_region
    action = tl.where(
        install,
        ACTION_INSTALL,
        tl.where(
            install_pre,
            ACTION_INSTALL_PRE,
            tl.where(clear, ACTION_CLEAR, ACTION_NONE),
        ),
    )
    tl.store(correction_action + transaction_rows, action, mask=row_mask)


_MTP_APPLY_ALIGNMENT_DYNAMIC_ARGS = (
    "row_regions",
    "row_positions",
    "row_source",
    "row_owner_block",
    "row_owner_block_index",
    "correction_action",
    "state_numerator",
    "state_denominator",
    "state_max_logits",
    "state_pre_numerator",
    "state_pre_denominator",
    "state_pre_max_logits",
    "mean_cache",
    "active_region_ids",
    "active_slot_by_region",
    "active_numerator",
    "denominator",
    "max_logits",
    "active_token_valid",
    "correction_free_slots",
    "correction_free_count",
    "correction_allocation_cursor",
)


@triton.jit
def _step4_reset_mtp_correction_allocator_kernel(
    free_count,
    allocation_cursor,
):
    tl.store(free_count, 0)
    tl.store(allocation_cursor, 0)


@triton.jit
def _step4_build_mtp_correction_free_slots_kernel(
    active_region_ids,
    active_slot_by_region,
    free_slots,
    free_count,
    active_capacity,
    total_regions,
    BLOCK_CAPACITY: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_CAPACITY + tl.arange(0, BLOCK_CAPACITY)
    in_range = offsets < active_capacity
    is_free = in_range & (
        tl.load(active_region_ids + offsets, mask=in_range, other=0) == -1
    )
    local_ranks = tl.cumsum(is_free.to(tl.int32), axis=0) - 1
    local_count = tl.sum(is_free.to(tl.int32), axis=0)
    output_base = tl.atomic_add(free_count, local_count)
    tl.store(
        free_slots + output_base + local_ranks,
        offsets,
        mask=is_free,
    )
    owners = tl.load(active_region_ids + offsets, mask=in_range, other=-1)
    valid_owner = in_range & (owners >= 0) & (owners < total_regions)
    safe_owner = tl.where(valid_owner, owners, 0)
    tl.store(active_slot_by_region + safe_owner, offsets, mask=valid_owner)


@triton.jit
def _step4_allocate_mtp_update_slots_kernel(
    active_region_ids,
    active_slot_by_region,
    allocation_success,
    active_token_valid,
    scratch_region_ids,
    scratch_row_map,
    scratch_reset_map,
    free_slots,
    free_count,
    allocation_cursor,
    region_block_size: tl.constexpr,
    active_capacity: tl.constexpr,
    active_token_valid_stride_slot: tl.constexpr,
):
    group_slot = tl.program_id(0)
    region_i32 = tl.load(scratch_region_ids + group_slot)
    if region_i32 < 0:
        return
    complete_row = tl.load(
        scratch_row_map + group_slot * region_block_size + region_block_size - 1
    )
    if complete_row >= 0:
        return

    region = region_i32.to(tl.int64)
    mapped_slot_i32 = tl.load(active_slot_by_region + region)
    mapped_slot = mapped_slot_i32.to(tl.int64)
    mapped_ok = (mapped_slot >= 0) & (mapped_slot < active_capacity)
    safe_mapped_slot = tl.where(mapped_ok, mapped_slot, 0)
    mapped_region = tl.load(
        active_region_ids + safe_mapped_slot,
        mask=mapped_ok,
        other=-1,
    )
    if mapped_region == region:
        return

    allocation_rank = tl.atomic_add(allocation_cursor, 1)
    available = tl.load(free_count)
    has_slot = allocation_rank < available
    slot = tl.load(
        free_slots + allocation_rank,
        mask=has_slot,
        other=active_capacity,
    ).to(tl.int64)
    slot_ok = has_slot & (slot >= 0) & (slot < active_capacity)
    allocation_ok = tl.full((), False, dtype=tl.int1)
    if slot_ok:
        old = tl.atomic_cas(
            active_region_ids + slot,
            tl.full((), -1, tl.int64),
            region,
        )
        allocation_ok = (old == -1) | (old == region)
        if allocation_ok:
            tl.store(active_slot_by_region + region, slot.to(tl.int32))
            if old == -1:
                tl.store(scratch_reset_map + group_slot, 1)
                offsets = tl.arange(0, region_block_size)
                tl.store(
                    active_token_valid
                    + slot * active_token_valid_stride_slot
                    + offsets,
                    0,
                )
    if ~allocation_ok:
        tl.store(allocation_success, 0)


@triton.jit(
    do_not_specialize_on_alignment=_MTP_APPLY_ALIGNMENT_DYNAMIC_ARGS,
)
def _step4_apply_mtp_correction_kernel(
    row_regions,
    row_positions,
    row_source,
    row_owner_block,
    row_owner_block_index,
    correction_action,
    state_numerator,
    state_denominator,
    state_max_logits,
    state_pre_numerator,
    state_pre_denominator,
    state_pre_max_logits,
    mean_cache,
    active_region_ids,
    active_slot_by_region,
    allocation_success,
    active_numerator,
    denominator,
    max_logits,
    active_token_valid,
    correction_free_slots,
    correction_free_count,
    correction_allocation_cursor,
    mean_stride_page: tl.constexpr,
    total_regions: tl.constexpr,
    summaries_per_page: tl.constexpr,
    active_capacity: tl.constexpr,
    proxy_dim: tl.constexpr,
    active_token_valid_stride_slot: tl.constexpr,
    region_block_size: tl.constexpr,
    ACTION_NONE: tl.constexpr,
    ACTION_CLEAR: tl.constexpr,
    ACTION_INSTALL: tl.constexpr,
    ACTION_INSTALL_PRE: tl.constexpr,
    APPLY_INSTALL: tl.constexpr,
    BLOCK_CAPACITY: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    region = tl.load(row_regions + row).to(tl.int64)
    action = tl.load(correction_action + row)
    if APPLY_INSTALL:  # noqa: SIM108 - keep Triton constexpr control flow explicit
        install_pre = action == ACTION_INSTALL_PRE
        apply = (action == ACTION_INSTALL) | install_pre
    else:
        apply = action == ACTION_CLEAR
    apply &= (region >= 0) & (region < total_regions)
    if ~apply:
        # The INSTALL launch is second and owns cleanup for every row,
        # including NONE, stale-owner rows, and rows already handled by CLEAR.
        if APPLY_INSTALL:
            tl.store(row_regions + row, -1)
            tl.store(row_positions + row, -1)
            tl.store(row_source + row, -1)
            tl.store(row_owner_block + row, -1)
            tl.store(row_owner_block_index + row, -1)
            tl.store(correction_action + row, ACTION_NONE)
        return
    safe_region = tl.where(apply, region, 0)
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < proxy_dim
    page = safe_region // summaries_per_page
    fragment = safe_region - page * summaries_per_page
    atom = dims // 128
    atom_dim = dims - atom * 128
    chunk = atom_dim // 16
    byte = atom_dim - chunk * 16
    row_in_block = fragment - (fragment // 8) * 8
    block_in_page = fragment // 8
    row_swizzle = safe_region % 8
    mean_offsets = (
        page * mean_stride_page
        + block_in_page * 2048
        + atom * 1024
        + row_in_block * 128
        + ((chunk ^ row_swizzle) * 16)
        + byte
    )
    has_state = tl.full((), True, dtype=tl.int1)
    if APPLY_INSTALL:
        post_numerator = tl.load(
            state_numerator + (row * proxy_dim + dims),
            mask=apply & ~install_pre & dim_mask,
            other=0.0,
        ).to(tl.float32)
        post_denominator = tl.load(
            state_denominator + (row * proxy_dim + dims),
            mask=apply & ~install_pre & dim_mask,
            other=0.0,
        ).to(tl.float32)
        post_max = tl.load(
            state_max_logits + row,
            mask=apply & ~install_pre,
            other=-float("inf"),
        ).to(tl.float32)
        pre_numerator = tl.load(
            state_pre_numerator + (row * proxy_dim + dims),
            mask=apply & install_pre & dim_mask,
            other=0.0,
        ).to(tl.float32)
        pre_denominator = tl.load(
            state_pre_denominator + (row * proxy_dim + dims),
            mask=apply & install_pre & dim_mask,
            other=0.0,
        ).to(tl.float32)
        pre_max = tl.load(
            state_pre_max_logits + row,
            mask=apply & install_pre,
            other=-float("inf"),
        ).to(tl.float32)
        numerator = tl.where(install_pre, pre_numerator, post_numerator)
        saved_denominator = tl.where(install_pre, pre_denominator, post_denominator)
        saved_max = tl.where(install_pre, pre_max, post_max)
        has_state = tl.max((saved_denominator > 0.0).to(tl.int32), axis=0) != 0
        summary = numerator / tl.maximum(saved_denominator, 1.0e-20)
        summary = tl.where(saved_denominator > 0.0, summary, 0.0)
        mean_fp8 = tl.cast(summary, tl.float8e4nv)
        mean_bytes = tl.cast(mean_fp8, tl.uint8, bitcast=True)
    else:
        numerator = tl.zeros((BLOCK_D,), dtype=tl.float32)
        saved_denominator = tl.zeros((BLOCK_D,), dtype=tl.float32)
        saved_max = -float("inf")
        mean_bytes = tl.zeros((BLOCK_D,), dtype=tl.uint8)
    tl.store(mean_cache + mean_offsets, mean_bytes, mask=apply & dim_mask)

    mapped_slot = tl.load(
        active_slot_by_region + safe_region,
        mask=apply,
        other=-1,
    ).to(tl.int64)
    mapped_ok = apply & (mapped_slot >= 0) & (mapped_slot < active_capacity)
    safe_mapped_slot = tl.where(mapped_ok, mapped_slot, 0)
    mapped_owner = tl.load(
        active_region_ids + safe_mapped_slot,
        mask=mapped_ok,
        other=-1,
    ).to(tl.int64)
    if mapped_ok & (mapped_owner == region):
        apply_csa = apply & has_state
        state_offsets = mapped_slot * proxy_dim + dims
        tl.store(
            active_numerator + state_offsets,
            numerator,
            mask=apply_csa & dim_mask,
        )
        tl.store(
            denominator + state_offsets,
            saved_denominator,
            mask=apply_csa & dim_mask,
        )
        tl.store(max_logits + mapped_slot, saved_max, mask=apply_csa)
        if APPLY_INSTALL:
            release_without_state = apply & ~has_state
            tl.store(
                active_numerator + state_offsets,
                0.0,
                mask=release_without_state & dim_mask,
            )
            tl.store(
                denominator + state_offsets,
                0.0,
                mask=release_without_state & dim_mask,
            )
            tl.store(
                max_logits + mapped_slot,
                -float("inf"),
                mask=release_without_state,
            )
        stage_offsets = tl.minimum(dims, region_block_size - 1)
        tl.store(
            active_token_valid
            + mapped_slot * active_token_valid_stride_slot
            + stage_offsets,
            0,
            mask=apply & (dims < region_block_size),
        )
        if APPLY_INSTALL:
            tl.store(
                active_region_ids + mapped_slot,
                -1,
                mask=apply & ~has_state,
            )
            tl.store(
                active_region_ids + mapped_slot,
                safe_region,
                mask=apply_csa,
            )
            tl.store(
                active_slot_by_region + safe_region,
                mapped_slot,
                mask=apply_csa,
            )
            tl.store(
                active_slot_by_region + safe_region,
                -1,
                mask=apply & ~apply_csa,
            )
            tl.store(row_regions + row, -1)
            tl.store(row_positions + row, -1)
            tl.store(row_source + row, -1)
            tl.store(row_owner_block + row, -1)
            tl.store(row_owner_block_index + row, -1)
            tl.store(correction_action + row, ACTION_NONE)
        else:
            tl.store(active_slot_by_region + safe_region, -1, mask=apply)
            tl.store(active_region_ids + mapped_slot, -1, mask=apply_csa)
        return

    if APPLY_INSTALL:
        # The preceding free-list launch repairs mappings for every live owner,
        # so an owner-invalid hint now proves that this region needs a slot.
        # Publishing the slot through the reverse-map CAS also makes duplicate
        # rows for one region converge on the same allocation.
        if mapped_slot != -1:
            tl.atomic_cas(
                active_slot_by_region + safe_region,
                mapped_slot.to(tl.int32),
                -1,
            )
        allocation_rank = tl.atomic_add(correction_allocation_cursor, 1)
        free_count = tl.load(correction_free_count)
        has_free_slot = allocation_rank < free_count
        candidate_slot = tl.load(
            correction_free_slots + allocation_rank,
            mask=has_free_slot,
            other=active_capacity,
        ).to(tl.int64)
        previous_slot = tl.full((), active_capacity, dtype=tl.int64)
        if has_free_slot:
            previous_slot = tl.atomic_cas(
                active_slot_by_region + safe_region,
                -1,
                candidate_slot.to(tl.int32),
            ).to(tl.int64)
        found_slot = tl.where(previous_slot == -1, candidate_slot, previous_slot)
    else:
        # The reverse map is a hint because reset and slot reuse can make it
        # stale before CLEAR. Owner-validated rows returned above, so only this
        # correctness path scans the bind-time capacity.
        candidates = tl.arange(0, BLOCK_CAPACITY)
        candidate_owners = tl.load(
            active_region_ids + candidates,
            mask=candidates < active_capacity,
            other=-1,
        ).to(tl.int64)
        found_slot = tl.min(
            tl.where(candidate_owners == region, candidates, active_capacity),
            axis=0,
        ).to(tl.int64)
    apply_csa = apply & has_state & (found_slot < active_capacity)
    if APPLY_INSTALL:
        tl.store(
            allocation_success,
            0,
            mask=apply & has_state & (found_slot >= active_capacity),
        )
    found_slot_valid = found_slot < active_capacity
    safe_found_slot = tl.where(found_slot_valid, found_slot, 0)
    safe_slot = tl.where(apply_csa, found_slot, 0)
    state_offsets = safe_slot * proxy_dim + dims
    tl.store(
        active_numerator + state_offsets,
        numerator,
        mask=apply_csa & dim_mask,
    )
    tl.store(
        denominator + state_offsets,
        saved_denominator,
        mask=apply_csa & dim_mask,
    )
    tl.store(max_logits + safe_slot, saved_max, mask=apply_csa)
    stage_offsets = tl.minimum(dims, region_block_size - 1)
    tl.store(
        active_token_valid
        + safe_found_slot * active_token_valid_stride_slot
        + stage_offsets,
        0,
        mask=apply & found_slot_valid & (dims < region_block_size),
    )
    if APPLY_INSTALL:
        release_empty = apply & ~has_state & found_slot_valid
        tl.store(
            active_region_ids + safe_found_slot,
            -1,
            mask=release_empty,
        )
        release_offsets = safe_found_slot * proxy_dim + dims
        tl.store(
            active_numerator + release_offsets,
            0.0,
            mask=release_empty & dim_mask,
        )
        tl.store(
            denominator + release_offsets,
            0.0,
            mask=release_empty & dim_mask,
        )
        tl.store(
            max_logits + safe_found_slot,
            -float("inf"),
            mask=release_empty,
        )
        tl.store(active_region_ids + safe_slot, safe_region, mask=apply_csa)
        tl.store(active_slot_by_region + safe_region, safe_slot, mask=apply_csa)
        tl.store(
            active_slot_by_region + safe_region,
            -1,
            mask=apply & ~apply_csa,
        )
    else:
        tl.store(active_slot_by_region + safe_region, -1, mask=apply)
        tl.store(active_region_ids + safe_slot, -1, mask=apply_csa)

    if APPLY_INSTALL:
        tl.store(row_regions + row, -1)
        tl.store(row_positions + row, -1)
        tl.store(row_source + row, -1)
        tl.store(row_owner_block + row, -1)
        tl.store(row_owner_block_index + row, -1)
        tl.store(correction_action + row, ACTION_NONE)


class Step4DSAMTPTransaction:
    """Fixed-capacity transaction payload owned by main attention."""

    @staticmethod
    def storage_size_bytes(
        *,
        max_num_reqs: int,
        max_rows_per_req: int,
        num_kv_heads: int,
        proxy_dim: int,
        source_map_rows: int,
        active_capacity: int,
    ) -> int:
        """Return the exact tensor payload allocated by one transaction."""
        num_rows = int(max_num_reqs) * int(max_rows_per_req)
        state_elements = num_rows * int(num_kv_heads) * int(proxy_dim)
        max_elements = num_rows * int(num_kv_heads)
        return (
            5 * num_rows * torch.tensor([], dtype=torch.long).element_size()
            + num_rows * torch.tensor([], dtype=torch.int8).element_size()
            + 4 * state_elements * torch.tensor([], dtype=torch.float32).element_size()
            + 2 * max_elements * torch.tensor([], dtype=torch.float32).element_size()
            + int(active_capacity) * torch.tensor([], dtype=torch.int32).element_size()
            + 2 * torch.tensor([], dtype=torch.int32).element_size()
            + int(source_map_rows) * torch.tensor([], dtype=torch.int32).element_size()
        )

    def __init__(
        self,
        *,
        max_num_reqs: int,
        max_rows_per_req: int,
        num_kv_heads: int,
        proxy_dim: int,
        source_map_rows: int,
        active_capacity: int,
        device: torch.device,
    ) -> None:
        self.max_num_reqs = int(max_num_reqs)
        self.max_rows_per_req = int(max_rows_per_req)
        self.num_rows = self.max_num_reqs * self.max_rows_per_req
        row_shape = (self.num_rows,)
        state_shape = (self.num_rows, int(num_kv_heads), int(proxy_dim))
        self.row_regions = torch.full(row_shape, -1, device=device, dtype=torch.long)
        self.row_positions = torch.full(row_shape, -1, device=device, dtype=torch.long)
        self.row_source = torch.full(row_shape, -1, device=device, dtype=torch.long)
        # Ownership fingerprint is captured with each post-token state. The
        # column is chosen past any shared prefix so correction can reject a
        # transaction whose request slot was recycled.
        self.row_owner_block = torch.full(
            row_shape, -1, device=device, dtype=torch.long
        )
        self.row_owner_block_index = torch.full(
            row_shape, -1, device=device, dtype=torch.long
        )
        self.correction_action = torch.full(
            row_shape, _MTP_ACTION_NONE, device=device, dtype=torch.int8
        )
        self.state_numerator = torch.zeros(
            state_shape, device=device, dtype=torch.float32
        )
        self.state_denominator = torch.zeros_like(self.state_numerator)
        self.state_max_logits = torch.full(
            (self.num_rows, int(num_kv_heads)),
            float("-inf"),
            device=device,
            dtype=torch.float32,
        )
        self.state_pre_numerator = torch.zeros(
            state_shape, device=device, dtype=torch.float32
        )
        self.state_pre_denominator = torch.zeros_like(self.state_pre_numerator)
        self.state_pre_max_logits = torch.full(
            (self.num_rows, int(num_kv_heads)),
            float("-inf"),
            device=device,
            dtype=torch.float32,
        )
        self.correction_free_slots = torch.empty(
            (int(active_capacity),), device=device, dtype=torch.int32
        )
        self.correction_free_count = torch.empty((1,), device=device, dtype=torch.int32)
        self.correction_allocation_cursor = torch.empty(
            (1,), device=device, dtype=torch.int32
        )
        # This map is persistent to avoid a full clear. The compact update
        # validates every hit against row_source and row_regions.
        self.source_to_transaction = torch.full(
            (int(source_map_rows),), -1, device=device, dtype=torch.int32
        )

    @torch.inference_mode()
    def reset_runtime_state(self) -> None:
        """Restore an empty transaction without reallocating captured storage."""
        for tensor in (
            self.row_regions,
            self.row_positions,
            self.row_source,
            self.row_owner_block,
            self.row_owner_block_index,
            self.source_to_transaction,
        ):
            tensor.fill_(-1)
        self.correction_action.fill_(_MTP_ACTION_NONE)
        for tensor in (
            self.state_numerator,
            self.state_denominator,
            self.state_pre_numerator,
            self.state_pre_denominator,
            self.correction_free_count,
            self.correction_allocation_cursor,
        ):
            tensor.zero_()
        self.state_max_logits.fill_(float("-inf"))
        self.state_pre_max_logits.fill_(float("-inf"))

    def invalidate_scheduler_blocks(
        self,
        block_ids: torch.Tensor,
        *,
        blocks_per_scheduler_block: int,
        summaries_per_page: int,
    ) -> None:
        if int(block_ids.numel()) == 0:
            return
        if self.row_regions.device.type != "cuda":
            ids = block_ids.to(device=self.row_regions.device, dtype=torch.long)
            if ids.numel() == 0:
                return
            scheduler_blocks = (self.row_regions // int(summaries_per_page)) // int(
                blocks_per_scheduler_block
            )
            reset = (self.row_regions >= 0) & torch.isin(scheduler_blocks, ids)
            if bool(reset.any()):
                self.row_regions[reset] = -1
                self.row_positions[reset] = -1
                self.row_source[reset] = -1
                self.row_owner_block[reset] = -1
                self.row_owner_block_index[reset] = -1
                self.correction_action[reset] = _MTP_ACTION_NONE
            return
        block_ids_per_program = 256
        grid = (
            int(self.num_rows),
            triton.cdiv(int(block_ids.numel()), block_ids_per_program),
        )
        _step4_invalidate_mtp_transaction_for_reset_blocks_kernel[grid](
            block_ids,
            self.row_regions,
            self.row_positions,
            self.row_source,
            self.row_owner_block,
            self.row_owner_block_index,
            self.correction_action,
            int(block_ids.numel()),
            summaries_per_page=int(summaries_per_page),
            blocks_per_scheduler_block=int(blocks_per_scheduler_block),
            ACTION_NONE=_MTP_ACTION_NONE,
            BLOCK_IDS=block_ids_per_program,
        )

    def warmup_invalidate_scheduler_blocks(
        self,
        *,
        summaries_per_page: int,
        blocks_per_scheduler_block: int,
    ) -> None:
        """Compile reset invalidation variants without mutating transaction state."""
        if self.row_regions.device.type != "cuda":
            return
        block_ids = torch.empty((1,), device=self.row_regions.device, dtype=torch.long)
        block_ids_per_program = 256
        _step4_invalidate_mtp_transaction_for_reset_blocks_kernel.warmup(
            block_ids,
            self.row_regions,
            self.row_positions,
            self.row_source,
            self.row_owner_block,
            self.row_owner_block_index,
            self.correction_action,
            1,
            summaries_per_page=int(summaries_per_page),
            blocks_per_scheduler_block=int(blocks_per_scheduler_block),
            ACTION_NONE=_MTP_ACTION_NONE,
            BLOCK_IDS=block_ids_per_program,
            grid=(int(self.num_rows), 1),
        )


class Step4DSAMTP:
    """Optional MTP transaction controller attached to existing attention."""

    def __init__(self, owner: object, num_speculative_tokens: int) -> None:
        self.owner = owner
        self.num_speculative_tokens = int(num_speculative_tokens)

    def _max_scratch_rows(self) -> int:
        transaction_rows = int(self.owner.max_num_seqs) * (
            self.num_speculative_tokens + 1
        )
        max_num_batched_tokens = int(
            getattr(self.owner, "max_num_batched_tokens", 0) or transaction_rows
        )
        return self.owner._round_up(
            max(max_num_batched_tokens, transaction_rows),
            64,
        )

    def runtime_state_size_bytes(
        self,
        *,
        num_kv_heads: int,
        proxy_dim: int,
        active_capacity: int,
    ) -> int:
        return Step4DSAMTPTransaction.storage_size_bytes(
            max_num_reqs=int(self.owner.max_num_seqs),
            max_rows_per_req=self.num_speculative_tokens + 1,
            num_kv_heads=num_kv_heads,
            proxy_dim=proxy_dim,
            source_map_rows=self._max_scratch_rows(),
            active_capacity=active_capacity,
        )

    def prepare_scratch(self, summary_cache: object) -> None:
        """Reserve the shared MTP scratch without creating layer state."""
        max_rows = self._max_scratch_rows()
        max_num_reqs = int(self.owner.max_num_seqs)
        device = summary_cache.sum_cache.device
        for name, dtype in (
            ("csa_prefill_scratch_region_ids", torch.int32),
            ("csa_prefill_scratch_reset_map", torch.int32),
        ):
            self.owner._get_dsa_tensor_buffer_at_least(
                name,
                (max_rows,),
                device=device,
                dtype=dtype,
            )
        self.owner._get_dsa_tensor_buffer_at_least(
            "csa_prefill_scratch_row_map",
            (max_rows, int(summary_cache.region_block_size)),
            device=device,
            dtype=torch.int32,
        )
        for name, dtype in (
            ("csa_mtp_prefill_flat_slot", torch.int64),
            ("csa_mtp_prefill_token_positions", torch.int64),
            ("csa_mtp_prefill_reset_slots", torch.int64),
            ("csa_mtp_prefill_token_valid", torch.bool),
            ("csa_mtp_q1_token_valid", torch.bool),
            ("csa_mtp_transaction_token_valid", torch.bool),
        ):
            self.owner._get_dsa_tensor_buffer_at_least(
                name,
                (max_rows,),
                device=device,
                dtype=dtype,
            )
        self.owner._get_dsa_tensor_buffer_at_least(
            "csa_mtp_prefill_query_start_loc",
            (max_num_reqs + 1,),
            device=device,
            dtype=torch.int32,
        )
        self.owner._get_dsa_tensor_buffer_at_least(
            "csa_mtp_prefill_seq_lens",
            (max_num_reqs,),
            device=device,
            dtype=torch.int32,
        )

    def initialize(self, summary_cache: object) -> None:
        max_num_reqs = int(self.owner.max_num_seqs)
        num_kv_heads = int(summary_cache.num_kv_heads)
        if num_kv_heads != 1:
            raise ValueError(
                "Step4 DSA MTP transaction supports exactly one summary "
                f"KV head, got {num_kv_heads}."
            )
        max_rows = self._max_scratch_rows()
        active_ids, _, _, _, _ = self.owner._csa_summary_state(summary_cache)
        transaction = Step4DSAMTPTransaction(
            max_num_reqs=max_num_reqs,
            max_rows_per_req=self.num_speculative_tokens + 1,
            num_kv_heads=num_kv_heads,
            proxy_dim=int(summary_cache.proxy_dim),
            source_map_rows=max_rows,
            active_capacity=int(active_ids.numel()),
            device=summary_cache.sum_cache.device,
        )
        summary_cache._step4_mtp_transaction = transaction

        # The attention owner binds common layout/index scratch. MTP owns its
        # correction transaction and mixed-prefix partition scratch; all
        # runtime calls below take active views from these fixed buffers.
        self.prepare_scratch(summary_cache)
        transaction.warmup_invalidate_scheduler_blocks(
            summaries_per_page=int(summary_cache.summaries_per_page),
            blocks_per_scheduler_block=max(
                1, int(summary_cache.blocks_per_scheduler_block)
            ),
        )

    def _update_summary_cache(
        self,
        *,
        summary_cache: object,
        layout: object,
        index_k: torch.Tensor,
        index_z: torch.Tensor,
        num_actual_tokens: int,
        use_decode_update: bool,
        preserve_completed_slots: bool,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        num_verifier_requests: int,
        num_verifier_tokens: int,
        has_q1_decode: bool,
        valid_requests: torch.Tensor,
        valid_tokens: torch.Tensor | None,
        step_metadata: object | None = None,
    ) -> None:
        if (
            not isinstance(valid_tokens, torch.Tensor)
            or valid_tokens.device != query_start_loc.device
            or valid_tokens.dtype != torch.int32
            or valid_tokens.ndim != 1
            or int(valid_tokens.numel()) != 1
            or not valid_tokens.is_contiguous()
        ):
            raise RuntimeError(
                "Step4 DSA MTP summary update requires dsa_valid_tokens as a "
                "contiguous device int32 tensor with shape [1]"
            )
        if (
            not isinstance(valid_requests, torch.Tensor)
            or valid_requests.device != query_start_loc.device
            or valid_requests.dtype != torch.int32
            or valid_requests.ndim != 1
            or int(valid_requests.numel()) != 1
            or not valid_requests.is_contiguous()
        ):
            raise RuntimeError(
                "Step4 DSA MTP summary update requires dsa_valid_requests "
                "as a contiguous device int32 tensor with shape [1]"
            )
        if preserve_completed_slots and has_q1_decode:
            # mtp_num_verifier_reqs is the complete decode prefix, so it can
            # contain ordinary q1 decodes alongside q2+ MTP verifiers. Keep
            # q1 on the ordinary compact-decode contract and mask those rows
            # out of the transaction update without assuming request order.
            q1_token_valid = self.owner._get_dsa_tensor_buffer_at_least(
                "csa_mtp_q1_token_valid",
                (num_verifier_tokens,),
                device=index_k.device,
                dtype=torch.bool,
            )
            transaction_token_valid = self.owner._get_dsa_tensor_buffer_at_least(
                "csa_mtp_transaction_token_valid",
                (int(layout.token_valid.shape[0]),),
                device=index_k.device,
                dtype=torch.bool,
            )
            q1_token_valid.zero_()
            # The transaction mask is a verifier-only mask.  Do not inherit
            # prefill validity from the full layout: those rows must stay on
            # the ordinary summary-update path below.
            transaction_token_valid.zero_()
            transaction = summary_cache._step4_mtp_transaction
            _step4_dsa_mtp_partition_decode_validity_kernel[
                (num_verifier_requests, int(transaction.max_rows_per_req))
            ](
                query_start_loc,
                layout.token_valid,
                q1_token_valid,
                transaction_token_valid,
                valid_requests,
                valid_tokens,
                int(num_verifier_requests),
                max_rows_per_req=int(transaction.max_rows_per_req),
            )
            q1_layout = type(layout)(
                token_flat_slot=layout.token_flat_slot[:num_verifier_tokens],
                token_positions=layout.token_positions[:num_verifier_tokens],
                reset_slots=layout.reset_slots[:num_verifier_tokens],
                token_valid=q1_token_valid,
            )
            self.owner._update_summary_cache_with_padded_layout(
                summary_cache=summary_cache,
                layout=q1_layout,
                index_k=index_k[:num_verifier_tokens],
                num_actual_tokens=num_verifier_tokens,
                proxy_dim=int(summary_cache.proxy_dim),
                use_decode_update=True,
                index_z=index_z[:num_verifier_tokens],
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                step_metadata=None,
            )
            transaction_layout = type(layout)(
                token_flat_slot=layout.token_flat_slot,
                token_positions=layout.token_positions,
                reset_slots=layout.reset_slots,
                token_valid=transaction_token_valid,
            )
            self._update_summary_cache(
                summary_cache=summary_cache,
                layout=transaction_layout,
                index_k=index_k,
                index_z=index_z,
                num_actual_tokens=num_actual_tokens,
                use_decode_update=use_decode_update,
                preserve_completed_slots=preserve_completed_slots,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                block_table=block_table,
                num_verifier_requests=num_verifier_requests,
                num_verifier_tokens=num_verifier_tokens,
                has_q1_decode=False,
                valid_requests=valid_requests,
                valid_tokens=valid_tokens,
                step_metadata=step_metadata,
            )
            return
        if preserve_completed_slots and 0 < num_verifier_tokens < num_actual_tokens:
            # Run the transaction-aware ordered update only on the verifier
            # prefix. Newly admitted prefills must use the ordinary summary
            # update, exactly as they do when scheduled alone. Feeding both
            # groups through the transaction kernel lets prefill rows mutate
            # verifier transaction state and makes the new request's summary
            # depend on an unrelated request sharing the batch.
            verifier_layout = type(layout)(
                token_flat_slot=layout.token_flat_slot[:num_verifier_tokens],
                token_positions=layout.token_positions[:num_verifier_tokens],
                reset_slots=layout.reset_slots[:num_verifier_tokens],
                token_valid=layout.token_valid[:num_verifier_tokens],
            )
            self._update_summary_cache(
                summary_cache=summary_cache,
                layout=verifier_layout,
                index_k=index_k[:num_verifier_tokens],
                index_z=index_z[:num_verifier_tokens],
                num_actual_tokens=num_verifier_tokens,
                use_decode_update=use_decode_update,
                preserve_completed_slots=True,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                block_table=block_table,
                num_verifier_requests=num_verifier_requests,
                num_verifier_tokens=num_verifier_tokens,
                has_q1_decode=False,
                valid_requests=valid_requests,
                valid_tokens=valid_tokens,
                step_metadata=step_metadata,
            )

            prefill_tokens = num_actual_tokens - num_verifier_tokens
            prefill_flat_slot = self.owner._get_dsa_tensor_buffer_at_least(
                "csa_mtp_prefill_flat_slot",
                (prefill_tokens,),
                device=index_k.device,
                dtype=torch.int64,
            )
            prefill_token_positions = self.owner._get_dsa_tensor_buffer_at_least(
                "csa_mtp_prefill_token_positions",
                (prefill_tokens,),
                device=index_k.device,
                dtype=torch.int64,
            )
            prefill_reset_slots = self.owner._get_dsa_tensor_buffer_at_least(
                "csa_mtp_prefill_reset_slots",
                (prefill_tokens,),
                device=index_k.device,
                dtype=torch.int64,
            )
            prefill_token_valid = self.owner._get_dsa_tensor_buffer_at_least(
                "csa_mtp_prefill_token_valid",
                (prefill_tokens,),
                device=index_k.device,
                dtype=torch.bool,
            )
            prefill_flat_slot.copy_(
                layout.token_flat_slot[num_verifier_tokens:num_actual_tokens]
            )
            prefill_token_positions.copy_(
                layout.token_positions[num_verifier_tokens:num_actual_tokens]
            )
            prefill_reset_slots.copy_(
                layout.reset_slots[num_verifier_tokens:num_actual_tokens]
            )
            prefill_token_valid.copy_(
                layout.token_valid[num_verifier_tokens:num_actual_tokens]
            )
            prefill_layout = type(layout)(
                token_flat_slot=prefill_flat_slot,
                token_positions=prefill_token_positions,
                reset_slots=prefill_reset_slots,
                token_valid=prefill_token_valid,
            )
            num_prefill_requests = (
                int(query_start_loc.shape[0]) - 1 - int(num_verifier_requests)
            )
            prefill_query_start_loc = self.owner._get_dsa_tensor_buffer_at_least(
                "csa_mtp_prefill_query_start_loc",
                (num_prefill_requests + 1,),
                device=index_k.device,
                dtype=torch.int32,
            )
            torch.sub(
                query_start_loc[num_verifier_requests:],
                num_verifier_tokens,
                out=prefill_query_start_loc,
            )
            prefill_seq_lens = self.owner._get_dsa_tensor_buffer_at_least(
                "csa_mtp_prefill_seq_lens",
                (num_prefill_requests,),
                device=index_k.device,
                dtype=torch.int32,
            )
            prefill_seq_lens.copy_(seq_lens[num_verifier_requests:])
            self.owner._update_summary_cache_with_padded_layout(
                summary_cache=summary_cache,
                layout=prefill_layout,
                index_k=index_k[num_verifier_tokens:num_actual_tokens],
                num_actual_tokens=prefill_tokens,
                proxy_dim=int(summary_cache.proxy_dim),
                use_decode_update=False,
                index_z=index_z[num_verifier_tokens:num_actual_tokens],
                query_start_loc=prefill_query_start_loc,
                seq_lens=prefill_seq_lens,
                step_metadata=None,
            )
            return
        if not preserve_completed_slots:
            self.owner._update_summary_cache_with_padded_layout(
                summary_cache=summary_cache,
                layout=layout,
                index_k=index_k,
                num_actual_tokens=num_actual_tokens,
                proxy_dim=int(summary_cache.proxy_dim),
                use_decode_update=use_decode_update,
                index_z=index_z,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                step_metadata=step_metadata,
            )
            return

        active_ids, active_slots, active_num, denominator, max_logits = (
            self.owner._csa_summary_state(summary_cache)
        )
        layout_rows = int(layout.token_flat_slot.shape[0])
        scratch_shape = (layout_rows,)
        scratch_region_ids = self.owner._get_dsa_tensor_buffer_at_least(
            "csa_prefill_scratch_region_ids",
            scratch_shape,
            device=index_k.device,
            dtype=torch.int32,
        )
        scratch_row_map = self.owner._get_dsa_tensor_buffer_at_least(
            "csa_prefill_scratch_row_map",
            (layout_rows, int(summary_cache.region_block_size)),
            device=index_k.device,
            dtype=torch.int32,
        )
        scratch_reset_map = self.owner._get_dsa_tensor_buffer_at_least(
            "csa_prefill_scratch_reset_map",
            scratch_shape,
            device=index_k.device,
            dtype=torch.int32,
        )
        transaction = summary_cache._step4_mtp_transaction
        allocation_success = self.owner._begin_csa_allocation_check(summary_cache)
        with torch.profiler.record_function(
            "step4_dsa.mtp.csa.compact_prefill_update_op"
        ):
            torch.ops.vllm.step4_dsa_mtp_csa_update(
                summary_cache.mean_cache,
                active_ids,
                active_slots,
                allocation_success,
                active_num,
                denominator,
                max_logits,
                summary_cache._step4_csa_active_token_k,
                summary_cache._step4_csa_active_token_z,
                summary_cache._step4_csa_active_token_valid,
                layout.token_flat_slot,
                layout.reset_slots,
                layout.token_valid,
                layout.token_positions,
                index_k,
                index_z,
                query_start_loc,
                seq_lens,
                block_table,
                transaction.source_to_transaction,
                transaction.row_source,
                transaction.row_regions,
                transaction.row_positions,
                transaction.row_owner_block,
                transaction.row_owner_block_index,
                transaction.correction_action,
                valid_requests,
                transaction.state_numerator,
                transaction.state_denominator,
                transaction.state_max_logits,
                transaction.state_pre_numerator,
                transaction.state_pre_denominator,
                transaction.state_pre_max_logits,
                int(summary_cache.region_block_size),
                int(summary_cache.page_size),
                int(transaction.max_rows_per_req),
                int(num_verifier_requests),
                scratch_region_ids,
                scratch_row_map,
                scratch_reset_map,
                valid_tokens,
                transaction.correction_free_slots,
                transaction.correction_free_count,
                transaction.correction_allocation_cursor,
            )
        self.owner._assert_csa_allocation_success(allocation_success)

    def correct(self, *, summary_cache: object, attn_metadata: object) -> None:
        transaction = summary_cache._step4_mtp_transaction
        query_start_loc = attn_metadata.query_start_loc
        valid_requests = getattr(attn_metadata, "dsa_valid_requests", None)
        if (
            not isinstance(valid_requests, torch.Tensor)
            or valid_requests.device != query_start_loc.device
            or valid_requests.dtype != torch.int32
            or valid_requests.ndim != 1
            or int(valid_requests.numel()) != 1
            or not valid_requests.is_contiguous()
        ):
            raise RuntimeError(
                "Step4 DSA MTP correction requires dsa_valid_requests as a "
                "contiguous device int32 tensor with shape [1]"
            )
        _step4_prepare_mtp_correction_kernel[(transaction.max_num_reqs,)](
            query_start_loc,
            attn_metadata.seq_lens,
            attn_metadata.block_table,
            transaction.row_regions,
            transaction.row_positions,
            transaction.row_owner_block,
            transaction.row_owner_block_index,
            transaction.correction_action,
            valid_requests,
            int(attn_metadata.block_table.shape[1]),
            int(attn_metadata.block_table.stride(0)),
            int(attn_metadata.block_table.stride(1)),
            page_size=int(summary_cache.page_size),
            summaries_per_page=int(summary_cache.summaries_per_page),
            max_rows_per_req=int(transaction.max_rows_per_req),
            BLOCK_ROWS=triton.next_power_of_2(transaction.max_rows_per_req),
            ACTION_NONE=_MTP_ACTION_NONE,
            ACTION_CLEAR=_MTP_ACTION_CLEAR,
            ACTION_INSTALL=_MTP_ACTION_INSTALL,
            ACTION_INSTALL_PRE=_MTP_ACTION_INSTALL_PRE,
        )
        active_ids, active_slots, active_num, denominator, max_logits = (
            self.owner._csa_summary_state(summary_cache)
        )
        active_token_valid = summary_cache._step4_csa_active_token_valid
        allocation_success = self.owner._begin_csa_allocation_check(summary_cache)
        # Clear and install are separate launches because Triton programs do
        # not have a grid-wide barrier. Each launch updates summary and CSA
        # together, so no intermediate indexing tensors are materialized.
        for apply_install in (False, True):
            if apply_install:
                free_slot_block = 256
                _step4_reset_mtp_correction_allocator_kernel[(1,)](
                    transaction.correction_free_count,
                    transaction.correction_allocation_cursor,
                )
                _step4_build_mtp_correction_free_slots_kernel[
                    (triton.cdiv(int(active_ids.numel()), free_slot_block),)
                ](
                    active_ids,
                    active_slots,
                    transaction.correction_free_slots,
                    transaction.correction_free_count,
                    int(active_ids.numel()),
                    int(
                        summary_cache.mean_cache.shape[0]
                        * summary_cache.mean_cache.shape[1]
                    ),
                    BLOCK_CAPACITY=free_slot_block,
                )
            _step4_apply_mtp_correction_kernel[(transaction.num_rows,)](
                transaction.row_regions,
                transaction.row_positions,
                transaction.row_source,
                transaction.row_owner_block,
                transaction.row_owner_block_index,
                transaction.correction_action,
                transaction.state_numerator,
                transaction.state_denominator,
                transaction.state_max_logits,
                transaction.state_pre_numerator,
                transaction.state_pre_denominator,
                transaction.state_pre_max_logits,
                summary_cache.mean_cache,
                active_ids,
                active_slots,
                allocation_success,
                active_num,
                denominator,
                max_logits,
                active_token_valid,
                transaction.correction_free_slots,
                transaction.correction_free_count,
                transaction.correction_allocation_cursor,
                mean_stride_page=int(summary_cache.mean_cache.stride(0)),
                total_regions=int(
                    summary_cache.mean_cache.shape[0]
                    * summary_cache.mean_cache.shape[1]
                ),
                summaries_per_page=int(summary_cache.summaries_per_page),
                active_capacity=int(active_ids.numel()),
                proxy_dim=int(summary_cache.proxy_dim),
                active_token_valid_stride_slot=int(active_token_valid.stride(0)),
                region_block_size=int(summary_cache.region_block_size),
                ACTION_NONE=_MTP_ACTION_NONE,
                ACTION_CLEAR=_MTP_ACTION_CLEAR,
                ACTION_INSTALL=_MTP_ACTION_INSTALL,
                ACTION_INSTALL_PRE=_MTP_ACTION_INSTALL_PRE,
                APPLY_INSTALL=apply_install,
                BLOCK_CAPACITY=triton.next_power_of_2(int(active_ids.numel())),
                BLOCK_D=triton.next_power_of_2(int(summary_cache.proxy_dim)),
            )
        self.owner._assert_csa_allocation_success(allocation_success)

    def update(
        self,
        *,
        summary_cache: object,
        attn_metadata: object,
        layout: object,
        index_k: torch.Tensor,
        index_z: torch.Tensor,
        num_actual_tokens: int,
        use_decode_update: bool,
        step_metadata: object | None = None,
    ) -> None:
        verifier_requests_value = getattr(attn_metadata, "mtp_num_verifier_reqs", None)
        if isinstance(verifier_requests_value, bool) or not isinstance(
            verifier_requests_value, Integral
        ):
            raise RuntimeError(
                "Step4 DSA MTP update requires mtp_num_verifier_reqs as a "
                "non-negative host integer; got type="
                f"{type(verifier_requests_value).__name__}"
            )
        verifier_requests = int(verifier_requests_value)
        if verifier_requests < 0:
            raise RuntimeError(
                "Step4 DSA MTP update requires mtp_num_verifier_reqs as a "
                f"non-negative host integer, got {verifier_requests}"
            )
        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        if (
            not isinstance(query_start_loc_cpu, torch.Tensor)
            or query_start_loc_cpu.device.type != "cpu"
            or query_start_loc_cpu.dtype != torch.int32
            or query_start_loc_cpu.ndim != 1
            or int(query_start_loc_cpu.numel()) <= verifier_requests
        ):
            raise RuntimeError(
                "Step4 DSA MTP update requires query_start_loc_cpu to cover "
                f"the verifier prefix, got {query_start_loc_cpu!r}"
            )
        num_verifier_tokens = int(query_start_loc_cpu[verifier_requests].item())
        if not 0 <= num_verifier_tokens <= num_actual_tokens:
            raise RuntimeError(
                "Step4 DSA MTP verifier token prefix is outside the active "
                f"token range: {num_verifier_tokens} not in "
                f"[0, {num_actual_tokens}]"
            )
        verifier_query_lens_cpu = (
            query_start_loc_cpu[1 : verifier_requests + 1]
            - query_start_loc_cpu[:verifier_requests]
        )
        has_q1_decode = bool((verifier_query_lens_cpu == 1).any().item())

        max_query_len_value = getattr(attn_metadata, "max_query_len", None)
        if isinstance(max_query_len_value, bool) or not isinstance(
            max_query_len_value, Integral
        ):
            raise RuntimeError(
                "Step4 DSA MTP update requires max_query_len as a positive "
                "host integer; got type="
                f"{type(max_query_len_value).__name__}"
            )
        max_query_len = int(max_query_len_value)
        if max_query_len <= 0:
            raise RuntimeError(
                "Step4 DSA MTP update requires max_query_len as a positive "
                f"host integer, got {max_query_len}"
            )

        valid_tokens = getattr(attn_metadata, "dsa_valid_tokens", None)

        with torch.profiler.record_function("step4_dsa.mtp.correction"):
            self.correct(
                summary_cache=summary_cache,
                attn_metadata=attn_metadata,
            )
        with torch.profiler.record_function("step4_dsa.summary.update_kernel"):
            self._update_summary_cache(
                summary_cache=summary_cache,
                layout=layout,
                index_k=index_k,
                index_z=index_z,
                num_actual_tokens=num_actual_tokens,
                use_decode_update=use_decode_update,
                preserve_completed_slots=(verifier_requests > 0 and max_query_len > 1),
                query_start_loc=attn_metadata.query_start_loc,
                seq_lens=attn_metadata.seq_lens,
                block_table=attn_metadata.block_table,
                num_verifier_requests=verifier_requests,
                num_verifier_tokens=num_verifier_tokens,
                has_q1_decode=has_q1_decode,
                valid_requests=getattr(attn_metadata, "dsa_valid_requests", None),
                valid_tokens=valid_tokens,
                step_metadata=step_metadata,
            )
