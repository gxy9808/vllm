# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.models.step4.nvidia.ops.cute_dsl.cutedsl_compile_cache import cached_compile_function
from vllm.models.step4.nvidia.ops.cute_dsl.utils import elem_pointer
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.kernel_utils import cute_utils as utils

_UNION_FILL_THREADS = 256
_UNION_COMPACT_THREADS = 512


# Grouped-union GQA prefill metadata builders.
@dsl_user_op
def _atomic_or_i32(ptr: cute.Pointer, val: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        nvvm.atomicrmw(
            op=nvvm.AtomicOpKind.OR,
            ptr=ptr.llvm_ptr,
            a=Int32(val).ir_value(loc=loc, ip=ip),
        )
    )


@dsl_user_op
def _popc_i32(x: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(x).ir_value(loc=loc, ip=ip)],
            "popc.b32 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _make_grouped_union_logical_bitset_fill_kernel(
    *,
    q_group: int,
    topk_windows: int,
    topk_width: int,
    num_region_bins: int,
    bit_words: int,
    bit_words_width: int,
    explicit_work_ranges: bool,
):
    @cute.kernel
    def _grouped_union_logical_bitset_fill_kernel(
        mRegionLogical: cute.Tensor,
        mRegionPhys: cute.Tensor,
        mRegionCounts: cute.Tensor,
        mBitset: cute.Tensor,
        mWorkQLocal: cute.Tensor,
        mWorkQLen: cute.Tensor,
        total_q: Int32,
    ):
        group_id, _, _ = cute.arch.block_idx()
        tx = cute.arch.thread_idx()[0]
        q_base = group_id * Int32(q_group)
        q_len = total_q
        if cutlass.const_expr(explicit_work_ranges):
            q_base = mWorkQLocal[group_id]
            q_len = mWorkQLen[group_id]

        for word_chunk in cutlass.range_constexpr(
            (bit_words_width + _UNION_FILL_THREADS - 1) // _UNION_FILL_THREADS
        ):
            word_idx = tx + Int32(word_chunk * _UNION_FILL_THREADS)
            if word_idx < Int32(bit_words):
                mBitset[group_id, word_idx] = Int32(0)
        cute.arch.sync_threads()

        num_values = Int32(q_group * topk_width)
        num_tiles = (num_values + Int32(_UNION_FILL_THREADS - 1)) // Int32(_UNION_FILL_THREADS)
        for tile in cutlass.range(num_tiles, unroll=1):
            value_offset = tile * Int32(_UNION_FILL_THREADS) + tx
            if value_offset < num_values:
                q_slot = value_offset // Int32(topk_width)
                win = value_offset - q_slot * Int32(topk_width)
                q_pos = q_base + q_slot
                if q_pos < total_q and q_pos < q_len:
                    count = mRegionCounts[q_pos]
                    if win < count and win < Int32(topk_windows):
                        logical_start = mRegionLogical[q_pos, win]
                        phys = mRegionPhys[q_pos, win]
                        logical_region = logical_start // Int32(8)
                        if (
                            logical_region >= Int32(0)
                            and logical_region < Int32(num_region_bins)
                            and phys >= Int32(0)
                        ):
                            word_idx = logical_region // Int32(32)
                            bit_idx = logical_region - word_idx * Int32(32)
                            bit_mask = Int32(1) << bit_idx
                            old_word = mBitset[group_id, word_idx]
                            if (old_word & bit_mask) == Int32(0):
                                _atomic_or_i32(
                                    elem_pointer(mBitset, (group_id, word_idx)),
                                    bit_mask,
                                )

    return _grouped_union_logical_bitset_fill_kernel


def _make_grouped_union_logical_bitset_compact_kernel(
    *,
    q_group: int,
    max_union_windows: int,
    num_region_bins: int,
    bit_words: int,
    bit_words_width: int,
    topk_windows: int,
    topk_width: int,
    emit_causal_limits: bool,
    emit_exact_mask: bool,
    explicit_work_ranges: bool,
):
    word_chunks = (bit_words_width + _UNION_COMPACT_THREADS - 1) // _UNION_COMPACT_THREADS

    @cute.kernel
    def _grouped_union_logical_bitset_compact_kernel(
        mRegionLogical: cute.Tensor,
        mRegionPhys: cute.Tensor,
        mRegionCounts: cute.Tensor,
        mBitset: cute.Tensor,
        mOutReqIdx: cute.Tensor,
        mWorkQGlobal: cute.Tensor,
        mWorkQLocal: cute.Tensor,
        mWorkQLen: cute.Tensor,
        mUnionPhys: cute.Tensor,
        mUnionLogical: cute.Tensor,
        mUnionCounts: cute.Tensor,
        mCausalLimits: cute.Tensor,
        mExactMaskBits: cute.Tensor,
        total_q: Int32,
        q_global_offset: Int32,
    ):
        group_id, _, _ = cute.arch.block_idx()
        tx = cute.arch.thread_idx()[0]
        q_base = group_id * Int32(q_group)
        q_global_base = q_base + q_global_offset
        q_len = total_q
        if cutlass.const_expr(explicit_work_ranges):
            q_base = mWorkQLocal[group_id]
            q_global_base = mWorkQGlobal[group_id]
            q_len = mWorkQLen[group_id]
        q_tile_count = cutlass.min(Int32(q_group), q_len - q_base)
        if q_tile_count < Int32(0):
            q_tile_count = Int32(0)

        @cute.struct
        class SharedStorage:
            words: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, bit_words_width], 128]
            prefix: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, bit_words_width], 128]
            popc: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, bit_words_width], 128]
            chunk_offsets: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, word_chunks], 128]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage, 16)
        sWords = storage.words.get_tensor(cute.make_layout((bit_words_width,), stride=(1,)))
        sPrefix = storage.prefix.get_tensor(cute.make_layout((bit_words_width,), stride=(1,)))
        sPopc = storage.popc.get_tensor(cute.make_layout((bit_words_width,), stride=(1,)))
        sChunkOffsets = storage.chunk_offsets.get_tensor(cute.make_layout((word_chunks,), stride=(1,)))

        for word_chunk in cutlass.range_constexpr(word_chunks):
            word_idx = tx + Int32(word_chunk * _UNION_COMPACT_THREADS)
            if word_idx < Int32(bit_words_width):
                word = Int32(0)
                if word_idx < Int32(bit_words):
                    word = mBitset[group_id, word_idx]
                pc = _popc_i32(word)
                sWords[word_idx] = word
                sPopc[word_idx] = pc
                sPrefix[word_idx] = pc
        cute.arch.sync_threads()

        for word_chunk in cutlass.range_constexpr(word_chunks):
            word_base = Int32(word_chunk * _UNION_COMPACT_THREADS)
            word_idx = word_base + tx
            for offset_i in cutlass.range_constexpr((_UNION_COMPACT_THREADS - 1).bit_length()):
                offset = Int32(1 << offset_i)
                addend = Int32(0)
                if word_idx < Int32(bit_words_width) and tx >= offset:
                    addend = sPrefix[word_idx - offset]
                cute.arch.sync_threads()
                if word_idx < Int32(bit_words_width) and tx >= offset:
                    sPrefix[word_idx] = sPrefix[word_idx] + addend
                cute.arch.sync_threads()
            if tx == Int32(0):
                chunk_last = Int32(
                    min((word_chunk + 1) * _UNION_COMPACT_THREADS, bit_words_width) - 1
                )
                sChunkOffsets[Int32(word_chunk)] = sPrefix[chunk_last]
            cute.arch.sync_threads()

        if tx == Int32(0):
            running = Int32(0)
            for word_chunk in cutlass.range_constexpr(word_chunks):
                chunk_total = sChunkOffsets[Int32(word_chunk)]
                sChunkOffsets[Int32(word_chunk)] = running
                running = running + chunk_total
        cute.arch.sync_threads()

        for word_chunk in cutlass.range_constexpr(word_chunks):
            word_idx = tx + Int32(word_chunk * _UNION_COMPACT_THREADS)
            if word_idx < Int32(bit_words_width):
                sPrefix[word_idx] = sPrefix[word_idx] + sChunkOffsets[Int32(word_chunk)]
        cute.arch.sync_threads()

        if cutlass.const_expr(emit_exact_mask):
            exact_values = Int32(max_union_windows)
            exact_tiles = (exact_values + Int32(_UNION_COMPACT_THREADS - 1)) // Int32(_UNION_COMPACT_THREADS)
            for tile in cutlass.range(exact_tiles, unroll=1):
                rank = tile * Int32(_UNION_COMPACT_THREADS) + tx
                if rank < exact_values:
                    mExactMaskBits[group_id, rank] = Int32(0)
            cute.arch.sync_threads()

        num_values = Int32(q_group * topk_width)
        num_tiles = (num_values + Int32(_UNION_COMPACT_THREADS - 1)) // Int32(_UNION_COMPACT_THREADS)
        for tile in cutlass.range(num_tiles, unroll=1):
            value_offset = tile * Int32(_UNION_COMPACT_THREADS) + tx
            if value_offset < num_values:
                q_slot = value_offset // Int32(topk_width)
                win = value_offset - q_slot * Int32(topk_width)
                q_pos = q_base + q_slot
                if q_pos < total_q and q_pos < q_len:
                    count = mRegionCounts[q_pos]
                    if win < count and win < Int32(topk_windows):
                        logical_start = mRegionLogical[q_pos, win]
                        phys = mRegionPhys[q_pos, win]
                        logical_region = logical_start // Int32(8)
                        if (
                            logical_region >= Int32(0)
                            and logical_region < Int32(num_region_bins)
                            and phys >= Int32(0)
                        ):
                            word_idx = logical_region // Int32(32)
                            bit_idx = logical_region - word_idx * Int32(32)
                            bit_mask = Int32(1) << bit_idx
                            word = sWords[word_idx]
                            if (word & bit_mask) != Int32(0):
                                before_words = Int32(0)
                                if word_idx > Int32(0):
                                    before_words = sPrefix[word_idx - Int32(1)]
                                rank = before_words + _popc_i32(word & (bit_mask - Int32(1)))
                                if rank < Int32(max_union_windows):
                                    mUnionPhys[group_id, rank] = phys
                                    if cutlass.const_expr(emit_exact_mask):
                                        mUnionLogical[group_id, rank] = logical_region
                                        _atomic_or_i32(
                                            elem_pointer(mExactMaskBits, (group_id, rank)),
                                            Int32(1) << q_slot,
                                        )
        cute.arch.sync_threads()

        if tx == Int32(0):
            union_count = cutlass.min(sPrefix[Int32(bit_words_width - 1)], Int32(max_union_windows))
            mUnionCounts[group_id] = union_count
            mOutReqIdx[group_id] = Int32(0)
            mWorkQGlobal[group_id] = q_global_base
            mWorkQLocal[group_id] = q_base
            mWorkQLen[group_id] = q_base + q_tile_count

        if cutlass.const_expr(emit_causal_limits):
            if tx < Int32(q_group):
                slot = tx
                q_local_s = q_base + slot
                q_pos_s = q_global_base + slot
                q_region = q_pos_s // Int32(8)
                q_word = q_region // Int32(32)
                q_bit = q_region - q_word * Int32(32)
                before_words = Int32(0)
                current_word = Int32(0)
                if q_word > Int32(0):
                    if q_word <= Int32(bit_words_width):
                        before_words = sPrefix[q_word - Int32(1)]
                if q_word < Int32(bit_words_width):
                    current_word = sWords[q_word]
                current_bit_mask = Int32(1) << q_bit
                lower_mask = current_bit_mask - Int32(1)
                before_in_word = _popc_i32(current_word & lower_mask)
                has_current = (current_word & current_bit_mask) != Int32(0)
                limit = (before_words + before_in_word) * Int32(8)
                if has_current:
                    limit = limit + Int32(8)
                if q_local_s >= total_q:
                    limit = Int32(0)
                if cutlass.const_expr(explicit_work_ranges):
                    if q_local_s >= q_len:
                        limit = Int32(0)
                mCausalLimits[group_id, slot] = limit

    return _grouped_union_logical_bitset_compact_kernel


def _make_launch_grouped_union_logical_bitset_fill_kernel(
    *,
    q_group: int,
    topk_windows: int,
    topk_width: int,
    num_region_bins: int,
    bit_words: int,
    bit_words_width: int,
    explicit_work_ranges: bool,
):
    kernel = _make_grouped_union_logical_bitset_fill_kernel(
        q_group=q_group,
        topk_windows=topk_windows,
        topk_width=topk_width,
        num_region_bins=num_region_bins,
        bit_words=bit_words,
        bit_words_width=bit_words_width,
        explicit_work_ranges=explicit_work_ranges,
    )

    @cute.jit
    def _launch_grouped_union_logical_bitset_fill_kernel(
        mRegionLogical: cute.Tensor,
        mRegionPhys: cute.Tensor,
        mRegionCounts: cute.Tensor,
        mBitset: cute.Tensor,
        mWorkQLocal: cute.Tensor,
        mWorkQLen: cute.Tensor,
        total_groups: int,
        total_q: int,
        stream,
    ):
        kernel(
            mRegionLogical,
            mRegionPhys,
            mRegionCounts,
            mBitset,
            mWorkQLocal,
            mWorkQLen,
            Int32(total_q),
        ).launch(
            grid=[total_groups, 1, 1],
            block=[_UNION_FILL_THREADS, 1, 1],
            stream=stream,
        )

    return _launch_grouped_union_logical_bitset_fill_kernel


def _make_launch_grouped_union_logical_bitset_compact_kernel(
    *,
    q_group: int,
    max_union_windows: int,
    num_region_bins: int,
    bit_words: int,
    bit_words_width: int,
    topk_windows: int,
    topk_width: int,
    emit_causal_limits: bool,
    emit_exact_mask: bool,
    explicit_work_ranges: bool,
):
    kernel = _make_grouped_union_logical_bitset_compact_kernel(
        q_group=q_group,
        max_union_windows=max_union_windows,
        num_region_bins=num_region_bins,
        bit_words=bit_words,
        bit_words_width=bit_words_width,
        topk_windows=topk_windows,
        topk_width=topk_width,
        emit_causal_limits=emit_causal_limits,
        emit_exact_mask=emit_exact_mask,
        explicit_work_ranges=explicit_work_ranges,
    )

    @cute.jit
    def _launch_grouped_union_logical_bitset_compact_kernel(
        mRegionLogical: cute.Tensor,
        mRegionPhys: cute.Tensor,
        mRegionCounts: cute.Tensor,
        mBitset: cute.Tensor,
        mOutReqIdx: cute.Tensor,
        mWorkQGlobal: cute.Tensor,
        mWorkQLocal: cute.Tensor,
        mWorkQLen: cute.Tensor,
        mUnionPhys: cute.Tensor,
        mUnionLogical: cute.Tensor,
        mUnionCounts: cute.Tensor,
        mCausalLimits: cute.Tensor,
        mExactMaskBits: cute.Tensor,
        total_groups: int,
        total_q: int,
        q_global_offset: int,
        stream,
    ):
        kernel(
            mRegionLogical,
            mRegionPhys,
            mRegionCounts,
            mBitset,
            mOutReqIdx,
            mWorkQGlobal,
            mWorkQLocal,
            mWorkQLen,
            mUnionPhys,
            mUnionLogical,
            mUnionCounts,
            mCausalLimits,
            mExactMaskBits,
            Int32(total_q),
            Int32(q_global_offset),
        ).launch(
            grid=[total_groups, 1, 1],
            block=[_UNION_COMPACT_THREADS, 1, 1],
            stream=stream,
        )

    return _launch_grouped_union_logical_bitset_compact_kernel


def _make_union_compile_placeholders(
    topk_windows: int,
    bit_words: int,
    max_union_windows: int,
    q_group: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((1, int(topk_windows)), dtype=torch.int32, device=device),
        torch.empty((1, int(topk_windows)), dtype=torch.int32, device=device),
        torch.empty((1,), dtype=torch.int32, device=device),
        torch.empty((1, int(bit_words)), dtype=torch.int32, device=device),
        torch.empty((1,), dtype=torch.int32, device=device),
        torch.empty((1,), dtype=torch.int32, device=device),
        torch.empty((1,), dtype=torch.int32, device=device),
        torch.empty((1,), dtype=torch.int32, device=device),
        torch.empty((1, int(max_union_windows)), dtype=torch.int32, device=device),
        torch.empty((1, int(max_union_windows)), dtype=torch.int32, device=device),
        torch.empty((1,), dtype=torch.int32, device=device),
        torch.empty((1, int(q_group)), dtype=torch.int32, device=device),
        torch.empty((1, int(max_union_windows)), dtype=torch.int32, device=device),
    )


@cached_compile_function
def _get_compiled_union_fill_kernel_for_shape(
    q_group: int,
    topk_windows: int,
    topk_width: int,
    num_region_bins: int,
    bit_words: int,
    bit_words_width: int,
    explicit_work_ranges: bool,
    device_key: tuple[str, int | None],
) -> cute.JitFunction:
    device = utils.device_from_cache_key(device_key)
    placeholders = _make_union_compile_placeholders(
        topk_windows=topk_windows,
        bit_words=bit_words,
        max_union_windows=1,
        q_group=q_group,
        device=device,
    )
    (
        p_logical,
        p_phys,
        p_counts,
        p_bitset,
        _p_out_req_idx,
        _p_work_q_global,
        p_work_q_local,
        p_work_q_len,
        *_,
    ) = placeholders
    launch = _make_launch_grouped_union_logical_bitset_fill_kernel(
        q_group=q_group,
        topk_windows=topk_windows,
        topk_width=topk_width,
        num_region_bins=num_region_bins,
        bit_words=bit_words,
        bit_words_width=bit_words_width,
        explicit_work_ranges=explicit_work_ranges,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        launch,
        utils.make_fake_tensor_like_with_dynamic_dim(p_logical, alignment=16, dynamic_shape_dim=0),
        utils.make_fake_tensor_like_with_dynamic_dim(p_phys, alignment=16, dynamic_shape_dim=0),
        utils.make_fake_tensor_like_with_dynamic_dim(p_counts, alignment=16, dynamic_shape_dim=0),
        utils.make_fake_tensor_like_with_dynamic_dim(p_bitset, alignment=16, dynamic_shape_dim=0),
        utils.make_fake_tensor_like_with_dynamic_dim(p_work_q_local, alignment=16, dynamic_shape_dim=0),
        utils.make_fake_tensor_like_with_dynamic_dim(p_work_q_len, alignment=16, dynamic_shape_dim=0),
        1,
        1,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


def _get_compiled_union_fill_kernel(
    *,
    logical_sq: torch.Tensor,
    q_group: int,
    topk_windows: int,
    topk_width: int,
    num_region_bins: int,
    bit_words: int,
    bit_words_width: int,
    explicit_work_ranges: bool,
) -> cute.JitFunction:
    return _get_compiled_union_fill_kernel_for_shape(
        int(q_group),
        int(topk_windows),
        int(topk_width),
        int(num_region_bins),
        int(bit_words),
        int(bit_words_width),
        bool(explicit_work_ranges),
        utils.device_cache_key(logical_sq.device),
    )


@cached_compile_function
def _get_compiled_union_compact_kernel_for_shape(
    q_group: int,
    topk_windows: int,
    topk_width: int,
    max_union_windows: int,
    num_region_bins: int,
    bit_words: int,
    bit_words_width: int,
    emit_causal_limits: bool,
    emit_exact_mask: bool,
    explicit_work_ranges: bool,
    device_key: tuple[str, int | None],
) -> cute.JitFunction:
    device = utils.device_from_cache_key(device_key)
    placeholders = _make_union_compile_placeholders(
        topk_windows=topk_windows,
        bit_words=bit_words,
        max_union_windows=max_union_windows,
        q_group=q_group,
        device=device,
    )
    (
        p_logical,
        p_phys,
        p_counts,
        p_bitset,
        p_out_req_idx,
        p_work_q_global,
        p_work_q_local,
        p_work_q_len,
        p_union_phys,
        p_union_logical,
        p_union_counts,
        p_causal_limits,
        p_exact_mask_bits,
    ) = placeholders
    if not emit_exact_mask:
        p_union_logical = torch.empty((1, 1), dtype=torch.int32, device=device)
        p_exact_mask_bits = torch.empty((1, 1), dtype=torch.int32, device=device)
    launch = _make_launch_grouped_union_logical_bitset_compact_kernel(
        q_group=q_group,
        max_union_windows=max_union_windows,
        num_region_bins=num_region_bins,
        bit_words=bit_words,
        bit_words_width=bit_words_width,
        topk_windows=topk_windows,
        topk_width=topk_width,
        emit_causal_limits=emit_causal_limits,
        emit_exact_mask=emit_exact_mask,
        explicit_work_ranges=explicit_work_ranges,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        launch,
        utils.make_fake_tensor_like_with_dynamic_dim(p_logical, alignment=16, dynamic_shape_dim=0),
        utils.make_fake_tensor_like_with_dynamic_dim(p_phys, alignment=16, dynamic_shape_dim=0),
        utils.make_fake_tensor_like_with_dynamic_dim(p_counts, alignment=16, dynamic_shape_dim=0),
        utils.make_fake_tensor_like_with_dynamic_dim(p_bitset, alignment=16, dynamic_shape_dim=0),
        utils.make_fake_tensor_like_with_dynamic_dim(p_out_req_idx, alignment=16, dynamic_shape_dim=0),
        utils.make_fake_tensor_like_with_dynamic_dim(p_work_q_global, alignment=16, dynamic_shape_dim=0),
        utils.make_fake_tensor_like_with_dynamic_dim(p_work_q_local, alignment=16, dynamic_shape_dim=0),
        utils.make_fake_tensor_like_with_dynamic_dim(p_work_q_len, alignment=16, dynamic_shape_dim=0),
        utils.make_fake_tensor_like_with_dynamic_dim(p_union_phys, alignment=16, dynamic_shape_dim=0),
        utils.make_fake_tensor_like_with_dynamic_dim(p_union_logical, alignment=16, dynamic_shape_dim=0),
        utils.make_fake_tensor_like_with_dynamic_dim(p_union_counts, alignment=16, dynamic_shape_dim=0),
        utils.make_fake_tensor_like_with_dynamic_dim(p_causal_limits, alignment=16, dynamic_shape_dim=0),
        utils.make_fake_tensor_like_with_dynamic_dim(p_exact_mask_bits, alignment=16, dynamic_shape_dim=0),
        1,
        1,
        0,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


def _get_compiled_union_compact_kernel(
    *,
    bitset: torch.Tensor,
    q_group: int,
    topk_windows: int,
    topk_width: int,
    max_union_windows: int,
    num_region_bins: int,
    bit_words: int,
    bit_words_width: int,
    emit_causal_limits: bool,
    emit_exact_mask: bool,
    explicit_work_ranges: bool,
) -> cute.JitFunction:
    return _get_compiled_union_compact_kernel_for_shape(
        int(q_group),
        int(topk_windows),
        int(topk_width),
        int(max_union_windows),
        int(num_region_bins),
        int(bit_words),
        int(bit_words_width),
        bool(emit_causal_limits),
        bool(emit_exact_mask),
        bool(explicit_work_ranges),
        utils.device_cache_key(bitset.device),
    )


def _launch_grouped_union_logical_bitset_cutedsl(
    *,
    logical_sq: torch.Tensor,
    phys_sq: torch.Tensor,
    counts_sq: torch.Tensor,
    bitset: torch.Tensor,
    out_req_idx: torch.Tensor,
    work_q_global: torch.Tensor,
    work_q_local: torch.Tensor,
    work_q_len: torch.Tensor,
    union_phys: torch.Tensor,
    union_logical: torch.Tensor,
    union_counts: torch.Tensor,
    causal_limits: torch.Tensor,
    exact_mask_bits: torch.Tensor,
    total_groups: int,
    total_q: int,
    q_global_offset: int,
    q_group: int,
    topk_windows: int,
    topk_width: int,
    max_union_windows: int,
    num_region_bins: int,
    bit_words: int,
    bit_words_width: int,
    emit_causal_limits: bool,
    emit_exact_mask: bool,
    explicit_work_ranges: bool,
) -> None:
    fill = _get_compiled_union_fill_kernel(
        logical_sq=logical_sq,
        q_group=q_group,
        topk_windows=topk_windows,
        topk_width=topk_width,
        num_region_bins=num_region_bins,
        bit_words=bit_words,
        bit_words_width=bit_words_width,
        explicit_work_ranges=explicit_work_ranges,
    )
    compact = _get_compiled_union_compact_kernel(
        bitset=bitset,
        q_group=q_group,
        topk_windows=topk_windows,
        topk_width=topk_width,
        max_union_windows=max_union_windows,
        num_region_bins=num_region_bins,
        bit_words=bit_words,
        bit_words_width=bit_words_width,
        emit_causal_limits=emit_causal_limits,
        emit_exact_mask=emit_exact_mask,
        explicit_work_ranges=explicit_work_ranges,
    )
    fill(
        logical_sq,
        phys_sq,
        counts_sq,
        bitset,
        work_q_local,
        work_q_len,
        int(total_groups),
        int(total_q),
    )
    compact(
        logical_sq,
        phys_sq,
        counts_sq,
        bitset,
        out_req_idx,
        work_q_global,
        work_q_local,
        work_q_len,
        union_phys,
        union_logical,
        union_counts,
        causal_limits,
        exact_mask_bits,
        int(total_groups),
        int(total_q),
        int(q_global_offset),
    )


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (int(x) - 1).bit_length()


def build_single_req_grouped_union_sparse_work_queue_gqa(
    *,
    total_q: int,
    q_group: int,
    region_counts: torch.Tensor,
    region_phys_indices: torch.Tensor,
    region_indices: torch.Tensor,
    max_union_windows: int | None = None,
    num_region_bins: int | None = None,
    return_exact_mask: bool = False,
    q_global_offset: int = 0,
    out_req_idx: torch.Tensor | None = None,
    work_q_global: torch.Tensor | None = None,
    work_q_local: torch.Tensor | None = None,
    work_q_len: torch.Tensor | None = None,
    union_counts: torch.Tensor | None = None,
    union_phys: torch.Tensor | None = None,
    union_logical: torch.Tensor | None = None,
    causal_limits: torch.Tensor | None = None,
    exact_mask_bits: torch.Tensor | None = None,
    bitset: torch.Tensor | None = None,
    total_groups: int | None = None,
    explicit_work_ranges: bool = False,
) -> tuple[torch.Tensor, ...]:
    if q_group not in (8, 16, 32):
        raise ValueError(f"GQA grouped-union prefill supports q_group in (8, 16, 32), got {q_group}")
    if total_q < 0:
        raise ValueError(f"total_q must be >= 0, got {total_q}")
    if region_phys_indices.device.type != "cuda" or region_counts.device.type != "cuda":
        raise ValueError("region_counts and region_phys_indices must be CUDA tensors")
    if region_indices is None:
        raise ValueError("GQA grouped-union preprocess requires logical region_indices")
    device = region_phys_indices.device
    if region_counts.device != device:
        raise ValueError("region_counts and region_phys_indices must live on the same CUDA device")
    if region_indices.device != device:
        raise ValueError("region_indices must live on the same CUDA device as region_phys_indices")

    if region_phys_indices.ndim != 2:
        raise ValueError(
            f"region_phys_indices must have shape [Tq, topk_regions], got {tuple(region_phys_indices.shape)}"
        )
    topk_windows = int(region_phys_indices.shape[1])
    if region_counts.ndim != 1:
        raise ValueError(f"region_counts must have shape [Tq], got {tuple(region_counts.shape)}")
    if region_indices.ndim != 2:
        raise ValueError(
            f"region_indices must have shape [Tq, topk_regions], got {tuple(region_indices.shape)}"
        )
    if int(region_phys_indices.shape[0]) != int(total_q) or int(region_counts.shape[0]) != int(total_q):
        raise ValueError("sparse metadata first dimension must match total_q")
    if int(region_indices.shape[1]) != int(topk_windows):
        raise ValueError(
            "region_indices topk dimension must match region_phys_indices, "
            f"got {int(region_indices.shape[1])} vs {topk_windows}"
        )
    if int(region_indices.shape[0]) != int(total_q):
        raise ValueError("region_indices first dimension must match total_q")
    if not region_phys_indices.is_contiguous():
        raise ValueError(
            f"region_phys_indices must be contiguous, got stride={tuple(region_phys_indices.stride())}"
        )
    if not region_indices.is_contiguous():
        raise ValueError(f"region_indices must be contiguous, got stride={tuple(region_indices.stride())}")
    if not region_counts.is_contiguous():
        raise ValueError(f"region_counts must be contiguous, got stride={tuple(region_counts.stride())}")

    if topk_windows <= 0:
        raise ValueError("GQA grouped-union metadata requires topk_windows > 0")

    if explicit_work_ranges:
        if total_groups is None:
            raise ValueError("explicit grouped-union work ranges require total_groups")
        total_groups = int(total_groups)
        if total_groups < 0:
            raise ValueError(f"total_groups must be >= 0, got {total_groups}")
    else:
        total_groups = (int(total_q) + int(q_group) - 1) // int(q_group)

    if num_region_bins is None:
        raise ValueError(
            "GQA grouped-union prefill requires explicit "
            "num_region_bins for token-independent JIT caching; inferring "
            f"from live total_q={int(total_q)} would make max_union_windows "
            "depend on the request length."
        )
    num_region_bins = _next_power_of_2(int(num_region_bins))
    if num_region_bins <= 0:
        raise ValueError("num_region_bins must be positive")

    max_possible_union = min(int(q_group) * topk_windows, int(num_region_bins))
    max_union_windows = int(max_union_windows) if max_union_windows is not None else max_possible_union
    if max_union_windows < max_possible_union:
        raise ValueError(
            "max_union_windows is too small for the lossless grouped-union contract, "
            f"got {max_union_windows} < {max_possible_union}"
        )
    def take_output(
        name: str,
        output: torch.Tensor | None,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        if output is None:
            return torch.empty(shape, dtype=torch.int32, device=device)
        if (
            output.device != device
            or output.dtype != torch.int32
            or not output.is_contiguous()
            or tuple(int(v) for v in output.shape) != shape
        ):
            raise ValueError(f"{name} must be contiguous CUDA int32 shape {shape}")
        return output

    out_req_idx = take_output("out_req_idx", out_req_idx, (total_groups,))
    work_q_global = take_output("work_q_global", work_q_global, (total_groups,))
    work_q_local = take_output("work_q_local", work_q_local, (total_groups,))
    work_q_len = take_output("work_q_len", work_q_len, (total_groups,))
    union_counts = take_output("union_counts", union_counts, (total_groups,))
    union_phys = take_output(
        "union_phys", union_phys, (total_groups, int(max_union_windows))
    )
    if return_exact_mask:
        union_logical = take_output(
            "union_logical", union_logical,
            (total_groups, int(max_union_windows)),
        )
        exact_mask_bits = take_output(
            "exact_mask_bits", exact_mask_bits,
            (total_groups, int(max_union_windows)),
        )
    else:
        union_logical = torch.empty((1, 1), dtype=torch.int32, device=device)
        exact_mask_bits = torch.empty((1, 1), dtype=torch.int32, device=device)
    causal_limits = take_output(
        "causal_limits", causal_limits, (total_groups, int(q_group))
    )
    if total_groups == 0:
        if return_exact_mask:
            return (
                out_req_idx,
                work_q_global,
                work_q_local,
                work_q_len,
                union_counts,
                union_phys,
                causal_limits,
                union_logical,
                exact_mask_bits,
            )
        return out_req_idx, work_q_global, work_q_local, work_q_len, union_counts, union_phys, causal_limits

    bit_words = (int(num_region_bins) + 31) // 32
    bit_words_width = _next_power_of_2(bit_words)
    bitset = take_output("bitset", bitset, (total_groups, bit_words))
    _launch_grouped_union_logical_bitset_cutedsl(
        logical_sq=region_indices,
        phys_sq=region_phys_indices,
        counts_sq=region_counts,
        bitset=bitset,
        out_req_idx=out_req_idx,
        work_q_global=work_q_global,
        work_q_local=work_q_local,
        work_q_len=work_q_len,
        union_phys=union_phys,
        union_logical=union_logical,
        union_counts=union_counts,
        causal_limits=causal_limits,
        exact_mask_bits=exact_mask_bits,
        total_groups=int(total_groups),
        total_q=int(total_q),
        q_global_offset=int(q_global_offset),
        q_group=int(q_group),
        topk_windows=int(topk_windows),
        topk_width=_next_power_of_2(int(topk_windows)),
        max_union_windows=int(max_union_windows),
        num_region_bins=int(num_region_bins),
        bit_words=int(bit_words),
        bit_words_width=int(bit_words_width),
        emit_causal_limits=True,
        emit_exact_mask=bool(return_exact_mask),
        explicit_work_ranges=bool(explicit_work_ranges),
    )
    if return_exact_mask:
        return (
            out_req_idx,
            work_q_global,
            work_q_local,
            work_q_len,
            union_counts,
            union_phys,
            causal_limits,
            union_logical,
            exact_mask_bits,
        )
    return out_req_idx, work_q_global, work_q_local, work_q_len, union_counts, union_phys, causal_limits


def build_grouped_union_sparse_work_queue_gqa(
    *,
    total_q: int,
    total_groups: int,
    q_group: int,
    region_counts: torch.Tensor,
    region_phys_indices: torch.Tensor,
    region_indices: torch.Tensor,
    work_q_global: torch.Tensor,
    work_q_local: torch.Tensor,
    work_q_len: torch.Tensor,
    max_union_windows: int | None = None,
    num_region_bins: int | None = None,
    return_exact_mask: bool = False,
    out_req_idx: torch.Tensor | None = None,
    union_counts: torch.Tensor | None = None,
    union_phys: torch.Tensor | None = None,
    union_logical: torch.Tensor | None = None,
    causal_limits: torch.Tensor | None = None,
    exact_mask_bits: torch.Tensor | None = None,
    bitset: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    return build_single_req_grouped_union_sparse_work_queue_gqa(
        total_q=total_q,
        total_groups=total_groups,
        q_group=q_group,
        region_counts=region_counts,
        region_phys_indices=region_phys_indices,
        region_indices=region_indices,
        max_union_windows=max_union_windows,
        num_region_bins=num_region_bins,
        return_exact_mask=return_exact_mask,
        out_req_idx=out_req_idx,
        work_q_global=work_q_global,
        work_q_local=work_q_local,
        work_q_len=work_q_len,
        union_counts=union_counts,
        union_phys=union_phys,
        union_logical=union_logical,
        causal_limits=causal_limits,
        exact_mask_bits=exact_mask_bits,
        bitset=bitset,
        explicit_work_ranges=True,
    )


__all__ = [
    "build_grouped_union_sparse_work_queue_gqa",
    "build_single_req_grouped_union_sparse_work_queue_gqa",
]
