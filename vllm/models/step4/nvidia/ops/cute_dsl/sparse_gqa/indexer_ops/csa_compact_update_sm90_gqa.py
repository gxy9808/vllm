# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import dsl_user_op

from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.kernel_utils import cute_utils as utils
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.kernel_utils.fp_compat import cvt_f32_to_e4m3
from vllm.models.step4.nvidia.ops.cute_dsl.utils import elem_pointer


_THREADS_PER_BLOCK = 256
_WARPS_PER_BLOCK = _THREADS_PER_BLOCK // 32
_MAX_PROXY_DIM = 1024
_MAX_DIMS_PER_THREAD = _MAX_PROXY_DIM // _THREADS_PER_BLOCK
_MAX_REGION_BLOCK_SIZE = 32
_MAX_DECODE_PROXY_DIM = 1024
_MAX_WARPS_PER_BLOCK = 32
_DECODE_STAGE_TOKENS = 8
_STAGE_THREADS_PER_BLOCK = 256
_MAX_STAGE_DIMS_PER_THREAD = _MAX_DECODE_PROXY_DIM // _STAGE_THREADS_PER_BLOCK
_LOG2_E = 1.4426950408889634
_NEG_INF = -3.4028234663852886e38


@cute.jit
def _wgmma_sw128_physical_offset(
    logical_dim: Int32,
    logical_region: Int64,
    page_slot: Int32,
) -> Int32:
    atom = logical_dim // Int32(128)
    atom_dim = logical_dim - atom * Int32(128)
    chunk = atom_dim // Int32(16)
    byte = atom_dim - chunk * Int32(16)
    # Keep one complete 8-row WGMMA swizzle period contiguous by atom.  This
    # is the physical contract consumed by the paged FP8 TMA path: atom 0 for
    # all eight rows is followed by atom 1, so the two atom streams can be
    # copied as one 8-row tile without a gather.
    row_in_block = page_slot - (page_slot // Int32(8)) * Int32(8)
    block_in_page = page_slot // Int32(8)
    row_swizzle = Int32(logical_region % Int64(8))
    return (
        block_in_page * Int32(8 * 256)
        + atom * Int32(8 * 128)
        + row_in_block * Int32(128)
        + (chunk ^ row_swizzle) * Int32(16)
        + byte
    )


def _decode_cluster_config(head_dim: int) -> tuple[int, int]:
    threads_per_block = ((int(head_dim) + 31) // 32) * 32
    return 1, threads_per_block


@dsl_user_op
def _atomic_cas_i64(
    ptr: cute.Pointer,
    compare: Int64,
    value: Int64,
    *,
    loc=None,
    ip=None,
) -> Int64:
    return Int64(
        nvvm.atomicrmw(
            op=nvvm.AtomicOpKind.CAS,
            ptr=ptr.llvm_ptr,
            a=Int64(compare).ir_value(loc=loc, ip=ip),
            b=Int64(value).ir_value(loc=loc, ip=ip),
        )
    )


@dsl_user_op
def _atomic_cas_i32(
    ptr: cute.Pointer,
    compare: Int32,
    value: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    return Int32(
        nvvm.atomicrmw(
            op=nvvm.AtomicOpKind.CAS,
            ptr=ptr.llvm_ptr,
            a=Int32(compare).ir_value(loc=loc, ip=ip),
            b=Int32(value).ir_value(loc=loc, ip=ip),
        )
    )


@cute.jit
def _min_i32(a: Int32, b: Int32) -> Int32:
    return cutlass.select_(a < b, a, b)


@cute.jit
def _find_active_slot_parallel(
    mActiveRegionIds: cute.Tensor,
    region: Int64,
    active_capacity: Int32,
    sWarpSlots: cute.Tensor,
) -> Int32:
    tidx, _, _ = cute.arch.thread_idx()
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    local_slot = active_capacity
    slot = Int32(tidx)
    while slot < active_capacity:
        if Int64(mActiveRegionIds[slot]) == region:
            local_slot = _min_i32(local_slot, slot)
        slot += Int32(_THREADS_PER_BLOCK)
    local_slot = utils.warp_reduce(local_slot, _min_i32)
    if lane_idx == Int32(0):
        sWarpSlots[warp_idx] = local_slot
    cute.arch.barrier()
    found_slot = active_capacity
    if lane_idx < Int32(_WARPS_PER_BLOCK):
        found_slot = sWarpSlots[lane_idx]
    found_slot = utils.warp_reduce(found_slot, _min_i32)
    cute.arch.barrier()
    return found_slot


@cute.kernel
def _csa_compact_update_kernel_sm90_gqa_hash_cta256(
    mSum: cute.Tensor,
    mCount: cute.Tensor,
    mActiveRegionIds: cute.Tensor,
    mDenominator: cute.Tensor,
    mMaxLogits: cute.Tensor,
    mFlatSlot: cute.Tensor,
    mResetSlots: cute.Tensor,
    mTokenValidU8: cute.Tensor,
    mTokenPositions: cute.Tensor,
    mIndexK: cute.Tensor,
    mIndexZ: cute.Tensor,
    source_rows: Int32,
    total_regions: Int64,
    active_capacity: Int32,
    num_kv_heads: cutlass.Constexpr[int],
    head_dim: cutlass.Constexpr[int],
    summaries_per_page: cutlass.Constexpr[int],
    region_block_size: cutlass.Constexpr[int],
):
    row, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    smem = cutlass.utils.SmemAllocator()
    sRows = smem.allocate_tensor(
        Int32,
        cute.make_layout((_MAX_REGION_BLOCK_SIZE,), stride=(1,)),
        byte_alignment=16,
    )
    sFlags = smem.allocate_tensor(
        Int32,
        cute.make_layout((8,), stride=(1,)),
        byte_alignment=16,
    )
    sRegion = smem.allocate_tensor(
        Int64,
        cute.make_layout((1,), stride=(1,)),
        byte_alignment=8,
    )
    sWarpSlots = smem.allocate_tensor(
        Int32,
        cute.make_layout((_WARPS_PER_BLOCK,), stride=(1,)),
        byte_alignment=16,
    )
    sWarpMax = smem.allocate_tensor(
        Float32,
        cute.make_layout((_WARPS_PER_BLOCK,), stride=(1,)),
        byte_alignment=16,
    )
    sSharedMax = smem.allocate_tensor(
        Float32,
        cute.make_layout((1,), stride=(1,)),
        byte_alignment=16,
    )
    if Int32(tidx) < Int32(region_block_size):
        sRows[Int32(tidx)] = Int32(-1)
    if Int32(tidx) < Int32(8):
        sFlags[Int32(tidx)] = Int32(0)
    cute.arch.barrier()

    if Int32(tidx) == Int32(0):
        region = Int64(mFlatSlot[row])
        if (
            Int32(mTokenValidU8[row]) != Int32(0)
            and region >= Int64(0)
            and region < total_regions
        ):
            sRegion[0] = region
            sFlags[0] = Int32(1)
            sFlags[6] = Int32(region // Int64(summaries_per_page))
            sFlags[7] = Int32(region % Int64(summaries_per_page))
    cute.arch.barrier()

    if sFlags[0] != Int32(0):
        region = sRegion[0]
        candidate = Int32(tidx)
        while candidate < source_rows:
            if (
                Int32(mTokenValidU8[candidate]) != Int32(0)
                and Int64(mFlatSlot[candidate]) == region
            ):
                if candidate < Int32(row):
                    sFlags[0] = Int32(0)
                position = Int64(mTokenPositions[candidate])
                offset = Int32(position % Int64(region_block_size))
                if offset >= Int32(0) and offset < Int32(region_block_size):
                    old = sRows[offset]
                    while old < Int32(0) or candidate < old:
                        observed = _atomic_cas_i32(
                            elem_pointer(sRows, (offset,)), old, candidate
                        )
                        if observed == old:
                            old = candidate
                        else:
                            old = observed
                    if offset == Int32(region_block_size - 1):
                        sFlags[3] = Int32(1)
                if Int64(mResetSlots[candidate]) == region:
                    sFlags[2] = Int32(1)
            candidate += Int32(_THREADS_PER_BLOCK)
    cute.arch.barrier()

    if sFlags[0] != Int32(0):
        region = sRegion[0]
        if Int32(tidx) == Int32(0):
            found_slot = active_capacity
            first_empty = active_capacity
            slot = Int32(region % Int64(active_capacity))
            probe = Int32(0)
            while probe < active_capacity and found_slot == active_capacity:
                active_region = Int64(mActiveRegionIds[slot])
                if active_region == region:
                    found_slot = slot
                elif active_region == Int64(-1):
                    first_empty = slot
                    probe = active_capacity
                else:
                    slot += Int32(1)
                    if slot == active_capacity:
                        slot = Int32(0)
                probe += Int32(1)
            sFlags[1] = found_slot
            sFlags[4] = first_empty
            sFlags[5] = Int32(found_slot == active_capacity)
        cute.arch.barrier()

        if sFlags[5] != Int32(0):
            # Preserve compatibility with state created by the former linear
            # allocator, whose region may live beyond the first hash-table hole.
            found_slot = _find_active_slot_parallel(
                mActiveRegionIds,
                region,
                active_capacity,
                sWarpSlots,
            )
            if Int32(tidx) == Int32(0):
                sFlags[1] = found_slot
            cute.arch.barrier()

        found_slot = sFlags[1]
        if found_slot == active_capacity and sFlags[3] == Int32(0):
            if Int32(tidx) == Int32(0):
                slot = sFlags[4]
                if slot == active_capacity:
                    slot = Int32(region % Int64(active_capacity))
                probe = Int32(0)
                while probe < active_capacity and found_slot == active_capacity:
                    old = _atomic_cas_i64(
                        elem_pointer(mActiveRegionIds, (slot,)),
                        Int64(-1),
                        region,
                    )
                    if old == Int64(-1) or old == region:
                        found_slot = slot
                    else:
                        slot += Int32(1)
                        if slot == active_capacity:
                            slot = Int32(0)
                    probe += Int32(1)
                sFlags[1] = found_slot
            cute.arch.barrier()
        found_slot = sFlags[1]
        reset = sFlags[2] != Int32(0)
        complete = sFlags[3] != Int32(0)
        summary_page = sFlags[6]
        summary_fragment = sFlags[7]

        head = Int32(0)
        while head < num_kv_heads:
            numerators = [Float32(0.0) for _ in range(_MAX_DIMS_PER_THREAD)]
            denominators = [Float32(0.0) for _ in range(_MAX_DIMS_PER_THREAD)]
            if Int32(tidx) == Int32(0):
                previous_max = Float32(_NEG_INF)
                if found_slot < active_capacity and not reset:
                    previous_max = Float32(mMaxLogits[found_slot, head])
                sSharedMax[Int32(0)] = previous_max
            cute.arch.barrier()
            previous_max = sSharedMax[Int32(0)]
            for part in cutlass.range_constexpr(_MAX_DIMS_PER_THREAD):
                d = Int32(tidx) + Int32(part * _THREADS_PER_BLOCK)
                if d < Int32(head_dim):
                    denominator = Float32(0.0)
                    if found_slot < active_capacity and not reset:
                        denominator = Float32(mDenominator[found_slot, head, d])
                    denominators[part] = denominator
                    numerators[part] = (
                        Float32(mSum[summary_page, summary_fragment, head, d])
                        * denominator
                    )

            local_max = Float32(_NEG_INF)
            for offset in range(region_block_size):
                matched_row = sRows[offset]
                if Int32(offset) < region_block_size and matched_row >= Int32(0):
                    for part in cutlass.range_constexpr(_MAX_DIMS_PER_THREAD):
                        d = Int32(tidx) + Int32(part * _THREADS_PER_BLOCK)
                        if d < Int32(head_dim):
                            local_max = cute.arch.fmax(
                                local_max,
                                Float32(mIndexZ[matched_row, head, d]),
                            )

            lane_idx = cute.arch.lane_idx()
            warp_idx = cute.arch.warp_idx()
            warp_max = utils.warp_reduce(local_max, cute.arch.fmax, width=32)
            if lane_idx == Int32(0):
                sWarpMax[warp_idx] = warp_max
            cute.arch.barrier()
            if Int32(tidx) == Int32(0):
                shared_max = sWarpMax[Int32(0)]
                for warp in cutlass.range_constexpr(1, _WARPS_PER_BLOCK):
                    shared_max = cute.arch.fmax(shared_max, sWarpMax[warp])
                sSharedMax[Int32(0)] = shared_max
            cute.arch.barrier()
            shared_max = sSharedMax[Int32(0)]

            for part in cutlass.range_constexpr(_MAX_DIMS_PER_THREAD):
                d = Int32(tidx) + Int32(part * _THREADS_PER_BLOCK)
                if d < Int32(head_dim):
                    old_scale = Float32(0.0)
                    if denominators[part] > Float32(0.0):
                        old_scale = cute.math.exp2(
                            (previous_max - shared_max) * Float32(_LOG2_E),
                            fastmath=True,
                        )
                    numerators[part] *= old_scale
                    denominators[part] *= old_scale

            for offset in range(region_block_size):
                matched_row = sRows[offset]
                if Int32(offset) < region_block_size and matched_row >= Int32(0):
                    for part in cutlass.range_constexpr(_MAX_DIMS_PER_THREAD):
                        d = Int32(tidx) + Int32(part * _THREADS_PER_BLOCK)
                        if d < Int32(head_dim):
                            logit = Float32(mIndexZ[matched_row, head, d])
                            weight = cute.math.exp2(
                                (logit - shared_max) * Float32(_LOG2_E),
                                fastmath=True,
                            )
                            value = Float32(mIndexK[matched_row, head, d])
                            numerators[part] += weight * value
                            denominators[part] += weight
            for part in cutlass.range_constexpr(_MAX_DIMS_PER_THREAD):
                d = Int32(tidx) + Int32(part * _THREADS_PER_BLOCK)
                if d < Int32(head_dim):
                    denominator = denominators[part]
                    output = Float32(0.0)
                    if denominator > Float32(0.0):
                        output = numerators[part] / cute.arch.fmax(
                            denominator,
                            Float32(1.0e-20),
                        )
                    mSum[summary_page, summary_fragment, head, d] = mSum.element_type(
                        output
                    )
                    if complete:
                        if found_slot < active_capacity:
                            mDenominator[found_slot, head, d] = Float32(0.0)
                    elif found_slot < active_capacity:
                        mDenominator[found_slot, head, d] = denominator
            if found_slot < active_capacity and Int32(tidx) == Int32(0):
                if complete:
                    mMaxLogits[found_slot, head] = Float32(float("-inf"))
                else:
                    mMaxLogits[found_slot, head] = shared_max
            if Int32(tidx) == Int32(0):
                mCount[summary_page, summary_fragment, head] = Float32(1.0)
            head += Int32(1)
            if head < num_kv_heads:
                cute.arch.barrier()

        if Int32(tidx) == Int32(0) and complete and found_slot < active_capacity:
            mActiveRegionIds[found_slot] = Int64(-1)


@cute.kernel
def _csa_compact_prefill_task_prefix_kernel_sm90_gqa(
    mQueryStartLoc: cute.Tensor,
    mSeqLens: cute.Tensor,
    mTaskPrefix: cute.Tensor,
    num_reqs: Int32,
    region_block_size: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == Int32(0):
        req = Int32(0)
        prefix = Int32(0)
        mTaskPrefix[0] = prefix
        while req < num_reqs:
            q_start = Int32(mQueryStartLoc[req])
            q_end = Int32(mQueryStartLoc[req + Int32(1)])
            q_len = q_end - q_start
            context_len = Int32(mSeqLens[req]) - q_len
            context_offset = context_len % Int32(region_block_size)
            task_count = (
                context_offset + q_len + Int32(region_block_size - 1)
            ) // Int32(region_block_size)
            prefix += cutlass.select_(q_len > Int32(0), task_count, Int32(0))
            mTaskPrefix[req + Int32(1)] = prefix
            req += Int32(1)


@cute.kernel
def _csa_compact_prefill_update_with_slots_kernel_sm90_gqa(
    mSum: cute.Tensor,
    mCount: cute.Tensor,
    mMean: cute.Tensor,
    mActiveRegionIds: cute.Tensor,
    mActiveSlotByRegion: cute.Tensor,
    mAllocationSuccess: cute.Tensor,
    mActiveNumerator: cute.Tensor,
    mDenominator: cute.Tensor,
    mMaxLogits: cute.Tensor,
    mFlatSlot: cute.Tensor,
    mResetSlots: cute.Tensor,
    mTokenValidU8: cute.Tensor,
    mTokenPositions: cute.Tensor,
    mQueryStartLoc: cute.Tensor,
    mSeqLens: cute.Tensor,
    mTaskPrefix: cute.Tensor,
    mIndexK: cute.Tensor,
    mIndexZ: cute.Tensor,
    source_rows: Int32,
    num_reqs: Int32,
    total_regions: Int64,
    active_capacity: Int32,
    head_dim: cutlass.Constexpr[int],
    summaries_per_page: cutlass.Constexpr[int],
    region_block_size: cutlass.Constexpr[int],
):
    task, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    smem = cutlass.utils.SmemAllocator()
    sRows = smem.allocate_tensor(
        Int32,
        cute.make_layout((_MAX_REGION_BLOCK_SIZE,), stride=(1,)),
        byte_alignment=16,
    )
    sState = smem.allocate_tensor(
        Int32,
        cute.make_layout((8,), stride=(1,)),
        byte_alignment=16,
    )
    sRegion = smem.allocate_tensor(
        Int64,
        cute.make_layout((1,), stride=(1,)),
        byte_alignment=8,
    )
    sWarpMax = smem.allocate_tensor(
        Float32,
        cute.make_layout((_WARPS_PER_BLOCK,), stride=(1,)),
        byte_alignment=16,
    )
    sSharedMax = smem.allocate_tensor(
        Float32,
        cute.make_layout((1,), stride=(1,)),
        byte_alignment=16,
    )
    sIndexLayout = cute.make_layout((8, 256), stride=(256, 1))
    sIndexK = smem.allocate_tensor(
        mIndexK.element_type,
        sIndexLayout,
        byte_alignment=16,
    )
    sIndexZ = smem.allocate_tensor(
        mIndexZ.element_type,
        sIndexLayout,
        byte_alignment=16,
    )
    if Int32(tidx) < Int32(region_block_size):
        sRows[Int32(tidx)] = Int32(-1)
    if Int32(tidx) < Int32(8):
        sState[Int32(tidx)] = Int32(0)
    cute.arch.barrier()

    # Each request reserves one task per distinct summary region touched by its
    # query rows. The device prefix kernel accounts for every request's context
    # alignment once; each payload CTA only performs a logarithmic lookup.
    row = source_rows
    region = Int64(-1)
    valid = False
    if Int32(tidx) == Int32(0) and num_reqs > Int32(0):
        lo = Int32(0)
        hi = num_reqs
        while lo < hi:
            mid = (lo + hi) // Int32(2)
            task_end = Int32(mTaskPrefix[mid + Int32(1)])
            if task < task_end:
                hi = mid
            else:
                lo = mid + Int32(1)
        req = lo
        if req >= Int32(0) and req < num_reqs:
            task_base = Int32(mTaskPrefix[req])
            task_end = Int32(mTaskPrefix[req + Int32(1)])
            q_start = Int32(mQueryStartLoc[req])
            q_end = Int32(mQueryStartLoc[req + Int32(1)])
            q_len = q_end - q_start
            context_len = Int32(mSeqLens[req]) - q_len
            context_offset = context_len % Int32(region_block_size)
            region_ordinal = Int32(task) - task_base
            if region_ordinal == Int32(0):
                row = q_start
            elif region_ordinal > Int32(0):
                first_span = Int32(region_block_size) - context_offset
                row = (
                    q_start
                    + first_span
                    + (region_ordinal - Int32(1)) * Int32(region_block_size)
                )
            if task >= task_end or row < q_start or row >= q_end or row >= source_rows:
                row = source_rows
        if row >= Int32(0) and row < source_rows:
            region = Int64(mFlatSlot[row])
            valid = (
                (Int32(mTokenValidU8[row]) != Int32(0))
                & (region >= Int64(0))
                & (region < total_regions)
            )
        else:
            region = Int64(-1)
            valid = False
        if valid:
            sRegion[0] = region
            sState[0] = Int32(1)
            sState[5] = Int32(region // Int64(summaries_per_page))
            sState[6] = Int32(region % Int64(summaries_per_page))
            for offset in range(region_block_size):
                candidate = row + Int32(offset)
                # Note(wangbojun/codex): CuTeDSL boolean conjunction does not
                # guarantee that metadata loads are short-circuited. Keep the
                # row bound in an outer branch so a tail CTA cannot issue an
                # out-of-range load before evaluating the predicate.
                if candidate >= Int32(0) and candidate < source_rows:
                    if Int32(mTokenValidU8[candidate]) != Int32(0):
                        if Int64(mFlatSlot[candidate]) == region:
                            position = Int64(mTokenPositions[candidate])
                            token_offset = Int32(position % Int64(region_block_size))
                            if token_offset >= Int32(0) and token_offset < Int32(
                                region_block_size
                            ):
                                sRows[token_offset] = candidate
                            if Int64(mResetSlots[candidate]) == region:
                                sState[2] = Int32(1)
                            if token_offset == Int32(region_block_size - 1):
                                sState[3] = Int32(1)
    cute.arch.barrier()

    if sState[0] != Int32(0):
        region = sRegion[0]
        reset = sState[2] != Int32(0)
        complete = sState[3] != Int32(0)
        summary_page = sState[5]
        summary_fragment = sState[6]
        if Int32(tidx) == Int32(0):
            found_slot = Int32(mActiveSlotByRegion[region])
            # CuTeDSL does not short-circuit boolean chains, so the
            # mActiveRegionIds load must sit behind an outer branch. A region
            # with no active slot stores -1 here, and folding the load into the
            # bound check issues mActiveRegionIds[-1] before the predicate.
            if found_slot < Int32(0) or found_slot >= active_capacity:
                found_slot = active_capacity
            elif Int64(mActiveRegionIds[found_slot]) != region:
                found_slot = active_capacity
            if found_slot == active_capacity and not complete:
                found_slot = Int32(region % Int64(active_capacity))
                probe = Int32(0)
                allocated = Int32(0)
                while probe < active_capacity and allocated == Int32(0):
                    old = _atomic_cas_i64(
                        elem_pointer(mActiveRegionIds, (found_slot,)),
                        Int64(-1),
                        region,
                    )
                    if old == Int64(-1) or old == region:
                        allocated = Int32(1)
                    else:
                        found_slot += Int32(1)
                        if found_slot == active_capacity:
                            found_slot = Int32(0)
                    probe += Int32(1)
                if allocated == Int32(0):
                    found_slot = active_capacity
                    mAllocationSuccess[0] = Int32(0)
                else:
                    mActiveSlotByRegion[region] = found_slot
            sState[1] = found_slot
            old_max = Float32(_NEG_INF)
            if found_slot < active_capacity and not reset:
                old_max = Float32(mMaxLogits[found_slot, 0])
            sSharedMax[0] = old_max
        cute.arch.barrier()

        found_slot = sState[1]
        previous_max = sSharedMax[0]
        # Note(wangbojun/codex): One warp owns one token so every lane starts at
        # lane * 8 in a contiguous 256-channel row. This proves 16-byte alignment
        # for BF16 K/Z and lets the weighted pass reuse Z from shared memory.
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        token_offset = Int32(warp_idx)
        channel_base = Int32(lane_idx) * Int32(8)
        matched_row = sRows[token_offset]
        vec_layout = cute.make_layout((8,), stride=(1,))
        s_k_ptr = elem_pointer(sIndexK, (token_offset, channel_base))
        s_z_ptr = elem_pointer(sIndexZ, (token_offset, channel_base))
        s_k_ptr = cute.make_ptr(
            mIndexK.element_type,
            s_k_ptr.toint(),
            sIndexK.memspace,
            assumed_align=16,
        )
        s_z_ptr = cute.make_ptr(
            mIndexZ.element_type,
            s_z_ptr.toint(),
            sIndexZ.memspace,
            assumed_align=16,
        )
        s_k_vec = cute.make_tensor(s_k_ptr, vec_layout)
        s_z_vec = cute.make_tensor(s_z_ptr, vec_layout)
        if matched_row >= Int32(0):
            g_k_ptr = elem_pointer(mIndexK, (matched_row, 0, channel_base))
            g_z_ptr = elem_pointer(mIndexZ, (matched_row, 0, channel_base))
            g_k_ptr = cute.make_ptr(
                mIndexK.element_type,
                g_k_ptr.toint(),
                mIndexK.memspace,
                assumed_align=16,
            )
            g_z_ptr = cute.make_ptr(
                mIndexZ.element_type,
                g_z_ptr.toint(),
                mIndexZ.memspace,
                assumed_align=16,
            )
            g_k_vec = cute.make_tensor(g_k_ptr, vec_layout)
            g_z_vec = cute.make_tensor(g_z_ptr, vec_layout)
            utils.vector_copy_with_explicit_width(
                g_k_vec,
                s_k_vec,
                num_copy_elems=8,
            )
            utils.vector_copy_with_explicit_width(
                g_z_vec,
                s_z_vec,
                num_copy_elems=8,
            )
        else:
            for vec_elem in cutlass.range_constexpr(8):
                s_k_vec[vec_elem] = mIndexK.element_type(0.0)
                s_z_vec[vec_elem] = mIndexZ.element_type(_NEG_INF)
        cute.arch.barrier()

        d = Int32(tidx)
        numerator = Float32(0.0)
        denominator = Float32(0.0)
        if d < Int32(head_dim) and found_slot < active_capacity and not reset:
            numerator = Float32(mActiveNumerator[found_slot, 0, d])
            denominator = Float32(mDenominator[found_slot, 0, d])

        local_max = Float32(_NEG_INF)
        if denominator > Float32(0.0):
            local_max = previous_max
        for offset in range(region_block_size):
            local_max = cute.arch.fmax(local_max, Float32(sIndexZ[offset, d]))
        warp_max = utils.warp_reduce(local_max, cute.arch.fmax, width=32)
        if lane_idx == Int32(0):
            sWarpMax[warp_idx] = warp_max
        cute.arch.barrier()
        if Int32(tidx) == Int32(0):
            shared_max = sWarpMax[0]
            for warp in cutlass.range_constexpr(1, _WARPS_PER_BLOCK):
                shared_max = cute.arch.fmax(shared_max, sWarpMax[warp])
            sSharedMax[0] = shared_max
        cute.arch.barrier()
        new_max = sSharedMax[0]

        if d < Int32(head_dim):
            old_scale = Float32(0.0)
            if denominator > Float32(0.0):
                old_scale = cute.math.exp2(
                    (previous_max - new_max) * Float32(_LOG2_E),
                    fastmath=True,
                )
            numerator *= old_scale
            denominator *= old_scale
            for offset in range(region_block_size):
                logit = Float32(sIndexZ[offset, d])
                weight = cute.math.exp2(
                    (logit - new_max) * Float32(_LOG2_E),
                    fastmath=True,
                )
                numerator += weight * Float32(sIndexK[offset, d])
                denominator += weight
            output = Float32(0.0)
            if denominator > Float32(0.0):
                output = numerator / cute.arch.fmax(denominator, Float32(1.0e-20))
            # Complete regions have no active scratch ownership.  Their
            # materialized FP8 mean above is the only persistent result.
            #
            # A non-complete region is *expected* to own a real slot, but slot
            # allocation above can genuinely fail (all slots taken), in which
            # case found_slot == active_capacity is a failure sentinel, not a
            # valid index.  Writing at that index corrupts whatever allocation
            # follows the active buffers - observed in production as a 1024-byte
            # fp32 write one row past mDenominator, which landed on a neighbour
            # scratch buffer and produced ~1e9 garbage when read back as int32.
            # Treat capacity exhaustion as "drop this update" instead: the
            # page-addressed FP8 mean below is still materialized, so only the
            # incremental active state for this region is lost.
            slot_valid = found_slot < active_capacity
            if not complete:
                if slot_valid:
                    mSum[found_slot, 0, 0, d] = output
            mean_offset = _wgmma_sw128_physical_offset(d, region, summary_fragment)
            mean_slot = mean_offset // Int32(head_dim)
            mean_d = mean_offset - mean_slot * Int32(head_dim)
            # Note(wangbojun/codex): prefill logits consumes the paged fp8
            # summary immediately, so the update kernel must materialize the
            # current snapshot here instead of depending on a later refresh.
            output_for_fp8 = Float32(BFloat16(output))
            mMean[summary_page, mean_slot, 0, mean_d] = cvt_f32_to_e4m3(
                output_for_fp8
            ).to(Uint8)
            if complete:
                if slot_valid:
                    mActiveNumerator[found_slot, 0, d] = Float32(0.0)
                    mDenominator[found_slot, 0, d] = Float32(0.0)
            else:
                if slot_valid:
                    mActiveNumerator[found_slot, 0, d] = numerator
                    mDenominator[found_slot, 0, d] = denominator
        cute.arch.barrier()
        if Int32(tidx) == Int32(0):
            if not complete:
                if found_slot < active_capacity:
                    mCount[found_slot, 0, 0] = Float32(1.0)
                    mMaxLogits[found_slot, 0] = new_max
            elif found_slot < active_capacity:
                mMaxLogits[found_slot, 0] = Float32(float("-inf"))
                mActiveSlotByRegion[region] = Int32(-1)
                mActiveRegionIds[found_slot] = Int64(-1)


@cute.jit
def _csa_compact_decode_update_with_slots_cluster_valid_row(
    mSum: cute.Tensor,
    mCount: cute.Tensor,
    mActiveRegionIds: cute.Tensor,
    mActiveSlotByRegion: cute.Tensor,
    mActiveNumerator: cute.Tensor,
    mDenominator: cute.Tensor,
    mMaxLogits: cute.Tensor,
    mResetSlots: cute.Tensor,
    mTokenPositions: cute.Tensor,
    mIndexK: cute.Tensor,
    mIndexZ: cute.Tensor,
    row: Int32,
    tidx: Int32,
    cta_rank: Int32,
    region: Int64,
    active_capacity: Int32,
    head_dim: cutlass.Constexpr[int],
    summaries_per_page: cutlass.Constexpr[int],
    region_block_size: cutlass.Constexpr[int],
    threads_per_block: cutlass.Constexpr[int],
    sState: cute.Tensor,
    sWarpMax: cute.Tensor,
    sSharedMax: cute.Tensor,
):
    if Int32(tidx) == Int32(0):
        reset = Int32(Int64(mResetSlots[row]) == region)
        position = Int64(mTokenPositions[row])
        complete = Int32(
            position % Int64(region_block_size) == Int64(region_block_size - Int32(1))
        )
        found_slot = Int32(mActiveSlotByRegion[region])
        if found_slot < Int32(0) or found_slot >= active_capacity:
            found_slot = active_capacity
        elif Int64(mActiveRegionIds[found_slot]) != region:
            found_slot = active_capacity
        if found_slot == active_capacity and complete == Int32(0):
            found_slot = Int32(region % Int64(active_capacity))
            probe = Int32(0)
            allocated = Int32(0)
            while probe < active_capacity and allocated == Int32(0):
                old = _atomic_cas_i64(
                    elem_pointer(mActiveRegionIds, (found_slot,)),
                    Int64(-1),
                    region,
                )
                if old == Int64(-1) or old == region:
                    allocated = Int32(1)
                else:
                    found_slot += Int32(1)
                    if found_slot == active_capacity:
                        found_slot = Int32(0)
                probe += Int32(1)
            if allocated == Int32(0):
                found_slot = active_capacity
            else:
                mActiveSlotByRegion[region] = found_slot
        sState[0] = found_slot
        sState[1] = reset
        sState[2] = complete
        sState[4] = Int32(region // Int64(summaries_per_page))
        sState[5] = Int32(region % Int64(summaries_per_page))
    cute.arch.barrier()

    found_slot = sState[0]
    reset = sState[1] != Int32(0)
    complete = sState[2] != Int32(0)
    summary_page = sState[4]
    summary_fragment = sState[5]
    d = Int32(cta_rank * Int32(threads_per_block)) + Int32(tidx)
    numerator = Float32(0.0)
    denominator = Float32(0.0)
    old_max_scalar = Float32(_NEG_INF)
    logit = Float32(_NEG_INF)
    if Int32(tidx) == Int32(0):
        if found_slot < active_capacity and not reset:
            old_max_scalar = Float32(mMaxLogits[found_slot, 0])
        sSharedMax[Int32(0)] = old_max_scalar
    cute.arch.barrier()
    old_max_scalar = sSharedMax[Int32(0)]
    if d < Int32(head_dim):
        if found_slot < active_capacity and not reset:
            numerator = Float32(mActiveNumerator[found_slot, 0, d])
            denominator = Float32(mDenominator[found_slot, 0, d])
        logit = Float32(mIndexZ[row, 0, d])
    local_max = logit
    if denominator > Float32(0.0):
        local_max = cute.arch.fmax(local_max, old_max_scalar)
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    warp_max = utils.warp_reduce(local_max, cute.arch.fmax, width=32)
    if lane_idx == Int32(0):
        sWarpMax[warp_idx] = warp_max
    cute.arch.barrier()
    if Int32(tidx) == Int32(0):
        shared_max = sWarpMax[Int32(0)]
        for warp in cutlass.range(1, threads_per_block // Int32(32)):
            shared_max = cute.arch.fmax(shared_max, sWarpMax[warp])
        sSharedMax[Int32(0)] = shared_max
    cute.arch.barrier()
    new_max = sSharedMax[Int32(0)]

    if d < Int32(head_dim):
        value = Float32(mIndexK[row, 0, d])
        old_scale = Float32(0.0)
        if denominator > Float32(0.0):
            old_scale = cute.math.exp2(
                (old_max_scalar - new_max) * Float32(_LOG2_E), fastmath=True
            )
        weight = cute.math.exp2((logit - new_max) * Float32(_LOG2_E), fastmath=True)
        numerator = numerator * old_scale + weight * value
        denominator = denominator * old_scale + weight
        # As in the prefill path, found_slot == active_capacity is the
        # allocation-failure sentinel, not a valid index. Keep every state
        # write guarded so capacity exhaustion drops this update instead of
        # corrupting the adjacent scratch allocation.
        slot_valid = found_slot < active_capacity
        if reset:
            if slot_valid:
                mSum[found_slot, 0, 0, d] = Float32(0.0)
        if complete:
            output = Float32(0.0)
            if denominator > Float32(0.0):
                output = numerator / cute.arch.fmax(denominator, Float32(1.0e-20))
            if slot_valid:
                mSum[found_slot, 0, 0, d] = output
            if found_slot < active_capacity:
                mActiveNumerator[found_slot, 0, d] = Float32(0.0)
                mDenominator[found_slot, 0, d] = Float32(0.0)
        elif found_slot < active_capacity:
            mActiveNumerator[found_slot, 0, d] = numerator
            mDenominator[found_slot, 0, d] = denominator
    if found_slot < active_capacity and Int32(tidx) == Int32(0):
        if complete:
            mMaxLogits[found_slot, 0] = Float32(float("-inf"))
        else:
            mMaxLogits[found_slot, 0] = new_max

    if complete:
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if (
            Int32(cta_rank) == Int32(0)
            and Int32(tidx) == Int32(0)
            and found_slot < active_capacity
        ):
            mCount[found_slot, 0, 0] = Float32(1.0)
            mActiveSlotByRegion[region] = Int32(-1)
            mActiveRegionIds[found_slot] = Int64(-1)
    elif (
        Int32(cta_rank) == Int32(0)
        and Int32(tidx) == Int32(0)
        and reset
        and found_slot < active_capacity
    ):
        mCount[found_slot, 0, 0] = Float32(0.0)


@cute.kernel
def _csa_compact_decode_update_with_slots_kernel_sm90_gqa(
    mSum: cute.Tensor,
    mCount: cute.Tensor,
    mActiveRegionIds: cute.Tensor,
    mActiveSlotByRegion: cute.Tensor,
    mActiveNumerator: cute.Tensor,
    mDenominator: cute.Tensor,
    mMaxLogits: cute.Tensor,
    mFlatSlot: cute.Tensor,
    mResetSlots: cute.Tensor,
    mTokenValidU8: cute.Tensor,
    mTokenPositions: cute.Tensor,
    mIndexK: cute.Tensor,
    mIndexZ: cute.Tensor,
    total_regions: Int64,
    active_capacity: Int32,
    head_dim: cutlass.Constexpr[int],
    summaries_per_page: cutlass.Constexpr[int],
    region_block_size: cutlass.Constexpr[int],
    cluster_size: cutlass.Constexpr[int],
    threads_per_block: cutlass.Constexpr[int],
):
    block, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    cta_rank = cute.arch.block_idx_in_cluster()
    row = Int32(block // Int32(cluster_size))
    region = Int64(mFlatSlot[row])
    valid = (
        (Int32(mTokenValidU8[row]) != Int32(0))
        & (region >= Int64(0))
        & (region < total_regions)
    )
    smem = cutlass.utils.SmemAllocator()
    sState = smem.allocate_tensor(
        Int32,
        cute.make_layout((6,), stride=(1,)),
        byte_alignment=16,
    )
    sWarpMax = smem.allocate_tensor(
        Float32,
        cute.make_layout((_MAX_WARPS_PER_BLOCK,), stride=(1,)),
        byte_alignment=16,
    )
    sSharedMax = smem.allocate_tensor(
        Float32,
        cute.make_layout((1,), stride=(1,)),
        byte_alignment=16,
    )
    # A graph-padding row is uniform across its entire CTA cluster, so every
    # peer either enters the helper and its barriers or skips them together.
    if valid:
        _csa_compact_decode_update_with_slots_cluster_valid_row(
            mSum,
            mCount,
            mActiveRegionIds,
            mActiveSlotByRegion,
            mActiveNumerator,
            mDenominator,
            mMaxLogits,
            mResetSlots,
            mTokenPositions,
            mIndexK,
            mIndexZ,
            row,
            Int32(tidx),
            Int32(cta_rank),
            region,
            active_capacity,
            head_dim,
            summaries_per_page,
            region_block_size,
            threads_per_block,
            sState,
            sWarpMax,
            sSharedMax,
        )


@cute.kernel
def _csa_compact_decode_stage_flush_with_slots_kernel_sm90_gqa(
    mSum: cute.Tensor,
    mCount: cute.Tensor,
    mMean: cute.Tensor,
    mActiveRegionIds: cute.Tensor,
    mActiveSlotByRegion: cute.Tensor,
    mAllocationSuccess: cute.Tensor,
    mActiveNumerator: cute.Tensor,
    mDenominator: cute.Tensor,
    mMaxLogits: cute.Tensor,
    mActiveTokenK: cute.Tensor,
    mActiveTokenZ: cute.Tensor,
    mActiveTokenValidU8: cute.Tensor,
    mFlatSlot: cute.Tensor,
    mResetSlots: cute.Tensor,
    mTokenValidU8: cute.Tensor,
    mTokenPositions: cute.Tensor,
    mIndexK: cute.Tensor,
    mIndexZ: cute.Tensor,
    total_regions: Int64,
    active_capacity: Int32,
    head_dim: cutlass.Constexpr[int],
    summaries_per_page: cutlass.Constexpr[int],
    region_block_size: cutlass.Constexpr[int],
):
    row, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    region = Int64(mFlatSlot[row])
    valid = (
        (Int32(mTokenValidU8[row]) != Int32(0))
        & (region >= Int64(0))
        & (region < total_regions)
    )

    smem = cutlass.utils.SmemAllocator()
    sState = smem.allocate_tensor(
        Int32,
        cute.make_layout((6,), stride=(1,)),
        byte_alignment=16,
    )
    sStageValid = smem.allocate_tensor(
        Int32,
        cute.make_layout((_DECODE_STAGE_TOKENS,), stride=(1,)),
        byte_alignment=16,
    )
    sWarpSlots = smem.allocate_tensor(
        Int32,
        cute.make_layout((_WARPS_PER_BLOCK,), stride=(1,)),
        byte_alignment=16,
    )
    sWarpMax = smem.allocate_tensor(
        Float32,
        cute.make_layout((_WARPS_PER_BLOCK,), stride=(1,)),
        byte_alignment=16,
    )
    sSharedMax = smem.allocate_tensor(
        Float32,
        cute.make_layout((1,), stride=(1,)),
        byte_alignment=16,
    )
    if valid:
        position = Int64(mTokenPositions[row])
        offset = Int32(position % Int64(_DECODE_STAGE_TOKENS))
        mapped_slot = Int32(mActiveSlotByRegion[region])
        mapped_slot_valid = (mapped_slot >= Int32(0)) & (mapped_slot < active_capacity)
        found_slot = mapped_slot
        map_mismatch = Int32(0)
        if mapped_slot_valid:
            map_mismatch = Int32(Int64(mActiveRegionIds[mapped_slot]) != region)
            if map_mismatch != Int32(0):
                found_slot = active_capacity
        else:
            found_slot = active_capacity
        reset = Int64(mResetSlots[row]) == region
        if map_mismatch != Int32(0):
            found_slot = _find_active_slot_parallel(
                mActiveRegionIds,
                region,
                active_capacity,
                sWarpSlots,
            )
            if Int32(tidx) == Int32(0):
                sState[0] = found_slot
                if found_slot < active_capacity:
                    mActiveSlotByRegion[region] = found_slot
            cute.arch.barrier()
            found_slot = sState[0]
            if found_slot == active_capacity:
                empty_slot = _find_active_slot_parallel(
                    mActiveRegionIds,
                    Int64(-1),
                    active_capacity,
                    sWarpSlots,
                )
                if Int32(tidx) == Int32(0):
                    newly_allocated = Int32(0)
                    if empty_slot < active_capacity:
                        old = _atomic_cas_i64(
                            elem_pointer(mActiveRegionIds, (empty_slot,)),
                            Int64(-1),
                            region,
                        )
                        if old == Int64(-1) or old == region:
                            found_slot = empty_slot
                            newly_allocated = Int32(old == Int64(-1))
                            mActiveSlotByRegion[region] = found_slot
                    sState[0] = found_slot
                    sState[1] = Int32(reset | (newly_allocated != Int32(0)))
                cute.arch.barrier()
                found_slot = sState[0]
                reset = sState[1] != Int32(0)
        if found_slot == active_capacity:
            if Int32(tidx) == Int32(0):
                found_slot = Int32(region % Int64(active_capacity))
                probe = Int32(0)
                allocated = Int32(0)
                newly_allocated = Int32(0)
                probe_limit = active_capacity
                if probe_limit > Int32(32):
                    probe_limit = Int32(32)
                while probe < probe_limit and allocated == Int32(0):
                    old = _atomic_cas_i64(
                        elem_pointer(mActiveRegionIds, (found_slot,)),
                        Int64(-1),
                        region,
                    )
                    if old == Int64(-1) or old == region:
                        allocated = Int32(1)
                        newly_allocated = Int32(old == Int64(-1))
                    else:
                        found_slot += Int32(1)
                        if found_slot == active_capacity:
                            found_slot = Int32(0)
                    probe += Int32(1)
                sState[2] = allocated
                sState[0] = found_slot if allocated != Int32(0) else active_capacity
                sState[1] = Int32(reset | (newly_allocated != Int32(0)))
                if allocated != Int32(0):
                    mActiveSlotByRegion[region] = found_slot
            cute.arch.barrier()
            found_slot = sState[0]
            reset = sState[1] != Int32(0)
            if sState[2] == Int32(0):
                found_slot = _find_active_slot_parallel(
                    mActiveRegionIds,
                    region,
                    active_capacity,
                    sWarpSlots,
                )
                if found_slot == active_capacity:
                    found_slot = _find_active_slot_parallel(
                        mActiveRegionIds,
                        Int64(-1),
                        active_capacity,
                        sWarpSlots,
                    )
                    if Int32(tidx) == Int32(0):
                        newly_allocated = Int32(0)
                        if found_slot < active_capacity:
                            old = _atomic_cas_i64(
                                elem_pointer(mActiveRegionIds, (found_slot,)),
                                Int64(-1),
                                region,
                            )
                            if old == Int64(-1) or old == region:
                                newly_allocated = Int32(old == Int64(-1))
                            else:
                                found_slot = active_capacity
                        sState[0] = found_slot
                        sState[1] = Int32(reset | (newly_allocated != Int32(0)))
                        if found_slot < active_capacity:
                            mActiveSlotByRegion[region] = found_slot
                    cute.arch.barrier()
                else:
                    if Int32(tidx) == Int32(0):
                        sState[0] = found_slot
                        mActiveSlotByRegion[region] = found_slot
                    cute.arch.barrier()
                found_slot = sState[0]
                reset = sState[1] != Int32(0)
        if found_slot == active_capacity and Int32(tidx) == Int32(0):
            mAllocationSuccess[0] = Int32(0)
        complete = offset == Int32(_DECODE_STAGE_TOKENS - 1)
        summary_page = Int32(region // Int64(summaries_per_page))
        summary_fragment = Int32(region % Int64(summaries_per_page))
        logical_region = region

        if found_slot < active_capacity:
            if reset:
                if Int32(tidx) < Int32(_DECODE_STAGE_TOKENS):
                    mActiveTokenValidU8[found_slot, Int32(tidx)] = Uint8(0)
                for part in cutlass.range_constexpr(_MAX_STAGE_DIMS_PER_THREAD):
                    d = Int32(tidx) + Int32(part * _STAGE_THREADS_PER_BLOCK)
                    if d < Int32(head_dim):
                        mean_offset = _wgmma_sw128_physical_offset(
                            d, logical_region, summary_fragment
                        )
                        mean_slot = mean_offset // Int32(head_dim)
                        mean_d = mean_offset - mean_slot * Int32(head_dim)
                        mActiveNumerator[found_slot, 0, d] = Float32(0.0)
                        mDenominator[found_slot, 0, d] = Float32(0.0)
                        mSum[found_slot, 0, 0, d] = Float32(0.0)
                        mMean[summary_page, mean_slot, 0, mean_d] = Uint8(0)
                if Int32(tidx) == Int32(0):
                    mMaxLogits[found_slot, 0] = Float32(float("-inf"))
                    mCount[found_slot, 0, 0] = Float32(0.0)
                cute.arch.barrier()

            if not complete:
                for part in cutlass.range_constexpr(_MAX_STAGE_DIMS_PER_THREAD):
                    d = Int32(tidx) + Int32(part * _STAGE_THREADS_PER_BLOCK)
                    if d < Int32(head_dim):
                        mActiveTokenK[found_slot, offset, 0, d] = (
                            mActiveTokenK.element_type(mIndexK[row, 0, d])
                        )
                        mActiveTokenZ[found_slot, offset, 0, d] = (
                            mActiveTokenZ.element_type(mIndexZ[row, 0, d])
                        )
                if Int32(tidx) == Int32(0):
                    mActiveTokenValidU8[found_slot, offset] = Uint8(1)
            else:
                if Int32(tidx) < Int32(_DECODE_STAGE_TOKENS):
                    sStageValid[Int32(tidx)] = Int32(
                        mActiveTokenValidU8[found_slot, Int32(tidx)]
                    )
                cute.arch.barrier()

                if Int32(tidx) == Int32(0):
                    prefix_max_scalar = Float32(_NEG_INF)
                    if not reset:
                        prefix_max_scalar = Float32(mMaxLogits[found_slot, 0])
                    sSharedMax[Int32(0)] = prefix_max_scalar
                cute.arch.barrier()
                prefix_max_scalar = sSharedMax[Int32(0)]
                local_max = prefix_max_scalar
                for part in cutlass.range_constexpr(_MAX_STAGE_DIMS_PER_THREAD):
                    d = Int32(tidx) + Int32(part * _STAGE_THREADS_PER_BLOCK)
                    if d < Int32(head_dim):
                        for token_offset in cutlass.range_constexpr(
                            _DECODE_STAGE_TOKENS
                        ):
                            use_current = Int32(token_offset) == offset
                            token_is_valid = (
                                sStageValid[token_offset] != Int32(0)
                            ) | use_current
                            if token_is_valid:
                                logit = Float32(
                                    mActiveTokenZ[found_slot, token_offset, 0, d]
                                )
                                if use_current:
                                    logit = Float32(mIndexZ[row, 0, d])
                                local_max = cute.arch.fmax(local_max, logit)
                lane_idx = cute.arch.lane_idx()
                warp_idx = cute.arch.warp_idx()
                warp_max = utils.warp_reduce(local_max, cute.arch.fmax, width=32)
                if lane_idx == Int32(0):
                    sWarpMax[warp_idx] = warp_max
                cute.arch.barrier()
                if Int32(tidx) == Int32(0):
                    shared_max = sWarpMax[Int32(0)]
                    for warp in cutlass.range_constexpr(1, _WARPS_PER_BLOCK):
                        shared_max = cute.arch.fmax(shared_max, sWarpMax[warp])
                    sSharedMax[Int32(0)] = shared_max
                cute.arch.barrier()
                shared_max = sSharedMax[Int32(0)]

                for part in cutlass.range_constexpr(_MAX_STAGE_DIMS_PER_THREAD):
                    d = Int32(tidx) + Int32(part * _STAGE_THREADS_PER_BLOCK)
                    if d < Int32(head_dim):
                        prefix_denominator = Float32(mDenominator[found_slot, 0, d])
                        prefix_numerator = Float32(0.0)
                        prefix_max = prefix_max_scalar
                        if prefix_denominator > Float32(0.0):
                            prefix_numerator = Float32(
                                mActiveNumerator[found_slot, 0, d]
                            )

                        token_logits = [
                            Float32(_NEG_INF) for _ in range(_DECODE_STAGE_TOKENS)
                        ]
                        for token_offset in cutlass.range_constexpr(
                            _DECODE_STAGE_TOKENS
                        ):
                            use_current = Int32(token_offset) == offset
                            token_is_valid = (
                                sStageValid[token_offset] != Int32(0)
                            ) | use_current
                            if token_is_valid:
                                logit = Float32(
                                    mActiveTokenZ[found_slot, token_offset, 0, d]
                                )
                                if use_current:
                                    logit = Float32(mIndexZ[row, 0, d])
                                token_logits[token_offset] = logit
                        # Keep the Triton contract: one scalar max per
                        # region/head is used for every proxy dimension.
                        max_acc = shared_max

                        numerator = Float32(0.0)
                        denominator = Float32(0.0)
                        if prefix_denominator > Float32(0.0):
                            prefix_scale = cute.math.exp2(
                                (prefix_max - max_acc) * Float32(_LOG2_E),
                                fastmath=True,
                            )
                            numerator = prefix_numerator * prefix_scale
                            denominator = prefix_denominator * prefix_scale
                        for token_offset in cutlass.range_constexpr(
                            _DECODE_STAGE_TOKENS
                        ):
                            use_current = Int32(token_offset) == offset
                            token_is_valid = (
                                sStageValid[token_offset] != Int32(0)
                            ) | use_current
                            if token_is_valid:
                                value = Float32(
                                    mActiveTokenK[found_slot, token_offset, 0, d]
                                )
                                if use_current:
                                    value = Float32(mIndexK[row, 0, d])
                                weight = cute.math.exp2(
                                    (token_logits[token_offset] - max_acc)
                                    * Float32(_LOG2_E),
                                    fastmath=True,
                                )
                                numerator += weight * value
                                denominator += weight
                        output = Float32(0.0)
                        if denominator > Float32(0.0):
                            output = numerator / cute.arch.fmax(
                                denominator, Float32(1.0e-20)
                            )
                        mSum[found_slot, 0, 0, d] = output
                        mean_offset = _wgmma_sw128_physical_offset(
                            d, logical_region, summary_fragment
                        )
                        mean_slot = mean_offset // Int32(head_dim)
                        mean_d = mean_offset - mean_slot * Int32(head_dim)
                        # Note(wangbojun/codex): Baseline summary logits cast
                        # the FP32 mean through BF16 before E4M3. Preserve
                        # that rounding contract in the direct cache writer.
                        output_for_fp8 = Float32(BFloat16(output))
                        mMean[summary_page, mean_slot, 0, mean_d] = cvt_f32_to_e4m3(
                            output_for_fp8
                        ).to(Uint8)
                        mActiveNumerator[found_slot, 0, d] = Float32(0.0)
                        mDenominator[found_slot, 0, d] = Float32(0.0)
                if Int32(tidx) == Int32(0):
                    mMaxLogits[found_slot, 0] = Float32(float("-inf"))
                if Int32(tidx) < Int32(_DECODE_STAGE_TOKENS):
                    mActiveTokenValidU8[found_slot, Int32(tidx)] = Uint8(0)
                cute.arch.barrier()
                if Int32(tidx) == Int32(0):
                    mCount[found_slot, 0, 0] = Float32(1.0)
                    mActiveSlotByRegion[region] = Int32(-1)
                    mActiveRegionIds[found_slot] = Int64(-1)


def _make_launch_csa_compact_update_kernel(
    *,
    num_kv_heads: int,
    head_dim: int,
    summaries_per_page: int,
    region_block_size: int,
):
    @cute.jit
    def _launch(
        mSum: cute.Tensor,
        mCount: cute.Tensor,
        mActiveRegionIds: cute.Tensor,
        mDenominator: cute.Tensor,
        mMaxLogits: cute.Tensor,
        mFlatSlot: cute.Tensor,
        mResetSlots: cute.Tensor,
        mTokenValidU8: cute.Tensor,
        mTokenPositions: cute.Tensor,
        mIndexK: cute.Tensor,
        mIndexZ: cute.Tensor,
        source_rows: int,
        total_regions: int,
        active_capacity: int,
        stream: cuda.CUstream,
    ):
        _csa_compact_update_kernel_sm90_gqa_hash_cta256(
            mSum,
            mCount,
            mActiveRegionIds,
            mDenominator,
            mMaxLogits,
            mFlatSlot,
            mResetSlots,
            mTokenValidU8,
            mTokenPositions,
            mIndexK,
            mIndexZ,
            Int32(source_rows),
            Int64(total_regions),
            Int32(active_capacity),
            num_kv_heads,
            head_dim,
            summaries_per_page,
            region_block_size,
        ).launch(
            grid=[source_rows, 1, 1],
            block=[_THREADS_PER_BLOCK, 1, 1],
            stream=stream,
        )

    return _launch


def _make_launch_csa_compact_prefill_update_with_slots_kernel(
    *,
    head_dim: int,
    summaries_per_page: int,
    region_block_size: int,
):
    @cute.jit
    def _launch(
        mSum: cute.Tensor,
        mCount: cute.Tensor,
        mMean: cute.Tensor,
        mActiveRegionIds: cute.Tensor,
        mActiveSlotByRegion: cute.Tensor,
        mAllocationSuccess: cute.Tensor,
        mActiveNumerator: cute.Tensor,
        mDenominator: cute.Tensor,
        mMaxLogits: cute.Tensor,
        mFlatSlot: cute.Tensor,
        mResetSlots: cute.Tensor,
        mTokenValidU8: cute.Tensor,
        mTokenPositions: cute.Tensor,
        mQueryStartLoc: cute.Tensor,
        mSeqLens: cute.Tensor,
        mTaskPrefix: cute.Tensor,
        mIndexK: cute.Tensor,
        mIndexZ: cute.Tensor,
        source_rows: int,
        num_reqs: int,
        total_regions: int,
        active_capacity: int,
        stream: cuda.CUstream,
    ):
        _csa_compact_prefill_task_prefix_kernel_sm90_gqa(
            mQueryStartLoc,
            mSeqLens,
            mTaskPrefix,
            Int32(num_reqs),
            region_block_size,
        ).launch(
            grid=[1, 1, 1],
            block=[32, 1, 1],
            stream=stream,
        )
        _csa_compact_prefill_update_with_slots_kernel_sm90_gqa(
            mSum,
            mCount,
            mMean,
            mActiveRegionIds,
            mActiveSlotByRegion,
            mAllocationSuccess,
            mActiveNumerator,
            mDenominator,
            mMaxLogits,
            mFlatSlot,
            mResetSlots,
            mTokenValidU8,
            mTokenPositions,
            mQueryStartLoc,
            mSeqLens,
            mTaskPrefix,
            mIndexK,
            mIndexZ,
            Int32(source_rows),
            Int32(num_reqs),
            Int64(total_regions),
            Int32(active_capacity),
            head_dim,
            summaries_per_page,
            region_block_size,
        ).launch(
            grid=[
                (source_rows + region_block_size - 1) // region_block_size
                + 2 * num_reqs,
                1,
                1,
            ],
            block=[_THREADS_PER_BLOCK, 1, 1],
            stream=stream,
        )

    return _launch


def _make_launch_csa_compact_decode_update_with_slots_kernel(
    *,
    head_dim: int,
    summaries_per_page: int,
    region_block_size: int,
):
    cluster_size, threads_per_block = _decode_cluster_config(head_dim)

    @cute.jit
    def _launch(
        mSum: cute.Tensor,
        mCount: cute.Tensor,
        mActiveRegionIds: cute.Tensor,
        mActiveSlotByRegion: cute.Tensor,
        mActiveNumerator: cute.Tensor,
        mDenominator: cute.Tensor,
        mMaxLogits: cute.Tensor,
        mFlatSlot: cute.Tensor,
        mResetSlots: cute.Tensor,
        mTokenValidU8: cute.Tensor,
        mTokenPositions: cute.Tensor,
        mIndexK: cute.Tensor,
        mIndexZ: cute.Tensor,
        source_rows: int,
        total_regions: int,
        active_capacity: int,
        stream: cuda.CUstream,
    ):
        _csa_compact_decode_update_with_slots_kernel_sm90_gqa(
            mSum,
            mCount,
            mActiveRegionIds,
            mActiveSlotByRegion,
            mActiveNumerator,
            mDenominator,
            mMaxLogits,
            mFlatSlot,
            mResetSlots,
            mTokenValidU8,
            mTokenPositions,
            mIndexK,
            mIndexZ,
            Int64(total_regions),
            Int32(active_capacity),
            head_dim,
            summaries_per_page,
            region_block_size,
            cluster_size,
            threads_per_block,
        ).launch(
            grid=[source_rows * cluster_size, 1, 1],
            block=[threads_per_block, 1, 1],
            cluster=(cluster_size, 1, 1),
            stream=stream,
        )

    return _launch


def _make_launch_csa_compact_decode_stage_flush_with_slots_kernel(
    *,
    head_dim: int,
    summaries_per_page: int,
    region_block_size: int,
):
    @cute.jit
    def _launch(
        mSum: cute.Tensor,
        mCount: cute.Tensor,
        mMean: cute.Tensor,
        mActiveRegionIds: cute.Tensor,
        mActiveSlotByRegion: cute.Tensor,
        mAllocationSuccess: cute.Tensor,
        mActiveNumerator: cute.Tensor,
        mDenominator: cute.Tensor,
        mMaxLogits: cute.Tensor,
        mActiveTokenK: cute.Tensor,
        mActiveTokenZ: cute.Tensor,
        mActiveTokenValidU8: cute.Tensor,
        mFlatSlot: cute.Tensor,
        mResetSlots: cute.Tensor,
        mTokenValidU8: cute.Tensor,
        mTokenPositions: cute.Tensor,
        mIndexK: cute.Tensor,
        mIndexZ: cute.Tensor,
        source_rows: int,
        total_regions: int,
        active_capacity: int,
        stream: cuda.CUstream,
    ):
        _csa_compact_decode_stage_flush_with_slots_kernel_sm90_gqa(
            mSum,
            mCount,
            mMean,
            mActiveRegionIds,
            mActiveSlotByRegion,
            mAllocationSuccess,
            mActiveNumerator,
            mDenominator,
            mMaxLogits,
            mActiveTokenK,
            mActiveTokenZ,
            mActiveTokenValidU8,
            mFlatSlot,
            mResetSlots,
            mTokenValidU8,
            mTokenPositions,
            mIndexK,
            mIndexZ,
            Int64(total_regions),
            Int32(active_capacity),
            head_dim,
            summaries_per_page,
            region_block_size,
        ).launch(
            grid=[source_rows, 1, 1],
            block=[_STAGE_THREADS_PER_BLOCK, 1, 1],
            stream=stream,
        )

    return _launch


def _fake_dynamic(tensor: torch.Tensor, *, leading_dim: int) -> cute.Tensor:
    return utils.make_fake_tensor_like_with_dynamic_dim(
        tensor,
        alignment=16,
        dynamic_layout_dim=leading_dim,
        dynamic_shape_dim=0,
    )


@functools.cache
def _get_compiled_csa_compact_prefill_update_with_slots_kernel(
    *,
    head_dim: int,
    summaries_per_page: int,
    sum_page_stride: int,
    count_page_stride: int,
    region_block_size: int,
    index_dtype: torch.dtype,
    device_key: tuple[str, int | None],
) -> cute.JitFunction:
    device = utils.device_from_cache_key(device_key)
    sum_tensor = torch.empty_strided(
        (1, 1, 1, head_dim),
        (sum_page_stride, head_dim, head_dim, 1),
        device=device,
        dtype=torch.float32,
    )
    count_tensor = torch.empty_strided(
        (1, 1, 1),
        (count_page_stride, 1, 1),
        device=device,
        dtype=torch.float32,
    )
    mean_tensor = torch.empty(
        (1, summaries_per_page, 1, head_dim),
        device=device,
        dtype=torch.uint8,
    )
    active_region_tensor = torch.empty((1,), device=device, dtype=torch.int64)
    active_slot_tensor = torch.empty((1,), device=device, dtype=torch.int32)
    allocation_success_tensor = torch.empty((1,), device=device, dtype=torch.int32)
    active_accumulator_tensor = torch.empty(
        (1, 1, head_dim), device=device, dtype=torch.float32
    )
    max_tensor = torch.empty((1, 1), device=device, dtype=torch.float32)
    metadata_i64 = torch.empty((1,), device=device, dtype=torch.int64)
    metadata_i32 = torch.empty((1,), device=device, dtype=torch.int32)
    valid_u8 = torch.empty((1,), device=device, dtype=torch.uint8)
    index_tensor = torch.empty((1, 1, head_dim), device=device, dtype=index_dtype)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    launch = _make_launch_csa_compact_prefill_update_with_slots_kernel(
        head_dim=head_dim,
        summaries_per_page=summaries_per_page,
        region_block_size=region_block_size,
    )
    return cute.compile(
        launch,
        _fake_dynamic(sum_tensor, leading_dim=3),
        _fake_dynamic(count_tensor, leading_dim=2),
        _fake_dynamic(mean_tensor, leading_dim=3),
        _fake_dynamic(active_region_tensor, leading_dim=0),
        _fake_dynamic(active_slot_tensor, leading_dim=0),
        _fake_dynamic(allocation_success_tensor, leading_dim=0),
        _fake_dynamic(active_accumulator_tensor, leading_dim=2),
        _fake_dynamic(active_accumulator_tensor, leading_dim=2),
        _fake_dynamic(max_tensor, leading_dim=0),
        _fake_dynamic(metadata_i64, leading_dim=0),
        _fake_dynamic(metadata_i64, leading_dim=0),
        _fake_dynamic(valid_u8, leading_dim=0),
        _fake_dynamic(metadata_i64, leading_dim=0),
        _fake_dynamic(metadata_i32, leading_dim=0),
        _fake_dynamic(metadata_i32, leading_dim=0),
        _fake_dynamic(metadata_i32, leading_dim=0),
        _fake_dynamic(index_tensor, leading_dim=2),
        _fake_dynamic(index_tensor, leading_dim=2),
        1,
        1,
        1,
        1,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


def prewarm_csa_compact_prefill_update_with_slots_sm90_gqa(
    *,
    device: torch.device | str,
    index_dtype: torch.dtype,
    head_dim: int,
    summaries_per_page: int,
    sum_page_stride: int,
    count_page_stride: int,
    region_block_size: int,
) -> None:
    """Compile the grouped-prefill CSA update for one cache geometry.

    This is intentionally a compile-only entry point. Callers should pass the
    exact cache strides and index dtype that the runtime update will use so the
    first real prefill cannot trigger CuTeDSL compilation after engine startup.
    """
    resolved_device = torch.device(device)
    head_dim = int(head_dim)
    summaries_per_page = int(summaries_per_page)
    sum_page_stride = int(sum_page_stride)
    count_page_stride = int(count_page_stride)
    region_block_size = int(region_block_size)
    if resolved_device.type != "cuda":
        raise ValueError(
            "CSA grouped-prefill prewarm requires a CUDA device, got "
            f"{resolved_device}."
        )
    if head_dim != 256:
        raise ValueError(
            f"CSA grouped-prefill prewarm requires proxy_dim=256, got {head_dim}."
        )
    if summaries_per_page <= 0 or summaries_per_page % 8 != 0:
        raise ValueError(
            "CSA grouped-prefill prewarm requires summaries_per_page to be a "
            f"positive multiple of 8, got {summaries_per_page}."
        )
    if sum_page_stride < head_dim or count_page_stride <= 0:
        raise ValueError(
            "CSA grouped-prefill prewarm requires positive cache strides, got "
            f"sum_page_stride={sum_page_stride}, "
            f"count_page_stride={count_page_stride}."
        )
    if region_block_size != _DECODE_STAGE_TOKENS:
        raise ValueError(
            "CSA grouped-prefill prewarm is specialized for "
            f"region_block_size={_DECODE_STAGE_TOKENS}, got "
            f"{region_block_size}."
        )
    if index_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            "CSA grouped-prefill prewarm requires fp16/bf16/fp32 index dtype, "
            f"got {index_dtype}."
        )
    _get_compiled_csa_compact_prefill_update_with_slots_kernel(
        head_dim=head_dim,
        summaries_per_page=summaries_per_page,
        sum_page_stride=sum_page_stride,
        count_page_stride=count_page_stride,
        region_block_size=region_block_size,
        index_dtype=index_dtype,
        device_key=utils.device_cache_key(resolved_device),
    )


@functools.cache
def _get_compiled_csa_compact_decode_update_with_slots_kernel(
    *,
    head_dim: int,
    summaries_per_page: int,
    sum_page_stride: int,
    count_page_stride: int,
    region_block_size: int,
    index_dtype: torch.dtype,
    device_key: tuple[str, int | None],
) -> cute.JitFunction:
    device = utils.device_from_cache_key(device_key)
    sum_tensor = torch.empty_strided(
        (1, summaries_per_page, 1, head_dim),
        (sum_page_stride, head_dim, head_dim, 1),
        device=device,
        dtype=torch.float32,
    )
    count_tensor = torch.empty_strided(
        (1, summaries_per_page, 1),
        (count_page_stride, 1, 1),
        device=device,
        dtype=torch.float32,
    )
    active_region_tensor = torch.empty((1,), device=device, dtype=torch.int64)
    active_slot_tensor = torch.empty((1,), device=device, dtype=torch.int32)
    active_accumulator_tensor = torch.empty(
        (1, 1, head_dim), device=device, dtype=torch.float32
    )
    max_tensor = torch.empty((1, 1), device=device, dtype=torch.float32)
    metadata_i64 = torch.empty((1,), device=device, dtype=torch.int64)
    valid_u8 = torch.empty((1,), device=device, dtype=torch.uint8)
    index_tensor = torch.empty((1, 1, head_dim), device=device, dtype=index_dtype)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    launch = _make_launch_csa_compact_decode_update_with_slots_kernel(
        head_dim=head_dim,
        summaries_per_page=summaries_per_page,
        region_block_size=region_block_size,
    )
    return cute.compile(
        launch,
        _fake_dynamic(sum_tensor, leading_dim=3),
        _fake_dynamic(count_tensor, leading_dim=2),
        _fake_dynamic(active_region_tensor, leading_dim=0),
        _fake_dynamic(active_slot_tensor, leading_dim=0),
        _fake_dynamic(active_accumulator_tensor, leading_dim=2),
        _fake_dynamic(active_accumulator_tensor, leading_dim=2),
        _fake_dynamic(max_tensor, leading_dim=0),
        _fake_dynamic(metadata_i64, leading_dim=0),
        _fake_dynamic(metadata_i64, leading_dim=0),
        _fake_dynamic(valid_u8, leading_dim=0),
        _fake_dynamic(metadata_i64, leading_dim=0),
        _fake_dynamic(index_tensor, leading_dim=2),
        _fake_dynamic(index_tensor, leading_dim=2),
        1,
        1,
        1,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


@functools.cache
def _get_compiled_csa_compact_decode_stage_flush_with_slots_kernel(
    *,
    head_dim: int,
    summaries_per_page: int,
    sum_page_stride: int,
    count_page_stride: int,
    region_block_size: int,
    index_dtype: torch.dtype,
    stage_dtype: torch.dtype,
    device_key: tuple[str, int | None],
) -> cute.JitFunction:
    device = utils.device_from_cache_key(device_key)
    sum_tensor = torch.empty_strided(
        (1, 1, 1, head_dim),
        (sum_page_stride, head_dim, head_dim, 1),
        device=device,
        dtype=torch.float32,
    )
    count_tensor = torch.empty_strided(
        (1, 1, 1),
        (count_page_stride, 1, 1),
        device=device,
        dtype=torch.float32,
    )
    mean_tensor = torch.empty(
        (1, summaries_per_page, 1, head_dim),
        device=device,
        dtype=torch.uint8,
    )
    active_region_tensor = torch.empty((1,), device=device, dtype=torch.int64)
    active_slot_tensor = torch.empty((1,), device=device, dtype=torch.int32)
    allocation_success_tensor = torch.empty((1,), device=device, dtype=torch.int32)
    active_accumulator_tensor = torch.empty(
        (1, 1, head_dim), device=device, dtype=torch.float32
    )
    max_tensor = torch.empty((1, 1), device=device, dtype=torch.float32)
    stage_tensor = torch.empty(
        (1, region_block_size, 1, head_dim), device=device, dtype=stage_dtype
    )
    stage_valid_u8 = torch.empty(
        (1, region_block_size), device=device, dtype=torch.uint8
    )
    metadata_i64 = torch.empty((1,), device=device, dtype=torch.int64)
    valid_u8 = torch.empty((1,), device=device, dtype=torch.uint8)
    index_tensor = torch.empty((1, 1, head_dim), device=device, dtype=index_dtype)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    launch = _make_launch_csa_compact_decode_stage_flush_with_slots_kernel(
        head_dim=head_dim,
        summaries_per_page=summaries_per_page,
        region_block_size=region_block_size,
    )
    return cute.compile(
        launch,
        _fake_dynamic(sum_tensor, leading_dim=3),
        _fake_dynamic(count_tensor, leading_dim=2),
        _fake_dynamic(mean_tensor, leading_dim=3),
        _fake_dynamic(active_region_tensor, leading_dim=0),
        _fake_dynamic(active_slot_tensor, leading_dim=0),
        _fake_dynamic(allocation_success_tensor, leading_dim=0),
        _fake_dynamic(active_accumulator_tensor, leading_dim=2),
        _fake_dynamic(active_accumulator_tensor, leading_dim=2),
        _fake_dynamic(max_tensor, leading_dim=0),
        _fake_dynamic(stage_tensor, leading_dim=3),
        _fake_dynamic(stage_tensor, leading_dim=3),
        _fake_dynamic(stage_valid_u8, leading_dim=1),
        _fake_dynamic(metadata_i64, leading_dim=0),
        _fake_dynamic(metadata_i64, leading_dim=0),
        _fake_dynamic(valid_u8, leading_dim=0),
        _fake_dynamic(metadata_i64, leading_dim=0),
        _fake_dynamic(index_tensor, leading_dim=2),
        _fake_dynamic(index_tensor, leading_dim=2),
        1,
        1,
        1,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


@functools.cache
def _get_compiled_csa_compact_update_kernel(
    *,
    num_kv_heads: int,
    head_dim: int,
    summaries_per_page: int,
    sum_page_stride: int,
    count_page_stride: int,
    region_block_size: int,
    index_dtype: torch.dtype,
    device_key: tuple[str, int | None],
) -> cute.JitFunction:
    device = utils.device_from_cache_key(device_key)
    sum_tensor = torch.empty_strided(
        (1, summaries_per_page, num_kv_heads, head_dim),
        (
            sum_page_stride,
            num_kv_heads * head_dim,
            head_dim,
            1,
        ),
        device=device,
        dtype=torch.float32,
    )
    count_tensor = torch.empty_strided(
        (1, summaries_per_page, num_kv_heads),
        (count_page_stride, num_kv_heads, 1),
        device=device,
        dtype=torch.float32,
    )
    active_tensor = torch.empty((1,), device=device, dtype=torch.int64)
    denominator_tensor = torch.empty(
        (1, num_kv_heads, head_dim), device=device, dtype=torch.float32
    )
    max_tensor = torch.empty((1, num_kv_heads), device=device, dtype=torch.float32)
    metadata_i64 = torch.empty((1,), device=device, dtype=torch.int64)
    valid_u8 = torch.empty((1,), device=device, dtype=torch.uint8)
    index_tensor = torch.empty(
        (1, num_kv_heads, head_dim), device=device, dtype=index_dtype
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    launch = _make_launch_csa_compact_update_kernel(
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        summaries_per_page=summaries_per_page,
        region_block_size=region_block_size,
    )
    return cute.compile(
        launch,
        _fake_dynamic(sum_tensor, leading_dim=3),
        _fake_dynamic(count_tensor, leading_dim=2),
        _fake_dynamic(active_tensor, leading_dim=0),
        _fake_dynamic(denominator_tensor, leading_dim=2),
        _fake_dynamic(max_tensor, leading_dim=1),
        _fake_dynamic(metadata_i64, leading_dim=0),
        _fake_dynamic(metadata_i64, leading_dim=0),
        _fake_dynamic(valid_u8, leading_dim=0),
        _fake_dynamic(metadata_i64, leading_dim=0),
        _fake_dynamic(index_tensor, leading_dim=2),
        _fake_dynamic(index_tensor, leading_dim=2),
        1,
        1,
        1,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


def _validate_csa_compact_update_inputs(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    active_region_ids: torch.Tensor,
    denominator: torch.Tensor,
    max_logits: torch.Tensor,
    flat_slot: torch.Tensor,
    reset_slots: torch.Tensor,
    token_valid: torch.Tensor,
    token_positions: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    region_block_size: int,
) -> tuple[int, int, int, int]:
    if sum_cache.ndim != 4:
        raise ValueError(f"sum_cache must be [P,R,H,D], got {tuple(sum_cache.shape)}")
    num_pages, summaries_per_page, num_kv_heads, head_dim = (
        int(v) for v in sum_cache.shape
    )
    total_regions = num_pages * summaries_per_page
    if index_k.ndim != 3 or index_z.ndim != 3:
        raise ValueError(
            "CSA compact index_k/index_z must be [S,H,D], "
            f"got {tuple(index_k.shape)} and {tuple(index_z.shape)}"
        )
    source_rows = int(index_k.shape[0])
    expected_index_shape = (source_rows, num_kv_heads, head_dim)
    if (
        tuple(index_k.shape) != expected_index_shape
        or tuple(index_z.shape) != expected_index_shape
    ):
        raise ValueError(
            "CSA compact index_k/index_z shape mismatch: "
            f"index_k={tuple(index_k.shape)}, index_z={tuple(index_z.shape)}, "
            f"expected={expected_index_shape}"
        )
    if index_k.dtype != index_z.dtype or index_k.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise TypeError(
            "CSA compact index_k/index_z must have matching fp16/bf16/fp32 dtype, "
            f"got {index_k.dtype} and {index_z.dtype}"
        )
    if sum_cache.dtype != torch.float32 or count_cache.dtype != torch.float32:
        raise TypeError("CSA compact sum_cache/count_cache must be float32")
    if denominator.dtype != torch.float32 or max_logits.dtype != torch.float32:
        raise TypeError("CSA compact denominator/max_logits must be float32")
    if active_region_ids.dtype != torch.int64:
        raise TypeError("CSA compact active_region_ids must be int64")
    for name, tensor in (
        ("flat_slot", flat_slot),
        ("reset_slots", reset_slots),
        ("token_positions", token_positions),
    ):
        if tensor.dtype != torch.int64:
            raise TypeError(f"CSA compact {name} must be int64, got {tensor.dtype}")
    if token_valid.dtype != torch.bool:
        raise TypeError(
            f"CSA compact token_valid must be bool, got {token_valid.dtype}"
        )
    if tuple(count_cache.shape) != (num_pages, summaries_per_page, num_kv_heads):
        raise ValueError("CSA compact count_cache shape mismatch")
    expected_sum_inner_stride = (
        num_kv_heads * head_dim,
        head_dim,
        1,
    )
    if tuple(int(v) for v in sum_cache.stride()[1:]) != expected_sum_inner_stride:
        raise ValueError(
            "CSA compact sum_cache must be contiguous within each page, "
            f"got stride={sum_cache.stride()}"
        )
    if int(sum_cache.stride(0)) < summaries_per_page * num_kv_heads * head_dim:
        raise ValueError(
            "CSA compact sum_cache page stride overlaps adjacent pages, "
            f"got stride={sum_cache.stride()}"
        )
    expected_count_inner_stride = (num_kv_heads, 1)
    if tuple(int(v) for v in count_cache.stride()[1:]) != expected_count_inner_stride:
        raise ValueError(
            "CSA compact count_cache must be contiguous within each page, "
            f"got stride={count_cache.stride()}"
        )
    if int(count_cache.stride(0)) < summaries_per_page * num_kv_heads:
        raise ValueError(
            "CSA compact count_cache page stride overlaps adjacent pages, "
            f"got stride={count_cache.stride()}"
        )
    if tuple(denominator.shape) != (
        int(active_region_ids.numel()),
        num_kv_heads,
        head_dim,
    ):
        raise ValueError("CSA compact denominator shape mismatch")
    if tuple(max_logits.shape) != (
        int(active_region_ids.numel()),
        num_kv_heads,
    ):
        raise ValueError(
            "CSA compact max_logits must be [active_slot, kv_head] scalar state, "
            f"got {tuple(max_logits.shape)}"
        )
    if int(active_region_ids.numel()) <= 0:
        raise ValueError("CSA compact active_region_ids must have positive capacity")
    if (
        min(
            int(flat_slot.numel()),
            int(reset_slots.numel()),
            int(token_valid.numel()),
            int(token_positions.numel()),
        )
        < source_rows
    ):
        raise ValueError("CSA compact metadata tensors are shorter than index_k")
    if head_dim <= 0 or head_dim > _MAX_PROXY_DIM:
        raise NotImplementedError(
            "CSA compact CuTeDSL supports proxy_dim in "
            f"[1, {_MAX_PROXY_DIM}], got {head_dim}"
        )
    region_block_size = int(region_block_size)
    if region_block_size <= 0 or region_block_size > 32:
        raise NotImplementedError(
            "CSA compact CuTeDSL supports region_block_size in [1, 32], "
            f"got {region_block_size}"
        )
    device = sum_cache.device
    if device.type != "cuda":
        raise RuntimeError("CSA compact CuTeDSL requires CUDA tensors")
    for name, tensor in (
        ("sum_cache", sum_cache),
        ("count_cache", count_cache),
        ("active_region_ids", active_region_ids),
        ("denominator", denominator),
        ("max_logits", max_logits),
        ("flat_slot", flat_slot),
        ("reset_slots", reset_slots),
        ("token_valid", token_valid),
        ("token_positions", token_positions),
        ("index_k", index_k),
        ("index_z", index_z),
    ):
        if tensor.device != device:
            raise ValueError(
                f"CSA compact {name} must be on {device}, got {tensor.device}"
            )
        if name not in ("sum_cache", "count_cache") and not tensor.is_contiguous():
            raise ValueError(f"CSA compact {name} must be contiguous")
        if int(tensor.numel()) > 0 and int(tensor.data_ptr()) % 16 != 0:
            raise ValueError(f"CSA compact {name} must be 16-byte aligned")
    return source_rows, total_regions, num_kv_heads, head_dim


def _validate_csa_compact_decode_update_with_slots_inputs(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    active_region_ids: torch.Tensor,
    active_slot_by_region: torch.Tensor,
    active_numerator: torch.Tensor,
    denominator: torch.Tensor,
    max_logits: torch.Tensor,
    flat_slot: torch.Tensor,
    reset_slots: torch.Tensor,
    token_valid: torch.Tensor,
    token_positions: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    region_block_size: int,
) -> tuple[int, int, int]:
    source_rows, total_regions, num_kv_heads, head_dim = (
        _validate_csa_compact_update_inputs(
            sum_cache,
            count_cache,
            active_region_ids,
            denominator,
            max_logits,
            flat_slot,
            reset_slots,
            token_valid,
            token_positions,
            index_k,
            index_z,
            region_block_size,
        )
    )
    if num_kv_heads != 1:
        raise NotImplementedError(
            "CSA compact decode with slots requires one local KV head, "
            f"got {num_kv_heads}"
        )
    if head_dim > _MAX_DECODE_PROXY_DIM:
        raise NotImplementedError(
            "CSA compact decode with slots supports proxy_dim <= "
            f"{_MAX_DECODE_PROXY_DIM}, got {head_dim}"
        )
    if active_slot_by_region.dtype != torch.int32:
        raise TypeError(
            "CSA compact active_slot_by_region must be int32, "
            f"got {active_slot_by_region.dtype}"
        )
    if int(active_slot_by_region.numel()) < total_regions:
        raise ValueError(
            "CSA compact active_slot_by_region is shorter than total regions: "
            f"{int(active_slot_by_region.numel())} < {total_regions}"
        )
    if active_numerator.dtype != torch.float32:
        raise TypeError(
            f"CSA compact active_numerator must be float32, got {active_numerator.dtype}"
        )
    if tuple(active_numerator.shape) != tuple(denominator.shape):
        raise ValueError(
            "CSA compact active_numerator shape must match denominator: "
            f"{tuple(active_numerator.shape)} != {tuple(denominator.shape)}"
        )
    for name, tensor in (
        ("active_slot_by_region", active_slot_by_region),
        ("active_numerator", active_numerator),
    ):
        if tensor.device != sum_cache.device:
            raise ValueError(
                f"CSA compact {name} must be on {sum_cache.device}, got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"CSA compact {name} must be contiguous")
        if int(tensor.numel()) > 0 and int(tensor.data_ptr()) % 16 != 0:
            raise ValueError(f"CSA compact {name} must be 16-byte aligned")
    return source_rows, total_regions, head_dim


def csa_compact_update_sm90_gqa(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    active_region_ids: torch.Tensor,
    denominator: torch.Tensor,
    max_logits: torch.Tensor,
    flat_slot: torch.Tensor,
    reset_slots: torch.Tensor,
    token_valid: torch.Tensor,
    token_positions: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    region_block_size: int,
    *,
    stream=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if stream is not None:
        raise ValueError("CSA CuTeDSL kernels use the TVM-FFI environment stream")
    source_rows, total_regions, num_kv_heads, head_dim = (
        _validate_csa_compact_update_inputs(
            sum_cache,
            count_cache,
            active_region_ids,
            denominator,
            max_logits,
            flat_slot,
            reset_slots,
            token_valid,
            token_positions,
            index_k,
            index_z,
            region_block_size,
        )
    )
    if source_rows == 0:
        return sum_cache, count_cache, denominator, max_logits
    valid_u8 = token_valid[:source_rows].view(torch.uint8)
    flat_slot = flat_slot[:source_rows]
    reset_slots = reset_slots[:source_rows]
    token_positions = token_positions[:source_rows]
    compiled = _get_compiled_csa_compact_update_kernel(
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        summaries_per_page=int(sum_cache.shape[1]),
        sum_page_stride=int(sum_cache.stride(0)),
        count_page_stride=int(count_cache.stride(0)),
        region_block_size=int(region_block_size),
        index_dtype=index_k.dtype,
        device_key=utils.device_cache_key(sum_cache.device),
    )
    compiled(
        sum_cache,
        count_cache,
        active_region_ids,
        denominator,
        max_logits,
        flat_slot,
        reset_slots,
        valid_u8,
        token_positions,
        index_k,
        index_z,
        source_rows,
        total_regions,
        int(active_region_ids.numel()),
    )
    return sum_cache, count_cache, denominator, max_logits


def csa_compact_prefill_update_with_slots_prevalidated_sm90_gqa(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    mean_cache: torch.Tensor,
    active_region_ids: torch.Tensor,
    active_slot_by_region: torch.Tensor,
    active_numerator: torch.Tensor,
    denominator: torch.Tensor,
    max_logits: torch.Tensor,
    flat_slot: torch.Tensor,
    reset_slots: torch.Tensor,
    token_valid: torch.Tensor,
    token_positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    task_prefix: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    region_block_size: int,
    allocation_success: torch.Tensor,
    *,
    stream=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if stream is not None:
        raise ValueError("CSA CuTeDSL kernels use the TVM-FFI environment stream")
    if int(region_block_size) != _DECODE_STAGE_TOKENS:
        raise ValueError(
            "CSA grouped prefill is specialized for region_block_size=8, got "
            f"{int(region_block_size)}"
        )
    if int(sum_cache.shape[2]) != 1 or int(sum_cache.shape[3]) != 256:
        raise ValueError(
            "CSA grouped prefill requires one local KV head and proxy_dim=256, "
            f"got sum_cache={tuple(sum_cache.shape)}"
        )
    if mean_cache.dtype != torch.uint8:
        raise ValueError(f"CSA FP8 mean_cache must be uint8, got {mean_cache.dtype}")
    if tuple(sum_cache.shape[1:]) != (1, 1, 256):
        raise ValueError(
            "CSA grouped prefill requires active sum_cache shape "
            f"[active_capacity, 1, 1, 256], got {tuple(sum_cache.shape)}"
        )
    if tuple(count_cache.shape[1:]) != (1, 1):
        raise ValueError(
            "CSA grouped prefill requires active count_cache shape "
            f"[active_capacity, 1, 1], got {tuple(count_cache.shape)}"
        )
    if tuple(mean_cache.shape[2:]) != tuple(sum_cache.shape[2:]):
        raise ValueError(
            "CSA FP8 mean_cache head shape must match sum_cache, got "
            f"mean_cache={tuple(mean_cache.shape)}, sum_cache={tuple(sum_cache.shape)}"
        )
    if int(mean_cache.shape[1]) <= 0 or int(mean_cache.shape[1]) % 8 != 0:
        raise ValueError(
            "CSA interleaved FP8 mean_cache requires summaries_per_page to be "
            f"a positive multiple of 8, got {int(mean_cache.shape[1])}"
        )
    if mean_cache.device != sum_cache.device or not mean_cache.is_contiguous():
        raise ValueError(
            "CSA FP8 mean_cache must be contiguous on the sum_cache device"
        )
    if (
        allocation_success.dtype != torch.int32
        or allocation_success.device != sum_cache.device
        or tuple(allocation_success.shape) != (1,)
        or not allocation_success.is_contiguous()
    ):
        raise ValueError(
            "CSA grouped prefill requires contiguous int32 "
            "allocation_success=[1] on the cache device"
        )
    if (
        query_start_loc.dtype != torch.int32
        or query_start_loc.device != sum_cache.device
        or query_start_loc.ndim != 1
        or not query_start_loc.is_contiguous()
    ):
        raise ValueError(
            "CSA grouped prefill requires contiguous int32 query_start_loc "
            "on the cache device"
        )
    if (
        seq_lens.dtype != torch.int32
        or seq_lens.device != sum_cache.device
        or seq_lens.ndim != 1
        or not seq_lens.is_contiguous()
    ):
        raise ValueError(
            "CSA grouped prefill requires contiguous int32 seq_lens on the cache device"
        )
    num_reqs = int(query_start_loc.shape[0]) - 1
    if num_reqs <= 0 or int(seq_lens.shape[0]) < num_reqs:
        raise ValueError(
            "CSA grouped prefill requires query_start_loc=[B+1] and "
            f"seq_lens=[B], got {tuple(query_start_loc.shape)} and "
            f"{tuple(seq_lens.shape)}"
        )
    if (
        task_prefix.dtype != torch.int32
        or task_prefix.device != sum_cache.device
        or task_prefix.ndim != 1
        or int(task_prefix.shape[0]) < num_reqs + 1
        or not task_prefix.is_contiguous()
    ):
        raise ValueError(
            "CSA grouped prefill requires caller-owned contiguous int32 "
            f"task_prefix=[B+1], got {tuple(task_prefix.shape)} "
            f"dtype={task_prefix.dtype} device={task_prefix.device}"
        )
    source_rows = int(index_k.shape[0])
    if source_rows == 0:
        return sum_cache, count_cache, denominator, max_logits
    compiled = _get_compiled_csa_compact_prefill_update_with_slots_kernel(
        head_dim=int(sum_cache.shape[3]),
        summaries_per_page=int(mean_cache.shape[1]),
        sum_page_stride=int(sum_cache.stride(0)),
        count_page_stride=int(count_cache.stride(0)),
        region_block_size=int(region_block_size),
        index_dtype=index_k.dtype,
        device_key=utils.device_cache_key(sum_cache.device),
    )
    compiled(
        sum_cache,
        count_cache,
        mean_cache,
        active_region_ids,
        active_slot_by_region,
        allocation_success,
        active_numerator,
        denominator,
        max_logits,
        flat_slot,
        reset_slots,
        token_valid.view(torch.uint8),
        token_positions,
        query_start_loc,
        seq_lens,
        task_prefix,
        index_k,
        index_z,
        source_rows,
        num_reqs,
        int(mean_cache.shape[0] * mean_cache.shape[1]),
        int(active_region_ids.numel()),
    )
    return sum_cache, count_cache, denominator, max_logits


def csa_compact_decode_update_sm90_gqa(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    active_region_ids: torch.Tensor,
    denominator: torch.Tensor,
    max_logits: torch.Tensor,
    flat_slot: torch.Tensor,
    reset_slots: torch.Tensor,
    token_valid: torch.Tensor,
    token_positions: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    region_block_size: int,
    *,
    stream=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return csa_compact_update_sm90_gqa(
        sum_cache,
        count_cache,
        active_region_ids,
        denominator,
        max_logits,
        flat_slot,
        reset_slots,
        token_valid,
        token_positions,
        index_k,
        index_z,
        region_block_size,
        stream=stream,
    )


def _launch_csa_compact_decode_update_with_slots_sm90_gqa(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    active_region_ids: torch.Tensor,
    active_slot_by_region: torch.Tensor,
    active_numerator: torch.Tensor,
    denominator: torch.Tensor,
    max_logits: torch.Tensor,
    flat_slot: torch.Tensor,
    reset_slots: torch.Tensor,
    token_valid: torch.Tensor,
    token_positions: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    region_block_size: int,
    *,
    source_rows: int,
    total_regions: int,
    head_dim: int,
    stream,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if stream is not None:
        raise ValueError("CSA CuTeDSL kernels use the TVM-FFI environment stream")
    if source_rows == 0:
        return sum_cache, count_cache, denominator, max_logits
    valid_u8 = token_valid.view(torch.uint8)
    compiled = _get_compiled_csa_compact_decode_update_with_slots_kernel(
        head_dim=head_dim,
        summaries_per_page=int(sum_cache.shape[1]),
        sum_page_stride=int(sum_cache.stride(0)),
        count_page_stride=int(count_cache.stride(0)),
        region_block_size=int(region_block_size),
        index_dtype=index_k.dtype,
        device_key=utils.device_cache_key(sum_cache.device),
    )
    compiled(
        sum_cache,
        count_cache,
        active_region_ids,
        active_slot_by_region,
        active_numerator,
        denominator,
        max_logits,
        flat_slot,
        reset_slots,
        valid_u8,
        token_positions,
        index_k,
        index_z,
        source_rows,
        total_regions,
        int(active_region_ids.numel()),
    )
    return sum_cache, count_cache, denominator, max_logits


def csa_compact_decode_update_with_slots_sm90_gqa(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    active_region_ids: torch.Tensor,
    active_slot_by_region: torch.Tensor,
    active_numerator: torch.Tensor,
    denominator: torch.Tensor,
    max_logits: torch.Tensor,
    flat_slot: torch.Tensor,
    reset_slots: torch.Tensor,
    token_valid: torch.Tensor,
    token_positions: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    region_block_size: int,
    *,
    stream=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source_rows, total_regions, head_dim = (
        _validate_csa_compact_decode_update_with_slots_inputs(
            sum_cache,
            count_cache,
            active_region_ids,
            active_slot_by_region,
            active_numerator,
            denominator,
            max_logits,
            flat_slot,
            reset_slots,
            token_valid,
            token_positions,
            index_k,
            index_z,
            region_block_size,
        )
    )
    return _launch_csa_compact_decode_update_with_slots_sm90_gqa(
        sum_cache,
        count_cache,
        active_region_ids,
        active_slot_by_region,
        active_numerator,
        denominator,
        max_logits,
        flat_slot,
        reset_slots,
        token_valid,
        token_positions,
        index_k,
        index_z,
        region_block_size,
        source_rows=source_rows,
        total_regions=total_regions,
        head_dim=head_dim,
        stream=stream,
    )


def csa_compact_decode_update_with_slots_prevalidated_sm90_gqa(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    active_region_ids: torch.Tensor,
    active_slot_by_region: torch.Tensor,
    active_numerator: torch.Tensor,
    denominator: torch.Tensor,
    max_logits: torch.Tensor,
    flat_slot: torch.Tensor,
    reset_slots: torch.Tensor,
    token_valid: torch.Tensor,
    token_positions: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    region_block_size: int,
    *,
    stream=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # vLLM validates these tensors before dispatch and retains every buffer for
    # the lifetime of the decode graph.
    return _launch_csa_compact_decode_update_with_slots_sm90_gqa(
        sum_cache,
        count_cache,
        active_region_ids,
        active_slot_by_region,
        active_numerator,
        denominator,
        max_logits,
        flat_slot,
        reset_slots,
        token_valid,
        token_positions,
        index_k,
        index_z,
        region_block_size,
        source_rows=int(index_k.shape[0]),
        total_regions=int(sum_cache.shape[0] * sum_cache.shape[1]),
        head_dim=int(sum_cache.shape[3]),
        stream=stream,
    )


def csa_compact_decode_stage_flush_with_slots_prevalidated_sm90_gqa(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    mean_cache: torch.Tensor,
    active_region_ids: torch.Tensor,
    active_slot_by_region: torch.Tensor,
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
    region_block_size: int,
    allocation_success: torch.Tensor,
    *,
    stream=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if stream is not None:
        raise ValueError("CSA CuTeDSL kernels use the TVM-FFI environment stream")
    if int(region_block_size) != _DECODE_STAGE_TOKENS:
        raise ValueError(
            "CSA decode staging is specialized for region_block_size=8, got "
            f"{int(region_block_size)}"
        )
    if mean_cache.dtype != torch.uint8:
        raise ValueError(f"CSA FP8 mean_cache must be uint8, got {mean_cache.dtype}")
    if tuple(sum_cache.shape[1:]) != (1, 1, 256):
        raise ValueError(
            "CSA decode staging requires active sum_cache shape "
            f"[active_capacity, 1, 1, 256], got {tuple(sum_cache.shape)}"
        )
    if tuple(count_cache.shape[1:]) != (1, 1):
        raise ValueError(
            "CSA decode staging requires active count_cache shape "
            f"[active_capacity, 1, 1], got {tuple(count_cache.shape)}"
        )
    if tuple(mean_cache.shape[2:]) != tuple(sum_cache.shape[2:]):
        raise ValueError(
            "CSA FP8 mean_cache head shape must match sum_cache, got "
            f"mean_cache={tuple(mean_cache.shape)}, sum_cache={tuple(sum_cache.shape)}"
        )
    if int(mean_cache.shape[1]) <= 0 or int(mean_cache.shape[1]) % 8 != 0:
        raise ValueError(
            "CSA interleaved FP8 mean_cache requires summaries_per_page to be "
            f"a positive multiple of 8, got {int(mean_cache.shape[1])}"
        )
    if mean_cache.device != sum_cache.device or not mean_cache.is_contiguous():
        raise ValueError(
            "CSA FP8 mean_cache must be contiguous on the sum_cache device"
        )
    if (
        allocation_success.dtype != torch.int32
        or allocation_success.device != sum_cache.device
        or tuple(allocation_success.shape) != (1,)
        or not allocation_success.is_contiguous()
    ):
        raise ValueError(
            "CSA decode staging requires contiguous int32 "
            "allocation_success=[1] on the cache device"
        )
    source_rows = int(index_k.shape[0])
    if source_rows == 0:
        return sum_cache, count_cache, denominator, max_logits
    compiled = _get_compiled_csa_compact_decode_stage_flush_with_slots_kernel(
        head_dim=int(sum_cache.shape[3]),
        summaries_per_page=int(mean_cache.shape[1]),
        sum_page_stride=int(sum_cache.stride(0)),
        count_page_stride=int(count_cache.stride(0)),
        region_block_size=int(region_block_size),
        index_dtype=index_k.dtype,
        stage_dtype=active_token_k.dtype,
        device_key=utils.device_cache_key(sum_cache.device),
    )
    compiled(
        sum_cache,
        count_cache,
        mean_cache,
        active_region_ids,
        active_slot_by_region,
        allocation_success,
        active_numerator,
        denominator,
        max_logits,
        active_token_k,
        active_token_z,
        active_token_valid.view(torch.uint8),
        flat_slot,
        reset_slots,
        token_valid.view(torch.uint8),
        token_positions,
        index_k,
        index_z,
        source_rows,
        int(mean_cache.shape[0] * mean_cache.shape[1]),
        int(active_region_ids.numel()),
    )
    return sum_cache, count_cache, denominator, max_logits


__all__ = [
    "csa_compact_decode_stage_flush_with_slots_prevalidated_sm90_gqa",
    "csa_compact_decode_update_sm90_gqa",
    "csa_compact_decode_update_with_slots_prevalidated_sm90_gqa",
    "csa_compact_decode_update_with_slots_sm90_gqa",
    "csa_compact_prefill_update_with_slots_prevalidated_sm90_gqa",
    "csa_compact_update_sm90_gqa",
    "prewarm_csa_compact_prefill_update_with_slots_sm90_gqa",
]
