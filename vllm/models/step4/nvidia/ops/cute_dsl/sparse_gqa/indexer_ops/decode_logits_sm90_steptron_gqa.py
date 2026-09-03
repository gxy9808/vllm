# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import operator
from typing import Optional

import torch

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int16, Int32, Uint8
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.kernel_utils import cute_utils as utils
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.kernel_utils.fp_compat import cvt_f32_to_e4m3
from vllm.models.step4.nvidia.ops.cute_dsl.utils import (
    elem_pointer,
    cvt_e4m3x2_to_f32x2,
    cvt_fp32x2_to_e4m3x2,
    warp_reduce,
    pack_uint8x2_to_int16,
)


_HEAD_DIM = 256
_Q_HEADS_PER_KV = 2
_SUPPORTED_Q_HEADS_PER_KV = (2, 4)
_MAX_Q_HEADS_PER_KV = 4
_REGIONS_PER_BLOCK = 128
_THREADS_PER_BLOCK = _REGIONS_PER_BLOCK
_WARP_REGIONS_PER_BLOCK = 16
_WARP_REGIONS_PER_THREAD_WARP = 2
_WARP_REGION_LANES = 16
_WARP_TILE_ITERS = 16
_RERANK_WARP_TILE_ITERS = 2
_WARP_THREADS_PER_BLOCK = (_WARP_REGIONS_PER_BLOCK // _WARP_REGIONS_PER_THREAD_WARP) * cute.arch.WARP_SIZE


@cute.jit
def _wgmma_sw128_physical_coord(
    logical_dim: Int32,
    logical_region: cutlass.Int64,
    page_slot: Int32,
) -> tuple[Int32, Int32]:
    """Map one logical FP8 summary element to the interleaved page layout."""
    atom = logical_dim // Int32(128)
    atom_dim = logical_dim - atom * Int32(128)
    chunk = atom_dim // Int32(16)
    byte = atom_dim - chunk * Int32(16)
    row_in_block = page_slot - (page_slot // Int32(8)) * Int32(8)
    block_in_page = page_slot // Int32(8)
    offset = (
        block_in_page * Int32(8 * 256)
        + atom * Int32(8 * 128)
        + row_in_block * Int32(128)
        + (chunk ^ Int32(logical_region % cutlass.Int64(8))) * Int32(16)
        + byte
    )
    mean_slot = offset // Int32(_HEAD_DIM)
    mean_dim = offset - mean_slot * Int32(_HEAD_DIM)
    return mean_slot, mean_dim

@dsl_user_op
def _mul_rn_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _add_rn_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _round_f32_to_e4m3_f32_pair(
    val0: Float32,
    val1: Float32,
) -> tuple[Float32, Float32]:
    packed = cvt_fp32x2_to_e4m3x2(val1, val0)
    return cvt_e4m3x2_to_f32x2(packed)


@cute.jit
def _round_f32_to_e4m3_f32(val: Float32) -> Float32:
    rounded, _ = _round_f32_to_e4m3_f32_pair(val, Float32(0.0))
    return rounded


@cute.jit
def _load_e4m3_u8_as_f32(val: Uint8) -> Float32:
    packed = pack_uint8x2_to_int16(val, Uint8(0))
    rounded, _ = cvt_e4m3x2_to_f32x2(packed)
    return rounded


@cute.jit
def _load_e4m3_u8_pair_as_f32(
    val0: Uint8,
    val1: Uint8,
) -> tuple[Float32, Float32]:
    packed = pack_uint8x2_to_int16(val0, val1)
    return cvt_e4m3x2_to_f32x2(packed)


@cute.jit
def _load_e4m3_u8_pair_global_as_f32(
    tensor: cute.Tensor,
    coord: tuple[Int32, Int32, Int32, Int32],
) -> tuple[Float32, Float32]:
    ptr = elem_pointer(tensor, coord)
    int_ptr = cute.make_ptr(
        Int16,
        ptr.toint(),
        tensor.memspace,
        assumed_align=2,
    )
    src = cute.make_tensor(int_ptr, cute.make_layout((1,), stride=(1,)))
    return cvt_e4m3x2_to_f32x2(Int16(src[0]))


@cute.kernel
def _decode_steptron_logits_kernel(
    mQ: cute.Tensor,
    mW: cute.Tensor,
    mK: cute.Tensor,
    mOut: cute.Tensor,
    batch_size: Int32,
    num_regions: Int32,
    q_heads_per_kv: cutlass.Constexpr[int],
):
    tile_idx, batch_idx, _ = cute.arch.block_idx()
    tx = cute.arch.thread_idx()[0]
    region_idx = tile_idx * Int32(_REGIONS_PER_BLOCK) + tx

    if batch_idx < batch_size and region_idx < num_regions:
        w = [Float32(0.0) for _ in range(_MAX_Q_HEADS_PER_KV)]
        dot = [Float32(0.0) for _ in range(_MAX_Q_HEADS_PER_KV)]
        for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
            valid_h = Int32(h) < Int32(q_heads_per_kv)
            safe_h = cutlass.select_(valid_h, Int32(h), Int32(0))
            w[h] = cutlass.select_(
                valid_h,
                Float32(mW[batch_idx, Int32(0), safe_h]),
                Float32(0.0),
            )
        for dim in cutlass.range_constexpr(_HEAD_DIM):
            k = _round_f32_to_e4m3_f32(
                Float32(mK[batch_idx, region_idx, Int32(0), Int32(dim)])
            )
            for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
                valid_h = Int32(h) < Int32(q_heads_per_kv)
                safe_h = cutlass.select_(valid_h, Int32(h), Int32(0))
                q = _round_f32_to_e4m3_f32(
                    Float32(mQ[batch_idx, Int32(0), safe_h, Int32(dim)])
                )
                dot[h] += cutlass.select_(valid_h, q * k, Float32(0.0))
        acc = Float32(0.0)
        for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
            relu = cutlass.select_(dot[h] > Float32(0.0), dot[h], Float32(0.0))
            acc = _add_rn_f32(acc, _mul_rn_f32(relu, w[h]))
        mOut[batch_idx, region_idx] = mOut.element_type(acc)



@cute.kernel
def _decode_steptron_logits_warp_kernel(
    mQ: cute.Tensor,
    mW: cute.Tensor,
    mK: cute.Tensor,
    mOut: cute.Tensor,
    batch_size: Int32,
    num_regions: Int32,
    q_heads_per_kv: cutlass.Constexpr[int],
):
    tile_idx, batch_idx, _ = cute.arch.block_idx()
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()
    tx = cute.arch.thread_idx()[0]
    half_idx = lane_idx // Int32(_WARP_REGION_LANES)
    lane_in_half = lane_idx - half_idx * Int32(_WARP_REGION_LANES)
    tile_region_base = tile_idx * Int32(_WARP_REGIONS_PER_BLOCK * _WARP_TILE_ITERS)

    @cute.struct
    class SharedStorage:
        q_cache: cute.struct.Align[
            cute.struct.MemRange[Float32, _MAX_Q_HEADS_PER_KV * _HEAD_DIM],
            128,
        ]

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sQ = storage.q_cache.get_tensor(
        cute.make_layout((_MAX_Q_HEADS_PER_KV, _HEAD_DIM), stride=(_HEAD_DIM, 1))
    )

    if batch_idx < batch_size and tx < Int32(_HEAD_DIM):
        dim = tx
        for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
            valid_h = Int32(h) < Int32(q_heads_per_kv)
            safe_h = cutlass.select_(valid_h, Int32(h), Int32(0))
            if valid_h:
                q = _round_f32_to_e4m3_f32(
                    Float32(mQ[batch_idx, Int32(0), safe_h, dim])
                )
                sQ[safe_h, dim] = q
    cute.arch.sync_threads()

    if batch_idx < batch_size:
        w = [Float32(0.0) for _ in range(_MAX_Q_HEADS_PER_KV)]
        for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
            valid_h = Int32(h) < Int32(q_heads_per_kv)
            safe_h = cutlass.select_(valid_h, Int32(h), Int32(0))
            w[h] = cutlass.select_(
                valid_h,
                Float32(mW[batch_idx, Int32(0), safe_h]),
                Float32(0.0),
            )
        for tile_iter in cutlass.range_constexpr(_WARP_TILE_ITERS):
            region_idx = (
                tile_region_base
                + Int32(tile_iter * _WARP_REGIONS_PER_BLOCK)
                + warp_idx * Int32(_WARP_REGIONS_PER_THREAD_WARP)
                + half_idx
            )
            if region_idx < num_regions:
                dot = [Float32(0.0) for _ in range(_MAX_Q_HEADS_PER_KV)]
                for dim_group in cutlass.range_constexpr(
                    _HEAD_DIM // (_WARP_REGION_LANES * 2)
                ):
                    dim0 = Int32(dim_group * _WARP_REGION_LANES * 2) + lane_in_half * Int32(2)
                    dim1 = dim0 + Int32(1)
                    if cutlass.const_expr(mK.element_type == Uint8):
                        k0, k1 = _load_e4m3_u8_pair_global_as_f32(
                            mK, (batch_idx, region_idx, Int32(0), dim0)
                        )
                    else:
                        k0, k1 = _round_f32_to_e4m3_f32_pair(
                            Float32(mK[batch_idx, region_idx, Int32(0), dim0]),
                            Float32(mK[batch_idx, region_idx, Int32(0), dim1]),
                        )
                    for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
                        valid_h = Int32(h) < Int32(q_heads_per_kv)
                        safe_h = cutlass.select_(valid_h, Int32(h), Int32(0))
                        v = sQ[safe_h, dim0] * k0 + sQ[safe_h, dim1] * k1
                        dot[h] += cutlass.select_(valid_h, v, Float32(0.0))
                for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
                    dot[h] = warp_reduce(dot[h], operator.add, width=_WARP_REGION_LANES)
                if lane_in_half == Int32(0):
                    acc = Float32(0.0)
                    for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
                        relu = cutlass.select_(dot[h] > Float32(0.0), dot[h], Float32(0.0))
                        acc = _add_rn_f32(acc, _mul_rn_f32(relu, w[h]))
                    mOut[batch_idx, region_idx] = mOut.element_type(acc)


@cute.kernel
def _decode_steptron_logits_paged_summary_warp_kernel(
    mQ: cute.Tensor,
    mW: cute.Tensor,
    mSum: cute.Tensor,
    mCount: cute.Tensor,
    mBlockTable: cute.Tensor,
    mOut: cute.Tensor,
    batch_size: Int32,
    num_regions: Int32,
    num_pages: Int32,
    summaries_per_page: Int32,
    q_heads_per_kv: cutlass.Constexpr[int],
):
    tile_idx, batch_idx, _ = cute.arch.block_idx()
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()
    tx = cute.arch.thread_idx()[0]
    half_idx = lane_idx // Int32(_WARP_REGION_LANES)
    lane_in_half = lane_idx - half_idx * Int32(_WARP_REGION_LANES)
    tile_region_base = tile_idx * Int32(_WARP_REGIONS_PER_BLOCK * _WARP_TILE_ITERS)

    @cute.struct
    class SharedStorage:
        q_cache: cute.struct.Align[
            cute.struct.MemRange[Float32, _MAX_Q_HEADS_PER_KV * _HEAD_DIM],
            128,
        ]

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sQ = storage.q_cache.get_tensor(
        cute.make_layout((_MAX_Q_HEADS_PER_KV, _HEAD_DIM), stride=(_HEAD_DIM, 1))
    )

    if batch_idx < batch_size and tx < Int32(_HEAD_DIM):
        dim = tx
        for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
            valid_h = Int32(h) < Int32(q_heads_per_kv)
            safe_h = cutlass.select_(valid_h, Int32(h), Int32(0))
            if valid_h:
                q = _round_f32_to_e4m3_f32(
                    Float32(mQ[batch_idx, Int32(0), safe_h, dim])
                )
                sQ[safe_h, dim] = q
    cute.arch.sync_threads()

    if batch_idx < batch_size:
        w = [Float32(0.0) for _ in range(_MAX_Q_HEADS_PER_KV)]
        for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
            valid_h = Int32(h) < Int32(q_heads_per_kv)
            safe_h = cutlass.select_(valid_h, Int32(h), Int32(0))
            w[h] = cutlass.select_(
                valid_h,
                Float32(mW[batch_idx, Int32(0), safe_h]),
                Float32(0.0),
            )
        for tile_iter in cutlass.range_constexpr(_WARP_TILE_ITERS):
            region_idx = (
                tile_region_base
                + Int32(tile_iter * _WARP_REGIONS_PER_BLOCK)
                + warp_idx * Int32(_WARP_REGIONS_PER_THREAD_WARP)
                + half_idx
            )
            if region_idx < num_regions:
                page_col = region_idx // summaries_per_page
                summary_slot = region_idx - page_col * summaries_per_page
                phys_page = mBlockTable[batch_idx, page_col]
                page_valid = (phys_page >= Int32(0)) and (phys_page < num_pages)
                safe_page = cutlass.select_(page_valid, phys_page, Int32(0))
                denom = Float32(1.0)
                if page_valid:
                    denom = cute.arch.fmax(
                        Float32(mCount[safe_page, summary_slot, Int32(0)]),
                        Float32(1.0),
                    )
                dot = [Float32(0.0) for _ in range(_MAX_Q_HEADS_PER_KV)]
                for dim_group in cutlass.range_constexpr(
                    _HEAD_DIM // (_WARP_REGION_LANES * 2)
                ):
                    dim0 = Int32(dim_group * _WARP_REGION_LANES * 2) + lane_in_half * Int32(2)
                    dim1 = dim0 + Int32(1)
                    k0 = Float32(0.0)
                    k1 = Float32(0.0)
                    if page_valid:
                        mean0 = Float32(mSum[safe_page, summary_slot, Int32(0), dim0]) / denom
                        mean1 = Float32(mSum[safe_page, summary_slot, Int32(0), dim1]) / denom
                        if cutlass.const_expr(mQ.element_type != Float32):
                            mean0 = Float32(mQ.element_type(mean0))
                            mean1 = Float32(mQ.element_type(mean1))
                        k0, k1 = _round_f32_to_e4m3_f32_pair(mean0, mean1)
                    for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
                        valid_h = Int32(h) < Int32(q_heads_per_kv)
                        safe_h = cutlass.select_(valid_h, Int32(h), Int32(0))
                        v = sQ[safe_h, dim0] * k0 + sQ[safe_h, dim1] * k1
                        dot[h] += cutlass.select_(valid_h, v, Float32(0.0))
                for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
                    dot[h] = warp_reduce(dot[h], operator.add, width=_WARP_REGION_LANES)
                if lane_in_half == Int32(0):
                    acc = Float32(0.0)
                    for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
                        relu = cutlass.select_(dot[h] > Float32(0.0), dot[h], Float32(0.0))
                        acc = _add_rn_f32(acc, _mul_rn_f32(relu, w[h]))
                    mOut[batch_idx, region_idx] = mOut.element_type(acc)


@cute.kernel
def _decode_steptron_logits_paged_mean_warp_kernel(
    mQ: cute.Tensor,
    mW: cute.Tensor,
    mMean: cute.Tensor,
    mBlockTable: cute.Tensor,
    mRowReqIdx: cute.Tensor,
    mRowTableIdx: cute.Tensor,
    mOut: cute.Tensor,
    mValidRequests: cute.Tensor,
    mValidTokens: cute.Tensor,
    batch_size: Int32,
    num_regions: Int32,
    num_pages: Int32,
    summaries_per_page: Int32,
    q_heads_per_kv: cutlass.Constexpr[int],
):
    tile_idx, batch_idx, _ = cute.arch.block_idx()
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()
    tx = cute.arch.thread_idx()[0]
    half_idx = lane_idx // Int32(_WARP_REGION_LANES)
    lane_in_half = lane_idx - half_idx * Int32(_WARP_REGION_LANES)
    tile_region_base = tile_idx * Int32(_WARP_REGIONS_PER_BLOCK * _WARP_TILE_ITERS)

    @cute.struct
    class SharedStorage:
        q_cache: cute.struct.Align[
            cute.struct.MemRange[Float32, _MAX_Q_HEADS_PER_KV * _HEAD_DIM],
            128,
        ]

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sQ = storage.q_cache.get_tensor(
        cute.make_layout((_MAX_Q_HEADS_PER_KV, _HEAD_DIM), stride=(_HEAD_DIM, 1))
    )

    if batch_idx < batch_size and tx < Int32(_HEAD_DIM):
        dim = tx
        for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
            valid_h = Int32(h) < Int32(q_heads_per_kv)
            safe_h = cutlass.select_(valid_h, Int32(h), Int32(0))
            if valid_h:
                q = _round_f32_to_e4m3_f32(
                    Float32(mQ[batch_idx, Int32(0), safe_h, dim])
                )
                sQ[safe_h, dim] = q
    cute.arch.sync_threads()

    req_idx = Int32(-1)
    table_idx = Int32(-1)
    if batch_idx < batch_size:
        req_idx = Int32(mRowReqIdx[batch_idx])
        table_idx = Int32(mRowTableIdx[batch_idx])
    row_valid = (
        (batch_idx < batch_size)
        & (req_idx >= Int32(0))
        & (req_idx < Int32(mValidRequests[0]))
        & (table_idx >= Int32(0))
        & (table_idx < batch_size)
        & (batch_idx < Int32(mValidTokens[0]))
    )
    if row_valid:
        w = [Float32(0.0) for _ in range(_MAX_Q_HEADS_PER_KV)]
        for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
            valid_h = Int32(h) < Int32(q_heads_per_kv)
            safe_h = cutlass.select_(valid_h, Int32(h), Int32(0))
            w[h] = cutlass.select_(
                valid_h,
                Float32(mW[batch_idx, Int32(0), safe_h]),
                Float32(0.0),
            )
        for tile_iter in cutlass.range_constexpr(_WARP_TILE_ITERS):
            region_idx = (
                tile_region_base
                + Int32(tile_iter * _WARP_REGIONS_PER_BLOCK)
                + warp_idx * Int32(_WARP_REGIONS_PER_THREAD_WARP)
                + half_idx
            )
            if region_idx < num_regions:
                page_col = region_idx // summaries_per_page
                summary_slot = region_idx - page_col * summaries_per_page
                phys_page = mBlockTable[table_idx, page_col]
                page_valid = (phys_page >= Int32(0)) and (phys_page < num_pages)
                safe_page = cutlass.select_(page_valid, phys_page, Int32(0))
                dot = [Float32(0.0) for _ in range(_MAX_Q_HEADS_PER_KV)]
                for dim_group in cutlass.range_constexpr(
                    _HEAD_DIM // (_WARP_REGION_LANES * 2)
                ):
                    dim0 = Int32(dim_group * _WARP_REGION_LANES * 2) + lane_in_half * Int32(2)
                    dim1 = dim0 + Int32(1)
                    mean_slot0, mean_dim0 = _wgmma_sw128_physical_coord(
                        dim0, cutlass.Int64(region_idx), summary_slot)
                    mean_slot1, mean_dim1 = _wgmma_sw128_physical_coord(
                        dim1, cutlass.Int64(region_idx), summary_slot)
                    k0 = Float32(0.0)
                    k1 = Float32(0.0)
                    if page_valid:
                        if cutlass.const_expr(mMean.element_type == Uint8):
                            k0 = _load_e4m3_u8_as_f32(
                                mMean[safe_page, mean_slot0, Int32(0), mean_dim0])
                            k1 = _load_e4m3_u8_as_f32(
                                mMean[safe_page, mean_slot1, Int32(0), mean_dim1])
                        else:
                            k0, k1 = _round_f32_to_e4m3_f32_pair(
                                Float32(mMean[safe_page, mean_slot0, Int32(0), mean_dim0]),
                                Float32(mMean[safe_page, mean_slot1, Int32(0), mean_dim1]),
                            )
                    for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
                        valid_h = Int32(h) < Int32(q_heads_per_kv)
                        safe_h = cutlass.select_(valid_h, Int32(h), Int32(0))
                        v = sQ[safe_h, dim0] * k0 + sQ[safe_h, dim1] * k1
                        dot[h] += cutlass.select_(valid_h, v, Float32(0.0))
                for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
                    dot[h] = warp_reduce(dot[h], operator.add, width=_WARP_REGION_LANES)
                if lane_in_half == Int32(0):
                    acc = Float32(0.0)
                    for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
                        relu = cutlass.select_(dot[h] > Float32(0.0), dot[h], Float32(0.0))
                        acc = _add_rn_f32(acc, _mul_rn_f32(relu, w[h]))
                    mOut[batch_idx, region_idx] = mOut.element_type(acc)


@cute.kernel
def _rerank_steptron_logits_paged_mean_warp_kernel_v2(
    mQ: cute.Tensor,
    mW: cute.Tensor,
    mMean: cute.Tensor,
    mBlockTable: cute.Tensor,
    mRegionIds: cute.Tensor,
    mOut: cute.Tensor,
    batch_size: Int32,
    num_candidates: Int32,
    num_regions: Int32,
    num_pages: Int32,
    summaries_per_page: Int32,
    q_heads_per_kv: cutlass.Constexpr[int],
):
    tile_idx, batch_idx, _ = cute.arch.block_idx()
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()
    tx = cute.arch.thread_idx()[0]
    half_idx = lane_idx // Int32(_WARP_REGION_LANES)
    lane_in_half = lane_idx - half_idx * Int32(_WARP_REGION_LANES)
    tile_base = tile_idx * Int32(
        _WARP_REGIONS_PER_BLOCK * _RERANK_WARP_TILE_ITERS)

    @cute.struct
    class SharedStorage:
        q_cache: cute.struct.Align[
            cute.struct.MemRange[Float32, _MAX_Q_HEADS_PER_KV * _HEAD_DIM],
            128,
        ]

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sQ = storage.q_cache.get_tensor(
        cute.make_layout(
            (_MAX_Q_HEADS_PER_KV, _HEAD_DIM), stride=(_HEAD_DIM, 1)))

    if batch_idx < batch_size and tx < Int32(_HEAD_DIM):
        for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
            valid_h = Int32(h) < Int32(q_heads_per_kv)
            safe_h = cutlass.select_(valid_h, Int32(h), Int32(0))
            if valid_h:
                if cutlass.const_expr(mQ.element_type == Uint8):
                    q = _load_e4m3_u8_as_f32(
                        mQ[batch_idx, Int32(0), safe_h, tx])
                else:
                    q = _round_f32_to_e4m3_f32(
                        Float32(mQ[batch_idx, Int32(0), safe_h, tx]))
                sQ[safe_h, tx] = q
    cute.arch.sync_threads()

    if batch_idx < batch_size:
        weights = [Float32(0.0) for _ in range(_MAX_Q_HEADS_PER_KV)]
        for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
            valid_h = Int32(h) < Int32(q_heads_per_kv)
            safe_h = cutlass.select_(valid_h, Int32(h), Int32(0))
            weights[h] = cutlass.select_(
                valid_h,
                Float32(mW[batch_idx, Int32(0), safe_h]),
                Float32(0.0),
            )
        for tile_iter in cutlass.range_constexpr(_RERANK_WARP_TILE_ITERS):
            candidate_pos = (
                tile_base
                + Int32(tile_iter * _WARP_REGIONS_PER_BLOCK)
                + warp_idx * Int32(_WARP_REGIONS_PER_THREAD_WARP)
                + half_idx
            )
            if candidate_pos < num_candidates:
                region_idx = mRegionIds[batch_idx, candidate_pos]
                region_valid = (
                    (region_idx >= Int32(0)) & (region_idx < num_regions))
                safe_region = cutlass.select_(
                    region_valid, region_idx, Int32(0))
                page_col = safe_region // summaries_per_page
                summary_slot = safe_region - page_col * summaries_per_page
                phys_page = mBlockTable[batch_idx, page_col]
                page_valid = (
                    region_valid
                    & (phys_page >= Int32(0))
                    & (phys_page < num_pages)
                )
                safe_page = cutlass.select_(page_valid, phys_page, Int32(0))
                dot = [Float32(0.0) for _ in range(_MAX_Q_HEADS_PER_KV)]
                for dim_group in cutlass.range_constexpr(
                    _HEAD_DIM // (_WARP_REGION_LANES * 2)
                ):
                    dim0 = Int32(
                        dim_group * _WARP_REGION_LANES * 2
                    ) + lane_in_half * Int32(2)
                    dim1 = dim0 + Int32(1)
                    mean_slot0, mean_dim0 = _wgmma_sw128_physical_coord(
                        dim0, cutlass.Int64(safe_region), summary_slot)
                    mean_slot1, mean_dim1 = _wgmma_sw128_physical_coord(
                        dim1, cutlass.Int64(safe_region), summary_slot)
                    k0 = Float32(0.0)
                    k1 = Float32(0.0)
                    if page_valid:
                        k0 = _load_e4m3_u8_as_f32(
                            mMean[safe_page, mean_slot0, Int32(0), mean_dim0])
                        k1 = _load_e4m3_u8_as_f32(
                            mMean[safe_page, mean_slot1, Int32(0), mean_dim1])
                    for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
                        valid_h = Int32(h) < Int32(q_heads_per_kv)
                        safe_h = cutlass.select_(valid_h, Int32(h), Int32(0))
                        value = sQ[safe_h, dim0] * k0 + sQ[safe_h, dim1] * k1
                        dot[h] += cutlass.select_(
                            valid_h, value, Float32(0.0))
                for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
                    dot[h] = warp_reduce(
                        dot[h], operator.add, width=_WARP_REGION_LANES)
                if lane_in_half == Int32(0):
                    score = Float32(0.0)
                    for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
                        relu = cutlass.select_(
                            dot[h] > Float32(0.0), dot[h], Float32(0.0))
                        score = _add_rn_f32(
                            score, _mul_rn_f32(relu, weights[h]))
                    if page_valid:
                        mOut[batch_idx, safe_region] = mOut.element_type(score)


@cute.kernel
def _materialize_steptron_paged_mean_cache_kernel(
    mSum: cute.Tensor,
    mCount: cute.Tensor,
    mMean: cute.Tensor,
    total_slots: Int32,
):
    slot_idx, _, _ = cute.arch.block_idx()
    tx = cute.arch.thread_idx()[0]
    if slot_idx < total_slots and tx < Int32(_HEAD_DIM // 2):
        page_idx = slot_idx // Int32(mSum.shape[1])
        summary_slot = slot_idx - page_idx * Int32(mSum.shape[1])
        dim0 = tx * Int32(2)
        dim1 = dim0 + Int32(1)
        denom = cute.arch.fmax(
            Float32(mCount[page_idx, summary_slot, Int32(0)]),
            Float32(1.0),
        )
        mean0 = Float32(mSum[page_idx, summary_slot, Int32(0), dim0]) / denom
        mean1 = Float32(mSum[page_idx, summary_slot, Int32(0), dim1]) / denom
        logical_region = cutlass.Int64(slot_idx)
        mean_slot0, mean_dim0 = _wgmma_sw128_physical_coord(
            dim0, logical_region, summary_slot)
        mean_slot1, mean_dim1 = _wgmma_sw128_physical_coord(
            dim1, logical_region, summary_slot)
        if cutlass.const_expr(mMean.element_type == Uint8):
            # Note(wangbojun/codex): Match the baseline decode kernel's
            # FP32 mean -> BF16 -> E4M3 conversion exactly. Skipping the
            # BF16 step changes E4M3 choices at midpoint boundaries.
            mean0 = Float32(BFloat16(mean0))
            mean1 = Float32(BFloat16(mean1))
            mMean[page_idx, mean_slot0, Int32(0), mean_dim0] = cvt_f32_to_e4m3(mean0).to(Uint8)
            mMean[page_idx, mean_slot1, Int32(0), mean_dim1] = cvt_f32_to_e4m3(mean1).to(Uint8)
        else:
            mMean[page_idx, mean_slot0, Int32(0), mean_dim0] = mMean.element_type(mean0)
            mMean[page_idx, mean_slot1, Int32(0), mean_dim1] = mMean.element_type(mean1)


@cute.kernel
def _materialize_steptron_selected_mean_cache_kernel(
    mSum: cute.Tensor,
    mCount: cute.Tensor,
    mMean: cute.Tensor,
    mRegionIds: cute.Tensor,
    num_regions: Int32,
):
    row = cute.arch.block_idx()[0]
    tx = cute.arch.thread_idx()[0]
    if row < num_regions and tx < Int32(_HEAD_DIM):
        region = Int32(mRegionIds[row])
        page_idx = region // Int32(mSum.shape[1])
        summary_slot = region - page_idx * Int32(mSum.shape[1])
        denom = cute.arch.fmax(
            Float32(mCount[page_idx, summary_slot, Int32(0)]),
            Float32(1.0),
        )
        mean = Float32(mSum[page_idx, summary_slot, Int32(0), tx]) / denom
        mean_slot, mean_dim = _wgmma_sw128_physical_coord(
            tx, cutlass.Int64(region), summary_slot)
        if cutlass.const_expr(mMean.element_type == Uint8):
            mean = Float32(BFloat16(mean))
            mMean[page_idx, mean_slot, Int32(0), mean_dim] = cvt_f32_to_e4m3(mean).to(Uint8)
        else:
            mMean[page_idx, mean_slot, Int32(0), mean_dim] = mMean.element_type(mean)


@cute.kernel
def _decode_steptron_logits_paged_summary_warp_splitk_partial_kernel(
    mQ: cute.Tensor,
    mSum: cute.Tensor,
    mCount: cute.Tensor,
    mBlockTable: cute.Tensor,
    mPartial: cute.Tensor,
    batch_size: Int32,
    num_regions: Int32,
    num_pages: Int32,
    summaries_per_page: Int32,
    split_k: cutlass.Constexpr[int],
    q_heads_per_kv: cutlass.Constexpr[int],
):
    tile_idx, batch_idx, split_idx = cute.arch.block_idx()
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()
    tx = cute.arch.thread_idx()[0]
    half_idx = lane_idx // Int32(_WARP_REGION_LANES)
    lane_in_half = lane_idx - half_idx * Int32(_WARP_REGION_LANES)
    tile_region_base = tile_idx * Int32(_WARP_REGIONS_PER_BLOCK * _WARP_TILE_ITERS)

    @cute.struct
    class SharedStorage:
        q_cache: cute.struct.Align[
            cute.struct.MemRange[Float32, _MAX_Q_HEADS_PER_KV * _HEAD_DIM],
            128,
        ]

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sQ = storage.q_cache.get_tensor(
        cute.make_layout((_MAX_Q_HEADS_PER_KV, _HEAD_DIM), stride=(_HEAD_DIM, 1))
    )

    if batch_idx < batch_size and tx < Int32(_HEAD_DIM):
        dim = tx
        for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
            valid_h = Int32(h) < Int32(q_heads_per_kv)
            safe_h = cutlass.select_(valid_h, Int32(h), Int32(0))
            if valid_h:
                q = _round_f32_to_e4m3_f32(
                    Float32(mQ[batch_idx, Int32(0), safe_h, dim])
                )
                sQ[safe_h, dim] = q
    cute.arch.sync_threads()

    if batch_idx < batch_size:
        for tile_iter in cutlass.range_constexpr(_WARP_TILE_ITERS):
            region_idx = (
                tile_region_base
                + Int32(tile_iter * _WARP_REGIONS_PER_BLOCK)
                + warp_idx * Int32(_WARP_REGIONS_PER_THREAD_WARP)
                + half_idx
            )
            if region_idx < num_regions:
                page_col = region_idx // summaries_per_page
                summary_slot = region_idx - page_col * summaries_per_page
                phys_page = mBlockTable[batch_idx, page_col]
                page_valid = (phys_page >= Int32(0)) and (phys_page < num_pages)
                safe_page = cutlass.select_(page_valid, phys_page, Int32(0))
                denom = Float32(1.0)
                if page_valid:
                    denom = cute.arch.fmax(
                        Float32(mCount[safe_page, summary_slot, Int32(0)]),
                        Float32(1.0),
                    )
                dot = [Float32(0.0) for _ in range(_MAX_Q_HEADS_PER_KV)]
                for dim_group in cutlass.range(
                    (_HEAD_DIM // split_k) // (_WARP_REGION_LANES * 2),
                    unroll=1,
                ):
                    dim0 = (
                        Int32(split_idx * (_HEAD_DIM // split_k))
                        + Int32(dim_group * _WARP_REGION_LANES * 2)
                        + lane_in_half * Int32(2)
                    )
                    dim1 = dim0 + Int32(1)
                    k0 = Float32(0.0)
                    k1 = Float32(0.0)
                    if page_valid:
                        mean0 = Float32(mSum[safe_page, summary_slot, Int32(0), dim0]) / denom
                        mean1 = Float32(mSum[safe_page, summary_slot, Int32(0), dim1]) / denom
                        if cutlass.const_expr(mQ.element_type != Float32):
                            mean0 = Float32(mQ.element_type(mean0))
                            mean1 = Float32(mQ.element_type(mean1))
                        k0, k1 = _round_f32_to_e4m3_f32_pair(mean0, mean1)
                    for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
                        valid_h = Int32(h) < Int32(q_heads_per_kv)
                        safe_h = cutlass.select_(valid_h, Int32(h), Int32(0))
                        v = sQ[safe_h, dim0] * k0 + sQ[safe_h, dim1] * k1
                        dot[h] += cutlass.select_(valid_h, v, Float32(0.0))
                for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
                    dot[h] = warp_reduce(dot[h], operator.add, width=_WARP_REGION_LANES)
                if lane_in_half == Int32(0):
                    for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
                        valid_h = Int32(h) < Int32(q_heads_per_kv)
                        safe_h = cutlass.select_(valid_h, Int32(h), Int32(0))
                        if valid_h:
                            mPartial[batch_idx, region_idx, split_idx, safe_h] = dot[h]


@cute.kernel
def _decode_steptron_logits_paged_summary_splitk_finalize_kernel(
    mW: cute.Tensor,
    mPartial: cute.Tensor,
    mOut: cute.Tensor,
    batch_size: Int32,
    num_regions: Int32,
    split_k: cutlass.Constexpr[int],
    q_heads_per_kv: cutlass.Constexpr[int],
):
    tile_idx, batch_idx, _ = cute.arch.block_idx()
    tx = cute.arch.thread_idx()[0]
    region_idx = tile_idx * Int32(_REGIONS_PER_BLOCK) + tx

    if batch_idx < batch_size and region_idx < num_regions:
        dot = [Float32(0.0) for _ in range(_MAX_Q_HEADS_PER_KV)]
        for si in cutlass.range(split_k, unroll=1):
            for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
                valid_h = Int32(h) < Int32(q_heads_per_kv)
                safe_h = cutlass.select_(valid_h, Int32(h), Int32(0))
                dot[h] += cutlass.select_(
                    valid_h,
                    Float32(mPartial[batch_idx, region_idx, si, safe_h]),
                    Float32(0.0),
                )
        acc = Float32(0.0)
        for h in cutlass.range_constexpr(_MAX_Q_HEADS_PER_KV):
            valid_h = Int32(h) < Int32(q_heads_per_kv)
            safe_h = cutlass.select_(valid_h, Int32(h), Int32(0))
            w = cutlass.select_(
                valid_h,
                Float32(mW[batch_idx, Int32(0), safe_h]),
                Float32(0.0),
            )
            relu = cutlass.select_(dot[h] > Float32(0.0), dot[h], Float32(0.0))
            acc = _add_rn_f32(acc, _mul_rn_f32(relu, w))
        mOut[batch_idx, region_idx] = mOut.element_type(acc)


def _make_launch_decode_steptron_logits_warp_kernel(q_heads_per_kv: int):
    @cute.jit
    def _launch_decode_steptron_logits_warp_kernel(
        mQ: cute.Tensor,
        mW: cute.Tensor,
        mK: cute.Tensor,
        mOut: cute.Tensor,
        batch_size: int,
        num_regions: int,
        stream,
    ):
        _decode_steptron_logits_warp_kernel(
            mQ,
            mW,
            mK,
            mOut,
            Int32(batch_size),
            Int32(num_regions),
            q_heads_per_kv,
        ).launch(
            grid=[
                cute.ceil_div(num_regions, _WARP_REGIONS_PER_BLOCK * _WARP_TILE_ITERS),
                batch_size,
                1,
            ],
            block=[_WARP_THREADS_PER_BLOCK, 1, 1],
            stream=stream,
        )

    return _launch_decode_steptron_logits_warp_kernel


def _make_launch_decode_steptron_logits_paged_summary_warp_kernel(q_heads_per_kv: int):
    @cute.jit
    def _launch_decode_steptron_logits_paged_summary_warp_kernel(
        mQ: cute.Tensor,
        mW: cute.Tensor,
        mSum: cute.Tensor,
        mCount: cute.Tensor,
        mBlockTable: cute.Tensor,
        mOut: cute.Tensor,
        batch_size: int,
        num_regions: int,
        num_pages: int,
        summaries_per_page: int,
        stream,
    ):
        _decode_steptron_logits_paged_summary_warp_kernel(
            mQ,
            mW,
            mSum,
            mCount,
            mBlockTable,
            mOut,
            Int32(batch_size),
            Int32(num_regions),
            Int32(num_pages),
            Int32(summaries_per_page),
            q_heads_per_kv,
        ).launch(
            grid=[
                cute.ceil_div(num_regions, _WARP_REGIONS_PER_BLOCK * _WARP_TILE_ITERS),
                batch_size,
                1,
            ],
            block=[_WARP_THREADS_PER_BLOCK, 1, 1],
            stream=stream,
        )

    return _launch_decode_steptron_logits_paged_summary_warp_kernel


def _make_launch_decode_steptron_logits_paged_mean_warp_kernel_v2(q_heads_per_kv: int):
    @cute.jit
    def _launch_decode_steptron_logits_paged_mean_warp_kernel(
        mQ: cute.Tensor,
        mW: cute.Tensor,
        mMean: cute.Tensor,
        mBlockTable: cute.Tensor,
        mRowReqIdx: cute.Tensor,
        mRowTableIdx: cute.Tensor,
        mOut: cute.Tensor,
        mValidRequests: cute.Tensor,
        mValidTokens: cute.Tensor,
        batch_size: int,
        num_regions: int,
        num_pages: int,
        summaries_per_page: int,
        stream,
    ):
        _decode_steptron_logits_paged_mean_warp_kernel(
            mQ,
            mW,
            mMean,
            mBlockTable,
            mRowReqIdx,
            mRowTableIdx,
            mOut,
            mValidRequests,
            mValidTokens,
            Int32(batch_size),
            Int32(num_regions),
            Int32(num_pages),
            Int32(summaries_per_page),
            q_heads_per_kv,
        ).launch(
            grid=[
                cute.ceil_div(num_regions, _WARP_REGIONS_PER_BLOCK * _WARP_TILE_ITERS),
                batch_size,
                1,
            ],
            block=[_WARP_THREADS_PER_BLOCK, 1, 1],
            stream=stream,
        )

    return _launch_decode_steptron_logits_paged_mean_warp_kernel


def _make_launch_rerank_steptron_logits_paged_mean_warp_kernel_v2(
    q_heads_per_kv: int,
):
    @cute.jit
    def _launch_rerank_steptron_logits_paged_mean_warp_kernel(
        mQ: cute.Tensor,
        mW: cute.Tensor,
        mMean: cute.Tensor,
        mBlockTable: cute.Tensor,
        mRegionIds: cute.Tensor,
        mOut: cute.Tensor,
        batch_size: int,
        num_candidates: int,
        num_regions: int,
        num_pages: int,
        summaries_per_page: int,
        stream,
    ):
        _rerank_steptron_logits_paged_mean_warp_kernel_v2(
            mQ,
            mW,
            mMean,
            mBlockTable,
            mRegionIds,
            mOut,
            Int32(batch_size),
            Int32(num_candidates),
            Int32(num_regions),
            Int32(num_pages),
            Int32(summaries_per_page),
            q_heads_per_kv,
        ).launch(
            grid=[
                cute.ceil_div(
                    num_candidates,
                    _WARP_REGIONS_PER_BLOCK * _RERANK_WARP_TILE_ITERS,
                ),
                batch_size,
                1,
            ],
            block=[_WARP_THREADS_PER_BLOCK, 1, 1],
            stream=stream,
        )

    return _launch_rerank_steptron_logits_paged_mean_warp_kernel


def _make_launch_materialize_steptron_paged_mean_cache_kernel():
    @cute.jit
    def _launch_materialize_steptron_paged_mean_cache_kernel(
        mSum: cute.Tensor,
        mCount: cute.Tensor,
        mMean: cute.Tensor,
        total_slots: int,
        stream,
    ):
        _materialize_steptron_paged_mean_cache_kernel(
            mSum,
            mCount,
            mMean,
            Int32(total_slots),
        ).launch(
            grid=[total_slots, 1, 1],
            block=[_HEAD_DIM // 2, 1, 1],
            stream=stream,
        )

    return _launch_materialize_steptron_paged_mean_cache_kernel


def _make_launch_materialize_steptron_selected_mean_cache_kernel():
    @cute.jit
    def _launch_materialize_steptron_selected_mean_cache_kernel(
        mSum: cute.Tensor,
        mCount: cute.Tensor,
        mMean: cute.Tensor,
        mRegionIds: cute.Tensor,
        num_regions: int,
        stream,
    ):
        _materialize_steptron_selected_mean_cache_kernel(
            mSum,
            mCount,
            mMean,
            mRegionIds,
            Int32(num_regions),
        ).launch(
            grid=[num_regions, 1, 1],
            block=[_HEAD_DIM, 1, 1],
            stream=stream,
        )

    return _launch_materialize_steptron_selected_mean_cache_kernel


def _make_launch_decode_steptron_logits_paged_summary_warp_splitk_kernel(
    split_k: int, q_heads_per_kv: int
):
    @cute.jit
    def _launch_decode_steptron_logits_paged_summary_warp_splitk_kernel(
        mQ: cute.Tensor,
        mW: cute.Tensor,
        mSum: cute.Tensor,
        mCount: cute.Tensor,
        mBlockTable: cute.Tensor,
        mPartial: cute.Tensor,
        mOut: cute.Tensor,
        batch_size: int,
        num_regions: int,
        num_pages: int,
        summaries_per_page: int,
        stream,
    ):
        _decode_steptron_logits_paged_summary_warp_splitk_partial_kernel(
            mQ,
            mSum,
            mCount,
            mBlockTable,
            mPartial,
            Int32(batch_size),
            Int32(num_regions),
            Int32(num_pages),
            Int32(summaries_per_page),
            split_k,
            q_heads_per_kv,
        ).launch(
            grid=[
                cute.ceil_div(num_regions, _WARP_REGIONS_PER_BLOCK * _WARP_TILE_ITERS),
                batch_size,
                split_k,
            ],
            block=[_WARP_THREADS_PER_BLOCK, 1, 1],
            stream=stream,
        )
        _decode_steptron_logits_paged_summary_splitk_finalize_kernel(
            mW,
            mPartial,
            mOut,
            Int32(batch_size),
            Int32(num_regions),
            split_k,
            q_heads_per_kv,
        ).launch(
            grid=[
                cute.ceil_div(num_regions, _REGIONS_PER_BLOCK),
                batch_size,
                1,
            ],
            block=[_THREADS_PER_BLOCK, 1, 1],
            stream=stream,
        )

    return _launch_decode_steptron_logits_paged_summary_warp_splitk_kernel


def _make_launch_decode_steptron_logits_kernel(q_heads_per_kv: int):
    @cute.jit
    def _launch_decode_steptron_logits_kernel(
        mQ: cute.Tensor,
        mW: cute.Tensor,
        mK: cute.Tensor,
        mOut: cute.Tensor,
        batch_size: int,
        num_regions: int,
        stream,
    ):
        _decode_steptron_logits_kernel(
            mQ,
            mW,
            mK,
            mOut,
            Int32(batch_size),
            Int32(num_regions),
            q_heads_per_kv,
        ).launch(
            grid=[
                cute.ceil_div(num_regions, _REGIONS_PER_BLOCK),
                batch_size,
                1,
            ],
            block=[_THREADS_PER_BLOCK, 1, 1],
            stream=stream,
        )

    return _launch_decode_steptron_logits_kernel


@functools.cache
def _get_compiled_decode_steptron_logits_kernel_for_shape(
    q_signature,
    weights_signature,
    summary_signature,
    out_signature,
    device_key: tuple[str, int | None],
) -> cute.JitFunction:
    device = utils.device_from_cache_key(device_key)
    q = utils.placeholder_from_signature(q_signature, device=device, dynamic_shape_fill=1)
    weights = utils.placeholder_from_signature(
        weights_signature, device=device, dynamic_shape_fill=1
    )
    summary = utils.placeholder_from_signature(
        summary_signature, device=device, dynamic_shape_fill=1
    )
    out = utils.placeholder_from_signature(out_signature, device=device, dynamic_shape_fill=1)
    launch_kernel = _make_launch_decode_steptron_logits_kernel(int(q.shape[2]))
    mQ = utils.make_fake_tensor_like_with_dynamic_dim(
        q, alignment=16, dynamic_shape_dims=(0,))
    mW = utils.make_fake_tensor_like_with_dynamic_dim(
        weights, alignment=16, dynamic_shape_dims=(0,))
    mK = utils.make_fake_tensor_like_with_dynamic_dim(
        summary,
        alignment=16,
        dynamic_shape_dims=(0, 1),
        dynamic_stride_dims=(0,),
    )
    mOut = utils.make_fake_tensor_like_with_dynamic_dim(
        out,
        alignment=16,
        dynamic_shape_dims=(0, 1),
        dynamic_stride_dims=(0,),
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        launch_kernel,
        mQ,
        mW,
        mK,
        mOut,
        1,
        1,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


def _get_compiled_decode_steptron_logits_kernel(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    summary: torch.Tensor,
    out: torch.Tensor,
) -> cute.JitFunction:
    return _get_compiled_decode_steptron_logits_kernel_for_shape(
        utils.tensor_signature_dynamic(index_q, dynamic_shape_dims=(0,)),
        utils.tensor_signature_dynamic(weights, dynamic_shape_dims=(0,)),
        utils.tensor_signature_dynamic(
            summary,
            dynamic_shape_dims=(0, 1),
            dynamic_stride_dims=(0,),
        ),
        utils.tensor_signature_dynamic(
            out,
            dynamic_shape_dims=(0, 1),
            dynamic_stride_dims=(0,),
        ),
        utils.device_cache_key(out.device),
    )



@functools.cache
def _get_compiled_decode_steptron_logits_warp_kernel_for_shape(
    q_signature,
    weights_signature,
    summary_signature,
    out_signature,
    device_key: tuple[str, int | None],
) -> cute.JitFunction:
    device = utils.device_from_cache_key(device_key)
    q = utils.placeholder_from_signature(q_signature, device=device, dynamic_shape_fill=1)
    weights = utils.placeholder_from_signature(weights_signature, device=device, dynamic_shape_fill=1)
    summary = utils.placeholder_from_signature(summary_signature, device=device, dynamic_shape_fill=1)
    out = utils.placeholder_from_signature(out_signature, device=device, dynamic_shape_fill=1)
    launch_kernel = _make_launch_decode_steptron_logits_warp_kernel(int(q.shape[2]))
    mQ = utils.make_fake_tensor_like_with_dynamic_dim(
        q, alignment=16, dynamic_shape_dims=(0,))
    mW = utils.make_fake_tensor_like_with_dynamic_dim(
        weights, alignment=16, dynamic_shape_dims=(0,))
    mK = utils.make_fake_tensor_like_with_dynamic_dim(
        summary, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,))
    mOut = utils.make_fake_tensor_like_with_dynamic_dim(
        out, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,))
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        launch_kernel,
        mQ,
        mW,
        mK,
        mOut,
        1,
        1,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


def _get_compiled_decode_steptron_logits_warp_kernel(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    summary: torch.Tensor,
    out: torch.Tensor,
) -> cute.JitFunction:
    return _get_compiled_decode_steptron_logits_warp_kernel_for_shape(
        utils.tensor_signature_dynamic(index_q, dynamic_shape_dims=(0,)),
        utils.tensor_signature_dynamic(weights, dynamic_shape_dims=(0,)),
        utils.tensor_signature_dynamic(
            summary, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)
        ),
        utils.tensor_signature_dynamic(
            out, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)
        ),
        utils.device_cache_key(out.device),
    )


@functools.cache
def _get_compiled_decode_steptron_logits_paged_summary_warp_kernel_for_shape(
    q_signature,
    weights_signature,
    sum_signature,
    count_signature,
    block_table_signature,
    out_signature,
    device_key: tuple[str, int | None],
) -> cute.JitFunction:
    device = utils.device_from_cache_key(device_key)
    q = utils.placeholder_from_signature(q_signature, device=device, dynamic_shape_fill=1)
    weights = utils.placeholder_from_signature(weights_signature, device=device, dynamic_shape_fill=1)
    sum_cache = utils.placeholder_from_signature(sum_signature, device=device, dynamic_shape_fill=1)
    count_cache = utils.placeholder_from_signature(count_signature, device=device, dynamic_shape_fill=1)
    block_table = utils.placeholder_from_signature(
        block_table_signature, device=device, dynamic_shape_fill=1
    )
    out = utils.placeholder_from_signature(out_signature, device=device, dynamic_shape_fill=1)
    launch_kernel = _make_launch_decode_steptron_logits_paged_summary_warp_kernel(int(q.shape[2]))
    mQ = utils.make_fake_tensor_like_with_dynamic_dim(
        q, alignment=16, dynamic_shape_dims=(0,))
    mW = utils.make_fake_tensor_like_with_dynamic_dim(
        weights, alignment=16, dynamic_shape_dims=(0,))
    mSum = utils.make_fake_tensor_like_with_dynamic_dim(
        sum_cache, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,))
    mCount = utils.make_fake_tensor_like_with_dynamic_dim(
        count_cache, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,))
    mBlockTable = utils.make_fake_tensor_like_with_dynamic_dim(
        block_table, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,))
    mOut = utils.make_fake_tensor_like_with_dynamic_dim(
        out, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        launch_kernel,
        mQ,
        mW,
        mSum,
        mCount,
        mBlockTable,
        mOut,
        1,
        1,
        1,
        1,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


def _get_compiled_decode_steptron_logits_paged_summary_warp_kernel(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    block_table: torch.Tensor,
    out: torch.Tensor,
) -> cute.JitFunction:
    return _get_compiled_decode_steptron_logits_paged_summary_warp_kernel_for_shape(
        utils.tensor_signature_dynamic(index_q, dynamic_shape_dims=(0,)),
        utils.tensor_signature_dynamic(weights, dynamic_shape_dims=(0,)),
        utils.tensor_signature_dynamic(
            sum_cache, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)
        ),
        utils.tensor_signature_dynamic(
            count_cache, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)
        ),
        utils.tensor_signature_dynamic(
            block_table, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)
        ),
        utils.tensor_signature_dynamic(
            out, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)
        ),
        utils.device_cache_key(out.device),
    )


@functools.cache
def _get_compiled_decode_steptron_logits_paged_mean_warp_kernel_for_shape(
    q_signature,
    weights_signature,
    mean_signature,
    block_table_signature,
    row_req_idx_signature,
    row_table_idx_signature,
    out_signature,
    valid_requests_signature,
    valid_tokens_signature,
    device_key: tuple[str, int | None],
) -> cute.JitFunction:
    device = utils.device_from_cache_key(device_key)
    q = utils.placeholder_from_signature(q_signature, device=device, dynamic_shape_fill=1)
    weights = utils.placeholder_from_signature(weights_signature, device=device, dynamic_shape_fill=1)
    mean = utils.placeholder_from_signature(mean_signature, device=device, dynamic_shape_fill=1)
    block_table = utils.placeholder_from_signature(
        block_table_signature, device=device, dynamic_shape_fill=1
    )
    row_req_idx = utils.placeholder_from_signature(
        row_req_idx_signature, device=device, dynamic_shape_fill=1
    )
    row_table_idx = utils.placeholder_from_signature(
        row_table_idx_signature, device=device, dynamic_shape_fill=1
    )
    out = utils.placeholder_from_signature(out_signature, device=device, dynamic_shape_fill=1)
    valid_requests = utils.placeholder_from_signature(
        valid_requests_signature, device=device, dynamic_shape_fill=1
    )
    valid_tokens = utils.placeholder_from_signature(
        valid_tokens_signature, device=device, dynamic_shape_fill=1
    )
    launch_kernel = _make_launch_decode_steptron_logits_paged_mean_warp_kernel_v2(
        int(q.shape[2])
    )
    mQ = utils.make_fake_tensor_like_with_dynamic_dim(
        q, alignment=16, dynamic_shape_dims=(0,))
    mW = utils.make_fake_tensor_like_with_dynamic_dim(
        weights, alignment=16, dynamic_shape_dims=(0,))
    mMean = utils.make_fake_tensor_like_with_dynamic_dim(
        mean, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,))
    mBlockTable = utils.make_fake_tensor_like_with_dynamic_dim(
        block_table, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,))
    mRowReqIdx = utils.make_fake_tensor_like_with_dynamic_dim(
        row_req_idx, alignment=4, dynamic_shape_dim=0)
    mRowTableIdx = utils.make_fake_tensor_like_with_dynamic_dim(
        row_table_idx, alignment=4, dynamic_shape_dim=0)
    mOut = utils.make_fake_tensor_like_with_dynamic_dim(
        out, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)
    )
    mValidRequests = utils.make_fake_tensor_like_with_dynamic_dim(
        valid_requests, alignment=4, dynamic_shape_dim=0
    )
    mValidTokens = utils.make_fake_tensor_like_with_dynamic_dim(
        valid_tokens, alignment=4, dynamic_shape_dim=0
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        launch_kernel,
        mQ,
        mW,
        mMean,
        mBlockTable,
        mRowReqIdx,
        mRowTableIdx,
        mOut,
        mValidRequests,
        mValidTokens,
        1,
        1,
        1,
        1,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


def _get_compiled_decode_steptron_logits_paged_mean_warp_kernel(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    mean_cache: torch.Tensor,
    block_table: torch.Tensor,
    out: torch.Tensor,
    *,
    row_req_idx: torch.Tensor | None = None,
    row_table_idx: torch.Tensor | None = None,
    valid_requests: torch.Tensor | None = None,
    valid_tokens: torch.Tensor | None = None,
) -> cute.JitFunction:
    batch_size = int(index_q.shape[0])
    if row_req_idx is None:
        row_req_idx = torch.arange(
            batch_size, dtype=torch.int32, device=index_q.device
        )
    if row_table_idx is None:
        row_table_idx = torch.arange(
            batch_size, dtype=torch.int32, device=index_q.device
        )
    if valid_requests is None:
        valid_requests = torch.full(
            (1,), batch_size, dtype=torch.int32, device=index_q.device
        )
    if valid_tokens is None:
        valid_tokens = torch.full(
            (1,), batch_size, dtype=torch.int32, device=index_q.device
        )
    return _get_compiled_decode_steptron_logits_paged_mean_warp_kernel_for_shape(
        utils.tensor_signature_dynamic(index_q, dynamic_shape_dims=(0,)),
        utils.tensor_signature_dynamic(weights, dynamic_shape_dims=(0,)),
        utils.tensor_signature_dynamic(
            mean_cache, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)
        ),
        utils.tensor_signature_dynamic(
            block_table, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)
        ),
        utils.tensor_signature_dynamic(row_req_idx, dynamic_shape_dims=(0,)),
        utils.tensor_signature_dynamic(row_table_idx, dynamic_shape_dims=(0,)),
        utils.tensor_signature_dynamic(
            out, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)
        ),
        utils.tensor_signature_dynamic(valid_requests, dynamic_shape_dims=(0,)),
        utils.tensor_signature_dynamic(valid_tokens, dynamic_shape_dims=(0,)),
        utils.device_cache_key(out.device),
    )


@functools.cache
def _get_compiled_rerank_steptron_logits_paged_mean_warp_kernel_for_shape(
    q_signature,
    weights_signature,
    mean_signature,
    block_table_signature,
    region_ids_signature,
    out_signature,
    device_key: tuple[str, int | None],
) -> cute.JitFunction:
    device = utils.device_from_cache_key(device_key)
    q = utils.placeholder_from_signature(q_signature, device=device, dynamic_shape_fill=1)
    weights = utils.placeholder_from_signature(weights_signature, device=device, dynamic_shape_fill=1)
    mean = utils.placeholder_from_signature(mean_signature, device=device, dynamic_shape_fill=1)
    block_table = utils.placeholder_from_signature(
        block_table_signature, device=device, dynamic_shape_fill=1
    )
    region_ids = utils.placeholder_from_signature(
        region_ids_signature, device=device, dynamic_shape_fill=1
    )
    out = utils.placeholder_from_signature(out_signature, device=device, dynamic_shape_fill=1)
    launch_kernel = _make_launch_rerank_steptron_logits_paged_mean_warp_kernel_v2(
        int(q.shape[2]))
    mQ = utils.make_fake_tensor_like_with_dynamic_dim(
        q, alignment=16, dynamic_shape_dims=(0,))
    mW = utils.make_fake_tensor_like_with_dynamic_dim(
        weights, alignment=16, dynamic_shape_dims=(0,))
    mMean = utils.make_fake_tensor_like_with_dynamic_dim(
        mean, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,))
    mBlockTable = utils.make_fake_tensor_like_with_dynamic_dim(
        block_table, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,))
    mRegionIds = utils.make_fake_tensor_like_with_dynamic_dim(
        region_ids, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,))
    mOut = utils.make_fake_tensor_like_with_dynamic_dim(
        out, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,))
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        launch_kernel,
        mQ,
        mW,
        mMean,
        mBlockTable,
        mRegionIds,
        mOut,
        1,
        1,
        1,
        1,
        1,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


def _get_compiled_rerank_steptron_logits_paged_mean_warp_kernel(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    mean_cache: torch.Tensor,
    block_table: torch.Tensor,
    region_ids: torch.Tensor,
    out: torch.Tensor,
) -> cute.JitFunction:
    return _get_compiled_rerank_steptron_logits_paged_mean_warp_kernel_for_shape(
        utils.tensor_signature_dynamic(index_q, dynamic_shape_dims=(0,)),
        utils.tensor_signature_dynamic(weights, dynamic_shape_dims=(0,)),
        utils.tensor_signature_dynamic(
            mean_cache, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)
        ),
        utils.tensor_signature_dynamic(
            block_table, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)
        ),
        utils.tensor_signature_dynamic(
            region_ids, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)
        ),
        utils.tensor_signature_dynamic(
            out, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)
        ),
        utils.device_cache_key(out.device),
    )


@functools.cache
def _get_compiled_materialize_steptron_paged_mean_cache_kernel_for_shape(
    sum_signature,
    count_signature,
    mean_signature,
    device_key: tuple[str, int | None],
) -> cute.JitFunction:
    device = utils.device_from_cache_key(device_key)
    launch_kernel = _make_launch_materialize_steptron_paged_mean_cache_kernel()
    sum_cache = utils.placeholder_from_signature(sum_signature, device=device, dynamic_shape_fill=1)
    count_cache = utils.placeholder_from_signature(count_signature, device=device, dynamic_shape_fill=1)
    mean_cache = utils.placeholder_from_signature(mean_signature, device=device, dynamic_shape_fill=1)
    mSum = utils.make_fake_tensor_like_with_dynamic_dim(
        sum_cache, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,))
    mCount = utils.make_fake_tensor_like_with_dynamic_dim(
        count_cache, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,))
    mMean = utils.make_fake_tensor_like_with_dynamic_dim(
        mean_cache, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,))
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        launch_kernel,
        mSum,
        mCount,
        mMean,
        1,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


def _get_compiled_materialize_steptron_paged_mean_cache_kernel(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    mean_cache: torch.Tensor,
) -> cute.JitFunction:
    return _get_compiled_materialize_steptron_paged_mean_cache_kernel_for_shape(
        utils.tensor_signature_dynamic(sum_cache, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)),
        utils.tensor_signature_dynamic(count_cache, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)),
        utils.tensor_signature_dynamic(mean_cache, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)),
        utils.device_cache_key(mean_cache.device),
    )


@functools.cache
def _get_compiled_materialize_steptron_selected_mean_cache_kernel_for_shape(
    sum_signature,
    count_signature,
    mean_signature,
    region_ids_signature,
    device_key: tuple[str, int | None],
) -> cute.JitFunction:
    device = utils.device_from_cache_key(device_key)
    launch_kernel = _make_launch_materialize_steptron_selected_mean_cache_kernel()
    sum_cache = utils.placeholder_from_signature(sum_signature, device=device, dynamic_shape_fill=1)
    count_cache = utils.placeholder_from_signature(count_signature, device=device, dynamic_shape_fill=1)
    mean_cache = utils.placeholder_from_signature(mean_signature, device=device, dynamic_shape_fill=1)
    region_ids = utils.placeholder_from_signature(
        region_ids_signature, device=device, dynamic_shape_fill=1
    )
    mSum = utils.make_fake_tensor_like_with_dynamic_dim(
        sum_cache, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,))
    mCount = utils.make_fake_tensor_like_with_dynamic_dim(
        count_cache, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,))
    mMean = utils.make_fake_tensor_like_with_dynamic_dim(
        mean_cache, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,))
    mRegionIds = utils.make_fake_tensor_like_with_dynamic_dim(
        region_ids,
        alignment=8 if region_ids.dtype == torch.int64 else 4,
        dynamic_shape_dims=(0,),
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        launch_kernel,
        mSum,
        mCount,
        mMean,
        mRegionIds,
        1,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


def _get_compiled_materialize_steptron_selected_mean_cache_kernel(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    mean_cache: torch.Tensor,
    region_ids: torch.Tensor,
) -> cute.JitFunction:
    return _get_compiled_materialize_steptron_selected_mean_cache_kernel_for_shape(
        utils.tensor_signature_dynamic(sum_cache, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)),
        utils.tensor_signature_dynamic(count_cache, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)),
        utils.tensor_signature_dynamic(mean_cache, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)),
        utils.tensor_signature_dynamic(region_ids, dynamic_shape_dims=(0,)),
        utils.device_cache_key(mean_cache.device),
    )


@functools.cache
def _get_compiled_decode_steptron_logits_paged_summary_warp_splitk_kernel_for_shape(
    q_signature,
    weights_signature,
    sum_signature,
    count_signature,
    block_table_signature,
    partial_signature,
    out_signature,
    split_k: int,
    device_key: tuple[str, int | None],
) -> cute.JitFunction:
    device = utils.device_from_cache_key(device_key)
    q = utils.placeholder_from_signature(q_signature, device=device, dynamic_shape_fill=1)
    weights = utils.placeholder_from_signature(weights_signature, device=device, dynamic_shape_fill=1)
    sum_cache = utils.placeholder_from_signature(sum_signature, device=device, dynamic_shape_fill=1)
    count_cache = utils.placeholder_from_signature(count_signature, device=device, dynamic_shape_fill=1)
    block_table = utils.placeholder_from_signature(block_table_signature, device=device, dynamic_shape_fill=1)
    partial = utils.placeholder_from_signature(partial_signature, device=device, dynamic_shape_fill=1)
    out = utils.placeholder_from_signature(out_signature, device=device, dynamic_shape_fill=1)
    launch_kernel = _make_launch_decode_steptron_logits_paged_summary_warp_splitk_kernel(split_k, int(q.shape[2]))
    mQ = utils.make_fake_tensor_like_with_dynamic_dim(
        q, alignment=16, dynamic_shape_dims=(0,))
    mW = utils.make_fake_tensor_like_with_dynamic_dim(
        weights, alignment=16, dynamic_shape_dims=(0,))
    mSum = utils.make_fake_tensor_like_with_dynamic_dim(
        sum_cache, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,))
    mCount = utils.make_fake_tensor_like_with_dynamic_dim(
        count_cache, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,))
    mBlockTable = utils.make_fake_tensor_like_with_dynamic_dim(
        block_table, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,))
    mPartial = utils.make_fake_tensor_like_with_dynamic_dim(
        partial, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,))
    mOut = utils.make_fake_tensor_like_with_dynamic_dim(
        out, alignment=16, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,))
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        launch_kernel,
        mQ,
        mW,
        mSum,
        mCount,
        mBlockTable,
        mPartial,
        mOut,
        1,
        1,
        1,
        1,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


def _get_compiled_decode_steptron_logits_paged_summary_warp_splitk_kernel(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    block_table: torch.Tensor,
    partial: torch.Tensor,
    out: torch.Tensor,
    split_k: int,
) -> cute.JitFunction:
    return _get_compiled_decode_steptron_logits_paged_summary_warp_splitk_kernel_for_shape(
        utils.tensor_signature_dynamic(index_q, dynamic_shape_dims=(0,)),
        utils.tensor_signature_dynamic(weights, dynamic_shape_dims=(0,)),
        utils.tensor_signature_dynamic(sum_cache, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)),
        utils.tensor_signature_dynamic(count_cache, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)),
        utils.tensor_signature_dynamic(block_table, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)),
        utils.tensor_signature_dynamic(partial, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)),
        utils.tensor_signature_dynamic(out, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)),
        int(split_k),
        utils.device_cache_key(out.device),
    )


def _decode_weighted_relu_logits_sum_sm90_steptron_gqa_impl(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    summary: torch.Tensor,
    out: torch.Tensor,
    stream=None,
    use_warp_kernel: bool = False,
) -> None:
    if stream is not None:
        raise ValueError("logits CuTeDSL kernels use the TVM-FFI environment stream")
    batch_size, proxy_kv_heads, q_heads_per_kv, head_dim_q = [int(v) for v in index_q.shape]
    summary_b, num_regions, summary_kv_heads, head_dim_k = [int(v) for v in summary.shape]
    if proxy_kv_heads != 1 or summary_kv_heads != 1:
        raise ValueError(
            "decode StepTron logits requires one proxy KV head, got "
            f"q={proxy_kv_heads}, summary={summary_kv_heads}"
        )
    if q_heads_per_kv not in _SUPPORTED_Q_HEADS_PER_KV:
        raise ValueError(
            f"decode StepTron logits requires q heads per KV in "
            f"{_SUPPORTED_Q_HEADS_PER_KV}, got {q_heads_per_kv}"
        )
    if head_dim_q != _HEAD_DIM or head_dim_k != _HEAD_DIM:
        raise ValueError(
            f"decode StepTron logits requires head_dim={_HEAD_DIM}, "
            f"got q={head_dim_q}, summary={head_dim_k}"
        )
    if summary_b != batch_size:
        raise ValueError(
            f"summary batch must match index_q batch, got {summary_b} vs {batch_size}"
        )
    if tuple(weights.shape) != (batch_size, proxy_kv_heads, q_heads_per_kv):
        raise ValueError(
            "weights shape must match index_q[:3], got "
            f"weights={tuple(weights.shape)}, index_q={tuple(index_q.shape)}"
        )
    if tuple(out.shape) != (batch_size, num_regions):
        raise ValueError(
            f"out must have shape {(batch_size, num_regions)}, got {tuple(out.shape)}"
        )
    if index_q.device != weights.device or index_q.device != summary.device or index_q.device != out.device:
        raise ValueError("index_q/weights/summary/out must live on the same device")
    if index_q.device.type != "cuda":
        raise RuntimeError("decode_weighted_relu_logits_sum_sm90_steptron_gqa requires CUDA tensors")
    if not index_q.is_contiguous() or not weights.is_contiguous() or not summary.is_contiguous() or not out.is_contiguous():
        raise ValueError(
            "decode_weighted_relu_logits_sum_sm90_steptron_gqa requires contiguous tensors, "
            f"q_stride={tuple(index_q.stride())}, weights_stride={tuple(weights.stride())}, "
            f"summary_stride={tuple(summary.stride())}, out_stride={tuple(out.stride())}"
        )
    if index_q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"index_q must be fp16/bf16/fp32, got {index_q.dtype}")
    if summary.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.uint8):
        raise ValueError(f"summary must be fp16/bf16/fp32/uint8-fp8bits, got {summary.dtype}")
    if weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"weights must be fp16/bf16/fp32, got {weights.dtype}")
    if out.dtype != torch.float32:
        raise ValueError(f"out must be torch.float32, got {out.dtype}")
    if batch_size <= 0 or num_regions <= 0:
        return

    if use_warp_kernel:
        compiled = _get_compiled_decode_steptron_logits_warp_kernel(
            index_q, weights, summary, out)
    else:
        compiled = _get_compiled_decode_steptron_logits_kernel(
            index_q, weights, summary, out)
    compiled(index_q, weights, summary, out, batch_size, num_regions)


def _decode_weighted_relu_logits_sum_paged_summary_warp_sm90_steptron_gqa_impl(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    block_table: torch.Tensor,
    out: torch.Tensor,
    stream=None,
) -> None:
    if stream is not None:
        raise ValueError("logits CuTeDSL kernels use the TVM-FFI environment stream")
    batch_size, proxy_kv_heads, q_heads_per_kv, head_dim_q = [int(v) for v in index_q.shape]
    num_pages, summaries_per_page, summary_kv_heads, head_dim_k = [int(v) for v in sum_cache.shape]
    if proxy_kv_heads != 1 or summary_kv_heads != 1:
        raise ValueError(
            "paged decode StepTron logits requires one proxy KV head, got "
            f"q={proxy_kv_heads}, summary={summary_kv_heads}"
        )
    if q_heads_per_kv not in _SUPPORTED_Q_HEADS_PER_KV:
        raise ValueError(
            f"paged decode StepTron logits requires q heads per KV in "
            f"{_SUPPORTED_Q_HEADS_PER_KV}, got {q_heads_per_kv}"
        )
    if head_dim_q != _HEAD_DIM or head_dim_k != _HEAD_DIM:
        raise ValueError(
            f"paged decode StepTron logits requires head_dim={_HEAD_DIM}, "
            f"got q={head_dim_q}, summary={head_dim_k}"
        )
    if tuple(count_cache.shape) != (num_pages, summaries_per_page, summary_kv_heads):
        raise ValueError(
            "count_cache shape must match sum_cache[:3], got "
            f"count_cache={tuple(count_cache.shape)}, sum_cache={tuple(sum_cache.shape)}"
        )
    if tuple(weights.shape) != (batch_size, proxy_kv_heads, q_heads_per_kv):
        raise ValueError(
            "weights shape must match index_q[:3], got "
            f"weights={tuple(weights.shape)}, index_q={tuple(index_q.shape)}"
        )
    if block_table.ndim != 2 or int(block_table.shape[0]) != batch_size:
        raise ValueError(
            "block_table must have shape [batch, pages], got "
            f"{tuple(block_table.shape)} for batch={batch_size}"
        )
    if out.ndim != 2 or int(out.shape[0]) != batch_size:
        raise ValueError(
            "out must have shape [batch, num_regions], got "
            f"{tuple(out.shape)} for batch={batch_size}"
        )
    if (
        index_q.device != weights.device
        or index_q.device != sum_cache.device
        or index_q.device != count_cache.device
        or index_q.device != block_table.device
        or index_q.device != out.device
    ):
        raise ValueError("paged decode logits inputs must live on the same device")
    if index_q.device.type != "cuda":
        raise RuntimeError(
            "decode_weighted_relu_logits_sum_paged_summary_warp_sm90_steptron_gqa "
            "requires CUDA tensors")
    if (
        not index_q.is_contiguous()
        or not weights.is_contiguous()
        or not block_table.is_contiguous()
        or not out.is_contiguous()
    ):
        raise ValueError(
            "paged decode logits requires contiguous q/weights/block_table/out, got "
            f"q_stride={tuple(index_q.stride())}, weights_stride={tuple(weights.stride())}, "
            f"sum_stride={tuple(sum_cache.stride())}, count_stride={tuple(count_cache.stride())}, "
            f"block_table_stride={tuple(block_table.stride())}, out_stride={tuple(out.stride())}"
        )
    if index_q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"index_q must be fp16/bf16/fp32, got {index_q.dtype}")
    if sum_cache.dtype != torch.float32 or count_cache.dtype != torch.float32:
        raise ValueError(
            "paged summary logits requires fp32 sum/count cache, got "
            f"sum={sum_cache.dtype}, count={count_cache.dtype}")
    if weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"weights must be fp16/bf16/fp32, got {weights.dtype}")
    if block_table.dtype != torch.int32:
        raise ValueError(f"block_table must be torch.int32, got {block_table.dtype}")
    if out.dtype != torch.float32:
        raise ValueError(f"out must be torch.float32, got {out.dtype}")
    if batch_size <= 0 or int(out.shape[1]) <= 0:
        return

    compiled = _get_compiled_decode_steptron_logits_paged_summary_warp_kernel(
        index_q, weights, sum_cache, count_cache, block_table, out)
    compiled(
        index_q,
        weights,
        sum_cache,
        count_cache,
        block_table,
        out,
        Int32(batch_size),
        Int32(out.shape[1]),
        Int32(num_pages),
        Int32(summaries_per_page),
    )


def _decode_weighted_relu_logits_sum_paged_mean_warp_sm90_steptron_gqa_impl(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    mean_cache: torch.Tensor,
    block_table: torch.Tensor,
    row_req_idx: torch.Tensor,
    row_table_idx: torch.Tensor,
    out: torch.Tensor,
    valid_requests: torch.Tensor,
    valid_tokens: torch.Tensor,
    stream=None,
) -> None:
    if stream is not None:
        raise ValueError("logits CuTeDSL kernels use the TVM-FFI environment stream")
    batch_size, proxy_kv_heads, q_heads_per_kv, head_dim_q = [int(v) for v in index_q.shape]
    num_pages, summaries_per_page, summary_kv_heads, head_dim_k = [int(v) for v in mean_cache.shape]
    if summaries_per_page <= 0 or summaries_per_page % 8 != 0:
        raise ValueError(
            "interleaved paged mean cache requires summaries_per_page to be "
            f"a positive multiple of 8, got {summaries_per_page}"
        )
    if proxy_kv_heads != 1 or summary_kv_heads != 1:
        raise ValueError(
            "paged mean decode StepTron logits requires one proxy KV head, got "
            f"q={proxy_kv_heads}, summary={summary_kv_heads}"
        )
    if q_heads_per_kv not in _SUPPORTED_Q_HEADS_PER_KV:
        raise ValueError(
            f"paged mean decode StepTron logits requires q heads per KV in "
            f"{_SUPPORTED_Q_HEADS_PER_KV}, got {q_heads_per_kv}"
        )
    if head_dim_q != _HEAD_DIM or head_dim_k != _HEAD_DIM:
        raise ValueError(
            f"paged mean decode StepTron logits requires head_dim={_HEAD_DIM}, "
            f"got q={head_dim_q}, summary={head_dim_k}"
        )
    if tuple(weights.shape) != (batch_size, proxy_kv_heads, q_heads_per_kv):
        raise ValueError(
            "weights shape must match index_q[:3], got "
            f"weights={tuple(weights.shape)}, index_q={tuple(index_q.shape)}"
        )
    if block_table.ndim != 2 or int(block_table.shape[0]) != batch_size:
        raise ValueError(
            "block_table must have shape [batch, pages], got "
            f"{tuple(block_table.shape)} for batch={batch_size}"
        )
    if out.ndim != 2 or int(out.shape[0]) != batch_size:
        raise ValueError(
            "out must have shape [batch, num_regions], got "
            f"{tuple(out.shape)} for batch={batch_size}"
        )
    if (
        row_req_idx.device != index_q.device
        or row_req_idx.dtype != torch.int32
        or row_req_idx.ndim != 1
        or tuple(row_req_idx.shape) != (batch_size,)
        or not row_req_idx.is_contiguous()
    ):
        raise ValueError(
            "row_req_idx must be a contiguous CUDA int32 tensor with shape "
            f"[{batch_size}]"
        )
    if (
        row_table_idx.device != index_q.device
        or row_table_idx.dtype != torch.int32
        or row_table_idx.ndim != 1
        or tuple(row_table_idx.shape) != (batch_size,)
        or not row_table_idx.is_contiguous()
    ):
        raise ValueError(
            "row_table_idx must be a contiguous CUDA int32 tensor with shape "
            f"[{batch_size}]"
        )
    if (
        index_q.device != weights.device
        or index_q.device != mean_cache.device
        or index_q.device != block_table.device
        or index_q.device != out.device
    ):
        raise ValueError("paged mean decode logits inputs must live on the same device")
    if index_q.device.type != "cuda":
        raise RuntimeError(
            "decode_weighted_relu_logits_sum_paged_mean_warp_sm90_steptron_gqa "
            "requires CUDA tensors")
    if (
        not index_q.is_contiguous()
        or not weights.is_contiguous()
        or not mean_cache.is_contiguous()
        or not block_table.is_contiguous()
        or not out.is_contiguous()
    ):
        raise ValueError(
            "paged mean decode logits requires contiguous q/weights/mean/block_table/out, got "
            f"q_stride={tuple(index_q.stride())}, weights_stride={tuple(weights.stride())}, "
            f"mean_stride={tuple(mean_cache.stride())}, "
            f"block_table_stride={tuple(block_table.stride())}, out_stride={tuple(out.stride())}"
        )
    if index_q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"index_q must be fp16/bf16/fp32, got {index_q.dtype}")
    if mean_cache.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.uint8):
        raise ValueError(
            f"mean_cache must be fp16/bf16/fp32/uint8-fp8bits, got {mean_cache.dtype}")
    if weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"weights must be fp16/bf16/fp32, got {weights.dtype}")
    if block_table.dtype != torch.int32:
        raise ValueError(f"block_table must be torch.int32, got {block_table.dtype}")
    if out.dtype != torch.float32:
        raise ValueError(f"out must be torch.float32, got {out.dtype}")
    if (
        valid_requests.device != index_q.device
        or valid_requests.dtype != torch.int32
        or valid_requests.ndim != 1
        or int(valid_requests.numel()) != 1
        or not valid_requests.is_contiguous()
    ):
        raise ValueError(
            "valid_requests must be a contiguous CUDA int32 tensor with shape [1]"
        )
    if (
        valid_tokens.device != index_q.device
        or valid_tokens.dtype != torch.int32
        or valid_tokens.ndim != 1
        or int(valid_tokens.numel()) != 1
        or not valid_tokens.is_contiguous()
    ):
        raise ValueError(
            "valid_tokens must be a contiguous CUDA int32 tensor with shape [1]"
        )
    if batch_size <= 0 or int(out.shape[1]) <= 0:
        return

    compiled = _get_compiled_decode_steptron_logits_paged_mean_warp_kernel(
        index_q,
        weights,
        mean_cache,
        block_table,
        out,
        row_req_idx=row_req_idx,
        row_table_idx=row_table_idx,
        valid_requests=valid_requests,
        valid_tokens=valid_tokens,
    )
    compiled(
        index_q,
        weights,
        mean_cache,
        block_table,
        row_req_idx,
        row_table_idx,
        out,
        valid_requests,
        valid_tokens,
        Int32(batch_size),
        Int32(out.shape[1]),
        Int32(num_pages),
        Int32(summaries_per_page),
    )


def _materialize_paged_summary_mean_cache_sm90_steptron_gqa_impl(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    mean_cache: torch.Tensor,
    stream=None,
) -> None:
    if stream is not None:
        raise ValueError("logits CuTeDSL kernels use the TVM-FFI environment stream")
    num_pages, summaries_per_page, summary_kv_heads, head_dim = [int(v) for v in sum_cache.shape]
    if summaries_per_page <= 0 or summaries_per_page % 8 != 0:
        raise ValueError(
            "interleaved paged mean cache requires summaries_per_page to be "
            f"a positive multiple of 8, got {summaries_per_page}"
        )
    if summary_kv_heads != 1:
        raise ValueError(
            "StepTron paged mean cache materialize requires one proxy KV head, got "
            f"{summary_kv_heads}"
        )
    if head_dim != _HEAD_DIM:
        raise ValueError(
            f"StepTron paged mean cache materialize requires head_dim={_HEAD_DIM}, got {head_dim}"
        )
    if tuple(count_cache.shape) != (num_pages, summaries_per_page, summary_kv_heads):
        raise ValueError(
            "count_cache shape must match sum_cache[:3], got "
            f"count_cache={tuple(count_cache.shape)}, sum_cache={tuple(sum_cache.shape)}"
        )
    if tuple(mean_cache.shape) != tuple(sum_cache.shape):
        raise ValueError(
            "mean_cache shape must match sum_cache, got "
            f"mean={tuple(mean_cache.shape)}, sum={tuple(sum_cache.shape)}"
        )
    if sum_cache.device != count_cache.device or sum_cache.device != mean_cache.device:
        raise ValueError("sum_cache/count_cache/mean_cache must live on the same device")
    if sum_cache.device.type != "cuda":
        raise RuntimeError("materialize_paged_summary_mean_cache_sm90_steptron_gqa requires CUDA tensors")
    if not mean_cache.is_contiguous():
        raise ValueError(
            "materialize paged mean cache requires contiguous output; "
            "sum/count may retain the KV-sidecar page stride, got "
            f"sum_stride={tuple(sum_cache.stride())}, count_stride={tuple(count_cache.stride())}, "
            f"mean_stride={tuple(mean_cache.stride())}"
        )
    if sum_cache.dtype != torch.float32 or count_cache.dtype != torch.float32:
        raise ValueError(
            "materialize paged mean cache requires fp32 sum/count, got "
            f"sum={sum_cache.dtype}, count={count_cache.dtype}"
        )
    if mean_cache.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.uint8):
        raise ValueError(
            f"mean_cache must be fp16/bf16/fp32/uint8-fp8bits, got {mean_cache.dtype}")
    if num_pages <= 0 or summaries_per_page <= 0:
        return

    compiled = _get_compiled_materialize_steptron_paged_mean_cache_kernel(
        sum_cache, count_cache, mean_cache)
    compiled(
        sum_cache,
        count_cache,
        mean_cache,
        num_pages * summaries_per_page,
    )


def _decode_weighted_relu_logits_sum_paged_summary_warp_splitk_sm90_steptron_gqa_impl(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    block_table: torch.Tensor,
    out: torch.Tensor,
    *,
    split_k: int,
    stream=None,
) -> None:
    if stream is not None:
        raise ValueError("logits CuTeDSL kernels use the TVM-FFI environment stream")
    if split_k not in (2, 4, 8):
        raise ValueError(f"split_k must be one of (2, 4, 8), got {split_k}")
    if _HEAD_DIM % split_k != 0 or (_HEAD_DIM // split_k) % (_WARP_REGION_LANES * 2) != 0:
        raise ValueError(f"unsupported split_k={split_k} for head_dim={_HEAD_DIM}")

    batch_size, proxy_kv_heads, q_heads_per_kv, head_dim_q = [int(v) for v in index_q.shape]
    num_pages, summaries_per_page, summary_kv_heads, head_dim_k = [int(v) for v in sum_cache.shape]
    if proxy_kv_heads != 1 or summary_kv_heads != 1:
        raise ValueError(
            "paged split-k decode StepTron logits requires one proxy KV head, got "
            f"q={proxy_kv_heads}, summary={summary_kv_heads}"
        )
    if q_heads_per_kv not in _SUPPORTED_Q_HEADS_PER_KV:
        raise ValueError(
            f"paged split-k decode StepTron logits requires q heads per KV in "
            f"{_SUPPORTED_Q_HEADS_PER_KV}, got {q_heads_per_kv}"
        )
    if head_dim_q != _HEAD_DIM or head_dim_k != _HEAD_DIM:
        raise ValueError(
            f"paged split-k decode StepTron logits requires head_dim={_HEAD_DIM}, "
            f"got q={head_dim_q}, summary={head_dim_k}"
        )
    if tuple(count_cache.shape) != (num_pages, summaries_per_page, summary_kv_heads):
        raise ValueError(
            "count_cache shape must match sum_cache[:3], got "
            f"count_cache={tuple(count_cache.shape)}, sum_cache={tuple(sum_cache.shape)}"
        )
    if tuple(weights.shape) != (batch_size, proxy_kv_heads, q_heads_per_kv):
        raise ValueError(
            "weights shape must match index_q[:3], got "
            f"weights={tuple(weights.shape)}, index_q={tuple(index_q.shape)}"
        )
    if block_table.ndim != 2 or int(block_table.shape[0]) != batch_size:
        raise ValueError(
            "block_table must have shape [batch, pages], got "
            f"{tuple(block_table.shape)} for batch={batch_size}"
        )
    if out.ndim != 2 or int(out.shape[0]) != batch_size:
        raise ValueError(
            "out must have shape [batch, num_regions], got "
            f"{tuple(out.shape)} for batch={batch_size}"
        )
    if (
        index_q.device != weights.device
        or index_q.device != sum_cache.device
        or index_q.device != count_cache.device
        or index_q.device != block_table.device
        or index_q.device != out.device
    ):
        raise ValueError("paged split-k decode logits inputs must live on the same device")
    if index_q.device.type != "cuda":
        raise RuntimeError(
            "decode_weighted_relu_logits_sum_paged_summary_warp_splitk_sm90_steptron_gqa "
            "requires CUDA tensors")
    if (
        not index_q.is_contiguous()
        or not weights.is_contiguous()
        or not block_table.is_contiguous()
        or not out.is_contiguous()
    ):
        raise ValueError(
            "paged split-k decode logits requires contiguous q/weights/block_table/out, got "
            f"q_stride={tuple(index_q.stride())}, weights_stride={tuple(weights.stride())}, "
            f"sum_stride={tuple(sum_cache.stride())}, count_stride={tuple(count_cache.stride())}, "
            f"block_table_stride={tuple(block_table.stride())}, out_stride={tuple(out.stride())}"
        )
    if index_q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"index_q must be fp16/bf16/fp32, got {index_q.dtype}")
    if sum_cache.dtype != torch.float32 or count_cache.dtype != torch.float32:
        raise ValueError(
            "paged split-k summary logits requires fp32 sum/count cache, got "
            f"sum={sum_cache.dtype}, count={count_cache.dtype}")
    if weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"weights must be fp16/bf16/fp32, got {weights.dtype}")
    if block_table.dtype != torch.int32:
        raise ValueError(f"block_table must be torch.int32, got {block_table.dtype}")
    if out.dtype != torch.float32:
        raise ValueError(f"out must be torch.float32, got {out.dtype}")
    num_regions = int(out.shape[1])
    if batch_size <= 0 or num_regions <= 0:
        return

    partial = torch.empty(
        (batch_size, num_regions, int(split_k), q_heads_per_kv),
        device=out.device,
        dtype=torch.float32,
    )

    compiled = _get_compiled_decode_steptron_logits_paged_summary_warp_splitk_kernel(
        index_q, weights, sum_cache, count_cache, block_table, partial, out, int(split_k))
    compiled(
        index_q,
        weights,
        sum_cache,
        count_cache,
        block_table,
        partial,
        out,
        batch_size,
        num_regions,
        num_pages,
        summaries_per_page,
    )


@torch.library.custom_op(
    "optimus_cutedsl::decode_weighted_relu_logits_sum_sm90_steptron_gqa_out",
    mutates_args=("out",),
    device_types="cuda",
)
def _decode_weighted_relu_logits_sum_sm90_steptron_gqa_out(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    summary: torch.Tensor,
    out: torch.Tensor,
) -> None:
    _decode_weighted_relu_logits_sum_sm90_steptron_gqa_impl(
        index_q,
        weights,
        summary,
        out,
        stream=None,
    )


def decode_weighted_relu_logits_sum_sm90_steptron_gqa(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    summary: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if index_q.dim() != 4:
        raise ValueError("index_q must be rank-4 [B, 1, q_heads_per_kv, head_dim]")
    if weights.dim() != 3:
        raise ValueError("weights must be rank-3 [B, 1, q_heads_per_kv]")
    if summary.dim() != 4:
        raise ValueError("summary must be rank-4 [B, R, 1, head_dim]")
    if out is None:
        out = torch.empty(
            (int(index_q.shape[0]), int(summary.shape[1])),
            device=index_q.device,
            dtype=torch.float32,
        )
    _decode_weighted_relu_logits_sum_sm90_steptron_gqa_out(
        index_q,
        weights,
        summary,
        out,
    )
    return out


@torch.library.custom_op(
    "optimus_cutedsl::decode_weighted_relu_logits_sum_warp_sm90_steptron_gqa_out",
    mutates_args=("out",),
    device_types="cuda",
)
def _decode_weighted_relu_logits_sum_warp_sm90_steptron_gqa_out(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    summary: torch.Tensor,
    out: torch.Tensor,
) -> None:
    _decode_weighted_relu_logits_sum_sm90_steptron_gqa_impl(
        index_q,
        weights,
        summary,
        out,
        stream=None,
        use_warp_kernel=True,
    )


def decode_weighted_relu_logits_sum_warp_sm90_steptron_gqa(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    summary: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if index_q.dim() != 4:
        raise ValueError("index_q must be rank-4 [B, 1, q_heads_per_kv, head_dim]")
    if weights.dim() != 3:
        raise ValueError("weights must be rank-3 [B, 1, q_heads_per_kv]")
    if summary.dim() != 4:
        raise ValueError("summary must be rank-4 [B, R, 1, head_dim]")
    if out is None:
        out = torch.empty(
            (int(index_q.shape[0]), int(summary.shape[1])),
            device=index_q.device,
            dtype=torch.float32,
        )
    _decode_weighted_relu_logits_sum_warp_sm90_steptron_gqa_out(
        index_q,
        weights,
        summary,
        out,
    )
    return out


@torch.library.custom_op(
    "optimus_cutedsl::decode_weighted_relu_logits_sum_paged_summary_warp_sm90_steptron_gqa_out",
    mutates_args=("out",),
    device_types="cuda",
)
def _decode_weighted_relu_logits_sum_paged_summary_warp_sm90_steptron_gqa_out(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    block_table: torch.Tensor,
    out: torch.Tensor,
) -> None:
    _decode_weighted_relu_logits_sum_paged_summary_warp_sm90_steptron_gqa_impl(
        index_q,
        weights,
        sum_cache,
        count_cache,
        block_table,
        out,
        stream=None,
    )


def decode_weighted_relu_logits_sum_paged_summary_warp_sm90_steptron_gqa(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if index_q.dim() != 4:
        raise ValueError("index_q must be rank-4 [B, 1, q_heads_per_kv, head_dim]")
    if weights.dim() != 3:
        raise ValueError("weights must be rank-3 [B, 1, q_heads_per_kv]")
    if sum_cache.dim() != 4:
        raise ValueError("sum_cache must be rank-4 [pages, summaries_per_page, 1, head_dim]")
    if count_cache.dim() != 3:
        raise ValueError("count_cache must be rank-3 [pages, summaries_per_page, 1]")
    if block_table.dim() != 2:
        raise ValueError("block_table must be rank-2 [B, pages]")
    if out is None:
        out = torch.empty(
            (int(index_q.shape[0]), int(block_table.shape[1]) * int(sum_cache.shape[1])),
            device=index_q.device,
            dtype=torch.float32,
        )
    _decode_weighted_relu_logits_sum_paged_summary_warp_sm90_steptron_gqa_out(
        index_q,
        weights,
        sum_cache,
        count_cache,
        block_table,
        out,
    )
    return out


def decode_weighted_relu_logits_sum_paged_mean_warp_sm90_steptron_gqa(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    mean_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    row_req_idx: Optional[torch.Tensor] = None,
    row_table_idx: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    valid_requests: Optional[torch.Tensor] = None,
    valid_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if index_q.dim() != 4:
        raise ValueError("index_q must be rank-4 [B, 1, q_heads_per_kv, head_dim]")
    if weights.dim() != 3:
        raise ValueError("weights must be rank-3 [B, 1, q_heads_per_kv]")
    if mean_cache.dim() != 4:
        raise ValueError("mean_cache must be rank-4 [pages, summaries_per_page, 1, head_dim]")
    if block_table.dim() != 2:
        raise ValueError("block_table must be rank-2 [B, pages]")
    if out is None:
        out = torch.empty(
            (int(index_q.shape[0]), int(block_table.shape[1]) * int(mean_cache.shape[1])),
            device=index_q.device,
            dtype=torch.float32,
        )
    if valid_requests is None:
        valid_requests = torch.full(
            (1,), int(index_q.shape[0]), dtype=torch.int32, device=index_q.device
        )
    if row_req_idx is None:
        row_req_idx = torch.arange(
            int(index_q.shape[0]), dtype=torch.int32, device=index_q.device
        )
    if row_table_idx is None:
        row_table_idx = torch.arange(
            int(index_q.shape[0]), dtype=torch.int32, device=index_q.device
        )
    if valid_tokens is None:
        valid_tokens = torch.full(
            (1,), int(index_q.shape[0]), dtype=torch.int32, device=index_q.device
        )
    _decode_weighted_relu_logits_sum_paged_mean_warp_sm90_steptron_gqa_impl(
        index_q,
        weights,
        mean_cache,
        block_table,
        row_req_idx,
        row_table_idx,
        out,
        valid_requests,
        valid_tokens,
        stream=None,
    )
    return out


def rerank_weighted_relu_logits_sum_paged_mean_warp_sm90_steptron_gqa(
    index_q_fp8: torch.Tensor,
    weights: torch.Tensor,
    mean_cache: torch.Tensor,
    block_table: torch.Tensor,
    region_ids: torch.Tensor,
    scores: torch.Tensor,
) -> None:
    batch = int(index_q_fp8.shape[0])
    if tuple(index_q_fp8.shape[1:]) != (1, 4, _HEAD_DIM):
        raise ValueError(
            "rerank requires Q shape [B, 1, 4, 256], got "
            f"{tuple(index_q_fp8.shape)}")
    if index_q_fp8.dtype != torch.uint8:
        raise ValueError("rerank requires uint8 FP8 Q bits")
    if tuple(weights.shape) != (batch, 1, 4) or weights.dtype != torch.float32:
        raise ValueError("rerank requires FP32 weights [B, 1, 4]")
    if mean_cache.ndim != 4 or mean_cache.dtype != torch.uint8:
        raise ValueError("rerank requires uint8 paged FP8 mean cache")
    if block_table.ndim != 2 or tuple(block_table.shape[:1]) != (batch,):
        raise ValueError("rerank block_table must be [B, pages]")
    if block_table.dtype != torch.int32:
        raise ValueError("rerank block_table must be int32")
    if region_ids.ndim != 2 or int(region_ids.shape[0]) != batch:
        raise ValueError("rerank region_ids must be int32 [B, candidates]")
    if region_ids.dtype != torch.int32:
        raise ValueError("rerank region_ids must be int32")
    if scores.ndim != 2 or int(scores.shape[0]) != batch:
        raise ValueError("rerank scores must be FP32 [B, regions]")
    if scores.dtype != torch.float32:
        raise ValueError("rerank scores must be FP32")
    tensors = (
        index_q_fp8,
        weights,
        mean_cache,
        block_table,
        region_ids,
        scores,
    )
    if any(t.device != scores.device for t in tensors):
        raise ValueError("rerank inputs must share one CUDA device")
    if scores.device.type != "cuda" or any(not t.is_contiguous() for t in tensors):
        raise ValueError("rerank inputs must be contiguous CUDA tensors")
    if batch <= 0 or int(region_ids.shape[1]) <= 0:
        return
    compiled = _get_compiled_rerank_steptron_logits_paged_mean_warp_kernel(
        index_q_fp8,
        weights,
        mean_cache,
        block_table,
        region_ids,
        scores,
    )
    compiled(
        index_q_fp8,
        weights,
        mean_cache,
        block_table,
        region_ids,
        scores,
        batch,
        int(region_ids.shape[1]),
        int(scores.shape[1]),
        int(mean_cache.shape[0]),
        int(mean_cache.shape[1]),
    )


def materialize_paged_summary_mean_cache_sm90_steptron_gqa(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if sum_cache.dim() != 4:
        raise ValueError("sum_cache must be rank-4 [pages, summaries_per_page, 1, head_dim]")
    if count_cache.dim() != 3:
        raise ValueError("count_cache must be rank-3 [pages, summaries_per_page, 1]")
    if out is None:
        out = torch.empty_like(sum_cache, dtype=out_dtype)
    elif out.dtype != out_dtype:
        raise ValueError(f"out dtype mismatch: got {out.dtype}, expected {out_dtype}")
    _materialize_paged_summary_mean_cache_sm90_steptron_gqa_impl(
        sum_cache,
        count_cache,
        out,
        stream=None,
    )
    return out


def materialize_selected_paged_summary_mean_cache_sm90_steptron_gqa(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    region_ids: torch.Tensor,
    mean_cache: torch.Tensor,
) -> None:
    """Refresh only the physical summary regions changed by decode CSA."""
    if sum_cache.dim() != 4 or count_cache.dim() != 3:
        raise ValueError("sum/count cache has invalid rank")
    if tuple(count_cache.shape) != tuple(sum_cache.shape[:3]):
        raise ValueError("count_cache shape must match sum_cache[:3]")
    if tuple(mean_cache.shape) != tuple(sum_cache.shape):
        raise ValueError("mean_cache shape must match sum_cache")
    if int(mean_cache.shape[1]) <= 0 or int(mean_cache.shape[1]) % 8 != 0:
        raise ValueError(
            "interleaved paged mean cache requires summaries_per_page to be "
            f"a positive multiple of 8, got {int(mean_cache.shape[1])}"
        )
    if sum_cache.dtype != torch.float32 or count_cache.dtype != torch.float32:
        raise ValueError("selected mean materialize requires fp32 sum/count")
    if mean_cache.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.uint8):
        raise ValueError(f"unsupported mean_cache dtype: {mean_cache.dtype}")
    if region_ids.dtype not in (torch.int32, torch.int64) or region_ids.ndim != 1:
        raise ValueError("region_ids must be contiguous int32/int64 [num_regions]")
    if any(t.device != sum_cache.device for t in (count_cache, region_ids, mean_cache)):
        raise ValueError("selected mean materialize inputs must share one device")
    if not region_ids.is_contiguous() or not mean_cache.is_contiguous():
        raise ValueError("region_ids and mean_cache must be contiguous")
    if sum_cache.device.type != "cuda":
        raise RuntimeError("selected mean materialize requires CUDA tensors")
    if int(region_ids.numel()) == 0:
        return
    compiled = _get_compiled_materialize_steptron_selected_mean_cache_kernel(
        sum_cache, count_cache, mean_cache, region_ids)
    compiled(sum_cache, count_cache, mean_cache, region_ids, int(region_ids.numel()))


def select_paged_summary_logits_split_k_sm90_steptron_gqa(
    batch_size: int,
    num_regions: int,
) -> Optional[int]:
    """Select split-k for StepTron paged summary logits.

    The thresholds come from H200 measurements with region size 8, so
    num_regions 1024/2048/4096/8192/16384 correspond to
    8k/16k/32k/64k/128k. Returns None when the no-split paged summary
    kernel is faster than any split-k variant.
    """
    batch_size = int(batch_size)
    num_regions = int(num_regions)
    if batch_size <= 0 or num_regions <= 0:
        return 2
    if num_regions <= 1024:
        if batch_size >= 64:
            return 2
        return 4 if batch_size >= 16 else 8
    if num_regions <= 2048:
        if batch_size >= 32:
            return 2
        return 4 if batch_size >= 8 else 8
    if num_regions <= 4096:
        if batch_size >= 16:
            return 2
        return 4 if batch_size >= 4 else 8
    if num_regions <= 8192:
        if batch_size >= 64:
            return None
        if batch_size >= 8:
            return 2
        return 4 if batch_size >= 2 else 8
    if batch_size >= 32:
        return None
    return 2 if batch_size >= 4 else 4


def decode_weighted_relu_logits_sum_paged_summary_warp_splitk_sm90_steptron_gqa(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    split_k: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if index_q.dim() != 4:
        raise ValueError("index_q must be rank-4 [B, 1, q_heads_per_kv, head_dim]")
    if weights.dim() != 3:
        raise ValueError("weights must be rank-3 [B, 1, q_heads_per_kv]")
    if sum_cache.dim() != 4:
        raise ValueError("sum_cache must be rank-4 [pages, summaries_per_page, 1, head_dim]")
    if count_cache.dim() != 3:
        raise ValueError("count_cache must be rank-3 [pages, summaries_per_page, 1]")
    if block_table.dim() != 2:
        raise ValueError("block_table must be rank-2 [B, pages]")
    if out is None:
        out = torch.empty(
            (int(index_q.shape[0]), int(block_table.shape[1]) * int(sum_cache.shape[1])),
            device=index_q.device,
            dtype=torch.float32,
        )
    _decode_weighted_relu_logits_sum_paged_summary_warp_splitk_sm90_steptron_gqa_impl(
        index_q,
        weights,
        sum_cache,
        count_cache,
        block_table,
        out,
        split_k=int(split_k),
        stream=None,
    )
    return out


def decode_weighted_relu_logits_sum_paged_summary_auto_sm90_steptron_gqa(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if index_q.dim() != 4:
        raise ValueError("index_q must be rank-4 [B, 1, q_heads_per_kv, head_dim]")
    if sum_cache.dim() != 4:
        raise ValueError("sum_cache must be rank-4 [pages, summaries_per_page, 1, head_dim]")
    if block_table.dim() != 2:
        raise ValueError("block_table must be rank-2 [B, pages]")
    if out is None:
        out = torch.empty(
            (int(index_q.shape[0]), int(block_table.shape[1]) * int(sum_cache.shape[1])),
            device=index_q.device,
            dtype=torch.float32,
        )
    split_k = select_paged_summary_logits_split_k_sm90_steptron_gqa(
        int(index_q.shape[0]),
        int(out.shape[1]),
    )
    if split_k is None:
        return decode_weighted_relu_logits_sum_paged_summary_warp_sm90_steptron_gqa(
            index_q,
            weights,
            sum_cache,
            count_cache,
            block_table,
            out=out,
        )
    return decode_weighted_relu_logits_sum_paged_summary_warp_splitk_sm90_steptron_gqa(
        index_q,
        weights,
        sum_cache,
        count_cache,
        block_table,
        split_k=split_k,
        out=out,
    )


__all__ = [
    "decode_weighted_relu_logits_sum_sm90_steptron_gqa",
    "decode_weighted_relu_logits_sum_warp_sm90_steptron_gqa",
    "decode_weighted_relu_logits_sum_paged_summary_warp_sm90_steptron_gqa",
    "decode_weighted_relu_logits_sum_paged_mean_warp_sm90_steptron_gqa",
    "rerank_weighted_relu_logits_sum_paged_mean_warp_sm90_steptron_gqa",
    "materialize_paged_summary_mean_cache_sm90_steptron_gqa",
    "materialize_selected_paged_summary_mean_cache_sm90_steptron_gqa",
    "decode_weighted_relu_logits_sum_paged_summary_warp_splitk_sm90_steptron_gqa",
    "decode_weighted_relu_logits_sum_paged_summary_auto_sm90_steptron_gqa",
    "select_paged_summary_logits_split_k_sm90_steptron_gqa",
]
