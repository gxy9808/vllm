# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from typing import Optional

import torch
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Float32, Float8E4M3FN, Int32, Uint8
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import if_generate

from vllm.models.step4.nvidia.ops.cute_dsl.flash_attn import copy_utils
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.kernel_utils import cute_utils as sparse_utils
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.kernel_utils import hopper_helpers as hop_helpers
from vllm.models.step4.nvidia.ops.cute_dsl.utils import elem_pointer


# NOTE(step4/TP4): this decoding specialization is the TP4/H4 path.  One CTA
# owns all q tokens of one request and expands WGMMA N from the host-visible
# q_len shape. HEAD_DIM is split into two exact K=128 segments.
Q_HEADS_PER_KV = 4
SUPPORTED_Q_HEADS_PER_KV = (4,)
MIN_Q_LEN = 1
MAX_Q_LEN = 16
HEAD_DIM = 256
GQA_COMPILED_Q_LEN = 1
GQA_COMPILED_WGMMA_N = 8
GQA_NUM_TMA_REGS = 32
GQA_NUM_MATH_REGS = 112
GQA_PREPARE_THREADS = 256
# A TMA transaction covers one complete 8-row WGMMA swizzle period.  The
# physical page can contain any multiple of this period (vLLM uses 32); page
# geometry stays runtime data and is never baked into the compiled kernel.
PAGED_TMA_ROWS = 8
PAGED_TMA_PAIR_BYTES = PAGED_TMA_ROWS * HEAD_DIM
PAGED_WGMMA_ATOM_BYTES = 128
PAGED_TMA_CHUNK_BYTES = PAGED_TMA_ROWS * PAGED_WGMMA_ATOM_BYTES
PAGED_TMA_CHUNKS_PER_PAIR = HEAD_DIM // PAGED_WGMMA_ATOM_BYTES
# The paged path uses a 256-region CTA tile so each resident CTA amortizes the
# runtime work-id lookup and query staging over more K rows. WGMMA remains a
# 64-row stage; four entries keep the producer/consumer ring fully asynchronous.
PAGED_TILE_KV = 256
PAGED_STAGE_KV = 64
PAGED_NUM_K_STAGES = PAGED_TILE_KV // PAGED_STAGE_KV
PAGED_K_SEGMENTS = 2
PAGED_K_SEGMENT = HEAD_DIM // PAGED_K_SEGMENTS
PAGED_WGMMA_ATOM_STRIDE = PAGED_STAGE_KV * PAGED_WGMMA_ATOM_BYTES
PAGED_PAIRS_PER_STAGE = PAGED_STAGE_KV // PAGED_TMA_ROWS
PAGED_STAGE_BYTES = PAGED_STAGE_KV * HEAD_DIM
PAGED_NUM_MATH_THREADS = 128
PAGED_NUM_PRODUCER_THREADS = 128
PAGED_NUM_MATH_WARPS = PAGED_NUM_MATH_THREADS // 32
PAGED_NUM_THREADS = PAGED_NUM_MATH_THREADS + PAGED_NUM_PRODUCER_THREADS
# Bound the persistent grid at eight CTAs/SM. The launch still takes the
# runtime minimum with max_work_items, so short/mixed varlen batches do not
# launch empty CTAs while long batches get one work item per CTA when possible.
PAGED_PERSISTENT_CTAS_PER_SM = 8


def _cute_compile(func, *args):
    return cute.compile(func, *args, options="--enable-tvm-ffi --opt-level 2")


def _validate_q_heads_per_kv(q_heads_per_kv: int) -> None:
    h = int(q_heads_per_kv)
    if h not in SUPPORTED_Q_HEADS_PER_KV:
        raise ValueError(
            "batch decode GQA logits supports q_heads_per_kv in "
            f"{SUPPORTED_Q_HEADS_PER_KV}, got {h}"
        )


def _wgmma_n_for_q_len(q_len: int) -> int:
    q_len = int(q_len)
    if not MIN_Q_LEN <= q_len <= MAX_Q_LEN:
        raise ValueError(
            f"batch decode GQA logits supports q_len in [{MIN_Q_LEN},{MAX_Q_LEN}], "
            f"got {q_len}"
        )
    logical_n = q_len * Q_HEADS_PER_KV
    return max(16, ((logical_n + 7) // 8) * 8)


def batch_decode_logits_wgmma_n(q_len: int) -> int:
    return _wgmma_n_for_q_len(q_len)


def _load_f32_global(tensor: cute.Tensor, coord: tuple[Int32, Int32]) -> Float32:
    src = cute.make_tensor(
        elem_pointer(tensor, coord),
        cute.make_layout((1,), stride=(1,)),
    )
    return Float32(src[0])


def _tensor_signature_dynamic(
    x: torch.Tensor,
    *,
    dynamic_shape_dims: tuple[int, ...],
    dynamic_stride_dims: tuple[int, ...] = (),
) -> tuple[torch.dtype, tuple[object, ...], tuple[object, ...]]:
    shape: list[object] = [int(v) for v in x.shape]
    stride: list[object] = [int(v) for v in x.stride()]
    for dim in dynamic_shape_dims:
        shape[int(dim)] = None
    for dim in dynamic_stride_dims:
        stride[int(dim)] = None
    return x.dtype, tuple(shape), tuple(stride)


def _placeholder_from_signature(
    signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    *,
    device: torch.device,
    dynamic_shape_fill: int,
    dynamic_stride_fill: int = 1,
) -> torch.Tensor:
    dtype, shape, stride = signature
    concrete_shape = tuple(
        int(dynamic_shape_fill) if dim is None else int(dim) for dim in shape
    )
    concrete_stride = tuple(
        int(dynamic_stride_fill) if dim is None else int(dim) for dim in stride
    )
    return torch.empty_strided(
        concrete_shape,
        concrete_stride,
        device=device,
        dtype=dtype,
    )


@cute.kernel
def _pack_fp8_qw_kernel(
    mQIn: cute.Tensor,
    mWIn: cute.Tensor,
    mQOut: cute.Tensor,
    mWOut: cute.Tensor,
    total_q_elems: Int32,
    total_w_elems: Int32,
    batch_size: Int32,
    q_len: cutlass.Constexpr[int],
    wgmma_n: cutlass.Constexpr[int],
):
    tid = (
        cute.arch.block_idx()[0] * Int32(GQA_PREPARE_THREADS)
        + cute.arch.thread_idx()[0]
    )
    total = total_q_elems + total_w_elems
    if tid < total_q_elems:
        dim = tid % Int32(HEAD_DIM)
        row = tid // Int32(HEAD_DIM)
        local_row = row % Int32(wgmma_n)
        batch_idx = row // Int32(wgmma_n)
        q_slot = local_row // Int32(Q_HEADS_PER_KV)
        head = local_row - q_slot * Int32(Q_HEADS_PER_KV)
        valid = (batch_idx < batch_size) & (q_slot < Int32(q_len))
        safe_batch = cutlass.select_(valid, batch_idx, Int32(0))
        safe_q_slot = cutlass.select_(valid, q_slot, Int32(0))
        safe_head = cutlass.select_(valid, head, Int32(0))
        value = mQIn[safe_batch, safe_q_slot, Int32(0), safe_head, dim]
        mQOut[row, dim] = cutlass.select_(valid, value, Float8E4M3FN(0.0))
    if (tid >= total_q_elems) & (tid < total):
        w_tid = tid - total_q_elems
        n = w_tid % Int32(wgmma_n)
        batch_idx = w_tid // Int32(wgmma_n)
        q_slot = n // Int32(Q_HEADS_PER_KV)
        head = n - q_slot * Int32(Q_HEADS_PER_KV)
        valid = (batch_idx < batch_size) & (q_slot < Int32(q_len))
        safe_batch = cutlass.select_(valid, batch_idx, Int32(0))
        safe_q_slot = cutlass.select_(valid, q_slot, Int32(0))
        safe_head = cutlass.select_(valid, head, Int32(0))
        value = mWIn[safe_batch, safe_q_slot, Int32(0), safe_head]
        mWOut[batch_idx, n] = cutlass.select_(
            valid, value, mWOut.element_type(Float32(0.0))
        )


def _make_pack_fp8_qw_call(q_len: int, wgmma_n: int):
    @cute.jit
    def _call(
        mQIn: cute.Tensor,
        mWIn: cute.Tensor,
        mQOut: cute.Tensor,
        mWOut: cute.Tensor,
        stream,
        grid_x: Int32,
        total_q_elems: Int32,
        total_w_elems: Int32,
        batch_size: Int32,
    ):
        _pack_fp8_qw_kernel(
            mQIn,
            mWIn,
            mQOut,
            mWOut,
            total_q_elems,
            total_w_elems,
            batch_size,
            q_len,
            wgmma_n,
        ).launch(grid=[grid_x, 1, 1], block=[GQA_PREPARE_THREADS, 1, 1], stream=stream)

    return _call


@functools.cache
def _compile_pack_fp8_qw_for_signature(
    device_key: tuple[str, Optional[int]],
    q_len: int,
    wgmma_n: int,
    q_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    w_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    q_out_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    w_out_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
):
    device = sparse_utils.device_from_cache_key(device_key)
    q = _placeholder_from_signature(q_signature, device=device, dynamic_shape_fill=1)
    w = _placeholder_from_signature(w_signature, device=device, dynamic_shape_fill=1)
    q_out = _placeholder_from_signature(
        q_out_signature, device=device, dynamic_shape_fill=wgmma_n
    )
    w_out = _placeholder_from_signature(
        w_out_signature, device=device, dynamic_shape_fill=1
    )
    f_q = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        q, alignment=16, dynamic_shape_dim=0
    )
    f_w = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        w, alignment=16, dynamic_shape_dim=0
    )
    f_q_out = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        q_out, alignment=16, dynamic_shape_dim=0, divisibility=wgmma_n
    )
    f_w_out = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        w_out, alignment=16, dynamic_shape_dim=0
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return _cute_compile(
        _make_pack_fp8_qw_call(q_len, wgmma_n),
        f_q,
        f_w,
        f_q_out,
        f_w_out,
        stream_fake,
        Int32(
            (q_out.numel() + w_out.numel() + GQA_PREPARE_THREADS - 1)
            // GQA_PREPARE_THREADS
        ),
        Int32(q_out.numel()),
        Int32(w_out.numel()),
        Int32(q.shape[0]),
    )


@cute.kernel
def _batch_decode_logits_paged_sm90_kernel(
    tma_atom_Q0: cute.CopyAtom,
    tma_tensor_Q0: cute.Tensor,
    tma_atom_Q1: cute.CopyAtom,
    tma_tensor_Q1: cute.Tensor,
    tma_atom_K0: cute.CopyAtom,
    tma_tensor_K0: cute.Tensor,
    tma_atom_K1: cute.CopyAtom,
    tma_tensor_K1: cute.Tensor,
    mQ: cute.Tensor,
    mKCache: cute.Tensor,
    mBlockTable: cute.Tensor,
    mRowReqIdx: cute.Tensor,
    mRowTableIdx: cute.Tensor,
    mRowStarts: cute.Tensor,
    mRowEnds: cute.Tensor,
    mValidRequests: cute.Tensor,
    mValidTokens: cute.Tensor,
    tiled_mma: cute.TiledMma,
    sQ_layout_fp8_0: cute.ComposedLayout,
    sQ_layout_fp8_1: cute.ComposedLayout,
    sK_layout_fp8_0: cute.ComposedLayout,
    sK_layout_fp8_1: cute.ComposedLayout,
    sQ_layout_single_outer_0: cute.Layout,
    sQ_layout_single_outer_1: cute.Layout,
    sK_layout_single_outer_0: cute.Layout,
    sK_layout_single_outer_1: cute.Layout,
    mW: cute.Tensor,
    mO: cute.Tensor,
    max_regions: Int32,
    query_rows: Int32,
    num_reqs: Int32,
    block_table_cols: Int32,
    num_pages: Int32,
    summaries_per_page: Int32,
    max_tiles_per_row: Int32,
    max_work_items: Int32,
):
    """Persistent varlen TP4 FP8 QK over paged FP8 summaries.

    Work is flattened over request rows and 256-region capacity tiles. Every
    CTA consumes all ``q_len`` tokens of one request through one variable-N
    WGMMA tile, so the paged summary is loaded once and reused by MTP0 through
    MTP15. Runtime row bounds reject unused graph capacity before payload work.
    """
    tidx, _, _ = cute.arch.thread_idx()
    q_len = GQA_COMPILED_Q_LEN
    wgmma_n = GQA_COMPILED_WGMMA_N
    warp_idx = cute.arch.warp_idx()
    work_id = cute.arch.block_idx()[0]
    work_stride = cute.arch.grid_dim()[0]

    if work_id < max_work_items:
        if warp_idx == 0 and (tidx % 32) == 0:
            cpasync.prefetch_descriptor(tma_atom_Q0)
            cpasync.prefetch_descriptor(tma_atom_Q1)
            cpasync.prefetch_descriptor(tma_atom_K0)
            cpasync.prefetch_descriptor(tma_atom_K1)

        smem = cutlass.utils.SmemAllocator()
        sQ0_struct = cute.struct.Align[
            cute.struct.MemRange[Float8E4M3FN, cute.cosize(sQ_layout_fp8_0)], 1024
        ]
        sQ1_struct = cute.struct.Align[
            cute.struct.MemRange[Float8E4M3FN, cute.cosize(sQ_layout_fp8_1)], 1024
        ]
        sK0_struct = cute.struct.Align[
            cute.struct.MemRange[Float8E4M3FN, cute.cosize(sK_layout_fp8_0)], 1024
        ]
        sK1_struct = cute.struct.Align[
            cute.struct.MemRange[Float8E4M3FN, cute.cosize(sK_layout_fp8_1)], 1024
        ]

        mbar_ptr_q_struct = cute.struct.MemRange[cutlass.Int64, 2]
        mbar_ptr_k_struct = cute.struct.MemRange[cutlass.Int64, PAGED_NUM_K_STAGES * 2]

        @cute.struct
        class SharedStorage:
            mbar_ptr_q: mbar_ptr_q_struct
            mbar_ptr_k: mbar_ptr_k_struct
            sQ0: sQ0_struct
            sQ1: sQ1_struct
            sK0: sK0_struct
            sK1: sK1_struct

        storage = smem.allocate(SharedStorage)
        sQ0 = storage.sQ0.get_tensor(
            sQ_layout_fp8_0.outer, swizzle=sQ_layout_fp8_0.inner
        )
        sQ1 = storage.sQ1.get_tensor(
            sQ_layout_fp8_1.outer, swizzle=sQ_layout_fp8_1.inner
        )
        sK0 = storage.sK0.get_tensor(
            sK_layout_fp8_0.outer, swizzle=sK_layout_fp8_0.inner
        )
        sK1 = storage.sK1.get_tensor(
            sK_layout_fp8_1.outer, swizzle=sK_layout_fp8_1.inner
        )
        # The K page cache is physically interleaved by the producer of the
        # cache. Expose the same allocation as a flat [bytes, pages] tensor for
        # TMA while retaining the WGMMA-composed tensor above for MMA.
        sK_pages0 = cute.make_tensor(
            cute.recast_ptr(storage.sK0.data_ptr(), dtype=Uint8),
            cute.make_layout(
                (PAGED_STAGE_BYTES // PAGED_K_SEGMENTS * PAGED_NUM_K_STAGES,),
                stride=(1,),
            ),
        )
        sK_pages1 = cute.make_tensor(
            cute.recast_ptr(storage.sK1.data_ptr(), dtype=Uint8),
            cute.make_layout(
                (PAGED_STAGE_BYTES // PAGED_K_SEGMENTS * PAGED_NUM_K_STAGES,),
                stride=(1,),
            ),
        )
        gQ0 = cute.local_tile(tma_tensor_Q0, (wgmma_n, PAGED_K_SEGMENT), (None, 0))
        gQ1 = cute.local_tile(tma_tensor_Q1, (wgmma_n, PAGED_K_SEGMENT), (None, 0))
        tQsQ0, tQgQ0 = cpasync.tma_partition(
            tma_atom_Q0,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ0, 0, cute.rank(sQ0) - 1),
            cute.group_modes(gQ0, 0, cute.rank(gQ0) - 1),
        )
        tQsQ1, tQgQ1 = cpasync.tma_partition(
            tma_atom_Q1,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ1, 0, cute.rank(sQ1) - 1),
            cute.group_modes(gQ1, 0, cute.rank(gQ1) - 1),
        )
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, PAGED_NUM_MATH_WARPS
        )
        pipeline_q = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_q.data_ptr(),
            num_stages=1,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=wgmma_n * HEAD_DIM,
        )
        pipeline_k = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_k.data_ptr(),
            num_stages=PAGED_NUM_K_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=PAGED_STAGE_BYTES,
        )
        q_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        k_prod = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, PAGED_NUM_K_STAGES
        )
        k_cons = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, PAGED_NUM_K_STAGES
        )

        if tidx >= PAGED_NUM_MATH_THREADS:
            cute.arch.warpgroup_reg_dealloc(GQA_NUM_TMA_REGS)
            producer_idx = tidx - Int32(PAGED_NUM_MATH_THREADS)
            lane_idx_tma = producer_idx % Int32(32)
            is_tma_leader = (producer_idx // Int32(32)) == Int32(
                0
            ) and lane_idx_tma == Int32(0)
            if is_tma_leader:
                producer_work_id = work_id
                while producer_work_id < max_work_items:
                    request_row = producer_work_id // max_tiles_per_row
                    tile_idx = producer_work_id - request_row * max_tiles_per_row
                    q_valid = request_row < query_rows
                    req_idx = Int32(-1)
                    table_idx = Int32(-1)
                    row_start = Int32(0)
                    row_end = Int32(0)
                    token_base = Int32(0)
                    active_q_len = Int32(0)
                    if q_valid:
                        req_idx = Int32(mRowReqIdx[request_row])
                        table_idx = Int32(mRowTableIdx[request_row])
                        token_base = request_row * Int32(q_len)
                        remaining_tokens = Int32(mValidTokens[0]) - token_base
                        active_q_len = cutlass.select_(
                            remaining_tokens < Int32(q_len),
                            remaining_tokens,
                            Int32(q_len),
                        )
                        active_q_len = cutlass.select_(
                            active_q_len > Int32(0), active_q_len, Int32(0)
                        )
                        q_valid = (
                            q_valid
                            & (req_idx >= Int32(0))
                            & (req_idx < Int32(mValidRequests[0]))
                        )
                        q_valid = q_valid & (active_q_len > Int32(0))
                        q_valid = (
                            q_valid & (table_idx >= Int32(0)) & (table_idx < num_reqs)
                        )
                    if q_valid:
                        row_start = Int32(mRowStarts[request_row])
                        for q_slot in cutlass.range_constexpr(q_len):
                            candidate_end = Int32(mRowEnds[token_base + Int32(q_slot)])
                            candidate_end = cutlass.select_(
                                Int32(q_slot) < active_q_len,
                                candidate_end,
                                Int32(0),
                            )
                            row_end = cutlass.select_(
                                candidate_end > row_end, candidate_end, row_end
                            )
                    req_len = row_end - row_start
                    found = (
                        q_valid
                        & (req_idx >= Int32(0))
                        & (table_idx < num_reqs)
                        & (row_start >= Int32(0))
                        & (row_end <= max_regions)
                        & (req_len > Int32(0))
                        & (tile_idx * Int32(PAGED_TILE_KV) < req_len)
                    )
                    if found == Int32(1):
                        pipeline_q.producer_acquire(q_prod)
                        cute.copy(
                            tma_atom_Q0,
                            tQgQ0[None, request_row],
                            tQsQ0[None, q_prod.index],
                            tma_bar_ptr=pipeline_q.producer_get_barrier(q_prod),
                        )
                        cute.copy(
                            tma_atom_Q1,
                            tQgQ1[None, request_row],
                            tQsQ1[None, q_prod.index],
                            tma_bar_ptr=pipeline_q.producer_get_barrier(q_prod),
                        )
                        q_prod.advance()
                        tile_base = row_start + tile_idx * Int32(PAGED_TILE_KV)
                        for stage_iter in cutlass.range_constexpr(PAGED_NUM_K_STAGES):
                            pipeline_k.producer_acquire(k_prod)
                            stage_base = tile_base + Int32(stage_iter * PAGED_STAGE_KV)
                            stage_offset = Int32(k_prod.index) * Int32(
                                PAGED_STAGE_BYTES // PAGED_K_SEGMENTS
                            )
                            for local_pair in cutlass.range_constexpr(
                                PAGED_PAIRS_PER_STAGE
                            ):
                                logical_region = stage_base + Int32(
                                    local_pair * PAGED_TMA_ROWS
                                )
                                valid_page = logical_region < row_end
                                logical_page = logical_region // summaries_per_page
                                valid_page = valid_page & (
                                    logical_page < block_table_cols
                                )
                                safe_col = cutlass.select_(
                                    valid_page, logical_page, Int32(0)
                                )
                                physical_page = mBlockTable[
                                    table_idx * block_table_cols + safe_col
                                ]
                                valid_page = (
                                    valid_page
                                    & (physical_page >= Int32(0))
                                    & (physical_page < num_pages)
                                )
                                safe_page = cutlass.select_(
                                    valid_page, physical_page, Int32(0)
                                )
                                page_slot = (
                                    logical_region - logical_page * summaries_per_page
                                )
                                pair_in_page = page_slot // Int32(PAGED_TMA_ROWS)
                                safe_pair_in_page = cutlass.select_(
                                    valid_page, pair_in_page, Int32(0)
                                )
                                row = Int32(local_pair * PAGED_TMA_ROWS)
                                row_in_atom = row % Int32(8)
                                row_atom = row // Int32(8)
                                row_base = (
                                    stage_offset
                                    + row_in_atom * Int32(PAGED_WGMMA_ATOM_BYTES)
                                    + row_atom * Int32(8 * PAGED_WGMMA_ATOM_BYTES)
                                )
                                gK_page0 = cute.domain_offset(
                                    (Int32(0), Int32(0), safe_pair_in_page, safe_page),
                                    tma_tensor_K0,
                                )
                                gK_page0 = cute.make_tensor(
                                    gK_page0.iterator,
                                    cute.make_layout(
                                        (PAGED_TMA_CHUNK_BYTES, 1),
                                        stride=(
                                            tma_tensor_K0.stride[0],
                                            tma_tensor_K0.stride[1],
                                        ),
                                    ),
                                )
                                gK_page1 = cute.domain_offset(
                                    (Int32(0), Int32(0), safe_pair_in_page, safe_page),
                                    tma_tensor_K1,
                                )
                                gK_page1 = cute.make_tensor(
                                    gK_page1.iterator,
                                    cute.make_layout(
                                        (PAGED_TMA_CHUNK_BYTES, 1),
                                        stride=(
                                            tma_tensor_K1.stride[0],
                                            tma_tensor_K1.stride[1],
                                        ),
                                    ),
                                )
                                sK_page_ptr0 = sparse_utils.elem_pointer_i64_offset(
                                    sK_pages0, (row_base,)
                                )
                                sK_page0 = cute.make_tensor(
                                    sK_page_ptr0,
                                    cute.make_layout(
                                        (PAGED_TMA_CHUNK_BYTES, 1), stride=(1, 1)
                                    ),
                                )
                                sK_page_ptr1 = sparse_utils.elem_pointer_i64_offset(
                                    sK_pages1, (row_base,)
                                )
                                sK_page1 = cute.make_tensor(
                                    sK_page_ptr1,
                                    cute.make_layout(
                                        (PAGED_TMA_CHUNK_BYTES, 1), stride=(1, 1)
                                    ),
                                )
                                load_K0, _, _ = copy_utils.tma_get_copy_fn(
                                    tma_atom_K0,
                                    0,
                                    cute.make_layout(1),
                                    gK_page0,
                                    sK_page0,
                                    single_stage=True,
                                )
                                load_K1, _, _ = copy_utils.tma_get_copy_fn(
                                    tma_atom_K1,
                                    0,
                                    cute.make_layout(1),
                                    gK_page1,
                                    sK_page1,
                                    single_stage=True,
                                )
                                load_K0(
                                    tma_bar_ptr=pipeline_k.producer_get_barrier(k_prod)
                                )
                                load_K1(
                                    tma_bar_ptr=pipeline_k.producer_get_barrier(k_prod)
                                )
                            k_prod.advance()
                    producer_work_id = producer_work_id + work_stride
        else:
            cute.arch.warpgroup_reg_alloc(GQA_NUM_MATH_REGS)
            warp_idx = tidx // Int32(32)
            warp_group_idx = cute.arch.make_warp_uniform(warp_idx // Int32(4))
            lane_idx = tidx % Int32(32)
            warp_group_thread_layout = cute.make_layout(
                PAGED_NUM_MATH_THREADS // 128, stride=128
            )
            thr_mma_thread = tiled_mma.get_slice(tidx)
            wg_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))
            cS = cute.make_identity_tensor((PAGED_STAGE_KV, wgmma_n))
            tScS = thr_mma_thread.partition_C(cS)
            work_id = cute.arch.block_idx()[0]
            while work_id < max_work_items:
                request_row = work_id // max_tiles_per_row
                tile_idx = work_id - request_row * max_tiles_per_row
                q_valid = request_row < query_rows
                req_idx = Int32(-1)
                table_idx = Int32(-1)
                row_start = Int32(0)
                row_end = Int32(0)
                token_base = Int32(0)
                active_q_len = Int32(0)
                if q_valid:
                    req_idx = Int32(mRowReqIdx[request_row])
                    table_idx = Int32(mRowTableIdx[request_row])
                    token_base = request_row * Int32(q_len)
                    remaining_tokens = Int32(mValidTokens[0]) - token_base
                    active_q_len = cutlass.select_(
                        remaining_tokens < Int32(q_len),
                        remaining_tokens,
                        Int32(q_len),
                    )
                    active_q_len = cutlass.select_(
                        active_q_len > Int32(0), active_q_len, Int32(0)
                    )
                    q_valid = (
                        q_valid
                        & (req_idx >= Int32(0))
                        & (req_idx < Int32(mValidRequests[0]))
                    )
                    q_valid = q_valid & (active_q_len > Int32(0))
                    q_valid = q_valid & (table_idx >= Int32(0)) & (table_idx < num_reqs)
                if q_valid:
                    row_start = Int32(mRowStarts[request_row])
                    for q_slot in cutlass.range_constexpr(q_len):
                        candidate_end = Int32(mRowEnds[token_base + Int32(q_slot)])
                        candidate_end = cutlass.select_(
                            Int32(q_slot) < active_q_len,
                            candidate_end,
                            Int32(0),
                        )
                        row_end = cutlass.select_(
                            candidate_end > row_end, candidate_end, row_end
                        )
                req_len = row_end - row_start
                found = (
                    q_valid
                    & (req_idx >= Int32(0))
                    & (table_idx < num_reqs)
                    & (row_start >= Int32(0))
                    & (row_end <= max_regions)
                    & (req_len > Int32(0))
                    & (tile_idx * Int32(PAGED_TILE_KV) < req_len)
                )
                if found == Int32(1):
                    pipeline_q.consumer_wait(q_cons)
                    q_stage = q_cons.index
                    warp_idx = tidx // Int32(32)
                    lane_idx = tidx % Int32(32)
                    warp_offset = warp_idx * Int32(16)
                    lane_group = lane_idx // Int32(4)
                    row0 = warp_offset + lane_group
                    row1 = row0 + Int32(8)
                    lane_mod4 = lane_idx % Int32(4)

                    for stage_iter in cutlass.range_constexpr(PAGED_NUM_K_STAGES):
                        pipeline_k.consumer_wait(k_cons)
                        k_stage = k_cons.index
                        sK_stage0 = cute.make_tensor(
                            elem_pointer(sK0, (0, 0, k_stage)),
                            sK_layout_single_outer_0,
                        )
                        sK_stage1 = cute.make_tensor(
                            elem_pointer(sK1, (0, 0, k_stage)),
                            sK_layout_single_outer_1,
                        )
                        sQ_stage0 = cute.make_tensor(
                            elem_pointer(sQ0, (0, 0, q_stage)),
                            sQ_layout_single_outer_0,
                        )
                        sQ_stage1 = cute.make_tensor(
                            elem_pointer(sQ1, (0, 0, q_stage)),
                            sQ_layout_single_outer_1,
                        )
                        tCrA0 = tiled_mma.make_fragment_A(wg_mma.partition_A(sK_stage0))
                        tCrB0 = tiled_mma.make_fragment_B(wg_mma.partition_B(sQ_stage0))
                        tCrA1 = tiled_mma.make_fragment_A(wg_mma.partition_A(sK_stage1))
                        tCrB1 = tiled_mma.make_fragment_B(wg_mma.partition_B(sQ_stage1))
                        acc = tiled_mma.make_fragment_C(tScS.layout)
                        hop_helpers.warpgroup_gemm_with_optional_swap_wait(
                            tiled_mma,
                            acc,
                            tCrA0,
                            tCrB0,
                            zero_init=True,
                            wg_wait=-1,
                        )
                        hop_helpers.warpgroup_gemm_with_optional_swap_wait(
                            tiled_mma,
                            acc,
                            tCrA1,
                            tCrB1,
                            zero_init=False,
                            wg_wait=0,
                        )
                        acc_val = acc.load()
                        pipeline_k.consumer_release(k_cons)
                        k_cons.advance()

                        kv_base = (
                            row_start
                            + tile_idx * Int32(PAGED_TILE_KV)
                            + Int32(stage_iter * PAGED_STAGE_KV)
                        )
                        sums0 = [Float32(0.0) for _ in range(q_len)]
                        sums1 = [Float32(0.0) for _ in range(q_len)]
                        for r in cutlass.range(cute.size(tScS), unroll_full=True):
                            m = tScS[r][0]
                            n = tScS[r][1]
                            physical_q_slot = n // Int32(Q_HEADS_PER_KV)
                            raw = acc_val[r]
                            relu = cutlass.select_(
                                raw > Float32(0.0), raw, Float32(0.0)
                            )
                            weight = _load_f32_global(mW, (request_row, n))
                            weighted = relu * weight
                            for q_slot in cutlass.range_constexpr(q_len):
                                sums0[q_slot] = if_generate(
                                    (m == row0) & (physical_q_slot == Int32(q_slot)),
                                    lambda current: current + weighted,
                                    lambda current: current,
                                    [sums0[q_slot]],
                                    [Float32],
                                )
                                sums1[q_slot] = if_generate(
                                    (m == row1) & (physical_q_slot == Int32(q_slot)),
                                    lambda current: current + weighted,
                                    lambda current: current,
                                    [sums1[q_slot]],
                                    [Float32],
                                )
                        global_k0 = kv_base + row0
                        global_k1 = kv_base + row1
                        for q_slot in cutlass.range_constexpr(q_len):
                            sums0[q_slot] = sums0[q_slot] + cute.arch.shuffle_sync_bfly(
                                sums0[q_slot], offset=1
                            )
                            sums1[q_slot] = sums1[q_slot] + cute.arch.shuffle_sync_bfly(
                                sums1[q_slot], offset=1
                            )
                            sums0[q_slot] = sums0[q_slot] + cute.arch.shuffle_sync_bfly(
                                sums0[q_slot], offset=2
                            )
                            sums1[q_slot] = sums1[q_slot] + cute.arch.shuffle_sync_bfly(
                                sums1[q_slot], offset=2
                            )
                            q_row = token_base + Int32(q_slot)
                            q_row_end = Int32(mRowEnds[q_row])
                            q_slot_valid = Int32(q_slot) < active_q_len
                            if (
                                (lane_mod4 == Int32(0))
                                & q_slot_valid
                                & (global_k0 < q_row_end)
                            ):
                                dst0 = cute.make_tensor(
                                    elem_pointer(
                                        mO,
                                        (q_row * max_regions + global_k0,),
                                    ),
                                    cute.make_layout((1,), stride=(1,)),
                                )
                                dst0[0] = mO.element_type(sums0[q_slot])
                            if (
                                (lane_mod4 == Int32(0))
                                & q_slot_valid
                                & (global_k1 < q_row_end)
                            ):
                                dst1 = cute.make_tensor(
                                    elem_pointer(
                                        mO,
                                        (q_row * max_regions + global_k1,),
                                    ),
                                    cute.make_layout((1,), stride=(1,)),
                                )
                                dst1[0] = mO.element_type(sums1[q_slot])
                    pipeline_q.consumer_release(q_cons)
                    q_cons.advance()
                work_id = work_id + work_stride


# Note(wangbojun/codex): CuTe's persistent compile cache does not include the
# bodies of called @cute.jit helpers in the entrypoint key.  Keep this
# entrypoint versioned whenever the WGMMA helper changes so serving warmup
# cannot load a cubin compiled with older accumulation semantics.
@cute.jit
def _batch_decode_logits_paged_sm90_call_v9_impl(
    mQ: cute.Tensor,
    mKCache: cute.Tensor,
    mBlockTable: cute.Tensor,
    mRowReqIdx: cute.Tensor,
    mRowTableIdx: cute.Tensor,
    mRowStarts: cute.Tensor,
    mRowEnds: cute.Tensor,
    mValidRequests: cute.Tensor,
    mValidTokens: cute.Tensor,
    mW: cute.Tensor,
    mO: cute.Tensor,
    stream,
    max_regions: Int32,
    query_rows: Int32,
    num_reqs: Int32,
    block_table_cols: Int32,
    num_pages: Int32,
    summaries_per_page: Int32,
    max_tiles_per_row: Int32,
    max_work_items: Int32,
    persistent_ctas: Int32,
    q_len: cutlass.Constexpr[int],
    wgmma_n: cutlass.Constexpr[int],
):
    mma_tiler_mnk = (PAGED_STAGE_KV, wgmma_n, PAGED_K_SEGMENT)
    tiled_mma = sm90_utils.make_trivial_tiled_mma(
        Float8E4M3FN,
        Float8E4M3FN,
        LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        Float32,
        atom_layout_mnk=(PAGED_STAGE_KV // 64, 1, 1),
        tiler_mn=(64, wgmma_n),
    )
    sQ_layout_fp8_0 = sm90_utils.make_smem_layout_b(
        LayoutEnum.ROW_MAJOR,
        mma_tiler_mnk,
        Float8E4M3FN,
        1,
    )
    sQ_layout_fp8_1 = sm90_utils.make_smem_layout_b(
        LayoutEnum.ROW_MAJOR,
        mma_tiler_mnk,
        Float8E4M3FN,
        1,
    )
    sK_layout_fp8_0 = sm90_utils.make_smem_layout_a(
        LayoutEnum.ROW_MAJOR,
        mma_tiler_mnk,
        Float8E4M3FN,
        PAGED_NUM_K_STAGES,
    )
    sK_layout_fp8_1 = sm90_utils.make_smem_layout_a(
        LayoutEnum.ROW_MAJOR,
        mma_tiler_mnk,
        Float8E4M3FN,
        PAGED_NUM_K_STAGES,
    )
    sQ_layout_single_0 = cute.slice_(sQ_layout_fp8_0, (None, None, 0))
    sQ_layout_single_1 = cute.slice_(sQ_layout_fp8_1, (None, None, 0))
    sK_layout_single_0 = cute.slice_(sK_layout_fp8_0, (None, None, 0))
    sK_layout_single_1 = cute.slice_(sK_layout_fp8_1, (None, None, 0))
    q_segment_layout = cute.make_layout(
        (mQ.shape[0], PAGED_K_SEGMENT),
        stride=(mQ.stride[0], mQ.stride[1]),
    )
    mQ_segment0 = cute.make_tensor(mQ.iterator, q_segment_layout)
    mQ_segment1_ptr = cute.domain_offset((Int32(0), Int32(PAGED_K_SEGMENT)), mQ)
    mQ_segment1 = cute.make_tensor(mQ_segment1_ptr.iterator, q_segment_layout)
    tma_atom_Q0, tma_tensor_Q0 = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mQ_segment0,
        sQ_layout_single_0,
        (wgmma_n, PAGED_K_SEGMENT),
    )
    tma_atom_Q1, tma_tensor_Q1 = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mQ_segment1,
        sQ_layout_single_1,
        (wgmma_n, PAGED_K_SEGMENT),
    )
    mKCache_pages = cute.make_tensor(
        cute.recast_ptr(mKCache.iterator, dtype=Uint8),
        cute.make_layout(
            (
                PAGED_TMA_CHUNK_BYTES,
                PAGED_TMA_CHUNKS_PER_PAIR,
                summaries_per_page // Int32(PAGED_TMA_ROWS),
                mKCache.shape[0],
            ),
            stride=(
                1,
                PAGED_TMA_CHUNK_BYTES,
                Int32(PAGED_TMA_PAIR_BYTES),
                summaries_per_page * Int32(HEAD_DIM),
            ),
        ),
    )
    k_page_layout = cute.make_layout(
        (
            PAGED_TMA_CHUNK_BYTES,
            1,
            summaries_per_page // Int32(PAGED_TMA_ROWS),
            mKCache.shape[0],
        ),
        stride=(
            1,
            PAGED_TMA_CHUNK_BYTES,
            Int32(PAGED_TMA_PAIR_BYTES),
            summaries_per_page * Int32(HEAD_DIM),
        ),
    )
    mKCache_pages0 = cute.make_tensor(mKCache_pages.iterator, k_page_layout)
    mKCache_pages1_ptr = cute.domain_offset(
        (Int32(0), Int32(1), Int32(0), Int32(0)), mKCache_pages
    )
    mKCache_pages1 = cute.make_tensor(mKCache_pages1_ptr.iterator, k_page_layout)
    sK_page_layout = cute.make_layout(
        (PAGED_TMA_CHUNK_BYTES, 1),
        stride=(1, 1),
    )
    tma_atom_K0, tma_tensor_K0 = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mKCache_pages0,
        sK_page_layout,
        (PAGED_TMA_CHUNK_BYTES, 1),
    )
    tma_atom_K1, tma_tensor_K1 = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mKCache_pages1,
        sK_page_layout,
        (PAGED_TMA_CHUNK_BYTES, 1),
    )
    grid = (persistent_ctas, 1, 1)
    _batch_decode_logits_paged_sm90_kernel(
        tma_atom_Q0,
        tma_tensor_Q0,
        tma_atom_Q1,
        tma_tensor_Q1,
        tma_atom_K0,
        tma_tensor_K0,
        tma_atom_K1,
        tma_tensor_K1,
        mQ,
        mKCache,
        mBlockTable,
        mRowReqIdx,
        mRowTableIdx,
        mRowStarts,
        mRowEnds,
        mValidRequests,
        mValidTokens,
        tiled_mma,
        sQ_layout_fp8_0,
        sQ_layout_fp8_1,
        sK_layout_fp8_0,
        sK_layout_fp8_1,
        sQ_layout_single_0.outer,
        sQ_layout_single_1.outer,
        sK_layout_single_0.outer,
        sK_layout_single_1.outer,
        mW,
        mO,
        max_regions,
        query_rows,
        num_reqs,
        block_table_cols,
        num_pages,
        summaries_per_page,
        max_tiles_per_row,
        max_work_items,
    ).launch(grid=grid, block=[PAGED_NUM_THREADS, 1, 1], stream=stream)


def _make_batch_decode_logits_paged_sm90_call_v9(q_len: int, wgmma_n: int):
    global GQA_COMPILED_Q_LEN
    global GQA_COMPILED_WGMMA_N
    GQA_COMPILED_Q_LEN = q_len
    GQA_COMPILED_WGMMA_N = wgmma_n

    @cute.jit
    def _call(
        mQ: cute.Tensor,
        mKCache: cute.Tensor,
        mBlockTable: cute.Tensor,
        mRowReqIdx: cute.Tensor,
        mRowTableIdx: cute.Tensor,
        mRowStarts: cute.Tensor,
        mRowEnds: cute.Tensor,
        mValidRequests: cute.Tensor,
        mValidTokens: cute.Tensor,
        mW: cute.Tensor,
        mO: cute.Tensor,
        stream,
        max_regions: Int32,
        query_rows: Int32,
        num_reqs: Int32,
        block_table_cols: Int32,
        num_pages: Int32,
        summaries_per_page: Int32,
        max_tiles_per_row: Int32,
        max_work_items: Int32,
        persistent_ctas: Int32,
    ):
        _batch_decode_logits_paged_sm90_call_v9_impl(
            mQ,
            mKCache,
            mBlockTable,
            mRowReqIdx,
            mRowTableIdx,
            mRowStarts,
            mRowEnds,
            mValidRequests,
            mValidTokens,
            mW,
            mO,
            stream,
            max_regions,
            query_rows,
            num_reqs,
            block_table_cols,
            num_pages,
            summaries_per_page,
            max_tiles_per_row,
            max_work_items,
            persistent_ctas,
            q_len,
            wgmma_n,
        )

    return _call


@functools.cache
def _compile_paged_for_signature(
    device_key: tuple[str, Optional[int]],
    q_len: int,
    wgmma_n: int,
    q_runtime_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    k_cache_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    block_table_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    row_req_idx_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    row_table_idx_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    row_starts_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    row_ends_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    valid_requests_signature: tuple[
        torch.dtype, tuple[object, ...], tuple[object, ...]
    ],
    valid_tokens_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    w_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    out_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
):
    device = sparse_utils.device_from_cache_key(device_key)
    q_runtime = _placeholder_from_signature(
        q_runtime_signature,
        device=device,
        dynamic_shape_fill=wgmma_n,
    )
    k_cache = _placeholder_from_signature(
        k_cache_signature,
        device=device,
        dynamic_shape_fill=1,
        # Keep a valid compact stride order while the outer page stride is
        # marked symbolic below.  The value is only a fake descriptor value.
        dynamic_stride_fill=8192,
    )
    block_table_flat = _placeholder_from_signature(
        block_table_signature,
        device=device,
        dynamic_shape_fill=1,
    )
    row_req_idx = _placeholder_from_signature(
        row_req_idx_signature,
        device=device,
        dynamic_shape_fill=1,
    )
    row_table_idx = _placeholder_from_signature(
        row_table_idx_signature,
        device=device,
        dynamic_shape_fill=1,
    )
    row_starts = _placeholder_from_signature(
        row_starts_signature,
        device=device,
        dynamic_shape_fill=1,
    )
    row_ends = _placeholder_from_signature(
        row_ends_signature,
        device=device,
        dynamic_shape_fill=1,
    )
    valid_requests = _placeholder_from_signature(
        valid_requests_signature,
        device=device,
        dynamic_shape_fill=1,
    )
    valid_tokens = _placeholder_from_signature(
        valid_tokens_signature,
        device=device,
        dynamic_shape_fill=1,
    )
    w_mat = _placeholder_from_signature(
        w_signature,
        device=device,
        dynamic_shape_fill=1,
    )
    out = _placeholder_from_signature(
        out_signature,
        device=device,
        dynamic_shape_fill=PAGED_TILE_KV,
    )
    fQ = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        q_runtime,
        alignment=16,
        dynamic_shape_dim=0,
        divisibility=wgmma_n,
    )
    fK = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        k_cache,
        alignment=16,
        dynamic_shape_dim=1,
        dynamic_stride_dims=(0,),
    )
    fK = fK.mark_compact_shape_dynamic(
        mode=0,
        stride_order=k_cache.dim_order(),
        divisibility=1,
    )
    fBlockTable = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        block_table_flat,
        alignment=16,
        dynamic_shape_dim=0,
    )
    fRowReqIdx = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        row_req_idx,
        alignment=16,
        dynamic_shape_dim=0,
    )
    fRowTableIdx = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        row_table_idx,
        alignment=16,
        dynamic_shape_dim=0,
    )
    fRowStarts = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        row_starts,
        alignment=16,
        dynamic_shape_dim=0,
    )
    fRowEnds = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        row_ends,
        alignment=16,
        dynamic_shape_dim=0,
    )
    fValidRequests = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        valid_requests,
        alignment=4,
        dynamic_shape_dim=0,
    )
    fValidTokens = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        valid_tokens,
        alignment=4,
        dynamic_shape_dim=0,
    )
    fW = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        w_mat,
        alignment=16,
        dynamic_shape_dim=0,
    )
    fO = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        out,
        alignment=16,
        dynamic_shape_dim=0,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return _cute_compile(
        _make_batch_decode_logits_paged_sm90_call_v9(q_len, wgmma_n),
        fQ,
        fK,
        fBlockTable,
        fRowReqIdx,
        fRowTableIdx,
        fRowStarts,
        fRowEnds,
        fValidRequests,
        fValidTokens,
        fW,
        fO,
        stream_fake,
        Int32(PAGED_TILE_KV),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(2),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
    )


def _batch_decode_weighted_relu_logits_sum_paged_sm90_steptron_gqa_impl(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    summary: torch.Tensor,
    block_table: torch.Tensor,
    row_req_idx: torch.Tensor,
    row_table_idx: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    q_runtime: torch.Tensor,
    kernel_weights: torch.Tensor,
    out: torch.Tensor,
    valid_requests: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> None:
    query_rows = int(index_q.shape[0])
    q_len = int(index_q.shape[1])
    q_heads_per_kv = int(index_q.shape[3])
    wgmma_n = _wgmma_n_for_q_len(q_len)
    max_regions = int(out.shape[2])
    summaries_per_page = int(summary.shape[1])
    if max_regions <= 0:
        raise ValueError("paged WGMMA logits output must have positive region capacity")
    if summaries_per_page <= 0 or summaries_per_page % 8 != 0:
        raise ValueError(
            "paged WGMMA logits requires summary rows per physical page to "
            "be a positive multiple of the 8-row WGMMA swizzle period, "
            f"got {summaries_per_page}"
        )
    required_block_cols = (max_regions + summaries_per_page - 1) // summaries_per_page
    if int(block_table.shape[1]) < required_block_cols:
        raise ValueError(
            "block_table does not cover the requested logical regions: "
            f"need {required_block_cols} columns, got {int(block_table.shape[1])}"
        )
    if index_q.dtype == torch.uint8:
        q_input = index_q.view(torch.float8_e4m3fn)
    elif index_q.dtype == torch.float8_e4m3fn:
        q_input = index_q
    else:
        raise ValueError(
            "paged WGMMA logits requires pre-quantized FP8 index_q "
            "(float8_e4m3fn or uint8 bits)"
        )
    if summary.dtype == torch.uint8:
        k_input = summary.view(torch.float8_e4m3fn)
    elif summary.dtype == torch.float8_e4m3fn:
        k_input = summary
    else:
        raise ValueError(
            "paged WGMMA logits requires pre-quantized FP8 summary "
            "(float8_e4m3fn or uint8 bits)"
        )
    if block_table.dtype != torch.int32:
        raise ValueError(f"block_table must be int32, got {block_table.dtype}")
    if (
        row_req_idx.dtype != torch.int32
        or row_req_idx.dim() != 1
        or tuple(row_req_idx.shape) != (query_rows,)
        or not row_req_idx.is_contiguous()
    ):
        raise ValueError(
            f"row_req_idx must be contiguous int32 [{query_rows}], got "
            f"{tuple(row_req_idx.shape)} {row_req_idx.dtype}"
        )
    if (
        row_table_idx.dtype != torch.int32
        or row_table_idx.dim() != 1
        or tuple(row_table_idx.shape) != (query_rows,)
        or not row_table_idx.is_contiguous()
    ):
        raise ValueError(
            f"row_table_idx must be contiguous int32 [{query_rows}], got "
            f"{tuple(row_table_idx.shape)} {row_table_idx.dtype}"
        )
    if (
        row_starts.dtype != torch.int32
        or row_starts.dim() != 1
        or tuple(row_starts.shape) != (query_rows,)
        or not row_starts.is_contiguous()
    ):
        raise ValueError(
            f"row_starts must be contiguous int32 [{query_rows}], got "
            f"{tuple(row_starts.shape)} {row_starts.dtype}"
        )
    if (
        row_ends.dtype != torch.int32
        or row_ends.dim() != 1
        or tuple(row_ends.shape) != (query_rows * q_len,)
        or not row_ends.is_contiguous()
    ):
        raise ValueError(
            f"row_ends must be contiguous int32 [{query_rows * q_len}], got "
            f"{tuple(row_ends.shape)} {row_ends.dtype}"
        )
    if out.dtype != torch.float32:
        raise ValueError(f"paged WGMMA logits output must be float32, got {out.dtype}")
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

    _validate_q_heads_per_kv(q_heads_per_kv)
    if (
        q_runtime.dtype != torch.float8_e4m3fn
        or tuple(q_runtime.shape) != (query_rows * wgmma_n, HEAD_DIM)
        or q_runtime.device != index_q.device
        or not q_runtime.is_contiguous()
    ):
        raise ValueError(
            "q_runtime must be caller-owned contiguous FP8 "
            f"[{query_rows * wgmma_n},{HEAD_DIM}] on {index_q.device}, got "
            f"shape={tuple(q_runtime.shape)}, dtype={q_runtime.dtype}, "
            f"device={q_runtime.device}"
        )
    if (
        kernel_weights.dtype != torch.float32
        or tuple(kernel_weights.shape) != (query_rows, wgmma_n)
        or kernel_weights.device != index_q.device
        or not kernel_weights.is_contiguous()
    ):
        raise ValueError(
            "kernel_weights must be caller-owned contiguous FP32 "
            f"[{query_rows},{wgmma_n}] on {index_q.device}, got "
            f"shape={tuple(kernel_weights.shape)}, dtype={kernel_weights.dtype}, "
            f"device={kernel_weights.device}"
        )
    pack = _compile_pack_fp8_qw_for_signature(
        sparse_utils.device_cache_key(index_q.device),
        q_len,
        wgmma_n,
        _tensor_signature_dynamic(q_input, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(weights, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(q_runtime, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(kernel_weights, dynamic_shape_dims=(0,)),
    )
    total_q = q_runtime.numel()
    total_w = kernel_weights.numel()
    pack(
        q_input,
        weights,
        q_runtime,
        kernel_weights,
        Int32((total_q + total_w + GQA_PREPARE_THREADS - 1) // GQA_PREPARE_THREADS),
        Int32(total_q),
        Int32(total_w),
        Int32(query_rows),
    )
    paged = _compile_paged_for_signature(
        sparse_utils.device_cache_key(index_q.device),
        q_len,
        wgmma_n,
        _tensor_signature_dynamic(q_runtime, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(
            k_input,
            dynamic_shape_dims=(0, 1),
            dynamic_stride_dims=(0,),
        ),
        _tensor_signature_dynamic(
            block_table.reshape(-1),
            dynamic_shape_dims=(0,),
        ),
        _tensor_signature_dynamic(row_req_idx, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(row_table_idx, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(row_starts, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(row_ends, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(valid_requests, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(valid_tokens, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(kernel_weights, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(out.reshape(-1), dynamic_shape_dims=(0,)),
    )
    paged(
        q_runtime,
        k_input,
        block_table.reshape(-1),
        row_req_idx,
        row_table_idx,
        row_starts,
        row_ends,
        valid_requests,
        valid_tokens,
        kernel_weights,
        out.reshape(-1),
        Int32(max_regions),
        Int32(query_rows),
        Int32(block_table.shape[0]),
        Int32(block_table.shape[1]),
        Int32(summary.shape[0]),
        Int32(summaries_per_page),
        Int32((max_regions + PAGED_TILE_KV - 1) // PAGED_TILE_KV),
        Int32(query_rows * ((max_regions + PAGED_TILE_KV - 1) // PAGED_TILE_KV)),
        Int32(
            min(
                query_rows * ((max_regions + PAGED_TILE_KV - 1) // PAGED_TILE_KV),
                max(
                    1,
                    int(
                        torch.cuda.get_device_properties(
                            index_q.device
                        ).multi_processor_count
                    )
                    * PAGED_PERSISTENT_CTAS_PER_SM,
                ),
            )
        ),
    )


@torch.library.custom_op(
    "optimus_cutedsl::batch_decode_weighted_relu_logits_sum_paged_sm90_steptron_gqa_out",
    mutates_args=("q_runtime", "kernel_weights", "out"),
    device_types="cuda",
)
def _batch_decode_weighted_relu_logits_sum_paged_sm90_steptron_gqa_out(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    summary: torch.Tensor,
    block_table: torch.Tensor,
    row_req_idx: torch.Tensor,
    row_table_idx: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    q_runtime: torch.Tensor,
    kernel_weights: torch.Tensor,
    out: torch.Tensor,
    valid_requests: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> None:
    _batch_decode_weighted_relu_logits_sum_paged_sm90_steptron_gqa_impl(
        index_q,
        weights,
        summary,
        block_table,
        row_req_idx,
        row_table_idx,
        row_starts,
        row_ends,
        q_runtime,
        kernel_weights,
        out,
        valid_requests,
        valid_tokens,
    )


def batch_decode_weighted_relu_logits_sum_paged_sm90_steptron_gqa(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    summary: torch.Tensor,
    block_table: torch.Tensor,
    row_req_idx: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    q_runtime: Optional[torch.Tensor] = None,
    kernel_weights: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    valid_requests: Optional[torch.Tensor] = None,
    valid_tokens: Optional[torch.Tensor] = None,
    row_table_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if index_q.dim() != 5 or tuple(index_q.shape[2:]) != (1, 4, HEAD_DIM):
        raise ValueError(
            f"paged WGMMA index_q must be [B,Q,1,4,256], got {tuple(index_q.shape)}"
        )
    q_len = int(index_q.shape[1])
    _wgmma_n_for_q_len(q_len)
    if weights.dim() != 4 or tuple(weights.shape) != (
        int(index_q.shape[0]),
        q_len,
        1,
        4,
    ):
        raise ValueError(
            f"paged WGMMA weights must be [B,Q,1,4], got {tuple(weights.shape)}"
        )
    if (
        summary.dim() != 4
        or int(summary.shape[1]) <= 0
        or int(summary.shape[1]) % 8 != 0
        or tuple(summary.shape[2:]) != (1, HEAD_DIM)
    ):
        raise ValueError(
            "paged WGMMA summary must be [num_pages,rows_multiple_of_8,1,256], got "
            f"{tuple(summary.shape)}"
        )
    if block_table.dim() != 2:
        raise ValueError(
            "paged WGMMA block_table must be [B,logical_pages], got "
            f"{tuple(block_table.shape)}"
        )
    if index_q.device.type != "cuda":
        raise RuntimeError(
            "batch_decode_weighted_relu_logits_sum_paged_sm90_steptron_gqa "
            "requires CUDA tensors"
        )
    if (
        not index_q.is_contiguous()
        or not weights.is_contiguous()
        or not summary.is_contiguous()
        or not block_table.is_contiguous()
        or not row_req_idx.is_contiguous()
        or not row_starts.is_contiguous()
        or not row_ends.is_contiguous()
    ):
        raise ValueError(
            "index_q/weights/summary/block_table/row metadata must be contiguous"
        )
    if index_q.dtype not in (torch.float8_e4m3fn, torch.uint8):
        raise ValueError("index_q must be pre-quantized FP8")
    if summary.dtype not in (torch.float8_e4m3fn, torch.uint8):
        raise ValueError("summary must be pre-quantized FP8")
    if out is None:
        raise ValueError(
            "paged WGMMA logits requires caller-owned out with padded logical "
            "region length; allocate it before CUDA Graph capture"
        )
    if q_runtime is None or kernel_weights is None:
        raise ValueError(
            "paged WGMMA logits requires caller-owned q_runtime and "
            "kernel_weights workspaces; allocate them before CUDA Graph capture"
        )
    if valid_requests is None:
        valid_requests = torch.full(
            (1,), int(index_q.shape[0]), dtype=torch.int32, device=index_q.device
        )
    if valid_tokens is None:
        valid_tokens = torch.full(
            (1,),
            int(index_q.shape[0]) * q_len,
            dtype=torch.int32,
            device=index_q.device,
        )
    if row_table_idx is None:
        row_table_idx = row_req_idx
    if not row_table_idx.is_contiguous():
        raise ValueError("row_table_idx must be contiguous")
    if out.device != index_q.device or out.dtype != torch.float32:
        raise ValueError("out must be CUDA float32 on the same device as index_q")
    if out.dim() != 3 or tuple(out.shape[:2]) != (int(index_q.shape[0]), q_len):
        raise ValueError(f"out must be [B,Q,padded_regions], got {tuple(out.shape)}")
    if tuple(row_ends.shape) != (int(index_q.shape[0]), q_len):
        raise ValueError(f"row_ends must be [B,Q], got {tuple(row_ends.shape)}")
    _batch_decode_weighted_relu_logits_sum_paged_sm90_steptron_gqa_out(
        index_q,
        weights,
        summary,
        block_table,
        row_req_idx,
        row_table_idx,
        row_starts,
        row_ends.reshape(-1),
        q_runtime,
        kernel_weights,
        out,
        valid_requests,
        valid_tokens,
    )
    return out


__all__ = [
    "batch_decode_logits_wgmma_n",
    "batch_decode_weighted_relu_logits_sum_paged_sm90_steptron_gqa",
]
