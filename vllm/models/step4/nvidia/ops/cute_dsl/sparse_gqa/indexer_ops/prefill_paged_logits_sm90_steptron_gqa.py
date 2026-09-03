from __future__ import annotations

import functools
from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import torch
from cutlass import Float32, Float8E4M3FN, Int32, Uint8
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import if_generate
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils

from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.kernel_utils import cute_utils as sparse_utils
from vllm.models.step4.nvidia.ops.cute_dsl.flash_attn import copy_utils
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.kernel_utils import hopper_helpers as hop_helpers
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.kernel_utils.fp_compat import cvt_f32_to_e4m3
from vllm.models.step4.nvidia.ops.cute_dsl.utils import elem_pointer


HEAD_DIM = 256
SUMMARY_KV_HEADS = 1
COMPILE_PLACEHOLDER_DIM = 1
SUPPORTED_Q_HEADS_PER_KV = (2, 4, 8)
PREFILL_BLOCK_QH = 128
PREFILL_BLOCK_KV = 256
PREFILL_K_SEGMENT = HEAD_DIM // 2
PREFILL_TMA_ROWS = 8
PREFILL_TMA_CHUNK_BYTES = PREFILL_TMA_ROWS * PREFILL_K_SEGMENT
PREFILL_TMA_PAGE_STRIDE_BYTES = PREFILL_TMA_ROWS * HEAD_DIM
PREFILL_Q_STAGES = 2
PREFILL_K_STAGES = 2
PREFILL_NUM_MATH_THREADS = 512
PREFILL_NUM_TMA_THREADS = 128
PREFILL_NUM_THREADS = PREFILL_NUM_MATH_THREADS + PREFILL_NUM_TMA_THREADS
LANES_PER_WARP = 32
WARPS_PER_WARPGROUP = 4
HEADS_PER_PAIR = 2
Q_LANES_PER_GROUP = PREFILL_TMA_ROWS // 2
ROWS_PER_WEIGHT_TILE = Q_LANES_PER_GROUP
WEIGHT_TILE_VALUES = PREFILL_BLOCK_QH // ROWS_PER_WEIGHT_TILE
OUTPUT_ROW_STRIDE = PREFILL_TMA_ROWS
ROWS_PER_WARP_STRIP = LANES_PER_WARP // 2
M_BARRIER_PTRS_PER_STAGE = 2
PREFILL_NUM_MATH_WARPS = PREFILL_NUM_MATH_THREADS // LANES_PER_WARP
PREFILL_NUM_TMA_REGS = 32
PREFILL_NUM_MATH_REGS = 112
PREFILL_PREPARE_THREADS = 256
PREFILL_MMA_TILE_M = PREFILL_BLOCK_KV // WARPS_PER_WARPGROUP
PREFILL_NUM_WARPGROUPS = PREFILL_NUM_MATH_THREADS // PREFILL_NUM_TMA_THREADS
PREFILL_WARPGROUP_STRIDE = PREFILL_NUM_TMA_THREADS
PREFILL_PERSISTENT_CTAS_PER_SM = 32
TMA_COPY_ALIGN_BYTES = 16
PREFILL_COMPILED_Q_HEADS_PER_KV = 4
PREFILL_COMPILED_Q_PAIR_GROUPS = PREFILL_COMPILED_Q_HEADS_PER_KV // HEADS_PER_PAIR
PREFILL_COMPILED_Q_SLOTS_PER_BLOCK = PREFILL_BLOCK_QH // (PREFILL_COMPILED_Q_HEADS_PER_KV * ROWS_PER_WEIGHT_TILE)
PREFILL_COMPILED_BLOCK_Q = PREFILL_BLOCK_QH // PREFILL_COMPILED_Q_HEADS_PER_KV
PREFILL_COMPILED_Q_SLOT_SHIFT = PREFILL_COMPILED_Q_PAIR_GROUPS.bit_length() + 1
PREFILL_COMPILED_PAIR_GROUP_MASK = PREFILL_COMPILED_Q_PAIR_GROUPS - 1


def _cute_compile(func, *args):
    return cute.compile(func, *args, options="--enable-tvm-ffi --opt-level 2")


def _prefill_block_q_for_heads(q_heads_per_kv: int) -> int:
    h = int(q_heads_per_kv)
    if h in SUPPORTED_Q_HEADS_PER_KV and PREFILL_BLOCK_QH % h == 0:
        return PREFILL_BLOCK_QH // h
    raise ValueError(
        "paged prefill logits supports q_heads_per_kv in "
        f"{SUPPORTED_Q_HEADS_PER_KV}, got {h}"
    )


@functools.cache
def _sm_count(device_key: tuple[str, Optional[int]]) -> int:
    device = sparse_utils.device_from_cache_key(device_key)
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


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


def _load_f32_2d(tensor: cute.Tensor, row: Int32, col: Int32) -> Float32:
    src = cute.make_tensor(
        elem_pointer(tensor, (row, col)),
        cute.make_layout((1,), stride=(1,)),
    )
    return Float32(src[0])


@cute.jit
def _select_tree_32(values: list[Float32], index: Int32) -> Float32:
    stage0 = [Float32(0.0) for _ in range(16)]
    stage1 = [Float32(0.0) for _ in range(8)]
    stage2 = [Float32(0.0) for _ in range(4)]
    stage3 = [Float32(0.0) for _ in range(2)]
    bit0 = ((index >> Int32(0)) & Int32(1)) == Int32(1)
    bit1 = ((index >> Int32(1)) & Int32(1)) == Int32(1)
    bit2 = ((index >> Int32(2)) & Int32(1)) == Int32(1)
    bit3 = ((index >> Int32(3)) & Int32(1)) == Int32(1)
    bit4 = ((index >> Int32(4)) & Int32(1)) == Int32(1)
    for i in cutlass.range_constexpr(16):
        stage0[i] = cutlass.select_(bit0, values[2 * i + 1], values[2 * i])
    for i in cutlass.range_constexpr(8):
        stage1[i] = cutlass.select_(bit1, stage0[2 * i + 1], stage0[2 * i])
    for i in cutlass.range_constexpr(4):
        stage2[i] = cutlass.select_(bit2, stage1[2 * i + 1], stage1[2 * i])
    for i in cutlass.range_constexpr(2):
        stage3[i] = cutlass.select_(bit3, stage2[2 * i + 1], stage2[2 * i])
    return cutlass.select_(bit4, stage3[1], stage3[0])


@cute.kernel
def _pack_prefill_qw_kernel(
    mQIn: cute.Tensor,
    mWIn: cute.Tensor,
    mQOut: cute.Tensor,
    mWOut: cute.Tensor,
    total_q_elems: Int32,
    total_w_elems: Int32,
    batch_q: Int32,
    q_heads_per_kv: cutlass.Constexpr[int],
    block_q: cutlass.Constexpr[int],
    q_slot_shift: cutlass.Constexpr[int],
    pair_group_mask: cutlass.Constexpr[int],
):
    tid = (
        cute.arch.block_idx()[0] * Int32(PREFILL_PREPARE_THREADS)
        + cute.arch.thread_idx()[0]
    )
    total = total_q_elems + total_w_elems
    if tid < total_q_elems:
        dim = tid % Int32(HEAD_DIM)
        packed_row = tid // Int32(HEAD_DIM)
        local_row = packed_row % Int32(PREFILL_BLOCK_QH)
        q_block = packed_row // Int32(PREFILL_BLOCK_QH)
        virtual_slot = local_row >> Int32(1)
        head_pair = local_row & Int32(HEADS_PER_PAIR - 1)
        q_group = virtual_slot >> Int32(q_slot_shift)
        q_lane = virtual_slot & Int32(Q_LANES_PER_GROUP - 1)
        pair_group = (virtual_slot >> Int32(2)) & Int32(pair_group_mask)
        q_in_block = (q_group << Int32(2)) | q_lane
        q_idx = q_block * Int32(block_q) + q_in_block
        head = (pair_group << Int32(1)) | head_pair
        valid = q_idx < batch_q
        safe_q = cutlass.select_(valid, q_idx, Int32(0))
        safe_head = cutlass.select_(valid, head, Int32(0))
        value = mQIn[safe_q, Int32(0), safe_head, dim]
        mQOut[packed_row, dim] = cutlass.select_(
            valid, cvt_f32_to_e4m3(Float32(value)).to(Uint8), Uint8(0)
        )
    if (tid >= total_q_elems) & (tid < total):
        weight_tid = tid - total_q_elems
        head = weight_tid % Int32(q_heads_per_kv)
        q_idx = weight_tid // Int32(q_heads_per_kv)
        valid = q_idx < batch_q
        safe_q = cutlass.select_(valid, q_idx, Int32(0))
        value = mWIn[safe_q, Int32(0), head]
        mWOut[q_idx, head] = cutlass.select_(
            valid, Float32(value), Float32(0.0)
        )


def _make_pack_prefill_qw_call(q_heads_per_kv: int, block_q: int):
    q_pair_groups = q_heads_per_kv // HEADS_PER_PAIR
    q_slot_shift = q_pair_groups.bit_length() + 1
    pair_group_mask = q_pair_groups - 1

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
        batch_q: Int32,
    ):
        _pack_prefill_qw_kernel(
            mQIn,
            mWIn,
            mQOut,
            mWOut,
            total_q_elems,
            total_w_elems,
            batch_q,
            q_heads_per_kv,
            block_q,
            q_slot_shift,
            pair_group_mask,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[PREFILL_PREPARE_THREADS, 1, 1],
            stream=stream,
        )

    return _call


@functools.cache
def _compile_pack_prefill_qw_for_signature(
    device_key: tuple[str, Optional[int]],
    q_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    w_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    q_out_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    w_out_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    q_heads_per_kv: int,
    block_q: int,
):
    device = sparse_utils.device_from_cache_key(device_key)
    q = _placeholder_from_signature(
        q_signature, device=device, dynamic_shape_fill=block_q
    )
    weights = _placeholder_from_signature(
        w_signature, device=device, dynamic_shape_fill=block_q
    )
    q_out = _placeholder_from_signature(
        q_out_signature, device=device, dynamic_shape_fill=PREFILL_BLOCK_QH
    )
    w_out = _placeholder_from_signature(
        w_out_signature, device=device, dynamic_shape_fill=block_q
    )
    f_q = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        q, alignment=TMA_COPY_ALIGN_BYTES, dynamic_shape_dim=0, divisibility=block_q
    )
    f_w = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        weights,
        alignment=TMA_COPY_ALIGN_BYTES,
        dynamic_shape_dim=0,
        divisibility=block_q,
    )
    f_q_out = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        q_out,
        alignment=TMA_COPY_ALIGN_BYTES,
        dynamic_shape_dim=0,
        divisibility=PREFILL_BLOCK_QH,
    )
    f_w_out = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        w_out,
        alignment=TMA_COPY_ALIGN_BYTES,
        dynamic_shape_dim=0,
        divisibility=block_q,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    return _cute_compile(
        _make_pack_prefill_qw_call(q_heads_per_kv, block_q),
        f_q,
        f_w,
        f_q_out,
        f_w_out,
        stream_fake,
        Int32((q_out.numel() + w_out.numel() + PREFILL_PREPARE_THREADS - 1) // PREFILL_PREPARE_THREADS),
        Int32(q_out.numel()),
        Int32(w_out.numel()),
        Int32(q.shape[0]),
    )


@cute.kernel
def _prefill_paged_logits_kernel(
    tma_atom_Q0: cute.CopyAtom,
    tma_tensor_Q0: cute.Tensor,
    tma_atom_Q1: cute.CopyAtom,
    tma_tensor_Q1: cute.Tensor,
    tma_atom_K0: cute.CopyAtom,
    tma_tensor_K0: cute.Tensor,
    tma_atom_K1: cute.CopyAtom,
    tma_tensor_K1: cute.Tensor,
    tma_atom_KFast0: cute.CopyAtom,
    tma_tensor_KFast0: cute.Tensor,
    tma_atom_KFast1: cute.CopyAtom,
    tma_tensor_KFast1: cute.Tensor,
    mQ: cute.Tensor,
    mKCache: cute.Tensor,
    mBlockTable: cute.Tensor,
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
    mCuSeqlensQ: cute.Tensor,
    mCuSeqlensK: cute.Tensor,
    seq_q: Int32,
    seq_k: Int32,
    block_table_cols: Int32,
    batch_size: Int32,
    num_pages: Int32,
    summaries_per_page: Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    q_block_idx = cute.arch.block_idx()[0]
    kv_block_idx = cute.arch.block_idx()[1]
    batch_idx = cute.arch.block_idx()[2]
    grid_q_blocks = cute.arch.grid_dim()[0]
    q_start = mCuSeqlensQ[batch_idx]
    q_end = mCuSeqlensQ[batch_idx + Int32(1)]
    k_end = mCuSeqlensK[batch_idx + Int32(1)]
    valid_seq_q = q_end - q_start
    valid_seq_k = k_end - mCuSeqlensK[batch_idx]
    q_heads_per_kv = PREFILL_COMPILED_Q_HEADS_PER_KV
    q_slots_per_block = PREFILL_COMPILED_Q_SLOTS_PER_BLOCK
    block_q = PREFILL_COMPILED_BLOCK_Q
    q_slot_shift = Int32(PREFILL_COMPILED_Q_SLOT_SHIFT)
    pair_group_mask = Int32(PREFILL_COMPILED_PAIR_GROUP_MASK)
    valid_q_blocks = cute.ceil_div(valid_seq_q, block_q)
    valid_k_blocks = cute.ceil_div(valid_seq_k, PREFILL_BLOCK_KV)

    if (
        (batch_idx < batch_size)
        & (valid_seq_q > Int32(0))
        & (valid_seq_k > Int32(0))
        & (q_block_idx < valid_q_blocks)
        & (kv_block_idx < valid_k_blocks)
    ):
        if warp_idx == 0 and (tidx % Int32(LANES_PER_WARP)) == 0:
            cpasync.prefetch_descriptor(tma_atom_Q0)
            cpasync.prefetch_descriptor(tma_atom_Q1)
            cpasync.prefetch_descriptor(tma_atom_K0)
            cpasync.prefetch_descriptor(tma_atom_K1)
            cpasync.prefetch_descriptor(tma_atom_KFast0)
            cpasync.prefetch_descriptor(tma_atom_KFast1)

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
        mbar_ptr_q_struct = cute.struct.MemRange[
            cutlass.Int64, PREFILL_Q_STAGES * M_BARRIER_PTRS_PER_STAGE
        ]
        mbar_ptr_k_struct = cute.struct.MemRange[
            cutlass.Int64, PREFILL_K_STAGES * M_BARRIER_PTRS_PER_STAGE
        ]

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
        sK_pages0 = cute.make_tensor(
            cute.recast_ptr(storage.sK0.data_ptr(), dtype=Uint8),
            cute.make_layout(
                (PREFILL_K_STAGES * PREFILL_BLOCK_KV * PREFILL_K_SEGMENT,),
                stride=(1,),
            ),
        )
        sK_pages1 = cute.make_tensor(
            cute.recast_ptr(storage.sK1.data_ptr(), dtype=Uint8),
            cute.make_layout(
                (PREFILL_K_STAGES * PREFILL_BLOCK_KV * PREFILL_K_SEGMENT,),
                stride=(1,),
            ),
        )

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, PREFILL_NUM_MATH_WARPS
        )
        pipeline_q = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_q.data_ptr(),
            num_stages=PREFILL_Q_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=PREFILL_BLOCK_QH * HEAD_DIM,
        )
        pipeline_k = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_k.data_ptr(),
            num_stages=PREFILL_K_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=PREFILL_BLOCK_KV * HEAD_DIM,
        )

        gQ0 = cute.local_tile(
            tma_tensor_Q0, (PREFILL_BLOCK_QH, PREFILL_K_SEGMENT), (None, 0)
        )
        gQ1 = cute.local_tile(
            tma_tensor_Q1, (PREFILL_BLOCK_QH, PREFILL_K_SEGMENT), (None, 0)
        )
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
        q_prod = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, PREFILL_Q_STAGES
        )
        q_cons = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, PREFILL_Q_STAGES
        )
        k_prod = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, PREFILL_K_STAGES
        )
        k_cons = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, PREFILL_K_STAGES
        )

        if tidx >= PREFILL_NUM_MATH_THREADS:
            cute.arch.warpgroup_reg_dealloc(PREFILL_NUM_TMA_REGS)
            tma_warp = PREFILL_NUM_MATH_THREADS // LANES_PER_WARP
            is_tma_leader = (
                (warp_idx == Int32(tma_warp))
                & ((tidx % Int32(LANES_PER_WARP)) == 0)
            )
            if is_tma_leader:
                producer_q_block = q_block_idx
                producer_q_global_block = q_start // Int32(block_q) + producer_q_block
                while producer_q_block < valid_q_blocks:
                    pipeline_q.producer_acquire(q_prod)
                    cute.copy(
                        tma_atom_Q0,
                        tQgQ0[None, producer_q_global_block],
                        tQsQ0[None, q_prod.index],
                        tma_bar_ptr=pipeline_q.producer_get_barrier(q_prod),
                    )
                    cute.copy(
                        tma_atom_Q1,
                        tQgQ1[None, producer_q_global_block],
                        tQsQ1[None, q_prod.index],
                        tma_bar_ptr=pipeline_q.producer_get_barrier(q_prod),
                    )
                    q_prod.advance()

                    for kv_block in cutlass.range(valid_k_blocks, unroll=1):
                        pipeline_k.producer_acquire(k_prod)
                        pairs_per_page = summaries_per_page // Int32(PREFILL_TMA_ROWS)
                        pages_per_tile = Int32(PREFILL_BLOCK_KV) // summaries_per_page
                        tile_page_aligned = (
                            (summaries_per_page >= Int32(PREFILL_TMA_ROWS))
                            & (summaries_per_page <= Int32(PREFILL_BLOCK_KV))
                            & (
                                Int32(PREFILL_BLOCK_KV)
                                == pages_per_tile * summaries_per_page
                            )
                        )
                        logical_page_base = (
                            kv_block * Int32(PREFILL_BLOCK_KV)
                        ) // summaries_per_page
                        fast_ok = (
                            tile_page_aligned
                            & (pairs_per_page > Int32(0))
                            & (logical_page_base + pages_per_tile <= block_table_cols)
                        )
                        safe_base_col = cutlass.select_(
                            fast_ok, logical_page_base, Int32(0)
                        )
                        physical_base = mBlockTable[
                            batch_idx * block_table_cols + safe_base_col
                        ]
                        fast_ok = (
                            fast_ok
                            & (physical_base >= Int32(0))
                            & (physical_base + pages_per_tile <= num_pages)
                        )
                        page = Int32(1)
                        while page < pages_per_tile:
                            expected_page = physical_base + page
                            safe_pair_col = cutlass.select_(
                                fast_ok,
                                logical_page_base + page,
                                Int32(0),
                            )
                            actual_page = mBlockTable[
                                batch_idx * block_table_cols + safe_pair_col
                            ]
                            fast_ok = fast_ok & (actual_page == expected_page)
                            page = page + Int32(1)

                        stage_offset = k_prod.index * Int32(
                            PREFILL_BLOCK_KV * PREFILL_K_SEGMENT
                        )
                        if fast_ok:
                            physical_pair_base = physical_base * pairs_per_page
                            gKFast0 = cute.domain_offset(
                                (Int32(0), physical_pair_base),
                                tma_tensor_KFast0,
                            )
                            gKFast1 = cute.domain_offset(
                                (Int32(0), physical_pair_base),
                                tma_tensor_KFast1,
                            )
                            gKFast0 = cute.make_tensor(
                                gKFast0.iterator,
                                cute.make_layout(
                                    (PREFILL_TMA_CHUNK_BYTES, PREFILL_BLOCK_KV // PREFILL_TMA_ROWS),
                                    stride=(tma_tensor_KFast0.stride[0], tma_tensor_KFast0.stride[1]),
                                ),
                            )
                            gKFast1 = cute.make_tensor(
                                gKFast1.iterator,
                                cute.make_layout(
                                    (PREFILL_TMA_CHUNK_BYTES, PREFILL_BLOCK_KV // PREFILL_TMA_ROWS),
                                    stride=(tma_tensor_KFast1.stride[0], tma_tensor_KFast1.stride[1]),
                                ),
                            )
                            sK_fast0 = cute.make_tensor(
                                sparse_utils.elem_pointer_i64_offset(
                                    sK_pages0, (stage_offset,)
                                ),
                                cute.make_layout(
                                    (PREFILL_TMA_CHUNK_BYTES, PREFILL_BLOCK_KV // PREFILL_TMA_ROWS),
                                    stride=(1, PREFILL_TMA_CHUNK_BYTES),
                                ),
                            )
                            sK_fast1 = cute.make_tensor(
                                sparse_utils.elem_pointer_i64_offset(
                                    sK_pages1, (stage_offset,)
                                ),
                                cute.make_layout(
                                    (PREFILL_TMA_CHUNK_BYTES, PREFILL_BLOCK_KV // PREFILL_TMA_ROWS),
                                    stride=(1, PREFILL_TMA_CHUNK_BYTES),
                                ),
                            )
                            load_KFast0, _, _ = copy_utils.tma_get_copy_fn(
                                tma_atom_KFast0,
                                0,
                                cute.make_layout(1),
                                gKFast0,
                                sK_fast0,
                                single_stage=True,
                            )
                            load_KFast1, _, _ = copy_utils.tma_get_copy_fn(
                                tma_atom_KFast1,
                                0,
                                cute.make_layout(1),
                                gKFast1,
                                sK_fast1,
                                single_stage=True,
                            )
                            load_KFast0(tma_bar_ptr=pipeline_k.producer_get_barrier(k_prod))
                            load_KFast1(tma_bar_ptr=pipeline_k.producer_get_barrier(k_prod))
                        else:
                            for pair in cutlass.range_constexpr(
                                PREFILL_BLOCK_KV // PREFILL_TMA_ROWS
                            ):
                                logical_region = (
                                    kv_block * Int32(PREFILL_BLOCK_KV)
                                    + Int32(pair * PREFILL_TMA_ROWS)
                                )
                                logical_page = logical_region // summaries_per_page
                                page_valid = logical_page < block_table_cols
                                safe_col = cutlass.select_(
                                    page_valid, logical_page, Int32(0)
                                )
                                physical_page = mBlockTable[
                                    batch_idx * block_table_cols + safe_col
                                ]
                                page_valid = (
                                    page_valid
                                    & (physical_page >= Int32(0))
                                    & (physical_page < num_pages)
                                )
                                safe_page = cutlass.select_(
                                    page_valid, physical_page, Int32(0)
                                )
                                page_slot = logical_region - logical_page * summaries_per_page
                                pair_in_page = page_slot // Int32(PREFILL_TMA_ROWS)
                                safe_pair = cutlass.select_(page_valid, pair_in_page, Int32(0))
                                dst_offset = stage_offset + Int32(
                                    pair * PREFILL_TMA_CHUNK_BYTES
                                )
                                gK0 = cute.domain_offset(
                                    (Int32(0), Int32(0), safe_pair, safe_page),
                                    tma_tensor_K0,
                                )
                                gK1 = cute.domain_offset(
                                    (Int32(0), Int32(0), safe_pair, safe_page),
                                    tma_tensor_K1,
                                )
                                gK0 = cute.make_tensor(
                                    gK0.iterator,
                                    cute.make_layout(
                                        (PREFILL_TMA_CHUNK_BYTES, 1),
                                        stride=(tma_tensor_K0.stride[0], tma_tensor_K0.stride[1]),
                                    ),
                                )
                                gK1 = cute.make_tensor(
                                    gK1.iterator,
                                    cute.make_layout(
                                        (PREFILL_TMA_CHUNK_BYTES, 1),
                                        stride=(tma_tensor_K1.stride[0], tma_tensor_K1.stride[1]),
                                    ),
                                )
                                sK0_ptr = sparse_utils.elem_pointer_i64_offset(sK_pages0, (dst_offset,))
                                sK1_ptr = sparse_utils.elem_pointer_i64_offset(sK_pages1, (dst_offset,))
                                sK0_chunk = cute.make_tensor(
                                    sK0_ptr,
                                    cute.make_layout((PREFILL_TMA_CHUNK_BYTES, 1), stride=(1, 1)),
                                )
                                sK1_chunk = cute.make_tensor(
                                    sK1_ptr,
                                    cute.make_layout((PREFILL_TMA_CHUNK_BYTES, 1), stride=(1, 1)),
                                )
                                load_K0, _, _ = copy_utils.tma_get_copy_fn(
                                    tma_atom_K0,
                                    0,
                                    cute.make_layout(1),
                                    gK0,
                                    sK0_chunk,
                                    single_stage=True,
                                )
                                load_K1, _, _ = copy_utils.tma_get_copy_fn(
                                    tma_atom_K1,
                                    0,
                                    cute.make_layout(1),
                                    gK1,
                                    sK1_chunk,
                                    single_stage=True,
                                )
                                load_K0(tma_bar_ptr=pipeline_k.producer_get_barrier(k_prod))
                                load_K1(tma_bar_ptr=pipeline_k.producer_get_barrier(k_prod))
                        k_prod.advance()
                    producer_q_block = producer_q_block + grid_q_blocks
                    producer_q_global_block = producer_q_global_block + grid_q_blocks
        else:
            cute.arch.warpgroup_reg_alloc(PREFILL_NUM_MATH_REGS)
            warp_group_idx = cute.arch.make_warp_uniform(
                warp_idx // Int32(WARPS_PER_WARPGROUP)
            )
            lane_idx = tidx % Int32(LANES_PER_WARP)
            warp_group_thread_layout = cute.make_layout(
                PREFILL_NUM_WARPGROUPS, stride=PREFILL_WARPGROUP_STRIDE
            )
            thr_mma_thread = tiled_mma.get_slice(tidx)
            wg_mma = tiled_mma.get_slice(
                warp_group_thread_layout(warp_group_idx)
            )
            cS = cute.make_identity_tensor((PREFILL_BLOCK_KV, PREFILL_BLOCK_QH))
            tScS = thr_mma_thread.partition_C(cS)
            consumer_q_block = q_block_idx
            while consumer_q_block < valid_q_blocks:
                pipeline_q.consumer_wait(q_cons)
                q_stage = q_cons.index
                sQ0_stage = cute.make_tensor(
                    elem_pointer(sQ0, (0, 0, q_stage)), sQ_layout_single_outer_0
                )
                sQ1_stage = cute.make_tensor(
                    elem_pointer(sQ1, (0, 0, q_stage)), sQ_layout_single_outer_1
                )
                warp_offset = warp_idx * Int32(ROWS_PER_WARP_STRIP)
                lane_group = lane_idx >> Int32(2)
                row0 = warp_offset + lane_group
                row1 = row0 + Int32(OUTPUT_ROW_STRIDE)
                lane_mod4 = lane_idx & Int32(Q_LANES_PER_GROUP - 1)
                q_weight_base = q_start + consumer_q_block * Int32(block_q)
                weight_tile = [
                    Float32(0.0)
                    for _ in range(WEIGHT_TILE_VALUES)
                ]
                for q_slot in cutlass.range_constexpr(q_slots_per_block):
                    q_weight_row = (
                        q_weight_base
                        + Int32(q_slot * ROWS_PER_WEIGHT_TILE)
                        + lane_mod4
                    )
                    for head_slot in cutlass.range_constexpr(q_heads_per_kv):
                        weight_tile[q_slot * q_heads_per_kv + head_slot] = _load_f32_2d(
                            mW, q_weight_row, Int32(head_slot)
                        )
                for kv_block in cutlass.range(valid_k_blocks, unroll=1):
                    pipeline_k.consumer_wait(k_cons)
                    k_stage = k_cons.index
                    sK0_stage = cute.make_tensor(
                        elem_pointer(sK0, (0, 0, k_stage)), sK_layout_single_outer_0
                    )
                    sK1_stage = cute.make_tensor(
                        elem_pointer(sK1, (0, 0, k_stage)), sK_layout_single_outer_1
                    )
                    tCrA0 = tiled_mma.make_fragment_A(wg_mma.partition_A(sK0_stage))
                    tCrB0 = tiled_mma.make_fragment_B(wg_mma.partition_B(sQ0_stage))
                    tCrA1 = tiled_mma.make_fragment_A(wg_mma.partition_A(sK1_stage))
                    tCrB1 = tiled_mma.make_fragment_B(wg_mma.partition_B(sQ1_stage))
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

                    sums0 = [Float32(0.0) for _ in range(q_slots_per_block)]
                    sums1 = [Float32(0.0) for _ in range(q_slots_per_block)]
                    for r in cutlass.range(cute.size(tScS), unroll_full=True):
                        m = tScS[r][0]
                        n = tScS[r][1]
                        raw = acc_val[r]
                        value = cutlass.select_(
                            raw > Float32(0.0), raw, Float32(0.0)
                        )
                        virtual_slot = n >> Int32(1)
                        head_pair = n & Int32(HEADS_PER_PAIR - 1)
                        local_q_slot = virtual_slot >> q_slot_shift
                        local_q_lane = virtual_slot & Int32(Q_LANES_PER_GROUP - 1)
                        pair_group = (virtual_slot >> Int32(2)) & pair_group_mask
                        head = (pair_group << Int32(1)) | head_pair
                        # Note(wangbojun/codex): keep the direct combined index here;
                        # the bit-tree shortcut regressed long-sequence prefill numerics.
                        combined_idx = local_q_slot * Int32(q_heads_per_kv) + head
                        weight = _select_tree_32(weight_tile, combined_idx)
                        weighted = value * weight
                        for q_slot in cutlass.range_constexpr(q_slots_per_block):
                            valid_lane = (
                                (local_q_slot == Int32(q_slot))
                                & (local_q_lane == lane_mod4)
                            )
                            sums0[q_slot] = if_generate(
                                (m == row0) & valid_lane,
                                lambda current: current + weighted,
                                lambda current: current,
                                [sums0[q_slot]],
                                [Float32],
                            )
                            sums1[q_slot] = if_generate(
                                (m == row1) & valid_lane,
                                lambda current: current + weighted,
                                lambda current: current,
                                [sums1[q_slot]],
                                [Float32],
                            )

                    kv_base = kv_block * Int32(PREFILL_BLOCK_KV)
                    q_base = q_start + consumer_q_block * Int32(block_q)
                    q_local_base = consumer_q_block * Int32(block_q)
                    for q_slot in cutlass.range_constexpr(q_slots_per_block):
                        q_idx = (
                            q_base + Int32(q_slot * ROWS_PER_WEIGHT_TILE) + lane_mod4
                        )
                        q_local_idx = (
                            q_local_base + Int32(q_slot * ROWS_PER_WEIGHT_TILE) + lane_mod4
                        )
                        key0 = kv_base + row0
                        key1 = kv_base + row1
                        if (q_local_idx < valid_seq_q) & (key0 < valid_seq_k):
                            # Note(wangbojun/codex): 64K x 8K prefill can reach a 2 GiB output buffer,
                            # so the writeback offset must stay 64-bit safe.
                            dst0 = cute.make_tensor(
                                sparse_utils.elem_pointer_i64_offset(mO, (q_idx, key0)),
                                cute.make_layout((1,), stride=(1,)),
                            )
                            dst0[0] = mO.element_type(sums0[q_slot])
                        if (q_local_idx < valid_seq_q) & (key1 < valid_seq_k):
                            dst1 = cute.make_tensor(
                                sparse_utils.elem_pointer_i64_offset(mO, (q_idx, key1)),
                                cute.make_layout((1,), stride=(1,)),
                            )
                            dst1[0] = mO.element_type(sums1[q_slot])
                pipeline_q.consumer_release(q_cons)
                q_cons.advance()
                consumer_q_block = consumer_q_block + grid_q_blocks


def _prefill_paged_logits_call_impl(
    mQ: cute.Tensor,
    mKCache: cute.Tensor,
    mBlockTable: cute.Tensor,
    mW: cute.Tensor,
    mO: cute.Tensor,
    mCuSeqlensQ: cute.Tensor,
    mCuSeqlensK: cute.Tensor,
    stream,
    seq_q: Int32,
    seq_k: Int32,
    block_table_cols: Int32,
    batch_size: Int32,
    num_pages: Int32,
    summaries_per_page: Int32,
    grid_q: Int32,
    q_heads_per_kv: cutlass.Constexpr[int],
    q_pair_groups: cutlass.Constexpr[int],
    q_slots_per_block: cutlass.Constexpr[int],
    block_q: cutlass.Constexpr[int],
):
    mma_tiler_mnk = (PREFILL_BLOCK_KV, PREFILL_BLOCK_QH, PREFILL_K_SEGMENT)
    tiled_mma = sm90_utils.make_trivial_tiled_mma(
        Float8E4M3FN,
        Float8E4M3FN,
        LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        Float32,
        atom_layout_mnk=(PREFILL_BLOCK_KV // PREFILL_MMA_TILE_M, 1, 1),
        tiler_mn=(PREFILL_MMA_TILE_M, PREFILL_BLOCK_QH),
    )
    sQ_layout_fp8_0 = sm90_utils.make_smem_layout_b(
        LayoutEnum.ROW_MAJOR,
        mma_tiler_mnk,
        Float8E4M3FN,
        PREFILL_Q_STAGES,
    )
    sQ_layout_fp8_1 = sm90_utils.make_smem_layout_b(
        LayoutEnum.ROW_MAJOR,
        mma_tiler_mnk,
        Float8E4M3FN,
        PREFILL_Q_STAGES,
    )
    sK_layout_fp8_0 = sm90_utils.make_smem_layout_a(
        LayoutEnum.ROW_MAJOR,
        mma_tiler_mnk,
        Float8E4M3FN,
        PREFILL_K_STAGES,
    )
    sK_layout_fp8_1 = sm90_utils.make_smem_layout_a(
        LayoutEnum.ROW_MAJOR,
        mma_tiler_mnk,
        Float8E4M3FN,
        PREFILL_K_STAGES,
    )
    sQ_layout_single_0 = cute.slice_(sQ_layout_fp8_0, (None, None, 0))
    sQ_layout_single_1 = cute.slice_(sQ_layout_fp8_1, (None, None, 0))
    sK_layout_single_0 = cute.slice_(sK_layout_fp8_0, (None, None, 0))
    sK_layout_single_1 = cute.slice_(sK_layout_fp8_1, (None, None, 0))
    q_segment_layout = cute.make_layout(
        (mQ.shape[0], PREFILL_K_SEGMENT),
        stride=(mQ.stride[0], mQ.stride[1]),
    )
    mQ_segment0 = cute.make_tensor(mQ.iterator, q_segment_layout)
    mQ_segment1_ptr = cute.domain_offset((Int32(0), Int32(PREFILL_K_SEGMENT)), mQ)
    mQ_segment1 = cute.make_tensor(mQ_segment1_ptr.iterator, q_segment_layout)
    tma_atom_Q0, tma_tensor_Q0 = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mQ_segment0,
        sQ_layout_single_0,
        (PREFILL_BLOCK_QH, PREFILL_K_SEGMENT),
    )
    tma_atom_Q1, tma_tensor_Q1 = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mQ_segment1,
        sQ_layout_single_1,
        (PREFILL_BLOCK_QH, PREFILL_K_SEGMENT),
    )
    mKCache_pages = cute.make_tensor(
        cute.recast_ptr(mKCache.iterator, dtype=Uint8),
        cute.make_layout(
            (
                PREFILL_TMA_CHUNK_BYTES,
                HEAD_DIM // PREFILL_K_SEGMENT,
                summaries_per_page // Int32(PREFILL_TMA_ROWS),
                mKCache.shape[0],
            ),
            stride=(
                1,
                PREFILL_TMA_CHUNK_BYTES,
                Int32(PREFILL_TMA_PAGE_STRIDE_BYTES),
                summaries_per_page * Int32(HEAD_DIM),
            ),
        ),
    )
    k_page_layout = cute.make_layout(
        (
            PREFILL_TMA_CHUNK_BYTES,
            1,
            summaries_per_page // Int32(PREFILL_TMA_ROWS),
            mKCache.shape[0],
        ),
        stride=(
            1,
            PREFILL_TMA_CHUNK_BYTES,
            Int32(PREFILL_TMA_PAGE_STRIDE_BYTES),
            summaries_per_page * Int32(HEAD_DIM),
        ),
    )
    mKCache_pages0 = cute.make_tensor(mKCache_pages.iterator, k_page_layout)
    mKCache_pages1_ptr = cute.domain_offset(
        (Int32(0), Int32(1), Int32(0), Int32(0)), mKCache_pages
    )
    mKCache_pages1 = cute.make_tensor(mKCache_pages1_ptr.iterator, k_page_layout)
    sK_page_layout = cute.make_layout(
        (PREFILL_TMA_CHUNK_BYTES, 1), stride=(1, 1)
    )
    tma_atom_K0, tma_tensor_K0 = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mKCache_pages0,
        sK_page_layout,
        (PREFILL_TMA_CHUNK_BYTES, 1),
    )
    tma_atom_K1, tma_tensor_K1 = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mKCache_pages1,
        sK_page_layout,
        (PREFILL_TMA_CHUNK_BYTES, 1),
    )
    sK_fast_layout = cute.make_layout(
        (PREFILL_TMA_CHUNK_BYTES, PREFILL_BLOCK_KV // PREFILL_TMA_ROWS),
        stride=(1, PREFILL_TMA_CHUNK_BYTES),
    )
    fast_pairs_per_page = summaries_per_page // Int32(PREFILL_TMA_ROWS)
    k_fast_source_layout = cute.make_layout(
        (PREFILL_TMA_CHUNK_BYTES, mKCache.shape[0] * fast_pairs_per_page),
        stride=(1, Int32(PREFILL_TMA_PAGE_STRIDE_BYTES)),
    )
    mKCache_fast0 = cute.make_tensor(mKCache_pages0.iterator, k_fast_source_layout)
    mKCache_fast1 = cute.make_tensor(mKCache_pages1.iterator, k_fast_source_layout)
    tma_atom_KFast0, tma_tensor_KFast0 = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mKCache_fast0,
        sK_fast_layout,
        (PREFILL_TMA_CHUNK_BYTES, PREFILL_BLOCK_KV // PREFILL_TMA_ROWS),
    )
    tma_atom_KFast1, tma_tensor_KFast1 = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mKCache_fast1,
        sK_fast_layout,
        (PREFILL_TMA_CHUNK_BYTES, PREFILL_BLOCK_KV // PREFILL_TMA_ROWS),
    )
    _prefill_paged_logits_kernel(
        tma_atom_Q0,
        tma_tensor_Q0,
        tma_atom_Q1,
        tma_tensor_Q1,
        tma_atom_K0,
        tma_tensor_K0,
        tma_atom_K1,
        tma_tensor_K1,
        tma_atom_KFast0,
        tma_tensor_KFast0,
        tma_atom_KFast1,
        tma_tensor_KFast1,
        mQ,
        mKCache,
        mBlockTable,
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
        mCuSeqlensQ,
        mCuSeqlensK,
        seq_q,
        seq_k,
        block_table_cols,
        batch_size,
        num_pages,
        summaries_per_page,
    ).launch(
        grid=(grid_q, 1, batch_size),
        block=[PREFILL_NUM_THREADS, 1, 1],
        stream=stream,
    )


def _make_prefill_paged_logits_call(q_heads_per_kv: int, block_q: int):
    q_pair_groups = q_heads_per_kv // HEADS_PER_PAIR
    q_slots_per_block = block_q // ROWS_PER_WEIGHT_TILE
    q_slot_shift = q_pair_groups.bit_length() + 1
    pair_group_mask = q_pair_groups - 1
    global PREFILL_COMPILED_Q_HEADS_PER_KV
    global PREFILL_COMPILED_Q_PAIR_GROUPS
    global PREFILL_COMPILED_Q_SLOTS_PER_BLOCK
    global PREFILL_COMPILED_BLOCK_Q
    global PREFILL_COMPILED_Q_SLOT_SHIFT
    global PREFILL_COMPILED_PAIR_GROUP_MASK
    PREFILL_COMPILED_Q_HEADS_PER_KV = q_heads_per_kv
    PREFILL_COMPILED_Q_PAIR_GROUPS = q_pair_groups
    PREFILL_COMPILED_Q_SLOTS_PER_BLOCK = q_slots_per_block
    PREFILL_COMPILED_BLOCK_Q = block_q
    PREFILL_COMPILED_Q_SLOT_SHIFT = q_slot_shift
    PREFILL_COMPILED_PAIR_GROUP_MASK = pair_group_mask

    @cute.jit
    def _call(
        mQ: cute.Tensor,
        mKCache: cute.Tensor,
        mBlockTable: cute.Tensor,
        mW: cute.Tensor,
        mO: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        stream,
        seq_q: Int32,
        seq_k: Int32,
        block_table_cols: Int32,
        batch_size: Int32,
        num_pages: Int32,
        summaries_per_page: Int32,
        grid_q: Int32,
    ):
        _prefill_paged_logits_call_impl(
            mQ,
            mKCache,
            mBlockTable,
            mW,
            mO,
            mCuSeqlensQ,
            mCuSeqlensK,
            stream,
            seq_q,
            seq_k,
            block_table_cols,
            batch_size,
            num_pages,
            summaries_per_page,
            grid_q,
            q_heads_per_kv,
            q_pair_groups,
            q_slots_per_block,
            block_q,
        )

    return _call


@functools.cache
def _compile_prefill_paged_logits_for_signature(
    device_key: tuple[str, Optional[int]],
    q_runtime_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    k_cache_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    block_table_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    weights_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    out_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    q_heads_per_kv: int,
    block_q: int,
):
    device = sparse_utils.device_from_cache_key(device_key)
    q_runtime = _placeholder_from_signature(
        q_runtime_signature, device=device, dynamic_shape_fill=PREFILL_BLOCK_QH
    )
    k_cache = _placeholder_from_signature(
        k_cache_signature,
        device=device,
        dynamic_shape_fill=PREFILL_TMA_ROWS,
        dynamic_stride_fill=PREFILL_TMA_PAGE_STRIDE_BYTES,
    )
    block_table_flat = _placeholder_from_signature(
        block_table_signature,
        device=device,
        dynamic_shape_fill=COMPILE_PLACEHOLDER_DIM,
    )
    weights = _placeholder_from_signature(
        weights_signature, device=device, dynamic_shape_fill=block_q
    )
    out = _placeholder_from_signature(
        out_signature, device=device, dynamic_shape_fill=PREFILL_BLOCK_KV
    )
    f_q = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        q_runtime,
        alignment=TMA_COPY_ALIGN_BYTES,
        dynamic_shape_dim=0,
        divisibility=PREFILL_BLOCK_QH,
    )
    f_k = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        k_cache,
        alignment=TMA_COPY_ALIGN_BYTES,
        dynamic_shape_dims=(0, 1),
        dynamic_stride_dims=(0,),
    )
    f_block_table = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        block_table_flat,
        alignment=TMA_COPY_ALIGN_BYTES,
        dynamic_shape_dim=0,
    )
    f_weights = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        weights,
        alignment=TMA_COPY_ALIGN_BYTES,
        dynamic_shape_dim=0,
        divisibility=block_q,
    )
    f_out = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        out,
        alignment=TMA_COPY_ALIGN_BYTES,
        dynamic_shape_dims=(0, 1),
        dynamic_stride_dims=(0,),
        divisibility=block_q,
    )
    cu_q = torch.empty((COMPILE_PLACEHOLDER_DIM + 1,), device=device, dtype=torch.int32)
    cu_k = torch.empty((COMPILE_PLACEHOLDER_DIM + 1,), device=device, dtype=torch.int32)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return _cute_compile(
        _make_prefill_paged_logits_call(q_heads_per_kv, block_q),
        f_q,
        f_k,
        f_block_table,
        f_weights,
        f_out,
        cu_q,
        cu_k,
        stream_fake,
        Int32(block_q),
        Int32(PREFILL_BLOCK_KV),
        Int32(COMPILE_PLACEHOLDER_DIM),
        Int32(COMPILE_PLACEHOLDER_DIM),
        Int32(COMPILE_PLACEHOLDER_DIM),
        Int32(PREFILL_TMA_ROWS),
        Int32(COMPILE_PLACEHOLDER_DIM),
    )


def _validate_inputs(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    summary: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_runtime: torch.Tensor,
    kernel_weights: torch.Tensor,
    out: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    q_heads_per_kv = int(index_q.shape[2])
    block_q = _prefill_block_q_for_heads(q_heads_per_kv)
    if tuple(index_q.shape[1:]) != (SUMMARY_KV_HEADS, q_heads_per_kv, HEAD_DIM):
        raise ValueError(
            f"paged prefill logits index_q must be [Q,{SUMMARY_KV_HEADS},H,{HEAD_DIM}], got "
            f"{tuple(index_q.shape)}"
        )
    query_rows = int(index_q.shape[0])
    max_regions = int(out.shape[1])
    if query_rows <= 0 or query_rows % block_q:
        raise ValueError(
            f"paged prefill logits Q capacity must be a positive multiple of "
            f"{block_q}, got {query_rows}"
        )
    if max_regions <= 0 or max_regions % PREFILL_BLOCK_KV:
        raise ValueError(
            f"paged prefill logits region capacity must be a positive multiple of "
            f"{PREFILL_BLOCK_KV}, got {max_regions}"
        )
    if tuple(weights.shape) != (query_rows, SUMMARY_KV_HEADS, q_heads_per_kv):
        raise ValueError(
            f"paged prefill logits weights must be [Q,{SUMMARY_KV_HEADS},H], got "
            f"{tuple(weights.shape)}"
        )
    if (
        summary.dim() != 4
        or int(summary.shape[1]) <= 0
        or int(summary.shape[1]) % PREFILL_TMA_ROWS
        or tuple(summary.shape[2:]) != (SUMMARY_KV_HEADS, HEAD_DIM)
    ):
        raise ValueError(
            f"paged prefill logits summary must be [pages,rows_multiple_of_{PREFILL_TMA_ROWS},{SUMMARY_KV_HEADS},{HEAD_DIM}], "
            f"got {tuple(summary.shape)}"
        )
    batch_size = int(block_table.shape[0]) if block_table.dim() == 2 else 0
    if (
        block_table.dtype != torch.int32
        or block_table.dim() != 2
        or batch_size <= 0
        or int(block_table.shape[1]) <= 0
        or not block_table.is_contiguous()
    ):
        raise ValueError(
            "paged prefill logits block_table must be contiguous int32 "
            "[B,pages] with B > 0 and pages > 0"
        )
    # ``max_regions`` is the graph/JIT-stable score capacity. Step4 rounds it
    # up to a selector bucket (and may raise it because of top-k), so it is not
    # the live logical KV width and must not be used to size ``block_table``.
    # The live width comes from ``cu_seqlens_k`` inside the kernel; individual
    # block-table accesses are guarded by ``block_table_cols``. The scheduler
    # owns the separate invariant that every live sequence fits its table.
    if tuple(q_runtime.shape) != (query_rows * q_heads_per_kv, HEAD_DIM):
        raise ValueError(
            f"paged prefill logits q_runtime must be [Q*H,{HEAD_DIM}], got "
            f"{tuple(q_runtime.shape)}"
        )
    if tuple(kernel_weights.shape) != (query_rows, q_heads_per_kv):
        raise ValueError(
            "paged prefill logits kernel_weights must be [Q,H], got "
            f"{tuple(kernel_weights.shape)}"
        )
    if tuple(out.shape) != (query_rows, max_regions):
        raise ValueError("paged prefill logits out must be [Q,padded_regions]")
    for name, tensor, dtype in (
        ("index_q", index_q, (torch.float16, torch.bfloat16, torch.float32)),
        ("summary", summary, (torch.float8_e4m3fn, torch.uint8)),
        ("q_runtime", q_runtime, (torch.float8_e4m3fn,)),
        ("kernel_weights", kernel_weights, (torch.float32,)),
        ("out", out, (torch.float32,)),
    ):
        if tensor.dtype not in dtype or not tensor.is_contiguous():
            raise ValueError(
                f"paged prefill logits {name} has unsupported dtype/layout: "
                f"dtype={tensor.dtype}, contiguous={tensor.is_contiguous()}"
            )
    for name, tensor in (
        ("weights", weights),
        ("block_table", block_table),
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_k", cu_seqlens_k),
    ):
        if tensor.device != index_q.device or not tensor.is_contiguous():
            raise ValueError(f"paged prefill logits {name} must be contiguous on index_q.device")
    if weights.dtype != torch.float32:
        raise ValueError(f"paged prefill logits weights must be float32, got {weights.dtype}")
    if (
        tuple(cu_seqlens_q.shape) != (batch_size + 1,)
        or tuple(cu_seqlens_k.shape) != (batch_size + 1,)
        or cu_seqlens_q.dtype != torch.int32
        or cu_seqlens_k.dtype != torch.int32
    ):
        raise ValueError(
            "paged prefill logits cu_seqlens_q/k must be contiguous int32 [B+1] "
            f"for B={batch_size}"
        )
    return query_rows, max_regions, q_heads_per_kv, block_q, batch_size


def _prefill_paged_weighted_relu_logits_impl(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    summary: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_runtime: torch.Tensor,
    kernel_weights: torch.Tensor,
    out: torch.Tensor,
) -> None:
    query_rows, max_regions, q_heads_per_kv, block_q, batch_size = _validate_inputs(
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
    if index_q.device.type != "cuda":
        raise RuntimeError("paged prefill WGMMA logits requires CUDA tensors")
    q_input = index_q
    k_input = (
        summary.view(torch.float8_e4m3fn)
        if summary.dtype == torch.uint8 else summary
    )
    pack = _compile_pack_prefill_qw_for_signature(
        sparse_utils.device_cache_key(index_q.device),
        _tensor_signature_dynamic(q_input, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(weights, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(q_runtime.view(torch.uint8), dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(kernel_weights, dynamic_shape_dims=(0,)),
        q_heads_per_kv,
        block_q,
    )
    total_q = q_runtime.numel()
    total_w = kernel_weights.numel()
    pack(
        q_input,
        weights,
        q_runtime.view(torch.uint8),
        kernel_weights,
        Int32((total_q + total_w + PREFILL_PREPARE_THREADS - 1) // PREFILL_PREPARE_THREADS),
        Int32(total_q),
        Int32(total_w),
        Int32(query_rows),
    )
    compiled = _compile_prefill_paged_logits_for_signature(
        sparse_utils.device_cache_key(index_q.device),
        _tensor_signature_dynamic(q_runtime, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(
            k_input,
            dynamic_shape_dims=(0, 1),
            dynamic_stride_dims=(0,),
        ),
        _tensor_signature_dynamic(block_table.reshape(-1), dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(kernel_weights, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(
            out, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)
        ),
        q_heads_per_kv,
        block_q,
    )
    sm_count = _sm_count(sparse_utils.device_cache_key(index_q.device))
    grid_q = min(
        query_rows // block_q,
        max(1, int(sm_count) * PREFILL_PERSISTENT_CTAS_PER_SM),
    )
    compiled(
        q_runtime,
        k_input,
        block_table.reshape(-1),
        kernel_weights,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        Int32(query_rows),
        Int32(max_regions),
        Int32(block_table.shape[1]),
        Int32(batch_size),
        Int32(summary.shape[0]),
        Int32(summary.shape[1]),
        Int32(grid_q),
    )


@torch.library.custom_op(
    "optimus_cutedsl::prefill_paged_weighted_relu_logits_sm90_steptron_gqa_out",
    mutates_args=("q_runtime", "kernel_weights", "out"),
    device_types="cuda",
)
def _prefill_paged_weighted_relu_logits_out(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    summary: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_runtime: torch.Tensor,
    kernel_weights: torch.Tensor,
    out: torch.Tensor,
) -> None:
    _prefill_paged_weighted_relu_logits_impl(
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


def prefill_paged_weighted_relu_logits_sm90_steptron_gqa(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    summary: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    q_runtime: Optional[torch.Tensor] = None,
    kernel_weights: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if q_runtime is None or kernel_weights is None or out is None:
        raise ValueError(
            "paged prefill WGMMA logits requires caller-owned q_runtime, "
            "kernel_weights, and out workspaces before CUDA Graph capture"
        )
    _prefill_paged_weighted_relu_logits_out(
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
    return out


__all__ = ["prefill_paged_weighted_relu_logits_sm90_steptron_gqa"]
