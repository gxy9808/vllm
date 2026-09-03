# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from typing import Optional

import torch
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync
from cutlass import Float32, Float8E4M3FN, Int32, Uint8
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cutlass_dsl import if_generate

from vllm.models.step4.nvidia.ops.cute_dsl.cutedsl_compile_cache import cached_compile_function
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.kernel_utils import cute_utils as sparse_utils
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.kernel_utils import hopper_helpers as hop_helpers
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.kernel_utils.fp_compat import cvt_f32_to_e4m3
from vllm.models.step4.nvidia.ops.cute_dsl.utils import elem_pointer


def _cute_compile(func, *args):
    return cute.compile(func, *args, options="--enable-tvm-ffi --opt-level 2")


def _load_f32_global(tensor: cute.Tensor, coord: tuple[Int32, Int32, Int32]) -> Float32:
    src = cute.make_tensor(
        elem_pointer(tensor, coord),
        cute.make_layout((1,), stride=(1,)),
    )
    return Float32(src[0])


Q_HEADS_PER_KV = 8
SUPPORTED_Q_HEADS_PER_KV = (8, 16)
HEAD_DIM = 64

GQA_BLOCK_Q = 16
GQA_BLOCK_Q_Q16 = 8
Q_HEADS_PER_KV_Q16 = 16
GQA_BLOCK_QH = 128
GQA_BLOCK_KV = 256
GQA_NUM_Q_STAGES = 3
GQA_NUM_KV_STAGES = 3
GQA_NUM_MATH_THREADS = 512
GQA_NUM_TMA_THREADS = 128
GQA_NUM_THREADS = GQA_NUM_MATH_THREADS + GQA_NUM_TMA_THREADS
GQA_NUM_WARPS = GQA_NUM_MATH_THREADS // 32
GQA_NUM_TMA_REGS = 32
GQA_NUM_MATH_REGS = 112
GQA_PREPARE_THREADS = 256


def _gqa_block_q_for_heads(q_heads_per_kv: int) -> int:
    if int(q_heads_per_kv) == Q_HEADS_PER_KV:
        return GQA_BLOCK_Q
    if int(q_heads_per_kv) == Q_HEADS_PER_KV_Q16:
        return GQA_BLOCK_Q_Q16
    raise ValueError(
        "GQA logits supports q_heads_per_kv in "
        f"{SUPPORTED_Q_HEADS_PER_KV}, got {int(q_heads_per_kv)}"
    )



def _tensor_signature(x: torch.Tensor) -> tuple[torch.dtype, tuple[int, ...], tuple[int, ...]]:
    return x.dtype, tuple(int(v) for v in x.shape), tuple(int(v) for v in x.stride())


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
def _prepare_gqa_logits_inputs_kernel(
    mQIn: cute.Tensor,
    mKIn: cute.Tensor,
    mWIn: cute.Tensor,
    mQOut: cute.Tensor,
    mKOut: cute.Tensor,
    mWOut: cute.Tensor,
    total_q_elems: Int32,
    total_k_elems: Int32,
    total_w_elems: Int32,
    seq_q: Int32,
    seq_k: Int32,
    kv_heads: Int32,
):
    tid = cute.arch.block_idx()[0] * Int32(GQA_PREPARE_THREADS) + cute.arch.thread_idx()[0]
    total = total_q_elems + total_k_elems + total_w_elems
    if tid < total:
        num_q_blocks = seq_q // Int32(GQA_BLOCK_Q)
        if tid < total_q_elems:
            dim = tid % Int32(HEAD_DIM)
            row = tid // Int32(HEAD_DIM)
            local_row = row % Int32(GQA_BLOCK_QH)
            block_linear = row // Int32(GQA_BLOCK_QH)
            block_q_idx = block_linear % num_q_blocks
            kv_head_idx = block_linear // num_q_blocks

            physical_q_slot = local_row // Int32(Q_HEADS_PER_KV)
            physical_head = local_row - physical_q_slot * Int32(Q_HEADS_PER_KV)
            q_slots_per_lane = Int32(GQA_BLOCK_Q // 4)
            slot_div = physical_q_slot - (physical_q_slot // q_slots_per_lane) * q_slots_per_lane
            head_pair = physical_q_slot // q_slots_per_lane
            lane = physical_head // Int32(2)
            bit = physical_head - lane * Int32(2)
            q_slot = slot_div * Int32(4) + lane
            head = head_pair * Int32(2) + bit
            global_q = block_q_idx * Int32(GQA_BLOCK_Q) + q_slot
            mQOut[row, dim] = cvt_f32_to_e4m3(
                Float32(mQIn[global_q, kv_head_idx, head, dim])
            ).to(Uint8)
        elif tid < total_q_elems + total_k_elems:
            k_tid = tid - total_q_elems
            dim = k_tid % Int32(HEAD_DIM)
            row = k_tid // Int32(HEAD_DIM)
            k_pos = row % seq_k
            kv_head_idx = row // seq_k
            mKOut[row, dim] = cvt_f32_to_e4m3(
                Float32(mKIn[k_pos, kv_head_idx, dim])
            ).to(Uint8)
        else:
            w_tid = tid - total_q_elems - total_k_elems
            head = w_tid % Int32(Q_HEADS_PER_KV)
            row = w_tid // Int32(Q_HEADS_PER_KV)
            q_pos = row % seq_q
            kv_head_idx = row // seq_q
            if kv_head_idx < kv_heads:
                mWOut[kv_head_idx, q_pos, head] = mWOut.element_type(
                    mWIn[q_pos, kv_head_idx, head]
                )


@cute.jit
def _prepare_gqa_logits_inputs_call(
    mQIn: cute.Tensor,
    mKIn: cute.Tensor,
    mWIn: cute.Tensor,
    mQOut: cute.Tensor,
    mKOut: cute.Tensor,
    mWOut: cute.Tensor,
    stream,
    grid_x: Int32,
    total_q_elems: Int32,
    total_k_elems: Int32,
    total_w_elems: Int32,
    seq_q: Int32,
    seq_k: Int32,
    kv_heads: Int32,
):
    _prepare_gqa_logits_inputs_kernel(
        mQIn,
        mKIn,
        mWIn,
        mQOut,
        mKOut,
        mWOut,
        total_q_elems,
        total_k_elems,
        total_w_elems,
        seq_q,
        seq_k,
        kv_heads,
    ).launch(grid=[grid_x, 1, 1], block=[GQA_PREPARE_THREADS, 1, 1], stream=stream)


@cached_compile_function
def _compile_prepare_gqa_logits_inputs_for_signature(
    device_key: tuple[str, Optional[int]],
    q_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    k_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    weights_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    q_runtime_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    k_runtime_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    w_mat_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
):
    device = sparse_utils.device_from_cache_key(device_key)
    q = _placeholder_from_signature(
        q_signature, device=device, dynamic_shape_fill=GQA_BLOCK_Q)
    k = _placeholder_from_signature(
        k_signature, device=device, dynamic_shape_fill=GQA_BLOCK_KV)
    weights = _placeholder_from_signature(
        weights_signature, device=device, dynamic_shape_fill=GQA_BLOCK_Q)
    q_runtime = _placeholder_from_signature(
        q_runtime_signature,
        device=device,
        dynamic_shape_fill=GQA_BLOCK_Q * Q_HEADS_PER_KV,
    )
    k_runtime = _placeholder_from_signature(
        k_runtime_signature, device=device, dynamic_shape_fill=GQA_BLOCK_KV)
    w_mat = _placeholder_from_signature(
        w_mat_signature, device=device, dynamic_shape_fill=GQA_BLOCK_Q)
    fQ = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        q,
        alignment=16,
        dynamic_shape_dim=0,
        divisibility=GQA_BLOCK_Q,
    )
    fK = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        k,
        alignment=16,
        dynamic_shape_dim=0,
        divisibility=GQA_BLOCK_KV,
    )
    fW = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        weights,
        alignment=16,
        dynamic_shape_dim=0,
        divisibility=GQA_BLOCK_Q,
    )
    fQOut = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        q_runtime,
        alignment=16,
        dynamic_shape_dim=0,
        divisibility=GQA_BLOCK_Q * Q_HEADS_PER_KV,
    )
    fKOut = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        k_runtime,
        alignment=16,
        dynamic_shape_dim=0,
        divisibility=GQA_BLOCK_KV,
    )
    fWOut = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        w_mat,
        alignment=16,
        dynamic_shape_dim=1,
        divisibility=GQA_BLOCK_Q,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return _cute_compile(
        _prepare_gqa_logits_inputs_call,
        fQ,
        fK,
        fW,
        fQOut,
        fKOut,
        fWOut,
        stream_fake,
        Int32((q_runtime.numel() + k_runtime.numel() + w_mat.numel() + GQA_PREPARE_THREADS - 1) // GQA_PREPARE_THREADS),
        Int32(q_runtime.numel()),
        Int32(k_runtime.numel()),
        Int32(w_mat.numel()),
        Int32(q.shape[0]),
        Int32(k.shape[0]),
        Int32(q.shape[1]),
    )


def _compile_prepare_gqa_logits_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    q_runtime: torch.Tensor,
    k_runtime: torch.Tensor,
    w_mat: torch.Tensor,
):
    return _compile_prepare_gqa_logits_inputs_for_signature(
        sparse_utils.device_cache_key(q.device),
        _tensor_signature_dynamic(q, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(k, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(weights, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(q_runtime, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(k_runtime, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(w_mat, dynamic_shape_dims=(1,)),
    )


@cute.kernel
def _prepare_gqa_logits_inputs_kernel_q16(
    mQIn: cute.Tensor,
    mKIn: cute.Tensor,
    mWIn: cute.Tensor,
    mQOut: cute.Tensor,
    mKOut: cute.Tensor,
    mWOut: cute.Tensor,
    total_q_elems: Int32,
    total_k_elems: Int32,
    total_w_elems: Int32,
    seq_q: Int32,
    seq_k: Int32,
    kv_heads: Int32,
):
    tid = cute.arch.block_idx()[0] * Int32(GQA_PREPARE_THREADS) + cute.arch.thread_idx()[0]
    total = total_q_elems + total_k_elems + total_w_elems
    if tid < total:
        num_q_blocks = seq_q // Int32(GQA_BLOCK_Q_Q16)
        if tid < total_q_elems:
            dim = tid % Int32(HEAD_DIM)
            row = tid // Int32(HEAD_DIM)
            local_row = row % Int32(GQA_BLOCK_QH)
            block_linear = row // Int32(GQA_BLOCK_QH)
            block_q_idx = block_linear % num_q_blocks
            kv_head_idx = block_linear // num_q_blocks

            physical_q_slot = local_row // Int32(Q_HEADS_PER_KV)
            physical_head = local_row - physical_q_slot * Int32(Q_HEADS_PER_KV)
            q_slots_per_lane = Int32(GQA_BLOCK_Q_Q16 // 4)
            slot_div = physical_q_slot - (physical_q_slot // q_slots_per_lane) * q_slots_per_lane
            head_pair = physical_q_slot // q_slots_per_lane
            lane = physical_head // Int32(2)
            bit = physical_head - lane * Int32(2)
            q_slot = slot_div * Int32(4) + lane
            head = head_pair * Int32(2) + bit
            global_q = block_q_idx * Int32(GQA_BLOCK_Q_Q16) + q_slot
            mQOut[row, dim] = cvt_f32_to_e4m3(
                Float32(mQIn[global_q, kv_head_idx, head, dim])
            ).to(Uint8)
        elif tid < total_q_elems + total_k_elems:
            k_tid = tid - total_q_elems
            dim = k_tid % Int32(HEAD_DIM)
            row = k_tid // Int32(HEAD_DIM)
            k_pos = row % seq_k
            kv_head_idx = row // seq_k
            mKOut[row, dim] = cvt_f32_to_e4m3(
                Float32(mKIn[k_pos, kv_head_idx, dim])
            ).to(Uint8)
        else:
            w_tid = tid - total_q_elems - total_k_elems
            head = w_tid % Int32(Q_HEADS_PER_KV_Q16)
            row = w_tid // Int32(Q_HEADS_PER_KV_Q16)
            q_pos = row % seq_q
            kv_head_idx = row // seq_q
            if kv_head_idx < kv_heads:
                mWOut[kv_head_idx, q_pos, head] = mWOut.element_type(
                    mWIn[q_pos, kv_head_idx, head]
                )


@cute.jit
def _prepare_gqa_logits_inputs_call_q16(
    mQIn: cute.Tensor,
    mKIn: cute.Tensor,
    mWIn: cute.Tensor,
    mQOut: cute.Tensor,
    mKOut: cute.Tensor,
    mWOut: cute.Tensor,
    stream,
    grid_x: Int32,
    total_q_elems: Int32,
    total_k_elems: Int32,
    total_w_elems: Int32,
    seq_q: Int32,
    seq_k: Int32,
    kv_heads: Int32,
):
    _prepare_gqa_logits_inputs_kernel_q16(
        mQIn,
        mKIn,
        mWIn,
        mQOut,
        mKOut,
        mWOut,
        total_q_elems,
        total_k_elems,
        total_w_elems,
        seq_q,
        seq_k,
        kv_heads,
    ).launch(grid=[grid_x, 1, 1], block=[GQA_PREPARE_THREADS, 1, 1], stream=stream)


@cached_compile_function
def _compile_prepare_gqa_logits_inputs_q16_for_signature(
    device_key: tuple[str, Optional[int]],
    q_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    k_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    weights_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    q_runtime_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    k_runtime_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    w_mat_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
):
    device = sparse_utils.device_from_cache_key(device_key)
    q = _placeholder_from_signature(
        q_signature, device=device, dynamic_shape_fill=GQA_BLOCK_Q_Q16)
    k = _placeholder_from_signature(
        k_signature, device=device, dynamic_shape_fill=GQA_BLOCK_KV)
    weights = _placeholder_from_signature(
        weights_signature, device=device, dynamic_shape_fill=GQA_BLOCK_Q_Q16)
    q_runtime = _placeholder_from_signature(
        q_runtime_signature,
        device=device,
        dynamic_shape_fill=GQA_BLOCK_QH,
    )
    k_runtime = _placeholder_from_signature(
        k_runtime_signature, device=device, dynamic_shape_fill=GQA_BLOCK_KV)
    w_mat = _placeholder_from_signature(
        w_mat_signature, device=device, dynamic_shape_fill=GQA_BLOCK_Q_Q16)
    fQ = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        q,
        alignment=16,
        dynamic_shape_dim=0,
        divisibility=GQA_BLOCK_Q_Q16,
    )
    fK = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        k,
        alignment=16,
        dynamic_shape_dim=0,
        divisibility=GQA_BLOCK_KV,
    )
    fW = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        weights,
        alignment=16,
        dynamic_shape_dim=0,
        divisibility=GQA_BLOCK_Q_Q16,
    )
    fQOut = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        q_runtime,
        alignment=16,
        dynamic_shape_dim=0,
        divisibility=GQA_BLOCK_QH,
    )
    fKOut = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        k_runtime,
        alignment=16,
        dynamic_shape_dim=0,
        divisibility=GQA_BLOCK_KV,
    )
    fWOut = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        w_mat,
        alignment=16,
        dynamic_shape_dim=1,
        divisibility=GQA_BLOCK_Q_Q16,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return _cute_compile(
        _prepare_gqa_logits_inputs_call_q16,
        fQ,
        fK,
        fW,
        fQOut,
        fKOut,
        fWOut,
        stream_fake,
        Int32((q_runtime.numel() + k_runtime.numel() + w_mat.numel() + GQA_PREPARE_THREADS - 1) // GQA_PREPARE_THREADS),
        Int32(q_runtime.numel()),
        Int32(k_runtime.numel()),
        Int32(w_mat.numel()),
        Int32(q.shape[0]),
        Int32(k.shape[0]),
        Int32(q.shape[1]),
    )


def _compile_prepare_gqa_logits_inputs_q16(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    q_runtime: torch.Tensor,
    k_runtime: torch.Tensor,
    w_mat: torch.Tensor,
):
    return _compile_prepare_gqa_logits_inputs_q16_for_signature(
        sparse_utils.device_cache_key(q.device),
        _tensor_signature_dynamic(q, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(k, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(weights, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(q_runtime, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(k_runtime, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(w_mat, dynamic_shape_dims=(1,)),
    )


@cute.kernel
def fp8_gqa_logits_fast_sm90_kernel(
    tma_atom_Q: cute.CopyAtom,
    tma_tensor_Q: cute.Tensor,
    tma_atom_K: cute.CopyAtom,
    tma_tensor_K: cute.Tensor,
    tiled_mma: cute.TiledMma,
    sQ_layout_fp8: cute.ComposedLayout,
    sK_layout_fp8: cute.ComposedLayout,
    sQ_layout_single_outer: cute.Layout,
    sK_layout_single_outer: cute.Layout,
    mW: cute.Tensor,
    mO: cute.Tensor,
    seq_q: Int32,
    seq_k: Int32,
    kv_heads: Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    q_block_idx = cute.arch.block_idx()[0]
    kv_head_idx = cute.arch.block_idx()[1]
    grid_q_blocks = cute.arch.grid_dim()[0]

    num_q_blocks_per_kv = cute.ceil_div(seq_q, GQA_BLOCK_Q)
    if q_block_idx < num_q_blocks_per_kv and kv_head_idx < kv_heads:
        # Note(wangbojun/codex): This kernel keeps only the current direct global-store
        # logits path; detached TMA store and approximate top-k experiments were removed.
        tma_prefetch_warp = GQA_NUM_MATH_THREADS // 32
        # Prefetch TMA descriptors.
        if warp_idx == 0 and (tidx % 32) == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)

        smem = cutlass.utils.SmemAllocator()
        sQ_struct = cute.struct.Align[
            cute.struct.MemRange[Float8E4M3FN, cute.cosize(sQ_layout_fp8)], 1024
        ]
        sK_struct = cute.struct.Align[
            cute.struct.MemRange[Float8E4M3FN, cute.cosize(sK_layout_fp8)], 1024
        ]
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, GQA_NUM_Q_STAGES * 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, GQA_NUM_KV_STAGES * 2]

        @cute.struct
        class SharedStorage:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            sQ: sQ_struct
            sK: sK_struct

        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout_fp8.outer, swizzle=sQ_layout_fp8.inner)
        sK = storage.sK.get_tensor(sK_layout_fp8.outer, swizzle=sK_layout_fp8.inner)

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, GQA_NUM_WARPS)
        tma_q_bytes = GQA_BLOCK_QH * HEAD_DIM
        tma_k_bytes = GQA_BLOCK_KV * HEAD_DIM
        pipeline_q = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_Q.data_ptr(),
            num_stages=GQA_NUM_Q_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tma_q_bytes,
        )
        pipeline_k = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=GQA_NUM_KV_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tma_k_bytes,
        )

        gQ = cute.local_tile(tma_tensor_Q, (GQA_BLOCK_QH, HEAD_DIM), (None, 0))
        gK = cute.local_tile(tma_tensor_K, (GQA_BLOCK_KV, HEAD_DIM), (None, 0))
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_atom_Q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, cute.rank(sQ) - 1),
            cute.group_modes(gQ, 0, cute.rank(gQ) - 1),
        )
        tKsK, tKgK = cpasync.tma_partition(
            tma_atom_K,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, cute.rank(sK) - 1),
            cute.group_modes(gK, 0, cute.rank(gK) - 1),
        )
        q_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, GQA_NUM_Q_STAGES)
        q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, GQA_NUM_Q_STAGES)
        k_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, GQA_NUM_KV_STAGES)
        k_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, GQA_NUM_KV_STAGES)

        if tidx >= GQA_NUM_MATH_THREADS:
            cute.arch.warpgroup_reg_dealloc(GQA_NUM_TMA_REGS)
            lane_idx_tma = tidx % Int32(32)
            is_tma_prefetch_warp = warp_idx == Int32(tma_prefetch_warp)
            if is_tma_prefetch_warp and lane_idx_tma == Int32(0):
                block_q_idx = q_block_idx
                if block_q_idx < num_q_blocks_per_kv:
                    pipeline_q.producer_acquire(q_prod)
                    q_src_idx = kv_head_idx * num_q_blocks_per_kv + block_q_idx
                    cute.copy(
                        tma_atom_Q,
                        tQgQ[None, q_src_idx],
                        tQsQ[None, q_prod.index],
                        tma_bar_ptr=pipeline_q.producer_get_barrier(q_prod),
                    )
                    q_prod.advance()

                while block_q_idx < num_q_blocks_per_kv:
                    next_block_q = block_q_idx + grid_q_blocks
                    if next_block_q < num_q_blocks_per_kv:
                        pipeline_q.producer_acquire(q_prod)
                        q_src_idx = kv_head_idx * num_q_blocks_per_kv + next_block_q
                        cute.copy(
                            tma_atom_Q,
                            tQgQ[None, q_src_idx],
                            tQsQ[None, q_prod.index],
                            tma_bar_ptr=pipeline_q.producer_get_barrier(q_prod),
                        )
                        q_prod.advance()

                    num_kv_blocks = cute.ceil_div(seq_k, GQA_BLOCK_KV)
                    for kv_block in cutlass.range(num_kv_blocks, unroll=1):
                        pipeline_k.producer_acquire(k_prod)
                        k_src_idx = kv_head_idx * num_kv_blocks + kv_block
                        cute.copy(
                            tma_atom_K,
                            tKgK[None, k_src_idx],
                            tKsK[None, k_prod.index],
                            tma_bar_ptr=pipeline_k.producer_get_barrier(k_prod),
                        )
                        k_prod.advance()

                    block_q_idx = next_block_q
        else:
            cute.arch.warpgroup_reg_alloc(GQA_NUM_MATH_REGS)
            warp_idx = tidx // Int32(32)
            warp_group_idx = cute.arch.make_warp_uniform(warp_idx // Int32(4))
            lane_idx = tidx % Int32(32)
            warp_group_thread_layout = cute.make_layout(
                GQA_NUM_MATH_THREADS // 128, stride=128
            )
            thr_mma_thread = tiled_mma.get_slice(tidx)
            wg_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))
            cS = cute.make_identity_tensor((GQA_BLOCK_KV, GQA_BLOCK_QH))
            tScS = thr_mma_thread.partition_C(cS)
            block_q_idx = q_block_idx
            num_kv_blocks = cute.ceil_div(seq_k, GQA_BLOCK_KV)
            while block_q_idx < num_q_blocks_per_kv:
                pipeline_q.consumer_wait(q_cons)
                q_stage = q_cons.index
                weight_tile = [Float32(0.0) for _ in range(GQA_BLOCK_Q * Q_HEADS_PER_KV)]
                global_q_weight_base = block_q_idx * GQA_BLOCK_Q
                q_slots_per_lane = GQA_BLOCK_Q // 4
                for qg in cutlass.range_constexpr(GQA_BLOCK_Q):
                    slot_div = qg % q_slots_per_lane
                    head_pair = qg // q_slots_per_lane
                    for head_slot in cutlass.range_constexpr(Q_HEADS_PER_KV):
                        lane = head_slot // 2
                        bit = head_slot % 2
                        global_q = global_q_weight_base + Int32(slot_div * 4 + lane)
                        logical_head = Int32(head_pair * 2 + bit)
                        weight_tile[qg * Q_HEADS_PER_KV + head_slot] = _load_f32_global(
                            mW, (kv_head_idx, global_q, logical_head)
                        )

                for kv_block in cutlass.range(num_kv_blocks, unroll=1):
                    pipeline_k.consumer_wait(k_cons)
                    k_stage = k_cons.index

                    sK_stage_ptr = elem_pointer(sK, (0, 0, k_stage))
                    sQ_stage_ptr = elem_pointer(sQ, (0, 0, q_stage))
                    sK_stage = cute.make_tensor(sK_stage_ptr, sK_layout_single_outer)
                    sQ_stage = cute.make_tensor(sQ_stage_ptr, sQ_layout_single_outer)

                    tCrA = tiled_mma.make_fragment_A(wg_mma.partition_A(sK_stage))
                    tCrB = tiled_mma.make_fragment_B(wg_mma.partition_B(sQ_stage))
                    acc = tiled_mma.make_fragment_C(tScS.layout)
                    hop_helpers.warpgroup_gemm_with_optional_swap_wait(
                        tiled_mma,
                        acc,
                        tCrA,
                        tCrB,
                        zero_init=True,
                        wg_wait=0,
                    )
                    acc_val = acc.load()
                    # Note(wangbojun/codex): Release the K-stage as soon as MMA
                    # finishes so the producer can advance while epilogue runs.
                    pipeline_k.consumer_release(k_cons)
                    k_cons.advance()
                    kv_base = kv_block * GQA_BLOCK_KV
                    warp_idx = tidx // Int32(32)
                    lane_idx = tidx % Int32(32)
                    warp_offset = warp_idx * Int32(16)
                    lane_group = lane_idx // Int32(4)
                    row0 = warp_offset + lane_group
                    row1 = row0 + Int32(8)
                    lane_mod4 = lane_idx % Int32(4)
                    lane_is1 = lane_mod4 == Int32(1)
                    lane_is2 = lane_mod4 == Int32(2)
                    lane_is3 = lane_mod4 == Int32(3)
                    # Note(wangbojun/codex): The GQA logits kernel is specialized for
                    # q_heads_per_kv=8. The packed inverse-Q layout keeps each lane's
                    # owned accumulator fragment lane-local and avoids the old quad reduction.
                    sum0 = [Float32(0.0) for _ in range(GQA_BLOCK_Q)]
                    sum1 = [Float32(0.0) for _ in range(GQA_BLOCK_Q)]
                    weight_lane_even = [Float32(0.0) for _ in range(GQA_BLOCK_Q)]
                    weight_lane_odd = [Float32(0.0) for _ in range(GQA_BLOCK_Q)]
                    for qg in cutlass.range_constexpr(GQA_BLOCK_Q):
                        w_even = weight_tile[qg * Q_HEADS_PER_KV]
                        w_odd = weight_tile[qg * Q_HEADS_PER_KV + 1]
                        w_even = cutlass.select_(
                            lane_is1,
                            weight_tile[qg * Q_HEADS_PER_KV + 2],
                            w_even,
                        )
                        w_odd = cutlass.select_(
                            lane_is1,
                            weight_tile[qg * Q_HEADS_PER_KV + 3],
                            w_odd,
                        )
                        w_even = cutlass.select_(
                            lane_is2,
                            weight_tile[qg * Q_HEADS_PER_KV + 4],
                            w_even,
                        )
                        w_odd = cutlass.select_(
                            lane_is2,
                            weight_tile[qg * Q_HEADS_PER_KV + 5],
                            w_odd,
                        )
                        w_even = cutlass.select_(
                            lane_is3,
                            weight_tile[qg * Q_HEADS_PER_KV + 6],
                            w_even,
                        )
                        w_odd = cutlass.select_(
                            lane_is3,
                            weight_tile[qg * Q_HEADS_PER_KV + 7],
                            w_odd,
                        )
                        weight_lane_even[qg] = w_even
                        weight_lane_odd[qg] = w_odd

                    q_slot_mask = Int32(q_slots_per_lane - 1)
                    for r in cutlass.range(cute.size(tScS), unroll_full=True):
                        m = tScS[r][0]
                        n = tScS[r][1]
                        cond_row0 = m == row0
                        cond_row1 = m == row1
                        v_raw = acc_val[r]
                        v_relu = cutlass.select_(
                            v_raw > Float32(0.0), v_raw, Float32(0.0)
                        )
                        physical_q_slot = n // Int32(Q_HEADS_PER_KV)
                        head_is_odd = (n & Int32(1)) == Int32(1)
                        local_q_slot = physical_q_slot & q_slot_mask
                        wv = cutlass.select_(
                            head_is_odd,
                            weight_lane_odd[0],
                            weight_lane_even[0],
                        )
                        for qg in cutlass.range_constexpr(1, GQA_BLOCK_Q):
                            wv_q = cutlass.select_(
                                head_is_odd,
                                weight_lane_odd[qg],
                                weight_lane_even[qg],
                            )
                            wv = cutlass.select_(
                                physical_q_slot == Int32(qg),
                                wv_q,
                                wv,
                            )
                        v = v_relu * wv
                        for q_slot in cutlass.range_constexpr(q_slots_per_lane):
                            cond_q = local_q_slot == Int32(q_slot)

                            def row0_then(s):
                                return s + v

                            def row1_then(s):
                                return s + v

                            sum0[q_slot] = if_generate(
                                cond_row0 & cond_q,
                                row0_then,
                                lambda s: s,
                                [sum0[q_slot]],
                                [Float32],
                            )
                            sum1[q_slot] = if_generate(
                                cond_row1 & cond_q,
                                row1_then,
                                lambda s: s,
                                [sum1[q_slot]],
                                [Float32],
                            )
                    global_q_base = kv_head_idx * seq_q + block_q_idx * GQA_BLOCK_Q
                    global_k0 = kv_base + row0
                    global_k1 = kv_base + row1
                    for q_slot in cutlass.range_constexpr(GQA_BLOCK_Q // 4):
                        qg_base = Int32(q_slot * 4)
                        sum0_lane = sum0[q_slot]
                        sum1_lane = sum1[q_slot]
                        global_q = global_q_base + qg_base + lane_mod4
                        dst0 = cute.make_tensor(
                            elem_pointer(mO, (global_q, global_k0)),
                            cute.make_layout((1,), stride=(1,)),
                        )
                        dst0[0] = mO.element_type(sum0_lane)
                        dst1 = cute.make_tensor(
                            elem_pointer(mO, (global_q, global_k1)),
                            cute.make_layout((1,), stride=(1,)),
                        )
                        dst1[0] = mO.element_type(sum1_lane)

                pipeline_q.consumer_release(q_cons)
                q_cons.advance()
                block_q_idx = block_q_idx + grid_q_blocks

@cute.jit
def fp8_gqa_logits_fast_sm90_call(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mW: cute.Tensor,
    mO: cute.Tensor,
    stream,
    seq_q: Int32,
    seq_k: Int32,
    kv_heads: Int32,
    grid_q: Int32,
):
    # Keep the original dynamic layouts from cute_tensor_like.
    mQ = mQ
    mK = mK

    mma_tiler_mnk = (GQA_BLOCK_KV, GQA_BLOCK_QH, HEAD_DIM)
    tiled_mma = sm90_utils.make_trivial_tiled_mma(
        Float8E4M3FN,
        Float8E4M3FN,
        LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        Float32,
        atom_layout_mnk=(GQA_BLOCK_KV // 64, 1, 1),
        tiler_mn=(64, GQA_BLOCK_QH),
    )
    sQ_layout_fp8 = sm90_utils.make_smem_layout_b(
        LayoutEnum.ROW_MAJOR,
        mma_tiler_mnk,
        Float8E4M3FN,
        GQA_NUM_Q_STAGES,
    )
    sK_layout_fp8 = sm90_utils.make_smem_layout_a(
        LayoutEnum.ROW_MAJOR,
        mma_tiler_mnk,
        Float8E4M3FN,
        GQA_NUM_KV_STAGES,
    )
    sQ_layout_single = cute.slice_(sQ_layout_fp8, (None, None, 0))
    sK_layout_single = cute.slice_(sK_layout_fp8, (None, None, 0))
    sQ_layout_single_outer = sQ_layout_single.outer
    sK_layout_single_outer = sK_layout_single.outer

    tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mQ,
        sQ_layout_single,
        (GQA_BLOCK_QH, HEAD_DIM),
    )
    tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mK,
        sK_layout_single,
        (GQA_BLOCK_KV, HEAD_DIM),
    )
    grid = (grid_q, kv_heads, 1)
    fp8_gqa_logits_fast_sm90_kernel(
        tma_atom_Q,
        tma_tensor_Q,
        tma_atom_K,
        tma_tensor_K,
        tiled_mma,
        sQ_layout_fp8,
        sK_layout_fp8,
        sQ_layout_single_outer,
        sK_layout_single_outer,
        mW,
        mO,
        seq_q,
        seq_k,
        kv_heads,
    ).launch(grid=grid, block=[GQA_NUM_THREADS, 1, 1], stream=stream)


@cute.kernel
def fp8_gqa_logits_fast_sm90_kernel_q16(
    tma_atom_Q: cute.CopyAtom,
    tma_tensor_Q: cute.Tensor,
    tma_atom_K: cute.CopyAtom,
    tma_tensor_K: cute.Tensor,
    tiled_mma: cute.TiledMma,
    sQ_layout_fp8: cute.ComposedLayout,
    sK_layout_fp8: cute.ComposedLayout,
    sQ_layout_single_outer: cute.Layout,
    sK_layout_single_outer: cute.Layout,
    mW: cute.Tensor,
    mO: cute.Tensor,
    seq_q: Int32,
    seq_k: Int32,
    kv_heads: Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    q_block_idx = cute.arch.block_idx()[0]
    kv_head_idx = cute.arch.block_idx()[1]
    grid_q_blocks = cute.arch.grid_dim()[0]

    num_q_blocks_per_kv = cute.ceil_div(seq_q, GQA_BLOCK_Q_Q16)
    if q_block_idx < num_q_blocks_per_kv and kv_head_idx < kv_heads:
        tma_prefetch_warp = GQA_NUM_MATH_THREADS // 32
        if warp_idx == 0 and (tidx % 32) == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)

        smem = cutlass.utils.SmemAllocator()
        sQ_struct = cute.struct.Align[
            cute.struct.MemRange[Float8E4M3FN, cute.cosize(sQ_layout_fp8)], 1024
        ]
        sK_struct = cute.struct.Align[
            cute.struct.MemRange[Float8E4M3FN, cute.cosize(sK_layout_fp8)], 1024
        ]
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, GQA_NUM_Q_STAGES * 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, GQA_NUM_KV_STAGES * 2]

        @cute.struct
        class SharedStorage:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            sQ: sQ_struct
            sK: sK_struct

        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout_fp8.outer, swizzle=sQ_layout_fp8.inner)
        sK = storage.sK.get_tensor(sK_layout_fp8.outer, swizzle=sK_layout_fp8.inner)

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, GQA_NUM_WARPS)
        pipeline_q = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_Q.data_ptr(),
            num_stages=GQA_NUM_Q_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=GQA_BLOCK_QH * HEAD_DIM,
        )
        pipeline_k = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=GQA_NUM_KV_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=GQA_BLOCK_KV * HEAD_DIM,
        )

        gQ = cute.local_tile(tma_tensor_Q, (GQA_BLOCK_QH, HEAD_DIM), (None, 0))
        gK = cute.local_tile(tma_tensor_K, (GQA_BLOCK_KV, HEAD_DIM), (None, 0))
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_atom_Q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, cute.rank(sQ) - 1),
            cute.group_modes(gQ, 0, cute.rank(gQ) - 1),
        )
        tKsK, tKgK = cpasync.tma_partition(
            tma_atom_K,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, cute.rank(sK) - 1),
            cute.group_modes(gK, 0, cute.rank(gK) - 1),
        )
        q_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, GQA_NUM_Q_STAGES)
        q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, GQA_NUM_Q_STAGES)
        k_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, GQA_NUM_KV_STAGES)
        k_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, GQA_NUM_KV_STAGES)

        if tidx >= GQA_NUM_MATH_THREADS:
            cute.arch.warpgroup_reg_dealloc(GQA_NUM_TMA_REGS)
            lane_idx_tma = tidx % Int32(32)
            is_tma_prefetch_warp = warp_idx == Int32(tma_prefetch_warp)
            if is_tma_prefetch_warp and lane_idx_tma == Int32(0):
                block_q_idx = q_block_idx
                if block_q_idx < num_q_blocks_per_kv:
                    pipeline_q.producer_acquire(q_prod)
                    q_src_idx = kv_head_idx * num_q_blocks_per_kv + block_q_idx
                    cute.copy(
                        tma_atom_Q,
                        tQgQ[None, q_src_idx],
                        tQsQ[None, q_prod.index],
                        tma_bar_ptr=pipeline_q.producer_get_barrier(q_prod),
                    )
                    q_prod.advance()

                while block_q_idx < num_q_blocks_per_kv:
                    next_block_q = block_q_idx + grid_q_blocks
                    if next_block_q < num_q_blocks_per_kv:
                        pipeline_q.producer_acquire(q_prod)
                        q_src_idx = kv_head_idx * num_q_blocks_per_kv + next_block_q
                        cute.copy(
                            tma_atom_Q,
                            tQgQ[None, q_src_idx],
                            tQsQ[None, q_prod.index],
                            tma_bar_ptr=pipeline_q.producer_get_barrier(q_prod),
                        )
                        q_prod.advance()

                    num_kv_blocks = cute.ceil_div(seq_k, GQA_BLOCK_KV)
                    for kv_block in cutlass.range(num_kv_blocks, unroll=1):
                        pipeline_k.producer_acquire(k_prod)
                        k_src_idx = kv_head_idx * num_kv_blocks + kv_block
                        cute.copy(
                            tma_atom_K,
                            tKgK[None, k_src_idx],
                            tKsK[None, k_prod.index],
                            tma_bar_ptr=pipeline_k.producer_get_barrier(k_prod),
                        )
                        k_prod.advance()

                    block_q_idx = next_block_q
        else:
            cute.arch.warpgroup_reg_alloc(GQA_NUM_MATH_REGS)
            warp_idx = tidx // Int32(32)
            warp_group_idx = cute.arch.make_warp_uniform(warp_idx // Int32(4))
            lane_idx = tidx % Int32(32)
            warp_group_thread_layout = cute.make_layout(
                GQA_NUM_MATH_THREADS // 128, stride=128
            )
            thr_mma_thread = tiled_mma.get_slice(tidx)
            wg_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))
            cS = cute.make_identity_tensor((GQA_BLOCK_KV, GQA_BLOCK_QH))
            tScS = thr_mma_thread.partition_C(cS)
            block_q_idx = q_block_idx
            num_kv_blocks = cute.ceil_div(seq_k, GQA_BLOCK_KV)
            while block_q_idx < num_q_blocks_per_kv:
                pipeline_q.consumer_wait(q_cons)
                q_stage = q_cons.index
                weight_tile = [Float32(0.0) for _ in range(GQA_BLOCK_QH)]
                global_q_weight_base = block_q_idx * GQA_BLOCK_Q_Q16
                q_slots_per_lane = GQA_BLOCK_Q_Q16 // 4
                for qg in cutlass.range_constexpr(GQA_BLOCK_QH // Q_HEADS_PER_KV):
                    slot_div = qg % q_slots_per_lane
                    head_pair = qg // q_slots_per_lane
                    for head_slot in cutlass.range_constexpr(Q_HEADS_PER_KV):
                        lane = head_slot // 2
                        bit = head_slot % 2
                        global_q = global_q_weight_base + Int32(slot_div * 4 + lane)
                        logical_head = Int32(head_pair * 2 + bit)
                        weight_tile[qg * Q_HEADS_PER_KV + head_slot] = _load_f32_global(
                            mW, (kv_head_idx, global_q, logical_head)
                        )

                for kv_block in cutlass.range(num_kv_blocks, unroll=1):
                    pipeline_k.consumer_wait(k_cons)
                    k_stage = k_cons.index

                    sK_stage_ptr = elem_pointer(sK, (0, 0, k_stage))
                    sQ_stage_ptr = elem_pointer(sQ, (0, 0, q_stage))
                    sK_stage = cute.make_tensor(sK_stage_ptr, sK_layout_single_outer)
                    sQ_stage = cute.make_tensor(sQ_stage_ptr, sQ_layout_single_outer)

                    tCrA = tiled_mma.make_fragment_A(wg_mma.partition_A(sK_stage))
                    tCrB = tiled_mma.make_fragment_B(wg_mma.partition_B(sQ_stage))
                    acc = tiled_mma.make_fragment_C(tScS.layout)
                    hop_helpers.warpgroup_gemm_with_optional_swap_wait(
                        tiled_mma,
                        acc,
                        tCrA,
                        tCrB,
                        zero_init=True,
                        wg_wait=0,
                    )
                    acc_val = acc.load()
                    pipeline_k.consumer_release(k_cons)
                    k_cons.advance()
                    kv_base = kv_block * GQA_BLOCK_KV
                    warp_idx = tidx // Int32(32)
                    lane_idx = tidx % Int32(32)
                    warp_offset = warp_idx * Int32(16)
                    lane_group = lane_idx // Int32(4)
                    row0 = warp_offset + lane_group
                    row1 = row0 + Int32(8)
                    lane_mod4 = lane_idx % Int32(4)
                    lane_is1 = lane_mod4 == Int32(1)
                    lane_is2 = lane_mod4 == Int32(2)
                    lane_is3 = lane_mod4 == Int32(3)
                    sum0 = [Float32(0.0) for _ in range(q_slots_per_lane)]
                    sum1 = [Float32(0.0) for _ in range(q_slots_per_lane)]
                    weight_lane_even = [
                        Float32(0.0) for _ in range(GQA_BLOCK_QH // Q_HEADS_PER_KV)
                    ]
                    weight_lane_odd = [
                        Float32(0.0) for _ in range(GQA_BLOCK_QH // Q_HEADS_PER_KV)
                    ]
                    for qg in cutlass.range_constexpr(GQA_BLOCK_QH // Q_HEADS_PER_KV):
                        w_even = weight_tile[qg * Q_HEADS_PER_KV]
                        w_odd = weight_tile[qg * Q_HEADS_PER_KV + 1]
                        w_even = cutlass.select_(
                            lane_is1,
                            weight_tile[qg * Q_HEADS_PER_KV + 2],
                            w_even,
                        )
                        w_odd = cutlass.select_(
                            lane_is1,
                            weight_tile[qg * Q_HEADS_PER_KV + 3],
                            w_odd,
                        )
                        w_even = cutlass.select_(
                            lane_is2,
                            weight_tile[qg * Q_HEADS_PER_KV + 4],
                            w_even,
                        )
                        w_odd = cutlass.select_(
                            lane_is2,
                            weight_tile[qg * Q_HEADS_PER_KV + 5],
                            w_odd,
                        )
                        w_even = cutlass.select_(
                            lane_is3,
                            weight_tile[qg * Q_HEADS_PER_KV + 6],
                            w_even,
                        )
                        w_odd = cutlass.select_(
                            lane_is3,
                            weight_tile[qg * Q_HEADS_PER_KV + 7],
                            w_odd,
                        )
                        weight_lane_even[qg] = w_even
                        weight_lane_odd[qg] = w_odd

                    q_slot_mask = Int32(q_slots_per_lane - 1)
                    for r in cutlass.range(cute.size(tScS), unroll_full=True):
                        m = tScS[r][0]
                        n = tScS[r][1]
                        cond_row0 = m == row0
                        cond_row1 = m == row1
                        v_raw = acc_val[r]
                        v_relu = cutlass.select_(
                            v_raw > Float32(0.0), v_raw, Float32(0.0)
                        )
                        physical_q_slot = n // Int32(Q_HEADS_PER_KV)
                        head_is_odd = (n & Int32(1)) == Int32(1)
                        local_q_slot = physical_q_slot & q_slot_mask
                        wv = cutlass.select_(
                            head_is_odd,
                            weight_lane_odd[0],
                            weight_lane_even[0],
                        )
                        for qg in cutlass.range_constexpr(1, GQA_BLOCK_QH // Q_HEADS_PER_KV):
                            wv_q = cutlass.select_(
                                head_is_odd,
                                weight_lane_odd[qg],
                                weight_lane_even[qg],
                            )
                            wv = cutlass.select_(
                                physical_q_slot == Int32(qg),
                                wv_q,
                                wv,
                            )
                        v = v_relu * wv

                        for q_slot in cutlass.range_constexpr(q_slots_per_lane):
                            cond_q = local_q_slot == Int32(q_slot)

                            def row0_then(s):
                                return s + v

                            def row1_then(s):
                                return s + v

                            sum0[q_slot] = if_generate(
                                cond_row0 & cond_q,
                                row0_then,
                                lambda s: s,
                                [sum0[q_slot]],
                                [Float32],
                            )
                            sum1[q_slot] = if_generate(
                                cond_row1 & cond_q,
                                row1_then,
                                lambda s: s,
                                [sum1[q_slot]],
                                [Float32],
                            )

                    global_q_base = kv_head_idx * seq_q + block_q_idx * GQA_BLOCK_Q_Q16
                    global_k0 = kv_base + row0
                    global_k1 = kv_base + row1
                    for q_slot in cutlass.range_constexpr(q_slots_per_lane):
                        qg_base = Int32(q_slot * 4)
                        global_q = global_q_base + qg_base + lane_mod4
                        dst0 = cute.make_tensor(
                            elem_pointer(mO, (global_q, global_k0)),
                            cute.make_layout((1,), stride=(1,)),
                        )
                        dst0[0] = mO.element_type(sum0[q_slot])
                        dst1 = cute.make_tensor(
                            elem_pointer(mO, (global_q, global_k1)),
                            cute.make_layout((1,), stride=(1,)),
                        )
                        dst1[0] = mO.element_type(sum1[q_slot])

                pipeline_q.consumer_release(q_cons)
                q_cons.advance()
                block_q_idx = block_q_idx + grid_q_blocks


@cute.jit
def fp8_gqa_logits_fast_sm90_call_q16(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mW: cute.Tensor,
    mO: cute.Tensor,
    stream,
    seq_q: Int32,
    seq_k: Int32,
    kv_heads: Int32,
    grid_q: Int32,
):
    mQ = mQ
    mK = mK

    mma_tiler_mnk = (GQA_BLOCK_KV, GQA_BLOCK_QH, HEAD_DIM)
    tiled_mma = sm90_utils.make_trivial_tiled_mma(
        Float8E4M3FN,
        Float8E4M3FN,
        LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        Float32,
        atom_layout_mnk=(GQA_BLOCK_KV // 64, 1, 1),
        tiler_mn=(64, GQA_BLOCK_QH),
    )
    sQ_layout_fp8 = sm90_utils.make_smem_layout_b(
        LayoutEnum.ROW_MAJOR,
        mma_tiler_mnk,
        Float8E4M3FN,
        GQA_NUM_Q_STAGES,
    )
    sK_layout_fp8 = sm90_utils.make_smem_layout_a(
        LayoutEnum.ROW_MAJOR,
        mma_tiler_mnk,
        Float8E4M3FN,
        GQA_NUM_KV_STAGES,
    )
    sQ_layout_single = cute.slice_(sQ_layout_fp8, (None, None, 0))
    sK_layout_single = cute.slice_(sK_layout_fp8, (None, None, 0))
    sQ_layout_single_outer = sQ_layout_single.outer
    sK_layout_single_outer = sK_layout_single.outer

    tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mQ,
        sQ_layout_single,
        (GQA_BLOCK_QH, HEAD_DIM),
    )
    tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mK,
        sK_layout_single,
        (GQA_BLOCK_KV, HEAD_DIM),
    )
    grid = (grid_q, kv_heads, 1)
    fp8_gqa_logits_fast_sm90_kernel_q16(
        tma_atom_Q,
        tma_tensor_Q,
        tma_atom_K,
        tma_tensor_K,
        tiled_mma,
        sQ_layout_fp8,
        sK_layout_fp8,
        sQ_layout_single_outer,
        sK_layout_single_outer,
        mW,
        mO,
        seq_q,
        seq_k,
        kv_heads,
    ).launch(grid=grid, block=[GQA_NUM_THREADS, 1, 1], stream=stream)


@cached_compile_function
def _compile_weighted_logits_sm90_gqa_for_signature(
    device_key: tuple[str, Optional[int]],
    kv_heads: int,
    out_dynamic_rows: bool,
    w_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    out_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
) -> cute.JitFunction:
    device = sparse_utils.device_from_cache_key(device_key)
    q_placeholder = torch.empty(
        (int(kv_heads) * int(GQA_BLOCK_Q) * int(Q_HEADS_PER_KV), int(HEAD_DIM)),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    k_placeholder = torch.empty(
        (int(kv_heads) * int(GQA_BLOCK_KV), int(HEAD_DIM)),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    w_mat = _placeholder_from_signature(
        w_signature,
        device=device,
        dynamic_shape_fill=GQA_BLOCK_Q,
        dynamic_stride_fill=GQA_BLOCK_Q * Q_HEADS_PER_KV,
    )
    out_placeholder = _placeholder_from_signature(
        out_signature,
        device=device,
        dynamic_shape_fill=int(kv_heads) * int(GQA_BLOCK_Q),
    )
    fQ = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        q_placeholder,
        alignment=16,
        dynamic_shape_dim=0,
        divisibility=GQA_BLOCK_Q * Q_HEADS_PER_KV,
    )
    fK = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        k_placeholder,
        alignment=16,
        dynamic_shape_dim=0,
        divisibility=GQA_BLOCK_KV,
    )
    fW = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        w_mat,
        alignment=16,
        dynamic_shape_dim=1,
        dynamic_stride_dims=(0,),
        divisibility=GQA_BLOCK_Q,
    )
    fO = (
        sparse_utils.make_fake_tensor_like_with_dynamic_dim(
            out_placeholder,
            alignment=16,
            dynamic_shape_dims=(0, 1),
            dynamic_stride_dims=(0,),
            divisibility=GQA_BLOCK_Q,
        )
        if out_dynamic_rows
        else sparse_utils.make_fake_tensor_like_with_dynamic_dim(
            out_placeholder, alignment=16)
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return _cute_compile(
        fp8_gqa_logits_fast_sm90_call,
        fQ,
        fK,
        fW,
        fO,
        stream_fake,
        Int32(GQA_BLOCK_Q),
        Int32(GQA_BLOCK_KV),
        Int32(kv_heads),
        Int32(1),
    )


def _validate_prepared_logits_tensors(
    q_runtime: torch.Tensor,
    k_runtime: torch.Tensor,
    w_mat: torch.Tensor,
    out_flat: torch.Tensor,
    *,
    seq_q: int,
    seq_k: int,
    kv_heads: int,
    q_heads_per_kv: int,
    block_q: int,
    kernel_name: str,
) -> tuple[int, int, int]:
    if q_runtime.dim() != 2:
        raise ValueError(f"{kernel_name} prepared q must be rank-2 [kv_head * seq_q * q_heads_per_kv, head_dim]")
    if k_runtime.dim() != 2:
        raise ValueError(f"{kernel_name} prepared k must be rank-2 [kv_head * seq_k, head_dim]")
    if w_mat.dim() != 3:
        raise ValueError(f"{kernel_name} prepared weights must be rank-3 [kv_head, seq_q, q_heads_per_kv]")
    seq_q = int(seq_q)
    seq_k = int(seq_k)
    kv_heads = int(kv_heads)
    q_heads_per_kv = int(q_heads_per_kv)
    block_q = int(block_q)
    if int(q_runtime.shape[1]) != HEAD_DIM or int(k_runtime.shape[1]) != HEAD_DIM:
        raise ValueError(f"{kernel_name} head_dim must be {HEAD_DIM}")
    expected_q = (kv_heads * seq_q * q_heads_per_kv, HEAD_DIM)
    if tuple(q_runtime.shape) != expected_q:
        raise ValueError(
            "prepared q shape must be [kv_heads * seq_q * q_heads_per_kv, head_dim], "
            f"got {tuple(q_runtime.shape)} expected {expected_q}"
        )
    expected_k = (kv_heads * seq_k, HEAD_DIM)
    if tuple(k_runtime.shape) != expected_k:
        raise ValueError(
            "prepared k shape must be [kv_heads * seq_k, head_dim], "
            f"got {tuple(k_runtime.shape)} expected {expected_k}"
        )
    expected_w = (kv_heads, seq_q, q_heads_per_kv)
    if tuple(w_mat.shape) != expected_w:
        raise ValueError(
            "prepared weights shape must be [kv_heads, seq_q, q_heads_per_kv], "
            f"got {tuple(w_mat.shape)} expected {expected_w}"
        )
    if w_mat.device != q_runtime.device or k_runtime.device != q_runtime.device:
        raise ValueError("prepared q/k/weights must be on the same device")
    if out_flat.device != q_runtime.device:
        raise ValueError("out_flat must be on the same device as q/k")
    if q_runtime.dtype != torch.float8_e4m3fn or k_runtime.dtype != torch.float8_e4m3fn:
        raise ValueError("prepared q/k must be torch.float8_e4m3fn")
    if out_flat.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("out_flat must be float16, bfloat16, or float32")
    expected_out = (kv_heads * seq_q, seq_k)
    if tuple(out_flat.shape) != expected_out:
        raise ValueError(
            "out_flat shape must be [kv_heads * seq_q, seq_k], got "
            f"{tuple(out_flat.shape)} expected {expected_out}"
        )
    if seq_q % block_q != 0:
        raise ValueError(f"seq_q must be divisible by {block_q} (gqa_block_q)")
    if seq_k % GQA_BLOCK_KV != 0:
        raise ValueError(f"seq_k must be divisible by {GQA_BLOCK_KV} (gqa_block_kv)")
    return seq_q, seq_k, kv_heads


def _compile_logits_sm90_gqa(
    q_runtime: torch.Tensor,
    k_runtime: torch.Tensor,
    w_mat: torch.Tensor,
    out_flat: torch.Tensor,
    *,
    seq_q: int,
    seq_k: int,
    kv_heads: int,
    stream: Optional[cutlass.backend.cuda.CUstream] = None,
    grid_q_multiplier: int = 2,
):
    if stream is not None:
        raise ValueError("GQA logits use the TVM-FFI environment stream")
    if grid_q_multiplier <= 0:
        raise ValueError("grid_q_multiplier must be > 0")
    seq_q, seq_k, kv_heads = _validate_prepared_logits_tensors(
        q_runtime,
        k_runtime,
        w_mat,
        out_flat,
        seq_q=seq_q,
        seq_k=seq_k,
        kv_heads=kv_heads,
        q_heads_per_kv=Q_HEADS_PER_KV,
        block_q=GQA_BLOCK_Q,
        kernel_name="GQA logits",
    )

    out_dynamic_rows = int(out_flat.stride(1)) == 1
    if out_dynamic_rows:
        out_signature = _tensor_signature_dynamic(
            out_flat,
            dynamic_shape_dims=(0, 1),
            dynamic_stride_dims=(0,),
        )
    else:
        out_signature = _tensor_signature(out_flat)
    num_sms = torch.cuda.get_device_properties(q_runtime.device).multi_processor_count
    grid_q = min(num_sms * grid_q_multiplier, seq_q // GQA_BLOCK_Q)
    w_signature = _tensor_signature_dynamic(
        w_mat,
        dynamic_shape_dims=(1, ),
        dynamic_stride_dims=(0, ),
    )
    mQ = q_runtime
    mK = k_runtime
    mW = w_mat
    mO = out_flat

    compiled_kernel = _compile_weighted_logits_sm90_gqa_for_signature(
        sparse_utils.device_cache_key(q_runtime.device),
        int(kv_heads),
        bool(out_dynamic_rows),
        w_signature,
        out_signature,
    )

    def compiled(mQ, mK, mW, mO, seq_q, seq_k, kv_heads):
        return compiled_kernel(
            mQ,
            mK,
            mW,
            mO,
            seq_q,
            seq_k,
            kv_heads,
            Int32(grid_q),
        )

    return compiled, (mQ, mK, mW, mO), out_flat, (q_runtime, k_runtime, w_mat)


@cached_compile_function
def _compile_weighted_logits_sm90_gqa_q16_for_signature(
    device_key: tuple[str, Optional[int]],
    kv_heads: int,
    out_dynamic_rows: bool,
    w_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    out_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
) -> cute.JitFunction:
    device = sparse_utils.device_from_cache_key(device_key)
    q_placeholder = torch.empty(
        (int(kv_heads) * int(GQA_BLOCK_QH), int(HEAD_DIM)),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    k_placeholder = torch.empty(
        (int(kv_heads) * int(GQA_BLOCK_KV), int(HEAD_DIM)),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    w_mat = _placeholder_from_signature(
        w_signature,
        device=device,
        dynamic_shape_fill=GQA_BLOCK_Q_Q16,
        dynamic_stride_fill=GQA_BLOCK_QH,
    )
    out_placeholder = _placeholder_from_signature(
        out_signature,
        device=device,
        dynamic_shape_fill=int(kv_heads) * int(GQA_BLOCK_Q_Q16),
    )
    fQ = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        q_placeholder,
        alignment=16,
        dynamic_shape_dim=0,
        divisibility=GQA_BLOCK_QH,
    )
    fK = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        k_placeholder,
        alignment=16,
        dynamic_shape_dim=0,
        divisibility=GQA_BLOCK_KV,
    )
    fW = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        w_mat,
        alignment=16,
        dynamic_shape_dim=1,
        dynamic_stride_dims=(0,),
        divisibility=GQA_BLOCK_Q_Q16,
    )
    fO = (
        sparse_utils.make_fake_tensor_like_with_dynamic_dim(
            out_placeholder,
            alignment=16,
            dynamic_shape_dims=(0, 1),
            dynamic_stride_dims=(0,),
            divisibility=GQA_BLOCK_Q_Q16,
        )
        if out_dynamic_rows
        else sparse_utils.make_fake_tensor_like_with_dynamic_dim(
            out_placeholder, alignment=16)
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return _cute_compile(
        fp8_gqa_logits_fast_sm90_call_q16,
        fQ,
        fK,
        fW,
        fO,
        stream_fake,
        Int32(GQA_BLOCK_Q_Q16),
        Int32(GQA_BLOCK_KV),
        Int32(kv_heads),
        Int32(1),
    )


def _compile_logits_sm90_gqa_q16(
    q_runtime: torch.Tensor,
    k_runtime: torch.Tensor,
    w_mat: torch.Tensor,
    out_flat: torch.Tensor,
    *,
    seq_q: int,
    seq_k: int,
    kv_heads: int,
    stream: Optional[cutlass.backend.cuda.CUstream] = None,
    grid_q_multiplier: int = 2,
):
    if stream is not None:
        raise ValueError("GQA q16 logits use the TVM-FFI environment stream")
    if grid_q_multiplier <= 0:
        raise ValueError("grid_q_multiplier must be > 0")
    seq_q, seq_k, kv_heads = _validate_prepared_logits_tensors(
        q_runtime,
        k_runtime,
        w_mat,
        out_flat,
        seq_q=seq_q,
        seq_k=seq_k,
        kv_heads=kv_heads,
        q_heads_per_kv=Q_HEADS_PER_KV_Q16,
        block_q=GQA_BLOCK_Q_Q16,
        kernel_name="GQA q16 logits",
    )

    out_dynamic_rows = int(out_flat.stride(1)) == 1
    if out_dynamic_rows:
        out_signature = _tensor_signature_dynamic(
            out_flat,
            dynamic_shape_dims=(0, 1),
            dynamic_stride_dims=(0,),
        )
    else:
        out_signature = _tensor_signature(out_flat)
    num_sms = torch.cuda.get_device_properties(q_runtime.device).multi_processor_count
    grid_q = min(num_sms * grid_q_multiplier, seq_q // GQA_BLOCK_Q_Q16)
    w_signature = _tensor_signature_dynamic(
        w_mat,
        dynamic_shape_dims=(1, ),
        dynamic_stride_dims=(0, ),
    )
    mQ = q_runtime
    mK = k_runtime
    mW = w_mat
    mO = out_flat

    compiled_kernel = _compile_weighted_logits_sm90_gqa_q16_for_signature(
        sparse_utils.device_cache_key(q_runtime.device),
        int(kv_heads),
        bool(out_dynamic_rows),
        w_signature,
        out_signature,
    )

    def compiled(mQ, mK, mW, mO, seq_q, seq_k, kv_heads):
        return compiled_kernel(
            mQ,
            mK,
            mW,
            mO,
            seq_q,
            seq_k,
            kv_heads,
            Int32(grid_q),
        )

    return compiled, (mQ, mK, mW, mO), out_flat, (q_runtime, k_runtime, w_mat)


def _same_tensor_view(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    return (
        lhs.device == rhs.device
        and lhs.dtype == rhs.dtype
        and tuple(lhs.shape) == tuple(rhs.shape)
        and tuple(lhs.stride()) == tuple(rhs.stride())
        and int(lhs.storage_offset()) == int(rhs.storage_offset())
        and int(lhs.data_ptr()) == int(rhs.data_ptr())
    )


def _launch_gqa_logits(
    q_runtime: torch.Tensor,
    k_runtime: torch.Tensor,
    w_mat: torch.Tensor,
    out_flat: torch.Tensor,
    *,
    seq_q: int,
    seq_k: int,
    kv_heads: int,
    grid_q_multiplier: int,
    stream: Optional[object] = None,
) -> tuple[torch.Tensor, ...]:
    if stream is not None:
        raise ValueError("GQA logits use the TVM-FFI environment stream")
    q_heads_per_kv = int(w_mat.shape[2])
    _gqa_block_q_for_heads(q_heads_per_kv)
    compile_fn = (
        _compile_logits_sm90_gqa
        if q_heads_per_kv == Q_HEADS_PER_KV
        else _compile_logits_sm90_gqa_q16
    )
    compiled, (mQ, mK, mW, mO), out_flat_impl, backing = compile_fn(
        q_runtime,
        k_runtime,
        w_mat,
        out_flat,
        seq_q=seq_q,
        seq_k=seq_k,
        kv_heads=kv_heads,
        stream=None,
        grid_q_multiplier=grid_q_multiplier,
    )
    if not _same_tensor_view(out_flat_impl, out_flat):
        raise RuntimeError("GQA logits compile did not reuse the requested out_flat tensor")
    compiled(mQ, mK, mW, mO, Int32(seq_q), Int32(seq_k), Int32(kv_heads))
    out_flat._fp8_gqa_backing = backing
    return backing


def _launch_prepare_gqa_logits_inputs(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    q_runtime: torch.Tensor,
    k_runtime: torch.Tensor,
    w_mat: torch.Tensor,
    *,
    stream: Optional[object] = None,
) -> tuple[torch.Tensor, ...]:
    if stream is not None:
        raise ValueError("GQA logits input packing uses the TVM-FFI environment stream")
    q_heads_per_kv = int(weights.shape[2])
    _gqa_block_q_for_heads(q_heads_per_kv)
    compile_prepare_fn = (
        _compile_prepare_gqa_logits_inputs
        if q_heads_per_kv == Q_HEADS_PER_KV
        else _compile_prepare_gqa_logits_inputs_q16
    )
    compiled = compile_prepare_fn(
        index_q,
        index_k,
        weights,
        q_runtime,
        k_runtime,
        w_mat,
    )
    compiled(
        index_q,
        index_k,
        weights,
        q_runtime,
        k_runtime,
        w_mat,
        Int32((q_runtime.numel() + k_runtime.numel() + w_mat.numel() + GQA_PREPARE_THREADS - 1) // GQA_PREPARE_THREADS),
        Int32(q_runtime.numel()),
        Int32(k_runtime.numel()),
        Int32(w_mat.numel()),
        Int32(index_q.shape[0]),
        Int32(index_k.shape[0]),
        Int32(index_q.shape[1]),
    )
    backing = (index_q, index_k, weights, q_runtime, k_runtime, w_mat)
    return backing


@torch.library.custom_op(
    "optimus_cutedsl::prepare_gqa_logits_inputs_sm90",
    mutates_args=("q_runtime", "k_runtime", "w_mat"),
    device_types="cuda",
)
def _prepare_gqa_logits_inputs_sm90(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    q_runtime: torch.Tensor,
    k_runtime: torch.Tensor,
    w_mat: torch.Tensor,
) -> None:
    _launch_prepare_gqa_logits_inputs(
        index_q,
        index_k,
        weights,
        q_runtime,
        k_runtime,
        w_mat,
        stream=None,
    )


@torch.library.custom_op(
    "optimus_cutedsl::weighted_relu_logits_sum_sm90_gqa_out",
    mutates_args=("out_flat",),
    device_types="cuda",
)
def _weighted_relu_logits_sum_sm90_gqa_out(
    q_runtime: torch.Tensor,
    k_runtime: torch.Tensor,
    w_mat: torch.Tensor,
    out_flat: torch.Tensor,
    seq_q: int,
    seq_k: int,
    kv_heads: int,
    grid_q_multiplier: int,
) -> None:
    _launch_gqa_logits(
        q_runtime,
        k_runtime,
        w_mat,
        out_flat,
        seq_q=seq_q,
        seq_k=seq_k,
        kv_heads=kv_heads,
        grid_q_multiplier=grid_q_multiplier,
        stream=None,
    )


def weighted_relu_logits_sum_sm90_gqa(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    *,
    tau: float,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if index_q.dim() != 4:
        raise ValueError("index_q must be rank-4 [seq_q, kv_head, q_heads_per_kv, head_dim]")
    if weights.dim() != 3:
        raise ValueError("weights must be rank-3 [seq_q, kv_head, q_heads_per_kv]")
    if index_k.dim() != 3:
        raise ValueError("index_k must be rank-3 [seq_k, kv_head, head_dim]")
    seq_q, num_kv_heads, q_heads_per_kv, head_dim_q = [int(v) for v in index_q.shape]
    seq_k, num_kv_heads_k, head_dim_k = [int(v) for v in index_k.shape]
    gqa_block_q = _gqa_block_q_for_heads(q_heads_per_kv)
    if head_dim_q != HEAD_DIM or head_dim_k != HEAD_DIM:
        raise ValueError(
            f"GQA logits kernel requires head_dim={HEAD_DIM}, "
            f"got q={head_dim_q}, k={head_dim_k}"
        )
    if tuple(weights.shape) != (seq_q, num_kv_heads, q_heads_per_kv):
        raise ValueError("weights shape must match index_q[:3]")
    if num_kv_heads_k != num_kv_heads:
        raise ValueError("kv_head mismatch between index_q and index_k")
    if index_q.device != index_k.device or index_q.device != weights.device:
        raise ValueError("index_q/weights/index_k must be on the same device")
    if index_q.device.type != "cuda":
        raise RuntimeError("weighted_relu_logits_sum_sm90_gqa requires CUDA tensors")
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("out_dtype must be float16, bfloat16, or float32")
    if not index_q.is_contiguous() or not index_k.is_contiguous() or not weights.is_contiguous():
        raise ValueError(
            "weighted_relu_logits_sum_sm90_gqa requires contiguous index_q/index_k/weights, "
            f"got q_stride={tuple(index_q.stride())}, "
            f"k_stride={tuple(index_k.stride())}, "
            f"weights_stride={tuple(weights.stride())}"
        )
    if abs(float(tau) - 1.0) > 1e-6:
        raise ValueError("GQA logits kernel requires tau=1.0")
    if weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            "weighted_relu_logits_sum_sm90_gqa requires fp16/bf16/fp32 weights, "
            f"got {weights.dtype}"
        )
    if seq_q <= 0 or seq_k <= 0:
        raise ValueError("weighted_relu_logits_sum_sm90_gqa requires non-empty q/k")
    if seq_q % gqa_block_q != 0:
        raise ValueError(f"seq_q must be divisible by {gqa_block_q}; no implicit padding is performed")
    if seq_k % GQA_BLOCK_KV != 0:
        raise ValueError(f"seq_k must be divisible by {GQA_BLOCK_KV}; no implicit padding is performed")

    q_runtime = torch.empty(
        (num_kv_heads * seq_q * q_heads_per_kv, HEAD_DIM),
        device=index_q.device,
        dtype=torch.float8_e4m3fn,
    )
    k_runtime = torch.empty(
        (num_kv_heads * seq_k, HEAD_DIM),
        device=index_q.device,
        dtype=torch.float8_e4m3fn,
    )
    q_runtime_bits = q_runtime.view(torch.uint8)
    k_runtime_bits = k_runtime.view(torch.uint8)
    kernel_weights = torch.empty(
        (num_kv_heads, seq_q, q_heads_per_kv),
        device=index_q.device,
        dtype=weights.dtype,
    )
    out_flat = torch.empty(
        (num_kv_heads * seq_q, seq_k),
        device=index_q.device,
        dtype=out_dtype,
    )
    _prepare_gqa_logits_inputs_sm90(
        index_q,
        index_k,
        weights,
        q_runtime_bits,
        k_runtime_bits,
        kernel_weights,
    )
    _weighted_relu_logits_sum_sm90_gqa_out(
        q_runtime,
        k_runtime,
        kernel_weights,
        out_flat,
        seq_q,
        seq_k,
        num_kv_heads,
        2,
    )
    backing = out_flat._fp8_gqa_backing
    out_flat._fp8_gqa_backing = backing
    return out_flat


__all__ = ["weighted_relu_logits_sum_sm90_gqa"]
