# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from typing import Optional

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint8

from vllm.models.step4.nvidia.ops.cute_dsl.cutedsl_compile_cache import cached_compile_function
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.kernel_utils import cute_utils as utils
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.kernel_utils.fp_compat import cvt_f32_to_e4m3


_HEAD_DIM = 64
_THREADS_PER_WARP = 32
_DIMS_PER_LANE = 2
_WARPS_PER_BLOCK = 8
_THREADS_PER_BLOCK = _THREADS_PER_WARP * _WARPS_PER_BLOCK
_GQA_SUMMARIES_PER_PAGE = 2
_REGION_SIZE = 8


def _shape_with_dynamic_dim(
    shape: tuple[int, ...],
    dynamic_shape_dim: int | None,
    dynamic_shape_dims: tuple[int, ...] | None,
    divisibility: int,
) -> tuple[object, ...]:
    if dynamic_shape_dim is not None and dynamic_shape_dims is not None:
        raise ValueError("pass dynamic_shape_dim or dynamic_shape_dims, not both")
    dynamic_dims = (
        {int(dynamic_shape_dim)} if dynamic_shape_dim is not None
        else {int(v) for v in (dynamic_shape_dims or ())}
    )
    if not dynamic_dims:
        return tuple(int(v) for v in shape)
    return tuple(
        cute.sym_int(divisibility=divisibility)
        if idx in dynamic_dims else int(v)
        for idx, v in enumerate(shape)
    )


def _stride_with_dynamic_dims(
    stride: tuple[int, ...],
    dynamic_stride_dims: tuple[int, ...],
    divisibility: int,
) -> tuple[object, ...]:
    dynamic_dims = {int(v) for v in dynamic_stride_dims}
    return tuple(
        cute.sym_int(divisibility=divisibility)
        if idx in dynamic_dims else int(v)
        for idx, v in enumerate(stride)
    )


def _make_fake_tensor_from_signature(
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    alignment: int,
    dynamic_shape_dim: int | None = None,
    dynamic_shape_dims: tuple[int, ...] | None = None,
    dynamic_stride_dims: tuple[int, ...] = (),
    divisibility: int = 1,
) -> cute.Tensor:
    if dynamic_shape_dim is not None and dynamic_shape_dims is not None:
        raise ValueError("pass dynamic_shape_dim or dynamic_shape_dims, not both")
    fake = cute.runtime.make_fake_tensor(
        utils._cutlass_dtype_from_torch_with_metadata(dtype),
        _shape_with_dynamic_dim(
            shape, dynamic_shape_dim, dynamic_shape_dims, divisibility),
        _stride_with_dynamic_dims(stride, dynamic_stride_dims, divisibility),
        assumed_align=int(alignment),
    )
    shape_modes = (
        (int(dynamic_shape_dim),) if dynamic_shape_dim is not None
        else tuple(int(v) for v in (dynamic_shape_dims or ()))
    )
    for shape_mode in shape_modes:
        marked = fake.mark_compact_shape_dynamic(
            mode=shape_mode,
            stride_order=tuple(range(len(shape))),
            divisibility=int(divisibility),
        )
        fake = fake if marked is None else marked
    return fake


def _require_contiguous_aligned_i32_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must be torch.int32, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tuple(tensor.stride())}")
    if int(tensor.numel()) > 0 and (int(tensor.data_ptr()) & 0xF) != 0:
        raise ValueError(
            f"{name} data_ptr must be 16-byte aligned for CuTe memref descriptor, "
            f"got data_ptr=0x{int(tensor.data_ptr()):x}, storage_offset={int(tensor.storage_offset())}"
        )
    return tensor


@cute.jit
def _find_req_idx_from_region_bounds(
    mReqRegionBounds: cute.Tensor,
    out_region_idx: Int32,
    num_reqs: Int32,
) -> Int32:
    lo = Int32(0)
    hi = num_reqs
    while lo < hi:
        mid = (lo + hi) // Int32(2)
        if out_region_idx < mReqRegionBounds[mid + Int32(1)]:
            hi = mid
        else:
            lo = mid + Int32(1)
    return lo


@cute.kernel
def _paged_summary_pack_kernel_gqa(
    mSumCache: cute.Tensor,
    mCountCache: cute.Tensor,
    mBlockTable: cute.Tensor,
    mReqRegionBounds: cute.Tensor,
    mTotalValidRegions: cute.Tensor,
    mOut: cute.Tensor,
    total_valid_regions_cap: Int32,
    num_reqs: Int32,
    out_heads: Int32,
    pack_heads: Int32,
    count_heads: Int32,
    kernel_pages_per_vllm_page: Int32,
    warps_per_block: cutlass.Constexpr[int],
):
    cta_idx, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()

    out_row = cta_idx * Int32(warps_per_block) + warp_idx
    total_rows = total_valid_regions_cap * out_heads
    if out_row < total_rows:
        out_region_idx = out_row // out_heads
        out_head_idx = out_row - out_region_idx * out_heads
        kv_head_idx = out_head_idx // pack_heads
        pack_head_idx = out_head_idx - kv_head_idx * pack_heads
        total_valid_regions = mTotalValidRegions[Int32(0)]
        is_valid = out_region_idx < total_valid_regions

        col0 = lane_idx * Int32(_DIMS_PER_LANE)
        col1 = col0 + Int32(1)
        proxy_col0 = pack_head_idx * Int32(_HEAD_DIM) + col0
        proxy_col1 = pack_head_idx * Int32(_HEAD_DIM) + col1
        if is_valid:
            req_idx = _find_req_idx_from_region_bounds(
                mReqRegionBounds,
                out_region_idx,
                num_reqs,
            )
            req_region_base = mReqRegionBounds[req_idx]
            local_region_idx = out_region_idx - req_region_base
            page_col = local_region_idx // Int32(_GQA_SUMMARIES_PER_PAGE)
            slot_col = local_region_idx - page_col * Int32(_GQA_SUMMARIES_PER_PAGE)
            phys_kernel_page = mBlockTable[req_idx, page_col]
            phys_page = phys_kernel_page // kernel_pages_per_vllm_page
            sub_page = phys_kernel_page - phys_page * kernel_pages_per_vllm_page
            summary_slot = sub_page * Int32(_GQA_SUMMARIES_PER_PAGE) + slot_col
            count_head_idx = Int32(0)
            if count_heads != Int32(1):
                count_head_idx = kv_head_idx

            denom = cute.arch.fmax(
                Float32(mCountCache[phys_page, summary_slot, count_head_idx]),
                Float32(1.0),
            )
            sum0 = Float32(mSumCache[phys_page, summary_slot, kv_head_idx, proxy_col0])
            sum1 = Float32(mSumCache[phys_page, summary_slot, kv_head_idx, proxy_col1])
            mean0 = sum0 / denom
            mean1 = sum1 / denom
            if cutlass.const_expr(mOut.element_type == Uint8):
                mOut[out_region_idx, out_head_idx, col0] = cvt_f32_to_e4m3(mean0).to(Uint8)
                mOut[out_region_idx, out_head_idx, col1] = cvt_f32_to_e4m3(mean1).to(Uint8)
            else:
                mOut[out_region_idx, out_head_idx, col0] = mOut.element_type(mean0)
                mOut[out_region_idx, out_head_idx, col1] = mOut.element_type(mean1)
        else:
            if cutlass.const_expr(mOut.element_type == Uint8):
                mOut[out_region_idx, out_head_idx, col0] = Uint8(0)
                mOut[out_region_idx, out_head_idx, col1] = Uint8(0)
            else:
                mOut[out_region_idx, out_head_idx, col0] = mOut.element_type(Float32(0.0))
                mOut[out_region_idx, out_head_idx, col1] = mOut.element_type(Float32(0.0))


def _make_launch_paged_summary_pack_kernel_gqa():
    @cute.jit
    def _launch_paged_summary_pack_kernel_gqa(
        mSumCache: cute.Tensor,
        mCountCache: cute.Tensor,
        mBlockTable: cute.Tensor,
        mReqRegionBounds: cute.Tensor,
        mTotalValidRegions: cute.Tensor,
        mOut: cute.Tensor,
        total_valid_regions_cap: Int32,
        num_reqs: Int32,
        out_heads: int,
        pack_heads: int,
        count_heads: int,
        kernel_pages_per_vllm_page: int,
        stream,
    ):
        _paged_summary_pack_kernel_gqa(
            mSumCache,
            mCountCache,
            mBlockTable,
            mReqRegionBounds,
            mTotalValidRegions,
            mOut,
            total_valid_regions_cap,
            num_reqs,
            Int32(out_heads),
            Int32(pack_heads),
            Int32(count_heads),
            Int32(kernel_pages_per_vllm_page),
            warps_per_block=_WARPS_PER_BLOCK,
        ).launch(
            grid=[(total_valid_regions_cap * out_heads + _WARPS_PER_BLOCK - 1) // _WARPS_PER_BLOCK, 1, 1],
            block=[_THREADS_PER_BLOCK, 1, 1],
            stream=stream,
        )

    return _launch_paged_summary_pack_kernel_gqa


@cached_compile_function
def _get_compiled_gqa_kernel_for_shape(
    sum_shape: tuple[int, ...],
    sum_stride: tuple[int, ...],
    count_shape: tuple[int, ...],
    count_stride: tuple[int, ...],
    block_table_shape: tuple[int, ...],
    req_region_bounds_shape: tuple[int, ...],
    total_valid_regions_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    out_dtype: torch.dtype,
    device_key: tuple[str, int | None],
    total_valid_regions_cap: int,
) -> cute.JitFunction:
    launch_kernel = _make_launch_paged_summary_pack_kernel_gqa()
    mSumCache = _make_fake_tensor_from_signature(
        dtype=torch.float32,
        shape=sum_shape,
        stride=sum_stride,
        alignment=16,
    )
    mCountCache = _make_fake_tensor_from_signature(
        dtype=torch.float32,
        shape=count_shape,
        stride=count_stride,
        alignment=16,
    )
    mBlockTable = _make_fake_tensor_from_signature(
        dtype=torch.int32,
        shape=block_table_shape,
        stride=(int(block_table_shape[1]), 1),
        alignment=16,
        dynamic_shape_dims=(0, 1),
        dynamic_stride_dims=(0,),
    )
    mReqRegionBounds = _make_fake_tensor_from_signature(
        dtype=torch.int32,
        shape=req_region_bounds_shape,
        stride=(1,),
        alignment=16,
        dynamic_shape_dim=0,
    )
    mTotalValidRegions = _make_fake_tensor_from_signature(
        dtype=torch.int32,
        shape=total_valid_regions_shape,
        stride=(1,),
        alignment=16,
        dynamic_shape_dim=0,
    )
    mOut = _make_fake_tensor_from_signature(
        dtype=out_dtype,
        shape=out_shape,
        stride=(int(out_shape[1]) * int(out_shape[2]), int(out_shape[2]), 1),
        alignment=16,
        dynamic_shape_dim=0,
        divisibility=1,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    proxy_dim = int(sum_shape[3])
    pack_heads = proxy_dim // _HEAD_DIM
    kernel_pages_per_vllm_page = int(sum_shape[1]) // _GQA_SUMMARIES_PER_PAGE
    return cute.compile(
        launch_kernel,
        mSumCache,
        mCountCache,
        mBlockTable,
        mReqRegionBounds,
        mTotalValidRegions,
        mOut,
        Int32(1),
        Int32(1),
        int(out_shape[1]),
        int(pack_heads),
        int(count_shape[2]),
        int(kernel_pages_per_vllm_page),
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


def _get_compiled_gqa_kernel(
    *,
    sum_flat: torch.Tensor,
    count_flat: torch.Tensor,
    block_table: torch.Tensor,
    req_region_bounds: torch.Tensor,
    total_valid_regions_i32: torch.Tensor,
    out: torch.Tensor,
    total_valid_regions_cap: int,
) -> cute.JitFunction:
    # Batch, request-region bounds, and output region capacity are runtime
    # values. Keep them out of the CuTeDSL cache key while retaining static
    # head/layout configuration in the compiled artifact.
    dynamic_block_table_shape = (1, _GQA_SUMMARIES_PER_PAGE)
    dynamic_req_bounds_shape = (1,)
    dynamic_out_shape = (1, int(out.shape[1]), int(out.shape[2]))
    return _get_compiled_gqa_kernel_for_shape(
        tuple(int(v) for v in sum_flat.shape),
        tuple(int(v) for v in sum_flat.stride()),
        tuple(int(v) for v in count_flat.shape),
        tuple(int(v) for v in count_flat.stride()),
        dynamic_block_table_shape,
        dynamic_req_bounds_shape,
        tuple(int(v) for v in total_valid_regions_i32.shape),
        dynamic_out_shape,
        out.dtype,
        utils.device_cache_key(out.device),
        1,
    )


def _pack_paged_summary_mean_sm90_gqa_impl(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    block_table: torch.Tensor,
    req_region_bounds: torch.Tensor,
    total_valid_regions_i32: torch.Tensor,
    out: torch.Tensor,
    stream=None,
) -> None:
    if stream is not None:
        raise ValueError(
            "paged summary pack uses the TVM-FFI environment stream"
        )
    num_pages, summaries_per_page, kv_heads, proxy_dim = [int(v) for v in sum_cache.shape]
    pack_heads = proxy_dim // _HEAD_DIM
    out_heads = kv_heads * pack_heads
    total_valid_regions_cap_i = int(out.shape[0])

    if out.dim() != 3:
        raise ValueError(f"out must be 3D, got shape={tuple(out.shape)}")
    if tuple(out.shape[1:]) != (out_heads, _HEAD_DIM):
        raise ValueError(
            "out shape mismatch: "
            f"got {tuple(out.shape)}, expected ({total_valid_regions_cap_i}, {out_heads}, {_HEAD_DIM})"
        )
    if out.device != sum_cache.device:
        raise ValueError("out must be on the same device as sum_cache")
    if total_valid_regions_cap_i <= 0:
        return
    # Note(wangbojun/codex): The vLLM DSA runtime already prepares these pack
    # metadata tensors as device-resident int32 views. Reuse them directly when
    # possible so the driver launch does not depend on short-lived call-local
    # staging buffers.
    req_region_bounds_i32 = _require_contiguous_aligned_i32_tensor(
        req_region_bounds, name="req_region_bounds")
    block_table_i32 = _require_contiguous_aligned_i32_tensor(block_table, name="block_table")
    # Note(wangbojun/codex): Keep the dynamic valid-region count on device so
    # this custom op remains CUDA-graph-safe; a host scalar read is illegal
    # while the stream is capturing.
    if int(total_valid_regions_i32.numel()) != 1:
        raise ValueError(
            "total_valid_regions_i32 must contain exactly one int32 scalar, got "
            f"shape={tuple(total_valid_regions_i32.shape)}"
        )
    total_valid_regions_i32 = _require_contiguous_aligned_i32_tensor(
        total_valid_regions_i32.reshape(1),
        name="total_valid_regions_i32",
    )
    if count_cache.dim() == 2:
        count_for_kernel = count_cache.view(num_pages, summaries_per_page, 1)
    else:
        count_for_kernel = count_cache
    compiled = _get_compiled_gqa_kernel(
        sum_flat=sum_cache,
        count_flat=count_for_kernel,
        block_table=block_table_i32,
        req_region_bounds=req_region_bounds_i32,
        total_valid_regions_i32=total_valid_regions_i32,
        out=out,
        total_valid_regions_cap=total_valid_regions_cap_i,
    )
    compiled(
        sum_cache,
        count_for_kernel,
        block_table_i32,
        req_region_bounds_i32,
        total_valid_regions_i32,
        out,
        total_valid_regions_cap_i,
        int(block_table_i32.shape[0]),
        out_heads,
        pack_heads,
        int(count_for_kernel.shape[2]),
        summaries_per_page // _GQA_SUMMARIES_PER_PAGE,
    )


@torch.library.custom_op(
    "optimus_cutedsl::pack_paged_summary_mean_sm90_gqa_out",
    mutates_args=("out",),
    device_types="cuda",
)
def _pack_paged_summary_mean_sm90_gqa_out(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    block_table: torch.Tensor,
    req_region_bounds: torch.Tensor,
    total_valid_regions_i32: torch.Tensor,
    out: torch.Tensor,
) -> None:
    # Keep the EnvStream launch opaque to torch.compile/CUDA Graph.
    _pack_paged_summary_mean_sm90_gqa_impl(
        sum_cache,
        count_cache,
        block_table,
        req_region_bounds,
        total_valid_regions_i32,
        out,
        stream=None,
    )


@torch.library.custom_op(
    "optimus_cutedsl::pack_paged_summary_mean_sm90_gqa_functional",
    mutates_args=(),
    device_types="cuda",
)
def _pack_paged_summary_mean_sm90_gqa_functional(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    block_table: torch.Tensor,
    req_region_bounds: torch.Tensor,
    total_valid_regions_i32: torch.Tensor,
    total_valid_regions_cap: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    # Note(wangbojun/codex): Keep allocation and kernel launch inside one
    # opaque op so cudagraph capture observes the compact summary regions as a
    # true op output instead of a graph-local allocation mutated by a later
    # custom kernel launch.
    out = torch.empty(
        (
            int(total_valid_regions_cap),
            sum_cache.shape[2] * (sum_cache.shape[3] // _HEAD_DIM),
            _HEAD_DIM,
        ),
        device=sum_cache.device,
        dtype=out_dtype,
    )
    _pack_paged_summary_mean_sm90_gqa_impl(
        sum_cache,
        count_cache,
        block_table,
        req_region_bounds,
        total_valid_regions_i32,
        out,
        stream=None,
    )
    return out


@_pack_paged_summary_mean_sm90_gqa_functional.register_fake
def _pack_paged_summary_mean_sm90_gqa_functional_fake(
    sum_cache: torch.Tensor,
    _count_cache: torch.Tensor,
    _block_table: torch.Tensor,
    _req_region_bounds: torch.Tensor,
    _total_valid_regions_i32: torch.Tensor,
    total_valid_regions_cap: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(
        (
            int(total_valid_regions_cap),
            sum_cache.shape[2] * (sum_cache.shape[3] // _HEAD_DIM),
            _HEAD_DIM,
        ),
        device=sum_cache.device,
        dtype=out_dtype,
    )


def _cu_seq_lens_to_region_bounds(cu_seq_lens: torch.Tensor) -> torch.Tensor:
    if cu_seq_lens.dtype != torch.int32:
        raise ValueError(f"cu_seq_lens must be torch.int32, got {cu_seq_lens.dtype}")
    if cu_seq_lens.device.type != "cuda":
        raise RuntimeError("cu_seq_lens must be a CUDA tensor")
    if cu_seq_lens.dim() != 1:
        raise ValueError(f"cu_seq_lens must be rank-1 [num_reqs + 1], got {tuple(cu_seq_lens.shape)}")
    if not cu_seq_lens.is_contiguous():
        raise ValueError(f"cu_seq_lens must be contiguous, got stride={tuple(cu_seq_lens.stride())}")
    if int(cu_seq_lens.numel()) < 2:
        raise ValueError(f"cu_seq_lens must contain at least [0, total], got shape={tuple(cu_seq_lens.shape)}")

    req_token_lens = torch.sub(cu_seq_lens[1:], cu_seq_lens[:-1])
    req_region_lens = torch.empty_like(req_token_lens)
    torch.add(req_token_lens, _REGION_SIZE - 1, out=req_region_lens)
    torch.div(
        req_region_lens,
        _REGION_SIZE,
        rounding_mode="floor",
        out=req_region_lens,
    )
    req_region_bounds = torch.empty_like(cu_seq_lens)
    req_region_bounds.zero_()
    torch.cumsum(req_region_lens, dim=0, out=req_region_bounds[1:])
    return req_region_bounds


def pack_paged_summary_mean_sm90_gqa(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    *,
    total_valid_regions: int,
    total_valid_regions_i32: Optional[torch.Tensor] = None,
    total_valid_regions_cap: Optional[int] = None,
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    req_region_bounds: Optional[torch.Tensor] = None,
    stream=None,
) -> torch.Tensor:
    if sum_cache.dim() != 4:
        raise ValueError(
            "sum_cache must be [num_pages, summaries_per_page, kv_head, head_dim], "
            f"got shape={tuple(sum_cache.shape)}"
        )
    if count_cache.dim() not in (2, 3):
        raise ValueError(
            "count_cache must be [num_pages, summaries_per_page] or "
            "[num_pages, summaries_per_page, kv_head], "
            f"got shape={tuple(count_cache.shape)}"
        )
    if block_table.dim() != 2:
        raise ValueError(
            "block_table must be [num_reqs, max_pages], "
            f"got shape={tuple(block_table.shape)}"
        )
    if cu_seq_lens.dim() != 1:
        raise ValueError(f"cu_seq_lens must be 1D [num_reqs + 1], got shape={tuple(cu_seq_lens.shape)}")
    if int(cu_seq_lens.numel()) != int(block_table.shape[0]) + 1:
        raise ValueError(
            "cu_seq_lens length must match block_table rows + 1: "
            f"{int(cu_seq_lens.numel())} vs {int(block_table.shape[0]) + 1}"
        )
    if sum_cache.device.type != "cuda":
        raise RuntimeError("pack_paged_summary_mean_sm90_gqa requires CUDA tensors")
    if count_cache.device != sum_cache.device or block_table.device != sum_cache.device:
        raise ValueError("sum_cache/count_cache/block_table must be on the same device")
    if cu_seq_lens.device != sum_cache.device:
        raise ValueError("cu_seq_lens must be on the same device as sum_cache")

    num_pages, summaries_per_page, kv_heads, proxy_dim = [int(v) for v in sum_cache.shape]
    if summaries_per_page % _GQA_SUMMARIES_PER_PAGE != 0:
        raise ValueError(
            "pack_paged_summary_mean_sm90_gqa requires summaries_per_page "
            f"to be divisible by {_GQA_SUMMARIES_PER_PAGE}, got {summaries_per_page}"
        )
    if proxy_dim % _HEAD_DIM != 0:
        raise ValueError(
            "pack_paged_summary_mean_sm90_gqa requires proxy_dim to be "
            f"divisible by {_HEAD_DIM}, got {proxy_dim}"
        )
    pack_heads = proxy_dim // _HEAD_DIM
    out_heads = kv_heads * pack_heads
    if count_cache.dim() == 2:
        expected_count_shapes = ((num_pages, summaries_per_page),)
    else:
        expected_count_shapes = (
            (num_pages, summaries_per_page, 1),
            (num_pages, summaries_per_page, kv_heads),
        )
    if tuple(count_cache.shape) not in expected_count_shapes:
        raise ValueError(
            "count_cache shape mismatch: "
            f"got {tuple(count_cache.shape)}, expected one of {expected_count_shapes}"
        )
    if cu_seq_lens.dtype != torch.int32:
        raise ValueError(f"cu_seq_lens must be int32, got {cu_seq_lens.dtype}")
    if block_table.dtype != torch.int32:
        raise ValueError(f"block_table must be int32, got {block_table.dtype}")
    if sum_cache.dtype != torch.float32 or count_cache.dtype != torch.float32:
        raise ValueError("sum_cache and count_cache must be float32")

    total_valid_regions = int(total_valid_regions)
    if total_valid_regions <= 0:
        return torch.empty(
            (0, out_heads, _HEAD_DIM),
            device=sum_cache.device,
            dtype=out_dtype,
        )
    total_valid_regions_cap_i = int(
        total_valid_regions if total_valid_regions_cap is None
        else total_valid_regions_cap)
    if total_valid_regions_cap_i < total_valid_regions:
        raise ValueError(
            "total_valid_regions_cap must be >= total_valid_regions, got "
            f"cap={total_valid_regions_cap_i}, total={total_valid_regions}"
        )
    if total_valid_regions_i32 is None:
        raise ValueError(
            "pack_paged_summary_mean_sm90_gqa requires caller-provided "
            "total_valid_regions_i32; creating a CUDA scalar in the wrapper "
            "would add a hidden hot-path torch op"
        )
    if total_valid_regions_i32.device != sum_cache.device:
        raise ValueError(
            "total_valid_regions_i32 must be on the same device as sum_cache")
    if total_valid_regions_i32.dtype != torch.int32:
        raise ValueError(
            f"total_valid_regions_i32 must be torch.int32, got {total_valid_regions_i32.dtype}"
        )
    if int(total_valid_regions_i32.numel()) != 1:
        raise ValueError(
            "total_valid_regions_i32 must contain exactly one scalar, got "
            f"shape={tuple(total_valid_regions_i32.shape)}"
        )
    total_valid_regions_i32 = total_valid_regions_i32.reshape(1)
    if req_region_bounds is None:
        req_region_bounds = _cu_seq_lens_to_region_bounds(cu_seq_lens)
    else:
        if req_region_bounds.device != sum_cache.device:
            raise ValueError(
                "req_region_bounds must be on the same device as sum_cache")
        if req_region_bounds.dtype != torch.int32:
            raise ValueError(
                f"req_region_bounds must be torch.int32, got {req_region_bounds.dtype}")
        if req_region_bounds.dim() != 1:
            raise ValueError(
                "req_region_bounds must be 1D [num_reqs + 1], got "
                f"shape={tuple(req_region_bounds.shape)}")
        if int(req_region_bounds.numel()) != int(block_table.shape[0]) + 1:
            raise ValueError(
                "req_region_bounds length must match block_table rows + 1: "
                f"{int(req_region_bounds.numel())} vs {int(block_table.shape[0]) + 1}")

    if out is not None:
        if out.device != sum_cache.device:
            raise ValueError("out must be on the same device as sum_cache")
        if out.dtype != out_dtype:
            raise ValueError(
                f"out dtype mismatch: got {out.dtype}, expected {out_dtype}"
            )
        expected_shape = (total_valid_regions_cap_i, out_heads, _HEAD_DIM)
        if tuple(int(v) for v in out.shape) != expected_shape:
            raise ValueError(
                "out shape mismatch: "
                f"got {tuple(int(v) for v in out.shape)}, expected {expected_shape}"
            )
    if stream is None:
        if out is None:
            out = _pack_paged_summary_mean_sm90_gqa_functional(
                sum_cache,
                count_cache,
                block_table,
                req_region_bounds,
                total_valid_regions_i32,
                total_valid_regions_cap_i,
                out_dtype,
            )
        else:
            _pack_paged_summary_mean_sm90_gqa_out(
                sum_cache,
                count_cache,
                block_table,
                req_region_bounds,
                total_valid_regions_i32,
                out,
            )
    else:
        if out is None:
            out = torch.empty(
                (total_valid_regions_cap_i, out_heads, _HEAD_DIM),
                device=sum_cache.device,
                dtype=out_dtype,
            )
        _pack_paged_summary_mean_sm90_gqa_impl(
            sum_cache,
            count_cache,
            block_table,
            req_region_bounds,
            total_valid_regions_i32,
            out,
            stream=stream,
        )
    return out


__all__ = [
    "pack_paged_summary_mean_sm90_gqa",
]
