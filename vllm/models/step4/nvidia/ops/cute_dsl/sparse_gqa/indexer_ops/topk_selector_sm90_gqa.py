# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from typing import Optional

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float16, Float32, Int16, Int32, Int64
from cutlass._mlir.dialects import llvm, nvvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.models.step4.nvidia.ops.cute_dsl.cutedsl_compile_cache import cached_compile_function
from vllm.models.step4.nvidia.ops.cute_dsl.indexer_ops.topk_selector_sm90 import (
    cutedsl_topk_selector_sm90_multi_cta,
)
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.indexer_ops.decode_sparse_meta_step3p5 import (
    convert_region_block_topk_to_sparse_meta_step3p5,
)
from vllm.models.step4.nvidia.ops.cute_dsl.utils import elem_pointer
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.kernel_utils import cute_utils as utils


_THREADS_PER_BLOCK = 1024


_RADIX = 256
_F32_TAIL_ROUNDS = 3
_F16_TAIL_ROUNDS = 1
_DEFAULT_MAX_CANDIDATES = 4096
# Below the fused single-CTA candidate-buffer kernel's real smem capacity
# (~28.9k candidates on H200) the launch would otherwise ValueError / silently
# cap. Route long inputs through the multi-CTA streaming selector (no
# O(num_regions) smem buffer) well before that limit.
_MULTI_CTA_DECODE_THRESHOLD = 16_384
_REGION_TOKENS = 8
_REGION_VALID_SHIFT = 24
_RAW_SELECTOR_DUMMY_BLOCK_TABLES: dict[tuple[str, int | None], torch.Tensor] = {}


def _is_torch_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is None:
        return False
    is_compiling = getattr(compiler, "is_compiling", None)
    if is_compiling is None:
        return False
    return bool(is_compiling())


def _make_fake_tensor_with_dynamic_dims(
    tensor: torch.Tensor,
    *,
    alignment: int,
    dynamic_shape_dims: tuple[int, ...],
    dynamic_stride_dims: tuple[int, ...] = (),
):
    """Build one descriptor whose request/sequence dimensions stay runtime.

    The selector launch receives batch, sequence length, and request count as
    scalar runtime arguments. Mark every corresponding tensor dimension here so
    the CuTe artifact is shared across graph captures with different shapes.
    """
    dims = tuple(sorted({int(dim) for dim in dynamic_shape_dims}))
    if not dims:
        return utils.make_fake_tensor_like_with_dynamic_dim(
            tensor,
            alignment=int(alignment),
            dynamic_stride_dims=tuple(int(dim) for dim in dynamic_stride_dims),
        )
    fake = utils.make_fake_tensor_like_with_dynamic_dim(
        tensor,
        alignment=int(alignment),
        dynamic_shape_dim=dims[0],
        dynamic_stride_dims=tuple(int(dim) for dim in dynamic_stride_dims),
    )
    stride_order = tensor.dim_order()
    for dim in dims[1:]:
        marked = fake.mark_compact_shape_dynamic(
            mode=int(dim),
            stride_order=stride_order,
            divisibility=1,
        )
        fake = fake if marked is None else marked
    return fake


@dsl_user_op
def _bitcast_f32_to_i32(x: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _bitcast_f16_to_i16(x: Float16, *, loc=None, ip=None) -> Int16:
    vec_f16 = vector.from_elements(
        T.vector(1, T.f16()),
        (Float16(x).ir_value(loc=loc, ip=ip),),
        loc=loc,
        ip=ip,
    )
    vec_i16 = vector.bitcast(T.vector(1, T.i16()), vec_f16)
    bits = vector.extract(vec_i16, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    return Int16(bits)


@dsl_user_op
def _atomic_add_i32(ptr: cute.Pointer, val: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        nvvm.atomicrmw(
            op=nvvm.AtomicOpKind.ADD,
            ptr=ptr.llvm_ptr,
            a=Int32(val).ir_value(loc=loc, ip=ip),
        )
    )


@cute.jit
def _ordered_u32_from_f32(x: Float32) -> Int32:
    bits = _bitcast_f32_to_i32(x)
    neg_bits = ~bits
    pos_bits = bits | Int32(0x80000000)
    return cutlass.select_(x < Float32(0.0), neg_bits, pos_bits)


@cute.jit
def _radix_byte_from_f32(x: Float32, *, shift: cutlass.Constexpr[int]) -> Int32:
    ordered = _ordered_u32_from_f32(x)
    return (ordered >> Int32(shift)) & Int32(0xFF)


@cute.jit
def _radix_byte_from_nonnegative_f16(x: Float16, *, shift: cutlass.Constexpr[int]) -> Int32:
    bits = Int32(_bitcast_f16_to_i16(x)) & Int32(0xFFFF)
    return (bits >> Int32(shift)) & Int32(0xFF)


@cute.jit
def _radix_byte_from_ordered_f16(x: Float16, *, shift: cutlass.Constexpr[int]) -> Int32:
    bits = Int32(_bitcast_f16_to_i16(x)) & Int32(0xFFFF)
    sign = bits & Int32(0x8000)
    ordered = cutlass.select_(sign != Int32(0), (~bits) & Int32(0xFFFF), bits | Int32(0x8000))
    return (ordered >> Int32(shift)) & Int32(0xFF)


@cute.jit
def _hist_prefix_desc_parallel(sHist: cute.Tensor, tx: Int32):
    partial = Int32(0)
    for i in cutlass.range_constexpr(8):
        offset = Int32(1 << i)
        cute.arch.sync_threads()
        if tx < Int32(_RADIX) - offset:
            partial = sHist[tx] + sHist[tx + offset]
        cute.arch.sync_threads()
        if tx < Int32(_RADIX) - offset:
            sHist[tx] = partial
    cute.arch.sync_threads()


@cute.jit
def _hist_select_threshold_parallel(
    sHist: cute.Tensor,
    sRemain: cute.Tensor,
    sThreshold: cute.Tensor,
    tx: Int32,
):
    remain = sRemain[Int32(0)]
    if tx < Int32(_RADIX):
        if sHist[tx] > remain and sHist[tx + Int32(1)] <= remain:
            sThreshold[Int32(0)] = tx
    cute.arch.sync_threads()
    threshold = Int32(0)
    if tx == Int32(0):
        threshold = sThreshold[Int32(0)]
        sRemain[Int32(0)] = remain - sHist[threshold + Int32(1)]
    cute.arch.sync_threads()


@cute.jit
def _pack_prefill_physical_region(
    selected_region: Int32,
    row_idx: Int32,
    mCuSeqLens: cute.Tensor,
    mBlockTable: cute.Tensor,
    seq_q: Int32,
    num_reqs: Int32,
    *,
    use_cu_seqlens: cutlass.Constexpr[bool],
) -> Int32:
    page_col = selected_region // Int32(2)
    page_slot = selected_region - page_col * Int32(2)
    req_idx = Int32(0)
    q_global = row_idx - (row_idx // seq_q) * seq_q
    q_pos = q_global
    if cutlass.const_expr(use_cu_seqlens):
        for req in cutlass.range(num_reqs, unroll=1):
            begin = mCuSeqLens[req]
            end = mCuSeqLens[req + Int32(1)]
            if q_global >= begin and q_global < end:
                req_idx = req
                q_pos = q_global - begin
    phys_page = mBlockTable[req_idx, page_col]
    phys_region = phys_page * Int32(2) + page_slot
    valid_tokens = q_pos + Int32(1) - selected_region * Int32(_REGION_TOKENS)
    if valid_tokens < Int32(0):
        valid_tokens = Int32(0)
    if valid_tokens > Int32(_REGION_TOKENS):
        valid_tokens = Int32(_REGION_TOKENS)
    return phys_region | (valid_tokens << Int32(_REGION_VALID_SHIFT))


def _round_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _next_power_of_2_at_least(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def topk_selector_decode_meta_capacity_class_sm90_gqa(
    score_capacity: int,
    *,
    k: int,
) -> Optional[int]:
    """Return the single-CTA shared-memory artifact class for a score capacity.

    ``None`` denotes the streaming path. The threshold remains an Optimus
    implementation detail so callers do not need to mirror its dispatch rule.
    """
    score_capacity = int(score_capacity)
    k = int(k)
    if score_capacity <= 0:
        raise ValueError(
            f"score_capacity must be > 0, got {score_capacity}"
        )
    if k <= 0 or k > score_capacity:
        raise ValueError(
            "decode-meta capacity classification requires "
            f"0 < k <= score_capacity, got k={k}, score_capacity={score_capacity}"
        )
    if score_capacity >= _MULTI_CTA_DECODE_THRESHOLD:
        return None
    required = max(_DEFAULT_MAX_CANDIDATES, k, score_capacity)
    return _next_power_of_2_at_least(required)


def _selector_required_smem_bytes(max_candidates: int) -> int:
    total = 0
    total = _round_up(total, 128) + (_RADIX + 1) * 4
    total = _round_up(total, 128) + 2 * int(max_candidates) * 4
    total = _round_up(total, 16) + 2 * 4
    total = _round_up(total, 8) + 4
    total = _round_up(total, 8) + 4
    return total


def _selector_max_candidates_cap(device: torch.device) -> int:
    props = torch.cuda.get_device_properties(device)
    max_smem = int(
        getattr(props, "shared_memory_per_block_optin", 0)
        or getattr(props, "shared_memory_per_block", 0)
        or 0
    )
    if max_smem <= 0:
        return _DEFAULT_MAX_CANDIDATES
    if _selector_required_smem_bytes(_DEFAULT_MAX_CANDIDATES) > max_smem:
        return _DEFAULT_MAX_CANDIDATES
    lo = _DEFAULT_MAX_CANDIDATES
    hi = lo
    while _selector_required_smem_bytes(hi) <= max_smem:
        lo = hi
        hi *= 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _selector_required_smem_bytes(mid) <= max_smem:
            lo = mid
        else:
            hi = mid
    return lo


def _selector_max_candidates_for_input(
    input_tensor: torch.Tensor,
    k: int,
    max_candidates: Optional[int] = None,
) -> int:
    # Low-entropy or monotonic scores can put all live regions in one high-order
    # radix bucket.  The refinement pass must retain the whole bucket; otherwise
    # long decode silently chooses from the first max_candidates regions only.
    required = max(int(_DEFAULT_MAX_CANDIDATES), int(k), int(input_tensor.shape[1]))
    explicit = max_candidates is not None
    if explicit:
        required = max(required, int(max_candidates))
    cap = int(_selector_max_candidates_cap(input_tensor.device))
    if explicit and required > cap:
        raise ValueError(
            "top-k selector max_candidates exceeds shared-memory capacity: "
            f"requested={required}, supported<={cap}"
        )
    return min(_next_power_of_2_at_least(required), cap)


def _make_topk_selector_kernel(
    max_candidates: int,
    *,
    use_f16_short_tail: bool,
    input_nonnegative: bool,
    use_row_starts: bool,
    use_cu_seqlens: bool,
    pack_output: bool,
):
    max_cand = int(max_candidates)
    max_cand_i32 = Int32(max_cand)
    use_f16_key = bool(use_f16_short_tail)
    use_f16_nonnegative_key = bool(use_f16_key and input_nonnegative)
    tail_rounds = _F16_TAIL_ROUNDS if use_f16_key else _F32_TAIL_ROUNDS
    first_shift = 8 if use_f16_key else 24

    @cute.kernel
    def _topk_selector_kernel_sm90(
        mInput: cute.Tensor,
        mOutIdx: cute.Tensor,
        mLengths: cute.Tensor,
        mRowStarts: cute.Tensor,
        mCuSeqLens: cute.Tensor,
        mBlockTable: cute.Tensor,
        batch: Int32,
        seq_len: Int32,
        seq_q: Int32,
        num_reqs: Int32,
        topk: Int32,
        length_delta: Int32,
    ):
        bx, _, _ = cute.arch.block_idx()
        tx = cute.arch.thread_idx()[0]
        threshold = Int32(0)

        @cute.struct
        class SharedStorage:
            hist: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, _RADIX + 1], 128]
            cand: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 2 * max_cand], 128
            ]
            cnt: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 2], 16]
            remain: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 8]
            threshold: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 8]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage, 16)
        sHist = storage.hist.get_tensor(cute.make_layout((_RADIX + 1,), stride=(1,)))
        sCand = storage.cand.get_tensor(
            cute.make_layout((2, max_cand), stride=(max_cand, 1))
        )
        sCnt = storage.cnt.get_tensor(cute.make_layout((2,), stride=(1,)))
        sRemain = storage.remain.get_tensor(cute.make_layout((1,), stride=(1,)))
        sThreshold = storage.threshold.get_tensor(cute.make_layout((1,), stride=(1,)))

        if bx < batch:
            num_out_tiles = (topk + Int32(_THREADS_PER_BLOCK - 1)) // Int32(
                _THREADS_PER_BLOCK
            )
            for tile in cutlass.range(num_out_tiles, unroll=1):
                out_pos = tile * Int32(_THREADS_PER_BLOCK) + tx
                if out_pos < topk:
                    mOutIdx[bx, out_pos] = Int32(-1)
            cute.arch.sync_threads()

            start_idx = Int32(0)
            if cutlass.const_expr(use_row_starts):
                start_idx = mRowStarts[bx]
            valid_len = mLengths[bx] + length_delta
            if valid_len < Int32(0):
                valid_len = Int32(0)
            end_idx = start_idx + valid_len
            if end_idx > seq_len:
                end_idx = seq_len
            if start_idx < Int32(0):
                start_idx = Int32(0)
            valid_len = end_idx - start_idx
            if valid_len < Int32(0):
                valid_len = Int32(0)
            threshold = Int32(0)

            for i_chunk in cutlass.range_constexpr(
                (_RADIX + 1 + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
            ):
                i = tx + Int32(i_chunk * _THREADS_PER_BLOCK)
                if i <= Int32(_RADIX):
                    sHist[i] = Int32(0)

            if tx < Int32(2):
                sCnt[tx] = Int32(0)
            if tx == Int32(0):
                sRemain[Int32(0)] = topk
                sThreshold[Int32(0)] = Int32(0)
            cute.arch.sync_threads()

            num_tiles = (valid_len + Int32(_THREADS_PER_BLOCK - 1)) // Int32(_THREADS_PER_BLOCK)
            for tile in cutlass.range(num_tiles, unroll=1):
                input_idx = start_idx + tile * Int32(_THREADS_PER_BLOCK) + tx
                threshold = Int32(0)
                if input_idx < end_idx:
                    if cutlass.const_expr(use_f16_nonnegative_key):
                        bin_id = _radix_byte_from_nonnegative_f16(
                            mInput[bx, input_idx], shift=first_shift)
                    elif cutlass.const_expr(use_f16_key):
                        bin_id = _radix_byte_from_ordered_f16(
                            mInput[bx, input_idx], shift=first_shift)
                    else:
                        bin_id = _radix_byte_from_f32(
                            Float32(mInput[bx, input_idx]),
                            shift=first_shift,
                        )
                    _atomic_add_i32(elem_pointer(sHist, (bin_id,)), Int32(1))
            cute.arch.sync_threads()

            _hist_prefix_desc_parallel(sHist, tx)
            _hist_select_threshold_parallel(sHist, sRemain, sThreshold, tx)

            for tile in cutlass.range(num_tiles, unroll=1):
                input_idx = start_idx + tile * Int32(_THREADS_PER_BLOCK) + tx
                if input_idx < end_idx:
                    if cutlass.const_expr(use_f16_nonnegative_key):
                        bin_id = _radix_byte_from_nonnegative_f16(
                            mInput[bx, input_idx], shift=first_shift)
                    elif cutlass.const_expr(use_f16_key):
                        bin_id = _radix_byte_from_ordered_f16(
                            mInput[bx, input_idx], shift=first_shift)
                    else:
                        bin_id = _radix_byte_from_f32(
                            Float32(mInput[bx, input_idx]),
                            shift=first_shift,
                        )
                    threshold = sThreshold[Int32(0)]
                    if bin_id > threshold:
                        pos_out = _atomic_add_i32(elem_pointer(sHist, (bin_id + Int32(1),)), Int32(1))
                        if pos_out < topk:
                            if cutlass.const_expr(pack_output):
                                mOutIdx[bx, pos_out] = _pack_prefill_physical_region(
                                    input_idx,
                                    bx,
                                    mCuSeqLens,
                                    mBlockTable,
                                    seq_q,
                                    num_reqs,
                                    use_cu_seqlens=use_cu_seqlens,
                                )
                            else:
                                mOutIdx[bx, pos_out] = input_idx
                    elif bin_id == threshold:
                        if sRemain[Int32(0)] > Int32(0):
                            pos_cand = _atomic_add_i32(elem_pointer(sCnt, (Int32(0),)), Int32(1))
                            if pos_cand < max_cand_i32:
                                sCand[Int32(0), pos_cand] = input_idx
            cute.arch.sync_threads()

            for round_i in cutlass.range_constexpr(tail_rounds):
                cur_buf = Int32(round_i & 1)
                nxt_buf = Int32((round_i + 1) & 1)

                if sRemain[Int32(0)] > Int32(0):
                    for i_chunk in cutlass.range_constexpr(
                        (_RADIX + 1 + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
                    ):
                        i = tx + Int32(i_chunk * _THREADS_PER_BLOCK)
                        if i <= Int32(_RADIX):
                            sHist[i] = Int32(0)
                    if tx == Int32(0):
                        sCnt[nxt_buf] = Int32(0)
                    cute.arch.sync_threads()

                    # Note(wangbojun/codex): The counter tracks the full
                    # threshold bucket cardinality, while sCand only stores
                    # max_candidates entries. Re-reading past that shared
                    # buffer corrupts memory on low-entropy/profile inputs.
                    num_input = cutlass.min(sCnt[cur_buf], max_cand_i32)
                    num_tiles_round = (num_input + Int32(_THREADS_PER_BLOCK - 1)) // Int32(
                        _THREADS_PER_BLOCK
                    )
                    for tile in cutlass.range(num_tiles_round, unroll=1):
                        cand_pos = tile * Int32(_THREADS_PER_BLOCK) + tx
                        if cand_pos < num_input:
                            input_idx = sCand[cur_buf, cand_pos]
                            if cutlass.const_expr(use_f16_nonnegative_key):
                                bin_id = _radix_byte_from_nonnegative_f16(
                                    mInput[bx, input_idx], shift=0)
                            elif cutlass.const_expr(use_f16_key):
                                bin_id = _radix_byte_from_ordered_f16(
                                    mInput[bx, input_idx], shift=0)
                            else:
                                bin_id = _radix_byte_from_f32(
                                    Float32(mInput[bx, input_idx]),
                                    shift=16 - round_i * 8,
                                )
                            _atomic_add_i32(elem_pointer(sHist, (bin_id,)), Int32(1))
                    cute.arch.sync_threads()

                    _hist_prefix_desc_parallel(sHist, tx)
                    _hist_select_threshold_parallel(sHist, sRemain, sThreshold, tx)

                    remain_old = sRemain[Int32(0)] + sHist[sThreshold[Int32(0)] + Int32(1)]
                    start_pos = topk - remain_old
                    for tile in cutlass.range(num_tiles_round, unroll=1):
                        cand_pos = tile * Int32(_THREADS_PER_BLOCK) + tx
                        threshold = Int32(0)
                        if cand_pos < num_input:
                            input_idx = sCand[cur_buf, cand_pos]
                            if cutlass.const_expr(use_f16_nonnegative_key):
                                bin_id = _radix_byte_from_nonnegative_f16(
                                    mInput[bx, input_idx], shift=0)
                            elif cutlass.const_expr(use_f16_key):
                                bin_id = _radix_byte_from_ordered_f16(
                                    mInput[bx, input_idx], shift=0)
                            else:
                                bin_id = _radix_byte_from_f32(
                                    Float32(mInput[bx, input_idx]),
                                    shift=16 - round_i * 8,
                                )
                            threshold = sThreshold[Int32(0)]
                            if bin_id > threshold:
                                pos_out = (
                                    _atomic_add_i32(elem_pointer(sHist, (bin_id + Int32(1),)), Int32(1))
                                    + start_pos
                                )
                                if pos_out < topk:
                                    if cutlass.const_expr(pack_output):
                                        mOutIdx[bx, pos_out] = _pack_prefill_physical_region(
                                            input_idx,
                                            bx,
                                            mCuSeqLens,
                                            mBlockTable,
                                            seq_q,
                                            num_reqs,
                                            use_cu_seqlens=use_cu_seqlens,
                                        )
                                    else:
                                        mOutIdx[bx, pos_out] = input_idx
                            elif bin_id == threshold:
                                if round_i == tail_rounds - 1:
                                    pos_tail = (
                                        _atomic_add_i32(
                                            elem_pointer(sHist, (bin_id + Int32(1),)), Int32(1)
                                        )
                                        + start_pos
                                    )
                                    if pos_tail < topk:
                                        if cutlass.const_expr(pack_output):
                                            mOutIdx[bx, pos_tail] = _pack_prefill_physical_region(
                                                input_idx,
                                                bx,
                                                mCuSeqLens,
                                                mBlockTable,
                                                seq_q,
                                                num_reqs,
                                                use_cu_seqlens=use_cu_seqlens,
                                            )
                                        else:
                                            mOutIdx[bx, pos_tail] = input_idx
                                elif sRemain[Int32(0)] > Int32(0):
                                    pos_next = _atomic_add_i32(elem_pointer(sCnt, (nxt_buf,)), Int32(1))
                                    if pos_next < max_cand_i32:
                                        sCand[nxt_buf, pos_next] = input_idx
                    cute.arch.sync_threads()

    return _topk_selector_kernel_sm90


def _make_launch_topk_selector_kernel(
    max_candidates: int,
    *,
    use_f16_short_tail: bool,
    input_nonnegative: bool,
    use_row_starts: bool,
    use_cu_seqlens: bool,
    pack_output: bool,
):
    kernel = _make_topk_selector_kernel(
        max_candidates,
        use_f16_short_tail=use_f16_short_tail,
        input_nonnegative=input_nonnegative,
        use_row_starts=use_row_starts,
        use_cu_seqlens=use_cu_seqlens,
        pack_output=pack_output,
    )

    @cute.jit
    def _launch_topk_selector_kernel(
        mInput: cute.Tensor,
        mOutIdx: cute.Tensor,
        mLengths: cute.Tensor,
        mRowStarts: cute.Tensor,
        mCuSeqLens: cute.Tensor,
        mBlockTable: cute.Tensor,
        batch: int,
        seq_len: int,
        seq_q: int,
        num_reqs: int,
        topk: int,
        length_delta: int,
        stream,
    ):
        kernel(
            mInput,
            mOutIdx,
            mLengths,
            mRowStarts,
            mCuSeqLens,
            mBlockTable,
            Int32(batch),
            Int32(seq_len),
            Int32(seq_q),
            Int32(num_reqs),
            Int32(topk),
            Int32(length_delta),
        ).launch(
            grid=[batch, 1, 1],
            block=[_THREADS_PER_BLOCK, 1, 1],
            stream=stream,
        )

    return _launch_topk_selector_kernel


@cached_compile_function
def _get_compiled_kernel(
    dtype: torch.dtype,
    device_key: tuple[str, int | None],
    topk: int,
    max_candidates: int,
    use_f16_short_tail: bool,
    input_nonnegative: bool,
    use_row_starts: bool,
    use_cu_seqlens: bool,
    pack_output: bool,
) -> cute.JitFunction:
    device = utils.device_from_cache_key(device_key)
    # Compile with tiny persistent buffers and dynamic request/sequence axes so
    # the artifact is independent of the current graph capture dimensions.
    placeholder_input = torch.empty((1, 1), dtype=dtype, device=device)
    placeholder_out = torch.empty((1, int(topk)), dtype=torch.int32, device=device)
    placeholder_lengths = torch.empty((1,), dtype=torch.int32, device=device)
    placeholder_row_starts = torch.empty((1,), dtype=torch.int32, device=device)
    placeholder_cu_seq_lens = torch.empty((2,), dtype=torch.int32, device=device)
    placeholder_block_table = torch.empty((1, 1), dtype=torch.int32, device=device)

    mInput = _make_fake_tensor_with_dynamic_dims(
        placeholder_input,
        alignment=16,
        dynamic_shape_dims=(0, 1),
        dynamic_stride_dims=(0,),
    )
    mOutIdx = utils.make_fake_tensor_like_with_dynamic_dim(
        placeholder_out, alignment=16, dynamic_shape_dim=0)
    mLengths = utils.make_fake_tensor_like_with_dynamic_dim(
        placeholder_lengths, alignment=16, dynamic_shape_dim=0)
    mRowStarts = utils.make_fake_tensor_like_with_dynamic_dim(
        placeholder_row_starts, alignment=16, dynamic_shape_dim=0)
    mCuSeqLens = _make_fake_tensor_with_dynamic_dims(
        placeholder_cu_seq_lens,
        alignment=16,
        dynamic_shape_dims=(0,),
    )
    mBlockTable = _make_fake_tensor_with_dynamic_dims(
        placeholder_block_table,
        alignment=16,
        dynamic_shape_dims=(0, 1),
        dynamic_stride_dims=(0,),
    )
    launch = _make_launch_topk_selector_kernel(
        max_candidates=int(max_candidates),
        use_f16_short_tail=bool(use_f16_short_tail),
        input_nonnegative=bool(input_nonnegative),
        use_row_starts=bool(use_row_starts),
        use_cu_seqlens=bool(use_cu_seqlens),
        pack_output=bool(pack_output),
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        launch,
        mInput,
        mOutIdx,
        mLengths,
        mRowStarts,
        mCuSeqLens,
        mBlockTable,
        1,
        1,
        1,
        1,
        int(topk),
        0,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


def _cutedsl_topk_selector_prefill_sm90_gqa_impl(
    input_tensor: torch.Tensor,
    lengths: torch.Tensor,
    row_starts: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    *,
    topk: int,
    max_candidates: int,
    out_idx: torch.Tensor,
    input_nonnegative: bool,
    use_row_starts: bool,
    use_cu_seqlens: bool,
    block_table: torch.Tensor,
    seq_q: int,
    active_rows: int,
    pack_output: bool,
    length_delta: int,
    stream=None,
) -> None:
    if stream is not None:
        raise ValueError("top-k CuTeDSL kernels use the TVM-FFI environment stream")

    if not input_tensor.is_contiguous():
        raise ValueError(
            "cutedsl_topk_selector_prefill_sm90_gqa requires contiguous input_tensor; "
            f"got shape={tuple(input_tensor.shape)}, stride={tuple(input_tensor.stride())}"
        )
    if (
        not lengths.is_contiguous()
        or not row_starts.is_contiguous()
        or not cu_seq_lens.is_contiguous()
    ):
        raise ValueError(
            "lengths/row_starts/cu_seq_lens must be contiguous, got "
            f"lengths_stride={tuple(lengths.stride())}, "
            f"row_starts_stride={tuple(row_starts.stride())}, "
            f"cu_seq_lens_stride={tuple(cu_seq_lens.stride())}"
        )
    if cu_seq_lens.dtype != torch.int32:
        raise ValueError(f"cu_seq_lens must be torch.int32, got {cu_seq_lens.dtype}")
    if cu_seq_lens.device != input_tensor.device:
        raise ValueError("cu_seq_lens must be on the same device as input_tensor")
    if not block_table.is_contiguous():
        raise ValueError(f"block_table must be contiguous, got stride={tuple(block_table.stride())}")
    if block_table.dtype != torch.int32:
        raise ValueError(f"block_table must be torch.int32, got {block_table.dtype}")
    if block_table.device != input_tensor.device:
        raise ValueError("block_table must be on the same device as input_tensor")

    capture_rows, seq_len = [int(v) for v in input_tensor.shape]
    batch = int(active_rows)
    seq_q_i = int(seq_q)
    if block_table.ndim != 2:
        raise ValueError(f"block_table must be [num_reqs, pages], got {tuple(block_table.shape)}")
    if seq_q_i <= 0:
        raise ValueError(f"seq_q must be > 0 for topk selector, got {seq_q_i}")
    if batch <= 0 or batch > capture_rows:
        raise ValueError(f"active_rows must be in (0, input rows], got active_rows={batch}, rows={capture_rows}")
    if int(lengths.shape[0]) < batch or int(row_starts.shape[0]) < batch:
        raise ValueError(
            "lengths/row_starts first dim must cover active_rows="
            f"{batch}, got lengths={tuple(lengths.shape)}, row_starts={tuple(row_starts.shape)}"
        )
    if int(out_idx.shape[0]) < batch:
        raise ValueError(f"out_idx first dim must cover active_rows={batch}, got {tuple(out_idx.shape)}")
    if bool(pack_output) and not bool(use_cu_seqlens) and int(batch) % seq_q_i != 0:
        raise ValueError(f"input rows={batch} must be divisible by seq_q={seq_q_i}")
    num_reqs = int(block_table.shape[0])
    if bool(pack_output):
        if bool(use_cu_seqlens):
            if cu_seq_lens.dim() != 1:
                raise ValueError("cu_seq_lens must be rank-1 [num_reqs + 1]")
            if int(cu_seq_lens.shape[0]) != num_reqs + 1:
                raise ValueError(
                    "cu_seq_lens length must match block_table requests + 1, got "
                    f"{tuple(cu_seq_lens.shape)} for block_table rows={num_reqs}"
                )
        elif num_reqs != 1:
            raise ValueError(
                "multi-request topk requires cu_seq_lens; "
                f"got block_table rows={num_reqs}"
            )
        if int(seq_len) > int(block_table.shape[1]) * 2:
            raise ValueError(
                f"block_table has too few pages for seq_len={seq_len}: shape={tuple(block_table.shape)}"
            )
    use_f16_short_tail = input_tensor.dtype == torch.float16

    compiled = _get_compiled_kernel(
        dtype=input_tensor.dtype,
        device_key=utils.device_cache_key(input_tensor.device),
        topk=topk,
        max_candidates=max_candidates,
        use_f16_short_tail=use_f16_short_tail,
        input_nonnegative=bool(input_nonnegative),
        use_row_starts=bool(use_row_starts),
        use_cu_seqlens=bool(use_cu_seqlens),
        pack_output=bool(pack_output),
    )
    compiled(
        input_tensor,
        out_idx,
        lengths,
        row_starts,
        cu_seq_lens,
        block_table,
        batch,
        seq_len,
        seq_q_i,
        num_reqs,
        topk,
        int(length_delta),
    )
def _validate_topk_public_inputs(
    input_tensor: torch.Tensor,
    lengths: torch.Tensor,
    row_starts: torch.Tensor,
    *,
    k: int,
    max_candidates: int,
    out_idx: Optional[torch.Tensor],
    active_rows: Optional[int],
) -> Optional[torch.Tensor]:
    compiling = _is_torch_compiling()
    if input_tensor.dim() != 2:
        raise ValueError("input_tensor must be [batch, seq_len]")
    if lengths.dim() != 1 or row_starts.dim() != 1:
        raise ValueError("lengths and row_starts must be [batch]")
    if input_tensor.device.type != "cuda":
        raise RuntimeError("cutedsl_topk_selector_prefill_sm90_gqa requires CUDA tensors")
    if input_tensor.dtype not in (torch.float16, torch.float32):
        raise ValueError("input_tensor dtype must be float16 or float32")
    if lengths.dtype != torch.int32 or row_starts.dtype != torch.int32:
        raise ValueError("lengths and row_starts dtype must be torch.int32")
    if not lengths.is_contiguous() or not row_starts.is_contiguous():
        raise ValueError(
            "lengths and row_starts must be contiguous, got "
            f"lengths_stride={tuple(lengths.stride())}, "
            f"row_starts_stride={tuple(row_starts.stride())}"
        )
    if (
        lengths.device != input_tensor.device
        or row_starts.device != input_tensor.device
    ):
        raise ValueError("input_tensor, lengths, and row_starts must be on the same device")
    batch = input_tensor.shape[0]
    active_rows_i = int(batch) if active_rows is None else int(active_rows)
    if active_rows_i <= 0 or active_rows_i > int(batch):
        raise ValueError(f"active_rows must be in (0, input rows], got active_rows={active_rows_i}, rows={int(batch)}")
    seq_len = int(input_tensor.shape[1])
    if (
        not compiling
        and (
            int(lengths.shape[0]) < active_rows_i
            or int(row_starts.shape[0]) < active_rows_i
        )
    ):
        raise ValueError("lengths and row_starts length must cover active_rows")
    k = int(k)
    if k <= 0:
        raise ValueError("k must be > 0")
    if k > seq_len:
        raise ValueError(f"k must be <= seq_len, got k={k}, seq_len={seq_len}")
    max_cand = int(max_candidates)
    if max_cand <= 0:
        raise ValueError("max_candidates must be > 0")
    if max_cand < k:
        raise ValueError(f"max_candidates must be >= k, got max_candidates={max_cand}, k={k}")
    max_supported = _selector_max_candidates_cap(input_tensor.device)
    if max_cand > max_supported:
        raise ValueError(
            "max_candidates exceeds selector shared-memory capacity on this device: "
            f"max_candidates={max_cand}, supported<={max_supported}"
        )

    if out_idx is not None:
        if out_idx.dtype != torch.int32:
            raise ValueError("out_idx dtype must be torch.int32")
        if out_idx.device != input_tensor.device:
            raise ValueError("out_idx must be on the same device as input_tensor")
        if not out_idx.is_contiguous():
            raise ValueError(f"out_idx must be contiguous, got stride={tuple(out_idx.stride())}")
        if (
            out_idx.ndim != 2
            or int(out_idx.shape[1]) != k
            or (not compiling and int(out_idx.shape[0]) < active_rows_i)
        ):
            raise ValueError(
                f"out_idx shape must cover ({active_rows_i}, {k}), got {tuple(out_idx.shape)}"
            )
    return out_idx


@torch.library.custom_op(
    "optimus_cutedsl::cutedsl_topk_selector_prefill_sm90_gqa_out",
    mutates_args=("out_idx",),
    device_types="cuda",
)
def _cutedsl_topk_selector_prefill_sm90_gqa_out(
    input_tensor: torch.Tensor,
    lengths: torch.Tensor,
    row_starts: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    out_idx: torch.Tensor,
    seq_q: int,
    active_rows: int,
    k: int,
    max_candidates: int,
    score_nonneg: bool,
    use_row_starts: bool,
    use_cu_seqlens: bool,
) -> None:
    _cutedsl_topk_selector_prefill_sm90_gqa_impl(
        input_tensor,
        lengths,
        row_starts,
        cu_seq_lens,
        topk=k,
        max_candidates=max_candidates,
        out_idx=out_idx,
        input_nonnegative=bool(score_nonneg),
        use_row_starts=bool(use_row_starts),
        use_cu_seqlens=bool(use_cu_seqlens),
        block_table=block_table,
        seq_q=int(seq_q),
        active_rows=int(active_rows),
        pack_output=True,
        length_delta=0,
        stream=None,
    )


@torch.library.custom_op(
    "optimus_cutedsl::cutedsl_topk_selector_prefill_sm90_gqa_functional",
    mutates_args=(),
    device_types="cuda",
)
def _cutedsl_topk_selector_prefill_sm90_gqa_functional(
    input_tensor: torch.Tensor,
    lengths: torch.Tensor,
    row_starts: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    seq_q: int,
    active_rows: int,
    k: int,
    max_candidates: int,
    score_nonneg: bool,
    use_row_starts: bool,
    use_cu_seqlens: bool,
) -> torch.Tensor:
    out_idx = torch.empty(
        (int(active_rows), int(k)),
        dtype=torch.int32,
        device=input_tensor.device,
    )
    _cutedsl_topk_selector_prefill_sm90_gqa_impl(
        input_tensor,
        lengths,
        row_starts,
        cu_seq_lens,
        topk=k,
        max_candidates=max_candidates,
        out_idx=out_idx,
        input_nonnegative=bool(score_nonneg),
        use_row_starts=bool(use_row_starts),
        use_cu_seqlens=bool(use_cu_seqlens),
        block_table=block_table,
        seq_q=int(seq_q),
        active_rows=int(active_rows),
        pack_output=True,
        length_delta=0,
        stream=None,
    )
    return out_idx


@_cutedsl_topk_selector_prefill_sm90_gqa_functional.register_fake
def _cutedsl_topk_selector_prefill_sm90_gqa_functional_fake(
    input_tensor: torch.Tensor,
    _lengths: torch.Tensor,
    _row_starts: torch.Tensor,
    _cu_seq_lens: torch.Tensor,
    _block_table: torch.Tensor,
    _seq_q: int,
    active_rows: int,
    k: int,
    _max_candidates: int,
    _score_nonneg: bool,
    _use_row_starts: bool,
    _use_cu_seqlens: bool,
) -> torch.Tensor:
    return torch.empty(
        (active_rows, k),
        dtype=torch.int32,
        device=input_tensor.device,
    )


def cutedsl_topk_selector_prefill_sm90_gqa(
    input_tensor: torch.Tensor,
    block_table: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    *,
    seq_q: int,
    active_rows: Optional[int] = None,
    row_starts: Optional[torch.Tensor] = None,
    cu_seq_lens: Optional[torch.Tensor] = None,
    score_nonneg: bool = True,
    out_idx: Optional[torch.Tensor] = None,
    max_candidates: Optional[int] = None,
    stream=None,
) -> torch.Tensor:
    max_candidates = _selector_max_candidates_for_input(
        input_tensor, k, max_candidates=max_candidates)
    use_row_starts = row_starts is not None
    use_cu_seqlens = cu_seq_lens is not None
    row_starts_tensor = row_starts if row_starts is not None else lengths
    out_idx = _validate_topk_public_inputs(
        input_tensor,
        lengths,
        row_starts_tensor,
        k=k,
        max_candidates=max_candidates,
        out_idx=out_idx,
        active_rows=active_rows,
    )
    if block_table.dtype != torch.int32:
        raise ValueError(f"block_table must be torch.int32, got {block_table.dtype}")
    if block_table.device != input_tensor.device:
        raise ValueError("block_table must be on the same device as input_tensor")
    if block_table.ndim != 2:
        raise ValueError(f"block_table must be [num_reqs, pages], got {tuple(block_table.shape)}")
    if not block_table.is_contiguous():
        raise ValueError(f"block_table must be contiguous, got stride={tuple(block_table.stride())}")
    seq_q_i = int(seq_q)
    if seq_q_i <= 0:
        raise ValueError(f"seq_q must be > 0, got {seq_q_i}")
    active_rows_i = int(input_tensor.shape[0]) if active_rows is None else int(active_rows)
    if active_rows_i <= 0 or active_rows_i > int(input_tensor.shape[0]):
        raise ValueError(
            f"active_rows must be in (0, input rows], got active_rows={active_rows_i}, "
            f"rows={int(input_tensor.shape[0])}"
        )
    if active_rows_i % seq_q_i != 0:
        raise ValueError(
            f"active_rows={active_rows_i} must be divisible by seq_q={seq_q_i}"
        )
    if use_cu_seqlens:
        cu_seq_lens_tensor = cu_seq_lens
        if cu_seq_lens_tensor is None:
            raise RuntimeError("unreachable")
        if cu_seq_lens_tensor.dtype != torch.int32:
            raise ValueError(f"cu_seq_lens must be torch.int32, got {cu_seq_lens_tensor.dtype}")
        if cu_seq_lens_tensor.device != input_tensor.device:
            raise ValueError("cu_seq_lens must be on the same device as input_tensor")
        if cu_seq_lens_tensor.ndim != 1:
            raise ValueError(f"cu_seq_lens must be rank-1, got {tuple(cu_seq_lens_tensor.shape)}")
        if int(cu_seq_lens_tensor.shape[0]) != int(block_table.shape[0]) + 1:
            raise ValueError(
                "cu_seq_lens length must equal block_table rows + 1, got "
                f"{tuple(cu_seq_lens_tensor.shape)} for block_table={tuple(block_table.shape)}"
            )
        if not cu_seq_lens_tensor.is_contiguous():
            raise ValueError(f"cu_seq_lens must be contiguous, got stride={tuple(cu_seq_lens_tensor.stride())}")
    else:
        if int(block_table.shape[0]) != 1:
            raise ValueError(
                "multi-request topk requires cu_seq_lens; "
                f"got block_table rows={int(block_table.shape[0])}"
            )
        cu_seq_lens_tensor = lengths
    if int(input_tensor.shape[1]) > int(block_table.shape[1]) * 2:
        raise ValueError(
            "block_table has too few pages for topk prefill selector: "
            f"seq_len={int(input_tensor.shape[1])}, block_table_shape={tuple(block_table.shape)}"
        )
    k = int(k)
    max_cand = int(max_candidates)
    if stream is None:
        if out_idx is None:
            return _cutedsl_topk_selector_prefill_sm90_gqa_functional(
                input_tensor,
                lengths,
                row_starts_tensor,
                cu_seq_lens_tensor,
                block_table,
                seq_q_i,
                active_rows_i,
                k,
                max_cand,
                bool(score_nonneg),
                bool(use_row_starts),
                bool(use_cu_seqlens),
            )
        _cutedsl_topk_selector_prefill_sm90_gqa_out(
            input_tensor,
            lengths,
            row_starts_tensor,
            cu_seq_lens_tensor,
            block_table,
            out_idx,
            seq_q_i,
            active_rows_i,
            k,
            max_cand,
            bool(score_nonneg),
            bool(use_row_starts),
            bool(use_cu_seqlens),
        )
    else:
        if out_idx is None:
            out_idx = torch.empty(
                (active_rows_i, k),
                dtype=torch.int32,
                device=input_tensor.device,
            )
        _cutedsl_topk_selector_prefill_sm90_gqa_impl(
            input_tensor,
            lengths,
            row_starts_tensor,
            cu_seq_lens_tensor,
            topk=k,
            max_candidates=max_cand,
            out_idx=out_idx,
            input_nonnegative=bool(score_nonneg),
            use_row_starts=bool(use_row_starts),
            use_cu_seqlens=bool(use_cu_seqlens),
            block_table=block_table,
            seq_q=seq_q_i,
            active_rows=active_rows_i,
            pack_output=True,
            length_delta=0,
            stream=stream,
        )
    return out_idx



def _make_topk_selector_decode_meta_kernel(
    max_candidates: int,
    *,
    topk: int,
    total_windows: int,
    sort_cap: int,
    sort_log2: int,
    use_f16_short_tail: bool,
    input_nonnegative: bool,
    use_row_starts: bool,
    sort_output: bool,
):
    max_cand = int(max_candidates)
    max_cand_i32 = Int32(max_cand)
    topk_const = int(topk)
    total_windows_const = int(total_windows)
    sort_cap_const = int(sort_cap)
    sort_log2_const = int(sort_log2)
    use_f16_key = bool(use_f16_short_tail)
    use_f16_nonnegative_key = bool(use_f16_key and input_nonnegative)
    tail_rounds = _F16_TAIL_ROUNDS if use_f16_key else _F32_TAIL_ROUNDS
    first_shift = 8 if use_f16_key else 24
    invalid_packed_const = 0x7FFF_FFFF_FFFF_FFFF

    @cute.kernel
    def _topk_selector_decode_meta_kernel_sm90(
        mInput: cute.Tensor,
        mLengths: cute.Tensor,
        mRowStarts: cute.Tensor,
        mQueryStartLoc: cute.Tensor,
        mSeqLens: cute.Tensor,
        mReqBlockOffsets: cute.Tensor,
        mBlockTable: cute.Tensor,
        mCountsOut: cute.Tensor,
        mPackedOut: cute.Tensor,
        batch: Int32,
        seq_len: Int32,
        logical_num_regions: Int32,
        num_reqs: Int32,
        topk_runtime: Int32,
        length_delta: Int32,
    ):
        bx, _, _ = cute.arch.block_idx()
        tx = cute.arch.thread_idx()[0]
        threshold = Int32(0)

        @cute.struct
        class SharedStorage:
            hist: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, _RADIX + 1], 128]
            cand: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 2 * max_cand], 128
            ]
            packed: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, sort_cap_const], 128
            ]
            cnt: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 2], 16]
            remain: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 8]
            threshold: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 8]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage, 16)
        sHist = storage.hist.get_tensor(cute.make_layout((_RADIX + 1,), stride=(1,)))
        sCand = storage.cand.get_tensor(
            cute.make_layout((2, max_cand), stride=(max_cand, 1))
        )
        sPacked = storage.packed.get_tensor(
            cute.make_layout((sort_cap_const,), stride=(1,))
        )
        sCnt = storage.cnt.get_tensor(cute.make_layout((2,), stride=(1,)))
        sRemain = storage.remain.get_tensor(cute.make_layout((1,), stride=(1,)))
        sThreshold = storage.threshold.get_tensor(cute.make_layout((1,), stride=(1,)))
        packed = Int64(0)

        if bx < batch:
            for tile in cutlass.range_constexpr(
                (topk_const + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
            ):
                pos = tx + Int32(tile * _THREADS_PER_BLOCK)
                if pos < Int32(topk_const):
                    mPackedOut[bx, pos] = Int64(-1)
            for tile in cutlass.range_constexpr(
                (sort_cap_const + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
            ):
                pos = tx + Int32(tile * _THREADS_PER_BLOCK)
                if pos < Int32(sort_cap_const):
                    sPacked[pos] = Int64(invalid_packed_const)
            cute.arch.sync_threads()

            start_idx = Int32(0)
            if cutlass.const_expr(use_row_starts):
                start_idx = mRowStarts[bx]
            valid_len = mLengths[bx] + length_delta
            if valid_len < Int32(0):
                valid_len = Int32(0)
            end_idx = start_idx + valid_len
            if end_idx > seq_len:
                end_idx = seq_len
            if end_idx > logical_num_regions:
                # Keep runtime access within the logical candidate width so
                # padded tail entries never contribute to decode meta.
                end_idx = logical_num_regions
            if start_idx < Int32(0):
                start_idx = Int32(0)
            valid_len = end_idx - start_idx
            if valid_len < Int32(0):
                valid_len = Int32(0)

            for i_chunk in cutlass.range_constexpr(
                (_RADIX + 1 + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
            ):
                i = tx + Int32(i_chunk * _THREADS_PER_BLOCK)
                if i <= Int32(_RADIX):
                    sHist[i] = Int32(0)

            if tx < Int32(2):
                sCnt[tx] = Int32(0)
            if tx == Int32(0):
                sRemain[Int32(0)] = topk_runtime
                sThreshold[Int32(0)] = Int32(0)
            cute.arch.sync_threads()

            num_tiles = (valid_len + Int32(_THREADS_PER_BLOCK - 1)) // Int32(_THREADS_PER_BLOCK)
            for tile in cutlass.range(num_tiles, unroll=1):
                input_idx = start_idx + tile * Int32(_THREADS_PER_BLOCK) + tx
                if input_idx < end_idx:
                    score_value = Float32(mInput[bx, input_idx])
                    if score_value != -Float32.inf:
                        if cutlass.const_expr(use_f16_nonnegative_key):
                            bin_id = _radix_byte_from_nonnegative_f16(
                                mInput[bx, input_idx], shift=first_shift)
                        elif cutlass.const_expr(use_f16_key):
                            bin_id = _radix_byte_from_ordered_f16(
                                mInput[bx, input_idx], shift=first_shift)
                        else:
                            bin_id = _radix_byte_from_f32(
                                score_value,
                                shift=first_shift,
                            )
                        _atomic_add_i32(elem_pointer(sHist, (bin_id,)), Int32(1))
            cute.arch.sync_threads()

            _hist_prefix_desc_parallel(sHist, tx)
            _hist_select_threshold_parallel(sHist, sRemain, sThreshold, tx)

            for tile in cutlass.range(num_tiles, unroll=1):
                input_idx = start_idx + tile * Int32(_THREADS_PER_BLOCK) + tx
                if input_idx < end_idx:
                    score_value = Float32(mInput[bx, input_idx])
                    if score_value != -Float32.inf:
                        if cutlass.const_expr(use_f16_nonnegative_key):
                            bin_id = _radix_byte_from_nonnegative_f16(
                                mInput[bx, input_idx], shift=first_shift)
                        elif cutlass.const_expr(use_f16_key):
                            bin_id = _radix_byte_from_ordered_f16(
                                mInput[bx, input_idx], shift=first_shift)
                        else:
                            bin_id = _radix_byte_from_f32(
                                score_value,
                                shift=first_shift,
                            )
                        threshold = sThreshold[Int32(0)]
                        if bin_id > threshold:
                            pos_out = _atomic_add_i32(elem_pointer(sHist, (bin_id + Int32(1),)), Int32(1))
                            if pos_out < topk_runtime:
                                mPackedOut[bx, pos_out] = Int64(input_idx)
            cute.arch.sync_threads()

            if tx == Int32(0):
                sCnt[Int32(0)] = Int32(0)
                if sRemain[Int32(0)] > Int32(0):
                    threshold = sThreshold[Int32(0)]
                    for tile in cutlass.range(num_tiles, unroll=1):
                        for lane in cutlass.range_constexpr(_THREADS_PER_BLOCK):
                            input_idx = start_idx + tile * Int32(_THREADS_PER_BLOCK) + Int32(lane)
                            if input_idx < end_idx:
                                score_value = Float32(mInput[bx, input_idx])
                                if score_value != -Float32.inf:
                                    if cutlass.const_expr(use_f16_nonnegative_key):
                                        bin_id = _radix_byte_from_nonnegative_f16(
                                            mInput[bx, input_idx], shift=first_shift)
                                    elif cutlass.const_expr(use_f16_key):
                                        bin_id = _radix_byte_from_ordered_f16(
                                            mInput[bx, input_idx], shift=first_shift)
                                    else:
                                        bin_id = _radix_byte_from_f32(
                                            score_value,
                                            shift=first_shift,
                                        )
                                    if bin_id == threshold:
                                        pos_cand = sCnt[Int32(0)]
                                        if pos_cand < max_cand_i32:
                                            sCand[Int32(0), pos_cand] = input_idx
                                        sCnt[Int32(0)] = pos_cand + Int32(1)
            cute.arch.sync_threads()

            for round_i in cutlass.range_constexpr(tail_rounds):
                cur_buf = Int32(round_i & 1)
                nxt_buf = Int32((round_i + 1) & 1)

                if sRemain[Int32(0)] > Int32(0):
                    for i_chunk in cutlass.range_constexpr(
                        (_RADIX + 1 + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
                    ):
                        i = tx + Int32(i_chunk * _THREADS_PER_BLOCK)
                        if i <= Int32(_RADIX):
                            sHist[i] = Int32(0)
                    if tx == Int32(0):
                        sCnt[nxt_buf] = Int32(0)
                    cute.arch.sync_threads()

                    num_input = cutlass.min(sCnt[cur_buf], max_cand_i32)
                    num_tiles_round = (num_input + Int32(_THREADS_PER_BLOCK - 1)) // Int32(
                        _THREADS_PER_BLOCK
                    )
                    for tile in cutlass.range(num_tiles_round, unroll=1):
                        cand_pos = tile * Int32(_THREADS_PER_BLOCK) + tx
                        if cand_pos < num_input:
                            input_idx = sCand[cur_buf, cand_pos]
                            if cutlass.const_expr(use_f16_nonnegative_key):
                                bin_id = _radix_byte_from_nonnegative_f16(
                                    mInput[bx, input_idx], shift=0)
                            elif cutlass.const_expr(use_f16_key):
                                bin_id = _radix_byte_from_ordered_f16(
                                    mInput[bx, input_idx], shift=0)
                            else:
                                bin_id = _radix_byte_from_f32(
                                    Float32(mInput[bx, input_idx]),
                                    shift=16 - round_i * 8,
                                )
                            _atomic_add_i32(elem_pointer(sHist, (bin_id,)), Int32(1))
                    cute.arch.sync_threads()

                    _hist_prefix_desc_parallel(sHist, tx)
                    _hist_select_threshold_parallel(sHist, sRemain, sThreshold, tx)

                    remain_old = sRemain[Int32(0)] + sHist[sThreshold[Int32(0)] + Int32(1)]
                    start_pos = topk_runtime - remain_old
                    for tile in cutlass.range(num_tiles_round, unroll=1):
                        cand_pos = tile * Int32(_THREADS_PER_BLOCK) + tx
                        if cand_pos < num_input:
                            input_idx = sCand[cur_buf, cand_pos]
                            if cutlass.const_expr(use_f16_nonnegative_key):
                                bin_id = _radix_byte_from_nonnegative_f16(
                                    mInput[bx, input_idx], shift=0)
                            elif cutlass.const_expr(use_f16_key):
                                bin_id = _radix_byte_from_ordered_f16(
                                    mInput[bx, input_idx], shift=0)
                            else:
                                bin_id = _radix_byte_from_f32(
                                    Float32(mInput[bx, input_idx]),
                                    shift=16 - round_i * 8,
                                )
                            threshold = sThreshold[Int32(0)]
                            if bin_id > threshold:
                                pos_out = (
                                    _atomic_add_i32(elem_pointer(sHist, (bin_id + Int32(1),)), Int32(1))
                                    + start_pos
                                )
                                if pos_out < topk_runtime:
                                    mPackedOut[bx, pos_out] = Int64(input_idx)
                    cute.arch.sync_threads()

                    if tx == Int32(0):
                        threshold = sThreshold[Int32(0)]
                        if round_i != tail_rounds - 1:
                            sCnt[nxt_buf] = Int32(0)
                        threshold_count = Int32(0)
                        if sRemain[Int32(0)] > Int32(0):
                            for cand_pos in cutlass.range(num_input, unroll=1):
                                input_idx = sCand[cur_buf, cand_pos]
                                if cutlass.const_expr(use_f16_nonnegative_key):
                                    bin_id = _radix_byte_from_nonnegative_f16(
                                        mInput[bx, input_idx], shift=0)
                                elif cutlass.const_expr(use_f16_key):
                                    bin_id = _radix_byte_from_ordered_f16(
                                        mInput[bx, input_idx], shift=0)
                                else:
                                    bin_id = _radix_byte_from_f32(
                                        Float32(mInput[bx, input_idx]),
                                        shift=16 - round_i * 8,
                                    )
                                if bin_id == threshold:
                                    if round_i == tail_rounds - 1:
                                        pos_tail = start_pos + threshold_count
                                        if pos_tail < topk_runtime:
                                            mPackedOut[bx, pos_tail] = Int64(input_idx)
                                    else:
                                        if threshold_count < max_cand_i32:
                                            sCand[nxt_buf, threshold_count] = input_idx
                                    threshold_count += Int32(1)
                            if round_i != tail_rounds - 1:
                                sCnt[nxt_buf] = threshold_count
                    cute.arch.sync_threads()

            req_idx = Int32(0)
            q_start = Int32(0)
            q_end = Int32(0)
            for req in cutlass.range(num_reqs, unroll=1):
                begin = mQueryStartLoc[req]
                end = mQueryStartLoc[req + Int32(1)]
                if bx >= begin and bx < end:
                    req_idx = req
                    q_start = begin
                    q_end = end
            valid_k = mSeqLens[req_idx]
            block_offset = mReqBlockOffsets[req_idx]
            q_len = q_end - q_start
            q_local = bx - q_start
            query_pos = valid_k - q_len + q_local
            valid_blocks = (valid_k + Int32(_REGION_TOKENS - 1)) // Int32(_REGION_TOKENS)
            local_insert_pos = topk_runtime
            if not cutlass.const_expr(sort_output):
                local_insert_pos = valid_len
                if local_insert_pos > topk_runtime:
                    local_insert_pos = topk_runtime

            for tile in cutlass.range_constexpr(
                (sort_cap_const + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
            ):
                win = tx + Int32(tile * _THREADS_PER_BLOCK)
                packed = Int64(invalid_packed_const)
                local_blk_safe = Int32(0)
                is_valid = False
                if win < Int32(sort_cap_const):
                    if win < local_insert_pos:
                        global_blk = Int32(mPackedOut[bx, win])
                        local_blk = global_blk - block_offset
                        is_valid = (global_blk >= Int32(0)) and (local_blk >= Int32(0)) and (local_blk < valid_blocks)
                        if is_valid:
                            local_blk_safe = local_blk
                    elif win == local_insert_pos:
                        local_blk = query_pos // Int32(_REGION_TOKENS)
                        is_valid = (local_blk >= Int32(0)) and (local_blk < valid_blocks)
                        if is_valid:
                            local_blk_safe = local_blk
                    if is_valid:
                        start_tok = local_blk_safe * Int32(_REGION_TOKENS)
                        page_idx = local_blk_safe // Int32(2)
                        phys_page = mBlockTable[req_idx, page_idx]
                        phys_blk8 = phys_page * Int32(2) + (local_blk_safe & Int32(1))
                        packed = Int64(start_tok) | (Int64(phys_blk8) << Int64(32))
                    sPacked[win] = packed
            cute.arch.sync_threads()

            if cutlass.const_expr(sort_output):
                for stage in cutlass.range_constexpr(sort_log2_const):
                    sort_size = Int32(1 << (stage + 1))
                    for sub_stage in cutlass.range_constexpr(stage + 1):
                        stride = Int32(1 << (stage - sub_stage))
                        for tile in cutlass.range_constexpr(
                            (sort_cap_const + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
                        ):
                            i = tx + Int32(tile * _THREADS_PER_BLOCK)
                            ixj = i ^ stride
                            if i < Int32(sort_cap_const) and ixj > i:
                                a = sPacked[i]
                                b = sPacked[ixj]
                                ascending = (i & sort_size) == Int32(0)
                                if (ascending and a > b) or ((not ascending) and a < b):
                                    sPacked[i] = b
                                    sPacked[ixj] = a
                        cute.arch.sync_threads()

            if tx == Int32(0):
                count = Int32(0)
                for pos in cutlass.range_constexpr(total_windows_const):
                    if sPacked[Int32(pos)] != Int64(invalid_packed_const):
                        count += Int32(1)
                mCountsOut[bx] = count
            cute.arch.sync_threads()

            for tile in cutlass.range_constexpr(
                (total_windows_const + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
            ):
                win = tx + Int32(tile * _THREADS_PER_BLOCK)
                if win < Int32(total_windows_const):
                    value = sPacked[win]
                    if value == Int64(invalid_packed_const):
                        value = Int64(0)
                    mPackedOut[bx, win] = value

    return _topk_selector_decode_meta_kernel_sm90


def _make_launch_topk_selector_decode_meta_kernel(
    max_candidates: int,
    *,
    topk: int,
    total_windows: int,
    sort_cap: int,
    sort_log2: int,
    use_f16_short_tail: bool,
    input_nonnegative: bool,
    use_row_starts: bool,
    sort_output: bool,
):
    kernel = _make_topk_selector_decode_meta_kernel(
        max_candidates,
        topk=int(topk),
        total_windows=int(total_windows),
        sort_cap=int(sort_cap),
        sort_log2=int(sort_log2),
        use_f16_short_tail=bool(use_f16_short_tail),
        input_nonnegative=bool(input_nonnegative),
        use_row_starts=bool(use_row_starts),
        sort_output=bool(sort_output),
    )

    @cute.jit
    def _launch_topk_selector_decode_meta_kernel(
        mInput: cute.Tensor,
        mLengths: cute.Tensor,
        mRowStarts: cute.Tensor,
        mQueryStartLoc: cute.Tensor,
        mSeqLens: cute.Tensor,
        mReqBlockOffsets: cute.Tensor,
        mBlockTable: cute.Tensor,
        mCountsOut: cute.Tensor,
        mPackedOut: cute.Tensor,
        batch: int,
        seq_len: int,
        logical_num_regions: int,
        num_reqs: int,
        topk_runtime: int,
        length_delta: int,
        stream,
    ):
        kernel(
            mInput,
            mLengths,
            mRowStarts,
            mQueryStartLoc,
            mSeqLens,
            mReqBlockOffsets,
            mBlockTable,
            mCountsOut,
            mPackedOut,
            Int32(batch),
            Int32(seq_len),
            Int32(logical_num_regions),
            Int32(num_reqs),
            Int32(topk_runtime),
            Int32(length_delta),
        ).launch(
            grid=[batch, 1, 1],
            block=[_THREADS_PER_BLOCK, 1, 1],
            stream=stream,
        )

    return _launch_topk_selector_decode_meta_kernel


@cached_compile_function
def _get_compiled_decode_meta_kernel(
    dtype: torch.dtype,
    device_key: tuple[str, int | None],
    topk: int,
    total_windows: int,
    sort_cap: int,
    sort_log2: int,
    max_candidates: int,
    use_f16_short_tail: bool,
    input_nonnegative: bool,
    use_row_starts: bool,
    sort_output: bool = True,
) -> cute.JitFunction:
    device = utils.device_from_cache_key(device_key)
    placeholder_input = torch.empty((1, 1), dtype=dtype, device=device)
    placeholder_lengths = torch.empty((1,), dtype=torch.int32, device=device)
    placeholder_row_starts = torch.empty((1,), dtype=torch.int32, device=device)
    placeholder_query_start_loc = torch.empty((2,), dtype=torch.int32, device=device)
    placeholder_seq_lens = torch.empty((1,), dtype=torch.int32, device=device)
    placeholder_req_offsets = torch.empty((1,), dtype=torch.int32, device=device)
    placeholder_block_table = torch.empty((1, 1), dtype=torch.int32, device=device)
    placeholder_counts = torch.empty((1,), dtype=torch.int32, device=device)
    placeholder_packed = torch.empty((1, int(total_windows)), dtype=torch.int64, device=device)

    mInput = _make_fake_tensor_with_dynamic_dims(
        placeholder_input,
        alignment=16,
        dynamic_shape_dims=(0, 1),
        dynamic_stride_dims=(0,),
    )
    mLengths = utils.make_fake_tensor_like_with_dynamic_dim(
        placeholder_lengths, alignment=16, dynamic_shape_dim=0)
    mRowStarts = utils.make_fake_tensor_like_with_dynamic_dim(
        placeholder_row_starts, alignment=16, dynamic_shape_dim=0)
    mQueryStartLoc = _make_fake_tensor_with_dynamic_dims(
        placeholder_query_start_loc,
        alignment=16,
        dynamic_shape_dims=(0,),
    )
    mSeqLens = _make_fake_tensor_with_dynamic_dims(
        placeholder_seq_lens,
        alignment=16,
        dynamic_shape_dims=(0,),
    )
    mReqBlockOffsets = _make_fake_tensor_with_dynamic_dims(
        placeholder_req_offsets,
        alignment=16,
        dynamic_shape_dims=(0,),
    )
    mBlockTable = _make_fake_tensor_with_dynamic_dims(
        placeholder_block_table,
        alignment=16,
        dynamic_shape_dims=(0, 1),
        dynamic_stride_dims=(0,),
    )
    mCounts = utils.make_fake_tensor_like_with_dynamic_dim(
        placeholder_counts, alignment=16, dynamic_shape_dim=0)
    mPacked = utils.make_fake_tensor_like_with_dynamic_dim(
        placeholder_packed, alignment=16, dynamic_shape_dim=0)
    launch = _make_launch_topk_selector_decode_meta_kernel(
        max_candidates=int(max_candidates),
        topk=int(topk),
        total_windows=int(total_windows),
        sort_cap=int(sort_cap),
        sort_log2=int(sort_log2),
        use_f16_short_tail=bool(use_f16_short_tail),
        input_nonnegative=bool(input_nonnegative),
        use_row_starts=bool(use_row_starts),
        sort_output=bool(sort_output),
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        launch,
        mInput,
        mLengths,
        mRowStarts,
        mQueryStartLoc,
        mSeqLens,
        mReqBlockOffsets,
        mBlockTable,
        mCounts,
        mPacked,
        1,
        1,
        1,
        1,
        int(topk),
        0,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


def prewarm_topk_selector_decode_meta_sm90_gqa(
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    k: int,
    max_score_capacity: int,
    score_capacity_step: int,
    score_nonneg: bool = True,
    sort_output: bool = True,
) -> tuple[int, ...]:
    """Compile each reachable single-CTA decode-meta capacity class once.

    ``score_capacity_step`` is the graph-stable score-buffer bucket size. For
    vLLM's 4096-region buckets this includes the 12288 capacity and therefore
    precompiles its 16384-candidate artifact before the streaming threshold.
    """
    max_score_capacity = int(max_score_capacity)
    score_capacity_step = int(score_capacity_step)
    k = int(k)
    if max_score_capacity <= 0:
        raise ValueError(
            f"max_score_capacity must be > 0, got {max_score_capacity}"
        )
    if score_capacity_step <= 0:
        raise ValueError(
            f"score_capacity_step must be > 0, got {score_capacity_step}"
        )

    single_cta_limit = min(max_score_capacity, _MULTI_CTA_DECODE_THRESHOLD)
    capacities = list(
        range(score_capacity_step, single_cta_limit, score_capacity_step)
    )
    capacities.append(single_cta_limit)
    capacity_classes = tuple(
        sorted(
            {
                capacity_class
                for score_capacity in capacities
                if (
                    capacity_class := topk_selector_decode_meta_capacity_class_sm90_gqa(
                        score_capacity,
                        k=k,
                    )
                )
                is not None
            }
        )
    )

    resolved_device = torch.device(device)
    max_supported = _selector_max_candidates_cap(resolved_device)
    if capacity_classes and capacity_classes[-1] > max_supported:
        raise ValueError(
            "decode-meta prewarm exceeds selector shared-memory capacity: "
            f"requested={capacity_classes[-1]}, supported<={max_supported}"
        )

    total_windows = k + 1
    sort_cap = 1 << (total_windows - 1).bit_length()
    sort_log2 = sort_cap.bit_length() - 1
    device_key = utils.device_cache_key(resolved_device)
    for max_candidates in capacity_classes:
        _get_compiled_decode_meta_kernel(
            dtype=dtype,
            device_key=device_key,
            topk=k,
            total_windows=total_windows,
            sort_cap=sort_cap,
            sort_log2=sort_log2,
            max_candidates=max_candidates,
            use_f16_short_tail=dtype == torch.float16,
            input_nonnegative=bool(score_nonneg),
            use_row_starts=True,
            sort_output=bool(sort_output),
        )
    return capacity_classes


def _cutedsl_topk_selector_decode_meta_sm90_gqa_impl(
    input_tensor: torch.Tensor,
    lengths: torch.Tensor,
    row_starts: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    req_block_offsets: torch.Tensor,
    block_table: torch.Tensor,
    *,
    k: int,
    active_rows: Optional[int] = None,
    score_nonneg: bool = True,
    block_size: int = _REGION_TOKENS,
    window: int = 0,
    block_counts_out: torch.Tensor,
    block_packed_indices_out: torch.Tensor,
    length_delta: int = 0,
    max_candidates: Optional[int] = None,
    logical_num_regions: Optional[int] = None,
    request_indices: Optional[torch.Tensor] = None,
    sort_output: bool = True,
    valid_rows: Optional[torch.Tensor] = None,
    stream=None,
) -> None:
    """Select decode top-k regions and emit sparse decode metadata."""
    if int(block_size) != _REGION_TOKENS:
        raise ValueError(
            "Step3p5 decode-meta selector requires block_size == 8, "
            f"got {block_size}"
        )
    if int(window) != 0:
        raise ValueError(
            "Step3p5 decode-meta selector currently supports window == 0, "
            f"got {window}"
        )
    if int(input_tensor.shape[1]) >= _MULTI_CTA_DECODE_THRESHOLD:
        active_rows_i = (
            int(input_tensor.shape[0]) if active_rows is None else int(active_rows))
        logical_num_regions_i = (
            int(input_tensor.shape[1]) if logical_num_regions is None
            else int(logical_num_regions)
        )
        if logical_num_regions_i <= 0 or logical_num_regions_i > int(input_tensor.shape[1]):
            raise ValueError(
                "logical_num_regions must be in (0, input_tensor.shape[1]], got "
                f"logical_num_regions={logical_num_regions_i}, seq_len={int(input_tensor.shape[1])}"
            )
        # Keep the selector topology tied to the graph-stable score capacity.
        # Row ends carry the logical width, including the valid_len < k case.
        scores_mc = input_tensor[:active_rows_i].contiguous()
        if row_starts is not None:
            starts_mc = row_starts[:active_rows_i].to(torch.int32).contiguous()
        else:
            starts_mc = torch.zeros(
                (active_rows_i,), dtype=torch.int32, device=input_tensor.device)
        ends_mc = torch.clamp(
            starts_mc + lengths[:active_rows_i].to(torch.int32) + int(length_delta),
            max=logical_num_regions_i,
        ).to(torch.int32).contiguous()
        raw_idx = cutedsl_topk_selector_sm90_multi_cta(
            scores_mc,
            starts_mc,
            ends_mc,
            topk=int(k),
            stable_sort=False,
            stream=stream,
        )
        counts, packed = convert_region_block_topk_to_sparse_meta_step3p5(
            raw_idx,
            query_start_loc,
            seq_lens,
            req_block_offsets,
            block_table,
            block_size=int(block_size),
            window=int(window),
            block_counts_out=block_counts_out,
            block_packed_indices_out=block_packed_indices_out,
            valid_seq_q=None if valid_rows is not None else active_rows_i,
            request_indices=request_indices,
            valid_rows=valid_rows,
        )
        if counts.ndim == 2:
            block_counts_out.copy_(counts[:, 0])
        if packed.ndim == 3:
            block_packed_indices_out.copy_(packed[:, 0, :])
        return
    logical_num_regions_i = (
        int(input_tensor.shape[1]) if logical_num_regions is None
        else int(logical_num_regions)
    )
    if logical_num_regions_i <= 0 or logical_num_regions_i > int(input_tensor.shape[1]):
        raise ValueError(
            "logical_num_regions must be in (0, input_tensor.shape[1]], got "
            f"logical_num_regions={logical_num_regions_i}, seq_len={int(input_tensor.shape[1])}"
        )
    max_candidates = _selector_max_candidates_for_input(
        input_tensor, k, max_candidates=max_candidates)
    _validate_topk_public_inputs(
        input_tensor,
        lengths,
        row_starts,
        k=int(k),
        max_candidates=max_candidates,
        out_idx=None,
        active_rows=active_rows,
    )
    active_rows_i = int(input_tensor.shape[0]) if active_rows is None else int(active_rows)
    total_windows = int(k) + 1
    if block_counts_out.device != input_tensor.device or block_packed_indices_out.device != input_tensor.device:
        raise ValueError("decode meta outputs must be on the same device as input_tensor")
    if valid_rows is not None and (
        valid_rows.device != input_tensor.device
        or valid_rows.dtype != torch.int32
        or valid_rows.ndim != 1
        or int(valid_rows.numel()) != 1
        or not valid_rows.is_contiguous()
    ):
        raise ValueError(
            "valid_rows must be a contiguous CUDA int32 tensor with shape [1]"
        )
    if block_counts_out.dtype != torch.int32:
        raise ValueError(f"block_counts_out must be torch.int32, got {block_counts_out.dtype}")
    if block_packed_indices_out.dtype != torch.int64:
        raise ValueError(
            f"block_packed_indices_out must be torch.int64, got {block_packed_indices_out.dtype}"
        )
    if tuple(int(v) for v in block_counts_out.shape) != (active_rows_i,):
        raise ValueError(
            f"block_counts_out shape must be {(active_rows_i,)}, got {tuple(block_counts_out.shape)}"
        )
    if tuple(int(v) for v in block_packed_indices_out.shape) != (active_rows_i, total_windows):
        raise ValueError(
            "block_packed_indices_out shape must be "
            f"{(active_rows_i, total_windows)}, got {tuple(block_packed_indices_out.shape)}"
        )
    if not block_counts_out.is_contiguous() or not block_packed_indices_out.is_contiguous():
        raise ValueError("decode meta outputs must be contiguous")
    for name, tensor, ndim in (
        ("query_start_loc", query_start_loc, 1),
        ("seq_lens", seq_lens, 1),
        ("req_block_offsets", req_block_offsets, 1),
        ("block_table", block_table, 2),
    ):
        if tensor.device != input_tensor.device:
            raise ValueError(f"{name} must be on the same device as input_tensor")
        if tensor.dtype != torch.int32:
            raise ValueError(f"{name} must be torch.int32, got {tensor.dtype}")
        if tensor.ndim != ndim:
            raise ValueError(f"{name} must be {ndim}D, got shape={tuple(tensor.shape)}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous, got stride={tuple(tensor.stride())}")
    num_reqs = int(seq_lens.shape[0])
    if int(query_start_loc.shape[0]) != num_reqs + 1:
        raise ValueError("query_start_loc length must equal len(seq_lens) + 1")
    if int(req_block_offsets.shape[0]) != num_reqs:
        raise ValueError("req_block_offsets length must equal len(seq_lens)")
    if int(block_table.shape[0]) != num_reqs:
        raise ValueError("block_table rows must equal len(seq_lens)")
    if active_rows_i <= 0:
        return
    if stream is not None:
        raise ValueError("top-k CuTeDSL kernels use the TVM-FFI environment stream")
    sort_cap = 1 << (int(total_windows) - 1).bit_length()
    sort_log2 = (int(sort_cap)).bit_length() - 1
    use_f16_short_tail = input_tensor.dtype == torch.float16
    compiled = _get_compiled_decode_meta_kernel(
        dtype=input_tensor.dtype,
        device_key=utils.device_cache_key(input_tensor.device),
        topk=int(k),
        total_windows=int(total_windows),
        sort_cap=int(sort_cap),
        sort_log2=int(sort_log2),
        max_candidates=int(
            _selector_max_candidates_for_input(
                input_tensor,
                k,
                max_candidates=max_candidates,
            )
        ),
        use_f16_short_tail=bool(use_f16_short_tail),
        input_nonnegative=bool(score_nonneg),
        use_row_starts=True,
        sort_output=bool(sort_output),
    )
    compiled(
        input_tensor,
        lengths,
        row_starts,
        query_start_loc,
        seq_lens,
        req_block_offsets,
        block_table,
        block_counts_out,
        block_packed_indices_out,
        active_rows_i,
        int(input_tensor.shape[1]),
        logical_num_regions_i,
        int(num_reqs),
        int(k),
        int(length_delta),
    )
def cutedsl_topk_selector_decode_meta_sm90_gqa(
    input_tensor: torch.Tensor,
    lengths: torch.Tensor,
    row_starts: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    req_block_offsets: torch.Tensor,
    block_table: torch.Tensor,
    *,
    k: int,
    active_rows: Optional[int] = None,
    score_nonneg: bool = True,
    block_size: int = _REGION_TOKENS,
    window: int = 0,
    block_counts_out: torch.Tensor,
    block_packed_indices_out: torch.Tensor,
    length_delta: int = 0,
    max_candidates: Optional[int] = None,
    logical_num_regions: Optional[int] = None,
    request_indices: Optional[torch.Tensor] = None,
    valid_rows: Optional[torch.Tensor] = None,
    stream=None,
) -> None:
    """Select decode top-k regions and emit sorted sparse decode metadata."""
    _cutedsl_topk_selector_decode_meta_sm90_gqa_impl(
        input_tensor,
        lengths,
        row_starts,
        query_start_loc,
        seq_lens,
        req_block_offsets,
        block_table,
        k=int(k),
        active_rows=active_rows,
        score_nonneg=bool(score_nonneg),
        block_size=int(block_size),
        window=int(window),
        block_counts_out=block_counts_out,
        block_packed_indices_out=block_packed_indices_out,
        length_delta=int(length_delta),
        max_candidates=max_candidates,
        logical_num_regions=logical_num_regions,
        request_indices=request_indices,
        sort_output=True,
        valid_rows=valid_rows,
        stream=stream,
    )


def cutedsl_topk_selector_decode_meta_ordered_sm90_gqa(
    input_tensor: torch.Tensor,
    lengths: torch.Tensor,
    row_starts: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    req_block_offsets: torch.Tensor,
    block_table: torch.Tensor,
    *,
    k: int,
    active_rows: Optional[int] = None,
    score_nonneg: bool = True,
    block_size: int = _REGION_TOKENS,
    window: int = 0,
    block_counts_out: torch.Tensor,
    block_packed_indices_out: torch.Tensor,
    length_delta: int = 0,
    max_candidates: Optional[int] = None,
    logical_num_regions: Optional[int] = None,
    request_indices: Optional[torch.Tensor] = None,
    stream=None,
) -> None:
    """Select decode top-k regions and emit previous-topk then current metadata."""
    _cutedsl_topk_selector_decode_meta_sm90_gqa_impl(
        input_tensor,
        lengths,
        row_starts,
        query_start_loc,
        seq_lens,
        req_block_offsets,
        block_table,
        k=int(k),
        active_rows=active_rows,
        score_nonneg=bool(score_nonneg),
        block_size=int(block_size),
        window=int(window),
        block_counts_out=block_counts_out,
        block_packed_indices_out=block_packed_indices_out,
        length_delta=int(length_delta),
        max_candidates=max_candidates,
        logical_num_regions=logical_num_regions,
        request_indices=request_indices,
        sort_output=False,
        stream=stream,
    )

def cutedsl_topk_selector_raw_sm90_gqa(
    input_tensor: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    *,
    active_rows: Optional[int] = None,
    row_starts: Optional[torch.Tensor] = None,
    score_nonneg: bool = True,
    out_idx: Optional[torch.Tensor] = None,
    length_delta: int = 0,
    max_candidates: Optional[int] = None,
    stream=None,
) -> torch.Tensor:
    """Select top-k column ids without prefill physical-region packing."""
    max_candidates = _selector_max_candidates_for_input(
        input_tensor, k, max_candidates=max_candidates)
    use_row_starts = row_starts is not None
    row_starts_tensor = row_starts if row_starts is not None else lengths
    out_idx = _validate_topk_public_inputs(
        input_tensor,
        lengths,
        row_starts_tensor,
        k=k,
        max_candidates=max_candidates,
        out_idx=out_idx,
        active_rows=active_rows,
    )
    active_rows_i = int(input_tensor.shape[0]) if active_rows is None else int(active_rows)
    if out_idx is None:
        out_idx = torch.empty(
            (active_rows_i, int(k)),
            dtype=torch.int32,
            device=input_tensor.device,
        )
    device_key = (input_tensor.device.type, input_tensor.device.index)
    dummy_block_table = _RAW_SELECTOR_DUMMY_BLOCK_TABLES.get(device_key)
    if dummy_block_table is None or dummy_block_table.device != input_tensor.device:
        dummy_block_table = torch.empty(
            (1, 1), dtype=torch.int32, device=input_tensor.device)
        _RAW_SELECTOR_DUMMY_BLOCK_TABLES[device_key] = dummy_block_table
    _cutedsl_topk_selector_prefill_sm90_gqa_impl(
        input_tensor,
        lengths,
        row_starts_tensor,
        lengths,
        topk=int(k),
        max_candidates=max_candidates,
        out_idx=out_idx,
        input_nonnegative=bool(score_nonneg),
        use_row_starts=bool(use_row_starts),
        use_cu_seqlens=False,
        block_table=dummy_block_table,
        seq_q=1,
        active_rows=active_rows_i,
        pack_output=False,
        length_delta=int(length_delta),
        stream=stream,
    )
    return out_idx


__all__ = [
    "cutedsl_topk_selector_prefill_sm90_gqa",
    "cutedsl_topk_selector_raw_sm90_gqa",
    "cutedsl_topk_selector_decode_meta_sm90_gqa",
    "cutedsl_topk_selector_decode_meta_ordered_sm90_gqa",
    "prewarm_topk_selector_decode_meta_sm90_gqa",
    "topk_selector_decode_meta_capacity_class_sm90_gqa",
]
