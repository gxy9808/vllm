from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.kernel_utils import cute_utils as sparse_utils
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.kernel_utils.cute_utils import convert_from_dlpack


_THREADS_PER_BLOCK = 1024
_WARP_SIZE = 32


_WARPS_PER_BLOCK = _THREADS_PER_BLOCK // _WARP_SIZE
_RADIX = 256
_F32_TAIL_ROUNDS = 3
# Note(wangbojun/codex): fp16 scores are still compared through the f32
# ordered-key path for exact tie membership compatibility. After byte24,
# byte16 and byte8 contain the fp16 payload; byte0 is always non-selective.
_F16_TAIL_ROUNDS = 2
_DEFAULT_MAX_CANDIDATES = 4096
_STABLE_SCORE_ROUNDS = 4
_MAX_STABLE_TOPK = 512

_COMPILE_CACHE: Dict[
    Tuple[int, torch.dtype, int, bool, Tuple[str, Optional[int]]],
    cute.JitFunction,
] = {}
_STREAMING_COMPILE_CACHE: Dict[
    Tuple[int, torch.dtype, bool, str, Tuple[str, Optional[int]]],
    cute.JitFunction,
] = {}
_STABLE_SORT_COMPILE_CACHE: Dict[
    Tuple[int, torch.dtype, Tuple[str, Optional[int]]],
    cute.JitFunction,
] = {}
_WORKSPACE_CACHE: Dict[Tuple[int, Tuple[str, Optional[int]]], "_SelectorWorkspace"] = {}
_MULTI_CTA_WORKSPACE_CACHE: Dict[
    Tuple[int, int, int, Tuple[str, Optional[int]]], torch.Tensor
] = {}
_SM_COUNT_CACHE: Dict[Tuple[str, Optional[int]], int] = {}
_COMPUTE_CAPABILITY_CACHE: Dict[Tuple[str, Optional[int]], Tuple[int, int]] = {}
_SEALED_COMPILE_DEVICES: set[Tuple[str, Optional[int]]] = set()


def _device_cache_key(device: torch.device) -> Tuple[str, Optional[int]]:
    if device.type == "cuda":
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        return (device.type, index)
    return (device.type, None)


def _device_sm_count(device: torch.device) -> int:
    key = _device_cache_key(device)
    cached = _SM_COUNT_CACHE.get(key)
    if cached is not None:
        return cached
    count = int(torch.cuda.get_device_properties(device).multi_processor_count)
    _SM_COUNT_CACHE[key] = count
    return count


def _device_is_sm90(device: torch.device) -> bool:
    key = _device_cache_key(device)
    capability = _COMPUTE_CAPABILITY_CACHE.get(key)
    if capability is None:
        capability = tuple(
            int(value) for value in torch.cuda.get_device_capability(device)
        )
        _COMPUTE_CAPABILITY_CACHE[key] = capability
    return capability[0] == 9


def _raise_if_compile_is_forbidden(
    *,
    device_key: Tuple[str, Optional[int]],
    kernel: str,
    key: tuple,
) -> None:
    if device_key in _SEALED_COMPILE_DEVICES:
        raise RuntimeError(
            "top-k CuTeDSL compilation is sealed after serving warmup; "
            f"missing prewarmed {kernel} variant key={key!r}"
        )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "top-k CuTeDSL cache miss during CUDA graph capture; "
            f"prewarm {kernel} variant key={key!r} before capture"
        )


def seal_cutedsl_topk_selector_sm90_compilation(
    device: torch.device | str | None = None,
) -> None:
    """Reject new selector compilation after the serving warmup boundary."""
    resolved = torch.device("cuda") if device is None else torch.device(device)
    device_key = _device_cache_key(resolved)
    _SEALED_COMPILE_DEVICES.add(device_key)

    # The stable selector may dispatch to the Hopper cluster specialization.
    # Seal its independent cache at the same service lifecycle boundary.
    _cluster_selector.seal_cutedsl_topk_selector_sm90_cluster_compilation(
        resolved
    )


def prewarm_cutedsl_topk_selector_sm90_compilation(
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    topk: int,
    max_seq_len: int,
    stable_sort: bool,
) -> None:
    """Compile the finite selector plan set used up to ``max_seq_len``."""
    resolved = torch.device(device)
    topk = int(topk)
    max_seq_len = int(max_seq_len)
    if topk <= 0 or max_seq_len < topk:
        raise ValueError(
            "top-k prewarm requires 0 < topk <= max_seq_len, "
            f"got topk={topk}, max_seq_len={max_seq_len}"
        )

    input_tensor = torch.empty((1, 1), dtype=dtype, device=resolved)
    starts = torch.empty((1,), dtype=torch.int32, device=resolved)
    ends = torch.empty((1,), dtype=torch.int32, device=resolved)
    out_idx = torch.empty((1, topk), dtype=torch.int32, device=resolved)

    _get_compiled_streaming_kernel(
        input_tensor=input_tensor,
        starts=starts,
        ends=ends,
        out_idx=out_idx,
        candidate_idx=out_idx,
        topk=topk,
        stable_sort=bool(stable_sort),
        source_kind="dense",
    )

    max_chunks = max(1, max_seq_len // topk)
    if max_chunks >= 2:
        candidate_idx = torch.empty(
            (2, topk), dtype=torch.int32, device=resolved
        )
        _get_compiled_streaming_kernel(
            input_tensor=input_tensor,
            starts=starts,
            ends=ends,
            out_idx=candidate_idx,
            candidate_idx=candidate_idx,
            topk=topk,
            stable_sort=False,
            source_kind="chunk",
        )
        _get_compiled_streaming_kernel(
            input_tensor=input_tensor,
            starts=starts,
            ends=ends,
            out_idx=out_idx,
            candidate_idx=candidate_idx,
            topk=topk,
            stable_sort=bool(stable_sort),
            source_kind="merge",
        )

    if stable_sort and topk == _MAX_STABLE_TOPK:
        cluster_input = torch.zeros(
            (1, _MAX_STABLE_TOPK), dtype=torch.float32, device=resolved
        )
        cluster_starts = torch.zeros((1,), dtype=torch.int32, device=resolved)
        cluster_ends = torch.full(
            (1,), _MAX_STABLE_TOPK, dtype=torch.int32, device=resolved
        )
        cluster_out = torch.empty(
            (1, _MAX_STABLE_TOPK), dtype=torch.int32, device=resolved
        )
        for cluster_size in (4, 8, 16):
            _cluster_selector.cutedsl_topk_selector_sm90_cluster(
                cluster_input,
                cluster_starts,
                cluster_ends,
                out_idx=cluster_out,
                cluster_size=cluster_size,
                threads_per_cta=512,
            )


@dataclass
class _SelectorWorkspace:
    hist: torch.Tensor
    cand: torch.Tensor
    cnt: torch.Tensor
    remain: torch.Tensor
    threshold: torch.Tensor


def _is_torch_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is None:
        return False
    is_compiling = getattr(compiler, "is_compiling", None)
    if is_compiling is None:
        return False
    return bool(is_compiling())


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
def _atomic_add_i32(ptr: cute.Pointer, val: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        nvvm.atomicrmw(
            op=nvvm.AtomicOpKind.ADD,
            ptr=ptr.llvm_ptr,
            a=Int32(val).ir_value(loc=loc, ip=ip),
        )
    )


@dsl_user_op
def _lane_mask_lt_u32(*, loc=None, ip=None) -> Uint32:
    return Uint32(nvvm.read_ptx_sreg_lanemask_lt(T.i32(), loc=loc, ip=ip))


@dsl_user_op
def _elem_pointer(
    x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None
) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


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
def _warp_prefix_sum_i32(value: Int32, lane_idx: Int32) -> Int32:
    for step in cutlass.range_constexpr(5):
        offset = 1 << step
        partial = cute.arch.shuffle_sync_up(value, offset=offset, mask_and_clamp=0)
        if lane_idx >= Int32(offset):
            value += partial
    return value


@cute.jit
def _hist_prefix_desc_inplace(sHist: cute.Tensor):
    running = Int32(0)
    i = Int32(_RADIX - 1)
    while i >= Int32(0):
        running += sHist[i]
        sHist[i] = running
        i -= Int32(1)


@cute.jit
def _hist_select_threshold(sHist: cute.Tensor, remain: Int32) -> Int32:
    threshold = Int32(0)
    i = Int32(0)
    while i < Int32(_RADIX - 1):
        if sHist[i] > remain and sHist[i + Int32(1)] <= remain:
            threshold = i
        i += Int32(1)
    return threshold


@cute.jit
def _hist_select_threshold_inclusive(sHist: cute.Tensor, remain: Int32) -> Int32:
    threshold = Int32(0)
    i = Int32(0)
    while i < Int32(_RADIX):
        if sHist[i] > remain and sHist[i + Int32(1)] <= remain:
            threshold = i
        i += Int32(1)
    return threshold


def _round_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _selector_required_smem_bytes(max_candidates: int) -> int:
    total = 0
    total = _round_up(total, 128) + (_RADIX + 1) * 4
    total = _round_up(total, 128) + _WARPS_PER_BLOCK * (_RADIX + 1) * 4
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


def _make_topk_selector_kernel(max_candidates: int, *, use_f16_short_tail: bool):
    max_cand = int(max_candidates)
    max_cand_i32 = Int32(max_cand)
    tail_rounds = _F16_TAIL_ROUNDS if use_f16_short_tail else _F32_TAIL_ROUNDS

    @cute.kernel
    def _topk_selector_kernel_sm90(
        mInput: cute.Tensor,
        mOutIdx: cute.Tensor,
        mStarts: cute.Tensor,
        mEnds: cute.Tensor,
        mHist: cute.Tensor,
        mCand: cute.Tensor,
        mCnt: cute.Tensor,
        mRemain: cute.Tensor,
        mThreshold: cute.Tensor,
        batch: Int32,
        seq_len: Int32,
        topk: Int32,
    ):
        bx, _, _ = cute.arch.block_idx()
        tx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()

        @cute.struct
        class SharedStorage:
            hist: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, _RADIX + 1], 128
            ]
            hist_warp: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, _WARPS_PER_BLOCK * (_RADIX + 1)],
                128,
            ]
            cand: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 2 * max_cand], 128
            ]
            cnt: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 2], 16]
            remain: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 8]
            threshold: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 8]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage, 16)
        sHist = storage.hist.get_tensor(cute.make_layout((_RADIX + 1,), stride=(1,)))
        sHistWarp = storage.hist_warp.get_tensor(
            cute.make_layout((_WARPS_PER_BLOCK, _RADIX + 1), stride=(_RADIX + 1, 1))
        )
        sCand = storage.cand.get_tensor(
            cute.make_layout((2, max_cand), stride=(max_cand, 1))
        )
        sCnt = storage.cnt.get_tensor(cute.make_layout((2,), stride=(1,)))
        sRemain = storage.remain.get_tensor(cute.make_layout((1,), stride=(1,)))
        sThreshold = storage.threshold.get_tensor(cute.make_layout((1,), stride=(1,)))
        remain = Int32(0)
        threshold = Int32(0)

        if bx < batch:
            start_idx = mStarts[bx]
            end_idx = mEnds[bx]
            if end_idx > seq_len:
                end_idx = seq_len
            if start_idx < Int32(0):
                start_idx = Int32(0)
            valid_len = end_idx - start_idx
            if valid_len < Int32(0):
                valid_len = Int32(0)

            for i_chunk in cutlass.range_constexpr(
                (_RADIX + 1 + _WARP_SIZE - 1) // _WARP_SIZE
            ):
                i = lane_idx + Int32(i_chunk * _WARP_SIZE)
                if i <= Int32(_RADIX):
                    sHistWarp[warp_idx, i] = Int32(0)

            if tx < Int32(2):
                sCnt[tx] = Int32(0)
            if tx == Int32(0):
                sRemain[Int32(0)] = topk
                sThreshold[Int32(0)] = Int32(0)
            cute.arch.sync_threads()

            num_tiles = (valid_len + Int32(_THREADS_PER_BLOCK - 1)) // Int32(
                _THREADS_PER_BLOCK
            )
            for tile in cutlass.range(num_tiles, unroll=1):
                input_idx = start_idx + tile * Int32(_THREADS_PER_BLOCK) + tx
                if input_idx < end_idx:
                    bin_id = _radix_byte_from_f32(
                        Float32(mInput[bx, input_idx]), shift=24
                    )
                    _atomic_add_i32(
                        _elem_pointer(sHistWarp, (warp_idx, bin_id)), Int32(1)
                    )
            cute.arch.sync_threads()

            for i_chunk in cutlass.range_constexpr(
                (_RADIX + 1 + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
            ):
                i = tx + Int32(i_chunk * _THREADS_PER_BLOCK)
                if i <= Int32(_RADIX):
                    hist_i = Int32(0)
                    for w in cutlass.range_constexpr(_WARPS_PER_BLOCK):
                        hist_i += sHistWarp[Int32(w), i]
                    sHist[i] = hist_i
            cute.arch.sync_threads()

            if tx == Int32(0):
                remain = sRemain[Int32(0)]
                _hist_prefix_desc_inplace(sHist)
                threshold = _hist_select_threshold(sHist, remain)
                sThreshold[Int32(0)] = threshold
                sRemain[Int32(0)] = remain - sHist[threshold + Int32(1)]
            cute.arch.sync_threads()

            for tile in cutlass.range(num_tiles, unroll=1):
                input_idx = start_idx + tile * Int32(_THREADS_PER_BLOCK) + tx
                if input_idx < end_idx:
                    bin_id = _radix_byte_from_f32(
                        Float32(mInput[bx, input_idx]), shift=24
                    )
                    threshold = sThreshold[Int32(0)]
                    if bin_id > threshold:
                        pos_out = _atomic_add_i32(
                            _elem_pointer(sHist, (bin_id + Int32(1),)), Int32(1)
                        )
                        if pos_out < topk:
                            mOutIdx[bx, pos_out] = input_idx
                    elif bin_id == threshold:
                        if sRemain[Int32(0)] > Int32(0):
                            pos_cand = _atomic_add_i32(
                                _elem_pointer(sCnt, (Int32(0),)), Int32(1)
                            )
                            if pos_cand < max_cand_i32:
                                sCand[Int32(0), pos_cand] = input_idx
            cute.arch.sync_threads()

            for round_i in cutlass.range_constexpr(tail_rounds):
                cur_buf = Int32(round_i & 1)
                nxt_buf = Int32((round_i + 1) & 1)

                if sRemain[Int32(0)] > Int32(0):
                    for i_chunk in cutlass.range_constexpr(
                        (_RADIX + 1 + _WARP_SIZE - 1) // _WARP_SIZE
                    ):
                        i = lane_idx + Int32(i_chunk * _WARP_SIZE)
                        if i <= Int32(_RADIX):
                            sHistWarp[warp_idx, i] = Int32(0)
                    if tx == Int32(0):
                        sCnt[nxt_buf] = Int32(0)
                    cute.arch.sync_threads()

                    # Note(wangbojun/codex): The counter tracks the full
                    # threshold bucket cardinality, while sCand only stores
                    # max_candidates entries. Re-reading past that shared
                    # buffer corrupts memory on low-entropy/profile inputs.
                    num_input = cutlass.min(sCnt[cur_buf], max_cand_i32)
                    num_tiles_round = (
                        num_input + Int32(_THREADS_PER_BLOCK - 1)
                    ) // Int32(_THREADS_PER_BLOCK)
                    for tile in cutlass.range(num_tiles_round, unroll=1):
                        cand_pos = tile * Int32(_THREADS_PER_BLOCK) + tx
                        if cand_pos < num_input:
                            input_idx = sCand[cur_buf, cand_pos]
                            bin_id = _radix_byte_from_f32(
                                Float32(mInput[bx, input_idx]),
                                shift=16 - round_i * 8,
                            )
                            _atomic_add_i32(
                                _elem_pointer(sHistWarp, (warp_idx, bin_id)), Int32(1)
                            )
                    cute.arch.sync_threads()

                    for i_chunk in cutlass.range_constexpr(
                        (_RADIX + 1 + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
                    ):
                        i = tx + Int32(i_chunk * _THREADS_PER_BLOCK)
                        if i <= Int32(_RADIX):
                            hist_i = Int32(0)
                            for w in cutlass.range_constexpr(_WARPS_PER_BLOCK):
                                hist_i += sHistWarp[Int32(w), i]
                            sHist[i] = hist_i
                    cute.arch.sync_threads()

                    if tx == Int32(0):
                        remain = sRemain[Int32(0)]
                        _hist_prefix_desc_inplace(sHist)
                        threshold = _hist_select_threshold(sHist, remain)
                        sThreshold[Int32(0)] = threshold
                        sRemain[Int32(0)] = remain - sHist[threshold + Int32(1)]
                    cute.arch.sync_threads()

                    remain_old = (
                        sRemain[Int32(0)] + sHist[sThreshold[Int32(0)] + Int32(1)]
                    )
                    start_pos = topk - remain_old
                    for tile in cutlass.range(num_tiles_round, unroll=1):
                        cand_pos = tile * Int32(_THREADS_PER_BLOCK) + tx
                        if cand_pos < num_input:
                            input_idx = sCand[cur_buf, cand_pos]
                            bin_id = _radix_byte_from_f32(
                                Float32(mInput[bx, input_idx]),
                                shift=16 - round_i * 8,
                            )
                            threshold = sThreshold[Int32(0)]
                            if bin_id > threshold:
                                pos_out = (
                                    _atomic_add_i32(
                                        _elem_pointer(sHist, (bin_id + Int32(1),)),
                                        Int32(1),
                                    )
                                    + start_pos
                                )
                                if pos_out < topk:
                                    mOutIdx[bx, pos_out] = input_idx
                            elif bin_id == threshold:
                                if round_i == tail_rounds - 1:
                                    pos_tail = (
                                        _atomic_add_i32(
                                            _elem_pointer(sHist, (bin_id + Int32(1),)),
                                            Int32(1),
                                        )
                                        + start_pos
                                    )
                                    if pos_tail < topk:
                                        mOutIdx[bx, pos_tail] = input_idx
                                elif sRemain[Int32(0)] > Int32(0):
                                    pos_next = _atomic_add_i32(
                                        _elem_pointer(sCnt, (nxt_buf,)), Int32(1)
                                    )
                                    if pos_next < max_cand_i32:
                                        sCand[nxt_buf, pos_next] = input_idx
                    cute.arch.sync_threads()

    return _topk_selector_kernel_sm90


def _make_launch_topk_selector_kernel(max_candidates: int, *, use_f16_short_tail: bool):
    kernel = _make_topk_selector_kernel(
        max_candidates, use_f16_short_tail=use_f16_short_tail
    )

    @cute.jit
    def _launch_topk_selector_kernel(
        mInput: cute.Tensor,
        mOutIdx: cute.Tensor,
        mStarts: cute.Tensor,
        mEnds: cute.Tensor,
        mHist: cute.Tensor,
        mCand: cute.Tensor,
        mCnt: cute.Tensor,
        mRemain: cute.Tensor,
        mThreshold: cute.Tensor,
        batch: int,
        seq_len: int,
        topk: int,
        stream,
    ):
        kernel(
            mInput,
            mOutIdx,
            mStarts,
            mEnds,
            mHist,
            mCand,
            mCnt,
            mRemain,
            mThreshold,
            Int32(batch),
            Int32(seq_len),
            Int32(topk),
        ).launch(
            grid=[batch, 1, 1],
            block=[_THREADS_PER_BLOCK, 1, 1],
            stream=stream,
        )

    return _launch_topk_selector_kernel


def _make_launch_topk_selector_kernel_k64(
    max_candidates: int, *, use_f16_short_tail: bool
):
    kernel = _make_topk_selector_kernel(
        max_candidates, use_f16_short_tail=use_f16_short_tail
    )

    @cute.jit
    def _launch_topk_selector_kernel_k64(
        mInput: cute.Tensor,
        mOutIdx: cute.Tensor,
        mStarts: cute.Tensor,
        mEnds: cute.Tensor,
        mHist: cute.Tensor,
        mCand: cute.Tensor,
        mCnt: cute.Tensor,
        mRemain: cute.Tensor,
        mThreshold: cute.Tensor,
        batch: int,
        seq_len: int,
        stream,
    ):
        kernel(
            mInput,
            mOutIdx,
            mStarts,
            mEnds,
            mHist,
            mCand,
            mCnt,
            mRemain,
            mThreshold,
            Int32(batch),
            Int32(seq_len),
            Int32(64),
        ).launch(
            grid=[batch, 1, 1],
            block=[_THREADS_PER_BLOCK, 1, 1],
            stream=stream,
        )

    return _launch_topk_selector_kernel_k64


def _make_topk_selector_streaming_kernel(
    topk: int,
    *,
    stable_sort: bool,
    source_kind: str = "dense",
):
    if source_kind not in ("dense", "chunk", "merge"):
        raise ValueError(f"unsupported streaming selector source_kind={source_kind}")

    topk_i32 = Int32(int(topk))
    use_stable_sort = bool(stable_sort)
    use_stable_membership = source_kind == "chunk"
    use_chunk_source = source_kind == "chunk"
    use_merge_source = source_kind == "merge"
    radix_rounds = _STABLE_SCORE_ROUNDS
    sort_items = 1 << (int(topk) - 1).bit_length()
    sort_stages = sort_items.bit_length() - 1
    selected_items = sort_items if use_stable_sort else 1

    @cute.kernel
    def _topk_selector_streaming_kernel_sm90(
        mInput: cute.Tensor,
        mOutIdx: cute.Tensor,
        mStarts: cute.Tensor,
        mEnds: cute.Tensor,
        mCandidateIdx: cute.Tensor,
        batch: Int32,
        seq_len: Int32,
        chunk_size: Int32,
        num_chunks: Int32,
    ):
        bx, _, _ = cute.arch.block_idx()
        tx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()

        @cute.struct
        class SharedStorage:
            hist: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, _RADIX + 1], 128
            ]
            hist_warp: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, _WARPS_PER_BLOCK * (_RADIX + 1)],
                128,
            ]
            cnt: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 2], 16]
            remain: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 8]
            threshold: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 8]
            prefix: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, radix_rounds], 16
            ]
            selected_ids: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, selected_items], 128
            ]
            selected_values: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, selected_items], 128
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage, 16)
        sHist = storage.hist.get_tensor(cute.make_layout((_RADIX + 1,), stride=(1,)))
        sHistWarp = storage.hist_warp.get_tensor(
            cute.make_layout((_WARPS_PER_BLOCK, _RADIX + 1), stride=(_RADIX + 1, 1))
        )
        sCnt = storage.cnt.get_tensor(cute.make_layout((2,), stride=(1,)))
        sRemain = storage.remain.get_tensor(cute.make_layout((1,), stride=(1,)))
        sThreshold = storage.threshold.get_tensor(cute.make_layout((1,), stride=(1,)))
        sPrefix = storage.prefix.get_tensor(
            cute.make_layout((radix_rounds,), stride=(1,))
        )
        sSelectedIds = storage.selected_ids.get_tensor(
            cute.make_layout((selected_items,), stride=(1,))
        )
        sSelectedValues = storage.selected_values.get_tensor(
            cute.make_layout((selected_items,), stride=(1,))
        )
        threshold = Int32(0)
        gt_count = Int32(0)
        fill_remaining = Int32(0)

        row = bx
        chunk_idx = Int32(0)
        if cutlass.const_expr(use_chunk_source):
            row = bx // num_chunks
            chunk_idx = bx - row * num_chunks

        if row < batch:
            if cutlass.const_expr(use_stable_sort):
                if tx < Int32(sort_items):
                    sSelectedIds[tx] = Int32(-1)
                    sSelectedValues[tx] = Float32(0.0)
            else:
                for tile in cutlass.range_constexpr(
                    (int(topk) + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
                ):
                    pos = tx + Int32(tile * _THREADS_PER_BLOCK)
                    if pos < topk_i32:
                        mOutIdx[bx, pos] = Int32(-1)

            start_idx = Int32(0)
            end_idx = num_chunks * topk_i32
            if cutlass.const_expr(not use_merge_source):
                start_idx = mStarts[row]
                end_idx = mEnds[row]
                if end_idx > seq_len:
                    end_idx = seq_len
                if start_idx < Int32(0):
                    start_idx = Int32(0)
            if cutlass.const_expr(use_chunk_source):
                chunk_begin = chunk_idx * chunk_size
                chunk_end = chunk_begin + chunk_size
                if chunk_end > seq_len:
                    chunk_end = seq_len
                if start_idx < chunk_begin:
                    start_idx = chunk_begin
                if end_idx > chunk_end:
                    end_idx = chunk_end
            valid_len = end_idx - start_idx
            if valid_len < Int32(0):
                valid_len = Int32(0)

            if tx < Int32(radix_rounds):
                sPrefix[tx] = Int32(0)
            if tx == Int32(0):
                sRemain[Int32(0)] = topk_i32
                sThreshold[Int32(0)] = Int32(0)
            cute.arch.sync_threads()

            num_tiles = (valid_len + Int32(_THREADS_PER_BLOCK - 1)) // Int32(
                _THREADS_PER_BLOCK
            )

            for round_i in cutlass.range_constexpr(radix_rounds):
                for i_chunk in cutlass.range_constexpr(
                    (_RADIX + 1 + _WARP_SIZE - 1) // _WARP_SIZE
                ):
                    i = lane_idx + Int32(i_chunk * _WARP_SIZE)
                    if i <= Int32(_RADIX):
                        sHistWarp[warp_idx, i] = Int32(0)
                cute.arch.sync_threads()

                for tile in cutlass.range(num_tiles, unroll=1):
                    input_idx = start_idx + tile * Int32(_THREADS_PER_BLOCK) + tx
                    item_id = input_idx
                    item_value = Float32(0.0)
                    item_valid = input_idx < end_idx
                    if cutlass.const_expr(use_merge_source):
                        item_valid = False
                        if input_idx < end_idx:
                            candidate_chunk = input_idx // topk_i32
                            candidate_slot = input_idx - candidate_chunk * topk_i32
                            item_id = mCandidateIdx[
                                row * num_chunks + candidate_chunk,
                                candidate_slot,
                            ]
                            if item_id >= Int32(0) and item_id < seq_len:
                                item_valid = True
                    if item_valid:
                        item_value = Float32(mInput[row, item_id])
                        matches_prefix = True
                        for prev_i in cutlass.range_constexpr(radix_rounds):
                            if prev_i < round_i:
                                prev_byte = _radix_byte_from_f32(
                                    item_value,
                                    shift=24 - prev_i * 8,
                                )
                                if prev_byte != sPrefix[Int32(prev_i)]:
                                    matches_prefix = False
                        if matches_prefix:
                            bin_id = _radix_byte_from_f32(
                                item_value,
                                shift=24 - round_i * 8,
                            )
                            _atomic_add_i32(
                                _elem_pointer(sHistWarp, (warp_idx, bin_id)), Int32(1)
                            )
                cute.arch.sync_threads()

                for i_chunk in cutlass.range_constexpr(
                    (_RADIX + 1 + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
                ):
                    i = tx + Int32(i_chunk * _THREADS_PER_BLOCK)
                    if i <= Int32(_RADIX):
                        hist_i = Int32(0)
                        for w in cutlass.range_constexpr(_WARPS_PER_BLOCK):
                            hist_i += sHistWarp[Int32(w), i]
                        sHist[i] = hist_i
                cute.arch.sync_threads()

                round_remain = sRemain[Int32(0)]
                bin_count = Int32(0)
                bin_id = Int32(0)
                inclusive = Int32(0)
                if tx < Int32(_RADIX):
                    bin_id = Int32(_RADIX - 1) - tx
                    bin_count = sHist[bin_id]
                    inclusive = _warp_prefix_sum_i32(bin_count, lane_idx)
                    if lane_idx == Int32(_WARP_SIZE - 1):
                        sHistWarp[warp_idx, Int32(0)] = inclusive
                cute.arch.sync_threads()

                if tx == Int32(0):
                    total = Int32(0)
                    for w in cutlass.range_constexpr(_RADIX // _WARP_SIZE):
                        warp_count = sHistWarp[Int32(w), Int32(0)]
                        sHistWarp[Int32(w), Int32(1)] = total
                        total += warp_count
                    sThreshold[Int32(0)] = Int32(0)
                    sPrefix[Int32(round_i)] = Int32(0)
                    sRemain[Int32(0)] = round_remain - (total - sHist[Int32(0)])
                cute.arch.sync_threads()

                if tx < Int32(_RADIX):
                    count_greater = (
                        sHistWarp[warp_idx, Int32(1)] + inclusive - bin_count
                    )
                    if (
                        bin_count > Int32(0)
                        and count_greater <= round_remain
                        and round_remain < count_greater + bin_count
                    ):
                        sThreshold[Int32(0)] = bin_id
                        sPrefix[Int32(round_i)] = bin_id
                        sRemain[Int32(0)] = round_remain - count_greater
                cute.arch.sync_threads()

            if tx < Int32(2):
                sCnt[tx] = Int32(0)
            cute.arch.sync_threads()

            for tile in cutlass.range(num_tiles, unroll=1):
                input_idx = start_idx + tile * Int32(_THREADS_PER_BLOCK) + tx
                item_id = input_idx
                item_value = Float32(0.0)
                item_valid = input_idx < end_idx
                if cutlass.const_expr(use_merge_source):
                    item_valid = False
                    if input_idx < end_idx:
                        candidate_chunk = input_idx // topk_i32
                        candidate_slot = input_idx - candidate_chunk * topk_i32
                        item_id = mCandidateIdx[
                            row * Int32(num_chunks) + candidate_chunk,
                            candidate_slot,
                        ]
                        if item_id >= Int32(0) and item_id < seq_len:
                            item_valid = True
                if item_valid:
                    item_value = Float32(mInput[row, item_id])
                    is_gt = False
                    is_eq = True
                    for cmp_i in cutlass.range_constexpr(radix_rounds):
                        byte = _radix_byte_from_f32(
                            item_value,
                            shift=24 - cmp_i * 8,
                        )
                        threshold = sPrefix[Int32(cmp_i)]
                        if is_eq and byte > threshold:
                            is_gt = True
                        if is_eq and byte != threshold:
                            is_eq = False
                    if is_gt:
                        pos_out = _atomic_add_i32(
                            _elem_pointer(sCnt, (Int32(0),)), Int32(1)
                        )
                        if pos_out < topk_i32:
                            if cutlass.const_expr(use_stable_sort):
                                sSelectedIds[pos_out] = item_id
                                sSelectedValues[pos_out] = item_value
                            elif cutlass.const_expr(not use_stable_membership):
                                mOutIdx[bx, pos_out] = item_id
            cute.arch.sync_threads()

            gt_count = sCnt[Int32(0)]
            fill_remaining = topk_i32 - gt_count
            if fill_remaining < Int32(0):
                fill_remaining = Int32(0)
            if cutlass.const_expr(use_stable_membership):
                if tx == Int32(0):
                    sThreshold[Int32(0)] = Int32(0)
            cute.arch.sync_threads()

            for tile in cutlass.range(num_tiles, unroll=1):
                input_idx = start_idx + tile * Int32(_THREADS_PER_BLOCK) + tx
                item_id = input_idx
                item_value = Float32(0.0)
                item_valid = input_idx < end_idx
                if cutlass.const_expr(use_merge_source):
                    item_valid = False
                    if input_idx < end_idx:
                        candidate_chunk = input_idx // topk_i32
                        candidate_slot = input_idx - candidate_chunk * topk_i32
                        item_id = mCandidateIdx[
                            row * Int32(num_chunks) + candidate_chunk,
                            candidate_slot,
                        ]
                        if item_id >= Int32(0) and item_id < seq_len:
                            item_valid = True
                is_eq = False
                is_gt = False
                if item_valid:
                    item_value = Float32(mInput[row, item_id])
                    is_eq = True
                    for cmp_i in cutlass.range_constexpr(radix_rounds):
                        byte = _radix_byte_from_f32(
                            item_value,
                            shift=24 - cmp_i * 8,
                        )
                        threshold = sPrefix[Int32(cmp_i)]
                        if is_eq and byte > threshold:
                            is_gt = True
                        if is_eq and byte != threshold:
                            is_eq = False

                if cutlass.const_expr(use_stable_membership):
                    eq_mask = cute.arch.vote_ballot_sync(is_eq)
                    eq_lane_rank = Int32(
                        cute.arch.popc(Uint32(eq_mask) & _lane_mask_lt_u32())
                    )
                    if lane_idx == Int32(0):
                        sHistWarp[warp_idx, Int32(0)] = Int32(cute.arch.popc(eq_mask))
                    cute.arch.sync_threads()

                    if tx == Int32(0):
                        eq_base = sCnt[Int32(1)]
                        eq_count = Int32(0)
                        for w in cutlass.range_constexpr(_WARPS_PER_BLOCK):
                            warp_count = sHistWarp[Int32(w), Int32(0)]
                            sHistWarp[Int32(w), Int32(1)] = eq_count
                            eq_count += warp_count
                        sRemain[Int32(0)] = eq_base
                        sCnt[Int32(1)] = eq_base + eq_count
                    cute.arch.sync_threads()

                    membership_eq_rank = (
                        sRemain[Int32(0)] + sHistWarp[warp_idx, Int32(1)] + eq_lane_rank
                    )
                    selected = is_gt | (is_eq & (membership_eq_rank < fill_remaining))
                    selected_mask = cute.arch.vote_ballot_sync(selected)
                    selected_lane_rank = Int32(
                        cute.arch.popc(Uint32(selected_mask) & _lane_mask_lt_u32())
                    )
                    if lane_idx == Int32(0):
                        sHistWarp[warp_idx, Int32(2)] = Int32(
                            cute.arch.popc(selected_mask)
                        )
                    cute.arch.sync_threads()

                    if tx == Int32(0):
                        selected_base = sThreshold[Int32(0)]
                        selected_count = Int32(0)
                        for w in cutlass.range_constexpr(_WARPS_PER_BLOCK):
                            warp_count = sHistWarp[Int32(w), Int32(2)]
                            sHistWarp[Int32(w), Int32(3)] = selected_count
                            selected_count += warp_count
                        sRemain[Int32(0)] = selected_base
                        sThreshold[Int32(0)] = selected_base + selected_count
                    cute.arch.sync_threads()

                    if selected:
                        out_pos = (
                            sRemain[Int32(0)]
                            + sHistWarp[warp_idx, Int32(3)]
                            + selected_lane_rank
                        )
                        if out_pos < topk_i32:
                            mOutIdx[bx, out_pos] = item_id
                    cute.arch.sync_threads()
                elif cutlass.const_expr(use_stable_sort):
                    eq_mask = cute.arch.vote_ballot_sync(is_eq)
                    lane_rank = Int32(
                        cute.arch.popc(Uint32(eq_mask) & _lane_mask_lt_u32())
                    )
                    if lane_idx == Int32(0):
                        sHistWarp[warp_idx, Int32(0)] = Int32(cute.arch.popc(eq_mask))
                    cute.arch.sync_threads()

                    if tx == Int32(0):
                        tile_base = sCnt[Int32(1)]
                        tile_count = Int32(0)
                        for w in cutlass.range_constexpr(_WARPS_PER_BLOCK):
                            warp_count = sHistWarp[Int32(w), Int32(0)]
                            sHistWarp[Int32(w), Int32(1)] = tile_count
                            tile_count += warp_count
                        sRemain[Int32(0)] = tile_base
                        sCnt[Int32(1)] = tile_base + tile_count
                    cute.arch.sync_threads()

                    if is_eq:
                        eq_rank = (
                            sRemain[Int32(0)]
                            + sHistWarp[warp_idx, Int32(1)]
                            + lane_rank
                        )
                        if eq_rank < fill_remaining:
                            out_pos = gt_count + eq_rank
                            if out_pos < topk_i32:
                                sSelectedIds[out_pos] = item_id
                                sSelectedValues[out_pos] = item_value
                    cute.arch.sync_threads()
                else:
                    if is_eq:
                        pos_eq = _atomic_add_i32(
                            _elem_pointer(sCnt, (Int32(1),)), Int32(1)
                        )
                        if pos_eq < fill_remaining:
                            out_pos = gt_count + pos_eq
                            if out_pos < topk_i32:
                                mOutIdx[bx, out_pos] = item_id

            if cutlass.const_expr(use_stable_sort):
                for stage in cutlass.range_constexpr(sort_stages):
                    for pass_i in cutlass.range_constexpr(stage + 1):
                        stride = Int32(1 << (stage - pass_i))
                        partner = tx ^ stride
                        if tx < Int32(sort_items) and tx < partner:
                            left_id = sSelectedIds[tx]
                            right_id = sSelectedIds[partner]
                            left_value = sSelectedValues[tx]
                            right_value = sSelectedValues[partner]
                            left_valid = left_id >= Int32(0)
                            right_valid = right_id >= Int32(0)
                            left_key = _ordered_u32_from_f32(left_value) ^ Int32(
                                -2147483648
                            )
                            right_key = _ordered_u32_from_f32(right_value) ^ Int32(
                                -2147483648
                            )
                            left_better = (left_valid & ~right_valid) | (
                                left_valid
                                & right_valid
                                & (
                                    (left_key > right_key)
                                    | ((left_key == right_key) & (left_id < right_id))
                                )
                            )
                            descending = (tx & Int32(1 << (stage + 1))) == Int32(0)
                            should_swap = cutlass.select_(
                                descending, ~left_better, left_better
                            )
                            if should_swap:
                                sSelectedIds[tx] = right_id
                                sSelectedValues[tx] = right_value
                                sSelectedIds[partner] = left_id
                                sSelectedValues[partner] = left_value
                        cute.arch.sync_threads()

                if tx < topk_i32:
                    mOutIdx[bx, tx] = sSelectedIds[tx]

    return _topk_selector_streaming_kernel_sm90


def _make_launch_topk_selector_streaming_kernel(
    topk: int,
    *,
    stable_sort: bool,
    source_kind: str = "dense",
):
    kernel = _make_topk_selector_streaming_kernel(
        topk,
        stable_sort=stable_sort,
        source_kind=source_kind,
    )

    @cute.jit
    def _launch_topk_selector_streaming_kernel(
        mInput: cute.Tensor,
        mOutIdx: cute.Tensor,
        mStarts: cute.Tensor,
        mEnds: cute.Tensor,
        mCandidateIdx: cute.Tensor,
        batch: int,
        seq_len: int,
        chunk_size: int,
        num_chunks: int,
        stream,
    ):
        kernel(
            mInput,
            mOutIdx,
            mStarts,
            mEnds,
            mCandidateIdx,
            Int32(batch),
            Int32(seq_len),
            Int32(chunk_size),
            Int32(num_chunks),
        ).launch(
            grid=[batch * num_chunks if source_kind == "chunk" else batch, 1, 1],
            block=[_THREADS_PER_BLOCK, 1, 1],
            stream=stream,
        )

    return _launch_topk_selector_streaming_kernel


def _make_stable_sort_selected_indices_kernel(topk: int):
    sort_items = 1 << (int(topk) - 1).bit_length()
    sort_stages = sort_items.bit_length() - 1

    @cute.kernel
    def _stable_sort_selected_indices_kernel(
        mInput: cute.Tensor,
        mOutIdx: cute.Tensor,
        batch: Int32,
        topk_runtime: Int32,
    ):
        bx, _, _ = cute.arch.block_idx()
        tx = cute.arch.thread_idx()[0]

        @cute.struct
        class SharedStorage:
            ids: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, sort_items], 128]
            values: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, sort_items], 128
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage, 16)
        s_ids = storage.ids.get_tensor(cute.make_layout((sort_items,), stride=(1,)))
        s_values = storage.values.get_tensor(
            cute.make_layout((sort_items,), stride=(1,))
        )

        if bx < batch:
            if tx < Int32(sort_items):
                if tx < topk_runtime:
                    logical_region = mOutIdx[bx, tx]
                    safe_region = cutlass.select_(
                        logical_region >= Int32(0), logical_region, Int32(0)
                    )
                    s_ids[tx] = logical_region
                    s_values[tx] = cutlass.select_(
                        logical_region >= Int32(0),
                        Float32(mInput[bx, safe_region]),
                        Float32(0.0),
                    )
                else:
                    s_ids[tx] = Int32(-1)
                    s_values[tx] = Float32(0.0)
            cute.arch.sync_threads()

            for stage in cutlass.range_constexpr(sort_stages):
                for pass_i in cutlass.range_constexpr(stage + 1):
                    stride = Int32(1 << (stage - pass_i))
                    partner = tx ^ stride
                    if tx < Int32(sort_items) and tx < partner:
                        left_id = s_ids[tx]
                        right_id = s_ids[partner]
                        left_value = s_values[tx]
                        right_value = s_values[partner]
                        left_valid = left_id >= Int32(0)
                        right_valid = right_id >= Int32(0)
                        left_key = _ordered_u32_from_f32(left_value) ^ Int32(
                            -2147483648
                        )
                        right_key = _ordered_u32_from_f32(right_value) ^ Int32(
                            -2147483648
                        )
                        left_better = (left_valid & ~right_valid) | (
                            left_valid
                            & right_valid
                            & (
                                (left_key > right_key)
                                | ((left_key == right_key) & (left_id < right_id))
                            )
                        )
                        descending = (tx & Int32(1 << (stage + 1))) == Int32(0)
                        should_swap = cutlass.select_(
                            descending, ~left_better, left_better
                        )
                        if should_swap:
                            s_ids[tx] = right_id
                            s_values[tx] = right_value
                            s_ids[partner] = left_id
                            s_values[partner] = left_value
                    cute.arch.sync_threads()

            if tx < topk_runtime:
                mOutIdx[bx, tx] = s_ids[tx]

    return _stable_sort_selected_indices_kernel


def _make_launch_stable_sort_selected_indices_kernel(topk: int):
    kernel = _make_stable_sort_selected_indices_kernel(topk)

    @cute.jit
    def _launch_stable_sort_selected_indices_kernel(
        mInput: cute.Tensor,
        mOutIdx: cute.Tensor,
        batch: int,
        topk_runtime: int,
        stream,
    ):
        kernel(
            mInput,
            mOutIdx,
            Int32(batch),
            Int32(topk_runtime),
        ).launch(
            grid=[batch, 1, 1],
            block=[_THREADS_PER_BLOCK, 1, 1],
            stream=stream,
        )

    return _launch_stable_sort_selected_indices_kernel


def _get_compiled_stable_sort_selected_indices_kernel(
    *,
    input_tensor: torch.Tensor,
    out_idx: torch.Tensor,
    topk: int,
) -> cute.JitFunction:
    device_key = _device_cache_key(input_tensor.device)
    key = (
        int(topk),
        input_tensor.dtype,
        device_key,
    )
    cached = _STABLE_SORT_COMPILE_CACHE.get(key)
    if cached is not None:
        return cached

    _raise_if_compile_is_forbidden(
        device_key=device_key,
        kernel="stable-sort",
        key=key,
    )

    m_input = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        input_tensor.detach(),
        alignment=16,
        dynamic_shape_dims=(0, 1),
        dynamic_stride_dims=(0,),
    )
    m_out_idx = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        out_idx.detach(), alignment=16, dynamic_shape_dim=0
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        _make_launch_stable_sort_selected_indices_kernel(int(topk)),
        m_input,
        m_out_idx,
        1,
        int(topk),
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )
    _STABLE_SORT_COMPILE_CACHE[key] = compiled
    return compiled


def _stable_sort_selected_indices(
    input_tensor: torch.Tensor,
    out_idx: torch.Tensor,
    *,
    topk: int,
    stream,
) -> None:
    if stream is not None:
        raise ValueError("top-k CuTeDSL kernels use the TVM-FFI environment stream")
    compiled = _get_compiled_stable_sort_selected_indices_kernel(
        input_tensor=input_tensor,
        out_idx=out_idx,
        topk=int(topk),
    )
    compiled(
        input_tensor,
        out_idx,
        int(input_tensor.shape[0]),
        int(topk),
    )


def _get_compiled_streaming_kernel(
    *,
    input_tensor: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    out_idx: torch.Tensor,
    candidate_idx: torch.Tensor,
    topk: int,
    stable_sort: bool,
    source_kind: str,
) -> cute.JitFunction:
    device_key = _device_cache_key(input_tensor.device)
    key = (
        int(topk),
        input_tensor.dtype,
        bool(stable_sort),
        str(source_kind),
        device_key,
    )
    cached = _STREAMING_COMPILE_CACHE.get(key)
    if cached is not None:
        return cached

    _raise_if_compile_is_forbidden(
        device_key=device_key,
        kernel=f"streaming-{source_kind}",
        key=key,
    )

    mInput = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        input_tensor.detach(),
        alignment=16,
        dynamic_shape_dims=(0, 1),
        dynamic_stride_dims=(0,),
    )
    mOutIdx = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        out_idx.detach(), alignment=16, dynamic_shape_dim=0
    )
    mStarts = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        starts.detach(), alignment=16, dynamic_shape_dim=0
    )
    mEnds = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        ends.detach(), alignment=16, dynamic_shape_dim=0
    )
    mCandidateIdx = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        candidate_idx.detach(), alignment=16, dynamic_shape_dim=0
    )
    launch = _make_launch_topk_selector_streaming_kernel(
        topk=int(topk),
        stable_sort=bool(stable_sort),
        source_kind=str(source_kind),
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        launch,
        mInput,
        mOutIdx,
        mStarts,
        mEnds,
        mCandidateIdx,
        1,
        1,
        1,
        1,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )
    _STREAMING_COMPILE_CACHE[key] = compiled
    return compiled


def _cutedsl_topk_selector_sm90_streaming_impl(
    input_tensor: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    *,
    topk: int,
    out_idx: torch.Tensor,
    stable_sort: bool,
    stream=None,
) -> None:
    if stream is not None:
        raise ValueError("top-k CuTeDSL kernels use the TVM-FFI environment stream")

    if not input_tensor.is_contiguous():
        raise ValueError(
            "cutedsl_topk_selector_sm90_multi_cta requires contiguous input_tensor; "
            f"got shape={tuple(input_tensor.shape)}, stride={tuple(input_tensor.stride())}"
        )
    starts_contig = starts.contiguous()
    ends_contig = ends.contiguous()

    batch, seq_len = [int(v) for v in input_tensor.shape]
    compiled = _get_compiled_streaming_kernel(
        input_tensor=input_tensor,
        starts=starts_contig,
        ends=ends_contig,
        out_idx=out_idx,
        candidate_idx=out_idx,
        topk=topk,
        stable_sort=bool(stable_sort),
        source_kind="dense",
    )
    compiled(
        input_tensor,
        out_idx,
        starts_contig,
        ends_contig,
        out_idx,
        batch,
        seq_len,
        seq_len,
        1,
    )


@lru_cache(maxsize=None)
def _choose_stable_multi_cta_chunks(
    *,
    batch: int,
    seq_len: int,
    topk: int,
    sm_count: int,
) -> int:
    """Choose a bounded power-of-two split from scan, wave, merge, and sort work."""
    if min(int(batch), int(seq_len), int(topk), int(sm_count)) <= 0:
        raise ValueError(
            "stable multi-CTA heuristic requires positive batch/seq_len/topk/SMs, "
            f"got batch={batch}, seq_len={seq_len}, topk={topk}, sm_count={sm_count}"
        )

    max_chunks = max(1, int(seq_len) // int(topk))
    candidates = [1]
    chunks = 2
    while chunks <= max_chunks:
        candidates.append(chunks)
        chunks *= 2

    scan_passes = _STABLE_SCORE_ROUNDS + 2
    radix_fixed_work = 24
    sort_items = 1 << (int(topk) - 1).bit_length()
    sort_stages = sort_items.bit_length() - 1
    sort_work = sort_stages * (sort_stages + 1) // 2
    extra_launch_work = 2 * scan_passes + 4

    def stage_waves(ctas: int) -> int:
        return (int(ctas) + int(sm_count) - 1) // int(sm_count)

    def scan_tiles(items: int) -> int:
        return (int(items) + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK

    single_cost = stage_waves(batch) * (
        scan_passes * scan_tiles(seq_len) + radix_fixed_work + sort_work
    )
    best_chunks = 1
    best_cost = single_cost
    for candidate in candidates[1:]:
        chunk_items = (int(seq_len) + candidate - 1) // candidate
        local_cost = stage_waves(int(batch) * candidate) * (
            scan_passes * scan_tiles(chunk_items) + radix_fixed_work
        )
        merge_cost = stage_waves(batch) * (
            scan_passes * scan_tiles(candidate * int(topk))
            + radix_fixed_work
            + sort_work
        )
        cost = local_cost + merge_cost + extra_launch_work
        if cost < best_cost:
            best_cost = cost
            best_chunks = candidate
    return best_chunks


@lru_cache(maxsize=None)
def _choose_stable_cluster_size(
    *,
    batch: int,
    seq_len: int,
    topk: int,
    sm_count: int,
) -> int:
    """Choose the cluster specialization from the H200 cluster-wave budget."""
    if min(int(batch), int(seq_len), int(topk), int(sm_count)) <= 0:
        raise ValueError(
            "stable cluster heuristic requires positive batch/seq_len/topk/SMs, "
            f"got batch={batch}, seq_len={seq_len}, topk={topk}, sm_count={sm_count}"
        )
    if int(topk) != 512 or int(seq_len) < 4096:
        return 0

    # Note(wangbojun/codex): C16 wins while four row-clusters fit the H200
    # low-latency wave, C8 wins through two CTA waves, and C4 only remains useful
    # for the partial third wave. Scale the measured crossovers by SM count.
    if int(batch) <= max(1, int(sm_count) // 32):
        return 16
    if int(batch) <= max(1, int(sm_count) // 4):
        return 8
    if int(batch) <= max(1, 3 * int(sm_count) // 8):
        return 4
    return 0


def _get_stable_multi_cta_workspace(
    *,
    batch: int,
    num_chunks: int,
    topk: int,
    device: torch.device,
) -> torch.Tensor:
    key = (
        int(batch),
        int(num_chunks),
        int(topk),
        _device_cache_key(device),
    )
    cached = _MULTI_CTA_WORKSPACE_CACHE.get(key)
    if cached is not None:
        return cached
    workspace = torch.empty(
        (int(batch) * int(num_chunks), int(topk)),
        dtype=torch.int32,
        device=device,
    )
    _MULTI_CTA_WORKSPACE_CACHE[key] = workspace
    return workspace


def _cutedsl_topk_selector_sm90_streaming_multi_cta_impl(
    input_tensor: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    *,
    topk: int,
    num_chunks: int,
    out_idx: torch.Tensor,
    merge_stable_sort: bool,
    stream=None,
) -> None:
    if stream is not None:
        raise ValueError("top-k CuTeDSL kernels use the TVM-FFI environment stream")
    if int(num_chunks) <= 1:
        _cutedsl_topk_selector_sm90_streaming_impl(
            input_tensor,
            starts,
            ends,
            topk=int(topk),
            out_idx=out_idx,
            stable_sort=bool(merge_stable_sort),
            stream=None,
        )
        return

    starts_contig = starts.contiguous()
    ends_contig = ends.contiguous()
    batch, seq_len = [int(v) for v in input_tensor.shape]
    chunk_size = (seq_len + int(num_chunks) - 1) // int(num_chunks)
    candidate_idx = _get_stable_multi_cta_workspace(
        batch=batch,
        num_chunks=int(num_chunks),
        topk=int(topk),
        device=input_tensor.device,
    )

    local_compiled = _get_compiled_streaming_kernel(
        input_tensor=input_tensor,
        starts=starts_contig,
        ends=ends_contig,
        out_idx=candidate_idx,
        candidate_idx=candidate_idx,
        topk=int(topk),
        stable_sort=False,
        source_kind="chunk",
    )
    local_compiled(
        input_tensor,
        candidate_idx,
        starts_contig,
        ends_contig,
        candidate_idx,
        batch,
        seq_len,
        chunk_size,
        int(num_chunks),
    )

    merge_compiled = _get_compiled_streaming_kernel(
        input_tensor=input_tensor,
        starts=starts_contig,
        ends=ends_contig,
        out_idx=out_idx,
        candidate_idx=candidate_idx,
        topk=int(topk),
        stable_sort=bool(merge_stable_sort),
        source_kind="merge",
    )
    merge_compiled(
        input_tensor,
        out_idx,
        starts_contig,
        ends_contig,
        candidate_idx,
        batch,
        seq_len,
        chunk_size,
        int(num_chunks),
    )


def _cutedsl_topk_selector_sm90_stable_multi_cta_impl(
    input_tensor: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    *,
    topk: int,
    num_chunks: int,
    out_idx: torch.Tensor,
    stream=None,
) -> None:
    _cutedsl_topk_selector_sm90_streaming_multi_cta_impl(
        input_tensor,
        starts,
        ends,
        topk=int(topk),
        num_chunks=int(num_chunks),
        out_idx=out_idx,
        merge_stable_sort=True,
        stream=stream,
    )


def _get_compiled_kernel(
    *,
    input_tensor: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    out_idx: torch.Tensor,
    hist: torch.Tensor,
    cand: torch.Tensor,
    cnt: torch.Tensor,
    remain: torch.Tensor,
    threshold: torch.Tensor,
    topk: int,
    max_candidates: int,
    use_f16_short_tail: bool,
) -> cute.JitFunction:
    device_key = _device_cache_key(input_tensor.device)
    key = (
        int(topk),
        input_tensor.dtype,
        int(max_candidates),
        bool(use_f16_short_tail),
        device_key,
    )
    cached = _COMPILE_CACHE.get(key)
    if cached is not None:
        return cached

    _raise_if_compile_is_forbidden(
        device_key=device_key,
        kernel="single-cta",
        key=key,
    )

    mInput = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        input_tensor.detach(),
        alignment=16,
        dynamic_shape_dims=(0, 1),
        dynamic_stride_dims=(0,),
    )
    mOutIdx = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        out_idx.detach(), alignment=16, dynamic_shape_dim=0
    )
    mStarts = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        starts.detach(), alignment=16, dynamic_shape_dim=0
    )
    mEnds = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        ends.detach(), alignment=16, dynamic_shape_dim=0
    )
    mHist = convert_from_dlpack(hist.detach(), leading_dim=1)
    mCand = convert_from_dlpack(cand.detach(), leading_dim=2)
    mCnt = convert_from_dlpack(cnt.detach(), leading_dim=1)
    mRemain = convert_from_dlpack(remain.detach(), leading_dim=0)
    mThreshold = convert_from_dlpack(threshold.detach(), leading_dim=0)
    if int(topk) == 64:
        launch = _make_launch_topk_selector_kernel_k64(
            max_candidates=int(max_candidates),
            use_f16_short_tail=bool(use_f16_short_tail),
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compiled = cute.compile(
            launch,
            mInput,
            mOutIdx,
            mStarts,
            mEnds,
            mHist,
            mCand,
            mCnt,
            mRemain,
            mThreshold,
            1,
            1,
            stream_fake,
            options="--enable-tvm-ffi --opt-level 2",
        )
    else:
        launch = _make_launch_topk_selector_kernel(
            max_candidates=int(max_candidates),
            use_f16_short_tail=bool(use_f16_short_tail),
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compiled = cute.compile(
            launch,
            mInput,
            mOutIdx,
            mStarts,
            mEnds,
            mHist,
            mCand,
            mCnt,
            mRemain,
            mThreshold,
            1,
            1,
            int(topk),
            stream_fake,
            options="--enable-tvm-ffi --opt-level 2",
        )
    _COMPILE_CACHE[key] = compiled
    return compiled


def _get_workspace(
    *,
    batch: int,
    max_candidates: int,
    device: torch.device,
) -> _SelectorWorkspace:
    del batch
    key = (int(max_candidates), _device_cache_key(device))
    cached = _WORKSPACE_CACHE.get(key)
    if cached is not None:
        return cached

    # Note(wangbojun/codex): Current selector keeps all per-row scratch in
    # shared memory. Keep the legacy ABI tensors as tiny placeholders instead
    # of allocating O(batch * max_candidates) global memory for 64k prompts.
    hist = torch.empty((1, 1), dtype=torch.int32, device=device)
    cand = torch.empty((1, 2, 1), dtype=torch.int32, device=device)
    cnt = torch.empty((1, 2), dtype=torch.int32, device=device)
    remain = torch.empty((1,), dtype=torch.int32, device=device)
    threshold = torch.empty((1,), dtype=torch.int32, device=device)

    ws = _SelectorWorkspace(
        hist=hist,
        cand=cand,
        cnt=cnt,
        remain=remain,
        threshold=threshold,
    )
    _WORKSPACE_CACHE[key] = ws
    return ws


def _cutedsl_topk_selector_sm90_impl(
    input_tensor: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    *,
    topk: int,
    max_candidates: int,
    out_idx: torch.Tensor,
    stream=None,
) -> None:
    if stream is not None:
        raise ValueError("top-k CuTeDSL kernels use the TVM-FFI environment stream")

    if not input_tensor.is_contiguous():
        raise ValueError(
            "cutedsl_topk_selector_sm90 requires contiguous input_tensor; "
            f"got shape={tuple(input_tensor.shape)}, stride={tuple(input_tensor.stride())}"
        )
    starts_contig = starts.contiguous()
    ends_contig = ends.contiguous()

    batch, seq_len = [int(v) for v in input_tensor.shape]
    use_f16_short_tail = input_tensor.dtype == torch.float16
    ws = _get_workspace(
        batch=batch, max_candidates=max_candidates, device=input_tensor.device
    )

    # The selector launch only needs row count at runtime; the candidate column
    # count remains a compile-time bucket. Keeping the row dimension dynamic
    # avoids recompiling the long-prefill selector for every exact prompt length.
    compiled = _get_compiled_kernel(
        input_tensor=input_tensor,
        starts=starts_contig,
        ends=ends_contig,
        out_idx=out_idx,
        hist=ws.hist,
        cand=ws.cand,
        cnt=ws.cnt,
        remain=ws.remain,
        threshold=ws.threshold,
        topk=topk,
        max_candidates=max_candidates,
        use_f16_short_tail=use_f16_short_tail,
    )
    if topk == 64:
        compiled(
            input_tensor,
            out_idx,
            starts_contig,
            ends_contig,
            ws.hist,
            ws.cand,
            ws.cnt,
            ws.remain,
            ws.threshold,
            batch,
            seq_len,
        )
    else:
        compiled(
            input_tensor,
            out_idx,
            starts_contig,
            ends_contig,
            ws.hist,
            ws.cand,
            ws.cnt,
            ws.remain,
            ws.threshold,
            batch,
            seq_len,
            topk,
        )


@torch.library.custom_op(
    "optimus_cutedsl::cutedsl_topk_selector_sm90_out",
    mutates_args=("out_idx",),
    device_types="cuda",
)
def _cutedsl_topk_selector_sm90_out(
    input_tensor: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    out_idx: torch.Tensor,
    topk: int,
    max_candidates: int,
) -> None:
    # Keep allocation and the EnvStream launch opaque to Dynamo/CUDA Graph.
    _cutedsl_topk_selector_sm90_impl(
        input_tensor,
        starts,
        ends,
        topk=topk,
        max_candidates=max_candidates,
        out_idx=out_idx,
        stream=None,
    )


@torch.library.custom_op(
    "optimus_cutedsl::cutedsl_topk_selector_sm90_functional",
    mutates_args=(),
    device_types="cuda",
)
def _cutedsl_topk_selector_sm90_functional(
    input_tensor: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    topk: int,
    max_candidates: int,
) -> torch.Tensor:
    # Note(wangbojun/codex): Keep allocation and kernel launch inside one
    # opaque op so piecewise cudagraph capture observes the selector result as
    # a true op output instead of an alias produced by mutating a graph-local
    # tensor allocation.
    out_idx = torch.full(
        (input_tensor.shape[0], int(topk)),
        -1,
        dtype=torch.int32,
        device=input_tensor.device,
    )
    _cutedsl_topk_selector_sm90_impl(
        input_tensor,
        starts,
        ends,
        topk=topk,
        max_candidates=max_candidates,
        out_idx=out_idx,
        stream=None,
    )
    return out_idx


@_cutedsl_topk_selector_sm90_functional.register_fake
def _cutedsl_topk_selector_sm90_functional_fake(
    input_tensor: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    topk: int,
    max_candidates: int,
) -> torch.Tensor:
    del starts, ends, max_candidates
    return torch.empty(
        (input_tensor.shape[0], topk),
        dtype=torch.int32,
        device=input_tensor.device,
    )


def cutedsl_topk_selector_sm90(
    input_tensor: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    *,
    topk: int,
    max_candidates: int = _DEFAULT_MAX_CANDIDATES,
    out_idx: Optional[torch.Tensor] = None,
    stream=None,
) -> torch.Tensor:
    compiling = _is_torch_compiling()
    if input_tensor.dim() != 2:
        raise ValueError("input_tensor must be [batch, seq_len]")
    if starts.dim() != 1 or ends.dim() != 1:
        raise ValueError("starts and ends must be [batch]")
    if input_tensor.device.type != "cuda":
        raise RuntimeError("cutedsl_topk_selector_sm90 requires CUDA tensors")
    if input_tensor.dtype not in (torch.float16, torch.float32):
        raise ValueError("input_tensor dtype must be float16 or float32")
    if starts.dtype != torch.int32 or ends.dtype != torch.int32:
        raise ValueError("starts and ends dtype must be torch.int32")
    if starts.device != input_tensor.device or ends.device != input_tensor.device:
        raise ValueError("input_tensor/starts/ends must be on the same device")
    batch = input_tensor.shape[0]
    seq_len = int(input_tensor.shape[1])
    if not compiling and (
        int(starts.shape[0]) != int(batch) or int(ends.shape[0]) != int(batch)
    ):
        raise ValueError("starts/ends length must match batch size")
    k = int(topk)
    if k <= 0:
        raise ValueError("topk must be > 0")
    if k > seq_len:
        raise ValueError(f"topk must be <= seq_len, got topk={k}, seq_len={seq_len}")
    max_cand = int(max_candidates)
    if max_cand <= 0:
        raise ValueError("max_candidates must be > 0")
    if max_cand < k:
        raise ValueError(
            f"max_candidates must be >= topk, got max_candidates={max_cand}, topk={k}"
        )
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
        if (
            out_idx.ndim != 2
            or int(out_idx.shape[1]) != k
            or (not compiling and int(out_idx.shape[0]) != int(batch))
        ):
            raise ValueError(
                f"out_idx shape must be ({batch}, {k}), got {tuple(out_idx.shape)}"
            )
    if stream is None:
        if out_idx is None:
            return _cutedsl_topk_selector_sm90_functional(
                input_tensor,
                starts,
                ends,
                k,
                max_cand,
            )
        _cutedsl_topk_selector_sm90_out(
            input_tensor,
            starts,
            ends,
            out_idx,
            k,
            max_cand,
        )
    else:
        if out_idx is None:
            out_idx = torch.full(
                (batch, k),
                -1,
                dtype=torch.int32,
                device=input_tensor.device,
            )
        _cutedsl_topk_selector_sm90_impl(
            input_tensor,
            starts,
            ends,
            topk=k,
            max_candidates=max_cand,
            out_idx=out_idx,
            stream=stream,
        )
    return out_idx


def _choose_multi_cta_selector_shape(
    *,
    seq_len: int,
    topk: int,
    max_supported: int,
) -> tuple[int, int, int]:
    if int(seq_len) <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    if int(topk) <= 0:
        raise ValueError(f"topk must be > 0, got {topk}")
    if int(max_supported) < int(topk):
        raise ValueError(
            "multi-CTA selector requires the per-CTA candidate capacity to "
            f"cover topk, got topk={topk}, supported<={max_supported}"
        )

    min_chunks = (int(seq_len) + int(max_supported) - 1) // int(max_supported)
    max_chunks = max(1, int(max_supported) // int(topk))
    if min_chunks > max_chunks:
        raise ValueError(
            "cannot build exact two-stage multi-CTA topk plan under the current "
            "shared-memory candidate cap: "
            f"seq_len={seq_len}, topk={topk}, supported<={max_supported}, "
            f"min_chunks={min_chunks}, max_chunks={max_chunks}"
        )

    num_chunks = max(1, min_chunks)
    chunk_size = (int(seq_len) + num_chunks - 1) // num_chunks
    candidate_cols = num_chunks * int(topk)
    if chunk_size > int(max_supported) or candidate_cols > int(max_supported):
        raise AssertionError(
            "internal multi-CTA selector plan exceeds single-CTA refinement cap: "
            f"chunk_size={chunk_size}, candidate_cols={candidate_cols}, "
            f"supported<={max_supported}"
        )
    return num_chunks, chunk_size, candidate_cols


def cutedsl_topk_selector_sm90_multi_cta(
    input_tensor: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    *,
    topk: int,
    out_idx: Optional[torch.Tensor] = None,
    stable_sort: bool = False,
    stream=None,
) -> torch.Tensor:
    """Exact long-row selector.

    The fp32 stable path uses a shape- and SM-aware local-selection/merge plan;
    the unstable path uses one streaming-radix kernel. Both avoid the
    ``max_candidates == seq_len`` shared-memory candidate buffer. Non-fp32
    inputs retain the older two-stage chunked path.
    """
    compiling = _is_torch_compiling()
    if input_tensor.dim() != 2:
        raise ValueError("input_tensor must be [batch, seq_len]")
    if starts.dim() != 1 or ends.dim() != 1:
        raise ValueError("starts and ends must be [batch]")
    if input_tensor.device.type != "cuda":
        raise RuntimeError("cutedsl_topk_selector_sm90_multi_cta requires CUDA tensors")
    if input_tensor.dtype not in (torch.float16, torch.float32):
        raise ValueError("input_tensor dtype must be float16 or float32")
    if starts.dtype != torch.int32 or ends.dtype != torch.int32:
        raise ValueError("starts and ends dtype must be torch.int32")
    if starts.device != input_tensor.device or ends.device != input_tensor.device:
        raise ValueError("input_tensor/starts/ends must be on the same device")
    if not input_tensor.is_contiguous():
        raise ValueError(
            "cutedsl_topk_selector_sm90_multi_cta requires contiguous input_tensor; "
            f"got shape={tuple(input_tensor.shape)}, stride={tuple(input_tensor.stride())}"
        )

    batch = int(input_tensor.shape[0])
    seq_len = int(input_tensor.shape[1])
    if not compiling and (int(starts.shape[0]) != batch or int(ends.shape[0]) != batch):
        raise ValueError("starts/ends length must match batch size")
    k = int(topk)
    if k <= 0:
        raise ValueError("topk must be > 0")
    if k > seq_len:
        raise ValueError(f"topk must be <= seq_len, got topk={k}, seq_len={seq_len}")
    if stable_sort and k > _MAX_STABLE_TOPK:
        raise NotImplementedError(
            "stable multi-CTA selection supports topk <= "
            f"{_MAX_STABLE_TOPK}, got topk={k}"
        )

    if out_idx is not None:
        if out_idx.dtype != torch.int32:
            raise ValueError("out_idx dtype must be torch.int32")
        if out_idx.device != input_tensor.device:
            raise ValueError("out_idx must be on the same device as input_tensor")
        if not out_idx.is_contiguous():
            raise ValueError(
                f"out_idx must be contiguous, got stride={tuple(out_idx.stride())}"
            )
        if (
            out_idx.ndim != 2
            or int(out_idx.shape[1]) != k
            or (not compiling and int(out_idx.shape[0]) != batch)
        ):
            raise ValueError(
                f"out_idx shape must be ({batch}, {k}), got {tuple(out_idx.shape)}"
            )

    if input_tensor.dtype == torch.float32:
        if out_idx is None:
            out_idx = torch.empty(
                (batch, k),
                dtype=torch.int32,
                device=input_tensor.device,
            )
        if stable_sort:
            sm_count = _device_sm_count(input_tensor.device)
            cluster_size = 0
            if not compiling and _device_is_sm90(input_tensor.device):
                cluster_size = _choose_stable_cluster_size(
                    batch=batch,
                    seq_len=seq_len,
                    topk=k,
                    sm_count=sm_count,
            )
            if cluster_size > 0:
                return _cluster_selector.cutedsl_topk_selector_sm90_cluster(
                    input_tensor,
                    starts,
                    ends,
                    out_idx=out_idx,
                    cluster_size=cluster_size,
                    threads_per_cta=512,
                    stream=stream,
                )
            num_chunks = _choose_stable_multi_cta_chunks(
                batch=batch,
                seq_len=seq_len,
                topk=k,
                sm_count=sm_count,
            )
            _cutedsl_topk_selector_sm90_stable_multi_cta_impl(
                input_tensor,
                starts,
                ends,
                topk=k,
                num_chunks=num_chunks,
                out_idx=out_idx,
                stream=stream,
            )
        else:
            sm_count = _device_sm_count(input_tensor.device)
            num_chunks = _choose_stable_multi_cta_chunks(
                batch=batch,
                seq_len=seq_len,
                topk=k,
                sm_count=sm_count,
            )
            _cutedsl_topk_selector_sm90_streaming_multi_cta_impl(
                input_tensor,
                starts,
                ends,
                topk=k,
                num_chunks=num_chunks,
                out_idx=out_idx,
                merge_stable_sort=False,
                stream=stream,
            )
        return out_idx

    if stable_sort:
        raise NotImplementedError(
            "stable multi-CTA selection currently requires float32 scores; "
            f"got {input_tensor.dtype}"
        )

    if stream is not None:
        raise NotImplementedError(
            "cutedsl_topk_selector_sm90_multi_cta fallback path currently uses "
            "PyTorch composite tensor ops between selector launches and does not "
            "accept an explicit CUDA driver stream"
        )

    max_supported = _selector_max_candidates_cap(input_tensor.device)
    num_chunks, chunk_size, candidate_cols = _choose_multi_cta_selector_shape(
        seq_len=seq_len,
        topk=k,
        max_supported=max_supported,
    )

    padded_len = num_chunks * chunk_size
    neg_inf = float("-inf")
    if padded_len == seq_len:
        chunked_scores = input_tensor.view(batch, num_chunks, chunk_size)
    else:
        padded = torch.full(
            (batch, padded_len),
            neg_inf,
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        padded[:, :seq_len].copy_(input_tensor)
        chunked_scores = padded.view(batch, num_chunks, chunk_size)
    chunked_scores_2d = chunked_scores.reshape(
        batch * num_chunks, chunk_size
    ).contiguous()

    seq_len_t = torch.tensor(seq_len, device=input_tensor.device, dtype=torch.int32)
    starts_clamped = torch.minimum(
        torch.maximum(starts.contiguous(), torch.zeros_like(starts)), seq_len_t
    )
    ends_clamped = torch.minimum(
        torch.maximum(ends.contiguous(), torch.zeros_like(ends)), seq_len_t
    )
    chunk_offsets = torch.arange(
        num_chunks, device=input_tensor.device, dtype=torch.int32
    ) * int(chunk_size)
    local_starts = (
        torch.clamp(
            starts_clamped.view(batch, 1) - chunk_offsets.view(1, num_chunks),
            min=0,
            max=int(chunk_size),
        )
        .reshape(batch * num_chunks)
        .contiguous()
    )
    local_ends = (
        torch.clamp(
            ends_clamped.view(batch, 1) - chunk_offsets.view(1, num_chunks),
            min=0,
            max=int(chunk_size),
        )
        .reshape(batch * num_chunks)
        .contiguous()
    )

    local_idx = torch.empty(
        (batch * num_chunks, k),
        dtype=torch.int32,
        device=input_tensor.device,
    )
    cutedsl_topk_selector_sm90(
        chunked_scores_2d,
        local_starts,
        local_ends,
        topk=k,
        max_candidates=int(chunk_size),
        out_idx=local_idx,
    )

    local_valid = local_idx >= 0
    safe_local_idx = torch.clamp(local_idx, min=0)
    local_scores = torch.gather(
        chunked_scores_2d, 1, safe_local_idx.to(dtype=torch.int64)
    )
    local_scores = torch.where(
        local_valid,
        local_scores,
        torch.full_like(local_scores, neg_inf),
    )
    flat_chunk_offsets = (
        chunk_offsets.view(1, num_chunks, 1)
        .expand(batch, num_chunks, k)
        .reshape(batch * num_chunks, k)
    )
    local_global_idx = torch.where(
        local_valid,
        local_idx + flat_chunk_offsets,
        torch.full_like(local_idx, -1),
    )

    candidate_scores = local_scores.view(batch, candidate_cols).contiguous()
    candidate_global_idx = local_global_idx.view(batch, candidate_cols).contiguous()
    candidate_starts = torch.zeros(
        (batch,), dtype=torch.int32, device=input_tensor.device
    )
    candidate_ends = torch.full(
        (batch,),
        int(candidate_cols),
        dtype=torch.int32,
        device=input_tensor.device,
    )
    candidate_topk_idx = torch.empty(
        (batch, k),
        dtype=torch.int32,
        device=input_tensor.device,
    )
    cutedsl_topk_selector_sm90(
        candidate_scores,
        candidate_starts,
        candidate_ends,
        topk=k,
        max_candidates=int(candidate_cols),
        out_idx=candidate_topk_idx,
    )

    final_valid = candidate_topk_idx >= 0
    final_idx = torch.gather(
        candidate_global_idx,
        1,
        torch.clamp(candidate_topk_idx, min=0).to(dtype=torch.int64),
    )
    final_idx = torch.where(final_valid, final_idx, torch.full_like(final_idx, -1))

    if out_idx is None:
        return final_idx.contiguous()
    out_idx.copy_(final_idx)
    return out_idx


def cutedsl_topk_selector_sm90_multi_cta_stable(
    input_tensor: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    *,
    topk: int,
    out_idx: Optional[torch.Tensor] = None,
    stream=None,
) -> torch.Tensor:
    """Stable long-row selector implemented entirely by CuTeDSL kernels."""
    if input_tensor.dtype != torch.float32:
        raise NotImplementedError(
            "stable multi-CTA selection requires float32 scores; "
            f"got {input_tensor.dtype}"
        )
    return cutedsl_topk_selector_sm90_multi_cta(
        input_tensor,
        starts,
        ends,
        topk=int(topk),
        out_idx=out_idx,
        stable_sort=True,
        stream=stream,
    )


# Note(wangbojun/codex): The cluster module imports the selector primitives
# defined above. Import the module only after those definitions exist so direct
# imports in either order share one module-level dependency without function-
# local imports or an initialization cycle.
import vllm.models.step4.nvidia.ops.cute_dsl.indexer_ops.topk_selector_cluster_sm90 as _cluster_selector


__all__ = [
    "cutedsl_topk_selector_sm90",
    "cutedsl_topk_selector_sm90_multi_cta",
    "cutedsl_topk_selector_sm90_multi_cta_stable",
    "prewarm_cutedsl_topk_selector_sm90_compilation",
    "seal_cutedsl_topk_selector_sm90_compilation",
]
