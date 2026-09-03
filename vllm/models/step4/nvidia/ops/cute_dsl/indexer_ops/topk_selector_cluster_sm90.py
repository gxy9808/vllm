from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.models.step4.nvidia.ops.cute_dsl.pdl import wait_for_dependencies
from vllm.models.step4.nvidia.ops.cute_dsl.indexer_ops.topk_selector_sm90 import (
    _RADIX,
    _STABLE_SCORE_ROUNDS,
    _WARP_SIZE,
    _atomic_add_i32,
    _elem_pointer,
    _lane_mask_lt_u32,
    _ordered_u32_from_f32,
    _radix_byte_from_f32,
    _warp_prefix_sum_i32,
)
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.kernel_utils import cute_utils as sparse_utils


_DEFAULT_CLUSTER_SIZE = 16
_DEFAULT_THREADS_PER_CTA = 512
_TOPK = 512
_COMPILE_CACHE: Dict[
    Tuple[int, int, torch.dtype, bool, Tuple[str, Optional[int]]], cute.JitFunction
] = {}
_SEALED_COMPILE_DEVICES: set[Tuple[str, Optional[int]]] = set()


def _device_cache_key(device: torch.device) -> Tuple[str, Optional[int]]:
    index = device.index
    if device.type == "cuda" and index is None:
        index = torch.cuda.current_device()
    return device.type, index


def seal_cutedsl_topk_selector_sm90_cluster_compilation(
    device: torch.device | str | None = None,
) -> None:
    resolved = torch.device("cuda") if device is None else torch.device(device)
    _SEALED_COMPILE_DEVICES.add(_device_cache_key(resolved))


@dsl_user_op
def _map_shared_rank(
    smem_ptr: cute.Pointer,
    peer_cta_rank: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                smem_ptr.toint(loc=loc, ip=ip).ir_value(),
                peer_cta_rank.ir_value(loc=loc, ip=ip),
            ],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _match_any_sync_i32(value: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [value.ir_value(loc=loc, ip=ip)],
            "match.any.sync.b32 $0, $1, 0xffffffff;",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _load_shared_remote_i32(
    smem_ptr: cute.Pointer,
    peer_cta_rank: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    remote_ptr = _map_shared_rank(smem_ptr, peer_cta_rank, loc=loc, ip=ip).ir_value(
        loc=loc, ip=ip
    )
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [remote_ptr],
            "ld.shared::cluster.u32 $0, [$1];",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _store_shared_remote_i32(
    value: Int32,
    smem_ptr: cute.Pointer,
    peer_cta_rank: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    remote_ptr = _map_shared_rank(smem_ptr, peer_cta_rank, loc=loc, ip=ip).ir_value(
        loc=loc, ip=ip
    )
    llvm.inline_asm(
        None,
        [remote_ptr, value.ir_value(loc=loc, ip=ip)],
        "st.shared::cluster.u32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _store_shared_remote_f32(
    value: Float32,
    smem_ptr: cute.Pointer,
    peer_cta_rank: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    remote_ptr = _map_shared_rank(smem_ptr, peer_cta_rank, loc=loc, ip=ip).ir_value(
        loc=loc, ip=ip
    )
    llvm.inline_asm(
        None,
        [remote_ptr, value.ir_value(loc=loc, ip=ip)],
        "st.shared::cluster.f32 [$0], $1;",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _cluster_sync():
    cute.arch.cluster_arrive_relaxed()
    cute.arch.cluster_wait()


def _make_cluster_selector_kernel(
    cluster_size: int,
    threads_per_cta: int,
    use_pdl: bool,
):
    warps_per_cta = threads_per_cta // _WARP_SIZE
    topk_i32 = Int32(_TOPK)

    @cute.kernel
    def _cluster_selector_kernel(
        mInput: cute.Tensor,
        mOutIdx: cute.Tensor,
        mStarts: cute.Tensor,
        mEnds: cute.Tensor,
        batch: Int32,
        seq_len: Int32,
        chunk_size: Int32,
    ):
        bx, _, _ = cute.arch.block_idx()
        tx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        cta_rank = cute.arch.block_idx_in_cluster()
        row = bx // Int32(cluster_size)

        @cute.struct
        class SharedStorage:
            hist_local: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 2 * (_RADIX + 1)], 128
            ]
            hist_global: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, _RADIX + 1], 128
            ]
            hist_warp: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, warps_per_cta * (_RADIX + 1)],
                128,
            ]
            counters: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]
            remain: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 8]
            threshold: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 8]
            prefix: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, _STABLE_SCORE_ROUNDS], 16
            ]
            selected_ids: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, _TOPK], 128
            ]
            selected_values: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, _TOPK], 128
            ]
            sort_temp_ids: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, _TOPK], 128
            ]
            sort_temp_values: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, _TOPK], 128
            ]

        storage = cutlass.utils.SmemAllocator().allocate(SharedStorage, 16)
        sHistLocal = storage.hist_local.get_tensor(
            cute.make_layout((2, _RADIX + 1), stride=(_RADIX + 1, 1))
        )
        sHistGlobal = storage.hist_global.get_tensor(
            cute.make_layout((_RADIX + 1,), stride=(1,))
        )
        sHistWarp = storage.hist_warp.get_tensor(
            cute.make_layout((warps_per_cta, _RADIX + 1), stride=(_RADIX + 1, 1))
        )
        sCounters = storage.counters.get_tensor(cute.make_layout((4,), stride=(1,)))
        sRemain = storage.remain.get_tensor(cute.make_layout((1,), stride=(1,)))
        sThreshold = storage.threshold.get_tensor(cute.make_layout((1,), stride=(1,)))
        sPrefix = storage.prefix.get_tensor(
            cute.make_layout((_STABLE_SCORE_ROUNDS,), stride=(1,))
        )
        sSelectedIds = storage.selected_ids.get_tensor(
            cute.make_layout((_TOPK,), stride=(1,))
        )
        sSelectedValues = storage.selected_values.get_tensor(
            cute.make_layout((_TOPK,), stride=(1,))
        )
        sSortTempIds = storage.sort_temp_ids.get_tensor(
            cute.make_layout((_TOPK,), stride=(1,))
        )
        sSortTempValues = storage.sort_temp_values.get_tensor(
            cute.make_layout((_TOPK,), stride=(1,))
        )

        if row < batch:
            start_idx = mStarts[row]
            end_idx = mEnds[row]
            if start_idx < Int32(0):
                start_idx = Int32(0)
            if end_idx > seq_len:
                end_idx = seq_len
            chunk_begin = cta_rank * chunk_size
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
            num_tiles = (valid_len + Int32(threads_per_cta - 1)) // Int32(
                threads_per_cta
            )

            if tx < Int32(_STABLE_SCORE_ROUNDS):
                sPrefix[tx] = Int32(0)
            if tx == Int32(0):
                sRemain[Int32(0)] = topk_i32
            cute.arch.sync_threads()

            if cutlass.const_expr(use_pdl):
                wait_for_dependencies()

            for round_i in cutlass.range_constexpr(_STABLE_SCORE_ROUNDS):
                hist_phase = round_i & 1
                for i_chunk in cutlass.range_constexpr(
                    (_RADIX + 1 + _WARP_SIZE - 1) // _WARP_SIZE
                ):
                    i = lane_idx + Int32(i_chunk * _WARP_SIZE)
                    if i <= Int32(_RADIX):
                        sHistWarp[warp_idx, i] = Int32(0)
                cute.arch.sync_threads()

                for tile in cutlass.range(num_tiles, unroll=1):
                    input_idx = start_idx + tile * Int32(threads_per_cta) + tx
                    if input_idx < end_idx:
                        value = Float32(mInput[row, input_idx])
                        matches_prefix = True
                        for prev_i in cutlass.range_constexpr(_STABLE_SCORE_ROUNDS):
                            if prev_i < round_i:
                                byte = _radix_byte_from_f32(
                                    value, shift=24 - prev_i * 8
                                )
                                if byte != sPrefix[Int32(prev_i)]:
                                    matches_prefix = False
                        if matches_prefix:
                            bin_id = _radix_byte_from_f32(value, shift=24 - round_i * 8)
                            _atomic_add_i32(
                                _elem_pointer(sHistWarp, (warp_idx, bin_id)),
                                Int32(1),
                            )
                cute.arch.sync_threads()

                if tx <= Int32(_RADIX):
                    local_count = Int32(0)
                    for w in cutlass.range_constexpr(warps_per_cta):
                        local_count += sHistWarp[Int32(w), tx]
                    sHistLocal[Int32(hist_phase), tx] = local_count
                cute.arch.sync_threads()
                _cluster_sync()

                if tx <= Int32(_RADIX):
                    global_count = Int32(0)
                    for peer in cutlass.range_constexpr(cluster_size):
                        global_count += _load_shared_remote_i32(
                            _elem_pointer(sHistLocal, (Int32(hist_phase), tx)),
                            Int32(peer),
                        )
                    sHistGlobal[tx] = global_count
                cute.arch.sync_threads()

                round_remain = sRemain[Int32(0)]
                bin_count = Int32(0)
                bin_id = Int32(0)
                inclusive = Int32(0)
                if tx < Int32(_RADIX):
                    bin_id = Int32(_RADIX - 1) - tx
                    bin_count = sHistGlobal[bin_id]
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
                    sPrefix[Int32(round_i)] = Int32(0)
                    sRemain[Int32(0)] = round_remain - (total - sHistGlobal[Int32(0)])
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
                        sPrefix[Int32(round_i)] = bin_id
                        sRemain[Int32(0)] = round_remain - count_greater
                cute.arch.sync_threads()

            if tx < Int32(2):
                sCounters[tx] = Int32(0)
            if cta_rank == Int32(0) and tx < topk_i32:
                sSelectedIds[tx] = Int32(-1)
                sSelectedValues[tx] = Float32(0.0)
            cute.arch.sync_threads()
            for tile in cutlass.range(num_tiles, unroll=1):
                input_idx = start_idx + tile * Int32(threads_per_cta) + tx
                value = Float32(0.0)
                is_gt = False
                is_eq = False
                if input_idx < end_idx:
                    value = Float32(mInput[row, input_idx])
                    is_eq = True
                    for cmp_i in cutlass.range_constexpr(_STABLE_SCORE_ROUNDS):
                        byte = _radix_byte_from_f32(value, shift=24 - cmp_i * 8)
                        if is_eq and byte > sPrefix[Int32(cmp_i)]:
                            is_gt = True
                        if is_eq and byte != sPrefix[Int32(cmp_i)]:
                            is_eq = False
                gt_mask = cute.arch.vote_ballot_sync(is_gt)
                eq_mask = cute.arch.vote_ballot_sync(is_eq)
                if lane_idx == Int32(0):
                    _atomic_add_i32(
                        _elem_pointer(sCounters, (Int32(0),)),
                        Int32(cute.arch.popc(gt_mask)),
                    )
                    _atomic_add_i32(
                        _elem_pointer(sCounters, (Int32(1),)),
                        Int32(cute.arch.popc(eq_mask)),
                    )
            cute.arch.sync_threads()
            _cluster_sync()

            if tx == Int32(0):
                global_gt = Int32(0)
                eq_before = Int32(0)
                selected_before = Int32(0)
                for peer in cutlass.range_constexpr(cluster_size):
                    peer_gt = _load_shared_remote_i32(
                        _elem_pointer(sCounters, (Int32(0),)), Int32(peer)
                    )
                    peer_eq = _load_shared_remote_i32(
                        _elem_pointer(sCounters, (Int32(1),)), Int32(peer)
                    )
                    global_gt += peer_gt
                    if Int32(peer) < cta_rank:
                        eq_before += peer_eq
                fill_remaining = topk_i32 - global_gt
                if fill_remaining < Int32(0):
                    fill_remaining = Int32(0)
                running_eq = Int32(0)
                for peer in cutlass.range_constexpr(cluster_size):
                    peer_gt = _load_shared_remote_i32(
                        _elem_pointer(sCounters, (Int32(0),)), Int32(peer)
                    )
                    peer_eq = _load_shared_remote_i32(
                        _elem_pointer(sCounters, (Int32(1),)), Int32(peer)
                    )
                    if Int32(peer) < cta_rank:
                        peer_ties = fill_remaining - running_eq
                        if peer_ties < Int32(0):
                            peer_ties = Int32(0)
                        if peer_ties > peer_eq:
                            peer_ties = peer_eq
                        selected_before += peer_gt + peer_ties
                    running_eq += peer_eq
                sCounters[Int32(2)] = fill_remaining
                sCounters[Int32(3)] = eq_before
                sThreshold[Int32(0)] = selected_before
                sCounters[Int32(0)] = Int32(0)
                sCounters[Int32(1)] = Int32(0)
            cute.arch.sync_threads()

            fill_remaining = sCounters[Int32(2)]
            eq_before = sCounters[Int32(3)]
            selected_before = sThreshold[Int32(0)]
            for tile in cutlass.range(num_tiles, unroll=1):
                input_idx = start_idx + tile * Int32(threads_per_cta) + tx
                value = Float32(0.0)
                is_gt = False
                is_eq = False
                if input_idx < end_idx:
                    value = Float32(mInput[row, input_idx])
                    is_eq = True
                    for cmp_i in cutlass.range_constexpr(_STABLE_SCORE_ROUNDS):
                        byte = _radix_byte_from_f32(value, shift=24 - cmp_i * 8)
                        if is_eq and byte > sPrefix[Int32(cmp_i)]:
                            is_gt = True
                        if is_eq and byte != sPrefix[Int32(cmp_i)]:
                            is_eq = False

                eq_mask = cute.arch.vote_ballot_sync(is_eq)
                eq_lane_rank = Int32(
                    cute.arch.popc(Uint32(eq_mask) & _lane_mask_lt_u32())
                )
                if lane_idx == Int32(0):
                    sHistWarp[warp_idx, Int32(0)] = Int32(cute.arch.popc(eq_mask))
                cute.arch.sync_threads()
                if tx == Int32(0):
                    eq_base = sCounters[Int32(0)]
                    tile_eq = Int32(0)
                    for w in cutlass.range_constexpr(warps_per_cta):
                        warp_count = sHistWarp[Int32(w), Int32(0)]
                        sHistWarp[Int32(w), Int32(1)] = tile_eq
                        tile_eq += warp_count
                    sRemain[Int32(0)] = eq_base
                    sCounters[Int32(0)] = eq_base + tile_eq
                cute.arch.sync_threads()

                eq_rank = (
                    eq_before
                    + sRemain[Int32(0)]
                    + sHistWarp[warp_idx, Int32(1)]
                    + eq_lane_rank
                )
                selected = is_gt | (is_eq & (eq_rank < fill_remaining))
                selected_mask = cute.arch.vote_ballot_sync(selected)
                selected_lane_rank = Int32(
                    cute.arch.popc(Uint32(selected_mask) & _lane_mask_lt_u32())
                )
                if lane_idx == Int32(0):
                    sHistWarp[warp_idx, Int32(2)] = Int32(cute.arch.popc(selected_mask))
                cute.arch.sync_threads()
                if tx == Int32(0):
                    selected_base = sCounters[Int32(1)]
                    tile_selected = Int32(0)
                    for w in cutlass.range_constexpr(warps_per_cta):
                        warp_count = sHistWarp[Int32(w), Int32(2)]
                        sHistWarp[Int32(w), Int32(3)] = tile_selected
                        tile_selected += warp_count
                    sRemain[Int32(0)] = selected_base
                    sCounters[Int32(1)] = selected_base + tile_selected
                cute.arch.sync_threads()
                if selected:
                    out_pos = (
                        selected_before
                        + sRemain[Int32(0)]
                        + sHistWarp[warp_idx, Int32(3)]
                        + selected_lane_rank
                    )
                    if out_pos < topk_i32:
                        if cta_rank == Int32(0):
                            sSelectedIds[out_pos] = input_idx
                            sSelectedValues[out_pos] = value
                        else:
                            _store_shared_remote_i32(
                                input_idx,
                                _elem_pointer(sSelectedIds, (out_pos,)),
                                Int32(0),
                            )
                            _store_shared_remote_f32(
                                value,
                                _elem_pointer(sSelectedValues, (out_pos,)),
                                Int32(0),
                            )
                cute.arch.sync_threads()
            _cluster_sync()

            if cta_rank == Int32(0):
                for radix_pass in cutlass.range_constexpr(4):
                    for i_chunk in cutlass.range_constexpr(
                        (_RADIX + 1 + _WARP_SIZE - 1) // _WARP_SIZE
                    ):
                        i = lane_idx + Int32(i_chunk * _WARP_SIZE)
                        if i <= Int32(_RADIX):
                            sHistWarp[warp_idx, i] = Int32(0)
                    cute.arch.sync_threads()

                    sort_active = tx < topk_i32
                    sort_id = Int32(-1)
                    sort_value = Float32(0.0)
                    if sort_active:
                        if cutlass.const_expr((radix_pass & 1) == 0):
                            sort_id = sSelectedIds[tx]
                            sort_value = sSelectedValues[tx]
                        else:
                            sort_id = sSortTempIds[tx]
                            sort_value = sSortTempValues[tx]
                    sort_key = Int32(-1)
                    if sort_id >= Int32(0):
                        sort_key = ~_ordered_u32_from_f32(sort_value)
                    sort_bin = (sort_key >> Int32(radix_pass * 8)) & Int32(0xFF)
                    match_input = sort_bin
                    if not sort_active:
                        match_input = Int32(_RADIX) + lane_idx
                    match_mask = _match_any_sync_i32(match_input)
                    sort_lane_rank = Int32(
                        cute.arch.popc(match_mask & _lane_mask_lt_u32())
                    )
                    if sort_active and sort_lane_rank == Int32(0):
                        sHistWarp[warp_idx, sort_bin] = Int32(
                            cute.arch.popc(match_mask)
                        )
                    cute.arch.sync_threads()

                    if tx < Int32(_RADIX):
                        warp_prefix = Int32(0)
                        for w in cutlass.range_constexpr(warps_per_cta):
                            warp_count = sHistWarp[Int32(w), tx]
                            sHistWarp[Int32(w), tx] = warp_prefix
                            warp_prefix += warp_count
                        sHistGlobal[tx] = warp_prefix
                    cute.arch.sync_threads()

                    bin_count = Int32(0)
                    inclusive = Int32(0)
                    if tx < Int32(_RADIX):
                        bin_count = sHistGlobal[tx]
                        inclusive = _warp_prefix_sum_i32(bin_count, lane_idx)
                        if lane_idx == Int32(_WARP_SIZE - 1):
                            sHistWarp[warp_idx, Int32(_RADIX)] = inclusive
                    cute.arch.sync_threads()

                    if tx == Int32(0):
                        warp_offset = Int32(0)
                        for w in cutlass.range_constexpr(_RADIX // _WARP_SIZE):
                            warp_total = sHistWarp[Int32(w), Int32(_RADIX)]
                            sHistWarp[Int32(w), Int32(_RADIX)] = warp_offset
                            warp_offset += warp_total
                    cute.arch.sync_threads()

                    if tx < Int32(_RADIX):
                        sHistGlobal[tx] = (
                            sHistWarp[warp_idx, Int32(_RADIX)] + inclusive - bin_count
                        )
                    cute.arch.sync_threads()

                    if sort_active:
                        sort_pos = (
                            sHistGlobal[sort_bin]
                            + sHistWarp[warp_idx, sort_bin]
                            + sort_lane_rank
                        )
                        if cutlass.const_expr((radix_pass & 1) == 0):
                            sSortTempIds[sort_pos] = sort_id
                            sSortTempValues[sort_pos] = sort_value
                        else:
                            sSelectedIds[sort_pos] = sort_id
                            sSelectedValues[sort_pos] = sort_value
                    cute.arch.sync_threads()
                if tx < topk_i32:
                    mOutIdx[row, tx] = sSelectedIds[tx]

    return _cluster_selector_kernel


def _make_cluster_selector_launch(
    cluster_size: int,
    threads_per_cta: int,
    use_pdl: bool,
):
    kernel = _make_cluster_selector_kernel(cluster_size, threads_per_cta, use_pdl)

    @cute.jit
    def _launch(
        mInput: cute.Tensor,
        mOutIdx: cute.Tensor,
        mStarts: cute.Tensor,
        mEnds: cute.Tensor,
        batch: int,
        seq_len: int,
        chunk_size: int,
        stream,
    ):
        kernel(
            mInput,
            mOutIdx,
            mStarts,
            mEnds,
            Int32(batch),
            Int32(seq_len),
            Int32(chunk_size),
        ).launch(
            grid=[batch * cluster_size, 1, 1],
            block=[threads_per_cta, 1, 1],
            cluster=(cluster_size, 1, 1),
            stream=stream,
            use_pdl=use_pdl,
        )

    return _launch


def cutedsl_topk_selector_sm90_cluster(
    input_tensor: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    *,
    out_idx: Optional[torch.Tensor] = None,
    cluster_size: Optional[int] = None,
    threads_per_cta: Optional[int] = None,
    use_pdl: bool = False,
    stream=None,
) -> torch.Tensor:
    if cluster_size is None:
        cluster_size = _DEFAULT_CLUSTER_SIZE
    if threads_per_cta is None:
        threads_per_cta = _DEFAULT_THREADS_PER_CTA
    cluster_size = int(cluster_size)
    threads_per_cta = int(threads_per_cta)
    if cluster_size not in (2, 4, 8, 16):
        raise ValueError(
            f"cluster_size must be one of 2, 4, 8, or 16; got {cluster_size}"
        )
    if threads_per_cta not in (512, 1024):
        raise ValueError(f"threads_per_cta must be 512 or 1024; got {threads_per_cta}")
    if input_tensor.ndim != 2 or input_tensor.dtype != torch.float32:
        raise ValueError(
            "cluster selector requires contiguous [batch, seq] fp32 scores"
        )
    if not input_tensor.is_cuda or not input_tensor.is_contiguous():
        raise ValueError("cluster selector requires contiguous CUDA scores")
    batch, seq_len = [int(v) for v in input_tensor.shape]
    if starts.shape != (batch,) or ends.shape != (batch,):
        raise ValueError("starts/ends must match batch")
    if starts.dtype != torch.int32 or ends.dtype != torch.int32:
        raise ValueError("starts/ends must be int32")
    if starts.device != input_tensor.device or ends.device != input_tensor.device:
        raise ValueError("input_tensor/starts/ends must be on the same CUDA device")
    if out_idx is None:
        out_idx = torch.empty(
            (batch, _TOPK), device=input_tensor.device, dtype=torch.int32
        )
    if out_idx.shape != (batch, _TOPK) or out_idx.dtype != torch.int32:
        raise ValueError(f"out_idx must be [{batch}, {_TOPK}] int32")
    if out_idx.device != input_tensor.device or not out_idx.is_contiguous():
        raise ValueError("out_idx must be contiguous and on the input CUDA device")

    if stream is not None:
        raise ValueError("cluster top-k uses the TVM-FFI environment stream")
    starts_contig = starts.contiguous()
    ends_contig = ends.contiguous()
    chunk_size = (seq_len + cluster_size - 1) // cluster_size
    device_key = _device_cache_key(input_tensor.device)
    key = (
        cluster_size,
        threads_per_cta,
        input_tensor.dtype,
        bool(use_pdl),
        device_key,
    )
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        if device_key in _SEALED_COMPILE_DEVICES:
            raise RuntimeError(
                "cluster top-k CuTeDSL compilation is sealed after serving "
                f"warmup; missing prewarmed variant key={key!r}"
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "cluster top-k CuTeDSL cache miss during CUDA graph capture; "
                f"prewarm variant key={key!r} before capture"
            )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compiled = cute.compile(
            _make_cluster_selector_launch(
                cluster_size,
                threads_per_cta,
                bool(use_pdl),
            ),
            sparse_utils.make_fake_tensor_like_with_dynamic_dim(
                input_tensor.detach(),
                alignment=16,
                dynamic_shape_dims=(0, 1),
                dynamic_stride_dims=(0,),
            ),
            sparse_utils.make_fake_tensor_like_with_dynamic_dim(
                out_idx.detach(), alignment=16, dynamic_shape_dim=0
            ),
            sparse_utils.make_fake_tensor_like_with_dynamic_dim(
                starts_contig.detach(), alignment=16, dynamic_shape_dim=0
            ),
            sparse_utils.make_fake_tensor_like_with_dynamic_dim(
                ends_contig.detach(), alignment=16, dynamic_shape_dim=0
            ),
            1,
            1,
            1,
            stream_fake,
            options="--enable-tvm-ffi --opt-level 2",
        )
        _COMPILE_CACHE[key] = compiled
    compiled(
        input_tensor,
        out_idx,
        starts_contig,
        ends_contig,
        batch,
        seq_len,
        chunk_size,
    )
    return out_idx


__all__ = [
    "cutedsl_topk_selector_sm90_cluster",
    "seal_cutedsl_topk_selector_sm90_cluster_compilation",
]
