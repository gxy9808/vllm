# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM90 GQA sparse paged-KV prefill using token-phys schedules."""

import functools
import math
from typing import Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.arch import ProxyKind, SharedSpace
from cutlass.cute.nvgpu import warp

from vllm.models.step4.nvidia.ops.cute_dsl.cutedsl_compile_cache import cached_compile_function
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.kernel_utils import mma_helpers as sm90_mma
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.kernel_utils import cute_utils as utils
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.kernel_utils.softmax import Softmax


GQA_REGION_VALID_SHIFT = 24
GQA_REGION_ID_MASK = (1 << GQA_REGION_VALID_SHIFT) - 1


class _TokenWiseSparsePagedPrefillGQA:
    """GQA-specific sparse paged-KV prefill kernel consuming physical region metadata directly."""

    arch: int = 90
    GROUP_SIZE_PAD_MMA = 16
    SUPPORTED_HEAD_DIMS = (128, 192)
    PAGE_SIZE = 16
    KV_HEADS = 1
    NUM_STAGES = 2
    PRODUCER_SUBGROUP_LANES = 8
    SUPERBLOCK_TOKENS = 64
    SUPPORTED_TOPOLOGIES = {
        (4, 16, 256),
        (4, 16, 320),
        (2, 16, 192),
        (2, 16, 224),
        (2, 16, 256),
        (1, 16, 128),
        (1, 16, 160),
        (1, 16, 192),
    }

    def __init__(
        self,
        *,
        dtype,
        logical_q_heads: int,
        head_dim: int,
        head_dim_v: int,
        block_n: int,
        num_threads: int,
        q_per_cta: int,
        topk_windows: int,
        q_load_copy_elems: int = 8,
        producer_async_copy_elems: int = 8,
    ) -> None:
        self.dtype = dtype
        self.logical_q_heads = int(logical_q_heads)
        self.HEAD_DIM = int(head_dim)
        self.HEAD_DIM_V = int(head_dim_v)
        self.head_dim_slabs = self.HEAD_DIM // 64
        self.head_dim_v_slabs = self.HEAD_DIM_V // 64
        self.GROUP_SIZE = 8 if self.logical_q_heads <= 8 else 16
        self.GROUP_SIZE_PAD = self.GROUP_SIZE
        self.block_n = int(block_n)
        self.num_threads = int(num_threads)
        self.q_per_cta = int(q_per_cta)
        self.topk_windows = int(topk_windows)
        self.q_load_copy_elems = int(q_load_copy_elems)
        self.producer_async_copy_elems = int(producer_async_copy_elems)
        self.num_stages = int(self.NUM_STAGES)
        self.num_warps = self.num_threads // 32
        self.num_compute_warps = self.q_per_cta
        self.num_producer_warps = self.num_warps - self.num_compute_warps
        self.producer_threads = self.num_producer_warps * 32
        self.producer_subgroup_lanes = int(self.PRODUCER_SUBGROUP_LANES)
        self.producer_groups = self.producer_threads // self.producer_subgroup_lanes
        self.max_rows_per_group = (self.q_per_cta * self.block_n + self.producer_groups - 1) // self.producer_groups
        self.k_stage_arrive_threads = self.producer_threads
        self.v_stage_arrive_threads = self.producer_threads
        self.fixed_count = self.topk_windows * 8
        self.fixed_split = (self.fixed_count + self.block_n - 1) // self.block_n
        if self.SUPERBLOCK_TOKENS % self.block_n != 0:
            raise ValueError(
                f"SUPERBLOCK_TOKENS={self.SUPERBLOCK_TOKENS} must be divisible by block_n={self.block_n}"
            )
        self.superblock_subtiles = self.SUPERBLOCK_TOKENS // self.block_n
        self.fixed_super_split = (self.fixed_split + self.superblock_subtiles - 1) // self.superblock_subtiles

        if self.dtype not in (cutlass.Float16, cutlass.BFloat16):
            raise TypeError("Only fp16/bf16 are supported")
        if self.logical_q_heads not in (4, 8, 16):
            raise ValueError(
                "GQA token-sparse prefill supports only logical_q_heads in {4, 8, 16}, "
                f"got {self.logical_q_heads}"
            )
        if self.HEAD_DIM not in self.SUPPORTED_HEAD_DIMS or self.HEAD_DIM_V not in self.SUPPORTED_HEAD_DIMS:
            raise ValueError(
                "GQA token-sparse prefill supports only head_dim/head_dim_v in "
                f"{self.SUPPORTED_HEAD_DIMS}, got ({self.HEAD_DIM}, {self.HEAD_DIM_V})"
            )
        if self.HEAD_DIM != self.HEAD_DIM_V:
            raise ValueError(
                "GQA token-sparse prefill keeps the equal-dim contract, "
                f"got head_dim={self.HEAD_DIM}, head_dim_v={self.HEAD_DIM_V}"
            )
        if self.HEAD_DIM % 64 != 0 or self.HEAD_DIM_V % 64 != 0:
            raise ValueError(
                "GQA token-sparse prefill requires head_dim/head_dim_v to be 64-aligned, "
                f"got ({self.HEAD_DIM}, {self.HEAD_DIM_V})"
            )
        if (self.q_per_cta, self.block_n, self.num_threads) not in self.SUPPORTED_TOPOLOGIES:
            raise ValueError(
                "GQA token-sparse prefill supports only "
                f"{sorted(self.SUPPORTED_TOPOLOGIES)}"
            )
        if self.num_producer_warps <= 0:
            raise ValueError("GQA token-sparse prefill requires producer warps")
        if self.topk_windows <= 0:
            raise ValueError("GQA token-sparse prefill requires topk_windows > 0")
        if self.producer_async_copy_elems not in (2, 4, 8):
            raise ValueError(
                "GQA token-sparse prefill supports only producer_async_copy_elems in {2, 4, 8}, "
                f"got {self.producer_async_copy_elems}"
            )
        if self.q_load_copy_elems not in (2, 4, 8):
            raise ValueError(
                "GQA token-sparse prefill supports only q_load_copy_elems in {2, 4, 8}, "
                f"got {self.q_load_copy_elems}"
            )

    def _get_tiled_mma(self):
        tiled_mma_qk = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (1, 1, 1),
            permutation_mnk=(16, 16, 16),
        )
        tiled_mma_pv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (1, 1, 1),
            permutation_mnk=(16, 16, 16),
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_smem_layouts(self):
        # Note(wangbojun/codex): Keep row stride padded to avoid bank-aligned rows on logical [M, K] storage.
        k_row_stride = self.HEAD_DIM + 8
        v_row_stride = self.HEAD_DIM_V + 8
        sK_layout = cute.make_layout(
            (self.block_n, self.HEAD_DIM, self.num_stages, self.q_per_cta),
            stride=(
                k_row_stride,
                1,
                self.block_n * k_row_stride,
                self.num_stages * self.block_n * k_row_stride,
            ),
        )
        sV_layout = cute.make_layout(
            (self.block_n, self.HEAD_DIM_V, self.num_stages, self.q_per_cta),
            stride=(
                v_row_stride,
                1,
                self.block_n * v_row_stride,
                self.num_stages * self.block_n * v_row_stride,
            ),
        )
        return sK_layout, sV_layout

    def _get_shared_storage_cls(self):
        sK_layout, sV_layout = self._get_smem_layouts()
        sQ_struct = cute.struct.Align[
            cute.struct.MemRange[
                self.dtype,
                self.q_per_cta * self.GROUP_SIZE_PAD_MMA * self.HEAD_DIM,
            ],
            128,
        ]
        sK_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(sK_layout)],
            128,
        ]
        sV_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(sV_layout)],
            128,
        ]
        mbar_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * self.q_per_cta]
        mbar_stage_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages]
        scalar_vec_struct = cute.struct.MemRange[Int32, self.q_per_cta]
        valid_tok_struct = cute.struct.MemRange[
            Int32,
            self.block_n * self.num_stages * self.q_per_cta,
        ]

        @cute.struct
        class SharedStorage:
            mbar_ptr_free: mbar_struct
            mbar_ptr_k_stage: mbar_stage_struct
            mbar_ptr_v_stage: mbar_stage_struct
            sQ: sQ_struct
            sK: sK_struct
            sV: sV_struct
            sCount: scalar_vec_struct
            sValidTok: valid_tok_struct

        return SharedStorage

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mKCache: cute.Tensor,
        mVCache: cute.Tensor,
        mRegionPhys: cute.Tensor,
        mRegionCount: cute.Tensor,
        mWorkQGlobal: cute.Tensor,
        mWorkQLocal: cute.Tensor,
        mWorkQLen: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        seq_q: Int32,
        softmax_scale: Float32,
        stream: cuda.CUstream,
    ) -> None:
        softmax_scale_log2 = Float32(softmax_scale * math.log2(math.e))

        def _assume_stride_divisible(stride_elem, *, element_width: int):
            if const_expr(isinstance(stride_elem, int)):
                return stride_elem
            return cute.assume(stride_elem, divby=128 // element_width)

        new_stride = lambda t: (
            *(
                _assume_stride_divisible(s, element_width=t.element_type.width)
                for s in t.stride[:-1]
            ),
            t.stride[-1],
        )
        mQ, mKCache, mVCache, mO = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            for t in (mQ, mKCache, mVCache, mO)
        ]
        if const_expr(mLSE is not None):
            mLSE = cute.make_tensor(
                mLSE.iterator,
                cute.make_layout(mLSE.shape, stride=new_stride(mLSE)),
            )

        mQ = utils.select(mQ, [0, 2, 1])
        mKCache = utils.select(mKCache, [1, 3, 2, 0])
        mVtCache = utils.select(mVCache, [3, 1, 2, 0])
        mO = utils.select(mO, [0, 2, 1])
        # Note(wangbojun/codex): Flatten paged-KV once and let the hot producer
        # derive physical token ids from compact physical region metadata directly.
        mKCache = cute.make_tensor(
            mKCache.iterator,
            cute.make_layout(
                (mKCache.shape[0] * mKCache.shape[3], mKCache.shape[1], mKCache.shape[2]),
                stride=(mKCache.stride[0], mKCache.stride[1], mKCache.stride[2]),
            ),
        )
        mVtCache = cute.make_tensor(
            mVtCache.iterator,
            cute.make_layout(
                (mVtCache.shape[0], mVtCache.shape[1] * mVtCache.shape[3], mVtCache.shape[2]),
                stride=(mVtCache.stride[0], mVtCache.stride[1], mVtCache.stride[2]),
            ),
        )
        mRegionPhys = cute.make_tensor(mRegionPhys.iterator, mRegionPhys.layout)
        mRegionCount = cute.make_tensor(mRegionCount.iterator, mRegionCount.layout)
        mWorkQGlobal = cute.make_tensor(mWorkQGlobal.iterator, mWorkQGlobal.layout)
        mWorkQLocal = cute.make_tensor(mWorkQLocal.iterator, mWorkQLocal.layout)
        mWorkQLen = cute.make_tensor(mWorkQLen.iterator, mWorkQLen.layout)
        if const_expr(mLSE is not None):
            mLSE = utils.select(mLSE, [0, 1])

        shape_Q_packed = (
            (self.GROUP_SIZE_PAD, mQ.shape[0]),
            mQ.shape[1],
            mKCache.shape[2],
        )
        stride_Q_packed = (
            (mQ.stride[2], mQ.stride[0]),
            mQ.stride[1],
            mQ.stride[2] * self.GROUP_SIZE_PAD,
        )
        mQ = cute.make_tensor(mQ.iterator, cute.make_layout(shape_Q_packed, stride=stride_Q_packed))

        shape_O_packed = (
            (self.GROUP_SIZE_PAD, mO.shape[0]),
            mO.shape[1],
            mKCache.shape[2],
        )
        stride_O_packed = (
            (mO.stride[2], mO.stride[0]),
            mO.stride[1],
            mO.stride[2] * self.GROUP_SIZE_PAD,
        )
        mO = cute.make_tensor(mO.iterator, cute.make_layout(shape_O_packed, stride=stride_O_packed))

        work_tiles = Int32(cute.size(mWorkQGlobal.shape[0]))
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        sK_layout, sV_layout = self._get_smem_layouts()
        SharedStorage = self._get_shared_storage_cls()

        self.kernel(
            mQ,
            mKCache,
            mVtCache,
            mRegionPhys,
            mRegionCount,
            mWorkQGlobal,
            mWorkQLocal,
            mWorkQLen,
            mO,
            mLSE,
            seq_q,
            softmax_scale_log2,
            tiled_mma_qk,
            tiled_mma_pv,
            sK_layout,
            sV_layout,
            SharedStorage,
        ).launch(
            grid=(work_tiles, 1, 1),
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mKCache: cute.Tensor,
        mVtCache: cute.Tensor,
        mRegionPhys: cute.Tensor,
        mRegionCount: cute.Tensor,
        mWorkQGlobal: cute.Tensor,
        mWorkQLocal: cute.Tensor,
        mWorkQLen: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        seq_q: Int32,
        softmax_scale_log2: Float32,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        sK_layout,
        sV_layout,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        work_q_block, _, _ = cute.arch.block_idx()

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tidx % 32
        is_producer = warp_idx < Int32(self.num_producer_warps)
        is_compute = warp_idx >= Int32(self.num_producer_warps)
        slot = warp_idx - Int32(self.num_producer_warps)

        q_base = mWorkQGlobal[work_q_block]
        q_local_base = mWorkQLocal[work_q_block]
        q_req_len = mWorkQLen[work_q_block]
        kv_head_idx = (q_base - q_local_base) // seq_q
        q_tile_count = q_req_len - q_local_base
        if q_tile_count < Int32(0):
            q_tile_count = Int32(0)
        if q_tile_count > Int32(self.q_per_cta):
            q_tile_count = Int32(self.q_per_cta)

        region_tokens_i32 = Int32(8)
        region_id_mask_i32 = Int32(GQA_REGION_ID_MASK)
        block_n_i32 = Int32(self.block_n)
        stage_count = Int32(self.num_stages)
        superblock_subtiles_i32 = Int32(self.superblock_subtiles)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        mbar_free = storage.mbar_ptr_free.data_ptr()
        mbar_k_stage = storage.mbar_ptr_k_stage.data_ptr()
        mbar_v_stage = storage.mbar_ptr_v_stage.data_ptr()

        sQ = storage.sQ.get_tensor(
            cute.make_layout(
                (self.q_per_cta, self.GROUP_SIZE_PAD_MMA, self.HEAD_DIM),
                stride=(self.GROUP_SIZE_PAD_MMA * self.HEAD_DIM, self.HEAD_DIM, 1),
            )
        )
        sK = storage.sK.get_tensor(sK_layout)
        sV = storage.sV.get_tensor(sV_layout)
        sCount = storage.sCount.get_tensor(cute.make_layout((self.q_per_cta,), stride=(1,)))
        sValidTok = storage.sValidTok.get_tensor(
            cute.make_layout(
                (self.block_n, self.num_stages, self.q_per_cta),
                stride=(1, self.block_n, self.block_n * self.num_stages),
            )
        )

        if tidx < Int32(self.num_stages * self.q_per_cta):
            cute.arch.mbarrier_init(mbar_free + tidx, 1)
        if tidx < Int32(self.num_stages):
            cute.arch.mbarrier_init(mbar_k_stage + tidx, Int32(self.k_stage_arrive_threads))
            cute.arch.mbarrier_init(mbar_v_stage + tidx, Int32(self.v_stage_arrive_threads))
        cute.arch.mbarrier_init_fence()

        if tidx < self.q_per_cta:
            slot_meta = tidx
            q_pos_meta = q_base + slot_meta
            count_meta = (
                mRegionCount[q_pos_meta] * region_tokens_i32
                if slot_meta < q_tile_count
                else Int32(0)
            )
            sCount[slot_meta] = count_meta

        cute.arch.sync_threads()

        # Note(wangbojun/codex): The runtime path can advertise up to 512 sparse
        # windows per row, but early prefill rows often touch only 0 or 1 region.
        # Driving the producer/consumer barrier protocol with the static
        # topk-derived split count forces hundreds of empty stage handoffs and
        # is what the long-context q4/256 path was corrupting on rows like
        # q={8,12}. Cap the outer stage loops to the actual max row count in the
        # current CTA so we only recycle stages that carry real payload.
        tile_count_tokens = Int32(0)
        for slot_i in cutlass.range_constexpr(self.q_per_cta):
            slot_count_i = sCount[Int32(slot_i)]
            if slot_count_i > tile_count_tokens:
                tile_count_tokens = slot_count_i
        tile_split_i32 = (tile_count_tokens + block_n_i32 - Int32(1)) // block_n_i32
        tile_super_split_i32 = (
            tile_split_i32 + superblock_subtiles_i32 - Int32(1)
        ) // superblock_subtiles_i32

        vec_k = Int32(8)
        vec_layout_8 = cute.make_layout((8,), stride=(1,))
        vec_layout_4 = cute.make_layout((4,), stride=(1,))
        vec_layout_2 = cute.make_layout((2,), stride=(1,))
        producer_subgroup_lanes = Int32(self.producer_subgroup_lanes)
        k_subgroup_lanes = producer_subgroup_lanes
        producer_threads = Int32(self.producer_threads)
        producer_groups = Int32(self.producer_groups)
        row_total = Int32(self.q_per_cta) * block_n_i32

        if is_producer:
            group_idx = tidx // k_subgroup_lanes
            lane_in_group = tidx - group_idx * k_subgroup_lanes
            col_base = lane_in_group * vec_k
            tok_buf = cute.make_fragment(self.max_rows_per_group * self.superblock_subtiles, Int32)
            valid_tok_buf = cute.make_fragment(self.max_rows_per_group * self.superblock_subtiles, Int32)
            row_linear_buf = cute.make_fragment(self.max_rows_per_group, Int32)
            slot_i = Int32(0)
            row = Int32(0)
            for super_idx in cutlass.range(tile_super_split_i32, unroll=1):
                local_idx_base = super_idx * superblock_subtiles_i32
                # Note(wangbojun/codex): The q4/256 topology is the only one
                # that satisfies producer_groups == block_n. Keeping a separate
                # "row==group_idx" fast branch there has been correlated with
                # the real vLLM corruption pattern (rows 8/12/24/28). The
                # generic row-linear mapping is semantically equivalent for that
                # topology, so use the single path for all producer shapes.
                for local_row in cutlass.range_constexpr(self.max_rows_per_group):
                    row_linear_buf[local_row] = group_idx + Int32(local_row) * producer_groups
                for subtile_idx in cutlass.range_constexpr(self.superblock_subtiles):
                    for local_row in cutlass.range_constexpr(self.max_rows_per_group):
                        row_linear = row_linear_buf[local_row]
                        phys_tok_pre = Int32(-1)
                        valid_tok_pre = Int32(0)
                        slot_i = Int32(0)
                        row = Int32(0)
                        if row_linear < row_total:
                            slot_i = row_linear // block_n_i32
                            row = row_linear - slot_i * block_n_i32
                            local_idx = local_idx_base + Int32(subtile_idx)
                            tok_idx = local_idx * block_n_i32 + row
                            if slot_i < q_tile_count:
                                count_slot = sCount[slot_i]
                                if (local_idx < tile_split_i32) and (tok_idx < count_slot):
                                    q_pos_cur = q_base + slot_i
                                    window_idx = tok_idx >> Int32(3)
                                    region_lane = tok_idx & Int32(7)
                                    packed_region = mRegionPhys[q_pos_cur, window_idx]
                                    valid_tok_pre = packed_region >> Int32(GQA_REGION_VALID_SHIFT)
                                    phys_region = packed_region & region_id_mask_i32
                                    if valid_tok_pre > Int32(0):
                                        phys_tok_pre = phys_region * region_tokens_i32 + region_lane
                        tok_buf[subtile_idx * self.max_rows_per_group + local_row] = phys_tok_pre
                        valid_tok_buf[subtile_idx * self.max_rows_per_group + local_row] = valid_tok_pre

                for subtile_idx in cutlass.range_constexpr(self.superblock_subtiles):
                    local_idx = local_idx_base + Int32(subtile_idx)
                    if local_idx < tile_split_i32:
                        stage = local_idx % stage_count

                        if local_idx >= stage_count and tidx < self.q_per_cta:
                            wait_phase = ((local_idx - stage_count - stage) // stage_count) & Int32(1)
                            bar_idx_wait = stage * Int32(self.q_per_cta) + tidx
                            cute.arch.mbarrier_wait(mbar_free + bar_idx_wait, wait_phase)
                        cute.arch.barrier(barrier_id=4, number_of_threads=producer_threads)

                        for local_row in cutlass.range_constexpr(self.max_rows_per_group):
                            phys_tok_cur = tok_buf[subtile_idx * self.max_rows_per_group + local_row]
                            row_linear = row_linear_buf[local_row]
                            if row_linear < row_total:
                                slot_i = row_linear // block_n_i32
                                row = row_linear - slot_i * block_n_i32
                                if lane_in_group == Int32(0):
                                    sValidTok[row, stage, slot_i] = valid_tok_buf[
                                        subtile_idx * self.max_rows_per_group + local_row
                                    ]
                            if phys_tok_cur >= Int32(0):
                                for slab in cutlass.range_constexpr(self.head_dim_slabs):
                                    col_slab_base = col_base + Int32(slab * 64)
                                    for vec_sub in cutlass.range_constexpr(8 // self.producer_async_copy_elems):
                                        vec_off = Int32(vec_sub * self.producer_async_copy_elems)
                                        g_ptr = utils.elem_pointer_i64_offset(
                                            mKCache, (phys_tok_cur, col_slab_base + vec_off, kv_head_idx)
                                        )
                                        s_ptr = utils.elem_pointer_i64_offset(
                                            sK, (row, col_slab_base + vec_off, stage, slot_i)
                                        )
                                        g_vec = cute.make_tensor(
                                            g_ptr,
                                            cute.make_layout((self.producer_async_copy_elems,), stride=(1,)),
                                        )
                                        s_vec = cute.make_tensor(
                                            s_ptr,
                                            cute.make_layout((self.producer_async_copy_elems,), stride=(1,)),
                                        )
                                        utils.vector_copy_with_explicit_width(
                                            g_vec,
                                            s_vec,
                                            num_copy_elems=self.producer_async_copy_elems,
                                            is_async=True,
                                        )
                        cute.arch.cp_async_commit_group()
                        cute.arch.barrier(barrier_id=6, number_of_threads=producer_threads)
                        cute.arch.cp_async_mbarrier_arrive_noinc(mbar_k_stage + stage)

                        for local_row in cutlass.range_constexpr(self.max_rows_per_group):
                            phys_tok_cur = tok_buf[subtile_idx * self.max_rows_per_group + local_row]
                            if phys_tok_cur >= Int32(0):
                                row_linear = row_linear_buf[local_row]
                                slot_i = row_linear // block_n_i32
                                row = row_linear - slot_i * block_n_i32
                                for slab in cutlass.range_constexpr(self.head_dim_v_slabs):
                                    col_slab_base = col_base + Int32(slab * 64)
                                    for vec_sub in cutlass.range_constexpr(8 // self.producer_async_copy_elems):
                                        vec_off = Int32(vec_sub * self.producer_async_copy_elems)
                                        g_ptr = utils.elem_pointer_i64_offset(
                                            mVtCache, (col_slab_base + vec_off, phys_tok_cur, kv_head_idx)
                                        )
                                        s_ptr = utils.elem_pointer_i64_offset(
                                            sV, (row, col_slab_base + vec_off, stage, slot_i)
                                        )
                                        g_vec = cute.make_tensor(
                                            g_ptr,
                                            cute.make_layout((self.producer_async_copy_elems,), stride=(1,)),
                                        )
                                        s_vec = cute.make_tensor(
                                            s_ptr,
                                            cute.make_layout((self.producer_async_copy_elems,), stride=(1,)),
                                        )
                                        utils.vector_copy_with_explicit_width(
                                            g_vec,
                                            s_vec,
                                            num_copy_elems=self.producer_async_copy_elems,
                                            is_async=True,
                                        )
                        cute.arch.cp_async_commit_group()
                        cute.arch.barrier(barrier_id=7, number_of_threads=producer_threads)
                        cute.arch.cp_async_mbarrier_arrive_noinc(mbar_v_stage + stage)

        if is_compute:
            q_pos_slot = q_base + slot
            q_local_pos_slot = q_local_base + slot
            q_valid_slot = slot < q_tile_count
            q_pos_safe = q_local_pos_slot if q_valid_slot else Int32(0)
            count_slot = sCount[slot]
            slot_active = q_valid_slot and (count_slot > Int32(0))
            logical_q_heads_i32 = Int32(self.logical_q_heads)
            lse_head_base = kv_head_idx * logical_q_heads_i32

            mQ_cur = mQ[None, None, kv_head_idx]
            gQ_slot = cute.local_tile(mQ_cur, (self.GROUP_SIZE_PAD, self.HEAD_DIM), (q_pos_safe, 0))
            q_vec_elems = Int32(8)
            q_vec_count = (self.GROUP_SIZE * self.HEAD_DIM) // 8
            q_logical_vec_count = (self.logical_q_heads * self.HEAD_DIM) // 8
            q_group_pad_vec_count = ((self.GROUP_SIZE - self.logical_q_heads) * self.HEAD_DIM) // 8
            q_pad_vec_count = ((self.GROUP_SIZE_PAD_MMA - self.GROUP_SIZE) * self.HEAD_DIM) // 8
            v_vecs_per_row = self.HEAD_DIM_V // 8
            v_block_vec_count = self.block_n * v_vecs_per_row

            if q_valid_slot:
                # Note(wangbojun/codex): The scalar `gQ_slot[h, d] if q_valid_slot else 0`
                # path compiled into `LDG.E.U16 -> SEL -> STS.U16` and dominated long scoreboard.
                # Load the valid heads as aligned 16B vectors so the common path does one branch
                # per slot instead of one predicate per element.
                for vec_idx in cutlass.range(lane, q_logical_vec_count, 32, unroll=1):
                    elem_idx = vec_idx * q_vec_elems
                    h = elem_idx // self.HEAD_DIM
                    d = elem_idx - h * self.HEAD_DIM
                    for vec_sub in cutlass.range_constexpr(8 // self.q_load_copy_elems):
                        vec_off = Int32(vec_sub * self.q_load_copy_elems)
                        g_ptr = utils.elem_pointer_i64_offset(gQ_slot, (h, d + vec_off))
                        s_ptr = utils.elem_pointer_i64_offset(sQ, (slot, h, d + vec_off))
                        if const_expr(self.q_load_copy_elems == 8):
                            g_vec = cute.make_tensor(g_ptr, vec_layout_8)
                            s_vec = cute.make_tensor(s_ptr, vec_layout_8)
                            utils.vector_copy_with_explicit_width(g_vec, s_vec, num_copy_elems=8)
                        elif const_expr(self.q_load_copy_elems == 4):
                            g_vec = cute.make_tensor(g_ptr, vec_layout_4)
                            s_vec = cute.make_tensor(s_ptr, vec_layout_4)
                            utils.vector_copy_with_explicit_width(g_vec, s_vec, num_copy_elems=4)
                        else:
                            g_vec = cute.make_tensor(g_ptr, vec_layout_2)
                            s_vec = cute.make_tensor(s_ptr, vec_layout_2)
                            utils.vector_copy_with_explicit_width(g_vec, s_vec, num_copy_elems=2)
                for vec_idx in cutlass.range(lane, q_group_pad_vec_count, 32, unroll=1):
                    elem_idx = vec_idx * q_vec_elems
                    h_pad = elem_idx // self.HEAD_DIM
                    d = elem_idx - h_pad * self.HEAD_DIM
                    s_ptr = utils.elem_pointer_i64_offset(sQ, (slot, logical_q_heads_i32 + h_pad, d))
                    s_vec = cute.make_tensor(s_ptr, vec_layout_8)
                    for vi in cutlass.range_constexpr(8):
                        s_vec[vi] = self.dtype(0)
            else:
                for vec_idx in cutlass.range(lane, q_vec_count, 32, unroll=1):
                    elem_idx = vec_idx * q_vec_elems
                    h = elem_idx // self.HEAD_DIM
                    d = elem_idx - h * self.HEAD_DIM
                    s_ptr = utils.elem_pointer_i64_offset(sQ, (slot, h, d))
                    s_vec = cute.make_tensor(s_ptr, vec_layout_8)
                    for vi in cutlass.range_constexpr(8):
                        s_vec[vi] = self.dtype(0)

            for vec_idx in cutlass.range(lane, q_pad_vec_count, 32, unroll=1):
                elem_idx = vec_idx * q_vec_elems
                h_pad = elem_idx // self.HEAD_DIM
                d = elem_idx - h_pad * self.HEAD_DIM
                s_ptr = utils.elem_pointer_i64_offset(sQ, (slot, self.GROUP_SIZE + h_pad, d))
                s_vec = cute.make_tensor(s_ptr, vec_layout_8)
                for vi in cutlass.range_constexpr(8):
                    s_vec[vi] = self.dtype(0)
            cute.arch.sync_warp()
            cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)

            smem_copy_atom_qk = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self.dtype,
            )
            smem_copy_atom_v = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                self.dtype,
            )
            thr_mma_qk = tiled_mma_qk.get_slice(lane)
            thr_mma_pv = tiled_mma_pv.get_slice(lane)
            smem_thr_copy_Q = utils.make_tiled_copy_A(smem_copy_atom_qk, tiled_mma_qk).get_slice(lane)
            smem_thr_copy_K = utils.make_tiled_copy_B(smem_copy_atom_qk, tiled_mma_qk).get_slice(lane)
            smem_thr_copy_V = utils.make_tiled_copy_B(smem_copy_atom_v, tiled_mma_pv).get_slice(lane)

            sQ_slot = sQ[slot, None, None]
            tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ_slot))
            tSsQ = smem_thr_copy_Q.partition_S(sQ_slot)
            tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
            cute.copy(smem_thr_copy_Q, tSsQ, tSrQ_copy_view)

            cS = cute.make_identity_tensor((self.GROUP_SIZE_PAD_MMA, self.block_n))
            tScS_mn = utils.make_acc_tensor_mn_view_from_mma(thr_mma_qk.partition_C(cS))
            cO = cute.make_identity_tensor((self.GROUP_SIZE_PAD_MMA, self.HEAD_DIM_V))
            tOcO_hv = utils.make_acc_tensor_mn_view_from_mma(thr_mma_pv.partition_C(cO))
            acc_shape_S = thr_mma_qk.partition_shape_C((self.GROUP_SIZE_PAD_MMA, self.block_n))
            rP = cute.make_fragment(acc_shape_S, self.dtype)
            tOrP = cute.make_tensor(rP.iterator, utils.acc_layout_to_frgA_split2(rP.layout))

            acc_shape_O = thr_mma_pv.partition_shape_C((self.GROUP_SIZE_PAD_MMA, self.HEAD_DIM_V))
            acc_O = cute.make_fragment(acc_shape_O, Float32)
            acc_O.fill(Float32.zero)
            softmax = Softmax.create(
                softmax_scale_log2,
                num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            )
            softmax.reset()
            nrow_s = const_expr(cute.size(tScS_mn.shape[0]))
            ncol_s = const_expr(cute.size(tScS_mn.shape[1]))
            nrow_o = const_expr(cute.size(tOcO_hv.shape[0]))
            ncol_o = const_expr(cute.size(tOcO_hv.shape[1]))
            mO_cur = mO[None, None, kv_head_idx]
            gO_slot = cute.local_tile(mO_cur, (self.GROUP_SIZE_PAD, self.HEAD_DIM_V), (q_pos_safe, 0))
            acc_O_hv = utils.make_acc_tensor_mn_view_from_mma(acc_O)

            if tile_split_i32 > Int32(0):
                cute.arch.mbarrier_wait(mbar_k_stage + Int32(0), Int32(0))
                cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
                cute.arch.sync_warp()

                if slot_active:
                    sK_stage_slot = sK[None, None, Int32(0), slot]
                    tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK_stage_slot))
                    tSsK = smem_thr_copy_K.partition_S(sK_stage_slot)
                    acc_S = cute.make_fragment(acc_shape_S, Float32)
                    acc_S.fill(Float32.zero)
                    sm90_mma.smem_prefetch_gemm_with_hook(
                        tiled_mma_qk,
                        acc_S,
                        tSrQ,
                        tSrK,
                        tSsQ,
                        tSsK,
                        smem_thr_copy_Q,
                        smem_thr_copy_K,
                        A_in_regs=cutlass.Boolean(True),
                    )
                    acc_S_mn = utils.make_acc_tensor_mn_view_from_mma(acc_S)
                    for r in cutlass.range_constexpr(nrow_s):
                        for c in cutlass.range_constexpr(ncol_s):
                            k_row = tScS_mn[r, c][1]
                            valid_tok = sValidTok[k_row, Int32(0), slot]
                            region_lane = k_row & Int32(7)
                            if valid_tok <= region_lane:
                                acc_S_mn[r, c] = -Float32.inf

                    row_scale_gen0 = softmax.online_softmax(
                        acc_S,
                        is_first=True,
                        check_inf=True,
                    )
                    softmax.rescale_O(acc_O, row_scale_gen0)
                    rP.store(acc_S.load().to(self.dtype))

                cute.arch.mbarrier_wait(mbar_v_stage + Int32(0), Int32(0))
                cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
                cute.arch.sync_warp()

                if slot_active:
                    sV_stage_slot = sV[None, None, Int32(0), slot]
                    if block_n_i32 > count_slot:
                        # Note(wangbojun/codex): Fixed-spec prefill can still hit a
                        # partial first tile on early rows. The QK side masks those
                        # columns away, but PV still reads the full shared-memory tile;
                        # leaving the inactive V rows untouched lets stale/poisoned
                        # values turn masked zeros into NaNs. Zero-pad the tail rows
                        # in place before the ldsm path consumes them.
                        for vec_idx in cutlass.range(lane, v_block_vec_count, 32, unroll=1):
                            row_zero = vec_idx // v_vecs_per_row
                            if row_zero >= count_slot:
                                d_zero = (vec_idx - row_zero * v_vecs_per_row) * q_vec_elems
                                s_ptr_zero = utils.elem_pointer_i64_offset(
                                    sV_stage_slot, (row_zero, d_zero)
                                )
                                s_vec_zero = cute.make_tensor(s_ptr_zero, vec_layout_8)
                                for vi in cutlass.range_constexpr(8):
                                    s_vec_zero[vi] = self.dtype(0)
                        cute.arch.sync_warp()
                        cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
                    sVt_stage_slot = utils.transpose_first_two_modes_view(sV_stage_slot)
                    tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt_stage_slot))
                    tOsVt = smem_thr_copy_V.partition_S(sVt_stage_slot)
                    sm90_mma.rs_smem_prefetch_gemm_with_hook(
                        tiled_mma_pv,
                        acc_O,
                        tOrP,
                        tOrVt,
                        tOsVt,
                        smem_thr_copy_V,
                    )
                if lane == 0 and stage_count < tile_split_i32:
                    cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
                    cute.arch.mbarrier_arrive(mbar_free + slot)

            for local_idx in cutlass.range(Int32(1), tile_split_i32, unroll=1):
                stage = local_idx % stage_count
                bar_idx = stage * Int32(self.q_per_cta) + slot
                wait_phase = ((local_idx - stage) // stage_count) & Int32(1)
                k_start = local_idx * block_n_i32
                block_active = slot_active and (k_start < count_slot)

                cute.arch.mbarrier_wait(mbar_k_stage + stage, wait_phase)
                cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
                cute.arch.sync_warp()

                if block_active:
                    sK_stage_slot = sK[None, None, stage, slot]
                    tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK_stage_slot))
                    tSsK = smem_thr_copy_K.partition_S(sK_stage_slot)
                    acc_S = cute.make_fragment(acc_shape_S, Float32)
                    acc_S.fill(Float32.zero)
                    sm90_mma.smem_prefetch_gemm_with_hook(
                        tiled_mma_qk,
                        acc_S,
                        tSrQ,
                        tSrK,
                        tSsQ,
                        tSsK,
                        smem_thr_copy_Q,
                        smem_thr_copy_K,
                        A_in_regs=cutlass.Boolean(True),
                    )
                    acc_S_mn = utils.make_acc_tensor_mn_view_from_mma(acc_S)
                    for r in cutlass.range_constexpr(nrow_s):
                        for c in cutlass.range_constexpr(ncol_s):
                            k_row = tScS_mn[r, c][1]
                            valid_tok = sValidTok[k_row, stage, slot]
                            region_lane = k_row & Int32(7)
                            if valid_tok <= region_lane:
                                acc_S_mn[r, c] = -Float32.inf

                    row_scale_genn = softmax.online_softmax(
                        acc_S,
                        is_first=False,
                        check_inf=True,
                    )
                    softmax.rescale_O(acc_O, row_scale_genn)
                    rP.store(acc_S.load().to(self.dtype))

                cute.arch.mbarrier_wait(mbar_v_stage + stage, wait_phase)
                cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
                cute.arch.sync_warp()

                if block_active:
                    sV_stage_slot = sV[None, None, stage, slot]
                    block_valid_tokens = count_slot - k_start
                    if block_valid_tokens < block_n_i32:
                        for vec_idx in cutlass.range(lane, v_block_vec_count, 32, unroll=1):
                            row_zero = vec_idx // v_vecs_per_row
                            if row_zero >= block_valid_tokens:
                                d_zero = (vec_idx - row_zero * v_vecs_per_row) * q_vec_elems
                                s_ptr_zero = utils.elem_pointer_i64_offset(
                                    sV_stage_slot, (row_zero, d_zero)
                                )
                                s_vec_zero = cute.make_tensor(s_ptr_zero, vec_layout_8)
                                for vi in cutlass.range_constexpr(8):
                                    s_vec_zero[vi] = self.dtype(0)
                        cute.arch.sync_warp()
                        cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
                    sVt_stage_slot = utils.transpose_first_two_modes_view(sV_stage_slot)
                    tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt_stage_slot))
                    tOsVt = smem_thr_copy_V.partition_S(sVt_stage_slot)
                    sm90_mma.rs_smem_prefetch_gemm_with_hook(
                        tiled_mma_pv,
                        acc_O,
                        tOrP,
                        tOrVt,
                        tOsVt,
                        smem_thr_copy_V,
                    )
                if lane == 0 and (local_idx + stage_count) < tile_split_i32:
                    cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
                    cute.arch.mbarrier_arrive(mbar_free + bar_idx)

            if q_valid_slot:
                if slot_active:
                    row_scale_gen_final = softmax.finalize()
                    softmax.rescale_O(acc_O, row_scale_gen_final)
                    for r in cutlass.range_constexpr(nrow_o):
                        for c in cutlass.range_constexpr(ncol_o):
                            h_out = tOcO_hv[r, c][0]
                            d_out = tOcO_hv[r, c][1]
                            if (h_out < logical_q_heads_i32) and (d_out < self.HEAD_DIM_V):
                                gO_slot[h_out, d_out] = self.dtype(acc_O_hv[r, c])

                    if const_expr(mLSE is not None):
                        for r in cutlass.range_constexpr(nrow_s):
                            h_lse = tScS_mn[r, 0][0]
                            key_lse = tScS_mn[r, 0][1]
                            if (key_lse == 0) and (h_lse < logical_q_heads_i32):
                                mLSE[q_pos_safe, lse_head_base + h_lse] = softmax.row_sum[r]
                else:
                    for idx_zero in cutlass.range(lane, self.logical_q_heads * self.HEAD_DIM_V, 32, unroll=1):
                        head_zero = idx_zero // self.HEAD_DIM_V
                        dim_zero = idx_zero - head_zero * self.HEAD_DIM_V
                        gO_slot[head_zero, dim_zero] = self.dtype(0)
                    if const_expr(mLSE is not None):
                        for head_lse in cutlass.range(lane, self.logical_q_heads, 32, unroll=1):
                            mLSE[q_pos_safe, lse_head_base + head_lse] = -Float32.inf


@functools.cache
def _get_prefill_kernel_variant(
    dtype,
    logical_q_heads: int,
    head_dim: int,
    head_dim_v: int,
    block_n: int,
    num_threads: int,
    q_per_cta: int,
    topk_windows: int,
    q_load_copy_elems: int,
    producer_async_copy_elems: int,
) -> _TokenWiseSparsePagedPrefillGQA:
    return _TokenWiseSparsePagedPrefillGQA(
        dtype=dtype,
        logical_q_heads=int(logical_q_heads),
        head_dim=int(head_dim),
        head_dim_v=int(head_dim_v),
        block_n=int(block_n),
        num_threads=int(num_threads),
        q_per_cta=int(q_per_cta),
        topk_windows=int(topk_windows),
        q_load_copy_elems=int(q_load_copy_elems),
        producer_async_copy_elems=int(producer_async_copy_elems),
    )


@cached_compile_function
def _get_compiled_prefill_kernel(
    dtype,
    logical_q_heads: int,
    head_dim: int,
    head_dim_v: int,
    block_n: int,
    num_threads: int,
    q_per_cta: int,
    topk_windows: int,
    q_load_copy_elems: int,
    producer_async_copy_elems: int,
    q_signature: tuple[object, ...],
    k_signature: tuple[object, ...],
    v_signature: tuple[object, ...],
    region_phys_signature: tuple[object, ...],
    region_count_signature: tuple[object, ...],
    work_q_global_signature: tuple[object, ...],
    work_q_local_signature: tuple[object, ...],
    work_q_len_signature: tuple[object, ...],
    out_signature: tuple[object, ...],
    lse_signature: tuple[object, ...] | None,
    q_align: int,
    k_align: int,
    v_align: int,
    region_phys_align: int,
    region_count_align: int,
    work_q_global_align: int,
    work_q_local_align: int,
    work_q_len_align: int,
    out_align: int,
    lse_align: int | None,
    softmax_scale: float,
    device_key: tuple[str, int | None],
):
    device = utils.device_from_cache_key(device_key)
    kernel_impl = _get_prefill_kernel_variant(
        dtype,
        int(logical_q_heads),
        int(head_dim),
        int(head_dim_v),
        int(block_n),
        int(num_threads),
        int(q_per_cta),
        int(topk_windows),
        int(q_load_copy_elems),
        int(producer_async_copy_elems),
    )
    q = utils.placeholder_from_signature(
        q_signature, device=device, dynamic_shape_fill=kernel_impl.q_per_cta)
    k_cache = utils.placeholder_from_signature(
        k_signature, device=device, dynamic_shape_fill=1)
    v_cache = utils.placeholder_from_signature(
        v_signature, device=device, dynamic_shape_fill=1)
    region_phys_indices = utils.placeholder_from_signature(
        region_phys_signature, device=device, dynamic_shape_fill=1)
    region_counts = utils.placeholder_from_signature(
        region_count_signature, device=device, dynamic_shape_fill=1)
    work_q_global = utils.placeholder_from_signature(
        work_q_global_signature, device=device, dynamic_shape_fill=1)
    work_q_local = utils.placeholder_from_signature(
        work_q_local_signature, device=device, dynamic_shape_fill=1)
    work_q_len = utils.placeholder_from_signature(
        work_q_len_signature, device=device, dynamic_shape_fill=1)
    out = utils.placeholder_from_signature(
        out_signature, device=device, dynamic_shape_fill=kernel_impl.q_per_cta)
    lse = (
        utils.placeholder_from_signature(
            lse_signature, device=device, dynamic_shape_fill=kernel_impl.q_per_cta)
        if lse_signature is not None
        else None
    )
    fQ = utils.make_fake_tensor_like_with_dynamic_dim(
        q, alignment=q_align, dynamic_shape_dim=0
    )
    fK = utils.make_fake_tensor_like_with_dynamic_dim(k_cache, alignment=k_align)
    fV = utils.make_fake_tensor_like_with_dynamic_dim(v_cache, alignment=v_align)
    fRegionPhys = utils.make_fake_tensor_like_with_dynamic_dim(
        region_phys_indices, alignment=region_phys_align, dynamic_shape_dim=0
    )
    fRegionCount = utils.make_fake_tensor_like_with_dynamic_dim(
        region_counts, alignment=region_count_align, dynamic_shape_dim=0
    )
    fWG = utils.make_fake_tensor_like_with_dynamic_dim(
        work_q_global, alignment=work_q_global_align, dynamic_shape_dim=0
    )
    fWL = utils.make_fake_tensor_like_with_dynamic_dim(
        work_q_local, alignment=work_q_local_align, dynamic_shape_dim=0
    )
    fWQ = utils.make_fake_tensor_like_with_dynamic_dim(
        work_q_len, alignment=work_q_len_align, dynamic_shape_dim=0
    )
    fO = utils.make_fake_tensor_like_with_dynamic_dim(
        out, alignment=out_align, dynamic_shape_dim=0
    )
    fLSE = (
        utils.make_fake_tensor_like_with_dynamic_dim(
            lse, alignment=lse_align, dynamic_shape_dim=0
        )
        if lse is not None
        else None
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel_impl,
        fQ,
        fK,
        fV,
        fRegionPhys,
        fRegionCount,
        fWG,
        fWL,
        fWQ,
        fO,
        fLSE,
        Int32(1),
        cutlass.Float32(softmax_scale),
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


class TokenWiseFlashAttnFwdSm90ManualMbarrierMmaPrefillPagedGQA:
    """GQA-specific sparse paged-KV prefill wrapper using direct physical region metadata."""

    arch: int = 90

    def __init__(
        self,
        *,
        dtype,
        logical_q_heads: int,
        head_dim: int = 128,
        head_dim_v: int = 128,
        block_n: int = 16,
        num_threads: int = 256,
        q_per_cta: int = 2,
    ) -> None:
        self.dtype = dtype
        self.logical_q_heads = int(logical_q_heads)
        self.HEAD_DIM = int(head_dim)
        self.HEAD_DIM_V = int(head_dim_v)
        self.block_n = int(block_n)
        self.num_threads = int(num_threads)
        self.q_per_cta = int(q_per_cta)
        self.superblock_tokens = int(_TokenWiseSparsePagedPrefillGQA.SUPERBLOCK_TOKENS)
        if self.superblock_tokens % self.block_n != 0:
            raise ValueError(
                f"superblock_tokens={self.superblock_tokens} must be divisible by block_n={self.block_n}"
            )
        self.superblock_subtiles = self.superblock_tokens // self.block_n

        if self.dtype not in (cutlass.Float16, cutlass.BFloat16):
            raise TypeError("Only fp16/bf16 are supported")
        if self.logical_q_heads not in (4, 8, 16):
            raise ValueError(
                "GQA token-sparse prefill wrapper supports only logical_q_heads in {4, 8, 16}, "
                f"got {self.logical_q_heads}"
            )
        if self.HEAD_DIM not in _TokenWiseSparsePagedPrefillGQA.SUPPORTED_HEAD_DIMS or self.HEAD_DIM_V not in _TokenWiseSparsePagedPrefillGQA.SUPPORTED_HEAD_DIMS:
            raise ValueError(
                "GQA token-sparse prefill wrapper supports only head_dim/head_dim_v in "
                f"{_TokenWiseSparsePagedPrefillGQA.SUPPORTED_HEAD_DIMS}, got ({self.HEAD_DIM}, {self.HEAD_DIM_V})"
            )
        if self.HEAD_DIM != self.HEAD_DIM_V:
            raise ValueError(
                "GQA token-sparse prefill keeps the equal-dim contract, "
                f"got head_dim={self.HEAD_DIM}, head_dim_v={self.HEAD_DIM_V}"
            )
        if (self.q_per_cta, self.block_n, self.num_threads) not in _TokenWiseSparsePagedPrefillGQA.SUPPORTED_TOPOLOGIES:
            raise ValueError(
                "GQA token-sparse prefill supports only "
                f"{sorted(_TokenWiseSparsePagedPrefillGQA.SUPPORTED_TOPOLOGIES)}"
            )

    def _select_vector_copy_elems(self, *, min_align: int, name: str) -> int:
        min_align = int(min_align)
        if min_align >= 16:
            return 8
        if min_align >= 8:
            return 4
        if min_align >= 4:
            return 2
        raise ValueError(
            f"GQA token-sparse prefill requires {name} pointer alignment >= 4 bytes, "
            f"got min_align={min_align}"
        )

    def _validate_128b_aligned_strides(self, t, *, name: str):
        align_elems = 128 // t.element_size() // 8
        for i, s in enumerate(t.stride()[:-1]):
            if int(s) % align_elems != 0:
                raise ValueError(
                    f"{name}.stride[{i}]={int(s)} is not divisible by {align_elems}; "
                    "this kernel requires 128-bit aligned logical strides"
                )

    def _validate_pointer_alignment(self, t, *, name: str, min_align: int) -> None:
        ptr = int(t.data_ptr())
        if ptr % int(min_align) != 0:
            raise ValueError(
                f"{name}.data_ptr()={ptr} is not {int(min_align)}-byte aligned; "
                "Step3p5 sparse prefill requires stable pointer alignment for "
                "precompiled CuTeDSL kernels"
            )

    def _validate_inputs(
        self,
        q,
        k_cache,
        v_cache,
        o,
        lse,
        region_phys_indices,
        region_counts,
        work_q_global,
        work_q_local,
        work_q_len,
        *,
        seq_q: int,
    ):
        import torch

        total_q, Hq, D = q.shape
        num_pages_k, page_size_k, Hkv, Dk = k_cache.shape
        num_pages_v, page_size_v, Hkv_v, Dv = v_cache.shape

        expected_q_heads = self.logical_q_heads * Hkv
        if (Hq, D) != (expected_q_heads, self.HEAD_DIM):
            raise ValueError(
                f"Expected q shape (*, {expected_q_heads}, {self.HEAD_DIM}) "
                f"for {Hkv} local KV heads, got {tuple(q.shape)}"
            )
        if Dk != self.HEAD_DIM or Dv != self.HEAD_DIM_V:
            raise ValueError(
                "Input head dimensions do not match GQA token-sparse prefill config "
                f"({self.HEAD_DIM}, {self.HEAD_DIM_V})"
            )
        if Hkv <= 0 or Hkv_v != Hkv:
            raise ValueError(f"Paged K/V local head mismatch: {Hkv} vs {Hkv_v}")
        if num_pages_k != num_pages_v:
            raise ValueError(f"Paged K/V page count mismatch: {num_pages_k} vs {num_pages_v}")
        if page_size_k != page_size_v:
            raise ValueError(f"Paged K/V page size mismatch: {page_size_k} vs {page_size_v}")
        if page_size_k != 16:
            raise ValueError(f"Expected page size 16, got {page_size_k}")
        if o.shape != (total_q, Hq, self.HEAD_DIM_V):
            raise ValueError(f"Expected o shape {(total_q, Hq, self.HEAD_DIM_V)}, got {tuple(o.shape)}")
        self._validate_128b_aligned_strides(q, name="q")
        self._validate_128b_aligned_strides(k_cache, name="k_cache")
        self._validate_128b_aligned_strides(v_cache, name="v_cache")
        self._validate_128b_aligned_strides(o, name="o")

        if lse is not None:
            if lse.shape != (total_q, Hq):
                raise ValueError(f"Expected lse shape {(total_q, Hq)}, got {tuple(lse.shape)}")
            if lse.dtype != torch.float32:
                raise ValueError("lse must be float32")
            if lse.device != q.device:
                raise ValueError("lse must be on the same device as q")

        if region_phys_indices is None:
            raise ValueError("GQA token-sparse prefill requires region_phys_indices")
        if region_phys_indices.dtype != torch.int32:
            raise ValueError(f"region_phys_indices must be int32, got {region_phys_indices.dtype}")
        if region_phys_indices.device != q.device:
            raise ValueError("region_phys_indices must be on the same device as q")

        if region_counts is None:
            raise ValueError("GQA token-sparse prefill requires region_counts")
        if region_counts.dtype != torch.int32:
            raise ValueError("region_counts must be int32")
        if region_counts.device != q.device:
            raise ValueError("region_counts must be on the same device as q")

        if region_phys_indices.ndim != 2:
            raise ValueError(
                f"region_phys_indices must be [Tq, topk_regions], got {tuple(region_phys_indices.shape)}"
            )
        if region_counts.ndim != 1:
            raise ValueError(f"region_counts must be [Tq], got {tuple(region_counts.shape)}")
        if not region_phys_indices.is_contiguous():
            raise ValueError(
                f"region_phys_indices must be contiguous, got stride={tuple(region_phys_indices.stride())}"
            )
        if not region_counts.is_contiguous():
            raise ValueError(
                f"region_counts must be contiguous, got stride={tuple(region_counts.stride())}"
            )
        expected_metadata_rows = int(seq_q) * Hkv
        if int(region_phys_indices.shape[0]) < expected_metadata_rows:
            raise ValueError(
                "region_phys_indices first dim must cover seq_q * local_kv_heads="
                f"{expected_metadata_rows}, got {int(region_phys_indices.shape[0])}"
            )
        if int(region_counts.shape[0]) < expected_metadata_rows:
            raise ValueError(
                "region_counts first dim must cover seq_q * local_kv_heads="
                f"{expected_metadata_rows}, got {int(region_counts.shape[0])}"
            )

        work_tensors = {
            "work_q_global": work_q_global,
            "work_q_local": work_q_local,
            "work_q_len": work_q_len,
        }
        work_size = None
        for name, tensor in work_tensors.items():
            if tensor is None:
                raise ValueError(f"{name} is required for GQA token-sparse prefill")
            if tensor.dtype != torch.int32:
                raise ValueError(f"{name} must be int32")
            if tensor.device != q.device:
                raise ValueError(f"{name} must be on the same device as q")
            if tensor.ndim != 1:
                raise ValueError(f"{name} must be 1D, got shape={tuple(tensor.shape)}")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous, got stride={tuple(tensor.stride())}")
            if work_size is None:
                work_size = int(tensor.numel())
            elif int(tensor.numel()) != work_size:
                raise ValueError("All work metadata tensors must have the same length")
        if work_size is None or work_size <= 0:
            raise ValueError("GQA token-sparse prefill requires at least one work tile")

        return (
            region_phys_indices,
            region_counts,
            work_tensors["work_q_global"],
            work_tensors["work_q_local"],
            work_tensors["work_q_len"],
        )

    def _run_gqa_phys_region(
        self,
        q,
        k_cache,
        v_cache,
        region_phys_indices,
        region_counts,
        work_q_global,
        work_q_local,
        work_q_len,
        o,
        *,
        seq_q: int,
        topk_windows: int,
        lse,
        softmax_scale: float,
    ):
        self._validate_pointer_alignment(q, name="q", min_align=16)
        self._validate_pointer_alignment(k_cache, name="k_cache", min_align=16)
        self._validate_pointer_alignment(v_cache, name="v_cache", min_align=16)
        self._validate_pointer_alignment(o, name="o", min_align=16)
        self._validate_pointer_alignment(
            region_phys_indices, name="region_phys_indices", min_align=4)
        self._validate_pointer_alignment(
            region_counts, name="region_counts", min_align=4)
        self._validate_pointer_alignment(
            work_q_global, name="work_q_global", min_align=4)
        self._validate_pointer_alignment(
            work_q_local, name="work_q_local", min_align=4)
        self._validate_pointer_alignment(
            work_q_len, name="work_q_len", min_align=4)
        if lse is not None:
            self._validate_pointer_alignment(lse, name="lse", min_align=4)
        q_align = 16
        k_align = 16
        v_align = 16
        o_align = 16
        region_phys_align = 4
        region_count_align = 4
        work_q_global_align = 4
        work_q_local_align = 4
        work_q_len_align = 4
        lse_align = 4 if lse is not None else None

        q_load_copy_elems = self._select_vector_copy_elems(
            min_align=q_align,
            name="Q",
        )
        producer_async_copy_elems = self._select_vector_copy_elems(
            min_align=min(k_align, v_align),
            name="K/V",
        )
        compiled = _get_compiled_prefill_kernel(
            self.dtype,
            int(self.logical_q_heads),
            int(self.HEAD_DIM),
            int(self.HEAD_DIM_V),
            int(self.block_n),
            int(self.num_threads),
            int(self.q_per_cta),
            int(topk_windows),
            int(q_load_copy_elems),
            int(producer_async_copy_elems),
            utils.tensor_signature_dynamic(q, dynamic_shape_dims=(0,)),
            utils.tensor_signature_dynamic(k_cache),
            utils.tensor_signature_dynamic(v_cache),
            utils.tensor_signature_dynamic(region_phys_indices, dynamic_shape_dims=(0,)),
            utils.tensor_signature_dynamic(region_counts, dynamic_shape_dims=(0,)),
            utils.tensor_signature_dynamic(work_q_global, dynamic_shape_dims=(0,)),
            utils.tensor_signature_dynamic(work_q_local, dynamic_shape_dims=(0,)),
            utils.tensor_signature_dynamic(work_q_len, dynamic_shape_dims=(0,)),
            utils.tensor_signature_dynamic(o, dynamic_shape_dims=(0,)),
            None if lse is None else utils.tensor_signature_dynamic(lse, dynamic_shape_dims=(0,)),
            int(q_align),
            int(k_align),
            int(v_align),
            int(region_phys_align),
            int(region_count_align),
            int(work_q_global_align),
            int(work_q_local_align),
            int(work_q_len_align),
            int(o_align),
            None if lse is None else int(lse_align),
            float(softmax_scale),
            utils.device_cache_key(q.device),
        )
        compiled(
            q,
            k_cache,
            v_cache,
            region_phys_indices,
            region_counts,
            work_q_global,
            work_q_local,
            work_q_len,
            o,
            lse,
            cutlass.Int32(seq_q),
            cutlass.Float32(softmax_scale),
        )
        return o

    def run(
        self,
        q,
        k_cache,
        v_cache,
        o,
        *,
        lse=None,
        region_counts=None,
        region_phys_indices=None,
        work_q_global=None,
        work_q_local=None,
        work_q_len=None,
        seq_q: Optional[int] = None,
        softmax_scale: Optional[float] = None,
        stream: Optional[cuda.CUstream] = None,
    ):
        import torch

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(float(self.HEAD_DIM))
        seq_q_i = int(q.shape[0]) if seq_q is None else int(seq_q)
        if seq_q_i <= 0 or seq_q_i > int(q.shape[0]):
            raise ValueError(f"seq_q must be in (0, q.shape[0]], got seq_q={seq_q_i}, q_shape={tuple(q.shape)}")
        if stream is not None:
            raise RuntimeError(
                "Explicit CUDA streams are not supported by the TVM-FFI AOT "
                "prefill path; invoke it under the desired current torch stream."
            )
        record_torch_stream = torch.cuda.current_stream(device=q.device)

        (
            phys_sq,
            counts_sq,
            work_q_global,
            work_q_local,
            work_q_len,
        ) = self._validate_inputs(
            q,
            k_cache,
            v_cache,
            o,
            lse,
            region_phys_indices,
            region_counts,
            work_q_global,
            work_q_local,
            work_q_len,
            seq_q=seq_q_i,
        )
        is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
        can_record_stream = not (is_capturing is not None and bool(is_capturing()))
        if can_record_stream:
            # Note(wangbojun/codex): This kernel consumes temporary sparse
            # metadata/work-queue tensors that are often produced immediately
            # before launch on the vLLM path. Because the launch goes through the
            # Cutlass/CUDA driver path instead of an ATen op, the allocator does
            # not automatically know these storages stay live on the stream.
            # Record the launch stream explicitly so post-launch allocations do
            # not recycle the metadata buffers while the kernel is still reading
            # them.
            for tensor in (
                q,
                k_cache,
                v_cache,
                o,
                lse,
                phys_sq,
                counts_sq,
                work_q_global,
                work_q_local,
                work_q_len,
            ):
                if tensor is not None:
                    tensor.record_stream(record_torch_stream)

        return self._run_gqa_phys_region(
            q,
            k_cache,
            v_cache,
            phys_sq,
            counts_sq,
            work_q_global,
            work_q_local,
            work_q_len,
            o,
            seq_q=seq_q_i,
            topk_windows=int(phys_sq.shape[1]),
            lse=lse,
            softmax_scale=float(softmax_scale),
        )


__all__ = ["TokenWiseFlashAttnFwdSm90ManualMbarrierMmaPrefillPagedGQA"]
