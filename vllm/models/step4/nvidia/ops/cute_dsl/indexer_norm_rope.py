from __future__ import annotations

from typing import Optional, Tuple
from functools import partial

import cuda.bindings.driver as cuda


import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

import torch
from torch import Tensor

ArgsAndKwargs = tuple[list[object], dict[str, object]]


def accuracy_case(*args, **kwargs):
    """No-op compatibility decorator for source-retained accuracy metadata."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda func: func

from vllm.models.step4.nvidia.ops.cute_dsl.cute_dsl_utils import (
    cache_get_or_create,
    cute_compile_with_spec,
    dynamic_dim0_specs,
)
from vllm.models.step4.nvidia.ops.cute_dsl.utils import torch2cute_dtype_map, row_reduce
from vllm.models.step4.nvidia.ops.cute_dsl.reduction_base import ReductionBase
import vllm.models.step4.nvidia.ops.cute_dsl.utils as utils
from vllm.models.step4.nvidia.ops.cute_dsl.flash_attn.copy_utils import get_copy_atom
import vllm.models.step4.nvidia.ops.cute_dsl.flash_attn.utils as copy_utils

class _TensorAnnotation:
    @classmethod
    def __class_getitem__(cls, _item):
        return Tensor


Float16Tensor = IntTensor = _TensorAnnotation


def benchmark_case(*args, **kwargs):
    """No-op compatibility decorator for source-retained benchmark metadata."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda func: func


# Fused Step3p5 sparse-indexer q/k norm + RoPE.
#
# Mirrors ``FusedQKNormRope`` from the sibling ``qknorm_rope.py`` module, but
# specializes it for the DSA
# sparse indexer, whose q/k projections are two *separate* tensors and whose
# norms differ:
#   * q head  -> RMSNorm      : y = x / sqrt(mean(x^2) + eps) * (w + q_norm_weight_bias)
#   * k head  -> LayerNorm    : y = (x - mean) / sqrt(var + eps) * w + b
# Both then receive the same NeoX-style partial RoPE (rotary_dim columns rotated,
# the rest passed through). One CTA handles one head; grid-z spans nq + nk heads.
class FusedIndexerNormRope(ReductionBase):
    def __init__(
        self,
        dtype: cutlass.Numeric,
        N: int,
        head_dim: int,
        num_q_head: int,
        num_k_head: int,
        rotary_dim: int,
        num_z_head: int = 0,
    ):
        # `head_dim` here plays the role of the indexer `proxy_dim`.
        super().__init__(dtype, N, stage=1)
        self.reload_from = None if N <= 16384 else "smem"
        self.delay_w_load = False
        self.head_dim = head_dim
        self.num_q_head = num_q_head
        self.num_k_head = num_k_head
        self.num_z_head = num_z_head
        self.rotary_dim = rotary_dim
        self.row_stages = 3 if head_dim == 192 else 1

    def _calculate_threads_per_head(self):
        """Number of threads cooperating on one head row."""
        N = self.head_dim
        if N <= 64:
            threads_per_head = 8
        elif N <= 128:
            threads_per_head = 16
        elif N <= 192:
            threads_per_head = 16
        elif N <= 3072:
            threads_per_head = 32
        elif N <= 6144:
            threads_per_head = 64
        elif N <= 16384:
            threads_per_head = 128
        else:
            threads_per_head = 256

        vals_per_thread = 64 // self.dtype.width
        rotary_threads = self.rotary_dim // vals_per_thread
        while threads_per_head < rotary_threads:
            threads_per_head *= 2
        return threads_per_head

    def _get_num_threads(self):
        return 128 if self.N <= 16384 else 256

    def _get_num_copy_bits(self):
        vals_per_thread_64b = 64 // self.dtype.width
        if self.head_dim % (self._calculate_threads_per_head() * vals_per_thread_64b) == 0 and \
           self.head_dim % (self._calculate_threads_per_head() * (128 // self.dtype.width)) != 0:
            return 64
        return 128

    def _get_tv_layout(self, num_copy_bits=128):
        vecsize = num_copy_bits // self.dtype.width
        assert self.head_dim % vecsize == 0, f"Input N {self.head_dim} is not divisible by vector size {vecsize}"
        num_threads = self._get_num_threads()
        assert num_threads % cute.arch.WARP_SIZE == 0

        threads_per_head = self._calculate_threads_per_head()
        num_blocks_N = cute.ceil_div(self.head_dim // vecsize, threads_per_head * self.cluster_n)
        cols_per_block = num_threads // threads_per_head
        tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_head)
        tv_layout = cute.make_layout(
            ((threads_per_head, cols_per_block), (vecsize, num_blocks_N)),
            stride=(
                (vecsize * cols_per_block, 1),
                (cols_per_block, cols_per_block * vecsize * threads_per_head),
            ),
        )
        return tiler_mn, tv_layout

    def _set_cluster_n(self):
        N = self.head_dim
        if const_expr(self.dtype.width == 16):
            if N <= 16 * 1024:
                cluster_n = 1
            elif N <= 32 * 1024:
                cluster_n = 2
            elif N <= 64 * 1024:
                cluster_n = 4
            elif N <= 128 * 1024:
                cluster_n = 8
            else:
                cluster_n = 16
        else:
            if N <= 32 * 1024:
                cluster_n = 1
            elif N <= 64 * 1024:
                cluster_n = 2
            elif N <= 128 * 1024:
                cluster_n = 4
            elif N <= 256 * 1024:
                cluster_n = 8
            else:
                cluster_n = 16
        self.cluster_n = cluster_n

    def _smem_size_in_bytes(self, tiler_mn, num_warps):
        return (
            self.row_stages * cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn))
            + self.stage * num_warps * self.cluster_n * (self.reduction_dtype.width // 8)
            + self.stage * (cutlass.Int64.width // 8)
        )

    @cute.jit
    def __call__(
        self,
        mQin: cute.Tensor,
        mKin: cute.Tensor,
        mQW: Optional[cute.Tensor],
        mKW: Optional[cute.Tensor],
        mKB: Optional[cute.Tensor],
        mCos: cute.Tensor,
        mSin: cute.Tensor,
        mPosId: cute.Tensor,
        mQout: cute.Tensor,
        mKout: cute.Tensor,
        mZin: Optional[cute.Tensor],
        mZout: Optional[cute.Tensor],
        head_dim: Int32,
        num_q_head: Int32,
        num_k_head: Int32,
        stream: cuda.CUstream,
        eps: Float32 = 1e-6,
        q_norm_weight_bias: Float32 = 1.0,
    ):
        # (tokens, n_head, head_dim) semistatic views. q/k/z may be strided
        # column-slices of one fused projection buffer, so the row stride need
        # only be 16B-aligned (not a multiple of head_dim). The middle stride is
        # never dereferenced (the kernel rebuilds per-head 2D views).
        new_stride = lambda t, n_head: (
            cute.assume(t.stride[0], divby=(128 // t.element_type.width)),
            self.head_dim,
            t.stride[1],
        )

        semistatic_shape_Q = (*mQin.shape[:-1], self.num_q_head, self.head_dim)
        semistatic_shape_K = (*mKin.shape[:-1], self.num_k_head, self.head_dim)
        mQin = cute.make_tensor(mQin.iterator, cute.make_layout(semistatic_shape_Q, stride=new_stride(mQin, self.num_q_head)))
        mKin = cute.make_tensor(mKin.iterator, cute.make_layout(semistatic_shape_K, stride=new_stride(mKin, self.num_k_head)))
        mQout = cute.make_tensor(mQout.iterator, cute.make_layout(semistatic_shape_Q, stride=new_stride(mQout, self.num_q_head)))
        mKout = cute.make_tensor(mKout.iterator, cute.make_layout(semistatic_shape_K, stride=new_stride(mKout, self.num_k_head)))
        if const_expr(mZin is not None):
            semistatic_shape_Z = (*mZin.shape[:-1], self.num_z_head, self.head_dim)
            mZin = cute.make_tensor(mZin.iterator, cute.make_layout(semistatic_shape_Z, stride=new_stride(mZin, self.num_z_head)))
            mZout = cute.make_tensor(mZout.iterator, cute.make_layout(semistatic_shape_Z, stride=new_stride(mZout, self.num_z_head)))

        semistatic_shape_Cos_Sin = (*mCos.shape[:-1], self.rotary_dim)
        new_stride_Cos_Sin = lambda t: (
            cute.assume(t.stride[0], divby=64 // t.element_type.width),
            t.stride[1],
        )
        mCos = cute.make_tensor(mCos.iterator, cute.make_layout(semistatic_shape_Cos_Sin, stride=new_stride_Cos_Sin(mCos)))
        mSin = cute.make_tensor(mSin.iterator, cute.make_layout(semistatic_shape_Cos_Sin, stride=new_stride_Cos_Sin(mSin)))

        self._set_cluster_n()
        largest_dtype_width = const_expr(mQin.element_type.width)
        tiler_mn, tv_layout = self._get_tv_layout(
            num_copy_bits=self._get_num_copy_bits() // largest_dtype_width * mQin.element_type.width
        )

        num_threads = cute.size(tv_layout, mode=[0])
        num_warps = num_threads // cute.arch.WARP_SIZE
        if const_expr(mQW is not None):
            mQW = cute.make_tensor(
                mQW.iterator,
                cute.prepend(mQW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))),
            )
        if const_expr(mKW is not None):
            mKW = cute.make_tensor(
                mKW.iterator,
                cute.prepend(mKW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))),
            )
        if const_expr(mKB is not None):
            mKB = cute.make_tensor(
                mKB.iterator,
                cute.prepend(mKB.layout, cute.make_layout((tiler_mn[0],), stride=(0,))),
            )

        self.kernel(
            mQin,
            mKin,
            mQW,
            mKW,
            mKB,
            mCos,
            mSin,
            mPosId,
            mQout,
            mKout,
            mZin,
            mZout,
            num_q_head,
            num_k_head,
            eps,
            q_norm_weight_bias,
            self.reload_from,
        ).launch(
            grid=[cute.ceil_div(mQin.shape[0], tiler_mn[0] * self.row_stages), self.cluster_n, self.num_q_head + self.num_k_head + self.num_z_head],
            block=[num_threads, 1, 1],
            cluster=([1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None),
            smem=self._smem_size_in_bytes(tiler_mn, num_warps),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQin: cute.Tensor,
        mKin: cute.Tensor,
        mQW: Optional[cute.Tensor],
        mKW: Optional[cute.Tensor],
        mKB: Optional[cute.Tensor],
        mCos: cute.Tensor,
        mSin: cute.Tensor,
        mPosId: cute.Tensor,
        mQout: cute.Tensor,
        mKout: cute.Tensor,
        mZin: Optional[cute.Tensor],
        mZout: Optional[cute.Tensor],
        num_q_head: Int32,
        num_k_head: Int32,
        eps: cute.Float32,
        q_norm_weight_bias: cute.Float32,
        reload_from: cutlass.Constexpr = None,
        delay_w_load: cutlass.Constexpr = False,
    ):
        largest_dtype_width = cutlass.const_expr(mQin.element_type.width)
        tiler_mn, tv_layout = self._get_tv_layout(
            num_copy_bits=self._get_num_copy_bits() // largest_dtype_width * mQin.element_type.width
        )
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, bidz = cute.arch.block_idx()

        VALS_PER_THREAD = 64 // mCos.element_type.width
        rotary_threads = self.rotary_dim // VALS_PER_THREAD
        LANES_PER_HEAD = tv_layout.shape[0][0]
        num_threads = cute.size(tv_layout, mode=[0])
        rowid_in_block = tidx // LANES_PER_HEAD

        if const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = const_expr(0)

        ROW_STAGES = cutlass.const_expr(self.row_stages)

        # smem: ROW_STAGES X-buffers + reduction buffer / mbar.
        smem = cutlass.utils.SmemAllocator()
        sX_stages = []
        for _stage in cutlass.range_constexpr(ROW_STAGES):
            sX_stages.append(
                smem.allocate_tensor(
                    mQin.element_type,
                    cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                    byte_alignment=16,
                )
            )

        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        num_tokens = mQin.shape[0]
        head_col = mQin.shape[2]  # per-head width == proxy_dim
        shape_X = (num_tokens, head_col)
        idX = cute.make_identity_tensor(shape_X)

        is_q = bidz < self.num_q_head
        is_qk = bidz < self.num_q_head + self.num_k_head

        # Per-head 2D views (tokens, head_dim). Out-of-range head offsets are
        # computed for the non-taken branch but never dereferenced.
        new_iterator = (mQin.iterator + bidz * mQin.shape[2]).align(16)
        mQin = cute.make_tensor(
            new_iterator,
            cute.make_layout((mQin.shape[0], mQin.shape[2]), stride=(mQin.stride[0], mQin.stride[2])),
        )
        new_iterator = (mQout.iterator + bidz * mQout.shape[2]).align(16)
        mQout = cute.make_tensor(
            new_iterator,
            cute.make_layout((mQout.shape[0], mQout.shape[2]), stride=(mQout.stride[0], mQout.stride[2])),
        )
        new_iterator = (mKin.iterator + (bidz - self.num_q_head) * mKin.shape[2]).align(16)
        mKin = cute.make_tensor(
            new_iterator,
            cute.make_layout((mKin.shape[0], mKin.shape[2]), stride=(mKin.stride[0], mKin.stride[2])),
        )
        new_iterator = (mKout.iterator + (bidz - self.num_q_head) * mKout.shape[2]).align(16)
        mKout = cute.make_tensor(
            new_iterator,
            cute.make_layout((mKout.shape[0], mKout.shape[2]), stride=(mKout.stride[0], mKout.stride[2])),
        )
        if const_expr(mZin is not None):
            new_iterator = (mZin.iterator + (bidz - self.num_q_head - self.num_k_head) * mZin.shape[2]).align(16)
            mZin = cute.make_tensor(
                new_iterator,
                cute.make_layout((mZin.shape[0], mZin.shape[2]), stride=(mZin.stride[0], mZin.stride[2])),
            )
            new_iterator = (mZout.iterator + (bidz - self.num_q_head - self.num_k_head) * mZout.shape[2]).align(16)
            mZout = cute.make_tensor(
                new_iterator,
                cute.make_layout((mZout.shape[0], mZout.shape[2]), stride=(mZout.stride[0], mZout.stride[2])),
            )

        num_copy_elems_X = tv_layout.shape[1][0]
        copy_atom_load_X_async = get_copy_atom(mQin.element_type, num_copy_elems_X, is_async=True)
        thr_copy_X = cute.make_tiled_copy(copy_atom_load_X_async, tv_layout, tiler_mn).get_slice(tidx)

        # Norm weight (q: RMS weight, k: LN weight) and k bias fragments.
        gQW = cute.local_tile(mQW, tiler_mn, (0, cluster_y))
        tXgQW = thr_copy_X.partition_S(gQW)
        tXrQW = cute.make_fragment_like(tXgQW)
        tXrQW.fill(0.0)

        gKW = cute.local_tile(mKW, tiler_mn, (0, cluster_y))
        tXgKW = thr_copy_X.partition_S(gKW)
        tXrKW = cute.make_fragment_like(tXgKW)
        tXrKW.fill(0.0)

        gKB = cute.local_tile(mKB, tiler_mn, (0, cluster_y))
        tXgKB = thr_copy_X.partition_S(gKB)
        tXrKB = cute.make_fragment_like(tXgKB)
        tXrKB.fill(0.0)

        gX_tmpl = cute.local_tile(mQin, tiler_mn, (0, cluster_y))
        tXgX_tmpl = thr_copy_X.partition_S(gX_tmpl)
        tXrX = cute.make_fragment_like(tXgX_tmpl)

        gQ_tmpl = cute.local_tile(mQout, tiler_mn, (0, cluster_y))
        tXrQ = cute.make_fragment_like(thr_copy_X.partition_D(gQ_tmpl))
        tXrK = cute.make_fragment_like(thr_copy_X.partition_D(gQ_tmpl))

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        is_even_N = cutlass.const_expr(shape_X[1] == tiler_mn[1] * self.cluster_n)
        cX = cute.local_tile(idX, tiler_mn, (0, cluster_y))
        tXpX = utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape_X[1]) if not is_even_N else None
        copy = partial(copy_utils.copy, pred=tXpX, num_copy_elems=num_copy_elems_X)

        head_idx = bidz
        threads_per_row = tv_layout.shape[0][0]
        lane = tidx % LANES_PER_HEAD

        # === Phase 1: issue cp.async of X for all stages (from q/k/z input) ===
        for stage_i in cutlass.range_constexpr(ROW_STAGES):
            stage_bidx = bidx * ROW_STAGES + stage_i
            stage_token_idx = stage_bidx * tiler_mn[0] + rowid_in_block
            tXsX_s = thr_copy_X.partition_D(sX_stages[stage_i])
            tXgXq_s = thr_copy_X.partition_S(cute.local_tile(mQin, tiler_mn, (stage_bidx, cluster_y)))
            tXgXk_s = thr_copy_X.partition_S(cute.local_tile(mKin, tiler_mn, (stage_bidx, cluster_y)))
            if stage_token_idx < num_tokens:
                if is_q:
                    copy(tXgXq_s, tXsX_s, is_async=True)
                elif is_qk:
                    copy(tXgXk_s, tXsX_s, is_async=True)
                else:
                    if const_expr(mZin is not None):
                        tXgXz_s = thr_copy_X.partition_S(cute.local_tile(mZin, tiler_mn, (stage_bidx, cluster_y)))
                        copy(tXgXz_s, tXsX_s, is_async=True)
            cute.arch.cp_async_commit_group()

        # === Load norm params once (q: RMS weight, k: LN weight+bias; z: none) ===
        if not delay_w_load:
            if is_q:
                copy(tXgQW, tXrQW)
            elif is_qk:
                copy(tXgKW, tXrKW)
                copy(tXgKB, tXrKB)

        # === Phase 2: per-stage norm + RoPE + store ===
        for stage_i in cutlass.range_constexpr(ROW_STAGES):
            stage_bidx = bidx * ROW_STAGES + stage_i
            stage_token_idx = stage_bidx * tiler_mn[0] + rowid_in_block

            sX_s = sX_stages[stage_i]
            tXsX_s = thr_copy_X.partition_D(sX_s)

            gQ_s = cute.local_tile(mQout, tiler_mn, (stage_bidx, cluster_y))
            gK_s = cute.local_tile(mKout, tiler_mn, (stage_bidx, cluster_y))
            tXgQ_s = thr_copy_X.partition_D(gQ_s)
            tXgK_s = thr_copy_X.partition_D(gK_s)
            if const_expr(mZout is not None):
                tXgZ_s = thr_copy_X.partition_D(cute.local_tile(mZout, tiler_mn, (stage_bidx, cluster_y)))

            cute.arch.cp_async_wait_group(ROW_STAGES - 1 - stage_i)

            cute.autovec_copy(tXsX_s, tXrX)
            x = tXrX.load().to(cute.Float32)

            if is_q:
                # RMSNorm
                sum_sq_x = row_reduce(
                    x * x,
                    cute.ReductionOp.ADD,
                    threads_per_row,
                    reduction_buffer[None, None, 0],
                    mbar_ptr,
                    init_val=0.0,
                    hook_fn=(cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None),
                )
                rstd = cute.math.rsqrt(sum_sq_x / shape_X[1] + eps, fastmath=True)

                if delay_w_load:
                    copy(tXgQW, tXrQW)

                if reload_from == "smem" or reload_from == "gmem":
                    if reload_from == "smem":
                        cute.autovec_copy(tXsX_s, tXrX)
                    else:
                        gX_s_reload = cute.local_tile(mQin, tiler_mn, (stage_bidx, cluster_y))
                        copy(thr_copy_X.partition_S(gX_s_reload), tXrX)
                    x = tXrX.load().to(cute.Float32)

                y = x * rstd * (tXrQW.load().to(cute.Float32) + q_norm_weight_bias)
                tXrQ.store(y.to(tXrQ.element_type))
                copy(tXrQ, tXsX_s)
            elif is_qk:
                # LayerNorm
                n_f = cute.Float32(shape_X[1])
                sum_x = row_reduce(
                    x,
                    cute.ReductionOp.ADD,
                    threads_per_row,
                    reduction_buffer[None, None, 0],
                    mbar_ptr,
                    init_val=0.0,
                    hook_fn=(cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None),
                )
                mean = sum_x / n_f
                sum_sq = row_reduce(
                    x * x,
                    cute.ReductionOp.ADD,
                    threads_per_row,
                    reduction_buffer[None, None, 0],
                    mbar_ptr,
                    init_val=0.0,
                )
                var = sum_sq / n_f - mean * mean
                rstd = cute.math.rsqrt(var + eps, fastmath=True)

                if delay_w_load:
                    copy(tXgKW, tXrKW)
                    copy(tXgKB, tXrKB)

                if reload_from == "smem" or reload_from == "gmem":
                    if reload_from == "smem":
                        cute.autovec_copy(tXsX_s, tXrX)
                    else:
                        gX_s_reload = cute.local_tile(mKin, tiler_mn, (stage_bidx, cluster_y))
                        copy(thr_copy_X.partition_S(gX_s_reload), tXrX)
                    x = tXrX.load().to(cute.Float32)

                y = (x - mean) * rstd * tXrKW.load().to(cute.Float32) + tXrKB.load().to(cute.Float32)
                tXrK.store(y.to(tXrK.element_type))
                copy(tXrK, tXsX_s)
            cute.arch.sync_warp()

            # Apply NeoX partial RoPE to this stage's smem buffer (q/k only; z is
            # a pure passthrough with no norm/rope).
            if const_expr(mCos is not None and mSin is not None and mPosId is not None):
                if is_qk and (lane < rotary_threads):
                    pos = (
                        cutlass.Int64(mPosId[stage_token_idx])
                        if stage_token_idx < num_tokens
                        else cutlass.Int64(0)
                    )

                    mCos_s = utils.domain_offset_i64((pos, 0), mCos)
                    mSin_s = utils.domain_offset_i64((pos, 0), mSin)
                    gCos = cute.local_tile(mCos_s, (1, self.rotary_dim), (0, 0))
                    gSin = cute.local_tile(mSin_s, (1, self.rotary_dim), (0, 0))

                    copy_atom_b16 = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        mCos.element_type,
                        num_bits_per_copy=64,
                    )

                    thr_layout_cs = cute.make_layout((1, rotary_threads), stride=(0, 1))
                    val_layout_cs = cute.make_layout((1, VALS_PER_THREAD), stride=(0, 1))
                    tiled_cs = cute.make_tiled_copy_tv(copy_atom_b16, thr_layout_cs, val_layout_cs)
                    thr_cs = tiled_cs.get_slice(lane)

                    tCgC = thr_cs.partition_S(gCos)
                    tSgS = thr_cs.partition_S(gSin)

                    cos_frag = cute.make_fragment_like(tCgC)
                    sin_frag = cute.make_fragment_like(tSgS)
                    cute.copy(copy_atom_b16, tCgC, cos_frag)
                    cute.copy(copy_atom_b16, tSgS, sin_frag)

                    sX0 = cute.make_tensor(
                        sX_s.iterator,
                        cute.make_layout((tiler_mn[0], self.rotary_dim), stride=(tiler_mn[1], 1)),
                    )
                    sX1 = cute.make_tensor(
                        sX_s.iterator + self.rotary_dim,
                        cute.make_layout((tiler_mn[0], self.rotary_dim), stride=(tiler_mn[1], 1)),
                    )

                    thr_layout = cute.make_layout((tiler_mn[0], rotary_threads), stride=(rotary_threads, 1))
                    tiled_copy = cute.make_tiled_copy_tv(copy_atom_b16, thr_layout, val_layout_cs)
                    thr_copy = tiled_copy.get_slice(rowid_in_block * rotary_threads + lane)

                    tXsX0 = thr_copy.partition_S(sX0)
                    tXrX0 = cute.make_fragment_like(tXsX0)
                    cute.copy(copy_atom_b16, tXsX0, tXrX0)

                    tXsX1 = thr_copy.partition_S(sX1)
                    tXrX1 = cute.make_fragment_like(tXsX1)
                    cute.copy(copy_atom_b16, tXsX1, tXrX1)

                    x0 = tXrX0.load().to(cute.Float32)
                    x1 = tXrX1.load().to(cute.Float32)
                    cos = cos_frag.load().to(cute.Float32)
                    sin = sin_frag.load().to(cute.Float32)
                    y0 = x0 * cos - x1 * sin
                    y1 = x0 * sin + x1 * cos
                    tXsX0.store(y0.to(tXsX0.element_type))
                    tXsX1.store(y1.to(tXsX1.element_type))

                cute.arch.sync_warp()

                if is_q:
                    if stage_token_idx < num_tokens:
                        copy(tXsX_s, tXgQ_s)
                elif is_qk:
                    if stage_token_idx < num_tokens:
                        copy(tXsX_s, tXgK_s)
                else:
                    if const_expr(mZout is not None):
                        if stage_token_idx < num_tokens:
                            copy(tXsX_s, tXgZ_s)


def _indexer_norm_rope_impl(
    index_q: Tensor,
    index_k: Tensor,
    qnorm_weight: Optional[Tensor],
    knorm_weight: Optional[Tensor],
    knorm_bias: Optional[Tensor],
    cos: Optional[Tensor],
    sin: Optional[Tensor],
    pos_id: Optional[Tensor],
    q_out: Tensor,
    k_out: Tensor,
    index_z: Optional[Tensor],
    z_out: Optional[Tensor],
    head_dim: int,
    num_q_head: int,
    num_k_head: int,
    num_z_head: int,
    rotary_dim: int,
    eps: float = 1e-6,
    q_norm_weight_bias: float = 1.0,
) -> None:
    qkv_dtype = index_q.dtype
    assert qkv_dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported dtype"
    assert index_k.dtype == qkv_dtype, "index_q/index_k dtype mismatch"
    for w in (qnorm_weight, knorm_weight, knorm_bias):
        if w is not None:
            assert w.dtype in [torch.float32, torch.bfloat16, torch.float16], (
                "norm weight/bias must be float32, float16 or bfloat16"
            )
    assert rotary_dim % 4 == 0, "rotary_dim must be divisible by 4"
    assert 2 * rotary_dim <= head_dim, "rotary_dim must fit within head_dim"

    dtype = torch2cute_dtype_map[index_q.dtype]

    current_stream = cuda.CUstream(torch.cuda.current_stream(index_q.device).cuda_stream)
    compile_key = _make_indexer_compile_key(
        head_dim,
        num_q_head,
        num_k_head,
        num_z_head,
        rotary_dim,
        q_norm_weight_bias,
        index_q.dtype,
        qnorm_weight,
        knorm_weight,
        knorm_bias,
    )

    def _compile() -> cute.JitFunction:
        op = FusedIndexerNormRope(
            dtype,
            head_dim * (num_q_head + num_k_head),
            head_dim,
            num_q_head,
            num_k_head,
            rotary_dim,
            num_z_head,
        )
        compile_options = "--enable-tvm-ffi"
        spec_names = ["mQin", "mKin", "mCos", "mSin", "mPosId", "mQout", "mKout"]
        if index_z is not None:
            spec_names += ["mZin", "mZout"]
        return cute_compile_with_spec(
            op,
            index_q,
            index_k,
            qnorm_weight,
            knorm_weight,
            knorm_bias,
            cos,
            sin,
            pos_id,
            q_out,
            k_out,
            index_z,
            z_out,
            head_dim,
            num_q_head,
            num_k_head,
            current_stream,
            eps,
            q_norm_weight_bias,
            tensor_specs_by_name=dynamic_dim0_specs(*spec_names),
            options=compile_options,
        )

    compiled = cache_get_or_create(
        _indexer_norm_rope_impl.compile_cache,
        compile_key,
        _compile,
        op_name="fused_indexer_norm_rope_forward_impl",
        phase="compile",
    )

    compiled(
        index_q,
        index_k,
        qnorm_weight,
        knorm_weight,
        knorm_bias,
        cos,
        sin,
        pos_id,
        q_out,
        k_out,
        index_z,
        z_out,
        head_dim,
        num_q_head,
        num_k_head,
        current_stream,
        eps,
        q_norm_weight_bias,
    )


_indexer_norm_rope_impl.compile_cache = {}


def _make_indexer_compile_key(
    head_dim: int,
    num_q_head: int,
    num_k_head: int,
    num_z_head: int,
    rotary_dim: int,
    q_norm_weight_bias: float,
    qkv_dtype: torch.dtype,
    qnorm_weight: Optional[torch.Tensor],
    knorm_weight: Optional[torch.Tensor],
    knorm_bias: Optional[torch.Tensor],
) -> tuple:
    return (
        head_dim,
        num_q_head,
        num_k_head,
        num_z_head,
        rotary_dim,
        float(q_norm_weight_bias),
        qkv_dtype,
        qnorm_weight.dtype if qnorm_weight is not None else None,
        knorm_weight.dtype if knorm_weight is not None else None,
        knorm_bias.dtype if knorm_bias is not None else None,
    )


def _ref_indexer_norm_rope(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    qnorm_weight: torch.Tensor,
    knorm_weight: torch.Tensor,
    knorm_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_id: torch.Tensor,
    head_dim: int,
    num_q_head: int,
    num_k_head: int,
    rotary_dim: int,
    eps: float = 1e-6,
    q_norm_weight_bias: float = 1.0,
    index_z: torch.Tensor | None = None,
):
    q = index_q.view(*index_q.shape[:-1], num_q_head, head_dim).to(torch.float32)
    k = index_k.view(*index_k.shape[:-1], num_k_head, head_dim).to(torch.float32)

    qw = qnorm_weight.to(torch.float32) + q_norm_weight_bias
    q_rms = torch.sqrt(q.pow(2).mean(dim=-1, keepdim=True) + eps)
    q_scaled = (q / q_rms) * qw.unsqueeze(0)

    kw = knorm_weight.to(torch.float32)
    kb = knorm_bias.to(torch.float32)
    k_mean = k.mean(dim=-1, keepdim=True)
    k_var = (k - k_mean).pow(2).mean(dim=-1, keepdim=True)
    k_scaled = (k - k_mean) / torch.sqrt(k_var + eps) * kw.unsqueeze(0) + kb.unsqueeze(0)

    pos = pos_id.to(torch.long)
    cos_sel = cos[pos].to(torch.float32)
    sin_sel = sin[pos].to(torch.float32)

    def _rope(t, n_head):
        out = t.clone()
        for i in range(n_head):
            real = t[:, i, :rotary_dim]
            imag = t[:, i, rotary_dim : 2 * rotary_dim]
            out[:, i, :rotary_dim] = real * cos_sel - imag * sin_sel
            out[:, i, rotary_dim : 2 * rotary_dim] = real * sin_sel + imag * cos_sel
        return out

    q_rope = _rope(q_scaled, num_q_head)
    k_rope = _rope(k_scaled, num_k_head)

    if index_z is None:
        return (
            q_rope.to(index_q.dtype).view(index_q.shape),
            k_rope.to(index_k.dtype).view(index_k.shape),
        )
    return (
        q_rope.to(index_q.dtype).view(index_q.shape),
        k_rope.to(index_k.dtype).view(index_k.shape),
        index_z.contiguous(),
    )


def _allocate_indexer_outputs(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(index_q),
        torch.empty_like(index_k),
    )


def _make_inputs_indexer() -> ArgsAndKwargs:
    tokens = 32
    head_dim = 128
    num_q_head = 8
    num_k_head = 1
    rotary_dim = 32
    index_q = torch.randn((tokens, num_q_head * head_dim), device="cuda", dtype=torch.float16)
    index_k = torch.randn((tokens, num_k_head * head_dim), device="cuda", dtype=torch.float16)
    qnorm_weight = torch.randn((head_dim,), device="cuda", dtype=torch.float16)
    knorm_weight = torch.randn((head_dim,), device="cuda", dtype=torch.float16)
    knorm_bias = torch.randn((head_dim,), device="cuda", dtype=torch.float16)
    cos = torch.randn((tokens, rotary_dim), device="cuda", dtype=torch.float16)
    sin = torch.randn((tokens, rotary_dim), device="cuda", dtype=torch.float16)
    pos_id = torch.arange(tokens, device="cuda", dtype=torch.int32)
    kwargs = {
        "head_dim": head_dim,
        "num_q_head": num_q_head,
        "num_k_head": num_k_head,
        "rotary_dim": rotary_dim,
        "eps": 1e-6,
        "q_norm_weight_bias": 1.0,
    }
    return [
        index_q,
        index_k,
        qnorm_weight,
        knorm_weight,
        knorm_bias,
        cos,
        sin,
        pos_id,
    ], kwargs


@torch.no_grad()
@benchmark_case(
    tag="small",
    description="32 tokens",
    axis_sizes={"tokens": 32, "seq": 32, "rotary_dim": 32, "head_dim": 128, "num_q_head": 8, "num_k_head": 1, "qhidden": 1024, "khidden": 128},
    inputs={
        "eps": {"kind": "scalar", "value": 1e-6, "scalar_type": "float"},
        "head_dim": {"kind": "scalar", "value": 128, "scalar_type": "int"},
        "num_q_head": {"kind": "scalar", "value": 8, "scalar_type": "int"},
        "num_k_head": {"kind": "scalar", "value": 1, "scalar_type": "int"},
        "rotary_dim": {"kind": "scalar", "value": 32, "scalar_type": "int"},
        "pos_id": {"generator": {"type": "randint", "low": 0, "high": 32}},
        "q_norm_weight_bias": {"kind": "scalar", "value": 1.0, "scalar_type": "float"},
        "index_z": {"kind": "none"},
    },
)
@accuracy_case(
    name="cutedsl_indexer_norm_rope",
    seed=17,
    make_inputs=_make_inputs_indexer,
    supports_cuda_graph=True,
)
def fused_indexer_norm_rope_forward_impl(
    index_q: Float16Tensor[torch.Tensor, "tokens qhidden"],
    index_k: Float16Tensor[torch.Tensor, "tokens khidden"],
    qnorm_weight: Float16Tensor[torch.Tensor, "head_dim"],
    knorm_weight: Float16Tensor[torch.Tensor, "head_dim"],
    knorm_bias: Float16Tensor[torch.Tensor, "head_dim"],
    cos: Float16Tensor[torch.Tensor, "seq rotary_dim"],
    sin: Float16Tensor[torch.Tensor, "seq rotary_dim"],
    pos_id: IntTensor[torch.Tensor, "tokens"],
    head_dim: int,
    num_q_head: int,
    num_k_head: int,
    rotary_dim: int,
    eps: float = 1e-6,
    q_norm_weight_bias: float = 1.0,
    index_z: Optional[torch.Tensor] = None,
):
    '''
    index_q: (tokens, num_q_head * head_dim)  -> RMSNorm per head + partial RoPE
    index_k: (tokens, num_k_head * head_dim)  -> LayerNorm per head + partial RoPE
    index_z: (tokens, num_k_head * head_dim)  -> optional passthrough (no norm/rope),
             may be a strided column-slice of the same fused projection buffer as q/k.
    qnorm_weight/knorm_weight/knorm_bias: (head_dim,)
    cos/sin: (seq, rotary_dim)   where rotary_dim == rope_dim // 2
    pos_id: (tokens,)
    return: (q_out, k_out) or, if index_z is given, (q_out, k_out, z_out).
    '''
    assert index_q.is_cuda and index_k.is_cuda
    assert qnorm_weight.is_cuda and knorm_weight.is_cuda and knorm_bias.is_cuda
    assert cos.is_cuda and sin.is_cuda and pos_id.is_cuda
    q_out = torch.empty(index_q.shape, dtype=index_q.dtype, device=index_q.device)
    k_out = torch.empty(index_k.shape, dtype=index_k.dtype, device=index_k.device)
    if index_z is None:
        z_out = None
        num_z_head = 0
    else:
        z_out = torch.empty(index_z.shape, dtype=index_z.dtype, device=index_z.device)
        num_z_head = int(index_z.shape[-1]) // head_dim
    _indexer_norm_rope_impl(
        index_q,
        index_k,
        qnorm_weight,
        knorm_weight,
        knorm_bias,
        cos,
        sin,
        pos_id,
        q_out,
        k_out,
        index_z,
        z_out,
        head_dim,
        num_q_head,
        num_k_head,
        num_z_head,
        rotary_dim,
        eps,
        q_norm_weight_bias,
    )
    if index_z is None:
        return q_out, k_out
    return q_out, k_out, z_out


fused_indexer_norm_rope_forward_impl.reference_op = _ref_indexer_norm_rope
