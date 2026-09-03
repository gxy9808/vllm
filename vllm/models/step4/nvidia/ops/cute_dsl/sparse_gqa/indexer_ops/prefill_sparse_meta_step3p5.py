from __future__ import annotations

import torch
import triton
import triton.language as tl


_REGION_VALID_SHIFT = 24
_KERNEL_PAGE_SIZE = 16
_PREFILL_SPARSE_META_ALIGNMENT_DYNAMIC_ARGS = (
    "raw_topk_ptr",
    "region_counts_ptr",
    "q_positions_ptr",
    "block_table_ptr",
    "out_ptr",
)
_PREFILL_UNION_META_ALIGNMENT_DYNAMIC_ARGS = (
    "raw_topk_ptr",
    "region_counts_ptr",
    "block_table_ptr",
    "out_phys_ptr",
    "out_logical_ptr",
)


@triton.jit(
    do_not_specialize=["seq_q", "block_table_pages"],
    do_not_specialize_on_alignment=_PREFILL_SPARSE_META_ALIGNMENT_DYNAMIC_ARGS,
)
def _prefill_region_topk_to_sparse_meta_kernel(
    raw_topk_ptr,
    region_counts_ptr,
    q_positions_ptr,
    block_table_ptr,
    out_ptr,
    stride_raw_row: tl.constexpr,
    stride_raw_col: tl.constexpr,
    stride_bt_page: tl.constexpr,
    stride_out_row: tl.constexpr,
    stride_out_col: tl.constexpr,
    topk: tl.constexpr,
    seq_q,
    region_block_size: tl.constexpr,
    block_table_pages,
    BLOCK_K: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    in_topk = cols < topk
    count = tl.load(region_counts_ptr + row).to(tl.int32)
    valid = in_topk & (cols < count)

    raw = tl.load(
        raw_topk_ptr + row * stride_raw_row + cols * stride_raw_col,
        mask=in_topk,
        other=-1,
    ).to(tl.int32)
    sentinel = tl.full((BLOCK_K,), 0x7FFFFFFF, tl.int32)
    sort_keys = tl.where(valid & (raw >= 0), raw, sentinel)
    sort_keys = tl.sort(sort_keys, descending=False)

    q_idx = row - (row // seq_q) * seq_q
    q_pos = tl.load(q_positions_ptr + q_idx).to(tl.int32)
    summaries_per_kernel_page: tl.constexpr = 16 // region_block_size
    page_col = sort_keys // summaries_per_kernel_page
    page_slot = sort_keys - page_col * summaries_per_kernel_page
    page_col_ok = page_col < block_table_pages
    safe_page_col = tl.minimum(page_col, block_table_pages - 1)
    phys_page = tl.load(
        block_table_ptr + safe_page_col * stride_bt_page,
        mask=valid & page_col_ok & (sort_keys != sentinel),
        other=0,
    ).to(tl.int32)
    phys_region = phys_page * summaries_per_kernel_page + page_slot
    valid_tokens = q_pos + 1 - sort_keys * region_block_size
    valid_tokens = tl.minimum(tl.maximum(valid_tokens, 0), region_block_size)
    packed = phys_region | (valid_tokens << 24)
    packed = tl.where(valid & page_col_ok & (sort_keys != sentinel), packed, -1)
    tl.store(
        out_ptr + row * stride_out_row + cols * stride_out_col,
        packed,
        mask=in_topk,
    )


def _validate_i32_cuda(name: str, tensor: torch.Tensor,
                       ndim: int) -> torch.Tensor:
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must be torch.int32, got {tensor.dtype}")
    if tensor.ndim != ndim:
        raise ValueError(
            f"{name} must be {ndim}D, got shape={tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous, got stride={tuple(tensor.stride())}")
    return tensor


def convert_prefill_region_topk_to_sparse_meta_step3p5(
    raw_topk_idx: torch.Tensor,
    region_counts: torch.Tensor,
    q_positions: torch.Tensor,
    block_table: torch.Tensor,
    *,
    seq_q: int,
    active_rows: int | None = None,
    region_block_size: int = 8,
    out_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sort logical prefill top-k ids and pack them for sparse GQA attention."""
    raw_topk_idx = _validate_i32_cuda("raw_topk_idx", raw_topk_idx, 2)
    region_counts = _validate_i32_cuda("region_counts", region_counts, 1)
    q_positions = _validate_i32_cuda("q_positions", q_positions, 1)
    block_table = _validate_i32_cuda("block_table", block_table, 2)
    if int(block_table.shape[0]) != 1:
        raise ValueError(
            "prefill sparse meta converter currently expects one request row, "
            f"got block_table shape={tuple(block_table.shape)}")
    if (int(region_block_size) <= 0
            or _KERNEL_PAGE_SIZE % int(region_block_size) != 0):
        raise ValueError(
            "region_block_size must be a positive divisor of 16, got "
            f"{region_block_size}")
    seq_q_i = int(seq_q)
    if seq_q_i <= 0:
        raise ValueError(f"seq_q must be positive, got {seq_q_i}")
    active_rows_i = (int(raw_topk_idx.shape[0])
                     if active_rows is None else int(active_rows))
    if active_rows_i <= 0 or active_rows_i > int(raw_topk_idx.shape[0]):
        raise ValueError(
            "active_rows must be within raw_topk_idx row capacity, got "
            f"active_rows={active_rows_i}, rows={int(raw_topk_idx.shape[0])}")
    if int(region_counts.shape[0]) < active_rows_i:
        raise ValueError(
            "region_counts must cover active rows, got "
            f"counts={tuple(region_counts.shape)}, active_rows={active_rows_i}")
    if int(q_positions.shape[0]) < seq_q_i:
        raise ValueError(
            "q_positions must cover seq_q, got "
            f"q_positions={tuple(q_positions.shape)}, seq_q={seq_q_i}")
    topk = int(raw_topk_idx.shape[1])
    if topk <= 0:
        raise ValueError("raw_topk_idx must have positive topk dimension")
    if topk > 1024:
        raise ValueError(f"topk must be <= 1024 for tl.sort, got {topk}")
    if out_idx is None:
        out_idx = torch.empty_like(raw_topk_idx)
    else:
        out_idx = _validate_i32_cuda("out_idx", out_idx, 2)
        if tuple(int(v) for v in out_idx.shape) != tuple(
                int(v) for v in raw_topk_idx.shape):
            raise ValueError(
                "out_idx shape must match raw_topk_idx, got "
                f"out={tuple(out_idx.shape)}, raw={tuple(raw_topk_idx.shape)}")
    block_k = 1 << (topk - 1).bit_length()
    _prefill_region_topk_to_sparse_meta_kernel[(active_rows_i,)](
        raw_topk_idx,
        region_counts,
        q_positions,
        block_table,
        out_idx,
        raw_topk_idx.stride(0),
        raw_topk_idx.stride(1),
        block_table.stride(1),
        out_idx.stride(0),
        out_idx.stride(1),
        int(topk),
        int(seq_q_i),
        int(region_block_size),
        int(block_table.shape[1]),
        BLOCK_K=int(block_k),
        num_warps=8 if block_k >= 256 else 4,
    )
    return out_idx


@triton.jit(
    do_not_specialize=[
        "active_rows",
        "block_table_pages",
    ],
    do_not_specialize_on_alignment=_PREFILL_UNION_META_ALIGNMENT_DYNAMIC_ARGS,
)
def _prefill_region_topk_to_union_meta_kernel(
    raw_topk_ptr,
    region_counts_ptr,
    block_table_ptr,
    out_phys_ptr,
    out_logical_ptr,
    stride_raw_row: tl.constexpr,
    stride_raw_col: tl.constexpr,
    stride_bt_page: tl.constexpr,
    stride_out_row: tl.constexpr,
    stride_out_col: tl.constexpr,
    topk: tl.constexpr,
    region_block_size: tl.constexpr,
    active_rows,
    block_table_pages,
    BLOCK_K: tl.constexpr,
) -> None:
    """Map selected logical regions to union metadata in one GPU launch.

    The union attention kernel needs the logical region start and the physical
    region id separately.  The ordinary prefill converter packs the physical
    id together with a tail-token count, which is intentionally not consumed
    by grouped-union attention.  Keep the sort and page-table lookup fused so
    the vLLM adapter does not add unpack, gather, or per-element Torch work.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    in_topk = cols < topk
    count = tl.load(region_counts_ptr + row).to(tl.int32)
    valid = in_topk & (cols < count)
    raw = tl.load(
        raw_topk_ptr + row * stride_raw_row + cols * stride_raw_col,
        mask=in_topk,
        other=-1,
    ).to(tl.int32)
    sentinel = tl.full((BLOCK_K,), 0x7FFFFFFF, tl.int32)
    logical_region = tl.sort(
        tl.where(valid & (raw >= 0), raw, sentinel), descending=False
    )
    page_span: tl.constexpr = 16 // region_block_size
    page_col = logical_region // page_span
    page_slot = logical_region - page_col * page_span
    page_col_ok = page_col < block_table_pages
    safe_page_col = tl.minimum(page_col, block_table_pages - 1)
    phys_page = tl.load(
        block_table_ptr + safe_page_col * stride_bt_page,
        mask=valid & page_col_ok & (logical_region != sentinel),
        other=0,
    ).to(tl.int32)
    phys_region = phys_page * page_span + page_slot
    keep = valid & page_col_ok & (logical_region != sentinel) & (phys_page >= 0)
    logical_start = logical_region * region_block_size
    tl.store(
        out_phys_ptr + row * stride_out_row + cols * stride_out_col,
        tl.where(keep, phys_region, -1),
        mask=in_topk,
    )
    tl.store(
        out_logical_ptr + row * stride_out_row + cols * stride_out_col,
        tl.where(keep, logical_start, -1),
        mask=in_topk,
    )


def convert_prefill_region_topk_to_union_meta_step3p5(
    raw_topk_idx: torch.Tensor,
    region_counts: torch.Tensor,
    block_table: torch.Tensor,
    *,
    region_block_size: int = 8,
    active_rows: int | None = None,
    out_phys: torch.Tensor | None = None,
    out_logical: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce separate physical/logical region ids for exact union attention."""
    raw_topk_idx = _validate_i32_cuda("raw_topk_idx", raw_topk_idx, 2)
    region_counts = _validate_i32_cuda("region_counts", region_counts, 1)
    block_table = _validate_i32_cuda("block_table", block_table, 2)
    if int(block_table.shape[0]) != 1:
        raise ValueError(
            "prefill union sparse meta converter expects one request row, "
            f"got block_table shape={tuple(block_table.shape)}"
        )
    if int(region_block_size) <= 0 or 16 % int(region_block_size) != 0:
        raise ValueError(
            "region_block_size must be a positive divisor of 16, got "
            f"{int(region_block_size)}"
        )
    if int(region_block_size) != 8:
        raise ValueError(
            "grouped-union GQA metadata currently requires region_block_size=8"
        )
    rows = int(raw_topk_idx.shape[0])
    active_rows_i = rows if active_rows is None else int(active_rows)
    if active_rows_i <= 0 or active_rows_i > rows:
        raise ValueError(
            f"active_rows must be within raw_topk_idx rows, got {active_rows_i}"
        )
    if int(region_counts.shape[0]) < active_rows_i:
        raise ValueError("region_counts does not cover active_rows")
    topk = int(raw_topk_idx.shape[1])
    if topk <= 0 or topk > 1024:
        raise ValueError(f"topk must be in [1, 1024], got {topk}")
    shape = (rows, topk)
    for name, out in (("out_phys", out_phys), ("out_logical", out_logical)):
        if out is None:
            continue
        _validate_i32_cuda(name, out, 2)
        if tuple(int(v) for v in out.shape) != shape:
            raise ValueError(f"{name} shape must be {shape}, got {tuple(out.shape)}")
    if out_phys is None:
        out_phys = torch.empty_like(raw_topk_idx)
    if out_logical is None:
        out_logical = torch.empty_like(raw_topk_idx)
    block_k = 1 << (topk - 1).bit_length()
    _prefill_region_topk_to_union_meta_kernel[(active_rows_i,)](
        raw_topk_idx,
        region_counts,
        block_table,
        out_phys,
        out_logical,
        raw_topk_idx.stride(0),
        raw_topk_idx.stride(1),
        block_table.stride(1),
        out_phys.stride(0),
        out_phys.stride(1),
        int(topk),
        int(region_block_size),
        active_rows_i,
        int(block_table.shape[1]),
        BLOCK_K=int(block_k),
        num_warps=8 if block_k >= 256 else 4,
    )
    return out_phys, out_logical


__all__ = [
    "convert_prefill_region_topk_to_sparse_meta_step3p5",
    "convert_prefill_region_topk_to_union_meta_step3p5",
]
