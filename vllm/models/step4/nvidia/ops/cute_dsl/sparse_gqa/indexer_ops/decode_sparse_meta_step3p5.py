from __future__ import annotations

import torch
import triton
import triton.language as tl


# Keep the binary-search unroll independent of runtime batch size. This single
# variant supports request counts through 4095 without a runtime compile.
_SEARCH_STEPS = 12

_DECODE_META_ALIGNMENT_DYNAMIC_ARGS = (
    "block_topk_idx_ptr",
    "query_start_loc_ptr",
    "request_indices_ptr",
    "valid_rows_ptr",
    "seq_lens_ptr",
    "req_block_offsets_ptr",
    "block_table_ptr",
    "out_counts_ptr",
    "out_packed_indices_ptr",
)
_DECODE_SUMMARY_TABLE_ALIGNMENT_DYNAMIC_ARGS = (
    "block_table_ptr",
    "seq_lens_ptr",
    "live_token_slots_ptr",
    "out_paged_block_table_ptr",
    "out_summary_valid_ptr",
)


def _validate_search_batch(num_reqs: int) -> None:
    if int(num_reqs) >= (1 << _SEARCH_STEPS):
        raise ValueError(
            "Step3p5 DSA decode metadata supports at most "
            f"{(1 << _SEARCH_STEPS) - 1} requests without runtime compilation, "
            f"got {int(num_reqs)}"
        )


@triton.jit
def _upper_bound_req_idx_scalar(
    query_start_loc_ptr,
    q_idx,
    num_reqs,
    SEARCH_STEPS: tl.constexpr,
):
    lo = tl.full((), 0, tl.int32)
    hi = tl.full((), num_reqs, tl.int32)
    for _ in range(SEARCH_STEPS):
        active = lo < hi
        mid = tl.where(active, (lo + hi) // 2, lo)
        mid_val = tl.load(
            query_start_loc_ptr + mid,
            mask=active,
            other=0,
        ).to(tl.int32)
        go_right = active & (mid_val <= q_idx)
        go_left = active & ~go_right
        lo = tl.where(go_right, mid + 1, lo)
        hi = tl.where(go_left, mid, hi)
    return lo - 1


@triton.jit
def _req_meta_scalar(
    query_start_loc_ptr,
    request_indices_ptr,
    seq_lens_ptr,
    req_block_offsets_ptr,
    q_idx,
    block_size,
    num_reqs,
    HAS_REQUEST_INDICES: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    if HAS_REQUEST_INDICES:
        req_idx = tl.load(request_indices_ptr + q_idx).to(tl.int32)
    else:
        req_idx = _upper_bound_req_idx_scalar(
            query_start_loc_ptr,
            q_idx,
            num_reqs,
            SEARCH_STEPS=SEARCH_STEPS,
        )
    q_start = tl.load(query_start_loc_ptr + req_idx).to(tl.int32)
    q_end = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int32)
    valid_k = tl.load(seq_lens_ptr + req_idx).to(tl.int32)
    block_offset = tl.load(req_block_offsets_ptr + req_idx).to(tl.int32)
    q_len = q_end - q_start
    q_local = q_idx - q_start
    query_pos = valid_k - q_len + q_local
    valid_blocks = (valid_k + block_size - 1) // block_size
    return req_idx, block_offset, valid_blocks, query_pos


@triton.jit(
    do_not_specialize=[
        "stride_blk_q",
        "stride_blk_w",
        "stride_bt_req",
        "stride_bt_page",
        "stride_out_counts_q",
        "stride_out_packed_q",
        "stride_out_packed_w",
        "num_reqs",
        "history_windows",
        "total_windows",
        "block_size",
        "window",
    ],
    do_not_specialize_on_alignment=_DECODE_META_ALIGNMENT_DYNAMIC_ARGS,
)
def _region_block_topk_to_decode_meta_kernel(
    block_topk_idx_ptr,
    query_start_loc_ptr,
    request_indices_ptr,
    valid_rows_ptr,
    seq_lens_ptr,
    req_block_offsets_ptr,
    block_table_ptr,
    out_counts_ptr,
    out_packed_indices_ptr,
    stride_blk_q,
    stride_blk_w,
    stride_bt_req,
    stride_bt_page,
    stride_out_counts_q,
    stride_out_packed_q,
    stride_out_packed_w,
    num_reqs,
    history_windows,
    total_windows,
    block_size,
    window,
    HAS_REQUEST_INDICES: tl.constexpr,
    HAS_VALID_ROWS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    q_idx = tl.program_id(0)
    if HAS_VALID_ROWS:
        valid_rows = tl.load(valid_rows_ptr).to(tl.int32)
        if q_idx >= valid_rows:
            return
    win = tl.arange(0, BLOCK_W)
    mask = win < total_windows

    req_idx, block_offset, valid_blocks, query_pos = _req_meta_scalar(
        query_start_loc_ptr,
        request_indices_ptr,
        seq_lens_ptr,
        req_block_offsets_ptr,
        q_idx,
        block_size,
        num_reqs,
        HAS_REQUEST_INDICES=HAS_REQUEST_INDICES,
        SEARCH_STEPS=SEARCH_STEPS,
    )

    global_blk = tl.load(
        block_topk_idx_ptr + q_idx * stride_blk_q + win * stride_blk_w,
        mask=win < history_windows,
        other=-1,
    ).to(tl.int32)
    local_blk = global_blk - block_offset
    history_valid = (
        (win < history_windows)
        & (global_blk >= 0)
        & (local_blk >= 0)
        & (local_blk < valid_blocks)
    )
    history_blk = tl.where(history_valid, local_blk, 0)
    start_tok = history_blk * block_size
    page_idx = history_blk // 2
    phys_page = tl.load(
        block_table_ptr + req_idx * stride_bt_req + page_idx * stride_bt_page,
        mask=history_valid,
        other=0,
    ).to(tl.int32)
    history_valid = history_valid & (phys_page >= 0)
    phys_blk8 = (phys_page << 1) + (history_blk & 1)
    phys_blk8 = tl.where(history_valid, phys_blk8, 0)

    window_start = tl.where(
        window > 0,
        tl.maximum(query_pos - window + 1, 0),
        query_pos,
    )
    local_begin = tl.minimum(window_start // block_size, valid_blocks)
    local_end = tl.minimum((query_pos // block_size) + 1, valid_blocks)
    local_count = tl.maximum(local_end - local_begin, 0)
    local_slot = win - history_windows
    local_valid = mask & (win >= history_windows) & (local_slot < local_count)
    local_blk = local_begin + local_slot
    local_blk_safe = tl.where(local_valid, local_blk, 0)
    local_start_tok = local_blk_safe * block_size
    local_page_idx = local_blk_safe // 2
    local_phys_page = tl.load(
        block_table_ptr + req_idx * stride_bt_req + local_page_idx * stride_bt_page,
        mask=local_valid,
        other=0,
    ).to(tl.int32)
    local_valid = local_valid & (local_phys_page >= 0)
    local_phys_blk8 = (local_phys_page << 1) + (local_blk_safe & 1)
    start_tok = tl.where(local_valid, local_start_tok, start_tok)
    phys_blk8 = tl.where(local_valid, local_phys_blk8, phys_blk8)

    valid = history_valid | local_valid
    invalid_packed = tl.full((BLOCK_W,), 0x7FFF_FFFF_FFFF_FFFF, tl.int64)
    packed_sort = start_tok.to(tl.int64) | (phys_blk8.to(tl.int64) << 32)
    packed_sort = tl.where(valid, packed_sort, invalid_packed)
    packed_sort = tl.sort(packed_sort, descending=False)

    sorted_valid = packed_sort != invalid_packed
    packed = tl.where(sorted_valid, packed_sort, tl.zeros((BLOCK_W,), dtype=tl.int64))
    tl.store(
        out_packed_indices_ptr
        + q_idx * stride_out_packed_q
        + win * stride_out_packed_w,
        packed,
        mask=mask,
    )
    tl.store(
        out_counts_ptr + q_idx * stride_out_counts_q,
        tl.sum(sorted_valid.to(tl.int32), axis=0),
    )


@triton.jit(
    do_not_specialize=[
        "stride_blk_q",
        "stride_blk_w",
        "stride_bt_req",
        "stride_bt_page",
        "stride_out_counts_q",
        "stride_out_packed_q",
        "stride_out_packed_w",
        "num_reqs",
        "total_windows",
    ],
    do_not_specialize_on_alignment=_DECODE_META_ALIGNMENT_DYNAMIC_ARGS,
)
def _region_block_topk_to_decode_meta_512_plus_one_kernel(
    block_topk_idx_ptr,
    query_start_loc_ptr,
    request_indices_ptr,
    valid_rows_ptr,
    seq_lens_ptr,
    req_block_offsets_ptr,
    block_table_ptr,
    out_counts_ptr,
    out_packed_indices_ptr,
    stride_blk_q,
    stride_blk_w,
    stride_bt_req,
    stride_bt_page,
    stride_out_counts_q,
    stride_out_packed_q,
    stride_out_packed_w,
    num_reqs,
    total_windows,
    SORT_OUTPUT: tl.constexpr,
    HAS_REQUEST_INDICES: tl.constexpr,
    HAS_VALID_ROWS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    q_idx = tl.program_id(0)
    if HAS_VALID_ROWS:
        valid_rows = tl.load(valid_rows_ptr).to(tl.int32)
        if q_idx >= valid_rows:
            return
    win_hist = tl.arange(0, 512)
    history_windows = total_windows - 1

    req_idx, block_offset, valid_blocks, query_pos = _req_meta_scalar(
        query_start_loc_ptr,
        request_indices_ptr,
        seq_lens_ptr,
        req_block_offsets_ptr,
        q_idx,
        8,
        num_reqs,
        HAS_REQUEST_INDICES=HAS_REQUEST_INDICES,
        SEARCH_STEPS=SEARCH_STEPS,
    )

    global_blk = tl.load(
        block_topk_idx_ptr + q_idx * stride_blk_q + win_hist * stride_blk_w,
        mask=win_hist < history_windows,
        other=-1,
    ).to(tl.int32)
    local_blk = global_blk - block_offset
    history_valid = (global_blk >= 0) & (local_blk >= 0) & (local_blk < valid_blocks)
    history_blk = tl.where(history_valid, local_blk, 0)
    start_tok = history_blk * 8
    page_idx = history_blk // 2
    phys_page = tl.load(
        block_table_ptr + req_idx * stride_bt_req + page_idx * stride_bt_page,
        mask=history_valid,
        other=0,
    ).to(tl.int32)
    history_valid = history_valid & (phys_page >= 0)
    phys_blk8 = (phys_page << 1) + (history_blk & 1)
    phys_blk8 = tl.where(history_valid, phys_blk8, 0)

    invalid_packed = tl.full((512,), 0x7FFF_FFFF_FFFF_FFFF, tl.int64)
    history_packed = start_tok.to(tl.int64) | (phys_blk8.to(tl.int64) << 32)
    history_packed = tl.where(history_valid, history_packed, invalid_packed)
    history_count = tl.sum((history_packed != invalid_packed).to(tl.int32), axis=0)
    if SORT_OUTPUT:
        history_packed = tl.sort(history_packed, descending=False)

    local_blk = query_pos // 8
    local_valid = (local_blk >= 0) & (local_blk < valid_blocks)
    local_blk_safe = tl.where(local_valid, local_blk, 0)
    local_page_idx = local_blk_safe // 2
    local_phys_page = tl.load(
        block_table_ptr + req_idx * stride_bt_req + local_page_idx * stride_bt_page,
        mask=local_valid,
        other=0,
    ).to(tl.int32)
    local_valid = local_valid & (local_phys_page >= 0)
    local_phys_blk8 = (local_phys_page << 1) + (local_blk_safe & 1)
    local_packed = tl.where(
        local_valid,
        (local_blk_safe * 8).to(tl.int64) | (local_phys_blk8.to(tl.int64) << 32),
        tl.zeros((), dtype=tl.int64),
    )

    tl.store(
        out_packed_indices_ptr
        + q_idx * stride_out_packed_q
        + win_hist * stride_out_packed_w,
        tl.zeros((512,), dtype=tl.int64),
        mask=(win_hist < total_windows)
        & (~local_valid | (win_hist != history_count)),
    )
    tl.store(
        out_packed_indices_ptr
        + q_idx * stride_out_packed_q
        + 512 * stride_out_packed_w,
        tl.zeros((), dtype=tl.int64),
        mask=(total_windows > 512) & (~local_valid | (history_count != 512)),
    )
    if SORT_OUTPUT:
        tl.store(
            out_packed_indices_ptr
            + q_idx * stride_out_packed_q
            + win_hist * stride_out_packed_w,
            history_packed,
            mask=win_hist < history_count,
        )
    else:
        history_rank = tl.cumsum(history_valid.to(tl.int32), axis=0) - 1
        tl.store(
            out_packed_indices_ptr
            + q_idx * stride_out_packed_q
            + history_rank * stride_out_packed_w,
            history_packed,
            mask=history_valid,
        )
    # Keep the current region as the terminal +1 slot so downstream decode can
    # treat the first `history_count` entries as historical windows.
    tl.store(
        out_packed_indices_ptr
        + q_idx * stride_out_packed_q
        + history_count * stride_out_packed_w,
        local_packed,
        mask=local_valid,
    )
    tl.store(
        out_counts_ptr + q_idx * stride_out_counts_q,
        history_count + local_valid.to(tl.int32),
    )


@triton.jit(
    do_not_specialize=[
        "stride_bt_req",
        "stride_bt_page",
        "stride_count_page",
        "stride_count_region",
        "stride_out_paged_req",
        "stride_out_paged_page",
        "stride_out_valid_req",
        "stride_out_valid_region",
        "num_pages",
        "block_table_cols",
        "required_pages",
        "num_regions",
        "summaries_per_page",
    ],
    do_not_specialize_on_alignment=_DECODE_SUMMARY_TABLE_ALIGNMENT_DYNAMIC_ARGS,
)
def _decode_paged_summary_block_table_and_valid_kernel(
    block_table_ptr,
    seq_lens_ptr,
    live_token_slots_ptr,
    out_paged_block_table_ptr,
    out_summary_valid_ptr,
    stride_bt_req,
    stride_bt_page,
    stride_out_paged_req,
    stride_out_paged_page,
    stride_out_valid_req,
    stride_out_valid_region,
    num_pages,
    block_table_cols,
    required_pages,
    num_regions,
    summaries_per_page,
    region_block_size: tl.constexpr,
    HAS_LIVE_TOKEN_SLOTS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    req = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask_paged = cols < required_pages
    col_mask_region = cols < num_regions

    live = tl.full((), True, tl.int1)
    if HAS_LIVE_TOKEN_SLOTS:
        live_slot = tl.load(live_token_slots_ptr + req)
        live = live_slot >= 0
    seq_len = tl.load(seq_lens_ptr + req).to(tl.int32)
    seq_len = tl.where(live, seq_len, 0)

    safe_page_col = tl.minimum(cols, block_table_cols - 1)
    page_valid = col_mask_paged & (cols < block_table_cols)
    physical_page = tl.load(
        block_table_ptr + req * stride_bt_req + safe_page_col * stride_bt_page,
        mask=page_valid,
        other=-1,
    ).to(tl.int32)
    physical_page_valid = (
        page_valid & live & (physical_page >= 0) & (physical_page < num_pages)
    )
    tl.store(
        out_paged_block_table_ptr + req * stride_out_paged_req + cols * stride_out_paged_page,
        tl.where(physical_page_valid, physical_page, -1),
        mask=col_mask_paged,
    )

    region_ids = cols
    logical_pages = region_ids // summaries_per_page
    safe_logical_pages = tl.minimum(logical_pages, block_table_cols - 1)
    summary_page_valid = col_mask_region & (logical_pages < block_table_cols)
    summary_physical_pages = tl.load(
        block_table_ptr + req * stride_bt_req + safe_logical_pages * stride_bt_page,
        mask=summary_page_valid,
        other=-1,
    ).to(tl.int32)
    summary_page_valid = (
        summary_page_valid
        & (summary_physical_pages >= 0)
        & (summary_physical_pages < num_pages)
    )
    # The CSA stage writes mean_cache only after a region has all of its
    # region_block_size tokens.  Validity therefore comes from the logical
    # sequence boundary and the physical block table; it must not depend on a
    # page-persistent count/validity sidecar.
    region_ends = (region_ids + 1) * region_block_size
    summary_valid = (
        col_mask_region
        & live
        & summary_page_valid
        & (region_ends <= seq_len)
    )
    tl.store(
        out_summary_valid_ptr + req * stride_out_valid_req + cols * stride_out_valid_region,
        summary_valid,
        mask=col_mask_region,
    )


def _as_i32_1d(
    name: str,
    tensor: torch.Tensor,
    *,
    device: torch.device,
    length: int | None = None,
) -> torch.Tensor:
    if tensor.device != device:
        raise ValueError(f"{name} must be on the same device as block_topk_idx")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={tuple(tensor.shape)}")
    if length is not None and int(tensor.numel()) != int(length):
        raise ValueError(
            f"{name} length mismatch: got {int(tensor.numel())}, "
            f"expected {int(length)}"
        )
    if tensor.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} dtype must be int32/int64, got {tensor.dtype}")
    return tensor.to(dtype=torch.int32).contiguous()


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


def _validate_output(
    name: str,
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if tensor.device != device:
        raise ValueError(f"{name} must be on the same device as block_topk_idx")
    if tensor.dtype != dtype:
        raise ValueError(
            f"{name} dtype mismatch: got {tensor.dtype}, expected {dtype}"
        )
    if tuple(int(v) for v in tensor.shape) != shape:
        raise ValueError(
            f"{name} shape mismatch: got {tuple(int(v) for v in tensor.shape)}, "
            f"expected {shape}"
        )
    return tensor


def convert_region_block_topk_to_sparse_meta_step3p5(
    block_topk_idx: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    req_block_offsets: torch.Tensor,
    block_table: torch.Tensor,
    *,
    block_size: int = 8,
    window: int = 0,
    block_counts_out: torch.Tensor | None = None,
    block_packed_indices_out: torch.Tensor | None = None,
    valid_seq_q: int | None = None,
    request_indices: torch.Tensor | None = None,
    sort_output: bool = True,
    valid_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert compact region top-k ids to sparse GQA decode metadata.

    ``block_topk_idx`` contains global compact region ids produced by the
    StepTron indexer selector. The output packed format is
    ``logical_start_token | physical_region << 32``, which is consumed directly
    by ``token_wise_flash_attn_decode_sm90_gqa_func``.
    """
    if int(block_size) != 8:
        raise ValueError(
            "Step3p5 sparse meta conversion requires block_size == 8, "
            f"got {block_size}"
        )
    if int(window) < 0:
        raise ValueError(f"window must be >= 0, got {window}")
    if block_topk_idx.device.type != "cuda":
        raise ValueError("block_topk_idx must be a CUDA tensor")
    device = block_topk_idx.device

    if block_topk_idx.ndim == 3:
        if int(block_topk_idx.shape[1]) != 1:
            raise ValueError(
                "block_topk_idx must have shape [T, W] or [T, 1, W], "
                f"got {tuple(block_topk_idx.shape)}"
            )
        block_idx_view = block_topk_idx[:, 0, :]
    elif block_topk_idx.ndim == 2:
        block_idx_view = block_topk_idx
    else:
        raise ValueError(
            "block_topk_idx must have shape [T, W] or [T, 1, W], "
            f"got {tuple(block_topk_idx.shape)}"
        )
    block_idx_view = block_idx_view.to(dtype=torch.int32).contiguous()

    capacity_q = int(block_idx_view.shape[0])
    if valid_rows is not None and valid_seq_q is not None:
        raise ValueError(
            "valid_rows and valid_seq_q are mutually exclusive; use the "
            "device-resident valid_rows for CUDA Graph replay"
        )
    use_valid_rows = valid_rows is not None
    total_q = capacity_q if use_valid_rows or valid_seq_q is None else int(valid_seq_q)
    if total_q < 0 or total_q > capacity_q:
        raise ValueError(
            "valid_seq_q must be within block_topk_idx row capacity, got "
            f"valid_seq_q={total_q}, capacity_q={capacity_q}"
        )
    block_idx_view = block_idx_view[:total_q]
    history_windows = int(block_idx_view.shape[1])
    total_windows = history_windows + triton.cdiv(int(window), int(block_size)) + 1
    if total_windows <= 0 or total_windows > 1024:
        raise ValueError(
            "Step3p5 sparse meta conversion requires 1 <= total_windows <= 1024, "
            f"got {total_windows}"
        )

    num_reqs = int(query_start_loc.numel()) - 1
    query_start_loc = _as_i32_1d(
        "query_start_loc", query_start_loc, device=device, length=num_reqs + 1)
    if request_indices is None:
        request_indices_arg = query_start_loc
        has_request_indices = False
    else:
        request_indices_arg = _as_i32_1d(
            "request_indices", request_indices, device=device, length=total_q)
        has_request_indices = True
    if valid_rows is None:
        valid_rows_arg = query_start_loc
    else:
        if valid_rows.device != device:
            raise ValueError("valid_rows must be on the same CUDA device")
        if valid_rows.dtype != torch.int32:
            raise ValueError(
                f"valid_rows must be torch.int32, got {valid_rows.dtype}"
            )
        if valid_rows.ndim != 1 or int(valid_rows.numel()) != 1:
            raise ValueError(
                "valid_rows must be a contiguous CUDA tensor with shape [1], "
                f"got shape={tuple(valid_rows.shape)}"
            )
        if not valid_rows.is_contiguous():
            raise ValueError("valid_rows must be contiguous")
        valid_rows_arg = valid_rows
    seq_lens = _as_i32_1d("seq_lens", seq_lens, device=device, length=num_reqs)
    req_block_offsets = _as_i32_1d(
        "req_block_offsets", req_block_offsets, device=device, length=num_reqs)
    if block_table.device != device:
        raise ValueError("block_table must be on the same device as block_topk_idx")
    if block_table.ndim != 2 or int(block_table.shape[0]) != num_reqs:
        raise ValueError(
            "block_table must have shape [num_reqs, pages], got "
            f"{tuple(block_table.shape)} for num_reqs={num_reqs}"
        )
    if block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"block_table dtype must be int32/int64, got {block_table.dtype}"
        )
    block_table = block_table.to(dtype=torch.int32).contiguous()

    counts_shape = (total_q, 1)
    packed_shape = (total_q, 1, total_windows)
    if block_counts_out is None:
        block_counts = torch.empty(
            counts_shape, dtype=torch.int32, device=device)
        block_counts_view = block_counts[:, 0]
    else:
        if tuple(int(v) for v in block_counts_out.shape) == (total_q,):
            block_counts = _validate_output(
                "block_counts_out", block_counts_out, device=device,
                dtype=torch.int32, shape=(total_q,))
            block_counts_view = block_counts
        else:
            block_counts = _validate_output(
                "block_counts_out", block_counts_out, device=device,
                dtype=torch.int32, shape=counts_shape)
            block_counts_view = block_counts[:, 0]
    if block_packed_indices_out is None:
        block_packed_indices = torch.empty(
            packed_shape, dtype=torch.int64, device=device)
        block_packed_view = block_packed_indices[:, 0, :]
    else:
        if tuple(int(v) for v in block_packed_indices_out.shape) == (
                total_q, total_windows):
            block_packed_indices = _validate_output(
                "block_packed_indices_out", block_packed_indices_out,
                device=device, dtype=torch.int64,
                shape=(total_q, total_windows))
            block_packed_view = block_packed_indices
        else:
            block_packed_indices = _validate_output(
                "block_packed_indices_out", block_packed_indices_out,
                device=device, dtype=torch.int64, shape=packed_shape)
            block_packed_view = block_packed_indices[:, 0, :]
    if total_q == 0:
        return block_counts, block_packed_indices
    _validate_search_batch(num_reqs)
    if (
        int(block_size) == 8
        and int(window) == 0
        and int(history_windows) <= 512
        and int(total_windows) <= 513
    ):
        _region_block_topk_to_decode_meta_512_plus_one_kernel[(total_q,)](
            block_idx_view,
            query_start_loc,
            request_indices_arg,
            valid_rows_arg,
            seq_lens,
            req_block_offsets,
            block_table,
            block_counts_view,
            block_packed_view,
            block_idx_view.stride(0),
            block_idx_view.stride(1),
            block_table.stride(0),
            block_table.stride(1),
            block_counts_view.stride(0),
            block_packed_view.stride(0),
            block_packed_view.stride(1),
            int(num_reqs),
            int(total_windows),
            SORT_OUTPUT=bool(sort_output),
            HAS_REQUEST_INDICES=has_request_indices,
            HAS_VALID_ROWS=use_valid_rows,
            SEARCH_STEPS=_SEARCH_STEPS,
            num_warps=4,
        )
        return block_counts, block_packed_indices
    block_w = 1 << (int(total_windows) - 1).bit_length()
    num_warps = 4 if block_w >= 256 else 2
    _region_block_topk_to_decode_meta_kernel[(total_q,)](
        block_idx_view,
        query_start_loc,
        request_indices_arg,
        valid_rows_arg,
        seq_lens,
        req_block_offsets,
        block_table,
        block_counts_view,
        block_packed_view,
        block_idx_view.stride(0),
        block_idx_view.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        block_counts_view.stride(0),
        block_packed_view.stride(0),
        block_packed_view.stride(1),
        int(num_reqs),
        int(history_windows),
        int(total_windows),
        int(block_size),
        int(window),
        HAS_REQUEST_INDICES=has_request_indices,
        HAS_VALID_ROWS=use_valid_rows,
        SEARCH_STEPS=_SEARCH_STEPS,
        BLOCK_W=block_w,
        num_warps=num_warps,
    )
    return block_counts, block_packed_indices


def build_decode_paged_summary_block_table_and_valid_step3p5(
    *,
    summary_cache,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    live_token_slots: torch.Tensor | None = None,
    num_regions: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if device.type != "cuda":
        raise ValueError(
            "build_decode_paged_summary_block_table_and_valid_step3p5 requires "
            "a CUDA device"
        )
    if int(getattr(summary_cache, "num_kv_heads", 1)) != 1:
        raise ValueError(
            "build_decode_paged_summary_block_table_and_valid_step3p5 "
            f"requires num_kv_heads == 1, got {int(summary_cache.num_kv_heads)}"
        )
    block_table = _validate_i32_cuda("block_table", block_table, 2)
    if int(block_table.shape[0]) != int(seq_lens.shape[0]):
        raise ValueError(
            "block_table and seq_lens must cover the same number of requests, "
            f"got block_table={tuple(block_table.shape)}, seq_lens={tuple(seq_lens.shape)}"
        )
    if int(num_regions) < 0:
        raise ValueError(f"num_regions must be >= 0, got {num_regions}")

    num_reqs = int(block_table.shape[0])
    num_regions = int(num_regions)
    summaries_per_page = int(summary_cache.summaries_per_page)
    required_pages = (num_regions + summaries_per_page - 1) // summaries_per_page
    seq_lens = _as_i32_1d("seq_lens", seq_lens, device=device, length=num_reqs)
    block_table = block_table.to(dtype=torch.int32).contiguous()
    has_live_token_slots = live_token_slots is not None
    if live_token_slots is None:
        live_token_slots_arg = seq_lens
    else:
        if live_token_slots.device != device:
            raise ValueError("live_token_slots must be on the same device")
        if live_token_slots.ndim != 1:
            raise ValueError(
                "live_token_slots must be 1D, "
                f"got shape={tuple(live_token_slots.shape)}")
        if live_token_slots.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                "live_token_slots dtype must be int32/int64, "
                f"got {live_token_slots.dtype}")
        if int(live_token_slots.numel()) < num_reqs:
            raise ValueError(
                "live_token_slots must cover every request, "
                f"got {int(live_token_slots.numel())}, expected at least {num_reqs}")
        live_token_slots_arg = live_token_slots.contiguous()

    paged_block_table = torch.empty(
        (num_reqs, required_pages), dtype=torch.int32, device=device)
    summary_valid = torch.empty(
        (num_reqs, num_regions), dtype=torch.bool, device=device)
    if num_reqs == 0 or num_regions == 0:
        return paged_block_table, summary_valid
    if int(block_table.shape[1]) <= 0:
        paged_block_table.fill_(-1)
        summary_valid.zero_()
        return paged_block_table, summary_valid

    block_n = 256
    grid = (num_reqs, triton.cdiv(max(required_pages, num_regions), block_n))
    _decode_paged_summary_block_table_and_valid_kernel[
        grid
    ](
        block_table,
        seq_lens,
        live_token_slots_arg,
        paged_block_table,
        summary_valid,
        block_table.stride(0),
        block_table.stride(1),
        paged_block_table.stride(0),
        paged_block_table.stride(1),
        summary_valid.stride(0),
        summary_valid.stride(1),
        int(summary_cache.num_pages),
        int(block_table.shape[1]),
        int(required_pages),
        int(num_regions),
        summaries_per_page,
        region_block_size=int(summary_cache.region_block_size),
        HAS_LIVE_TOKEN_SLOTS=bool(has_live_token_slots),
        BLOCK_N=block_n,
        num_warps=4,
    )
    return paged_block_table, summary_valid


__all__ = [
    "build_decode_paged_summary_block_table_and_valid_step3p5",
    "convert_region_block_topk_to_sparse_meta_step3p5",
]
