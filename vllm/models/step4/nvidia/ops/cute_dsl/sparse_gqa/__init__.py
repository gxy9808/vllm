# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy Step4 GQA token-sparse attention and indexer kernel surface."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

__all__ = [
    "batch_decode_logits_wgmma_n",
    "batch_decode_weighted_relu_logits_sum_paged_sm90_steptron_gqa",
    "build_grouped_union_sparse_work_queue_gqa",
    "csa_compact_decode_stage_flush_with_slots_prevalidated_sm90_gqa",
    "csa_compact_prefill_update_with_slots_prevalidated_sm90_gqa",
    "cutedsl_topk_selector_decode_meta_sm90_gqa",
    "decode_weighted_relu_logits_sum_paged_mean_warp_sm90_steptron_gqa",
    "merge_dynamic_split_nat_lse_states_sm90_gqa",
    "merge_variable_split_nat_lse_states_sm90_gqa",
    "prefill_paged_weighted_relu_logits_sm90_steptron_gqa",
    "prewarm_csa_compact_prefill_update_with_slots_sm90_gqa",
    "prewarm_topk_selector_decode_meta_sm90_gqa",
    "token_wise_flash_attn_decode_sm90_gqa_func",
    "token_wise_flash_attn_decode_sm90_gqa_plan",
    "token_wise_flash_attn_prefill_union_sm90_gqa_func",
]

_INDEXER_INTERFACE = (
    "vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.indexer_ops.interface"
)
_TOKEN_INTERFACE = (
    "vllm.models.step4.nvidia.ops.cute_dsl."
    "sparse_gqa.token_sparse_attn.interface"
)
_SPLITKV = (
    "vllm.models.step4.nvidia.ops.cute_dsl."
    "sparse_gqa.token_sparse_attn.splitkv_merge_sm90_gqa"
)

_TOKEN_EXPORTS = {
    "token_wise_flash_attn_decode_sm90_gqa_func",
    "token_wise_flash_attn_decode_sm90_gqa_plan",
    "token_wise_flash_attn_prefill_union_sm90_gqa_func",
}
_SPLITKV_EXPORTS = {
    "merge_dynamic_split_nat_lse_states_sm90_gqa",
    "merge_variable_split_nat_lse_states_sm90_gqa",
}
_EXPORTS = {
    name: (
        _TOKEN_INTERFACE
        if name in _TOKEN_EXPORTS
        else _SPLITKV
        if name in _SPLITKV_EXPORTS
        else _INDEXER_INTERFACE,
        name,
    )
    for name in __all__
}


def _make_deferred_callable(name: str) -> Callable[..., Any]:
    def _deferred(*args: Any, **kwargs: Any) -> Any:
        from .._cutlass_compat import apply_patches

        apply_patches()
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value(*args, **kwargs)

    _deferred.__name__ = name
    _deferred.__qualname__ = name
    _deferred.__module__ = __name__
    return _deferred


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = _make_deferred_callable(name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])
