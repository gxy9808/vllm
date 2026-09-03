# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GQA sparse indexer kernels."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from vllm.models.step4.nvidia.ops.cute_dsl._cutlass_compat import apply_patches

__all__ = [
    "batch_decode_logits_wgmma_n",
    "batch_decode_weighted_relu_logits_sum_paged_sm90_steptron_gqa",
    "prefill_paged_weighted_relu_logits_sm90_steptron_gqa",
    "csa_compact_decode_stage_flush_with_slots_prevalidated_sm90_gqa",
    "csa_compact_decode_update_sm90_gqa",
    "csa_compact_decode_update_with_slots_prevalidated_sm90_gqa",
    "csa_compact_decode_update_with_slots_sm90_gqa",
    "csa_compact_prefill_update_with_slots_prevalidated_sm90_gqa",
    "csa_compact_update_sm90_gqa",
    "prewarm_csa_compact_prefill_update_with_slots_sm90_gqa",
    "decode_weighted_relu_logits_sum_paged_mean_warp_sm90_steptron_gqa",
    "decode_weighted_relu_logits_sum_paged_summary_auto_sm90_steptron_gqa",
    "decode_weighted_relu_logits_sum_paged_summary_warp_sm90_steptron_gqa",
    "decode_weighted_relu_logits_sum_paged_summary_warp_splitk_sm90_steptron_gqa",
    "decode_weighted_relu_logits_sum_sm90_steptron_gqa",
    "decode_weighted_relu_logits_sum_warp_sm90_steptron_gqa",
    "materialize_paged_summary_mean_cache_sm90_steptron_gqa",
    "materialize_selected_paged_summary_mean_cache_sm90_steptron_gqa",
    "rerank_weighted_relu_logits_sum_paged_mean_warp_sm90_steptron_gqa",
    "select_paged_summary_logits_split_k_sm90_steptron_gqa",
    "build_grouped_union_sparse_work_queue_gqa",
    "build_single_req_grouped_union_sparse_work_queue_gqa",
    "convert_region_block_topk_to_sparse_meta_step3p5",
    "convert_prefill_region_topk_to_sparse_meta_step3p5",
    "convert_prefill_region_topk_to_union_meta_step3p5",
    "cutedsl_topk_selector_decode_meta_sm90_gqa",
    "cutedsl_topk_selector_decode_meta_ordered_sm90_gqa",
    "cutedsl_topk_selector_prefill_sm90_gqa",
    "cutedsl_topk_selector_raw_sm90_gqa",
    "prewarm_topk_selector_decode_meta_sm90_gqa",
    "topk_selector_decode_meta_capacity_class_sm90_gqa",
    "pack_paged_summary_mean_sm90_gqa",
    "weighted_relu_logits_sum_sm90_gqa",
    "weighted_relu_logits_sum_sm90_steptron_gqa",
]

_EXPORTS = {
    name: ("vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.indexer_ops.interface", name)
    for name in __all__
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    apply_patches()
    module_name, attr = _EXPORTS[name]
    value = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))
