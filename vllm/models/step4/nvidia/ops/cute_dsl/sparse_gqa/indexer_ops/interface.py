# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public GQA indexer kernel interface."""

from __future__ import annotations

from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.indexer_ops.logits_sm90_gqa import (
    weighted_relu_logits_sum_sm90_gqa,
)
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.indexer_ops.logits_sm90_steptron_gqa import (
    weighted_relu_logits_sum_sm90_steptron_gqa,
)
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.indexer_ops.batch_decode_logits_sm90_steptron_gqa import (
    batch_decode_logits_wgmma_n,
    batch_decode_weighted_relu_logits_sum_paged_sm90_steptron_gqa,
)
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.indexer_ops.prefill_paged_logits_sm90_steptron_gqa import (
    prefill_paged_weighted_relu_logits_sm90_steptron_gqa,
)
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.indexer_ops.csa_compact_update_sm90_gqa import (
    csa_compact_decode_stage_flush_with_slots_prevalidated_sm90_gqa,
    csa_compact_decode_update_sm90_gqa,
    csa_compact_decode_update_with_slots_prevalidated_sm90_gqa,
    csa_compact_decode_update_with_slots_sm90_gqa,
    csa_compact_prefill_update_with_slots_prevalidated_sm90_gqa,
    csa_compact_update_sm90_gqa,
    prewarm_csa_compact_prefill_update_with_slots_sm90_gqa,
)
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.indexer_ops.decode_logits_sm90_steptron_gqa import (
    decode_weighted_relu_logits_sum_paged_mean_warp_sm90_steptron_gqa,
    decode_weighted_relu_logits_sum_paged_summary_auto_sm90_steptron_gqa,
    decode_weighted_relu_logits_sum_paged_summary_warp_sm90_steptron_gqa,
    decode_weighted_relu_logits_sum_paged_summary_warp_splitk_sm90_steptron_gqa,
    decode_weighted_relu_logits_sum_sm90_steptron_gqa,
    decode_weighted_relu_logits_sum_warp_sm90_steptron_gqa,
    materialize_paged_summary_mean_cache_sm90_steptron_gqa,
    materialize_selected_paged_summary_mean_cache_sm90_steptron_gqa,
    rerank_weighted_relu_logits_sum_paged_mean_warp_sm90_steptron_gqa,
    select_paged_summary_logits_split_k_sm90_steptron_gqa,
)
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.indexer_ops.paged_summary_pack_sm90_gqa import (
    pack_paged_summary_mean_sm90_gqa,
)
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.indexer_ops.region_sparse_meta_gqa import (
    build_grouped_union_sparse_work_queue_gqa,
    build_single_req_grouped_union_sparse_work_queue_gqa,
)
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.indexer_ops.decode_sparse_meta_step3p5 import (
    convert_region_block_topk_to_sparse_meta_step3p5,
)
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.indexer_ops.prefill_sparse_meta_step3p5 import (
    convert_prefill_region_topk_to_sparse_meta_step3p5,
    convert_prefill_region_topk_to_union_meta_step3p5,
)
from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.indexer_ops.topk_selector_sm90_gqa import (
    cutedsl_topk_selector_decode_meta_sm90_gqa,
    cutedsl_topk_selector_decode_meta_ordered_sm90_gqa,
    cutedsl_topk_selector_prefill_sm90_gqa,
    cutedsl_topk_selector_raw_sm90_gqa,
    prewarm_topk_selector_decode_meta_sm90_gqa,
    topk_selector_decode_meta_capacity_class_sm90_gqa,
)

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
