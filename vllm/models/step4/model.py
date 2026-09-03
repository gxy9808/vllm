# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Step4 model."""

import copy
import functools
import math
import typing
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import regex as re
import torch
from torch import nn
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SwigluStepAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import (
    get_attention_context,
    unified_kv_cache_update,
)
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer,
)
from vllm.model_executor.layers.batch_invariant import (
    linear_fp32_batch_invariant,
)
from vllm.model_executor.layers.layernorm import RMSNorm as NaiveRMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    get_spec_layer_idx_from_weight_name,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.model_executor.parameter import BasevLLMParameter, ModelWeightParameter
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionType

from .kernels import (
    Step4SparseConfig,
    checkpoint_has_step4_sparse_config,
    fused_indexer_norm_rope_forward_impl,
    fused_qknorm_rope_cache_forward_impl,
    fused_qknorm_rope_forward_impl,
    get_step4_sparse_config,
    has_optimus_moe_matmul_fp32,
    is_supported_optimus_qknorm_cache_rotary,
    router_bias_func,
)
from .layernorm import OptimusLayerNorm, OptimusRMSNorm

logger = init_logger(__name__)


def step4_materialize_gate_input(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clone()


def step4_materialize_gate_input_fake(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(tensor)


direct_register_custom_op(
    op_name="step4_materialize_gate_input",
    op_func=step4_materialize_gate_input,
    fake_impl=step4_materialize_gate_input_fake,
)


@eager_break_during_capture
@maybe_transfer_kv_layer
def step4_dsa_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    dsa_proxy_query: torch.Tensor,
    dsa_proxy_key: torch.Tensor,
    dsa_proxy_weights: torch.Tensor,
    dsa_proxy_z: torch.Tensor | None = None,
    dsa_summary_cache_sum: torch.Tensor | None = None,
    dsa_summary_cache_count: torch.Tensor | None = None,
    dsa_csa_active_region_ids: torch.Tensor | None = None,
    dsa_csa_active_slot_by_region: torch.Tensor | None = None,
    dsa_csa_numerator: torch.Tensor | None = None,
    dsa_csa_denominator: torch.Tensor | None = None,
    dsa_csa_max: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
    dsa_summary_cache_mean: torch.Tensor | None = None,
    dsa_csa_active_token_k: torch.Tensor | None = None,
    dsa_csa_active_token_z: torch.Tensor | None = None,
    dsa_csa_active_token_valid: torch.Tensor | None = None,
    dsa_mtp_source_to_transaction: torch.Tensor | None = None,
    dsa_mtp_row_source: torch.Tensor | None = None,
    dsa_mtp_row_regions: torch.Tensor | None = None,
    dsa_mtp_row_positions: torch.Tensor | None = None,
    dsa_mtp_row_owner_block: torch.Tensor | None = None,
    dsa_mtp_row_owner_block_index: torch.Tensor | None = None,
    dsa_mtp_correction_action: torch.Tensor | None = None,
    dsa_mtp_state_numerator: torch.Tensor | None = None,
    dsa_mtp_state_denominator: torch.Tensor | None = None,
    dsa_mtp_state_max_logits: torch.Tensor | None = None,
    dsa_mtp_state_pre_numerator: torch.Tensor | None = None,
    dsa_mtp_state_pre_denominator: torch.Tensor | None = None,
    dsa_mtp_state_pre_max_logits: torch.Tensor | None = None,
    dsa_mtp_correction_free_slots: torch.Tensor | None = None,
    dsa_mtp_correction_free_count: torch.Tensor | None = None,
    dsa_mtp_correction_allocation_cursor: torch.Tensor | None = None,
    dsa_order_token: torch.Tensor | None = None,
) -> None:
    # kv_cache_dummy_dep orders the split KV update before DSA attention.  The
    # summary tensors are explicit mutated args so torch.compile/CUDA graph see
    # the side effects performed through the Step4-local summary sidecar.
    # dsa_order_token is a model-wide dependency token: each DSA layer has
    # private CSA state tensors, while its metadata/scratch pool is shared.
    # Mutating this token forces compiled calls to remain in layer order.
    if dsa_order_token is not None:
        # Use a non-idempotent write rather than zero_(): it makes the
        # dependency visible even to aggressive compiler dead-write
        # elimination, while the token is reset at capture/wake boundaries.
        dsa_order_token.add_(1)
    del (
        kv_cache_dummy_dep,
        dsa_summary_cache_sum,
        dsa_summary_cache_count,
        dsa_csa_active_region_ids,
        dsa_csa_active_slot_by_region,
        dsa_csa_numerator,
        dsa_csa_denominator,
        dsa_csa_max,
        dsa_summary_cache_mean,
        dsa_csa_active_token_k,
        dsa_csa_active_token_z,
        dsa_csa_active_token_valid,
        dsa_mtp_source_to_transaction,
        dsa_mtp_row_source,
        dsa_mtp_row_regions,
        dsa_mtp_row_positions,
        dsa_mtp_row_owner_block,
        dsa_mtp_row_owner_block_index,
        dsa_mtp_correction_action,
        dsa_mtp_state_numerator,
        dsa_mtp_state_denominator,
        dsa_mtp_state_max_logits,
        dsa_mtp_state_pre_numerator,
        dsa_mtp_state_pre_denominator,
        dsa_mtp_state_pre_max_logits,
        dsa_mtp_correction_free_slots,
        dsa_mtp_correction_free_count,
        dsa_mtp_correction_allocation_cursor,
        dsa_order_token,
    )
    with torch.profiler.record_function("step4_dsa.op.get_attention_context"):
        attn_metadata, attn_layer, kv_cache, _ = get_attention_context(layer_name)
    with torch.profiler.record_function("step4_dsa.op.impl_forward"):
        attn_layer.impl.forward(
            attn_layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output=output,
            dsa_proxy_query=dsa_proxy_query,
            dsa_proxy_key=dsa_proxy_key,
            dsa_proxy_weights=dsa_proxy_weights,
            dsa_proxy_z=dsa_proxy_z,
        )


def step4_dsa_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    dsa_proxy_query: torch.Tensor,
    dsa_proxy_key: torch.Tensor,
    dsa_proxy_weights: torch.Tensor,
    dsa_proxy_z: torch.Tensor | None = None,
    dsa_summary_cache_sum: torch.Tensor | None = None,
    dsa_summary_cache_count: torch.Tensor | None = None,
    dsa_csa_active_region_ids: torch.Tensor | None = None,
    dsa_csa_active_slot_by_region: torch.Tensor | None = None,
    dsa_csa_numerator: torch.Tensor | None = None,
    dsa_csa_denominator: torch.Tensor | None = None,
    dsa_csa_max: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
    dsa_summary_cache_mean: torch.Tensor | None = None,
    dsa_csa_active_token_k: torch.Tensor | None = None,
    dsa_csa_active_token_z: torch.Tensor | None = None,
    dsa_csa_active_token_valid: torch.Tensor | None = None,
    dsa_mtp_source_to_transaction: torch.Tensor | None = None,
    dsa_mtp_row_source: torch.Tensor | None = None,
    dsa_mtp_row_regions: torch.Tensor | None = None,
    dsa_mtp_row_positions: torch.Tensor | None = None,
    dsa_mtp_row_owner_block: torch.Tensor | None = None,
    dsa_mtp_row_owner_block_index: torch.Tensor | None = None,
    dsa_mtp_correction_action: torch.Tensor | None = None,
    dsa_mtp_state_numerator: torch.Tensor | None = None,
    dsa_mtp_state_denominator: torch.Tensor | None = None,
    dsa_mtp_state_max_logits: torch.Tensor | None = None,
    dsa_mtp_state_pre_numerator: torch.Tensor | None = None,
    dsa_mtp_state_pre_denominator: torch.Tensor | None = None,
    dsa_mtp_state_pre_max_logits: torch.Tensor | None = None,
    dsa_mtp_correction_free_slots: torch.Tensor | None = None,
    dsa_mtp_correction_free_count: torch.Tensor | None = None,
    dsa_mtp_correction_allocation_cursor: torch.Tensor | None = None,
    dsa_order_token: torch.Tensor | None = None,
) -> None:
    return


direct_register_custom_op(
    op_name="step4_dsa_attention_with_output",
    op_func=step4_dsa_attention_with_output,
    mutates_args=[
        "output",
        "dsa_summary_cache_sum",
        "dsa_summary_cache_count",
        "dsa_csa_active_region_ids",
        "dsa_csa_active_slot_by_region",
        "dsa_csa_numerator",
        "dsa_csa_denominator",
        "dsa_csa_max",
        "dsa_summary_cache_mean",
        "dsa_csa_active_token_k",
        "dsa_csa_active_token_z",
        "dsa_csa_active_token_valid",
        "dsa_mtp_source_to_transaction",
        "dsa_mtp_row_source",
        "dsa_mtp_row_regions",
        "dsa_mtp_row_positions",
        "dsa_mtp_row_owner_block",
        "dsa_mtp_row_owner_block_index",
        "dsa_mtp_correction_action",
        "dsa_mtp_state_numerator",
        "dsa_mtp_state_denominator",
        "dsa_mtp_state_max_logits",
        "dsa_mtp_state_pre_numerator",
        "dsa_mtp_state_pre_denominator",
        "dsa_mtp_state_pre_max_logits",
        "dsa_mtp_correction_free_slots",
        "dsa_mtp_correction_free_count",
        "dsa_mtp_correction_allocation_cursor",
        "dsa_order_token",
    ],
    fake_impl=step4_dsa_attention_with_output_fake,
)


_OPTIONAL_FP8_ATTN_SCALE_SUFFIXES = (
    ".attn.q_scale",
    ".attn.k_scale",
    ".attn.v_scale",
    ".attn.q_quant_scale",
    ".attn.k_quant_scale",
    ".attn.v_quant_scale",
    ".attn.prob_scale",
)

STEP4_PACKED_MODULES_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "qkvg_proj": ["q_proj", "k_proj", "v_proj", "g_proj"],
    "qkv_indexer_proj": [
        "q_proj",
        "k_proj",
        "v_proj",
        "sparse_indexer_q",
        "sparse_indexer_k",
        "sparse_indexer_z",
        "g_proj",
    ],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _require_resolved_valid_vocab_size(model_config: ModelConfig) -> int:
    if model_config.valid_vocab_size is None:
        raise ValueError(
            "Step4 requires valid_vocab_size to be resolved from the tokenizer "
            "before model construction. Call VllmConfig.resolve_valid_vocab_size() "
            "or pass --valid-vocab-size when tokenizer initialization is skipped."
        )
    return model_config.get_valid_vocab_size()


def _step_layer_types(config: Any) -> list[str]:
    """Per-layer attention types spanning the dense stack and the MTP layers.

    `config.layer_types` only covers the dense stack so that transformers' own
    length validation passes; the MTP block indexes past it.
    """
    return (
        getattr(config, "layer_types_with_mtp", None)
        or getattr(config, "layer_types", None)
        or []
    )


def _parse_step4_layer_indices(
    value: str | Iterable[int] | None,
    *,
    name: str,
) -> set[int] | None:
    if value is None:
        return None
    raw_values = value.split(",") if isinstance(value, str) else value
    try:
        indices = [int(item) for item in raw_values if str(item).strip()]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Step4 {name} must contain integer layer indices.") from exc
    if len(indices) != len(set(indices)):
        raise ValueError(f"Step4 {name} contains duplicate layer indices.")
    return set(indices)


def _get_step4_moe_layer_indices(config: Any) -> set[int]:
    enum_indices = _parse_step4_layer_indices(
        getattr(config, "moe_layers_enum", None),
        name="moe_layers_enum",
    )
    list_indices = _parse_step4_layer_indices(
        getattr(config, "moe_layer_list", None),
        name="moe_layer_list",
    )
    if (
        enum_indices is not None
        and list_indices is not None
        and enum_indices != list_indices
    ):
        raise ValueError(
            "Step4 moe_layers_enum and moe_layer_list describe different layers."
        )

    indices = enum_indices if enum_indices is not None else list_indices
    if indices is None:
        indices = set(range(1, int(config.num_hidden_layers)))
    total_layers = int(config.num_hidden_layers) + int(
        getattr(config, "num_nextn_predict_layers", 0) or 0
    )
    invalid = sorted(index for index in indices if not 0 <= index < total_layers)
    if invalid:
        raise ValueError(
            f"Step4 MoE layer indices must be in [0, {total_layers}), got {invalid}."
        )
    return indices


def _set_step4_moe_protocol_metadata(
    model: Any,
    example_layer: Any | None,
) -> None:
    """Populate the MoE protocol for the layers local to this PP rank."""
    model.num_moe_layers = len(model.moe_layers)
    model.num_expert_groups = 1
    model.num_shared_experts = 0
    if example_layer is None:
        # A valid pipeline stage can contain only dense layers. Reporting zero
        # local MoE layers keeps that rank out of EPLB while other stages still
        # expose their local expert runners.
        model.num_logical_experts = 0
        model.num_physical_experts = 0
        model.num_local_physical_experts = 0
        model.num_routed_experts = 0
        model.num_redundant_experts = 0
        return

    model.num_logical_experts = example_layer.n_logical_experts
    model.num_physical_experts = example_layer.n_physical_experts
    model.num_local_physical_experts = example_layer.n_local_physical_experts
    model.num_routed_experts = example_layer.n_routed_experts
    model.num_redundant_experts = example_layer.n_redundant_experts


def _verify_step4_dsa_kv_transfer_compatibility(vllm_config: VllmConfig) -> None:
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is not None and kv_transfer_config.is_kv_transfer_instance:
        raise ValueError(
            "Step4 DSA sparse summary side storage is not compatible with KV "
            "transfer/offload. Disable kv_transfer_config for Step4."
        )


def _step4_groups_per_rank(total_groups: int, tp_size: int) -> int:
    """Return the number of contiguous groups visible on one TP rank."""
    total_groups = int(total_groups)
    tp_size = int(tp_size)
    if total_groups <= 0 or tp_size <= 0:
        raise ValueError(
            "Step4 parallel group counts must be positive, got "
            f"groups={total_groups}, tp_size={tp_size}."
        )
    if total_groups % tp_size == 0:
        return total_groups // tp_size
    if tp_size % total_groups == 0:
        return 1
    raise ValueError(
        f"Step4 tensor parallel size {tp_size} is incompatible with "
        f"group count {total_groups}: one must divide the other."
    )


def _step4_replicated_group_rank(
    *,
    total_groups: int,
    tp_size: int,
    tp_rank: int,
) -> tuple[int, int]:
    """Map a TP rank to one replicated provider/KV group.

    The production DSA kernels consume exactly one local KV/provider group.
    Consequently global groups may be replicated across ranks, but multiple
    groups cannot be co-located on one rank.
    """
    total_groups = int(total_groups)
    tp_size = int(tp_size)
    tp_rank = int(tp_rank)
    if total_groups <= 0 or tp_size <= 0:
        raise ValueError(
            "Step4 replicated group counts must be positive, got "
            f"groups={total_groups}, tp_size={tp_size}."
        )
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"Step4 TP rank must be in [0, {tp_size}), got {tp_rank}.")
    if tp_size < total_groups or tp_size % total_groups != 0:
        raise ValueError(
            "Step4 DSA production kernels require exactly one local "
            "provider/KV group, so the global group count must divide tensor "
            "parallel size and cannot exceed it; got "
            f"groups={total_groups}, tp_size={tp_size}."
        )
    ranks_per_group = tp_size // total_groups
    return tp_rank // ranks_per_group, ranks_per_group


def _validate_step4_dsa_parallel_geometry(
    *,
    total_num_heads: int,
    total_num_kv_heads: int,
    indexer_num_heads: int,
    index_tp_size: int,
    tp_size: int,
) -> None:
    """Fail fast when DSA provider and attention GQA groups diverge.

    The sparse indexer reduces scores within provider groups, while the
    attention kernel consumes the corresponding KV groups.  These partitions
    must describe the same global and per-rank geometry; otherwise the model
    still constructs but selects regions for the wrong query/KV heads.
    """
    total_num_heads = int(total_num_heads)
    total_num_kv_heads = int(total_num_kv_heads)
    indexer_num_heads = int(indexer_num_heads)
    index_tp_size = int(index_tp_size)
    tp_size = int(tp_size)
    if total_num_heads <= 0 or total_num_kv_heads <= 0:
        raise ValueError(
            "Step4 attention head counts must be positive, got "
            f"num_heads={total_num_heads}, num_kv_heads={total_num_kv_heads}."
        )
    if index_tp_size <= 0:
        raise ValueError(
            "Step4 DSA provider group count must be positive, got "
            f"index_tp_size={index_tp_size}."
        )
    if index_tp_size != total_num_kv_heads:
        raise ValueError(
            "Step4 DSA provider groups must align with attention KV groups: "
            f"{index_tp_size} != {total_num_kv_heads}"
        )
    if indexer_num_heads <= 0 or indexer_num_heads % index_tp_size:
        raise ValueError(
            f"{indexer_num_heads} sparse indexer heads do not divide into "
            f"{index_tp_size} provider groups."
        )
    local_kv_groups = _step4_groups_per_rank(total_num_kv_heads, tp_size)
    local_provider_groups = _step4_groups_per_rank(index_tp_size, tp_size)
    if local_provider_groups != local_kv_groups:
        raise ValueError(
            "local DSA provider groups must align with local KV groups: "
            f"{local_provider_groups} != {local_kv_groups}"
        )
    if local_kv_groups != 1:
        raise ValueError(
            "Step4 DSA production kernels require exactly one local "
            "KV/provider group per tensor-parallel rank, got "
            f"local_groups={local_kv_groups} "
            f"(global_groups={total_num_kv_heads}, tp_size={tp_size})."
        )
    if total_num_heads % tp_size:
        raise ValueError(
            "Step4 attention heads must be divisible by tensor parallel size: "
            f"num_heads={total_num_heads}, tp_size={tp_size}."
        )
    local_q_heads = total_num_heads // tp_size
    if local_q_heads % local_kv_groups:
        raise ValueError(
            f"{local_q_heads} local Q heads do not divide into "
            f"{local_kv_groups} local KV groups."
        )
    q_heads_per_kv = local_q_heads // local_kv_groups
    if q_heads_per_kv not in (4, 8, 16):
        raise ValueError(
            "Step4 DSA production kernels require 4, 8, or 16 local query "
            "heads per KV group, got "
            f"{q_heads_per_kv} (num_heads={total_num_heads}, tp_size={tp_size})."
        )


def _per_layer_value(
    values: list[Any] | tuple[Any, ...] | None,
    layer_idx: int,
    *,
    name: str,
    default: Any,
) -> Any:
    if not values:
        return default
    if layer_idx >= len(values):
        raise ValueError(
            f"Step4 {name} has {len(values)} entries, but layer {layer_idx} "
            "requires an entry."
        )
    return values[layer_idx]


def _is_step4_full_attention_layer(
    config: Any,
    layer_idx: int,
    sparse_config: Any | None = None,
) -> bool:
    layer_types = _step_layer_types(config)
    layer_type = (
        layer_types[layer_idx] if layer_idx < len(layer_types) else "full_attention"
    )
    if sparse_config is not None:
        apply_to = getattr(
            sparse_config,
            "apply_to_layer_types",
            ("full_attention",),
        )
        return layer_type in apply_to
    return layer_type == "full_attention"


def _mark_optional_fp8_attention_scales_loaded(
    loaded_params: set[str],
    params_dict: dict[str, torch.nn.Parameter],
) -> None:
    # Step FP8 attention scale tensors are optional calibration metadata. Older
    # fp8 checkpoints do not carry them; the attention post-load hook validates
    # partial scale sets and fills the missing all-or-nothing case.
    loaded_params.update(
        name for name in params_dict if name.endswith(_OPTIONAL_FP8_ATTN_SCALE_SUFFIXES)
    )


def RMSNormFactory(
    hidden_size: int,
    eps: float = 1e-6,
    zero_centered: bool = False,
    dtype: torch.dtype | None = None,
):
    if zero_centered:
        return OptimusRMSNorm(hidden_size, eps, zero_centered, dtype=dtype)
    return NaiveRMSNorm(hidden_size, eps, dtype=dtype)


_NORM_DTYPE_TO_TORCH_DTYPE = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "float": torch.float32,
}


def get_norm_dtype(config: Any) -> torch.dtype:
    norm_dtype = config.norm_dtype.lower()
    if norm_dtype in _NORM_DTYPE_TO_TORCH_DTYPE:
        return _NORM_DTYPE_TO_TORCH_DTYPE[norm_dtype]
    raise ValueError(f"Unknown norm_dtype: {norm_dtype!r}")


def check_loaded_weight_dtype(
    loaded_weight: torch.Tensor,
    param: torch.nn.Parameter,
) -> None:
    assert loaded_weight.dtype == param.dtype, (
        f"Loaded weight dtype {loaded_weight.dtype} does not match "
        f"parameter dtype {param.dtype}."
    )


def pad_param(
    weight: torch.Tensor,
    name: str,
    param: torch.nn.Parameter,
    quant_config: QuantizationConfig | None = None,
) -> torch.Tensor:
    """Pad 2D weight for groupwise quantization TP sharding.

    Decide whether to pad based on `param.quant_method`:
    - None / unquantized => no padding
    - otherwise => pad when using `groupwise_quant`
    """
    if weight.dim() != 2:
        return weight

    quant_method = getattr(param, "quant_method", None)
    if (
        quant_config is None
        or quant_config.get_name() != "groupwise_quant"
        or not quant_method
    ):
        return weight

    world_size = get_tensor_model_parallel_world_size()
    group_size = quant_config.group_size

    if ("down_proj.scales" in name) or ("w2_weight_scale" in name):
        group_size = 1

    ic, oc = weight.shape
    if ("down" in name) or ("w2" in name):
        ic_pad = (
            int(math.ceil(ic / group_size / world_size) * world_size * group_size) - ic
        )
        out = torch.nn.functional.pad(weight, (0, 0, 0, ic_pad), "constant", 0)
    else:
        oc_pad = (
            int(math.ceil(oc / group_size / world_size) * world_size * group_size) - oc
        )
        out = torch.nn.functional.pad(weight, (0, oc_pad, 0, 0), "constant", 0)

    logger.debug(
        "padding %s, quant_config=%s, original weight.shape=%s, padded weight.shape=%s",
        name,
        quant_config,
        tuple(weight.shape),
        tuple(out.shape),
    )
    return out


def _pad_size_for_groupwise_quant(
    size: int,
    quant_config: QuantizationConfig | None = None,
) -> int:
    """Pad `size` to a multiple of `group_size * tensor_parallel_world_size`."""
    if quant_config is None or quant_config.get_name() != "groupwise_quant":
        return size

    group_size = getattr(quant_config, "group_size", None)
    if not isinstance(group_size, int) or group_size <= 0:
        return size

    world_size = get_tensor_model_parallel_world_size()
    multiple = group_size * world_size
    return int(math.ceil(size / multiple) * multiple)


def _is_mxfp4_moe_quant_config(quant_config: QuantizationConfig | None) -> bool:
    return quant_config is not None and quant_config.get_name() == "mxfp4"


class FP32ReplicatedLinear(ReplicatedLinear):
    """
    Use FP32 for higher precision.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # OptimusMoe is an optional extension.  Resolve its availability while
        # constructing the layer, rather than from ``forward``: calling the
        # cached capability probe from a torch.compile-traced function makes
        # Dynamo trace ``functools.cache`` and can produce a graph break (and
        # potentially different compiled graphs across ranks).
        self._use_optimus_matmul_fp32 = has_optimus_moe_matmul_fp32()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        # The optional Optimus op upcasts internally. The portable path must
        # upcast both operands explicitly because this layer keeps model dtype.
        # Bias is always disabled, matching the None returned alongside.
        if envs.VLLM_BATCH_INVARIANT and current_platform.is_cuda():
            router_logits = linear_fp32_batch_invariant(x, self.weight.data)
        elif self._use_optimus_matmul_fp32:
            router_logits = torch.ops.vllm.optimus_matmul_fp32(x, self.weight.data)
        else:
            router_logits = torch.nn.functional.linear(
                x.to(torch.float32), self.weight.to(torch.float32)
            )
        return router_logits, None


class Step4FusedQKVIndexerLinear(QKVParallelLinear):
    """Main-attention qkv and the sparse-indexer q/k/z/g in one GEMM.

    Both projections consume the same normed hidden state, so they live in a
    single weight `[qkv_rows | indexer_rows]` and one `hidden @ W` replaces two
    GEMMs. The indexer rows are appended to the qkv rows of the base class, which
    keeps every q/k/v shard offset (and the kv-head replication for
    total_num_kv_heads < tp_size) inside `QKVParallelLinear.weight_loader_v2`
    instead of being reimplemented here.
    """

    INDEXER_SHARD_IDS = ("index_q", "index_k", "index_z", "index_g")

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        *,
        indexer_q_output_size: int,
        indexer_kv_output_size: int,
        gate_output_size: int | None,
        proxy_dim: int,
        index_tp_size: int,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            hidden_size,
            head_size,
            total_num_heads,
            total_num_kv_heads,
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
        )
        if not isinstance(self.quant_method, UnquantizedLinearMethod):
            raise ValueError(
                "Step4 DSA layers fuse qkv with the sparse indexer into one "
                "plain GEMM and cannot be quantized, but "
                f"{prefix}.weight resolved to "
                f"{type(self.quant_method).__name__}."
            )

        tp_size = self.tp_size
        self.proxy_dim = int(proxy_dim)
        self.has_gate = gate_output_size is not None and int(gate_output_size) > 0
        self._fused_checkpoint_shard_sizes = (
            ("q", self.total_num_heads * self.head_size),
            ("k", self.total_num_kv_heads * self.head_size),
            ("v", self.total_num_kv_heads * self.v_head_size),
            ("index_q", int(indexer_q_output_size)),
            ("index_k", int(indexer_kv_output_size)),
            ("index_z", int(indexer_kv_output_size)),
            ("index_g", int(gate_output_size or 0)),
        )

        total_index_q_heads = int(indexer_q_output_size) // self.proxy_dim
        total_index_kv_heads = int(indexer_kv_output_size) // self.proxy_dim
        if tp_size >= total_index_kv_heads:
            self.local_index_kv_heads = 1
            self.index_kv_head_replicas = tp_size // total_index_kv_heads
        else:
            self.local_index_kv_heads = total_index_kv_heads // tp_size
            self.index_kv_head_replicas = 1
        # The sparse indexer q heads are sharded into `index_tp_size` global
        # provider groups (default 4, matching the main-attention KV groups).
        # Production kernels consume one group per rank, so each group is
        # replicated across tp_size // index_tp_size ranks. This keeps the
        # per-rank index q-head set -- and therefore the GQA sparse-topk
        # aggregation -- identical at TP4/TP8/TP16. Splitting q across the full
        # tp_size instead would shrink the aggregation group and corrupt sparse
        # region selection.
        index_tp_size = int(index_tp_size)
        if index_tp_size <= 0:
            index_tp_size = tp_size
        self.index_tp_rank, ranks_per_index = _step4_replicated_group_rank(
            total_groups=index_tp_size,
            tp_size=tp_size,
            tp_rank=self.tp_rank,
        )
        if total_index_q_heads <= 0 or total_index_q_heads % index_tp_size != 0:
            raise ValueError(
                "Step4 sparse indexer q heads must be divisible by "
                f"index_tp_size, got total_index_q_heads={total_index_q_heads}, "
                f"index_tp_size={index_tp_size}."
            )
        if total_index_kv_heads <= 0:
            raise ValueError(
                "Step4 sparse indexer requires at least one KV head, got "
                f"{total_index_kv_heads}."
            )
        if total_index_kv_heads > tp_size:
            if total_index_kv_heads % tp_size != 0:
                raise ValueError(
                    "Step4 sparse indexer KV heads must divide the tensor "
                    "parallel size when sharded, got "
                    f"total_index_kv_heads={total_index_kv_heads}, "
                    f"tp_size={tp_size}."
                )
        elif tp_size % total_index_kv_heads != 0:
            raise ValueError(
                "Step4 sparse indexer KV heads must divide tensor parallel "
                "size when replicated, got "
                f"total_index_kv_heads={total_index_kv_heads}, tp_size={tp_size}."
            )
        _validate_step4_dsa_parallel_geometry(
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
            indexer_num_heads=total_index_q_heads,
            index_tp_size=index_tp_size,
            tp_size=tp_size,
        )
        self.local_index_q_heads = total_index_q_heads // index_tp_size
        self.index_q_head_replicas = ranks_per_index
        self.local_gate_size = int(gate_output_size) // tp_size if self.has_gate else 0

        self.qkv_size = int(self.output_size_per_partition)
        self.indexer_size = (
            self.local_index_q_size
            + self.local_index_k_size
            + self.local_index_z_size
            + self.local_gate_size
        )
        # The base class only allocated the qkv rows; replace that parameter with
        # one that also covers the indexer rows, since a single GEMM requires a
        # single storage. Done before any weight loading, so this is the only
        # weight this module ever owns.
        weight = ModelWeightParameter(
            data=torch.zeros(
                self.qkv_size + self.indexer_size,
                self.input_size_per_partition,
                dtype=self.params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=self.weight_loader_v2,
        )
        self.register_parameter("weight", weight)
        self.update_param_tp_status()
        self._loaded_shards: set[str] = set()

    @property
    def local_index_q_size(self) -> int:
        return int(self.local_index_q_heads * self.proxy_dim)

    @property
    def local_index_k_size(self) -> int:
        return int(self.local_index_kv_heads * self.proxy_dim)

    @property
    def local_index_z_size(self) -> int:
        return int(self.local_index_kv_heads * self.proxy_dim)

    def validate_shard_id(self, loaded_shard_id: str | None) -> None:
        if loaded_shard_id in self.INDEXER_SHARD_IDS:
            return
        super().validate_shard_id(loaded_shard_id)

    def _indexer_source_offset(self, shard_id: str) -> tuple[int, int]:
        if shard_id == "index_q":
            # q is sharded into index_tp_size provider groups; the ranks that
            # share an index group load the identical q shard (replicated).
            return (
                self.index_tp_rank * self.local_index_q_size,
                self.local_index_q_size,
            )
        if shard_id in ("index_k", "index_z"):
            shard_index = self.tp_rank // self.index_kv_head_replicas
            return shard_index * self.local_index_k_size, self.local_index_k_size
        if shard_id == "index_g":
            if not self.has_gate:
                raise ValueError("Sparse indexer gate is disabled.")
            return self.tp_rank * self.local_gate_size, self.local_gate_size
        raise ValueError(f"Unsupported sparse indexer shard: {shard_id}.")

    def _indexer_dest_offset(self, shard_id: str) -> int:
        offset = self.qkv_size
        if shard_id == "index_q":
            return offset
        offset += self.local_index_q_size
        if shard_id == "index_k":
            return offset
        offset += self.local_index_k_size
        if shard_id == "index_z":
            return offset
        offset += self.local_index_z_size
        if shard_id == "index_g":
            return offset
        raise ValueError(f"Unsupported sparse indexer shard: {shard_id}.")

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ) -> None:
        if loaded_shard_id is None:
            self._load_fused_checkpoint_weight(param, loaded_weight)
            return
        if loaded_shard_id in self.INDEXER_SHARD_IDS:
            src_offset, shard_size = self._indexer_source_offset(loaded_shard_id)
            dst_offset = self._indexer_dest_offset(loaded_shard_id)
            shard = loaded_weight.narrow(0, src_offset, shard_size)
            param.data.narrow(0, dst_offset, shard_size).copy_(shard)
        else:
            super().weight_loader_v2(param, loaded_weight, loaded_shard_id)
        self._loaded_shards.add(loaded_shard_id)

    def _load_fused_checkpoint_weight(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        if loaded_weight.ndim != 2 or loaded_weight.shape[1] != param.data.shape[1]:
            raise ValueError(
                "Step4 fused qkv+indexer checkpoint weight must be a 2D tensor "
                f"with input dimension {param.data.shape[1]}, got "
                f"{tuple(loaded_weight.shape)}."
            )
        if loaded_weight.shape == param.data.shape:
            param.data.copy_(loaded_weight)
            self._loaded_shards.update(self.required_shard_ids())
            return

        shard_sizes = [
            (shard_id, shard_size)
            for shard_id, shard_size in self._fused_checkpoint_shard_sizes
            if shard_size > 0
        ]
        expected_rows = sum(shard_size for _, shard_size in shard_sizes)
        if loaded_weight.shape[0] != expected_rows:
            raise ValueError(
                "Step4 fused qkv+indexer checkpoint weight has an unexpected "
                f"output dimension: expected local {param.data.shape[0]} or "
                f"global {expected_rows}, got {loaded_weight.shape[0]}."
            )

        offset = 0
        for shard_id, shard_size in shard_sizes:
            shard = loaded_weight.narrow(0, offset, shard_size)
            self.weight_loader_v2(param, shard, shard_id)
            offset += shard_size

    def required_shard_ids(self) -> set[str]:
        required = {"q", "k", "v", "index_q", "index_k", "index_z"}
        if self.has_gate:
            required.add("index_g")
        return required

    def assert_fully_loaded(self, prefix: str) -> None:
        """Fail unless this load covered every shard, then forget the record.

        Tracking per load (instead of latching a flag on first success) is what
        makes weight reloading safe: a reload materializes fresh uninitialized
        storage for the whole parameter, so a shard the new weight stream omits
        would otherwise silently become garbage.
        """
        required = self.required_shard_ids()
        missing = sorted(required - self._loaded_shards)
        if missing:
            raise ValueError(
                f"Incomplete fused qkv+indexer weight for {prefix}: "
                f"missing shards {missing}."
            )
        self._loaded_shards.clear()

    def reset_load_state(self) -> None:
        self._loaded_shards.clear()

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Both halves are column slices, so the Optimus consumers see a dim-0
        # stride wider than the slice. That is supported: those kernels compile
        # dim-0 stride as a symbolic value, dim 1 stays contiguous, and the base
        # pointer alignment is inferred from the tensor rather than assumed.
        combined = torch.nn.functional.linear(hidden_states, self.weight)
        return combined[..., : self.qkv_size], combined[..., self.qkv_size :]

    def split_indexer(
        self,
        qkzg: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        sections = [
            self.local_index_q_size,
            self.local_index_k_size,
            self.local_index_z_size,
        ]
        if not self.has_gate:
            index_q, index_k, index_z = qkzg.split(sections, dim=-1)
            return index_q, index_k, index_z, None
        sections.append(self.local_gate_size)
        index_q, index_k, index_z, gate = qkzg.split(sections, dim=-1)
        return index_q, index_k, index_z, gate


def _reset_fused_qkv_indexer_load_state(module: nn.Module) -> None:
    for submodule in module.modules():
        if isinstance(submodule, Step4FusedQKVIndexerLinear):
            submodule.reset_load_state()


def _validate_fused_qkv_indexer_weights(module: nn.Module) -> set[str]:
    loaded_params: set[str] = set()
    for name, submodule in module.named_modules():
        if not isinstance(submodule, Step4FusedQKVIndexerLinear):
            continue
        param_name = f"{name}.weight" if name else "weight"
        submodule.assert_fully_loaded(param_name)
        loaded_params.add(param_name)
    return loaded_params


class Step4SparseIndexerIndexTPLinear(nn.Module):
    """Sparse-indexer linear sharded by its own tensor-parallel size."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        params_dtype: torch.dtype,
        index_tp_size: int,
    ) -> None:
        super().__init__()
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        index_tp_size = int(index_tp_size)
        if index_tp_size <= 0:
            index_tp_size = tp_size
        self.index_tp_rank, self.index_tp_rank_replicas = _step4_replicated_group_rank(
            total_groups=index_tp_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        if output_size <= 0 or output_size % index_tp_size != 0:
            raise ValueError(
                "Step4 sparse indexer output size must be divisible by "
                f"index_tp_size, got output_size={output_size}, "
                f"index_tp_size={index_tp_size}."
            )
        self.output_size_per_partition = output_size // index_tp_size
        self.weight = Parameter(
            torch.empty(
                self.output_size_per_partition,
                input_size,
                dtype=params_dtype,
            )
        )
        self.weight.weight_loader = self.weight_loader
        self.weight.data.zero_()

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor) -> None:
        start = self.index_tp_rank * self.output_size_per_partition
        shard = loaded_weight.narrow(0, start, self.output_size_per_partition)
        param.data.copy_(shard)

    def forward(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        return torch.nn.functional.linear(input_, self.weight, None), None


class Step4MLP(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        intermediate_size = _pad_size_for_groupwise_quant(
            intermediate_size, quant_config
        )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )

        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.prefix = prefix
        self.hidden_size = hidden_size
        self.limit = None
        layer_idx = extract_layer_index(prefix)
        swiglu_limit = _per_layer_value(
            getattr(config, "swiglu_limits_shared", None),
            layer_idx,
            name="swiglu_limits_shared",
            default=None,
        )
        if swiglu_limit not in (None, 0):
            self.limit = swiglu_limit
            self.act_fn = SwigluStepAndMul(limit=self.limit, compile_native=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        intermediate_act = self.act_fn(gate_up)
        output, _ = self.down_proj(intermediate_act)
        return output


def _step4_moe_reduce_policy(tp_size: int, dp_size: int) -> tuple[bool, bool]:
    """Return combined-reduce and per-path-reduce settings for Step4 MoE."""
    fuse_all_reduce = tp_size > 1 and dp_size == 1
    return fuse_all_reduce, not fuse_all_reduce


class Step4Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float | list[float] | None = 10000,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        rope_scaling: dict[str, Any] | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        # Step4 specific args
        sliding_window: int | None = None,
        use_head_wise_attn_gate: bool = False,
        layer_types: list = None,
        use_rope_layers: list = None,
        yarn_only_types: list = None,
        swa_num_attention_heads: int | None = None,
        partial_rotary_factor: float = 1.0,
        zero_centered: bool = True,
        vllm_config: VllmConfig | None = None,
        sparse_config: Step4SparseConfig | None = None,
        model_has_dsa_layers: bool = False,
        norm_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_size
        self.layer_idx = extract_layer_index(prefix)
        self.prefix = prefix
        sparse_indexer_base_rope_parameters = (
            dict(rope_scaling) if rope_scaling is not None else {}
        )
        default_layer_type = (
            "sliding_attention" if self.layer_idx % 2 == 0 else "full_attention"
        )
        layer_type = _per_layer_value(
            layer_types,
            self.layer_idx,
            name="layer_types",
            default=default_layer_type,
        )
        enable_sliding_window = layer_type == "sliding_attention"
        if yarn_only_types and layer_type not in yarn_only_types:
            rope_scaling = None

        if sliding_window is not None and enable_sliding_window:
            sliding_window = sliding_window
            if swa_num_attention_heads is not None:
                num_heads = swa_num_attention_heads
                self.total_num_heads = swa_num_attention_heads
        else:
            sliding_window = None

        if isinstance(rope_theta, list):
            if not rope_theta:
                raise ValueError("Step4 rope_theta cannot be an empty list.")
            rope_theta = _per_layer_value(
                rope_theta,
                self.layer_idx,
                name="rope_theta",
                default=None,
            )

        self.rank = get_tensor_model_parallel_rank()
        if self.total_num_heads <= 0 or self.total_num_heads % tp_size != 0:
            raise ValueError(
                "Step4 attention heads must be positive and divisible by tensor "
                f"parallel size, got num_heads={self.total_num_heads}, "
                f"tp_size={tp_size}."
            )
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads <= 0:
            raise ValueError(
                "Step4 attention requires a positive number of KV heads, got "
                f"{self.total_num_kv_heads}."
            )
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            if self.total_num_kv_heads % tp_size != 0:
                raise ValueError(
                    "Step4 KV heads must be divisible by tensor parallel size "
                    "when sharded, got "
                    f"num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
                )
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            if tp_size % self.total_num_kv_heads != 0:
                raise ValueError(
                    "Step4 KV heads must divide tensor parallel size when "
                    "replicated, got "
                    f"num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
                )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        if head_dim is None and hidden_size % self.total_num_heads != 0:
            raise ValueError(
                "Step4 hidden_size must be divisible by num_heads when head_dim "
                f"is omitted, got hidden_size={hidden_size}, "
                f"num_heads={self.total_num_heads}."
            )
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        if self.head_dim <= 0:
            raise ValueError(f"Step4 head_dim must be positive, got {self.head_dim}.")
        self.partial_rotary_factor = float(partial_rotary_factor)
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if (
            self.partial_rotary_factor <= 0.0
            or self.partial_rotary_factor > 1.0
            or self.rotary_dim <= 0
            or self.rotary_dim % 2 != 0
        ):
            raise ValueError(
                "Step4 partial_rotary_factor must produce a positive, even "
                "rotary dimension no larger than head_dim, got "
                f"head_dim={self.head_dim}, "
                f"partial_rotary_factor={self.partial_rotary_factor}, "
                f"rotary_dim={self.rotary_dim}."
            )
        if max_position is None or int(max_position) <= 0:
            raise ValueError(
                f"Step4 max_position must be a positive integer, got {max_position}."
            )
        max_position = int(max_position)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        linear_quant_config = (
            quant_config
            if quant_config is None or quant_config.get_name() != "fp8"
            else None
        )

        # Q/K/V and the per-head attention gate consume the same normalized
        # hidden states.  When explicitly enabled, pack them into one
        # column-parallel GEMM if KV heads are partitioned (rather than
        # replicated) across TP ranks.  The local output layout is
        # [Q, K, V, gate].
        self.fuse_qkv_gate = (
            envs.VLLM_STEP4_ENABLE_QKVG_PROJ
            and use_head_wise_attn_gate
            and self.total_num_kv_heads >= tp_size
        )
        self.sparse_config = sparse_config
        self.use_dsa_backend = sparse_config is not None and sliding_window is None
        if self.use_dsa_backend:
            if vllm_config is None:
                raise ValueError("Step4 DSA requires a complete VllmConfig.")
            _verify_step4_dsa_kv_transfer_compatibility(vllm_config)
            if not current_platform.is_device_capability((9, 0)):
                raise NotImplementedError(
                    "Step4 DSA currently requires an NVIDIA Hopper GPU with "
                    "compute capability 9.0; observed "
                    f"{current_platform.get_device_capability()}."
                )
            if self.num_kv_heads != 1:
                raise ValueError(
                    "Step4 DSA requires exactly one local KV head, got "
                    f"{self.num_kv_heads} (global={self.total_num_kv_heads}, "
                    f"tp_size={tp_size})."
                )
            if self.num_heads % self.num_kv_heads != 0:
                raise ValueError(
                    "Step4 DSA query heads must be divisible by local KV heads, "
                    f"got query_heads={self.num_heads}, "
                    f"kv_heads={self.num_kv_heads}."
                )
            queries_per_kv_head = self.num_heads // self.num_kv_heads
            if queries_per_kv_head not in (4, 8, 16):
                raise ValueError(
                    "Step4 DSA supports 4, 8, or 16 local query heads per KV "
                    f"head, got {queries_per_kv_head}."
                )
            if self.head_dim not in (128, 192):
                raise ValueError(
                    "Step4 DSA supports attention head dimensions 128 or 192, "
                    f"got {self.head_dim}."
                )
            if vllm_config.parallel_config.use_ubatching:
                raise ValueError(
                    "Step4 DSA does not support dual-batch overlap or "
                    "microbatching yet. Disable --enable-dbo and set "
                    "--ubatch-size 1."
                )
            sparse_config = typing.cast(Step4SparseConfig, sparse_config)
            _validate_step4_dsa_parallel_geometry(
                total_num_heads=self.total_num_heads,
                total_num_kv_heads=self.total_num_kv_heads,
                indexer_num_heads=int(sparse_config.sparse_indexer_num_heads),
                index_tp_size=int(sparse_config.index_tp_size),
                tp_size=tp_size,
            )
            if qkv_bias:
                raise ValueError(
                    "Step4 DSA layers fuse qkv with the sparse indexer into "
                    "one bias-free GEMM, but attention_bias is enabled."
                )
            proxy_dim = int(sparse_config.proxy_dim)
            # The sparse-indexer architecture comes from model config. Register
            # it before weight loading so loader threads only copy tensors and
            # never mutate the module/parameter structure.
            self.qkv_indexer_proj = Step4FusedQKVIndexerLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                indexer_q_output_size=(
                    int(sparse_config.sparse_indexer_num_heads) * proxy_dim
                ),
                indexer_kv_output_size=(
                    int(sparse_config.sparse_indexer_num_k_heads) * proxy_dim
                ),
                gate_output_size=(
                    self.total_num_heads if use_head_wise_attn_gate else None
                ),
                proxy_dim=proxy_dim,
                index_tp_size=int(sparse_config.index_tp_size),
                quant_config=linear_quant_config,
                prefix=f"{prefix}.qkv_indexer_proj",
            )
            self.params_dtype = self.qkv_indexer_proj.params_dtype
        elif self.fuse_qkv_gate:
            self.qkvg_proj = MergedColumnParallelLinear(
                hidden_size,
                [
                    self.total_num_heads * self.head_dim,
                    self.total_num_kv_heads * self.head_dim,
                    self.total_num_kv_heads * self.head_dim,
                    self.total_num_heads,
                ],
                bias=qkv_bias,
                quant_config=linear_quant_config,
                # Keep the original prefix so quantization configuration that
                # targets qkv_proj continues to apply to the fused module.
                prefix=f"{prefix}.qkv_proj",
            )
            if self.qkvg_proj.bias is not None:
                # g_proj is bias-free.  The fused allocation includes a gate
                # bias only because qkv_bias applies to the first three shards.
                with torch.no_grad():
                    self.qkvg_proj.bias[-self.num_heads :].zero_()
            self.params_dtype = self.qkvg_proj.params_dtype
        else:
            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=qkv_bias,
                quant_config=linear_quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            self.params_dtype = self.qkv_proj.params_dtype
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=linear_quant_config,
            prefix=f"{prefix}.o_proj",
        )

        rope_parameters: dict[str, Any] = (
            dict(rope_scaling) if rope_scaling is not None else {}
        )
        rope_parameters.setdefault("rope_type", "default")
        if self.rope_theta is not None:
            rope_parameters["rope_theta"] = self.rope_theta
        rope_parameters["partial_rotary_factor"] = partial_rotary_factor

        sparse_indexer_base_rope_parameters.setdefault("rope_type", "default")
        if self.rope_theta is not None:
            sparse_indexer_base_rope_parameters["rope_theta"] = self.rope_theta
        sparse_indexer_base_rope_parameters["partial_rotary_factor"] = (
            partial_rotary_factor
        )

        self._sparse_indexer_base_rope_parameters = dict(
            sparse_indexer_base_rope_parameters
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            # This path passes the cache directly to Optimus instead of calling
            # RotaryEmbedding.forward(), so it must request the activation
            # dtype up front rather than relying on the forward-time cast.
            dtype=self.params_dtype,
        )

        self.zero_centered = zero_centered
        self.q_norm = RMSNormFactory(
            self.head_dim,
            eps=rms_norm_eps,
            zero_centered=self.zero_centered,
            dtype=norm_dtype,
        )
        self.k_norm = RMSNormFactory(
            self.head_dim,
            eps=rms_norm_eps,
            zero_centered=self.zero_centered,
            dtype=norm_dtype,
        )
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        if (
            use_head_wise_attn_gate
            and not self.use_dsa_backend
            and not self.fuse_qkv_gate
        ):
            self.g_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads,
                bias=False,
                quant_config=linear_quant_config,
                prefix=f"{prefix}.g_proj",
            )

        self.use_rope = bool(
            _per_layer_value(
                use_rope_layers,
                self.layer_idx,
                name="use_rope_layers",
                default=True,
            )
        )

        # Non-DSA layers use the same KV layout as DSA layers because vLLM
        # shares one raw allocation across KV cache groups.
        if self.use_dsa_backend:
            from .sparse_attention import Step4DSAAttentionBackend

            attn_backend = Step4DSAAttentionBackend
        elif model_has_dsa_layers:
            from .sparse_attention import Step4SplitKVFlashAttentionBackend

            attn_backend = Step4SplitKVFlashAttentionBackend
        else:
            attn_backend = None
        sparse_impl_args = {}
        if self.use_dsa_backend:
            dsa_vllm_config = typing.cast(VllmConfig, vllm_config)
            model_config = dsa_vllm_config.model_config
            scheduler_config = dsa_vllm_config.scheduler_config
            speculative_config = dsa_vllm_config.speculative_config
            sparse_impl_args = {
                "sparse_config": sparse_config,
                "max_model_len": int(model_config.max_model_len),
                "max_num_seqs": int(scheduler_config.max_num_seqs),
                "num_speculative_tokens": (
                    int(speculative_config.num_speculative_tokens)
                    if getattr(speculative_config, "method", None) == "mtp"
                    else 0
                ),
            }
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
            attn_backend=attn_backend,
            **sparse_impl_args,
        )
        if cache_config is not None and model_has_dsa_layers:
            # Step4 H200 cold-vs-prefix-hit parity requires at least one cache
            # page of prompt replay. This also applies when the sparse backend
            # is disabled: a DSA-capable checkpoint's dense fallback reproduces
            # the same short-tail divergence. Native dense Step4 variants do
            # not opt into this model-specific replay contract.
            self.attn.kv_cache_prefix_recompute_tokens = int(cache_config.block_size)
        if self.use_dsa_backend:
            from .sparse_summary_cache import Step4SparseSummaryCacheLayer

            sparse_config = typing.cast(Step4SparseConfig, sparse_config)
            proxy_dim = int(sparse_config.proxy_dim)
            indexer_num_heads = int(sparse_config.sparse_indexer_num_heads)
            self.sparse_indexer_w = Step4SparseIndexerIndexTPLinear(
                self.hidden_size,
                indexer_num_heads,
                params_dtype=self.params_dtype,
                index_tp_size=int(sparse_config.index_tp_size),
            )
            # The deployed indexer scores with weighted ReLU rather than a
            # scalable softmax, so this checkpoint tensor is intentionally
            # dormant. Keep it registered to make weight loading explicit
            # instead of silently dropping a model-owned tensor.
            self.ssmax_s = Parameter(
                torch.zeros(self.total_num_heads, dtype=torch.float32),
                requires_grad=False,
            )
            self.sparse_indexer_q_norm = RMSNormFactory(
                proxy_dim,
                eps=rms_norm_eps,
                zero_centered=self.zero_centered,
            )
            self.sparse_indexer_k_norm = OptimusLayerNorm(
                proxy_dim,
                eps=rms_norm_eps,
            )
            self.attn.kv_cache_requires_zeroing = True
            self.attn.impl.summary_cache_num_proxy_kv_heads = 1
            self.sparse_summary_cache = Step4SparseSummaryCacheLayer(
                prefix=f"{prefix}.attn.sparse_summary_cache",
                target_impl=self.attn.impl,
                sparse_config=sparse_config,
                main_layer_name=f"{prefix}.attn",
                static_forward_context=(
                    vllm_config.compilation_config.static_forward_context
                ),
            )
            self.attn.kv_cache_extra_budget_page_size_bytes = (
                self.sparse_summary_cache._set_summary_cache_budget
            )
            self.attn.kv_cache_extra_budget_fixed_size_bytes = (
                self.sparse_summary_cache._fixed_runtime_state_budget
            )
        else:
            self.sparse_summary_cache = None
        self.max_position_embeddings = max_position
        self.sparse_attn = (
            Step4SparseAttention(self, sparse_config) if self.use_dsa_backend else None
        )

        self.rotary_cache = self.rotary_emb.cos_sin_cache
        self.rope_cos, self.rope_sin = self.rotary_cache.chunk(2, dim=-1)
        self.use_optimus_qknorm = self.use_rope
        self.use_optimus_qknorm_cache = (
            self.use_rope
            and envs.VLLM_STEP_CC_LEVEL >= 1
            and current_platform.is_cuda()
            # STEP4_FLASH_ATTN is FlashAttention on the split KV layout, so the
            # fused cache kernel applies to it as well.
            and self.attn.attn_backend.get_name() in ("FLASH_ATTN", "STEP4_FLASH_ATTN")
            and not self.attn.attn_backend.forward_includes_kv_cache_update
            and self.attn.kv_sharing_target_layer_name is None
            and self.attn.attn_type == AttentionType.DECODER
            and self.attn.head_size_v == self.head_dim
            and self.attn.kv_cache_torch_dtype == self.attn.dtype
            and self.head_dim in (64, 128, 192, 256)
            and is_supported_optimus_qknorm_cache_rotary(
                self.head_dim,
                self.rotary_dim // 2,
            )
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        reduce_scatter_output: bool = False,
    ) -> torch.Tensor:
        # kv_cache_dummy_dep preserves ordering when the fused QKNorm path
        # writes KV before attention executes.
        if self.use_dsa_backend:
            qkv, qkzg = self.qkv_indexer_proj(hidden_states)
            extra_dims = None
        elif self.fuse_qkv_gate:
            qkvg, _ = self.qkvg_proj(hidden_states)
            qkv, extra_dims = qkvg.split(
                [self.q_size + 2 * self.kv_size, self.num_heads], dim=-1
            )
            # The fused projection may have a padded physical row stride.
            # Materialize both views before they cross compilation boundaries:
            # Optimus QKNorm requires contiguous QKV rows, and the gate must
            # not retain a logical stride that differs from the GEMM output.
            qkv = qkv.contiguous()
            extra_dims = extra_dims.contiguous()
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            qkzg = None
            extra_dims = None

        kv_cache_dummy_dep = None
        if self.use_optimus_qknorm_cache:
            eps = self.q_norm.variance_epsilon
            q, k, v, kv_cache_dummy_dep = fused_qknorm_rope_cache_forward_impl(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rope_cos,
                self.rope_sin,
                positions,
                self.head_dim,
                self.num_heads,
                self.num_kv_heads,
                self.rotary_dim // 2,
                self.attn.layer_name,
                eps,
                norm_weight_bias=1.0 if self.zero_centered else 0.0,
            )
        elif self.use_optimus_qknorm:
            eps = self.q_norm.variance_epsilon
            q, k, v = fused_qknorm_rope_forward_impl(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rope_cos,
                self.rope_sin,
                positions,
                self.head_dim,
                self.num_heads,
                self.num_kv_heads,
                self.rotary_dim // 2,
                eps,
                norm_weight_bias=1.0 if self.zero_centered else 0.0,
            )
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            # Add qk-norm inline similar to Qwen3 MOE attention
            q_by_head = q.view(
                *q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim
            )
            q_by_head = self.q_norm(q_by_head.contiguous())
            q = q_by_head.view(q.shape)

            k_by_head = k.view(
                *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
            )
            k_by_head = self.k_norm(k_by_head.contiguous())
            k = k_by_head.view(k.shape)
            if self.use_rope:
                q, k = self.rotary_emb(positions, q, k)

        if self.sparse_attn is not None:
            attn_output, extra_dims = self.sparse_attn.forward(
                positions, hidden_states, q, k, v, qkzg=qkzg
            )
        else:
            attn_output = self.attn(q, k, v, kv_cache_dummy_dep=kv_cache_dummy_dep)
        if extra_dims is None and self.use_head_wise_attn_gate:
            extra_dims, _ = self.g_proj(hidden_states)

        if extra_dims is not None:
            # Inductor can miscompile the fused head-wise gate -> o_proj path
            # for Step4 sparse models. Use an opaque materialization op so
            # the broadcasted gate multiply consumes stable BF16 buffers.
            attn_output = torch.ops.vllm.step4_materialize_gate_input(attn_output)
            extra_dims = torch.ops.vllm.step4_materialize_gate_input(extra_dims)

        if self.use_head_wise_attn_gate:
            output = (
                attn_output.view(*attn_output.shape[:-1], self.num_heads, self.head_dim)
                * extra_dims.unsqueeze(-1).sigmoid()
            )
            attn_output = output.view(*attn_output.shape)
        if reduce_scatter_output:
            output, _ = self.o_proj(
                attn_output, reduce_scatter_results=True, reduce_scatter_dim=0
            )
        else:
            output, _ = self.o_proj(attn_output)
        return output


class Step4SparseAttention:
    def __init__(
        self,
        owner: Step4Attention,
        sparse_config: Step4SparseConfig,
    ) -> None:
        self.owner = owner
        self.sparse_config = sparse_config
        self.sparse_rotary_emb = self._build_sparse_rotary_emb()
        owner.add_module("sparse_indexer_rotary_emb", self.sparse_rotary_emb)
        # cos/sin for the fused indexer norm+RoPE kernel are precomputed here
        # (outside forward) so the chunk/contiguous is not re-traced into the
        # compiled decode graph every step — mirroring how main attention builds
        # self.rope_cos/self.rope_sin in __init__.
        self._fused_indexer_ok: bool | None = None
        cos, sin = self.sparse_rotary_emb.cos_sin_cache.chunk(2, dim=-1)
        weight_dtype = owner.params_dtype
        self._sparse_rope_cos = cos.to(weight_dtype).contiguous()
        self._sparse_rope_sin = sin.to(weight_dtype).contiguous()

    def _get_dsa_order_token(self, device: torch.device) -> torch.Tensor | None:
        """Return the backend-owned ordering token when available.

        The model wrapper owns the DSA projection path, while the actual
        summary/CSA implementation (``attn.impl``) owns the shared scratch
        workspace.  Keep the lookup at this seam so eager and compiled paths
        use the same per-virtual-engine token without assuming every test
        backend provides Step4 scratch state.
        """
        impl = getattr(self.owner.attn, "impl", None)
        getter = getattr(impl, "_get_dsa_order_token", None)
        if getter is None:
            return None
        return getter(device)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        qkzg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        dsa_proxy_query, dsa_proxy_key, dsa_proxy_weights, dsa_proxy_z, extra_dims = (
            self._project_sparse_indexer(positions, hidden_states, qkzg)
        )
        attn_output = self._forward_dsa_attention(
            query,
            key,
            value,
            dsa_proxy_query=dsa_proxy_query,
            dsa_proxy_key=dsa_proxy_key,
            dsa_proxy_weights=dsa_proxy_weights,
            dsa_proxy_z=dsa_proxy_z,
        )
        return attn_output, extra_dims

    def _forward_dsa_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        dsa_proxy_query: torch.Tensor,
        dsa_proxy_key: torch.Tensor,
        dsa_proxy_weights: torch.Tensor,
        dsa_proxy_z: torch.Tensor,
    ) -> torch.Tensor:
        attn = self.owner.attn
        if attn.calculate_kv_scales:
            torch.ops.vllm.maybe_calc_kv_scales(query, key, value, attn.layer_name)
        hidden_size = attn.num_heads * attn.head_size_v
        output = torch.empty(
            (query.shape[0], hidden_size),
            dtype=query.dtype,
            device=query.device,
        )

        query = query.view(-1, attn.num_heads, attn.head_size)
        output = output.view(-1, attn.num_heads, attn.head_size_v)
        key = key.view(-1, attn.num_kv_heads, attn.head_size)
        value = value.view(-1, attn.num_kv_heads, attn.head_size_v)

        kv_cache_dummy_dep = None
        (
            dsa_summary_cache_sum,
            dsa_summary_cache_count,
            dsa_summary_cache_mean,
            dsa_csa_active_region_ids,
            dsa_csa_active_slot_by_region,
            dsa_csa_numerator,
            dsa_csa_denominator,
            dsa_csa_max,
            dsa_csa_active_token_k,
            dsa_csa_active_token_z,
            dsa_csa_active_token_valid,
            dsa_mtp_source_to_transaction,
            dsa_mtp_row_source,
            dsa_mtp_row_regions,
            dsa_mtp_row_positions,
            dsa_mtp_row_owner_block,
            dsa_mtp_row_owner_block_index,
            dsa_mtp_correction_action,
            dsa_mtp_state_numerator,
            dsa_mtp_state_denominator,
            dsa_mtp_state_max_logits,
            dsa_mtp_state_pre_numerator,
            dsa_mtp_state_pre_denominator,
            dsa_mtp_state_pre_max_logits,
            dsa_mtp_correction_free_slots,
            dsa_mtp_correction_free_count,
            dsa_mtp_correction_allocation_cursor,
            dsa_order_token,
        ) = self._dsa_summary_cache_side_effect_tensors()
        dsa_side_effect_tensors = (
            dsa_summary_cache_mean,
            dsa_csa_active_token_k,
            dsa_csa_active_token_z,
            dsa_csa_active_token_valid,
            dsa_mtp_source_to_transaction,
            dsa_mtp_row_source,
            dsa_mtp_row_regions,
            dsa_mtp_row_positions,
            dsa_mtp_row_owner_block,
            dsa_mtp_row_owner_block_index,
            dsa_mtp_correction_action,
            dsa_mtp_state_numerator,
            dsa_mtp_state_denominator,
            dsa_mtp_state_max_logits,
            dsa_mtp_state_pre_numerator,
            dsa_mtp_state_pre_denominator,
            dsa_mtp_state_pre_max_logits,
            dsa_mtp_correction_free_slots,
            dsa_mtp_correction_free_count,
            dsa_mtp_correction_allocation_cursor,
            dsa_order_token,
        )
        if attn.use_direct_call:
            if (
                not attn.attn_backend.forward_includes_kv_cache_update
                and attn.kv_sharing_target_layer_name is None
            ):
                kv_cache_dummy_dep = unified_kv_cache_update(
                    key, value, attn.layer_name
                )
            step4_dsa_attention_with_output(
                query,
                key,
                value,
                output,
                attn.layer_name,
                dsa_proxy_query,
                dsa_proxy_key,
                dsa_proxy_weights,
                dsa_proxy_z,
                dsa_summary_cache_sum,
                dsa_summary_cache_count,
                dsa_csa_active_region_ids,
                dsa_csa_active_slot_by_region,
                dsa_csa_numerator,
                dsa_csa_denominator,
                dsa_csa_max,
                kv_cache_dummy_dep,
                *dsa_side_effect_tensors,
            )
        else:
            if (
                not attn.attn_backend.forward_includes_kv_cache_update
                and attn.kv_sharing_target_layer_name is None
            ):
                kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
                    key, value, attn.layer_name
                )
            torch.ops.vllm.step4_dsa_attention_with_output(
                query,
                key,
                value,
                output,
                attn.layer_name,
                dsa_proxy_query,
                dsa_proxy_key,
                dsa_proxy_weights,
                dsa_proxy_z,
                dsa_summary_cache_sum,
                dsa_summary_cache_count,
                dsa_csa_active_region_ids,
                dsa_csa_active_slot_by_region,
                dsa_csa_numerator,
                dsa_csa_denominator,
                dsa_csa_max,
                kv_cache_dummy_dep,
                *dsa_side_effect_tensors,
            )
        return output.view(-1, hidden_size)

    def _dsa_summary_cache_side_effect_tensors(
        self,
    ) -> tuple[
        torch.Tensor | None,
        ...,
    ]:
        summary_cache = getattr(self.owner.attn.impl, "_summary_cache", None)
        if summary_cache is None:
            return (None,) * 28
        (
            csa_active_region_ids,
            csa_active_slot_by_region,
            csa_numerator,
            csa_denominator,
            csa_max,
        ) = self.owner.attn.impl._csa_summary_state(summary_cache)
        transaction = getattr(summary_cache, "_step4_mtp_transaction", None)
        return (
            summary_cache.sum_cache,
            summary_cache.count_cache,
            summary_cache.mean_cache,
            csa_active_region_ids,
            csa_active_slot_by_region,
            csa_numerator,
            csa_denominator,
            csa_max,
            getattr(summary_cache, "_step4_csa_active_token_k", None),
            getattr(summary_cache, "_step4_csa_active_token_z", None),
            getattr(summary_cache, "_step4_csa_active_token_valid", None),
            getattr(transaction, "source_to_transaction", None),
            getattr(transaction, "row_source", None),
            getattr(transaction, "row_regions", None),
            getattr(transaction, "row_positions", None),
            getattr(transaction, "row_owner_block", None),
            getattr(transaction, "row_owner_block_index", None),
            getattr(transaction, "correction_action", None),
            getattr(transaction, "state_numerator", None),
            getattr(transaction, "state_denominator", None),
            getattr(transaction, "state_max_logits", None),
            getattr(transaction, "state_pre_numerator", None),
            getattr(transaction, "state_pre_denominator", None),
            getattr(transaction, "state_pre_max_logits", None),
            getattr(transaction, "correction_free_slots", None),
            getattr(transaction, "correction_free_count", None),
            getattr(transaction, "correction_allocation_cursor", None),
            self._get_dsa_order_token(summary_cache.sum_cache.device),
        )

    def _sparse_indexer_rope_parameters(
        self,
        *,
        proxy_dim: int,
        rope_dim: int,
    ) -> dict[str, Any]:
        rope_parameters = dict(self.owner._sparse_indexer_base_rope_parameters)
        rope_type = str(rope_parameters.get("rope_type", "default")).strip().lower()
        if rope_type == "none":
            rope_parameters["rope_type"] = "default"
        rope_parameters["partial_rotary_factor"] = float(rope_dim) / float(proxy_dim)
        return rope_parameters

    def _build_sparse_rotary_emb(self) -> nn.Module:
        rope_dim = int(self.sparse_config.sparse_indexer_rope_dim)
        proxy_dim = int(self.sparse_config.proxy_dim)
        return get_rope(
            head_size=proxy_dim,
            max_position=self.owner.max_position_embeddings,
            rope_parameters=self._sparse_indexer_rope_parameters(
                proxy_dim=proxy_dim,
                rope_dim=rope_dim,
            ),
            dtype=self.owner.params_dtype,
        )

    def _apply_sparse_indexer_rope(
        self,
        positions: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sparse_rotary_emb = self.sparse_rotary_emb
        proxy_dim = int(index_q.shape[-1])
        positions = positions.reshape(-1).to(device=index_q.device, dtype=torch.long)
        q_shape = tuple(index_q.shape)
        k_shape = tuple(index_k.shape)
        q = index_q.reshape(q_shape[0], -1, proxy_dim)
        k = index_k.reshape(k_shape[0], -1, proxy_dim)
        q, k = sparse_rotary_emb(positions, q, k)
        k = typing.cast(torch.Tensor, k)
        return q.reshape(q_shape), k.reshape(k_shape)

    def _resolve_fused_indexer_support(self, proxy_dim: int) -> bool:
        """Resolve (once) whether the fused indexer norm+RoPE kernel applies.

        The kernel does per-head RMSNorm(q) + LayerNorm(k) + NeoX partial RoPE
        in a single launch, matching the eager sparse_indexer_q_norm (RMSNorm),
        sparse_indexer_k_norm (LayerNorm) and _apply_sparse_indexer_rope path.
        """
        cached = self._fused_indexer_ok
        if cached is not None:
            return cached
        if not envs.VLLM_STEP4_FUSE_INDEXER_NORM:
            # A/B switch: force the eager indexer norm+RoPE path.
            self._fused_indexer_ok = False
            return False
        rope_dim = int(self.sparse_config.sparse_indexer_rope_dim)
        interleaved = getattr(self.sparse_rotary_emb, "is_neox_style", True) is False
        ok = (
            not interleaved
            and rope_dim > 0
            and rope_dim % 8 == 0
            and rope_dim <= proxy_dim
            and proxy_dim % 8 == 0
        )
        self._fused_indexer_ok = ok
        return ok

    def _try_fused_indexer_norm_rope(
        self,
        positions: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        index_z: torch.Tensor,
        proxy_dim: int,
        num_q_heads: int,
        num_k_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        # z is a pure passthrough (no norm/rope); q/k/z are strided column-slices
        # of the fused qkv+indexer GEMM output.
        if not self._resolve_fused_indexer_support(proxy_dim):
            return None
        q_norm = self.owner.sparse_indexer_q_norm
        k_norm = self.owner.sparse_indexer_k_norm
        rope_dim = int(self.sparse_config.sparse_indexer_rope_dim)
        return fused_indexer_norm_rope_forward_impl(
            index_q,
            index_k,
            index_z,
            q_norm.weight,
            k_norm.weight,
            k_norm.bias,
            self._sparse_rope_cos,
            self._sparse_rope_sin,
            positions.reshape(-1),
            proxy_dim,
            num_q_heads,
            num_k_heads,
            rope_dim // 2,
            float(q_norm.variance_epsilon),
            1.0 if getattr(q_norm, "zero_centered", False) else 0.0,
        )

    def _project_sparse_indexer(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        qkzg: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        index_q, index_k, index_z, extra_dims = (
            self.owner.qkv_indexer_proj.split_indexer(qkzg)
        )
        weights, _ = self.owner.sparse_indexer_w(hidden_states)

        num_tokens = int(hidden_states.shape[0])
        proxy_dim = int(self.sparse_config.proxy_dim)
        q_width = int(index_q.shape[-1])
        k_width = int(index_k.shape[-1])
        num_q_heads = q_width // proxy_dim
        num_k_heads = k_width // proxy_dim
        num_local_kv_heads = int(self.owner.num_kv_heads)
        q_heads_per_kv = num_q_heads // num_local_kv_heads

        fused = self._try_fused_indexer_norm_rope(
            positions, index_q, index_k, index_z, proxy_dim, num_q_heads, num_k_heads
        )
        if fused is not None:
            # z is a contiguous passthrough emitted by the kernel; q/k are
            # normed+roped. This avoids the eager split/contiguous + rope ops.
            index_q, index_k, index_z = fused
            index_q = index_q.view(
                num_tokens, num_local_kv_heads, q_heads_per_kv, proxy_dim
            )
            index_k = index_k.view(num_tokens, num_k_heads, proxy_dim)
            index_z = index_z.view(num_tokens, num_k_heads, proxy_dim)
        else:
            index_k = self.owner.sparse_indexer_k_norm(index_k.contiguous())
            index_q = index_q.view(
                num_tokens,
                num_local_kv_heads,
                q_heads_per_kv,
                proxy_dim,
            ).contiguous()
            index_q = self.owner.sparse_indexer_q_norm(index_q)
            index_k = index_k.view(num_tokens, num_k_heads, proxy_dim).contiguous()
            index_q, index_k = self._apply_sparse_indexer_rope(
                positions, index_q, index_k
            )
            index_z = index_z.view(num_tokens, num_k_heads, proxy_dim).contiguous()

        weights = weights.view(
            num_tokens,
            num_local_kv_heads,
            q_heads_per_kv,
        )
        weights = weights.to(dtype=torch.float32)

        weights = weights * (float(q_heads_per_kv) ** -0.5)
        if self.owner.use_head_wise_attn_gate and extra_dims is None:
            raise RuntimeError("Sparse indexer gate output is missing.")
        return index_q, index_k, weights, index_z, extra_dims


class FusedMoEBlock(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        from vllm.model_executor.layers.fused_moe import FusedMoEFactory

        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer_idx = extract_layer_index(prefix)

        self.ep_size = get_ep_group().device_group.size()
        self.ep_rank = get_ep_group().device_group.rank()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        self.enable_eplb = parallel_config.enable_eplb
        self.n_routed_experts = config.moe_num_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        if self.tp_size > config.moe_num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.moe_num_experts}."
            )

        self.gate = FP32ReplicatedLinear(
            config.hidden_size,
            config.moe_num_experts,
            bias=False,
            quant_config=None,
            # params_dtype=torch.float32,  # Use FP32 for higher precision.
            prefix=f"{prefix}.gate",
        )
        self.use_moe_router_bias = config.use_moe_router_bias
        if not self.use_moe_router_bias:
            raise ValueError("Step4 MoE currently requires use_moe_router_bias=true.")
        self.routed_scaling_factor = config.moe_router_scaling_factor
        self.router_bias = nn.Parameter(
            torch.zeros(config.moe_num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.need_fp32_gate = config.need_fp32_gate
        if not self.need_fp32_gate:
            raise ValueError(
                "Step4 MoE requires need_fp32_gate=true for stable router logits."
            )

        activation = "silu"
        swiglu_limits = config.swiglu_limits or []
        swiglu_limit = (
            swiglu_limits[self.layer_idx]
            if self.layer_idx < len(swiglu_limits)
            else None
        )
        if swiglu_limit not in (None, 0):
            swiglu_limit = float(swiglu_limit)
            if swiglu_limit != 7.0:
                raise ValueError(
                    "Step4 fused MoE supports only swiglu_limit=7.0, got "
                    f"{swiglu_limit}."
                )
            activation = "swiglustep"
            logger.debug(
                "step4 layer_idx: %s, activation: %s, limit: %s",
                self.layer_idx,
                activation,
                swiglu_limit,
            )

        # CustomRoutingRouter does not forward the correction bias or routed
        # scaling factor to custom routing functions, so bind both here.
        # router_bias is loaded in place, making the captured Parameter stable.
        custom_routing_function = functools.partial(
            router_bias_func,
            router_bias=self.router_bias,
            routed_scaling_factor=config.moe_router_scaling_factor,
        )
        share_expert_dim = _pad_size_for_groupwise_quant(
            config.share_expert_dim, quant_config
        )
        moe_intermediate_size = _pad_size_for_groupwise_quant(
            config.moe_intermediate_size, quant_config
        )
        self.fuse_all_reduce, reduce_results = _step4_moe_reduce_policy(
            self.tp_size,
            get_dp_group().world_size,
        )
        effective_sequence_parallel = (
            vllm_config.compilation_config.pass_config.enable_sp and self.tp_size > 1
        )

        self.share_expert = Step4MLP(
            config=config,
            hidden_size=self.hidden_size,
            intermediate_size=share_expert_dim,
            hidden_act="silu",
            reduce_results=reduce_results,
            is_sequence_parallel=effective_sequence_parallel,
            quant_config=quant_config
            if quant_config and quant_config.get_name() != "fp8"
            else None,
            prefix=f"{prefix}.share_expert",
        )
        # Keep the shared expert outside FusedMoEFactory so Step4 can combine
        # shared and routed outputs in FP32 before the final all-reduce.
        kwargs = {"custom_routing_function": custom_routing_function}
        self.experts = FusedMoEFactory(
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            hidden_size=config.hidden_size,
            intermediate_size=moe_intermediate_size,
            reduce_results=reduce_results,
            renormalize=config.norm_expert_weight,
            quant_config=quant_config,
            activation=activation,
            prefix=f"{prefix}.experts",
            e_score_correction_bias=self.router_bias,
            routed_scaling_factor=config.moe_router_scaling_factor,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=effective_sequence_parallel,
            router_logits_dtype=torch.float32,
            **kwargs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_is_sequence_parallel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        # Sequence-parallel state is carried by the runner's FusedMoEConfig.
        if (
            self.experts.moe_config.is_sequence_parallel
            and not input_is_sequence_parallel
        ):
            hidden_states = sequence_parallel_chunk(hidden_states)

        shared_output = self.share_expert(hidden_states)

        if self.experts.is_internal_router:
            routed_output = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        else:
            # TODO(bnell): this gate could be moved into the MoERunner?
            router_logits, _ = self.gate(hidden_states)
            routed_output = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )

        # Kept separate so _forward_ffn can combine in fp32 and all-reduce after.
        return shared_output, routed_output


class Step4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.fp32_residual_connection = config.fp32_residual_connection
        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        # Step4 uses layer_types to decide which layers are SWA. Preserve
        # the cache-config window, then clear it only on this layer's copy;
        # otherwise a window supplied via --sliding-window is lost and every
        # layer is registered with a FullAttentionSpec.
        sliding_window = getattr(config, "sliding_window", None)
        if sliding_window is None:
            # Multimodal step4 keeps sliding_window under the nested text config;
            # hf_text_config resolves to it (or to hf_config when not nested).
            sliding_window = getattr(
                vllm_config.model_config.hf_text_config, "sliding_window", None
            )
        if sliding_window is None and cache_config is not None:
            sliding_window = cache_config.sliding_window
        if cache_config is not None:
            cache_config = copy.copy(cache_config)
            cache_config.sliding_window = None
        sparse_config = get_step4_sparse_config(config)
        checkpoint_has_dsa_layers = checkpoint_has_step4_sparse_config(config)
        use_dsa_for_layer = (
            sparse_config is not None
            and _is_step4_full_attention_layer(config, layer_idx, sparse_config)
        )
        if config.att_impl_type == "GQA":
            norm_dtype = get_norm_dtype(config)
            num_attention_heads = None
            num_attention_groups = None
            head_dim = None
            layer_types = _step_layer_types(config)
            if (
                getattr(config, "attention_other_setting", None)
                and layer_idx < len(layer_types)
                and layer_types[layer_idx]
                == config.attention_other_setting["attention_type"]
            ):
                num_attention_heads = config.attention_other_setting[
                    "num_attention_heads"
                ]
                num_attention_groups = config.attention_other_setting[
                    "num_attention_groups"
                ]
                head_dim = config.attention_other_setting["head_dim"]
            partial_rotary_factors = getattr(config, "partial_rotary_factors", [])
            partial_rotary_factor = float(
                _per_layer_value(
                    partial_rotary_factors,
                    layer_idx,
                    name="partial_rotary_factors",
                    default=1.0,
                )
            )
            max_position = getattr(config, "max_position_embeddings", None)
            if max_position is None:
                max_position = vllm_config.model_config.max_model_len
            self.self_attn = Step4Attention(
                hidden_size=self.hidden_size,
                num_heads=num_attention_heads
                if num_attention_heads
                else config.num_attention_heads,
                max_position=max_position,
                num_kv_heads=num_attention_groups
                if num_attention_groups
                else config.num_attention_groups,
                rope_theta=config.rope_theta,
                rms_norm_eps=config.rms_norm_eps,
                qkv_bias=getattr(config, "attention_bias", False),
                head_dim=head_dim if head_dim else getattr(config, "head_dim", None),
                cache_config=cache_config,
                quant_config=quant_config,
                rope_scaling=getattr(config, "rope_scaling", None),
                sliding_window=sliding_window,
                use_head_wise_attn_gate=getattr(
                    config, "use_head_wise_attn_gate", False
                ),
                layer_types=layer_types,
                use_rope_layers=getattr(config, "use_rope_layers", []),
                yarn_only_types=getattr(config, "yarn_only_types", []),
                swa_num_attention_heads=getattr(
                    config, "swa_num_attention_heads", None
                ),
                partial_rotary_factor=partial_rotary_factor,
                prefix=f"{prefix}.self_attn",
                zero_centered=config.zero_centered,
                vllm_config=vllm_config,
                sparse_config=sparse_config if use_dsa_for_layer else None,
                model_has_dsa_layers=checkpoint_has_dsa_layers,
                norm_dtype=norm_dtype,
            )
        else:
            raise ValueError(
                f"Unsupported attention implementation: {config.att_impl_type}"
            )
        self.use_moe = False
        self.tp_group = get_tp_group()
        self.use_fused_all_reduce = (
            get_tensor_model_parallel_world_size() > 1
            and get_dp_group().world_size == 1
        )
        if self.use_fused_all_reduce:
            logger.warning_once("Enable custom fused all reduce...")
        else:
            logger.warning_once("Disable custom fused all reduce...")

        moe_layers_idx = _get_step4_moe_layer_indices(config)
        if layer_idx in moe_layers_idx:
            self.moe = FusedMoEBlock(
                vllm_config,
                prefix=f"{prefix}.moe",
            )
            self.use_moe = True
        else:
            self.mlp = Step4MLP(
                config=config,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act="silu",
                quant_config=quant_config
                if quant_config and quant_config.get_name() != "fp8"
                else None,
                reduce_results=True,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNormFactory(
            config.hidden_size,
            eps=config.rms_norm_eps,
            zero_centered=config.zero_centered,
            dtype=norm_dtype,
        )
        self.post_attention_layernorm = RMSNormFactory(
            config.hidden_size,
            eps=config.rms_norm_eps,
            zero_centered=config.zero_centered,
            dtype=norm_dtype,
        )
        self.prefix = prefix
        self.use_attention_o_proj_reduce_scatter = (
            envs.VLLM_STEP4_O_PROJ_REDUCE_SCATTER
            and self.use_moe
            and self.moe.experts.moe_config.is_sequence_parallel
            and self.tp_group.world_size > 1
        )
        if self.use_attention_o_proj_reduce_scatter:
            logger.warning_once("Enable Step4 attention o_proj reduce-scatter path.")

    def add_and_maybe_inplace_all_reduce(
        self, in1: torch.Tensor, in2: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self._cast_for_residual(in1) + self._cast_for_residual(in2)
        if not self.use_fused_all_reduce:
            return hidden_states
        return self.tp_group.all_reduce(hidden_states)

    def _cast_for_param_op(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.fp32_residual_connection:
            return hidden_states
        return hidden_states.to(torch.bfloat16)

    def _cast_for_residual(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.fp32_residual_connection:
            return hidden_states
        return hidden_states.to(torch.float32)

    def _forward_ffn(
        self,
        hidden_states: torch.Tensor,
        input_is_sequence_parallel: bool = False,
        residual: torch.Tensor | None = None,
        orig_num_tokens: int | None = None,
    ) -> torch.Tensor:
        if self.use_moe:
            shared_output, moe_output = self.moe(
                hidden_states, input_is_sequence_parallel=input_is_sequence_parallel
            )
            if self.moe.experts.moe_config.is_sequence_parallel:
                if input_is_sequence_parallel:
                    assert residual is not None
                    assert orig_num_tokens is not None
                    ffn_output = self._cast_for_residual(
                        moe_output
                    ) + self._cast_for_residual(shared_output)
                    hidden_states = tensor_model_parallel_all_gather(
                        ffn_output + residual, dim=0
                    )
                    return hidden_states[:orig_num_tokens]
                ffn_output = tensor_model_parallel_all_gather(
                    self._cast_for_residual(moe_output)
                    + self._cast_for_residual(shared_output),
                    dim=0,
                )
                return ffn_output[: hidden_states.shape[0]]
            # Combine shared and routed expert outputs
            combined = self._cast_for_residual(moe_output) + self._cast_for_residual(
                shared_output
            )
            # When fuse_all_reduce=True, the runner does NOT
            # all-reduce (reduce_results=False), so we must all-reduce
            # the combined output here. When fuse_all_reduce=False,
            # routed output is either already reduced by the combine kernel or
            # reduced by _maybe_reduce_output. The shared expert path is a
            # separate RowParallelLinear, so DP/EP paths configure it to reduce
            # internally before it is combined with routed output.
            if self.moe.fuse_all_reduce:
                if self.use_fused_all_reduce:
                    combined = self.tp_group.all_reduce(combined)
                else:
                    combined = tensor_model_parallel_all_reduce(combined)
                return combined
            return combined
        return self.mlp(hidden_states)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        use_attention_o_proj_reduce_scatter = (
            self.use_attention_o_proj_reduce_scatter and hidden_states.dim() == 2
        )
        orig_num_tokens = hidden_states.shape[0]
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._cast_for_param_op(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            reduce_scatter_output=use_attention_o_proj_reduce_scatter,
        )
        hidden_states = self._cast_for_residual(hidden_states)
        if use_attention_o_proj_reduce_scatter:
            residual = sequence_parallel_chunk(residual)
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self._cast_for_param_op(hidden_states)

        ffn_output = self._forward_ffn(
            hidden_states,
            input_is_sequence_parallel=use_attention_o_proj_reduce_scatter,
            residual=residual if use_attention_o_proj_reduce_scatter else None,
            orig_num_tokens=(
                orig_num_tokens if use_attention_o_proj_reduce_scatter else None
            ),
        )
        if use_attention_o_proj_reduce_scatter:
            return ffn_output
        ffn_output = self._cast_for_residual(ffn_output)
        hidden_states = ffn_output + residual
        return hidden_states


@support_torch_compile
class Step4Model(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        self.vocab_size = config.vocab_size
        self.config = config
        self.fp32_residual_connection = config.fp32_residual_connection
        logger.info(
            "Step4 fp32_residual_connection: %s",
            self.fp32_residual_connection,
        )

        self.moe_num_experts = config.moe_num_experts
        self.parallel_config = vllm_config.parallel_config

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Step4DecoderLayer(
                vllm_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            norm_dtype = get_norm_dtype(config)
            self.norm = RMSNormFactory(
                config.hidden_size,
                eps=config.rms_norm_eps,
                zero_centered=config.zero_centered,
                dtype=norm_dtype,
            )
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _cast_for_residual(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.fp32_residual_connection:
            return hidden_states
        return hidden_states.to(torch.float32)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        hidden_states = self._cast_for_residual(hidden_states)
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                }
            )

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.layers.fused_moe import (
            fused_moe_make_expert_params_mapping,
        )
        from vllm.model_executor.model_loader.weight_utils import (
            default_weight_loader,
            maybe_remap_kv_scale_name,
        )

        config = self.config
        quant_config = self.vllm_config.quant_config
        if config.num_attention_groups <= 1:
            raise ValueError(
                "Step4 weight loading currently supports only GQA "
                "(num_attention_groups > 1)."
            )
        qkv_params_mapping = []
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkvg_proj", "q_proj", 0),
            ("qkvg_proj", "k_proj", 1),
            ("qkvg_proj", "v_proj", 2),
            ("qkvg_proj", "g_proj", 3),
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("qkv_indexer_proj", "q_proj", "q"),
            ("qkv_indexer_proj", "k_proj", "k"),
            ("qkv_indexer_proj", "v_proj", "v"),
            ("qkv_indexer_proj", "sparse_indexer_q", "index_q"),
            ("qkv_indexer_proj", "sparse_indexer_k", "index_k"),
            ("qkv_indexer_proj", "sparse_indexer_z", "index_z"),
            ("qkv_indexer_proj", "g_proj", "index_g"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        # Quantized expert wrappers insert base_layer into parameter names.
        base_layer = (
            "base_layer." if any(".base_layer." in name for name in params_dict) else ""
        )
        is_mxfp4_moe_quant = _is_mxfp4_moe_quant_config(quant_config)

        # Old packed 3D format: .moe.gate_proj.weight [num_experts, out, in]
        expert_params_mapping = [
            (
                f".moe.experts.routed_experts.{base_layer}w13_weight",
                ".moe.gate_proj.weight",
                "w1",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w13_weight",
                ".moe.up_proj.weight",
                "w3",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w2_weight",
                ".moe.down_proj.weight",
                "w2",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w13_weight_scale_2",
                ".moe.gate_proj.weight_scale_2",
                "w1",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w13_weight_scale_2",
                ".moe.up_proj.weight_scale_2",
                "w3",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w2_weight_scale_2",
                ".moe.down_proj.weight_scale_2",
                "w2",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w13_weight_scale",
                ".moe.gate_proj.weight_scale",
                "w1",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w13_weight_scale",
                ".moe.up_proj.weight_scale",
                "w3",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w2_weight_scale",
                ".moe.down_proj.weight_scale",
                "w2",
            ),
            # Required due to the Step3 HF model's packed expert format:
            # input scales are stored as moe.{gate,up,down}_proj.input_scale
            # rather than the standard per-expert format handled generically.
            (
                f".moe.experts.routed_experts.{base_layer}w13_input_scale",
                ".moe.gate_proj.input_scale",
                "w1",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w13_input_scale",
                ".moe.up_proj.input_scale",
                "w3",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w2_input_scale",
                ".moe.down_proj.input_scale",
                "w2",
            ),
        ]
        if is_mxfp4_moe_quant:
            expert_params_mapping = [
                (".moe.experts.w13_weight_scale", ".moe.gate_proj.weight_scale", "w1"),
                (".moe.experts.w13_weight_scale", ".moe.up_proj.weight_scale", "w3"),
                (".moe.experts.w2_weight_scale", ".moe.down_proj.weight_scale", "w2"),
                (".moe.experts.w13_weight", ".moe.gate_proj.weight", "w1"),
                (".moe.experts.w13_weight", ".moe.up_proj.weight", "w3"),
                (".moe.experts.w2_weight", ".moe.down_proj.weight", "w2"),
            ]

        is_groupwise_quant = (
            quant_config is not None and quant_config.get_name() == "groupwise_quant"
        )
        if is_groupwise_quant:
            expert_params_mapping = [
                (".moe.experts.w13_weight", ".moe.gate_proj.qweight", "w1"),
                (".moe.experts.w13_weight", ".moe.up_proj.qweight", "w3"),
                (".moe.experts.w2_weight", ".moe.down_proj.qweight", "w2"),
                (".moe.experts.w13_weight_scale", ".moe.gate_proj.scales", "w1"),
                (".moe.experts.w13_weight_scale", ".moe.up_proj.scales", "w3"),
                (".moe.experts.w2_weight_scale", ".moe.down_proj.scales", "w2"),
            ]

        # New per-expert format: .moe.experts.E.gate_proj.weight_packed [out, in]
        per_expert_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.moe_num_experts,
        )

        disable_moe_stacked_params = [data[1] for data in expert_params_mapping]

        def _as_mxfp4_param_dtype(
            param: torch.nn.Parameter, weight: torch.Tensor
        ) -> torch.Tensor:
            fp8_e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
            raw_dtypes = (torch.int8,)
            if fp8_e8m0_dtype is not None:
                raw_dtypes = raw_dtypes + (fp8_e8m0_dtype,)
            if param.dtype == torch.uint8 and weight.dtype in raw_dtypes:
                return weight.contiguous().view(torch.uint8)
            return weight

        def _load_mxfp4_weight(
            param: torch.nn.Parameter,
            weight: torch.Tensor,
            name: str,
            shard_id: str,
        ) -> None:
            mxfp4_block = 32
            use_ep = self.parallel_config.enable_expert_parallel
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            intermediate_size = self.config.moe_intermediate_size
            intermediate_size_block = intermediate_size // mxfp4_block
            per_rank_intermediate_size_block = math.ceil(
                intermediate_size_block / tp_size
            )
            per_rank_intermediate_size = per_rank_intermediate_size_block * mxfp4_block
            tp_rank_start = tp_rank * per_rank_intermediate_size
            tp_rank_end = min(
                (tp_rank + 1) * per_rank_intermediate_size, intermediate_size
            )

            if use_ep:
                ep_size = get_ep_group().world_size
                ep_rank = get_ep_group().rank
                experts_per_rank = self.config.moe_num_experts // ep_size
                expert_slice = slice(
                    ep_rank * experts_per_rank, (ep_rank + 1) * experts_per_rank
                )
            else:
                expert_slice = slice(None)

            if ".w13_weight_scale" in name:
                weight_slice = (
                    weight[expert_slice, ...]
                    if use_ep
                    else weight[:, tp_rank_start:tp_rank_end, ...]
                )
                weight_slice = _as_mxfp4_param_dtype(param, weight_slice)
                dest = param.data[: weight_slice.shape[0]]
                rows, cols = weight_slice.shape[1], weight_slice.shape[2]
                # MXFP4 backend layout conversion expects w13 as gate/up
                # row pairs, not contiguous gate and up blocks.
                if shard_id == "w1":
                    dest[:, : 2 * rows : 2, :cols].copy_(weight_slice)
                elif shard_id == "w3":
                    dest[:, 1 : 2 * rows : 2, :cols].copy_(weight_slice)
                else:
                    dest[:, :rows, :cols].copy_(weight_slice)
                return

            if ".w2_weight_scale" in name:
                start = tp_rank_start // mxfp4_block
                end = tp_rank_end // mxfp4_block
                weight_slice = (
                    weight[expert_slice, ...] if use_ep else weight[..., start:end]
                )
                weight_slice = _as_mxfp4_param_dtype(param, weight_slice)
                dest = param.data[: weight_slice.shape[0]]
                rows, cols = weight_slice.shape[1], weight_slice.shape[2]
                dest[:, :rows, :cols].copy_(weight_slice)
                return

            if ".w13_weight" in name:
                weight_slice = (
                    weight[expert_slice, ...]
                    if use_ep
                    else weight[:, tp_rank_start:tp_rank_end, ...]
                )
                weight_slice = _as_mxfp4_param_dtype(param, weight_slice)
                dest = param.data[: weight_slice.shape[0]]
                rows, cols = weight_slice.shape[1], weight_slice.shape[2]
                # MXFP4 backend layout conversion expects w13 as gate/up
                # row pairs, not contiguous gate and up blocks.
                if shard_id == "w1":
                    dest[:, : 2 * rows : 2, :cols].copy_(weight_slice)
                elif shard_id == "w3":
                    dest[:, 1 : 2 * rows : 2, :cols].copy_(weight_slice)
                else:
                    dest[:, :rows, :cols].copy_(weight_slice)
                return

            if ".w2_weight" in name:
                start = tp_rank_start // 2
                end = tp_rank_end // 2
                weight_slice = (
                    weight[expert_slice, ...] if use_ep else weight[..., start:end]
                )
                weight_slice = _as_mxfp4_param_dtype(param, weight_slice)
                dest = param.data[: weight_slice.shape[0]]
                rows, cols = weight_slice.shape[1], weight_slice.shape[2]
                dest[:, :rows, :cols].copy_(weight_slice)

        import threading

        # for name, loaded_weight in weights:
        def _worker(name, loaded_weight):
            loaded_params: set[str] = set()
            if name.startswith("model."):
                local_name = name[len("model.") :]
                full_name = name
            else:
                local_name = name
                full_name = f"model.{name}" if name else "model"

            spec_layer = get_spec_layer_idx_from_weight_name(config, full_name)
            if spec_layer is not None:
                # continue  # skip spec decode layers for main model
                return loaded_params

            # Skip any layers beyond the main model's depth (e.g., MTP layers)
            if full_name.startswith("model.layers."):
                parts = full_name.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    layer_idx = int(parts[2])
                    if layer_idx >= config.num_hidden_layers:
                        return loaded_params

            remapped_name = maybe_remap_kv_scale_name(local_name, params_dict)
            if remapped_name is None:
                return loaded_params
            local_name = remapped_name

            # Per-expert MoE weights (new format from LLM Compressor):
            # .moe.experts.{E}.{gate,up,down}_proj.{weight_packed,scale,...}
            # Each weight is individual per-expert, not stacked 3D.
            if ".moe.experts." in local_name:
                is_expert_weight = False
                for mapping in per_expert_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in local_name:
                        continue
                    is_expert_weight = True
                    name_mapped = local_name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if name_mapped not in params_dict:
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    loaded_weight_padded = pad_param(
                        loaded_weight,
                        name_mapped,
                        param,
                        quant_config,
                    )
                    success = weight_loader(
                        param,
                        loaded_weight_padded,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(name_mapped)
                        break
                else:
                    if (
                        not is_expert_weight
                        and not is_pp_missing_parameter(local_name, self)
                        and local_name in params_dict
                    ):
                        # Not an expert proj — use default loader
                        # (e.g. share_expert weights if they matched)
                        param = params_dict[local_name]
                        weight_loader = getattr(
                            param,
                            "weight_loader",
                            default_weight_loader,
                        )
                        loaded_weight_padded = pad_param(
                            loaded_weight,
                            local_name,
                            param,
                            quant_config,
                        )
                        weight_loader(param, loaded_weight_padded)
                        loaded_params.add(local_name)
                return loaded_params

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in local_name:
                    continue
                if any(
                    disable_moe_stacked_param in local_name
                    for disable_moe_stacked_param in disable_moe_stacked_params
                ):
                    continue
                replaced_name = local_name.replace(weight_name, param_name)
                if is_pp_missing_parameter(replaced_name, self):
                    continue
                if replaced_name not in params_dict:
                    continue
                param = params_dict[replaced_name]
                weight_loader = param.weight_loader
                loaded_weight_padded = pad_param(
                    loaded_weight,
                    replaced_name,
                    param,
                    quant_config,
                )
                weight_loader(param, loaded_weight_padded, shard_id)
                loaded_params.add(replaced_name)
                break
            else:
                for param_name, weight_name, shard_id in expert_params_mapping:
                    if weight_name not in local_name:
                        continue
                    replaced_name = local_name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(replaced_name, self):
                        continue
                    if (
                        replaced_name.endswith(".bias")
                        or replaced_name.endswith("_bias")
                    ) and replaced_name not in params_dict:
                        continue
                    if replaced_name not in params_dict:
                        continue
                    param = params_dict[replaced_name]
                    weight_loader = param.weight_loader
                    moe_expert_num = self.moe_num_experts
                    if is_mxfp4_moe_quant:
                        _load_mxfp4_weight(
                            param, loaded_weight, replaced_name, shard_id
                        )
                        loaded_params.add(replaced_name)
                        break
                    # Per-tensor global scales (e.g. weight_global_scale)
                    # have shape [1] in compressed-tensors NVFP4 checkpoints.
                    # Expand to per-expert before the iteration loop.
                    if loaded_weight.ndim == 0:
                        loaded_weight = loaded_weight.unsqueeze(0).expand(
                            moe_expert_num
                        )
                    elif (
                        loaded_weight.shape[0] == 1
                        and loaded_weight.shape[0] != moe_expert_num
                    ):
                        loaded_weight = loaded_weight.expand(
                            moe_expert_num, *loaded_weight.shape[1:]
                        )
                    assert loaded_weight.shape[0] == moe_expert_num
                    for expert_id in range(moe_expert_num):
                        loaded_weight_expert = pad_param(
                            loaded_weight[expert_id],
                            replaced_name,
                            param,
                            quant_config,
                        )
                        weight_loader(
                            param,
                            loaded_weight_expert,
                            replaced_name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    loaded_params.add(replaced_name)
                    break
                else:
                    for (
                        param_name,
                        weight_name,
                        start_idx,
                        end_idx,
                    ) in qkv_params_mapping:
                        if weight_name not in local_name:
                            continue
                        replaced_name = local_name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(replaced_name, self):
                            continue
                        if replaced_name not in params_dict:
                            continue
                        param = params_dict[replaced_name]
                        dim = param.shape[param.output_dim]
                        begin_idx = int(start_idx * dim)
                        end_idx = int(end_idx * dim)
                        param_slice = param.narrow(
                            param.output_dim, begin_idx, end_idx - begin_idx
                        )
                        param_slice.copy_(loaded_weight)
                        loaded_params.add(replaced_name)
                        break
                    else:
                        if is_pp_missing_parameter(local_name, self):
                            return loaded_params
                        if "expert_bias" in local_name:
                            logger.warning_once("ignore expert_bias")
                            return loaded_params
                        if local_name not in params_dict:
                            return loaded_params
                        param = params_dict[local_name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        try:
                            loaded_weight_padded = pad_param(
                                loaded_weight,
                                local_name,
                                param,
                                quant_config,
                            )
                            weight_loader(param, loaded_weight_padded)
                        except Exception as e:
                            logger.error(
                                "shape: %s, param shape: %s, %s",
                                loaded_weight.shape,
                                param.shape,
                                local_name,
                            )
                            raise e

                        loaded_params.add(local_name)
            return loaded_params

        worker_num = 8
        logger.info("Loading weights by %s workers... %s", worker_num, type(weights))

        # Limited concurrency to make tqdm happy.
        throttle = threading.BoundedSemaphore(worker_num)
        futures = []
        with ThreadPoolExecutor(worker_num) as executor:
            for name, loaded_weight in weights:
                throttle.acquire()
                futures.append(executor.submit(_worker, name, loaded_weight))
                futures[-1].add_done_callback(lambda _: throttle.release())
        for future in as_completed(futures):
            loaded_params |= future.result()
        _mark_optional_fp8_attention_scales_loaded(loaded_params, params_dict)
        return loaded_params


class Step4ForCausalLM(nn.Module, SupportsPP, MixtureOfExperts):
    # Required so quantization exclude lists match fused module prefixes.
    packed_modules_mapping = STEP4_PACKED_MODULES_MAPPING
    # The custom loader accounts for optional/online FP8 parameters, so a
    # quantized Step4 checkpoint must retain the default missing-weight check.
    _enable_weights_track_by_default = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_regex={
            re.compile(r"^vit_large_projector\.weight$"): None,
        },
        orig_to_new_substr={".share_expert.": ".moe.share_expert."},
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        if not current_platform.is_cuda():
            raise NotImplementedError(
                "Step4 currently requires the CUDA/Optimus backend; no "
                "Ascend/CPU model adapter is registered for this architecture."
            )
        self.vllm_config = vllm_config
        model_config = vllm_config.model_config
        valid_vocab_size = _require_resolved_valid_vocab_size(model_config)
        config = model_config.hf_config
        self.config = config
        self.fp32_residual_connection = config.fp32_residual_connection
        self.model = Step4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config
                if vllm_config.quant_config
                and vllm_config.quant_config.get_name() != "fp8"
                else None,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
            self.logits_processor = LogitsProcessor(
                config.vocab_size,
                valid_vocab_size=valid_vocab_size,
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.moe_layers: list[Any] = []
        example_layer: FusedMoEBlock | None = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            assert isinstance(layer, Step4DecoderLayer)
            if hasattr(layer, "moe") and isinstance(layer.moe, FusedMoEBlock):
                example_layer = layer.moe
                self.moe_layers.append(layer.moe.experts)

        _set_step4_moe_protocol_metadata(self, example_layer)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.fp32_residual_connection:
            hidden_states = hidden_states.to(torch.bfloat16)
        hidden_states = self.model.norm(hidden_states)
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        if self.num_local_physical_experts != num_local_physical_experts:
            raise ValueError(
                "Step4 EPLB cannot change the number of local physical experts: "
                f"expected={self.num_local_physical_experts}, "
                f"got={num_local_physical_experts}."
            )
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if not isinstance(layer, Step4DecoderLayer):
                continue
            moe = getattr(layer, "moe", None)
            if not isinstance(moe, FusedMoEBlock):
                continue
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.model_loader.mtp_validation import (
            is_mtp_completeness_check_enabled,
        )

        validate_completeness = is_mtp_completeness_check_enabled()
        if validate_completeness:
            _reset_fused_qkv_indexer_load_state(self.model)
        skip_prefixes = ["vision_model."]
        if self.config.tie_word_embeddings:
            skip_prefixes.append("lm_head.")
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes,
        )
        loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        if validate_completeness:
            loaded_params.update(
                f"model.{name}"
                for name in _validate_fused_qkv_indexer_weights(self.model)
            )
        return loaded_params
