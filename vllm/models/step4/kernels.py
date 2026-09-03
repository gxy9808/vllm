# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from dataclasses import dataclass, fields
from functools import cache
from typing import Any

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


@dataclass(frozen=True)
class Step4SparseConfig:
    enabled: bool = True
    proxy_dim: int = 256
    sparse_indexer_rope_dim: int = 32
    sparse_indexer_use_rope: bool = True
    sparse_indexer_num_heads: int = 16
    sparse_indexer_num_k_heads: int = 1
    sparse_indexer_q_norm_type: str = "rmsnorm"
    sparse_indexer_k_norm_type: str = "layernorm"
    sparse_indexer_csa_z_norm_type: str = "none"
    index_tp_size: int = 4
    topk: int = 512
    region_block_size: int = 8
    compression_method: str = "csa_block_compress"
    attention_impl: str = "sparse_gqa"
    decode_split_max: int = 16
    sparse_indexer_softmax_variant: str = "softmax"
    sparse_indexer_ssmax_s_granularity: str = "q_head"
    apply_to_layer_types: tuple[str, ...] = ("full_attention",)


_STEP4_SPARSE_SECTION_NAMES = (
    "step4_sparse_config",
    "step3p5_sparse_config",
    "sparse_config",
)
_STEP4_SPARSE_ENABLE_ENV = "VLLM_STEP4_SPARSE"
_STEP4_SPARSE_ENV = {
    "proxy_dim": "VLLM_STEP4_SPARSE_PROXY_DIM",
    "sparse_indexer_rope_dim": "VLLM_STEP4_SPARSE_INDEXER_ROPE_DIM",
    "index_tp_size": "VLLM_STEP4_DSA_INDEX_TP_SIZE",
    "topk": "VLLM_STEP4_SPARSE_TOPK",
    "region_block_size": "VLLM_STEP4_SPARSE_REGION_BLOCK_SIZE",
    "attention_impl": "VLLM_STEP4_SPARSE_ATTENTION_IMPL",
    "decode_split_max": "VLLM_STEP4_SPARSE_DECODE_SPLIT_MAX",
}

_SUPPORTED_SOFTMAX_VARIANTS = {"softmax", "standard", "ssmax"}
_SUPPORTED_INDEXER_Q_HEADS_PER_PROVIDER = (2, 4)
_SUPPORTED_DECODE_SPLIT_MAX = (1, 2, 4, 16)
_STEP4_SPARSE_PROXY_DIM = 256
_STEP4_SPARSE_REGION_BLOCK_SIZE = 8
_STEP4_SPARSE_TOPK_MAX = 1024
# The stable CuTeDSL selector currently uses a fixed 512-entry shared-memory
# selection buffer. Keep config validation aligned with that implementation;
# allowing larger values would defer the failure until warmup/first request.
_STEP4_SPARSE_STABLE_TOPK_MAX = 512
_STEP4_SPARSE_FIXED_OPTIONS = {
    "sparse_indexer_use_rope": True,
    "sparse_indexer_q_norm_type": "rmsnorm",
    "sparse_indexer_k_norm_type": "layernorm",
    "sparse_indexer_csa_z_norm_type": "none",
    "compression_method": "csa_block_compress",
    "attention_impl": "sparse_gqa",
    "sparse_indexer_ssmax_s_granularity": "q_head",
}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _config_value(config: Any, name: str) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _get_sparse_config_section(config: Any) -> Any:
    for name in _STEP4_SPARSE_SECTION_NAMES:
        value = _config_value(config, name)
        if value is not None:
            return value
    return None


def checkpoint_has_step4_sparse_config(config: Any) -> bool:
    """Return whether the checkpoint declares the Step4 sparse layout.

    The runtime ``VLLM_STEP4_SPARSE`` switch controls whether DSA kernels are
    enabled, but it must not erase model capabilities that affect shared cache
    layout and replay contracts.  Presence of a recognized sparse section is
    therefore the structural capability bit, including a section serialized
    with ``enabled: false``. Native dense checkpoints without a sparse section
    stay out of the model-specific contract.
    """
    section = _get_sparse_config_section(config)
    return section is not None


def get_step4_sparse_config(config: Any) -> Step4SparseConfig | None:
    defaults = Step4SparseConfig()
    section = _get_sparse_config_section(config)
    if envs.is_set(_STEP4_SPARSE_ENABLE_ENV):
        enabled = envs.VLLM_STEP4_SPARSE
        if enabled and section is None:
            raise ValueError(
                "VLLM_STEP4_SPARSE=1 requires a checkpoint-declared "
                "Step4 sparse config section; refusing to apply DSA defaults "
                "to a native dense checkpoint."
            )
    elif section is None:
        enabled = False
    else:
        # The model config may carry a sparse_config section with
        # ``enabled: false``.  Preserve that switch, while still allowing the
        # environment variable above to force-enable/disable the path.
        enabled = _truthy(_config_value(section, "enabled"))
        if _config_value(section, "enabled") is None:
            enabled = True
    if not enabled:
        return None
    softmax_variant = _config_value(
        section, "sparse_indexer_softmax_variant"
    ) or _config_value(section, "softmax_variant")
    if softmax_variant is not None and str(softmax_variant).lower() not in (
        _SUPPORTED_SOFTMAX_VARIANTS
    ):
        raise ValueError(
            "Step4 sparse indexer softmax variant is unsupported by the "
            f"current kernels: {softmax_variant!r}. Supported variants are "
            f"{sorted(_SUPPORTED_SOFTMAX_VARIANTS)}."
        )
    if softmax_variant is not None and str(softmax_variant).lower() == "ssmax":
        logger.warning_once(
            "Step4 sparse config carries ssmax metadata, but the deployed DSA "
            "selector uses its fixed weighted-ReLU scoring path; ssmax_s is "
            "loaded for checkpoint completeness but is not consumed."
        )
    values: dict[str, Any] = {}
    for field in fields(Step4SparseConfig):
        default = getattr(defaults, field.name)
        if field.name == "apply_to_layer_types":
            continue
        env_name = _STEP4_SPARSE_ENV.get(field.name)
        value = (
            getattr(envs, env_name)
            if env_name is not None and envs.is_set(env_name)
            else None
        )
        if value is None:
            value = _config_value(section, field.name)
        if field.name == "sparse_indexer_softmax_variant" and value is None:
            value = _config_value(section, "softmax_variant")
        if field.name == "index_tp_size" and value is None:
            # The checkpoint schema calls this field num_provider_groups.
            value = _config_value(section, "num_provider_groups")
        if value is None or value == "":
            value = default
        elif isinstance(default, bool):
            value = _truthy(value)
        elif isinstance(default, int):
            value = int(value)
        elif isinstance(default, str):
            value = str(value)
        values[field.name] = value
    # Reconcile the serialized field with the enable switch resolved above.
    # Reaching this point means DSA is enabled, including an environment
    # force-enable over ``section.enabled=false``.
    values["enabled"] = True
    apply_to_layer_types = _config_value(section, "apply_to_layer_types")
    if apply_to_layer_types is None:
        apply_to_layer_types = defaults.apply_to_layer_types
    if isinstance(apply_to_layer_types, str):
        apply_to_layer_types = tuple(
            item.strip().lower()
            for item in apply_to_layer_types.split(",")
            if item.strip()
        )
    else:
        apply_to_layer_types = tuple(
            str(item).strip().lower()
            for item in apply_to_layer_types
            if str(item).strip()
        )
    if apply_to_layer_types != ("full_attention",):
        raise ValueError(
            "Step4 DSA production integration requires "
            "apply_to_layer_types=('full_attention',); sliding, empty, or "
            f"mixed layer selections are unsupported, got {apply_to_layer_types!r}."
        )
    values["apply_to_layer_types"] = apply_to_layer_types
    proxy_dim = int(values["proxy_dim"])
    if proxy_dim != _STEP4_SPARSE_PROXY_DIM:
        raise ValueError(
            f"Step4 DSA production kernels require proxy_dim=256, got {proxy_dim}."
        )
    region_block_size = int(values["region_block_size"])
    if region_block_size != _STEP4_SPARSE_REGION_BLOCK_SIZE:
        raise ValueError(
            "Step4 DSA production kernels require region_block_size=8, got "
            f"{region_block_size}."
        )
    topk = int(values["topk"])
    if not 1 <= topk <= _STEP4_SPARSE_TOPK_MAX:
        raise ValueError(
            f"Step4 DSA production kernels require topk in [1, 1024], got {topk}."
        )
    if envs.VLLM_STEP4_DSA_FORCE_STABLE_TOPK and topk > _STEP4_SPARSE_STABLE_TOPK_MAX:
        raise ValueError(
            "Step4 DSA stable top-k currently requires topk in [1, 512] "
            "because the CuTeDSL stable selector supports at most 512 "
            f"entries; got topk={topk}."
        )
    indexer_num_k_heads = int(values["sparse_indexer_num_k_heads"])
    if indexer_num_k_heads != 1:
        raise ValueError(
            "Step4 sparse indexer currently requires exactly one KV head, got "
            f"sparse_indexer_num_k_heads={indexer_num_k_heads}."
        )
    index_tp_size = int(values["index_tp_size"])
    if index_tp_size <= 0:
        raise ValueError(
            f"Step4 sparse indexer index_tp_size must be positive, got {index_tp_size}."
        )
    indexer_num_heads = int(values["sparse_indexer_num_heads"])
    if indexer_num_heads <= 0 or indexer_num_heads % index_tp_size != 0:
        raise ValueError(
            "Step4 sparse_indexer_num_heads must be positive and divisible by "
            "index_tp_size, got "
            f"sparse_indexer_num_heads={indexer_num_heads}, "
            f"index_tp_size={index_tp_size}."
        )
    indexer_q_heads_per_provider = indexer_num_heads // index_tp_size
    if indexer_q_heads_per_provider not in _SUPPORTED_INDEXER_Q_HEADS_PER_PROVIDER:
        raise ValueError(
            "Step4 DSA production indexer logits kernels require "
            "sparse_indexer_num_heads / index_tp_size in (2, 4), got "
            f"{indexer_q_heads_per_provider} "
            f"({indexer_num_heads} / {index_tp_size})."
        )
    indexer_rope_dim = int(values["sparse_indexer_rope_dim"])
    if (
        indexer_rope_dim <= 0
        or indexer_rope_dim > proxy_dim
        or indexer_rope_dim % 2 != 0
    ):
        raise ValueError(
            "Step4 sparse indexer RoPE dimension must be positive, even, and "
            "no larger than proxy_dim, got "
            f"sparse_indexer_rope_dim={indexer_rope_dim}, "
            f"proxy_dim={proxy_dim}."
        )
    for name, expected in _STEP4_SPARSE_FIXED_OPTIONS.items():
        actual = values[name]
        normalized = (
            bool(actual) if isinstance(expected, bool) else str(actual).strip().lower()
        )
        if normalized != expected:
            raise ValueError(
                f"Step4 DSA production kernels require {name}={expected!r}, "
                f"got {actual!r}."
            )
        values[name] = expected
    decode_split_max = int(values["decode_split_max"])
    if decode_split_max not in _SUPPORTED_DECODE_SPLIT_MAX:
        raise ValueError(
            "Step4 DSA decode_split_max must be one of {1, 2, 4, 16}, got "
            f"{decode_split_max}."
        )
    return Step4SparseConfig(**values)


def is_supported_optimus_qknorm_cache_rotary(
    head_dim: int,
    rotary_pairs: int,
) -> bool:
    """Whether the native QKNorm+RoPE+cache kernel supports this RoPE shape."""
    if head_dim not in (64, 128, 192, 256):
        return False
    return 0 <= rotary_pairs <= head_dim // 2 and rotary_pairs % 4 == 0


def _get_stepfun_qknorm_cache_op() -> Any | None:
    """Return the selected native cache op when it was built into vLLM."""
    op_name = (
        "optimus_fused_qknorm_rope_cache"
        if envs.VLLM_STEP_CC_LEVEL == 3
        else "optimus_fused_qknorm_rope_cache_bitwise"
    )
    namespace = getattr(torch.ops, "_C", None)
    if namespace is None or not hasattr(namespace, op_name):
        return None
    return getattr(namespace, op_name)


@cache
def _get_step4_cute_dsl():
    return importlib.import_module("vllm.models.step4.nvidia.ops.cute_dsl")


def _fused_qknorm_rope_forward_impl(*args: Any, **kwargs: Any):
    return _get_step4_cute_dsl().fused_qknorm_rope_forward_impl(*args, **kwargs)


def _fused_indexer_norm_rope_forward_impl(*args: Any, **kwargs: Any):
    return _get_step4_cute_dsl().fused_indexer_norm_rope_forward_impl(*args, **kwargs)


def _fused_qknorm_rope_forward_impl_fake(
    qkv: torch.Tensor,
    qnorm_weight: torch.Tensor,
    knorm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_id: torch.Tensor,
    head_dim: int,
    num_q_head: int,
    num_kv_head: int,
    rotary_dim: int,
    eps: float = 1e-5,
    norm_weight_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = qkv.shape[0]
    q = torch.empty((tokens, num_q_head * head_dim), device=qkv.device, dtype=qkv.dtype)
    k = torch.empty(
        (tokens, num_kv_head * head_dim), device=qkv.device, dtype=qkv.dtype
    )
    v = torch.empty(
        (tokens, num_kv_head * head_dim), device=qkv.device, dtype=qkv.dtype
    )
    return q, k, v


def _fused_qknorm_rope_forward_impl_op(
    qkv: torch.Tensor,
    qnorm_weight: torch.Tensor,
    knorm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_id: torch.Tensor,
    head_dim: int,
    num_q_head: int,
    num_kv_head: int,
    rotary_dim: int,
    eps: float = 1e-5,
    norm_weight_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _fused_qknorm_rope_forward_impl(
        qkv,
        qnorm_weight,
        knorm_weight,
        cos,
        sin,
        pos_id,
        head_dim,
        num_q_head,
        num_kv_head,
        rotary_dim,
        eps,
        norm_weight_bias,
    )


direct_register_custom_op(
    op_name="fused_qknorm_rope_forward_impl",
    op_func=_fused_qknorm_rope_forward_impl_op,
    mutates_args=[],
    fake_impl=_fused_qknorm_rope_forward_impl_fake,
)


def fused_qknorm_rope_forward_impl(
    qkv: torch.Tensor,
    qnorm_weight: torch.Tensor,
    knorm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_id: torch.Tensor,
    head_dim: int,
    num_q_head: int,
    num_kv_head: int,
    rotary_dim: int,
    eps: float = 1e-5,
    norm_weight_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Keep the same Python signature, but route to a custom op so
    # torch.compile(fullgraph=True) can use the fake_impl for tracing.
    return torch.ops.vllm.fused_qknorm_rope_forward_impl(
        qkv,
        qnorm_weight,
        knorm_weight,
        cos,
        sin,
        pos_id,
        head_dim,
        num_q_head,
        num_kv_head,
        rotary_dim,
        eps,
        norm_weight_bias,
    )


def _fused_qknorm_rope_cache_forward_impl_fake(
    qkv: torch.Tensor,
    qnorm_weight: torch.Tensor,
    knorm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_id: torch.Tensor,
    head_dim: int,
    num_q_head: int,
    num_kv_head: int,
    rotary_dim: int,
    layer_name: str,
    eps: float = 1e-5,
    norm_weight_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = _fused_qknorm_rope_forward_impl_fake(
        qkv,
        qnorm_weight,
        knorm_weight,
        cos,
        sin,
        pos_id,
        head_dim,
        num_q_head,
        num_kv_head,
        rotary_dim,
        eps,
        norm_weight_bias,
    )
    dummy = torch.empty(0, device=qkv.device, dtype=qkv.dtype)
    return q, k, v, dummy


def _fused_qknorm_rope_cache_forward_impl_op(
    qkv: torch.Tensor,
    qnorm_weight: torch.Tensor,
    knorm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_id: torch.Tensor,
    head_dim: int,
    num_q_head: int,
    num_kv_head: int,
    rotary_dim: int,
    layer_name: str,
    eps: float = 1e-5,
    norm_weight_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Import lazily: attention.py imports model definitions during startup.
    from vllm.model_executor.layers.attention.attention import get_attention_context

    _, attn_layer, kv_cache, slot_mapping = get_attention_context(layer_name)
    if slot_mapping is None:
        # The QKV view can come from a wider fused QKVG projection.  Optimus
        # CutDSL requires aligned rows, while that view's token stride is not
        # necessarily 16-byte aligned (for example, 1804 bf16 elements).
        # This path is used before KV caches are initialized; the native fused
        # cache path below supports the wider row stride directly.
        q, k, v = _fused_qknorm_rope_forward_impl(
            qkv.contiguous(),
            qnorm_weight,
            knorm_weight,
            cos,
            sin,
            pos_id,
            head_dim,
            num_q_head,
            num_kv_head,
            rotary_dim,
            eps,
            norm_weight_bias,
        )
    else:
        # With DSA enabled every attention layer uses the split
        # (2, num_blocks, block_size, heads, dim) layout because
        # Step4's CuTeDSL sparse-GQA kernels require each half dense row-major
        # and vLLM shares one raw allocation across KV cache groups (see
        # Step4SplitKVFlashAttentionBackend). Without DSA, FlashAttention packs
        # K and V into the content dim as
        # (num_blocks, heads, block_size, 2 * dim). The
        # fused cache kernel below reads the cache strides instead of assuming
        # contiguity, so a transposed split view of the packed layout works
        # directly.
        if kv_cache.dim() == 5:
            key_cache, value_cache = kv_cache.unbind(0)
        else:
            key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)
        qknorm_cache_op = _get_stepfun_qknorm_cache_op()
        use_native_fusion = (
            qknorm_cache_op is not None
            and qkv.is_cuda
            and qkv.dtype in (torch.float16, torch.bfloat16)
            and key_cache.dtype == qkv.dtype
            and value_cache.dtype == qkv.dtype
            and is_supported_optimus_qknorm_cache_rotary(head_dim, rotary_dim)
        )
        if use_native_fusion:
            q, k, v, _ = _fused_qknorm_rope_cache_forward_impl_fake(
                qkv,
                qnorm_weight,
                knorm_weight,
                cos,
                sin,
                pos_id,
                head_dim,
                num_q_head,
                num_kv_head,
                rotary_dim,
                layer_name,
                eps,
                norm_weight_bias,
            )
            assert qknorm_cache_op is not None
            qknorm_cache_op(
                q,
                k,
                v,
                qkv,
                qnorm_weight,
                knorm_weight,
                cos,
                sin,
                pos_id.contiguous(),
                slot_mapping.contiguous(),
                key_cache,
                value_cache,
                head_dim,
                num_q_head,
                num_kv_head,
                rotary_dim,
                eps,
                norm_weight_bias,
            )
        else:
            # Preserve correctness for cache dtypes that the native kernel does
            # not handle yet, and for precompiled vLLM extensions that do not
            # contain the StepFun operators, while keeping one outer custom-op
            # dependency.
            if qknorm_cache_op is None:
                logger.warning_once(
                    "Step4 native QK-norm+RoPE+KV-cache op is unavailable; "
                    "falling back to CuTeDSL plus the attention backend's KV "
                    "cache update. Build vLLM from this source tree to enable "
                    "the native fused path."
                )
            q, k, v = _fused_qknorm_rope_forward_impl(
                qkv.contiguous(),
                qnorm_weight,
                knorm_weight,
                cos,
                sin,
                pos_id,
                head_dim,
                num_q_head,
                num_kv_head,
                rotary_dim,
                eps,
                norm_weight_bias,
            )
            attn_layer.impl.do_kv_cache_update(
                attn_layer,
                k.view(-1, num_kv_head, head_dim),
                v.view(-1, num_kv_head, head_dim),
                kv_cache,
                slot_mapping,
            )
    dummy = torch.empty(0, device=qkv.device, dtype=qkv.dtype)
    return q, k, v, dummy


direct_register_custom_op(
    op_name="fused_qknorm_rope_cache_forward_impl",
    op_func=_fused_qknorm_rope_cache_forward_impl_op,
    mutates_args=[],
    fake_impl=_fused_qknorm_rope_cache_forward_impl_fake,
)


def fused_qknorm_rope_cache_forward_impl(
    qkv: torch.Tensor,
    qnorm_weight: torch.Tensor,
    knorm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_id: torch.Tensor,
    head_dim: int,
    num_q_head: int,
    num_kv_head: int,
    rotary_dim: int,
    layer_name: str,
    eps: float = 1e-5,
    norm_weight_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.fused_qknorm_rope_cache_forward_impl(
        qkv,
        qnorm_weight,
        knorm_weight,
        cos,
        sin,
        pos_id,
        head_dim,
        num_q_head,
        num_kv_head,
        rotary_dim,
        layer_name,
        eps,
        norm_weight_bias,
    )


def _fused_indexer_norm_rope_forward_impl_fake(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(index_q),
        torch.empty_like(index_k),
        torch.empty_like(index_z),
    )


def _fused_indexer_norm_rope_forward_impl_op(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _fused_indexer_norm_rope_forward_impl(
        index_q,
        index_k,
        qnorm_weight,
        knorm_weight,
        knorm_bias,
        cos,
        sin,
        pos_id,
        head_dim,
        num_q_head,
        num_k_head,
        rotary_dim,
        eps,
        q_norm_weight_bias,
        index_z,
    )


direct_register_custom_op(
    op_name="fused_indexer_norm_rope_forward_impl",
    op_func=_fused_indexer_norm_rope_forward_impl_op,
    mutates_args=[],
    fake_impl=_fused_indexer_norm_rope_forward_impl_fake,
)


def fused_indexer_norm_rope_forward_impl(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.fused_indexer_norm_rope_forward_impl(
        index_q,
        index_k,
        index_z,
        qnorm_weight,
        knorm_weight,
        knorm_bias,
        cos,
        sin,
        pos_id,
        head_dim,
        num_q_head,
        num_k_head,
        rotary_dim,
        eps,
        q_norm_weight_bias,
    )


def apply_optimus_matmul_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.ops.OptimusMoe.matmul_fp32(x, weight.t())


def apply_optimus_matmul_fp32_fake(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return torch.empty(
        x.shape[0], weight.shape[0], device=x.device, dtype=torch.float32
    )


direct_register_custom_op(
    op_name="optimus_matmul_fp32",
    op_func=apply_optimus_matmul_fp32,
    mutates_args=[],
    fake_impl=apply_optimus_matmul_fp32_fake,
)


@cache
def has_optimus_moe_matmul_fp32() -> bool:
    """Whether the closed-source `step-optimus` OptimusMoe extension is loaded.

    Callers must fall back to a portable implementation when this is False.
    """
    return hasattr(torch.ops, "OptimusMoe") and hasattr(
        torch.ops.OptimusMoe, "matmul_fp32"
    )


def router_bias_func(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    router_bias: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
    nan_row_i_out: int = 0,
    indices_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert renormalize
    if indices_dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "Step4 router indices_dtype must be torch.int32 or torch.int64, "
            f"got {indices_dtype}."
        )

    step4_router_bias_triton_func = importlib.import_module(
        "vllm.models.step4.nvidia.ops.triton.router_bias"
    ).router_bias_triton_func
    return step4_router_bias_triton_func(
        gating_output,
        router_bias,
        topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        nan_row_i_out=nan_row_i_out,
        indices_dtype=indices_dtype,
    )
