"""Generated Step-4 minimal inference model.py; do not edit by hand.

Regenerate with ``python3 -m dsa_parity.build_step4_minimal_release``.
Numerical bodies come from the source manifest below.  Only merged-module
imports/metadata and the experimental Triton sliding-attention branch are
rewritten; sliding attention is the validated PyTorch SDPA implementation.
"""

from __future__ import annotations

# Source manifest (step4-minimal-inference-release-v1):
#   configuration_step4.py  sha256=8a500f91005b9cea302239086dc0115b57380b242741894f6f6cec6770499d02
#   inference/tp_layers.py  sha256=3d9401df8461e24dd23cdd0b98cccccaf3fccef5c0cc71008392fac3a7c3c8f4
#   modeling_step4_attention.py  sha256=342ae006f36350537bd5a2d66b3f644e6ad61119178c80cf956a829c52b26875
#   modeling_step4.py  sha256=3919cd7fdcea583372c9a4919a4aeae84f7d57bbb02a6de3e727899ff5f9259b

from kernel import (
    DSAGeometry,
    DSALayerCache,
    Step4SparseIndexer,
    build_rope_cache,
    decode_metadata,
    prefill_metadata,
    update_summaries_decode,
    update_summaries_prefill,
    sparse_attention_decode,
    sparse_attention_prefill,
    fused_qknorm_rope,
    linear_fp8_or_bf16,
    clamped_swiglu as triton_clamped_swiglu,
    weighted_topk_gather,
)


# ============================================================================
# merged source: configuration_step4.py
# ============================================================================
"""Configuration for StepFun step4.

The field names here are the checkpoint's, not HuggingFace's. That is deliberate: the
released ``config.json`` is the one production serves from, and renaming its keys would
mean shipping a config that only this file can read. Where a HuggingFace utility expects a
conventional name, it is exposed as a read-only property (``num_key_value_heads``) rather
than duplicated as state that could drift.

Three things about the layer geometry are easy to misread:

**The serialized ``layer_types`` is one longer than ``num_hidden_layers``.** The main
stack is layers ``0 .. num_hidden_layers - 1``; the extra trailing entry describes the MTP
layer, which lives at index ``num_hidden_layers`` and is a speculative-decoding accelerator
rather than part of the model's output. ``rope_theta`` and ``partial_rotary_factors`` are
the same length and follow the same rule.

**Attention type and FFN type are independent axes.** Layer 91 is ``full_attention`` (so it
carries a sparse indexer) *and* dense-MLP (so it has no experts). Deriving one from the
other happens to work for 92 of the 93 layers, which is exactly the kind of coincidence
that survives a smoke test.

**RoPE is per layer and gated twice.** ``rope_theta`` and ``partial_rotary_factors`` are
indexed by layer; ``rope_scaling`` applies only to layers whose type is listed in
``yarn_only_types``. On step4 that means full-attention layers get theta 5e6, a third of
the head rotated, and llama3 scaling, while sliding layers get theta 1e4, the whole head
rotated, and no scaling.
"""


from typing import Any

from transformers.configuration_utils import PretrainedConfig


# Incremented whenever the on-disk TP tensor layout changes incompatibly.  Version 2 keeps
# v1's corrected GQA/provider-group mapping and additionally TP-shards the shared expert.
# The latter is required because the deployed MoE combines the local shared and routed
# branches in FP32 and performs one all-reduce; replicating the shared expert would sum TP
# identical copies.
STEP4_TP_LAYOUT_VERSION = "gqa-provider-shared-tp-v2"


class Step4SparseConfig:
    """The ``sparse_config`` block, as a plain attribute holder.

    Kept as its own object rather than flattened onto the parent because the DSA code reads
    it as a unit, and because the checkpoint nests it -- flattening would require rewriting
    ``config.json`` on the way in and out.
    """

    def __init__(
        self,
        enabled: bool = True,
        proxy_dim: int = 256,
        sparse_indexer_rope_dim: int = 32,
        sparse_indexer_use_rope: bool = True,
        sparse_indexer_num_heads: int = 16,
        sparse_indexer_num_k_heads: int = 1,
        sparse_indexer_q_norm_type: str = "rmsnorm",
        sparse_indexer_k_norm_type: str = "layernorm",
        sparse_indexer_csa_z_norm_type: str = "none",
        sparse_indexer_softmax_variant: str = "ssmax",
        sparse_indexer_ssmax_s_granularity: str = "q_head",
        num_provider_groups: int = 4,
        topk: int = 512,
        region_block_size: int = 8,
        compression_method: str = "csa_block_compress",
        attention_impl: str = "sparse_gqa",
        apply_to_layer_types: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.enabled = enabled
        self.proxy_dim = proxy_dim
        self.sparse_indexer_rope_dim = sparse_indexer_rope_dim
        self.sparse_indexer_use_rope = sparse_indexer_use_rope
        self.sparse_indexer_num_heads = sparse_indexer_num_heads
        self.sparse_indexer_num_k_heads = sparse_indexer_num_k_heads
        self.sparse_indexer_q_norm_type = sparse_indexer_q_norm_type
        self.sparse_indexer_k_norm_type = sparse_indexer_k_norm_type
        self.sparse_indexer_csa_z_norm_type = sparse_indexer_csa_z_norm_type
        # Declared but unused. The indexer scores with a weighted ReLU, not a softmax, so
        # there is nothing for an ``ssmax`` variant to scale; the checkpoint's ``ssmax_s``
        # is bit-identical across all heads and layers, i.e. still at initialisation.
        # See :class:`Step4Attention` in ``modeling_step4.py``.
        self.sparse_indexer_softmax_variant = sparse_indexer_softmax_variant
        self.sparse_indexer_ssmax_s_granularity = sparse_indexer_ssmax_s_granularity
        self.num_provider_groups = num_provider_groups
        self.topk = topk
        self.region_block_size = region_block_size
        self.compression_method = compression_method
        self.attention_impl = attention_impl
        self.apply_to_layer_types = apply_to_layer_types or ["full_attention"]
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class Step4Config(PretrainedConfig):
    model_type = "step4"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 13824,
        num_hidden_layers: int = 92,
        vocab_size: int = 128896,
        num_attention_heads: int = 64,
        num_attention_groups: int = 4,
        head_dim: int = 192,
        rms_norm_eps: float = 1e-5,
        norm_dtype: str = "float32",
        fp32_residual_connection: bool = True,
        use_head_wise_attn_gate: bool = True,
        rope_theta: float | list[float] = 10000.0,
        partial_rotary_factors: list[float] | None = None,
        layer_types: list[str] | None = None,
        sliding_window: int | None = 512,
        rope_scaling: dict[str, Any] | None = None,
        yarn_only_types: list[str] | None = None,
        max_position_embeddings: int = 524288,
        use_moe: bool = True,
        moe_intermediate_size: int = 1536,
        share_expert_dim: int = 1536,
        moe_num_experts: int = 352,
        moe_top_k: int = 8,
        moe_layer_list: list[int] | None = None,
        use_moe_router_bias: bool = True,
        need_fp32_gate: bool = True,
        norm_expert_weight: bool = True,
        moe_router_scaling_factor: float = 3.0,
        swiglu_limits: list[float] | None = None,
        swiglu_limits_shared: list[float] | None = None,
        num_nextn_predict_layers: int = 1,
        sparse_config: dict[str, Any] | Step4SparseConfig | None = None,
        tie_word_embeddings: bool = False,
        bos_token_id: int = 0,
        eos_token_id: int | list[int] | None = None,
        tp_size: int = 1,
        tp_layout_version: str | None = None,
        **kwargs: Any,
    ) -> None:
        if tp_size < 1:
            raise ValueError(f"tp_size must be positive, got {tp_size}")
        self.tp_size = tp_size
        self.tp_layout_version = tp_layout_version
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.norm_dtype = norm_dtype
        self.fp32_residual_connection = fp32_residual_connection
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        self.max_position_embeddings = max_position_embeddings
        self.sliding_window = sliding_window
        self.rope_scaling = rope_scaling
        self.yarn_only_types = yarn_only_types or []

        # The checkpoint has one entry per layer *including* the MTP layer. Transformers 5
        # validates ``layer_types`` as a main-stack-only field, so retain the trailing MTP
        # entries separately and join them again in ``to_dict``. Runtime access continues
        # through ``layer_type`` for both the main stack and MTP layers.
        total_layers = num_hidden_layers + num_nextn_predict_layers
        serialized_layer_types = (
            list(layer_types)
            if layer_types is not None
            else ["full_attention"] * total_layers
        )
        if len(serialized_layer_types) != total_layers:
            raise ValueError(
                "layer_types must contain one entry per main and next-token "
                f"prediction layer: expected {total_layers}, got "
                f"{len(serialized_layer_types)}"
            )
        self.layer_types = serialized_layer_types[:num_hidden_layers]
        self._mtp_layer_types = serialized_layer_types[num_hidden_layers:]
        self.rope_theta = (
            list(rope_theta)
            if isinstance(rope_theta, (list, tuple))
            else [float(rope_theta)] * total_layers
        )
        self.partial_rotary_factors = partial_rotary_factors or [1.0] * total_layers

        self.use_moe = use_moe
        self.moe_intermediate_size = moe_intermediate_size
        self.share_expert_dim = share_expert_dim
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_layer_list = moe_layer_list or []
        self.use_moe_router_bias = use_moe_router_bias
        self.need_fp32_gate = need_fp32_gate
        self.norm_expert_weight = norm_expert_weight
        self.moe_router_scaling_factor = moe_router_scaling_factor
        self.swiglu_limits = swiglu_limits or []
        self.swiglu_limits_shared = swiglu_limits_shared or []
        self.num_nextn_predict_layers = num_nextn_predict_layers

        if isinstance(sparse_config, Step4SparseConfig):
            self.sparse_config = sparse_config
        elif sparse_config:
            self.sparse_config = Step4SparseConfig(**sparse_config)
        else:
            self.sparse_config = None

        self._validate_parallel_geometry()

        # ``num_experts_per_tok`` and ``moe_top_k`` are the same quantity under two names;
        # the checkpoint carries both. Trust ``moe_top_k`` and drop the duplicate rather
        # than storing two fields that can disagree.
        kwargs.pop("num_experts_per_tok", None)
        self._consume_quantization_config(kwargs)
        # Emitted on ``save_pretrained`` so a published checkpoint resolves these classes
        # under ``trust_remote_code``. Defaulted rather than required, so a config that
        # already carries its own mapping keeps it.
        kwargs.setdefault(
            "auto_map",
            {
                "AutoConfig": "model.Step4Config",
                "AutoModelForCausalLM": "model.Step4ForCausalLM",
            },
        )
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id if eos_token_id is not None else [1, 2],
            **kwargs,
        )

    @property
    def num_key_value_heads(self) -> int:
        """HuggingFace's name for ``num_attention_groups``."""
        return self.num_attention_groups

    @property
    def num_experts_per_tok(self) -> int:
        return self.moe_top_k

    @property
    def tp(self) -> int:
        """Active tensor-parallel degree (defaults to 1, i.e. single-GPU)."""
        return self.tp_size

    @property
    def num_attention_heads_per_rank(self) -> int:
        """Heads owned by each rank. ``q_proj``/``o_proj`` slice the head dim by this."""
        return self.num_attention_heads // self.tp

    @property
    def heads_per_group_per_rank(self) -> int:
        """Local attention Q heads served by each local KV group."""
        return self.num_attention_heads_per_rank // self.num_kv_groups_per_rank

    @property
    def num_kv_groups_per_rank(self) -> int:
        """KV heads needed by one rank's contiguous Q-head slice.

        With step4's 64 Q heads, 4 KV heads and TP=8 this is one, not four.  Ranks
        ``0,1`` share global KV0, ranks ``2,3`` share KV1, and so on.  Replicating all four
        KV heads and interpreting eight local Q heads as four two-head groups changes the
        GQA function for 48 of the 64 heads.
        """
        return self._groups_per_rank(self.num_attention_groups)

    @property
    def num_provider_groups_per_rank(self) -> int:
        """DSA provider groups aligned with the local KV groups."""
        if self.sparse_config is None:
            return 0
        return self._groups_per_rank(self.sparse_config.num_provider_groups)

    @property
    def sparse_indexer_heads_per_provider_group(self) -> int:
        """The checkpoint carries four indexer Q/W heads per provider group."""
        if self.sparse_config is None:
            return 0
        return (
            self.sparse_config.sparse_indexer_num_heads
            // self.sparse_config.num_provider_groups
        )

    @property
    def sparse_indexer_num_heads_per_rank(self) -> int:
        """Indexer Q/W heads stored on this rank (four for the production TP=8 layout)."""
        return (
            self.num_provider_groups_per_rank
            * self.sparse_indexer_heads_per_provider_group
        )

    @property
    def intermediate_per_rank(self) -> int:
        """Dense MLP gate/up output dim and down input dim, per rank."""
        return self.intermediate_size // self.tp

    @property
    def moe_intermediate_per_rank(self) -> int:
        return self.moe_intermediate_size // self.tp

    @property
    def share_expert_per_rank(self) -> int:
        return self.share_expert_dim // self.tp

    @property
    def vocab_per_rank(self) -> int:
        return self.vocab_size // self.tp

    def _groups_per_rank(self, total_groups: int) -> int:
        """Number of contiguous GQA/provider groups intersecting one rank's Q slice.

        Supported TP layouts either partition groups across ranks (``tp <= groups``) or
        duplicate one group over several ranks (``tp > groups``).  Requiring divisibility
        in one direction prevents a rank's Q slice from straddling a fractional group.
        """
        if total_groups % self.tp == 0:
            return total_groups // self.tp
        if self.tp % total_groups == 0:
            return 1
        raise ValueError(
            f"tp_size={self.tp} and group count {total_groups} are incompatible: "
            "one must divide the other"
        )

    def _validate_parallel_geometry(self) -> None:
        if self.num_attention_heads % self.tp:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} is not divisible by "
                f"tp_size={self.tp}"
            )
        if self.num_attention_heads % self.num_attention_groups:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} is not divisible by "
                f"num_attention_groups={self.num_attention_groups}"
            )

        local_kv = self.num_kv_groups_per_rank
        local_q = self.num_attention_heads_per_rank
        if local_q % local_kv:
            raise ValueError(
                f"{local_q} local Q heads do not divide into {local_kv} local KV groups"
            )

        sparse = self.sparse_config
        if sparse is not None:
            if sparse.num_provider_groups != self.num_attention_groups:
                raise ValueError(
                    "DSA provider groups must align with attention KV groups: "
                    f"{sparse.num_provider_groups} != {self.num_attention_groups}"
                )
            if sparse.sparse_indexer_num_heads % sparse.num_provider_groups:
                raise ValueError(
                    f"{sparse.sparse_indexer_num_heads} sparse indexer heads do not divide "
                    f"into {sparse.num_provider_groups} provider groups"
                )
            if self.num_provider_groups_per_rank != local_kv:
                raise ValueError(
                    "local DSA provider groups must align with local KV groups: "
                    f"{self.num_provider_groups_per_rank} != {local_kv}"
                )

        if self.tp > 1 and self.tp_layout_version != STEP4_TP_LAYOUT_VERSION:
            raise ValueError(
                "incompatible or unversioned Step4 TP shards: expected "
                f"tp_layout_version={STEP4_TP_LAYOUT_VERSION!r}, got "
                f"{self.tp_layout_version!r}. Older layouts either replicated all "
                "KV/provider groups or replicated the shared expert; both are numerically "
                "wrong. Regenerate every rank with "
                "convert.py (do not resume old shards)."
            )

    def layer_type(self, layer_idx: int) -> str:
        if layer_idx < 0:
            raise IndexError(f"layer index must be non-negative, got {layer_idx}")
        if layer_idx < self.num_hidden_layers:
            return self.layer_types[layer_idx]
        mtp_idx = layer_idx - self.num_hidden_layers
        if mtp_idx >= len(self._mtp_layer_types):
            raise IndexError(
                f"layer index {layer_idx} is outside the "
                f"{self.num_hidden_layers + len(self._mtp_layer_types)} configured layers"
            )
        return self._mtp_layer_types[mtp_idx]

    def is_sparse_layer(self, layer_idx: int) -> bool:
        """Whether DSA runs on this layer.

        Sparse attention is opt-in per *layer type*, not per layer index, and it is also
        gated on the sparse block being present and enabled. Both halves matter: dropping
        the type check would put an indexer on all 93 layers, and dropping the enabled
        check would ignore a config that asks for the dense path.
        """
        sparse = self.sparse_config
        if sparse is None or not sparse.enabled:
            return False
        return self.layer_type(layer_idx) in sparse.apply_to_layer_types

    def is_moe_layer(self, layer_idx: int) -> bool:
        return self.use_moe and layer_idx in self.moe_layer_list

    def _consume_quantization_config(self, kwargs: dict[str, Any]) -> None:
        """Take the fp8 block off the config object, keeping only what this model needs.

        The block has to leave the attribute namespace, and the reason is mechanical:
        ``transformers`` decides to run its own quantizer purely on
        ``hasattr(config, "quantization_config")``. Its ``fp8`` quantizer then rewrites *every*
        ``nn.Linear`` in the model into a quantized one and attaches a ``weight_scale_inv`` to
        each -- including the BF16 attention and MLP projections this checkpoint never
        quantized. Those invented parameters have no data in the checkpoint, so they stay on
        the meta device and the model detonates on the first ``.to(device)``. Measured: 38
        such parameters over four layers.

        That behaviour is right for a checkpoint whose fp8 layout is the one transformers
        expects. It is wrong here: the experts are stored as three stacked
        ``[experts, out, in]`` tensors and are dequantised by ``inference/fp8_gemm.py`` at the
        point of use. So the block is parsed into the one fact this model needs -- which layers
        ship bf16 experts -- and stashed for re-emission by :meth:`to_dict`, so a saved config
        is still byte-faithful to the one that was loaded.
        """
        quantization = kwargs.pop("quantization_config", None)
        if quantization is not None and not isinstance(quantization, dict):
            quantization = (
                quantization.to_dict() if hasattr(quantization, "to_dict") else dict(quantization)
            )
        self._quantization_config = quantization

        # Entries name tensors in the *serving* framework's vocabulary, not the checkpoint's.
        # The layer index is the only part that survives a rename, so it is the only part read.
        excluded = (quantization or {}).get("modules_to_not_convert") or ()
        layers: set[int] = set()
        for name in excluded:
            for part in str(name).split("."):
                if part.isdigit():
                    layers.add(int(part))
                    break
        self.bf16_expert_layers = sorted(layers)

    def is_fp8_expert_layer(self, layer_idx: int) -> bool:
        """Whether this layer's routed experts are block-scaled fp8 rather than bf16."""
        return self.is_moe_layer(layer_idx) and layer_idx not in self.bf16_expert_layers

    def sliding_window_for(self, layer_idx: int) -> int | None:
        """The window this layer attends over, or ``None`` for unbounded.

        A layer is windowed because its *type* says so. ``sliding_window`` in the config is
        the size, not the switch -- it stays set even on full-attention layers, where the
        DSA metadata builder ignores it entirely.
        """
        if self.layer_type(layer_idx) != "sliding_attention":
            return None
        return self.sliding_window

    def rope_scaling_for(self, layer_idx: int) -> dict[str, Any] | None:
        if self.rope_scaling is None:
            return None
        if self.yarn_only_types and self.layer_type(layer_idx) not in self.yarn_only_types:
            return None
        return self.rope_scaling

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        mtp_layer_types = output.pop("_mtp_layer_types", [])
        output["layer_types"] = list(self.layer_types) + list(mtp_layer_types)
        if isinstance(output.get("sparse_config"), Step4SparseConfig):
            output["sparse_config"] = output["sparse_config"].to_dict()
        # Re-attached on the way out only. See :meth:`_consume_quantization_config` for why it
        # must not be an attribute while the model is being built.
        output.pop("_quantization_config", None)
        if self._quantization_config is not None:
            output["quantization_config"] = self._quantization_config
        return output


# ============================================================================
# merged source: inference/tp_layers.py
# ============================================================================
"""Tensor-parallel linear layers for step4 on H200.

Mirrors DeepSeek-V4-Flash's ``convert.py`` + vLLM's ``ColumnParallelLinear`` /
``RowParallelLinear`` convention. Weights are pre-sharded per rank by the convert
step into ``model{r}-mp{TP}.safetensors``; these wrappers just wrap the loaded
weight and fire the right collective.

Two conventions that are easy to get wrong, and that the modeling files rely on:

* :class:`ColumnParallelLinear` slices the **output** dim and **does not** all-reduce.
  The caller's upstream must have produced a full input (via its upstream's reduce), and
  the downstream must tolerate a partial output (typically a :class:`RowParallelLinear`
  that will reduce). The model only ever uses this in a ``qkv / gate / up / indexer``
  role, immediately followed by a RowParallel that finishes the block.
* :class:`RowParallelLinear` slices the **input** dim and normally all-reduces the output.
  The caller passes the partial output of an upstream ColumnParallel as the input and
  gets back the full output (same on every rank). Used for ``o_proj`` and ``down_proj``,
  which add into the full residual stream.  ``reduce_results=False`` exposes the local
  partial when a caller needs to combine several TP/EP branches before one shared
  FP32 all-reduce (the Step4 MoE shared+routed path).

All collectives go through :func:`torch.distributed` and only fire when ``tp_size > 1``
and dist is initialized. A TP=1 build keeps the arithmetic bit-identical to a regular
``nn.Linear`` — that is the contract the rest of the model relies on for the
single-GPU path.
"""


import torch
from torch import nn
import torch.distributed as dist


_TP_GROUP: dist.ProcessGroup | None = None


def set_tp_group(group: dist.ProcessGroup | None) -> None:
    """Override the collective target. Defaults to the world group when None."""
    global _TP_GROUP
    _TP_GROUP = group


def tp_group() -> dist.ProcessGroup | None:
    if not (dist.is_available() and dist.is_initialized()):
        return None
    return _TP_GROUP


def tp_rank() -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return 0
    return dist.get_rank()


def tp_size() -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return 1
    return dist.get_world_size()


def _local_size(total: int, tp: int) -> int:
    if tp < 1:
        raise ValueError(f"tp_size must be positive, got {tp}")
    if total % tp:
        raise ValueError(f"{total} not divisible by tp_size={tp}")
    return total // tp


def _all_reduce(x: torch.Tensor) -> torch.Tensor:
    if tp_size() <= 1:
        return x
    group = _TP_GROUP
    if group is None:
        group = dist.group.WORLD
    torch.distributed.all_reduce(x, group=group)
    return x


class ColumnParallelLinear(nn.Module):
    """Output-sliced linear, no internal reduce.

    Weight shape per rank: ``[out/tp, in]``. Forward computes ``x @ W.T`` locally and
    returns the partial output of shape ``[..., out/tp]``. The caller is responsible for
    ensuring the upstream produced a full input and the downstream handles partial output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tp_size: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
        gather_output: bool = False,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size
        out_local = _local_size(out_features, tp_size)
        factory = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty((out_local, in_features), **factory), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_local, **factory), requires_grad=False)
            if bias
            else None
        )
        self.gather_output = gather_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.gather_output and tp_size() > 1:
            out = _all_gather(out)
        return out


def _all_gather(x: torch.Tensor) -> torch.Tensor:
    group = _TP_GROUP
    if group is None:
        group = dist.group.WORLD
    n = dist.get_world_size(group)
    out = [torch.empty_like(x) for _ in range(n)]
    torch.distributed.all_gather(out, x, group=group)
    return torch.cat(out, dim=-1)


def vocab_parallel_argmax(
    local_logits: torch.Tensor,
    *,
    expected_tp_size: int,
) -> torch.Tensor:
    """Return greedy token ids after reconstructing the full TP vocabulary.

    :class:`VocabParallelLinear` intentionally leaves the vocabulary sharded: rank
    ``r`` returns the contiguous window ``[r * vocab_local, (r + 1) *
    vocab_local)``.  Taking ``argmax`` on that tensor directly produces a *local
    column index*, not a token id, and (worse) lets every rank feed a different token
    into the next decode collective.

    This helper follows the reference DeepSeek inference path: all-gather the
    vocabulary windows in rank order, concatenate them, then take the argmax.  The
    returned tensor has shape ``local_logits.shape[:-1]`` and is identical on every
    TP rank.  For TP=1 it is a plain local argmax and does not require a process
    group.

    ``expected_tp_size`` is explicit rather than inferred so a TP checkpoint cannot
    silently run generation with an uninitialised or wrong-sized process group.
    """
    if local_logits.ndim == 0:
        raise ValueError("vocab logits must have at least one dimension")
    if expected_tp_size < 1:
        raise ValueError(f"expected_tp_size must be positive, got {expected_tp_size}")
    if expected_tp_size == 1:
        return local_logits.argmax(dim=-1)

    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(
            f"TP={expected_tp_size} vocab argmax requires an initialized "
            "torch.distributed process group"
        )
    group = _TP_GROUP if _TP_GROUP is not None else dist.group.WORLD
    actual_tp_size = dist.get_world_size(group)
    if actual_tp_size != expected_tp_size:
        raise RuntimeError(
            "vocab argmax process-group size does not match the checkpoint: "
            f"expected TP={expected_tp_size}, got {actual_tp_size}"
        )

    full_logits = _all_gather(local_logits.contiguous())
    return full_logits.argmax(dim=-1)


class RowParallelLinear(nn.Module):
    """Input-sliced linear with an optional internal all-reduce.

    Weight shape per rank: ``[out, in/tp]``. The input is the partial output of an
    upstream :class:`ColumnParallelLinear`; each rank multiplies its slice and returns
    the full output (all-reduced across ranks) by default.  With
    ``reduce_results=False`` it returns the local GEMM result instead.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tp_size: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size
        self.reduce_results = reduce_results
        in_local = _local_size(in_features, tp_size)
        factory = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty((out_features, in_local), **factory), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_features, **factory), requires_grad=False)
            if bias
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.reduce_results and self.tp_size > 1:
            return _all_reduce(out)
        return out


class VocabParallelLinear(nn.Module):
    """Output-vocab-sliced linear, no internal reduce (no all-gather).

    Used for ``lm_head`` in forward-only numeric check. Each rank produces logits for
    its vocab slice; a serving path would all-gather, but the gate only needs each
    rank's slice to be finite.

    The conversion step ships the **full** ``[vocab, in]`` weight on every rank. The
    parameter therefore starts full-sized so both the legacy Transformers loader and the
    direct tensor-injection loader in Transformers 5 see an exact checkpoint shape.
    :meth:`shard_` narrows it to this rank's contiguous vocabulary window immediately
    after loading and before the model is moved to the GPU.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tp_size: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size
        self.out_features = out_features
        self._out_features_per_rank = _local_size(out_features, tp_size)
        self._vocab_sharded = False
        factory = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_features, **factory), requires_grad=False)
            if bias
            else None
        )

    def shard_(self, rank: int | None = None) -> "VocabParallelLinear":
        """Keep only this rank's vocabulary window after checkpoint loading."""
        if self._vocab_sharded:
            return self
        local = self._out_features_per_rank
        rank = tp_rank() if rank is None else rank
        if rank < 0 or rank >= self.tp_size:
            raise ValueError(f"rank {rank} is outside TP range [0, {self.tp_size})")
        start = rank * local
        self.weight = nn.Parameter(
            self.weight.narrow(0, start, local).contiguous(),
            requires_grad=False,
        )
        if self.bias is not None:
            self.bias = nn.Parameter(
                self.bias.narrow(0, start, local).contiguous(),
                requires_grad=False,
            )
        self._vocab_sharded = True
        return self

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ) -> None:
        # Keep reloading an already-sharded module compatible with the legacy
        # state-dict path. Normal from_pretrained loading starts full-sized and
        # is narrowed exactly once by shard_ after the checkpoint is loaded.
        rank = tp_rank() if self.tp_size > 1 else 0
        for param_name in ("weight", "bias"):
            key = prefix + param_name
            if key not in state_dict:
                continue
            loaded = state_dict[key]
            is_bias = param_name == "bias"
            param = self.bias if is_bias else self.weight
            if param is None:
                continue
            if loaded.shape[0] > param.shape[0]:
                state_dict[key] = loaded.narrow(0, rank * param.shape[0], param.shape[0])
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


# ============================================================================
# merged source: modeling_step4_attention.py
# ============================================================================
"""step4 attention: a windowed dense path and a DSA sparse path, plus the cache both use.

step4 alternates three ``sliding_attention`` layers with one ``full_attention`` layer, and
the two are different enough that sharing an implementation would obscure both:

============  ==============================  ================================
\\             sliding_attention (70 layers)   full_attention (23 layers)
============  ==============================  ================================
history       last 512 tokens                 all of it, sparsely selected
RoPE theta    1e4                             5e6
RoPE span     all 192 dims                    64 dims (a third)
RoPE scaling  none                            llama3
indexer       --                              yes, drives region selection
cache         ring buffer, 512 deep           full K/V + region summaries
============  ==============================  ================================

Two of those rows are traps. The ``sliding_window: 512`` in the config is *not* a property
of the model, it is a property of the sliding layers; the DSA metadata builder ignores it
entirely, and a path that applied it to full-attention layers would quietly truncate the
long-context behaviour that is the entire point of the sparse path. And ``rope_scaling`` is
gated by ``yarn_only_types``, so applying it everywhere -- the obvious reading of a
top-level config key -- changes every sliding layer's positional encoding.

The ring buffer for sliding layers stores tokens at ``position % window`` and does not
attempt to keep them ordered. That is safe because softmax is permutation-invariant over
the key axis, and it is what removes the need to shift the cache every step.
"""


import torch
from torch import nn


# Queries per chunk in the sliding-window prefill. Bounds the attention mask to
# ``chunk x (chunk + window)`` so a 256k prefill never materialises a square mask.
SLIDING_PREFILL_CHUNK = 2048


class Step4RMSNorm(nn.Module):
    """Zero-centered RMSNorm: ``x / rms(x) * (1 + w)``, reduced in fp32.

    The ``1 +`` is not a stylistic variant -- the checkpoint's weights are centred on zero,
    so reading them as a conventional RMSNorm scale multiplies every activation by roughly
    zero. It fails loudly at the first layer, which is the good case; what makes it worth a
    comment is that the same weights are a *valid* conventional scale, so nothing in the
    tensor shapes or dtypes objects.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        normed = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (normed * (self.weight.float() + 1.0)).to(dtype)


class DenseLayerCache:
    """Ring buffer of the last ``window`` tokens for one sliding-attention layer."""

    def __init__(
        self,
        *,
        batch: int,
        window: int,
        num_kv_groups: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.window = window
        self.key = torch.zeros(
            (batch, window, num_kv_groups, head_dim), device=device, dtype=dtype
        )
        self.value = torch.zeros_like(self.key)

    def write(self, request: int, start: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Store one request's tokens, keeping only the most recent ``window`` of them."""
        count = key.shape[0]
        if count > self.window:
            key, value = key[-self.window :], value[-self.window :]
            start += count - self.window
            count = self.window
        slots = (torch.arange(count, device=key.device) + start) % self.window
        self.key[request, slots] = key
        self.value[request, slots] = value

    def valid_mask(self, lengths: torch.Tensor) -> torch.Tensor:
        """``[batch, window]`` bool: which slots hold a live token.

        ``min(length, window)`` covers both regimes with one expression -- below the window
        only the first ``length`` slots have been written, and at or above it every slot has.
        """
        slots = torch.arange(self.window, device=lengths.device)
        return slots.view(1, -1) < lengths.clamp_max(self.window).view(-1, 1)


class Step4Cache:
    """Per-layer KV state for a batch of sequences, plus the lengths both paths read.

    One object rather than a list of independent caches because ``lengths`` is shared: every
    layer advances together, and duplicating the counter per layer is how a decode step ends
    up scoring against a context one token stale on some layers and not others.
    """

    def __init__(self, config, *, batch: int, max_tokens: int, device: torch.device, dtype: torch.dtype) -> None:
        sparse = config.sparse_config
        geometry = DSAGeometry(
            proxy_dim=sparse.proxy_dim,
            topk=sparse.topk,
            region_size=sparse.region_block_size,
        )
        self.config = config
        self.batch = batch
        self.lengths = torch.zeros((batch,), device=device, dtype=torch.int32)
        self.layers: list[DSALayerCache | DenseLayerCache] = []
        # Q heads are column-sharded, so each rank only caches the KV groups intersecting
        # its contiguous Q-head slice.  For production TP=8 this is one group: ranks 0/1
        # cache KV0, 2/3 cache KV1, etc.  The pre-sharder duplicates that one group across
        # the two ranks that consume it.
        local_kv_groups = config.num_kv_groups_per_rank
        for layer_idx in range(config.num_hidden_layers):
            if config.is_sparse_layer(layer_idx):
                self.layers.append(
                    DSALayerCache(
                        batch=batch,
                        max_tokens=max_tokens,
                        num_kv_groups=local_kv_groups,
                        head_dim=config.head_dim,
                        geometry=geometry,
                        device=device,
                        dtype=dtype,
                    )
                )
            else:
                # A full-attention layer only reaches here when the sparse path is disabled,
                # and then its history is genuinely unbounded -- reusing ``sliding_window``
                # would silently truncate it to 512 and look like a quality regression rather
                # than a configuration error.
                window = config.sliding_window_for(layer_idx) or max_tokens
                self.layers.append(
                    DenseLayerCache(
                        batch=batch,
                        window=window,
                        num_kv_groups=local_kv_groups,
                        head_dim=config.head_dim,
                        device=device,
                        dtype=dtype,
                    )
                )

    def advance(self, seq_lens: torch.Tensor) -> torch.Tensor:
        """Record ``seq_lens`` new tokens per request and return the lengths from before."""
        past = self.lengths.clone()
        self.lengths = self.lengths + seq_lens.to(self.lengths.dtype)
        for layer in self.layers:
            if isinstance(layer, DSALayerCache):
                layer.lengths = self.lengths
        return past


class Step4Attention(nn.Module):
    """One attention layer. Sparse or windowed according to the layer's type.

    The indexer's parameters live directly on this module (``sparse_indexer_q`` and friends)
    rather than inside a submodule, because that is where the checkpoint puts them. The
    logic that drives them is in :class:`~inference.dsa_attention.Step4SparseIndexer`, which
    borrows them -- see its docstring.
    """

    def __init__(self, config, layer_idx: int, device=None, dtype=None) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads_per_rank
        self.num_kv_groups = config.num_kv_groups_per_rank
        self.head_dim = config.head_dim
        if self.num_heads % self.num_kv_groups:
            raise ValueError(
                f"{self.num_heads} local Q heads do not divide into "
                f"{self.num_kv_groups} local KV groups"
            )
        self.heads_per_group = self.num_heads // self.num_kv_groups
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window_for(layer_idx)
        self.is_sparse = config.is_sparse_layer(layer_idx)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_groups * self.head_dim

        # Keep the three checkpoint-facing names while loading, then pack them into the
        # single [Q_local, K_local, V_local] matrix used by vLLM's QKVParallelLinear.
        # Separate GEMM shapes are not numerically interchangeable in BF16: on H200 the
        # Q result happens to agree, while sparse K/V elements cross rounding boundaries.
        # ``pack_qkv_`` rebinds these parameters to views of the non-persistent packed
        # buffer, so matching the deployed fused GEMM costs no steady-state duplicate.
        self.q_proj = ColumnParallelLinear(
            config.hidden_size, config.num_attention_heads * config.head_dim,
            tp_size=config.tp, bias=False, **factory,
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.kv_size, bias=False, **factory
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.kv_size, bias=False, **factory
        )
        self.register_buffer(
            "_qkv_weight",
            torch.empty(0, **factory),
            persistent=False,
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * config.head_dim, config.hidden_size,
            tp_size=config.tp, bias=False, **factory,
        )
        self.q_norm = Step4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Step4RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.use_head_wise_attn_gate = config.use_head_wise_attn_gate
        if self.use_head_wise_attn_gate:
            self.g_proj = ColumnParallelLinear(
                config.hidden_size, config.num_attention_heads,
                tp_size=config.tp, bias=False, **factory,
            )

        rotary_span = int(round(self.head_dim * config.partial_rotary_factors[layer_idx]))
        self.rotary_span = rotary_span - rotary_span % 2
        self.register_buffer("rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)

        if self.is_sparse:
            self._build_indexer(config, layer_idx, factory)
        else:
            self.indexer = None

    @property
    def qkv_is_packed(self) -> bool:
        """Whether the runtime fused QKV matrix has been materialised."""
        return self._qkv_weight.numel() != 0

    def _invalidate_packed_qkv(self) -> None:
        # Preserve device/dtype (including meta during low-memory construction) without
        # retaining the packed storage. The three named parameters keep their own
        # references until a subsequent load or repack replaces them.
        self._qkv_weight = self.q_proj.weight.new_empty(0)

    def _validate_packed_qkv(self) -> None:
        expected_shape = (
            self.q_size + 2 * self.kv_size,
            self.config.hidden_size,
        )
        if tuple(self._qkv_weight.shape) != expected_shape:
            raise RuntimeError(
                f"layer {self.layer_idx}: packed QKV shape "
                f"{tuple(self._qkv_weight.shape)} != {expected_shape}"
            )

        parameters = (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight)
        expected_rows = (self.q_size, self.kv_size, self.kv_size)
        expected_offset = 0
        packed_storage = self._qkv_weight.untyped_storage().data_ptr()
        for name, weight, rows in zip(("q", "k", "v"), parameters, expected_rows):
            expected = (rows, self.config.hidden_size)
            if tuple(weight.shape) != expected:
                raise RuntimeError(
                    f"layer {self.layer_idx}: {name}_proj shape "
                    f"{tuple(weight.shape)} != {expected}"
                )
            if weight.device != self._qkv_weight.device or weight.dtype != self._qkv_weight.dtype:
                raise RuntimeError(
                    f"layer {self.layer_idx}: {name}_proj device/dtype does not "
                    "match packed QKV"
                )
            if weight.untyped_storage().data_ptr() != packed_storage:
                raise RuntimeError(
                    f"layer {self.layer_idx}: {name}_proj no longer shares packed "
                    "QKV storage; repack after moving or loading the model"
                )
            if weight.storage_offset() != expected_offset:
                raise RuntimeError(
                    f"layer {self.layer_idx}: {name}_proj storage offset "
                    f"{weight.storage_offset()} != {expected_offset}"
                )
            expected_offset += weight.numel()

    def pack_qkv_(self) -> Step4Attention:
        """Prepack Q/K/V weights for one fused projection, without steady-state copies.

        The checkpoint intentionally retains the separate ``q_proj``/``k_proj``/
        ``v_proj`` keys. Call this after loading and placing the model on its execution
        device. The first forward also calls it lazily for compatibility with direct
        model construction.
        """
        if self.qkv_is_packed:
            self._validate_packed_qkv()
            return self

        weights = (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight)
        if any(weight.is_meta for weight in weights):
            raise RuntimeError(
                f"layer {self.layer_idx}: cannot pack QKV weights on the meta device; "
                "load the checkpoint first"
            )
        devices = {weight.device for weight in weights}
        dtypes = {weight.dtype for weight in weights}
        if len(devices) != 1 or len(dtypes) != 1:
            raise RuntimeError(
                f"layer {self.layer_idx}: Q/K/V device or dtype mismatch: "
                f"devices={devices}, dtypes={dtypes}"
            )

        # ``run_full_92`` is itself under inference_mode. Temporarily disable it so the
        # long-lived parameters remain ordinary tensors that can safely be transformed
        # again by a later ``model.to(...)``.
        with torch.inference_mode(False), torch.no_grad():
            packed = torch.cat(weights, dim=0).contiguous()
        self._qkv_weight = packed

        offset = 0
        for projection, rows in (
            (self.q_proj, self.q_size),
            (self.k_proj, self.kv_size),
            (self.v_proj, self.kv_size),
        ):
            view = packed.narrow(0, offset, rows)
            projection.weight = nn.Parameter(view, requires_grad=False)
            offset += rows

        self._validate_packed_qkv()
        return self

    def _apply(self, fn, recurse: bool = True):
        """Keep the packed buffer and checkpoint-facing parameter views coalesced."""
        was_packed = self.qkv_is_packed
        result = super()._apply(fn, recurse=recurse)
        if was_packed:
            # Module._apply transforms the buffer and each registered parameter
            # independently. Repack immediately to restore one shared storage.
            self._invalidate_packed_qkv()
            self.pack_qkv_()
        return result

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # Loading after a previous inference run must not leave a stale packed buffer.
        # Child q/k/v modules are populated after this module-level callback returns.
        if self.qkv_is_packed:
            self._invalidate_packed_qkv()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _build_indexer(self, config, layer_idx: int, factory: dict) -> None:
        sparse = config.sparse_config
        proxy_dim = sparse.proxy_dim
        # DSA provider groups follow the same partition/duplication as KV groups.  The
        # production checkpoint has 16 indexer heads = 4 provider groups x 4 heads; TP=8
        # therefore constructs only four Q/W heads on each rank.  K and Z stay single-head
        # MQA projections and are replicated by the pre-sharder.
        heads = config.sparse_indexer_num_heads_per_rank
        self.sparse_indexer_q = nn.Linear(config.hidden_size, heads * proxy_dim, bias=False, **factory)
        self.sparse_indexer_k = nn.Linear(config.hidden_size, proxy_dim, bias=False, **factory)
        self.sparse_indexer_z = nn.Linear(config.hidden_size, proxy_dim, bias=False, **factory)
        self.sparse_indexer_w = nn.Linear(config.hidden_size, heads, bias=False, dtype=torch.float32, device=factory["device"])
        self.sparse_indexer_q_norm = Step4RMSNorm(proxy_dim, eps=config.rms_norm_eps)
        self.sparse_indexer_k_norm = nn.LayerNorm(proxy_dim, eps=config.rms_norm_eps, dtype=torch.float32, device=factory["device"])
        # Present in the checkpoint at its initialisation value on every head of every
        # layer, and unused by the deployed kernels: the indexer scores with a weighted
        # ReLU, which has no softmax for a scalable-softmax scale to act on. Registered so
        # loading is clean and the tensor is not silently dropped. Shape is the full head
        # count (the convert step ships ssmax_s as a replicate, see ``_SHARD_TABLE``); the
        # indexer that would read it keys off the full set of heads.
        self.register_buffer(
            "ssmax_s", torch.zeros(config.num_attention_heads, dtype=torch.float32, device=factory["device"])
        )
        self.register_buffer("sparse_indexer_rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("sparse_indexer_rope_sin", torch.empty(0), persistent=False)
        self.indexer = Step4SparseIndexer(
            self,
            geometry=DSAGeometry(
                proxy_dim=proxy_dim, topk=sparse.topk, region_size=sparse.region_block_size
            ),
            num_kv_groups=self.num_kv_groups,
        )

    def build_rope(
        self,
        *,
        max_position: int,
        device: torch.device,
        shared_cache: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]]
        | None = None,
    ) -> None:
        """Materialise this layer's cos/sin tables, optionally sharing equal tables.

        Step4 has 92 layers but only a few distinct RoPE geometries.  At the production
        524288-token capacity, independently materialising every table costs many GiB per
        rank. vLLM caches by RoPE parameters; ``Step4Model.build_rope`` passes a model-local
        cache here to do the same without introducing process-global state.
        """
        config = self.config
        theta = config.rope_theta[self.layer_idx]
        scaling = config.rope_scaling_for(self.layer_idx)
        common_key = (
            float(theta),
            int(max_position),
            str(device),
            repr(scaling),
        )
        main_key = ("attention", self.rotary_span, torch.bfloat16, *common_key)
        if self.rope_cos.numel() == 0:
            cached = shared_cache.get(main_key) if shared_cache is not None else None
            if cached is None:
                cached = build_rope_cache(
                    rotary_span=self.rotary_span,
                    theta=theta,
                    max_position=max_position,
                    device=device,
                    # vLLM constructs frequencies/trigonometry in FP32, then
                    # stores the cache in the model's default BF16 dtype.
                    dtype=torch.bfloat16,
                    scaling=scaling,
                )
                if shared_cache is not None:
                    shared_cache[main_key] = cached
            cos, sin = cached
            self.rope_cos, self.rope_sin = cos, sin
        elif shared_cache is not None:
            shared_cache.setdefault(main_key, (self.rope_cos, self.rope_sin))
        if self.is_sparse and self.sparse_indexer_rope_cos.numel() == 0:
            indexer_span = self.config.sparse_config.sparse_indexer_rope_dim
            indexer_key = (
                "sparse_indexer",
                indexer_span,
                torch.bfloat16,
                *common_key,
            )
            cached = (
                shared_cache.get(indexer_key)
                if shared_cache is not None
                else None
            )
            if cached is None:
                cached = build_rope_cache(
                    rotary_span=indexer_span,
                    theta=theta,
                    max_position=max_position,
                    device=device,
                    dtype=torch.bfloat16,
                    scaling=scaling,
                )
                if shared_cache is not None:
                    shared_cache[indexer_key] = cached
            cos, sin = cached
            self.sparse_indexer_rope_cos, self.sparse_indexer_rope_sin = cos, sin
        elif self.is_sparse and shared_cache is not None:
            indexer_span = self.config.sparse_config.sparse_indexer_rope_dim
            indexer_key = (
                "sparse_indexer",
                indexer_span,
                torch.bfloat16,
                *common_key,
            )
            shared_cache.setdefault(
                indexer_key,
                (self.sparse_indexer_rope_cos, self.sparse_indexer_rope_sin),
            )

    def _apply_rope(self, tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate the leading ``rotary_span`` dims of every head, NEOX-paired.

        Dims past the span are returned untouched, which is what "partial" means here: on
        full-attention layers two thirds of each head carries no positional signal at all.

        The deployed fused QK-norm/RoPE path has a precise rounding boundary: norm first
        produces BF16 Q/K; the FP32-built, BF16-stored cache and Q/K are then widened for
        the RoPE arithmetic, and only the rotated result is rounded back to BF16. Performing
        the multiply itself in BF16 drifts from the oracle at every layer.
        """
        half = self.rotary_span // 2
        cos = self.rope_cos[positions].unsqueeze(1).float()
        sin = self.rope_sin[positions].unsqueeze(1).float()
        real = tensor[..., :half].float()
        imaginary = tensor[..., half : 2 * half].float()
        rotated = torch.cat(
            (real * cos - imaginary * sin, real * sin + imaginary * cos), dim=-1
        ).to(tensor.dtype)
        return torch.cat((rotated, tensor[..., 2 * half :]), dim=-1)

    def _project_packed_qkv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the deployed single QKV GEMM and retain its packed row layout."""
        if not self.qkv_is_packed:
            self.pack_qkv_()
        return torch.nn.functional.linear(hidden_states, self._qkv_weight)

    def _split_packed_qkv(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return TP-local Q/K/V head views without materialising another copy."""
        tokens = qkv.shape[0]
        query, key, value = qkv.split(
            (self.q_size, self.kv_size, self.kv_size), dim=-1
        )
        query = query.view(tokens, self.num_heads, self.head_dim)
        key = key.view(tokens, self.num_kv_groups, self.head_dim)
        value = value.view(tokens, self.num_kv_groups, self.head_dim)
        return query, key, value

    def _project_qkv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compatibility helper returning raw TP-local heads after one fused GEMM."""
        return self._split_packed_qkv(self._project_packed_qkv(hidden_states))

    def _project(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self._project_packed_qkv(hidden_states)
        if qkv.is_cuda and self.head_dim == 192:
            # Keep the projection packed all the way into the Triton kernel.  Besides
            # avoiding three launches and an intermediate concat, this reproduces the
            # deployed norm -> BF16 materialisation -> RoPE rounding boundary.
            query, key, value = fused_qknorm_rope(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rope_cos,
                self.rope_sin,
                positions.contiguous(),
                head_dim=self.head_dim,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_groups,
                rotary_pairs=self.rotary_span // 2,
                eps=self.q_norm.variance_epsilon,
                norm_weight_bias=1.0,
            )
            tokens = qkv.shape[0]
            return (
                query.view(tokens, self.num_heads, self.head_dim),
                key.view(tokens, self.num_kv_groups, self.head_dim),
                value.view(tokens, self.num_kv_groups, self.head_dim),
            )

        # CPU and non-Step4 geometries retain a transparent reference fallback for
        # configuration tooling and the small assembled-model tests.
        query, key, value = self._split_packed_qkv(qkv)
        query = self._apply_rope(self.q_norm(query), positions)
        key = self._apply_rope(self.k_norm(key), positions)
        return query, key, value

    def _gate_and_project(self, attn_output: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the per-head sigmoid gate, then ``o_proj``.

        The gate is applied *before* the output projection and per head, so it can suppress
        an entire head's contribution. Applying it after ``o_proj`` would need a different
        weight shape and is not the same function.
        """
        if self.use_head_wise_attn_gate:
            gate = self.g_proj(hidden_states).sigmoid().unsqueeze(-1)
            attn_output = attn_output * gate.to(attn_output.dtype)
        return self.o_proj(attn_output.reshape(attn_output.shape[0], -1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cache: Step4Cache,
        seq_lens: torch.Tensor,
        past_lens: torch.Tensor,
        *,
        is_decode: bool,
    ) -> torch.Tensor:
        """Attend over a packed batch of tokens.

        Args:
            hidden_states: ``[total_tokens, hidden]``, requests concatenated.
            positions: ``[total_tokens]`` int32, absolute position of each token.
            seq_lens: ``[batch]`` int32, tokens contributed by each request now.
            past_lens: ``[batch]`` int32, tokens already in the cache.
            is_decode: exactly one token per request. A separate flag rather than
                ``seq_lens.max() == 1`` because a one-token prefill is a real case and takes
                the prefill path.
        """
        query, key, value = self._project(hidden_states, positions)
        layer_cache = cache.layers[self.layer_idx]

        if not is_decode and bool((past_lens > 0).any()):
            # Chunked prefill is unsupported on *both* paths, and each fails differently.
            # The dense path would need the ring buffer's live span joined to the chunk's own
            # keys, and the join is where the window bound gets lost. The sparse path is
            # worse: ``update_summaries_prefill`` numbers regions from zero for whatever it is
            # given, so a second chunk would overwrite the first chunk's summaries with its
            # own tokens rather than appending -- corruption, not staleness, and silent.
            raise NotImplementedError(
                "prefill on top of an existing cache is not implemented; prefill once, then "
                "decode one token at a time"
            )

        if self.is_sparse:
            attn_output = self._sparse_attention(
                query, key, value, layer_cache, hidden_states, positions,
                seq_lens, past_lens, is_decode=is_decode,
            )
        else:
            attn_output = self._sliding_attention(
                query, key, value, layer_cache, seq_lens, past_lens, is_decode=is_decode
            )
        return self._gate_and_project(attn_output, hidden_states)

    def _sparse_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_cache: DSALayerCache,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        past_lens: torch.Tensor,
        *,
        is_decode: bool,
    ) -> torch.Tensor:
        index_q, index_k, index_z, weights = self.indexer.project(hidden_states, positions)

        offset = 0
        for request in range(layer_cache.batch):
            length = int(seq_lens[request])
            layer_cache.write_kv(
                request, int(past_lens[request]), key[offset : offset + length], value[offset : offset + length]
            )
            offset += length

        if is_decode:
            update_summaries_decode(layer_cache, index_k, index_z)
            packed, counts = decode_metadata(self.indexer, layer_cache, index_q, weights)
            out, _ = sparse_attention_decode(
                query,
                layer_cache.key,
                layer_cache.value,
                packed,
                counts,
                layer_cache.lengths,
                num_kv_groups=self.num_kv_groups,
                region_size=layer_cache.geometry.region_size,
                softmax_scale=self.scaling,
            )
            return out

        update_summaries_prefill(layer_cache, index_k, index_z, seq_lens)
        packed, counts = prefill_metadata(
            self.indexer, layer_cache, index_q, weights, seq_lens, past_lens
        )
        out, _ = sparse_attention_prefill(
            query,
            layer_cache.key,
            layer_cache.value,
            packed,
            counts,
            num_kv_groups=self.num_kv_groups,
            region_size=layer_cache.geometry.region_size,
            softmax_scale=self.scaling,
        )
        return out

    def _sliding_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_cache: DenseLayerCache,
        seq_lens: torch.Tensor,
        past_lens: torch.Tensor,
        *,
        is_decode: bool,
    ) -> torch.Tensor:
        window = layer_cache.window
        out = torch.empty_like(query)
        if is_decode:
            lengths = past_lens + 1
            batch = layer_cache.key.shape[0]
            for request in range(batch):
                layer_cache.write(
                    request,
                    int(past_lens[request]),
                    key[request : request + 1],
                    value[request : request + 1],
                )
            out = torch.nn.functional.scaled_dot_product_attention(
                # One query token per request: [batch, heads, 1, head_dim].
                query.unsqueeze(2),
                layer_cache.key.transpose(1, 2),
                layer_cache.value.transpose(1, 2),
                attn_mask=layer_cache.valid_mask(lengths).view(batch, 1, 1, -1),
                scale=self.scaling,
                enable_gqa=True,
            )
            return out.reshape(batch, self.num_heads, self.head_dim)

        offset = 0
        for request in range(layer_cache.key.shape[0]):
            length = int(seq_lens[request])
            past = int(past_lens[request])
            if past:
                raise NotImplementedError(
                    "dense prefill on top of an existing cache is not implemented; chunked "
                    "prefill would need the ring buffer's live span joined to the chunk's own "
                    "keys, and the join is where the window bound gets lost"
                )
            span = slice(offset, offset + length)
            out[span] = self._sliding_prefill_chunked(
                query[span], key[span], value[span], window
            )
            layer_cache.write(request, 0, key[span], value[span])
            offset += length
        return out

    def _sliding_prefill_chunked(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, window: int
    ) -> torch.Tensor:
        """Causal attention limited to the last ``window`` tokens, one query chunk at a time.

        Chunking bounds the mask rather than the compute: a 256k prefill with a 512-token
        window would otherwise build a 256k x 256k mask to express a band 512 wide.
        """
        length = query.shape[0]
        out = torch.empty_like(query)
        positions = torch.arange(length, device=query.device)

        for start in range(0, length, SLIDING_PREFILL_CHUNK):
            stop = min(start + SLIDING_PREFILL_CHUNK, length)
            key_start = max(0, start - window + 1)
            q_positions = positions[start:stop].view(-1, 1)
            k_positions = positions[key_start:stop].view(1, -1)
            distance = q_positions - k_positions
            mask = (distance >= 0) & (distance < window)
            chunk = torch.nn.functional.scaled_dot_product_attention(
                query[start:stop].transpose(0, 1).unsqueeze(0),
                key[key_start:stop].transpose(0, 1).unsqueeze(0),
                value[key_start:stop].transpose(0, 1).unsqueeze(0),
                attn_mask=mask.view(1, 1, stop - start, -1),
                scale=self.scaling,
                enable_gqa=True,
            )
            out[start:stop] = chunk.squeeze(0).transpose(0, 1)
        return out


# ============================================================================
# merged source: modeling_step4.py
# ============================================================================
"""StepFun step4 for HuggingFace transformers.

Adapted from StepFun's own ``stepfun-ai/Step-3.7-Flash`` release (Apache 2.0), whose dense
backbone -- zero-centered RMSNorm, clamped SwiGLU, sigmoid router with a selection-only
bias, per-head attention gate -- step4 shares. What step4 adds is DSA sparse attention on
its full-attention layers, which lives in :mod:`modeling_step4_attention` and
:mod:`inference`.

Three pieces of arithmetic here look like details and are not:

**The residual stream is fp32.** ``fp32_residual_connection`` is set, and it means the
accumulation is fp32 while the matmul inputs are bf16 -- not that the weights are fp32. At 92
layers the difference is visible.

**The SwiGLU clamps are asymmetric.** The gate is clamped from above only; the up projection
is clamped on both sides; their product is not clamped at all. Symmetrising the gate clamp
would change the function on every negative activation, which is most of them.

**The router's expert weight is not the routed score.** Selection ranks
``sigmoid(logits) + bias``; the weight is that score with the bias *subtracted back out*,
which is arithmetically ``sigmoid(logits)`` but not bit-identically so. The deployed kernel
subtracts, so this does too -- see :meth:`Step4MoE.route`.
"""


import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel


# The renormalisation guard from the deployed router kernel. Small enough to be a no-op on
# any real top-k sum, and reproduced rather than replaced so the arithmetic matches.
ROUTER_RENORM_EPS = 1e-20


class Step4MLP(nn.Module):
    """Clamped SwiGLU. Used for the dense layers and for every MoE layer's shared expert.

    ``tp`` defaults to ``config.tp`` for the dense FFN (ColumnParallel/RowParallel, TP-sliced
    on the intermediate dim). The shared expert uses the same TP slicing but sets
    ``reduce_results=False``: its local down-projection is added to the local EP-routed
    output in FP32 and those two branches are all-reduced exactly once.
    """

    def __init__(
        self,
        config: Step4Config,
        intermediate_size: int,
        limit: float,
        *,
        tp: int | None = None,
        reduce_results: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        tp = config.tp if tp is None else tp
        factory = {"device": device, "dtype": dtype}
        # ``tp`` is the slice the ColumnParallel / RowParallel wrappers apply. Pre-slicing
        # ``intermediate_size`` here and then passing ``tp_size=tp`` would double-slice --
        # each rank ends up with ``inter / tp^2`` instead of ``inter / tp`` and the
        # checkpoint shape check fails. ``intermediate_size`` is the full out dim; the
        # wrapper does the projection onto this rank's output slice.
        self.gate_proj = ColumnParallelLinear(config.hidden_size, intermediate_size, tp_size=tp, bias=False, **factory)
        self.up_proj = ColumnParallelLinear(config.hidden_size, intermediate_size, tp_size=tp, bias=False, **factory)
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            tp_size=tp,
            bias=False,
            reduce_results=reduce_results,
            **factory,
        )
        self.limit = limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        return self.down_proj(clamped_swiglu(gate, up, self.limit))


def clamped_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """``silu(gate).clamp(max=limit) * up.clamp(-limit, limit)``, computed in fp32.

    The clamps bound the activation's magnitude so the fp8 expert GEMMs downstream keep their
    dynamic range. The gate has no lower clamp because ``silu`` already bounds it from below at
    about -0.28; adding one would be inert on real values and wrong on the boundary.

    The fp32 is deliberate and worth defending, because the obvious reference disagrees.
    StepFun's own HF release does this arithmetic in the tensor's own dtype (bf16), and so does
    vLLM's ``SwigluStepAndMul.forward_native``. But neither is the deployed path: the CUDA path
    is a Triton kernel that loads both halves ``.to(tl.float32)``, computes silu, clamps and
    multiplies in fp32, and casts only the result back. Since precision follows the deployed
    kernel, this matches that and not the two reference implementations -- so a future reader
    comparing against the HF release will find a difference here, and it is the release that
    deviates.
    """
    if gate.is_cuda:
        return triton_clamped_swiglu(
            gate.contiguous(), up.contiguous(), limit=limit
        )

    # CPU fallback keeps shape-only/unit tests usable. Production inference is
    # CUDA/Hopper and therefore always takes the pure-Triton path above.
    activated = torch.nn.functional.silu(gate.float()).clamp(max=limit)
    bounded = up.float().clamp(-limit, limit)
    return (activated * bounded).to(gate.dtype)


class _StackedExpertWeight(nn.Module):
    """One ``[n_local_experts, out, in]`` weight plus its block scale, addressed one expert at a time.

    Named to match the checkpoint (``moe.gate_proj.weight``), so these are direct children of
    the MoE block rather than of an ``experts`` container -- an extra level of nesting here
    would rename every expert tensor.

    ``weight_scale_inv`` is declared only for the fp8 layers. The alternative, always
    declaring it and leaving it unset for bf16 layers, means the dispatcher's "is the scale
    present" test becomes "is the scale meaningful", and there is no value that makes an
    unquantised weight's scale meaningful.

    EP, not TP-of-inner-dim. Each rank owns ``n_experts // tp_size`` contiguous experts and
    keeps each expert's *full* inner dim. That is the only layout under which the fp8 block
    scale's K-block grid (``ceil(in_features / 128)``) stays aligned with the GEMM's K-iters:
    the pre-shard step slices the experts dim only, leaving each expert's ``(N, K)`` weight
    and ``(N_blocks, K_blocks)`` scale intact. Slicing the inner dim would put a single
    K-iter across a weight-block boundary -- forcing a per-tile scale gather the kernel does
    not do -- and break the block-scaling assumption.
    """

    def __init__(
        self,
        n_local_experts: int,
        out_features: int,
        in_features: int,
        *,
        quantized: bool,
        block: int = 128,
        device=None,
    ) -> None:
        super().__init__()
        self.n_local_experts = n_local_experts
        self.block = block
        weight_dtype = torch.float8_e4m3fn if quantized else torch.bfloat16
        self.weight = nn.Parameter(
            torch.empty(
                (n_local_experts, out_features, in_features),
                device=device,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        if quantized:
            self.weight_scale_inv = nn.Parameter(
                torch.empty(
                    (
                        n_local_experts,
                        -(-out_features // block),
                        -(-in_features // block),
                    ),
                    device=device,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        else:
            self.weight_scale_inv = None

    def apply_expert(self, expert_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        # ``hidden_states`` is the routed tokens for this expert on this rank; it is not
        # sliced. Each rank owns its experts' full weights, so the GEMM runs over the full
        # in dim and the per-rank output is one expert's worth of contribution. There is no
        # inner-dim TP slice to sum across ranks -- all-reduce is the responsibility of the
        # caller, summing only the routed-expert outputs (one row per token) across ranks.
        scale = None if self.weight_scale_inv is None else self.weight_scale_inv[expert_idx]
        return linear_fp8_or_bf16(hidden_states, self.weight[expert_idx], scale)


class Step4MoE(nn.Module):
    """Sigmoid router with a selection-only bias, over 352 stacked experts.

    The shared expert is *not* part of this block. The checkpoint stores it as a sibling
    (``layers.N.share_expert``), so the decoder layer owns it and adds its output; owning it
    here would nest it under ``moe`` and rename all three of its tensors.

    EP, not TP-of-inner-dim: each rank owns ``moe_num_experts // tp`` contiguous experts
    with the full inner dim per expert, so the pre-shard's experts-dim slice is load-bearing
    here (see :class:`_StackedExpertWeight`). The router weight and bias are replicated --
    every rank scores all experts and selects the same top-k -- so the routed tokens are the
    same everywhere and ``(expert_ids == global_idx)`` matches the same rows on every rank.
    """

    def __init__(self, config: Step4Config, layer_idx: int, device=None, dtype=None) -> None:
        super().__init__()
        self.top_k = config.moe_top_k
        self.scaling_factor = config.moe_router_scaling_factor
        self.renormalize = config.norm_expert_weight
        self.limit = config.swiglu_limits[layer_idx] if config.swiglu_limits else 7.0
        self.tp = config.tp
        # ``tp_rank()`` is 0 when dist is not initialised (the single-GPU path), which is
        # correct: with tp=1 every rank owns every expert and start_idx/end_idx span [0, N).
        self.rank = tp_rank()

        n_routed = config.moe_num_experts
        n_local = n_routed // self.tp
        assert n_routed % self.tp == 0, f"{n_routed} not divisible by tp={self.tp}"
        self.n_local_experts = n_local
        self.experts_start_idx = self.rank * n_local
        self.experts_end_idx = self.experts_start_idx + n_local

        # Router weight is replicated: every rank computes all ``moe_num_experts`` logits per token
        # and picks the same top-k, so a single rank could in principle make the decision. Keeping
        # it full across ranks means the routed tokens are the same everywhere -- which is the
        # precondition for expert slicing: the same ``(rows_for_expert_e)`` on every rank.
        self.gate = nn.Linear(
            config.hidden_size, config.moe_num_experts, bias=False, device=device, dtype=dtype
        )
        self.router_bias = nn.Parameter(
            torch.zeros(config.moe_num_experts, dtype=torch.float32, device=device),
            requires_grad=False,
        )

        quantized = config.is_fp8_expert_layer(layer_idx)
        self.gate_proj = _StackedExpertWeight(n_local, config.moe_intermediate_size, config.hidden_size, quantized=quantized, device=device)
        self.up_proj = _StackedExpertWeight(n_local, config.moe_intermediate_size, config.hidden_size, quantized=quantized, device=device)
        self.down_proj = _StackedExpertWeight(n_local, config.hidden_size, config.moe_intermediate_size, quantized=quantized, device=device)

    def route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick ``top_k`` experts per token and weight them."""
        logits = torch.nn.functional.linear(hidden_states.float(), self.gate.weight.float())
        gate_prob = torch.sigmoid(logits)
        scores = gate_prob + self.router_bias.view(1, -1)
        top_scores, expert_ids = torch.topk(scores, self.top_k, dim=-1)

        weights = top_scores - self.router_bias[expert_ids]
        if self.renormalize:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + ROUTER_RENORM_EPS)
        return weights * self.scaling_factor, expert_ids

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return this rank's BF16 routed-expert contribution without a collective.

        Iterates over this rank's local experts (the contiguous index slice
        ``[start, end)``) and gathers, for each, the tokens routed to that expert's global
        index. The residual stream is full on every rank, so the routed tokens are the same
        everywhere. Expert GEMMs are grouped by expert for efficiency, but the deployed
        ``ep_gather`` accumulates their BF16 outputs in top-k *slot* order using an FP32
        accumulator and only then rounds the local result to BF16. Keep that order here;
        the decoder combines this local result with its local shared-expert result and
        performs one FP32 all-reduce.
        """
        weights, expert_ids = self.route(hidden_states)
        slot_output = torch.zeros(
            (*hidden_states.shape[:-1], self.top_k, hidden_states.shape[-1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        for local in range(self.n_local_experts):
            global_idx = self.experts_start_idx + local
            rows, slots = (expert_ids == global_idx).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            tokens = hidden_states[rows]
            hidden = clamped_swiglu(
                self.gate_proj.apply_expert(local, tokens),
                self.up_proj.apply_expert(local, tokens),
                self.limit,
            )
            contribution = self.down_proj.apply_expert(local, hidden)
            slot_output[rows, slots] = contribution.to(hidden_states.dtype)

        if slot_output.is_cuda:
            return weighted_topk_gather(
                slot_output.contiguous(), weights.contiguous()
            )

        # CPU fallback mirrors the Triton kernel's logical slot order.
        out = torch.zeros(
            hidden_states.shape, device=hidden_states.device, dtype=torch.float32
        )
        for slot in range(self.top_k):
            out = out + (
                slot_output[:, slot].float() * weights[:, slot].unsqueeze(-1)
            )
        return out.to(hidden_states.dtype)


class Step4DecoderLayer(nn.Module):
    def __init__(self, config: Step4Config, layer_idx: int, device=None, dtype=None) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.fp32_residual = config.fp32_residual_connection
        self.self_attn = Step4Attention(config, layer_idx, device=device, dtype=dtype)
        self.input_layernorm = Step4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Step4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.is_moe_layer(layer_idx):
            self.moe = Step4MoE(config, layer_idx, device=device, dtype=dtype)
            shared_limit = (
                config.swiglu_limits_shared[layer_idx] if config.swiglu_limits_shared else 7.0
            )
            # The shared expert is TP-sharded. Its RowParallel down projection deliberately
            # leaves the local partial unreduced so it can be combined with the local
            # expert-parallel routed output before one FP32 collective.
            self.share_expert = Step4MLP(
                config,
                config.share_expert_dim,
                shared_limit,
                tp=config.tp,
                reduce_results=False,
                device=device,
                dtype=dtype,
            )
            self.mlp = None
        else:
            limit = config.swiglu_limits[layer_idx] if config.swiglu_limits else 7.0
            self.mlp = Step4MLP(config, config.intermediate_size, limit, device=device, dtype=dtype)
            self.moe = None
            self.share_expert = None

    def _feed_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.moe is None:
            return self.mlp(hidden_states)
        # Both branches are local BF16 values. The NV path widens each independently,
        # adds in FP32, then performs a single TP all-reduce. Reducing either branch first
        # changes both the function (for a replicated shared expert) and rounding order.
        combined = self.moe(hidden_states).float()
        combined = combined + self.share_expert(hidden_states).float()
        if self.moe.tp > 1:
            combined = _all_reduce(combined)
        return combined

    def forward(
        self,
        residual: torch.Tensor,
        positions: torch.Tensor,
        cache: Step4Cache,
        seq_lens: torch.Tensor,
        past_lens: torch.Tensor,
        *,
        is_decode: bool,
    ) -> torch.Tensor:
        """Take and return the fp32 residual stream, with bf16 into the matmuls."""
        hidden = self.input_layernorm(residual).to(self.self_attn.q_proj.weight.dtype)
        attn_out = self.self_attn(
            hidden, positions, cache, seq_lens, past_lens, is_decode=is_decode
        )
        residual = residual + attn_out.float()

        hidden = self.post_attention_layernorm(residual).to(attn_out.dtype)
        return residual + self._feed_forward(hidden).float()


class Step4PreTrainedModel(PreTrainedModel):
    config_class = Step4Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Step4DecoderLayer"]
    _supports_sdpa = True
    # The MTP layer sits at index ``num_hidden_layers`` and exists only to accelerate
    # speculative decoding. Its weights do not affect the model's output, so they are not
    # built. Declaring them ignored says so; letting them surface as "unexpected keys" would
    # read as a loading failure.
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\.9[2-9]\."]


class Step4Model(Step4PreTrainedModel):
    def __init__(self, config: Step4Config) -> None:
        super().__init__(config)
        dtype = torch.bfloat16
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            Step4DecoderLayer(config, layer_idx, dtype=dtype)
            for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = Step4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def build_rope(self, device: torch.device) -> None:
        shared_cache: dict[
            tuple[object, ...], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        for layer in self.layers:
            layer.self_attn.build_rope(
                max_position=self.config.max_position_embeddings,
                device=device,
                shared_cache=shared_cache,
            )

    def pack_qkv_(self) -> Step4Model:
        """Prepack every layer's TP-local Q/K/V weights one layer at a time."""
        for layer in self.layers:
            layer.self_attn.pack_qkv_()
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cache: Step4Cache,
        seq_lens: torch.Tensor,
        past_lens: torch.Tensor,
        *,
        is_decode: bool,
    ) -> torch.Tensor:
        residual = hidden_states.float()
        for layer in self.layers:
            residual = layer(
                residual, positions, cache, seq_lens, past_lens, is_decode=is_decode
            )
        # vLLM's compute_logits narrows the FP32 residual to the parameter dtype *before*
        # final RMSNorm. Normalising FP32 and casting only at lm_head is measurably different.
        norm_input = (
            residual.to(self.embed_tokens.weight.dtype)
            if self.config.fp32_residual_connection
            else residual
        )
        return self.norm(norm_input)


class Step4ForCausalLM(Step4PreTrainedModel):
    """step4 with a language-modelling head.

    Input is a packed batch: ``[total_tokens, ...]`` with per-request lengths, rather than a
    padded ``[batch, seq]``. That is the layout the DSA kernels want, and converting at the
    boundary means padding never reaches the cache -- a padded token written into the region
    summaries would be scored and selected like a real one.
    """

    _tied_weights_keys: list[str] = []

    def __init__(self, config: Step4Config) -> None:
        super().__init__(config)
        self.model = Step4Model(config)
        self.lm_head = VocabParallelLinear(
            config.hidden_size, config.vocab_size, tp_size=config.tp, bias=False, dtype=torch.bfloat16
        )
        self.post_init()

    def allocate_cache(self, *, batch: int, max_tokens: int, device: torch.device) -> Step4Cache:
        self.model.pack_qkv_()
        self.model.build_rope(device)
        return Step4Cache(
            self.config, batch=batch, max_tokens=max_tokens, device=device, dtype=torch.bfloat16
        )

    def pack_qkv_(self) -> Step4ForCausalLM:
        """Materialise the inference-only fused QKV layout after loading/device move."""
        self.model.pack_qkv_()
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Step4Cache,
        seq_lens: torch.Tensor,
        *,
        is_decode: bool = False,
        return_all_logits: bool = False,
    ) -> CausalLMOutputWithPast:
        """Run one prefill or one decode step over a packed batch.

        Args:
            input_ids: ``[total_tokens]`` int64, requests concatenated.
            seq_lens: ``[batch]`` int32, tokens contributed by each request.
            return_all_logits: emit a logit row per token instead of one per request. Off by
                default because at a 128896-wide vocabulary the prefill logits are larger
                than the activations that produced them.
        """
        past_lens = cache.advance(seq_lens)
        positions = _packed_positions(seq_lens, past_lens)
        hidden = self.model.embed_tokens(input_ids)
        hidden = self.model(
            hidden, positions, cache, seq_lens, past_lens, is_decode=is_decode
        )

        if not return_all_logits:
            hidden = hidden[_last_token_indices(seq_lens)]
        logits = self.lm_head(hidden.to(self.lm_head.weight.dtype))
        return CausalLMOutputWithPast(logits=logits.float(), past_key_values=cache)

    @torch.inference_mode()
    def generate_greedy(
        self,
        prompt_ids: list[list[int]],
        *,
        max_new_tokens: int,
        device: torch.device,
        eos_token_ids: set[int] | None = None,
    ) -> list[list[int]]:
        """Minimal greedy decode, so the model can be exercised without a serving stack.

        Deliberately not ``transformers.generate``: that expects a padded batch and a
        ``Cache`` it can slice, and adapting the DSA cache to that interface would mean
        pretending its region summaries are sliceable per token.
        """
        batch = len(prompt_ids)
        lengths = [len(ids) for ids in prompt_ids]
        cache = self.allocate_cache(
            batch=batch, max_tokens=max(lengths) + max_new_tokens, device=device
        )
        eos = eos_token_ids or set(
            self.config.eos_token_id
            if isinstance(self.config.eos_token_id, (list, tuple))
            else [self.config.eos_token_id]
        )

        flat = torch.tensor([t for ids in prompt_ids for t in ids], device=device, dtype=torch.long)
        seq_lens = torch.tensor(lengths, device=device, dtype=torch.int32)
        logits = self(flat, cache, seq_lens).logits

        outputs: list[list[int]] = [[] for _ in range(batch)]
        finished = [False] * batch
        ones = torch.ones((batch,), device=device, dtype=torch.int32)
        for step in range(max_new_tokens):
            # ``lm_head`` owns only a contiguous vocab slice on each TP rank.
            # A local argmax is merely an index into that slice and makes every
            # rank feed a different token into the next collective.  Reconstruct
            # the full vocabulary exactly as the DeepSeek reference inference
            # path does, so every rank advances the cache with the same global id.
            tokens = vocab_parallel_argmax(
                logits, expected_tp_size=self.config.tp
            )
            for request in range(batch):
                token = int(tokens[request])
                if finished[request]:
                    continue
                outputs[request].append(token)
                if token in eos:
                    finished[request] = True
            # ``logits`` already produced this iteration's token.  A decode
            # forward is needed only when another sampling iteration follows;
            # running it after the final requested token mutates the cache and
            # launches TP collectives whose result is never consumed.
            if all(finished) or step + 1 == max_new_tokens:
                break
            logits = self(tokens, cache, ones, is_decode=True).logits
        return outputs


def _packed_positions(seq_lens: torch.Tensor, past_lens: torch.Tensor) -> torch.Tensor:
    """Absolute position of every token in a packed batch."""
    device = seq_lens.device
    spans = [
        torch.arange(int(past), int(past) + int(length), device=device, dtype=torch.int32)
        for past, length in zip(past_lens.tolist(), seq_lens.tolist())
    ]
    return torch.cat(spans)


def _last_token_indices(seq_lens: torch.Tensor) -> torch.Tensor:
    ends = torch.cumsum(seq_lens.to(torch.long), dim=0)
    return ends - 1


__all__ = [
    "STEP4_TP_LAYOUT_VERSION",
    "Step4SparseConfig",
    "Step4Config",
    "set_tp_group",
    "tp_group",
    "tp_rank",
    "tp_size",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelLinear",
    "vocab_parallel_argmax",
    "DenseLayerCache",
    "Step4Attention",
    "Step4Cache",
    "Step4RMSNorm",
    "Step4DecoderLayer",
    "Step4ForCausalLM",
    "Step4MLP",
    "Step4MoE",
    "Step4Model",
    "clamped_swiglu",
]
