# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.model_loader.mtp_validation import (
    disable_mtp_completeness_check,
)
from vllm.model_executor.models.interfaces import is_mixture_of_experts
from vllm.models.step4 import kernels as step4_kernels
from vllm.models.step4 import layernorm as step4_layernorm
from vllm.models.step4 import model as step4_model
from vllm.models.step4 import mtp as step4_mtp
from vllm.models.step4.kernels import (
    checkpoint_has_step4_sparse_config,
    get_step4_sparse_config,
    is_supported_optimus_qknorm_cache_rotary,
)


def _make_fused_projection(*loaded_shards: str, has_gate: bool = True):
    projection = step4_model.Step4FusedQKVIndexerLinear.__new__(
        step4_model.Step4FusedQKVIndexerLinear
    )
    nn.Module.__init__(projection)
    projection.has_gate = has_gate
    projection._loaded_shards = set(loaded_shards)
    return projection


def test_step4_moe_layer_indices_accept_matching_config_aliases():
    config = SimpleNamespace(
        num_hidden_layers=4,
        num_nextn_predict_layers=1,
        moe_layers_enum="1,3",
        moe_layer_list=[3, 1],
    )

    assert step4_model._get_step4_moe_layer_indices(config) == {1, 3}


def test_step4_moe_layer_indices_falls_back_to_list():
    config = SimpleNamespace(
        num_hidden_layers=4,
        num_nextn_predict_layers=0,
        moe_layers_enum=None,
        moe_layer_list=[2],
    )

    assert step4_model._get_step4_moe_layer_indices(config) == {2}


def test_step4_moe_layer_indices_rejects_conflicting_aliases():
    config = SimpleNamespace(
        num_hidden_layers=4,
        num_nextn_predict_layers=0,
        moe_layers_enum="1,3",
        moe_layer_list=[1, 2],
    )

    with pytest.raises(ValueError, match="describe different layers"):
        step4_model._get_step4_moe_layer_indices(config)


def test_step4_pp_stage_without_local_moe_layers_is_valid():
    model = SimpleNamespace(moe_layers=[])

    step4_model._set_step4_moe_protocol_metadata(model, None)

    assert model.num_moe_layers == 0
    assert model.num_logical_experts == 0
    assert model.num_physical_experts == 0
    assert model.num_local_physical_experts == 0
    assert model.num_routed_experts == 0
    assert model.num_redundant_experts == 0


@pytest.mark.parametrize(
    ("tp_size", "dp_size", "fuse_all_reduce", "reduce_results"),
    [
        (1, 1, False, True),
        (4, 1, True, False),
        (4, 2, False, True),
    ],
)
def test_step4_moe_reduce_policy(
    tp_size,
    dp_size,
    fuse_all_reduce,
    reduce_results,
):
    assert step4_model._step4_moe_reduce_policy(tp_size, dp_size) == (
        fuse_all_reduce,
        reduce_results,
    )


class _FakeMoeRunner(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.eplb_calls = []
        self.update_expert_map_calls = 0

    def get_expert_weights(self):
        return [self.weight]

    def set_eplb_state(self, **kwargs):
        self.eplb_calls.append(kwargs)

    def update_expert_map(self):
        self.update_expert_map_calls += 1


def _make_mtp_with_optional_moe(*, use_moe: bool):
    mtp = object.__new__(step4_mtp.Step4MTP)
    nn.Module.__init__(mtp)
    mtp.model = nn.Module()

    predictor_layer = nn.Module()
    predictor_layer.mtp_block = nn.Module()
    predictor_layer.mtp_block.use_moe = use_moe
    if use_moe:
        moe = step4_model.FusedMoEBlock.__new__(step4_model.FusedMoEBlock)
        nn.Module.__init__(moe)
        moe.experts = _FakeMoeRunner()
        moe.n_logical_experts = 8
        moe.n_physical_experts = 10
        moe.n_local_physical_experts = 5
        moe.n_routed_experts = 8
        moe.n_redundant_experts = 2
        predictor_layer.mtp_block.moe = moe

    mtp.model.layers = nn.ModuleDict({"4": predictor_layer})
    moe_blocks = step4_mtp._get_step4_mtp_moe_blocks(mtp.model)
    mtp.moe_layers = [moe.experts for moe in moe_blocks]
    step4_model._set_step4_moe_protocol_metadata(
        mtp,
        moe_blocks[0] if moe_blocks else None,
    )
    return mtp


def test_dense_step4_mtp_is_not_registered_as_moe():
    mtp = _make_mtp_with_optional_moe(use_moe=False)

    assert mtp.moe_layers == []
    assert mtp.num_moe_layers == 0
    assert not mtp.model.layers["4"].mtp_block.use_moe
    assert step4_mtp._is_step4_mtp_dense(mtp.model)
    assert not is_mixture_of_experts(mtp)


def test_step4_mtp_dense_flag_is_false_when_a_draft_block_is_moe():
    mtp = _make_mtp_with_optional_moe(use_moe=True)

    assert not step4_mtp._is_step4_mtp_dense(mtp.model)


def test_step4_mtp_exposes_moe_runner_to_eplb_and_updates_metadata():
    mtp = _make_mtp_with_optional_moe(use_moe=True)
    runner = mtp.moe_layers[0]
    expert_load_view = torch.zeros(1)
    logical_to_physical_map = torch.zeros(1, dtype=torch.int32)
    logical_replica_count = torch.ones(1, dtype=torch.int32)

    assert is_mixture_of_experts(mtp)
    mtp.set_eplb_state(
        expert_load_view,
        logical_to_physical_map,
        logical_replica_count,
    )
    mtp.update_physical_experts_metadata(
        num_physical_experts=12,
        num_local_physical_experts=5,
    )

    assert mtp.expert_weights == [[runner.weight]]
    assert runner.eplb_calls == [
        {
            "moe_layer_idx": 0,
            "expert_load_view": expert_load_view,
            "logical_to_physical_map": logical_to_physical_map,
            "logical_replica_count": logical_replica_count,
        }
    ]
    assert runner.update_expert_map_calls == 1
    assert mtp.num_physical_experts == 12
    assert mtp.num_redundant_experts == 4


def test_step4_mtp_rejects_local_physical_expert_count_changes():
    mtp = _make_mtp_with_optional_moe(use_moe=True)

    with pytest.raises(ValueError, match="cannot change the number of local"):
        mtp.update_physical_experts_metadata(
            num_physical_experts=12,
            num_local_physical_experts=4,
        )


def test_step4_dsa_rejects_kv_transfer_before_cache_binding():
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_transfer_instance=True)
    )

    with pytest.raises(ValueError, match="not compatible with KV transfer"):
        step4_model._verify_step4_dsa_kv_transfer_compatibility(vllm_config)


@pytest.mark.parametrize("tp_size", [4, 8, 16])
def test_step4_dsa_parallel_geometry_accepts_production_layout(tp_size):
    step4_model._validate_step4_dsa_parallel_geometry(
        total_num_heads=64,
        total_num_kv_heads=4,
        indexer_num_heads=16,
        index_tp_size=4,
        tp_size=tp_size,
    )


@pytest.mark.parametrize(("tp_size", "local_groups"), [(1, 4), (2, 2)])
def test_step4_dsa_parallel_geometry_rejects_multiple_local_groups(
    tp_size,
    local_groups,
):
    with pytest.raises(
        ValueError,
        match=rf"exactly one local.*local_groups={local_groups}",
    ):
        step4_model._validate_step4_dsa_parallel_geometry(
            total_num_heads=64,
            total_num_kv_heads=4,
            indexer_num_heads=16,
            index_tp_size=4,
            tp_size=tp_size,
        )


def test_step4_tp16_provider_replication_matches_main_kv_geometry():
    mappings = [
        step4_model._step4_replicated_group_rank(
            total_groups=4,
            tp_size=16,
            tp_rank=rank,
        )
        for rank in range(16)
    ]

    assert mappings == [
        *((0, 4),) * 4,
        *((1, 4),) * 4,
        *((2, 4),) * 4,
        *((3, 4),) * 4,
    ]


def test_step4_dsa_parallel_geometry_rejects_provider_group_mismatch():
    with pytest.raises(ValueError, match="provider groups must align"):
        step4_model._validate_step4_dsa_parallel_geometry(
            total_num_heads=64,
            total_num_kv_heads=4,
            indexer_num_heads=16,
            index_tp_size=8,
            tp_size=8,
        )


def test_step4_dsa_parallel_geometry_rejects_incompatible_local_groups():
    with pytest.raises(ValueError, match="incompatible with group count"):
        step4_model._validate_step4_dsa_parallel_geometry(
            total_num_heads=64,
            total_num_kv_heads=3,
            indexer_num_heads=15,
            index_tp_size=3,
            tp_size=8,
        )


def test_fused_qkv_indexer_validation_rejects_missing_shard():
    module = nn.Module()
    module.projection = _make_fused_projection(
        "q",
        "k",
        "v",
        "index_q",
        "index_k",
        "index_z",
    )

    with pytest.raises(ValueError, match=r"missing shards \['index_g'\]"):
        step4_model._validate_fused_qkv_indexer_weights(module)


def test_fused_qkv_indexer_validation_reports_and_resets_complete_weight():
    module = nn.Module()
    module.projection = _make_fused_projection(
        "q",
        "k",
        "v",
        "index_q",
        "index_k",
        "index_z",
        "index_g",
    )

    assert step4_model._validate_fused_qkv_indexer_weights(module) == {
        "projection.weight"
    }
    assert module.projection._loaded_shards == set()


def test_fused_qkv_indexer_accepts_local_fused_checkpoint_weight():
    projection = _make_fused_projection()
    projection._fused_checkpoint_shard_sizes = ()
    param = nn.Parameter(torch.zeros(8, 4))
    loaded_weight = torch.arange(32, dtype=torch.float32).view(8, 4)

    projection.weight_loader_v2(param, loaded_weight)

    torch.testing.assert_close(param, loaded_weight)
    assert projection._loaded_shards == projection.required_shard_ids()


def test_fused_qkv_indexer_rejects_malformed_fused_checkpoint_weight():
    projection = _make_fused_projection()
    projection._fused_checkpoint_shard_sizes = (
        ("q", 2),
        ("k", 1),
        ("v", 1),
        ("index_q", 2),
        ("index_k", 1),
        ("index_z", 1),
        ("index_g", 1),
    )
    param = nn.Parameter(torch.zeros(8, 4))

    with pytest.raises(ValueError, match="expected local 8 or global 9"):
        projection.weight_loader_v2(param, torch.zeros(7, 4))


def test_step4_full_load_validates_but_weight_transfer_can_be_partial(monkeypatch):
    calls: list[str] = []

    class FakeStep4Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = _make_fused_projection()

    class FakeLoader:
        def __init__(self, _module, **_kwargs):
            pass

        def load_weights(self, _weights, *, mapper):
            del mapper
            calls.append("load")
            return set()

    model = SimpleNamespace(
        config=SimpleNamespace(tie_word_embeddings=False),
        hf_to_vllm_mapper=object(),
        model=FakeStep4Model(),
    )
    monkeypatch.setattr(step4_model, "AutoWeightsLoader", FakeLoader)

    with pytest.raises(ValueError, match="Incomplete fused qkv"):
        step4_model.Step4ForCausalLM.load_weights(model, [])

    with disable_mtp_completeness_check():
        assert step4_model.Step4ForCausalLM.load_weights(model, []) == set()
    assert calls == ["load", "load"]


def test_step4_sparse_config_prefers_native_section_and_keeps_legacy_alias():
    native = SimpleNamespace(
        step4_sparse_config={"enabled": False},
        step3p5_sparse_config={"enabled": True},
        sparse_config={"enabled": True},
    )
    legacy = SimpleNamespace(step3p5_sparse_config={"enabled": True})

    assert get_step4_sparse_config(native) is None
    assert get_step4_sparse_config(legacy) is not None


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (SimpleNamespace(step4_sparse_config={"enabled": True}), True),
        (SimpleNamespace(step4_sparse_config={}), True),
        # ``enabled`` is a runtime DSA switch; section presence still
        # advertises the split-KV/replay contract required by the checkpoint.
        (SimpleNamespace(step4_sparse_config={"enabled": False}), True),
        (SimpleNamespace(), False),
    ],
)
def test_step4_sparse_checkpoint_capability_tracks_declared_layout(config, expected):
    assert checkpoint_has_step4_sparse_config(config) is expected


def test_step4_sparse_checkpoint_capability_honors_force_enable_for_declared_section(
    monkeypatch,
):
    monkeypatch.setattr(step4_kernels.envs, "is_set", lambda _name: True)
    monkeypatch.setattr(
        step4_kernels.envs,
        "VLLM_STEP4_SPARSE",
        True,
        raising=False,
    )

    assert checkpoint_has_step4_sparse_config(
        SimpleNamespace(step4_sparse_config={"enabled": False})
    )
    assert not checkpoint_has_step4_sparse_config(SimpleNamespace())


def test_step4_sparse_force_enable_rejects_native_dense_checkpoint(monkeypatch):
    monkeypatch.setattr(step4_kernels.envs, "is_set", lambda _name: True)
    monkeypatch.setattr(
        step4_kernels.envs,
        "VLLM_STEP4_SPARSE",
        True,
        raising=False,
    )

    with pytest.raises(ValueError, match="requires a checkpoint-declared"):
        get_step4_sparse_config(SimpleNamespace())


def test_step4_sparse_config_retains_ssmax_compatibility_metadata():
    config = SimpleNamespace(
        step4_sparse_config={
            "enabled": True,
            "sparse_indexer_softmax_variant": "ssmax",
        }
    )

    sparse_config = get_step4_sparse_config(config)

    assert sparse_config is not None
    assert sparse_config.sparse_indexer_softmax_variant == "ssmax"


def test_step4_sparse_config_rejects_stable_topk_above_selector_limit():
    config = SimpleNamespace(
        step4_sparse_config={
            "enabled": True,
            "topk": 513,
        }
    )

    with pytest.raises(ValueError, match=r"stable top-k currently requires topk"):
        get_step4_sparse_config(config)


def test_step4_packed_mapping_exposes_dsa_partitions():
    assert step4_model.Step4ForCausalLM.packed_modules_mapping["qkv_indexer_proj"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "sparse_indexer_q",
        "sparse_indexer_k",
        "sparse_indexer_z",
        "g_proj",
    ]


def test_step4_model_requires_resolved_valid_vocab_size():
    unresolved = SimpleNamespace(valid_vocab_size=None)

    with pytest.raises(ValueError, match="resolve_valid_vocab_size"):
        step4_model._require_resolved_valid_vocab_size(unresolved)

    resolved = SimpleNamespace(
        valid_vocab_size=128815,
        get_valid_vocab_size=lambda: 128815,
    )
    assert step4_model._require_resolved_valid_vocab_size(resolved) == 128815


@pytest.mark.parametrize(
    "model_cls",
    [
        step4_model.Step4ForCausalLM,
        step4_mtp.Step4MTP,
    ],
)
def test_step4_quantized_models_enable_weight_tracking_by_default(model_cls):
    assert model_cls._enable_weights_track_by_default


def test_step4_mtp_local_argmax_selects_the_current_step_head():
    calls: list[tuple[object, torch.Tensor]] = []
    hidden_states = torch.ones(2, 3)

    class _SharedHead:
        def __init__(self, name):
            self.head = name

        def __call__(self, states):
            return states + 1

    predictor = SimpleNamespace(
        num_mtp_layers=2,
        mtp_start_layer_idx=10,
        layers={
            "10": SimpleNamespace(shared_head=_SharedHead("head-0")),
            "11": SimpleNamespace(shared_head=_SharedHead("head-1")),
        },
        logits_processor=SimpleNamespace(
            get_top_tokens=lambda head, states: (
                calls.append((head, states)),
                torch.tensor([7, 8]),
            )[1]
        ),
    )

    result = step4_mtp.Step4MultiTokenPredictor.get_top_tokens(
        predictor,
        hidden_states,
        spec_step_idx=3,
    )

    assert torch.equal(result, torch.tensor([7, 8]))
    assert len(calls) == 1
    assert calls[0][0] == "head-1"
    torch.testing.assert_close(calls[0][1], hidden_states + 1)


def test_step4_mtp_completeness_ignores_loaded_optional_and_extra_params():
    params_dict = {
        "required.weight": nn.Parameter(torch.zeros(1)),
        "attention.k_scale": nn.Parameter(
            torch.ones(1),
            requires_grad=False,
        ),
    }

    assert (
        step4_mtp._get_missing_required_mtp_params(
            params_dict,
            {
                "required.weight",
                "attention.k_scale",
                "checkpoint.alias",
            },
        )
        == set()
    )
    assert step4_mtp._get_missing_required_mtp_params(
        params_dict,
        {"attention.k_scale"},
    ) == {"required.weight"}


def test_step4_rms_norm_falls_back_without_closed_optimus_op(monkeypatch):
    monkeypatch.setenv("VLLM_STEP_CC_LEVEL", "0")
    monkeypatch.setattr(step4_layernorm, "_has_optimus_rms_norm_op", lambda: False)
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    weight = torch.tensor([0.25, -0.5], dtype=torch.float32)

    actual = step4_layernorm.apply_optimus_rms_norm(
        x,
        weight,
        1e-6,
        zero_centered=True,
    )
    expected = step4_layernorm._optimus_rms_norm_native(
        x,
        weight,
        1e-6,
        True,
    )

    torch.testing.assert_close(actual, expected)


def test_step4_fused_rms_norm_falls_back_without_custom_ops(monkeypatch):
    monkeypatch.setenv("VLLM_STEP_CC_LEVEL", "0")
    monkeypatch.setattr(step4_layernorm, "_has_optimus_rms_norm_op", lambda: False)
    monkeypatch.setattr(
        step4_layernorm,
        "_has_stepfun_fused_add_rms_norm_op",
        lambda: False,
    )
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    residual = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    weight = torch.tensor([0.25, -0.5], dtype=torch.float32)

    actual, residual_out = step4_layernorm.apply_optimus_fused_add_rms_norm(
        x,
        residual,
        weight,
        1e-6,
        zero_centered=True,
    )
    expected_residual = x + residual
    expected = step4_layernorm._optimus_rms_norm_native(
        expected_residual,
        weight,
        1e-6,
        True,
    )

    torch.testing.assert_close(residual_out, expected_residual)
    torch.testing.assert_close(actual, expected)


def test_step4_fused_rms_norm_fp16_fallback_preserves_residual_dtype(monkeypatch):
    monkeypatch.setenv("VLLM_STEP_CC_LEVEL", "0")
    monkeypatch.setattr(step4_layernorm, "_has_optimus_rms_norm_op", lambda: False)
    monkeypatch.setattr(
        step4_layernorm,
        "_has_stepfun_fused_add_rms_norm_op",
        lambda: False,
    )
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    residual = torch.tensor([[3.0, 4.0]], dtype=torch.float16)
    weight = torch.tensor([0.25, -0.5], dtype=torch.float32)

    actual, residual_out = step4_layernorm.apply_optimus_fused_add_rms_norm(
        x,
        residual,
        weight,
        1e-6,
        zero_centered=True,
    )
    expected_residual = (x.float() + residual.float()).to(torch.float16)
    expected = step4_layernorm._optimus_rms_norm_native(
        expected_residual,
        weight,
        1e-6,
        True,
    )

    assert residual_out.dtype == torch.float16
    assert actual.dtype == torch.float16
    torch.testing.assert_close(residual_out, expected_residual)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qknorm_cache_falls_back_when_stepfun_extension_op_is_missing(monkeypatch):
    device = torch.device("cuda")
    qkv = torch.zeros((1, 6), dtype=torch.bfloat16, device=device)
    q = torch.full((1, 2), 1, dtype=qkv.dtype, device=device)
    k = torch.full((1, 2), 2, dtype=qkv.dtype, device=device)
    v = torch.full((1, 2), 3, dtype=qkv.dtype, device=device)
    kv_cache = torch.zeros((2, 1, 4, 1, 2), dtype=qkv.dtype, device=device)
    slot_mapping = torch.zeros((1,), dtype=torch.int64, device=device)
    updates: list[tuple[torch.Tensor, torch.Tensor]] = []

    class FakeImpl:
        @staticmethod
        def do_kv_cache_update(
            _layer,
            key,
            value,
            _kv_cache,
            _slot_mapping,
        ):
            updates.append((key, value))

    attn_layer = SimpleNamespace(impl=FakeImpl())
    monkeypatch.setattr(
        "vllm.model_executor.layers.attention.attention.get_attention_context",
        lambda _layer_name: (None, attn_layer, kv_cache, slot_mapping),
    )
    monkeypatch.setattr(
        step4_kernels,
        "_get_stepfun_qknorm_cache_op",
        lambda: None,
    )
    monkeypatch.setattr(
        step4_kernels,
        "_fused_qknorm_rope_forward_impl",
        lambda *_args, **_kwargs: (q, k, v),
    )

    actual_q, actual_k, actual_v, _ = (
        step4_kernels._fused_qknorm_rope_cache_forward_impl_op(
            qkv,
            torch.ones(2, dtype=torch.float32, device=device),
            torch.ones(2, dtype=torch.float32, device=device),
            torch.ones((4, 1), dtype=qkv.dtype, device=device),
            torch.zeros((4, 1), dtype=qkv.dtype, device=device),
            torch.zeros((1,), dtype=torch.int64, device=device),
            head_dim=2,
            num_q_head=1,
            num_kv_head=1,
            rotary_dim=0,
            layer_name="layers.0.self_attn",
        )
    )

    assert actual_q is q
    assert actual_k is k
    assert actual_v is v
    assert len(updates) == 1
    torch.testing.assert_close(updates[0][0], k.view(1, 1, 2))
    torch.testing.assert_close(updates[0][1], v.view(1, 1, 2))


@pytest.mark.parametrize(
    ("cc_level", "op_name"),
    [
        (0, "optimus_fused_qknorm_rope_cache_bitwise"),
        (3, "optimus_fused_qknorm_rope_cache"),
    ],
)
def test_qknorm_cache_selects_stepfun_op_for_cc_level(
    monkeypatch,
    cc_level,
    op_name,
):
    selected_op = object()
    namespace = SimpleNamespace(**{op_name: selected_op})
    monkeypatch.setattr(
        step4_kernels,
        "torch",
        SimpleNamespace(ops=SimpleNamespace(_C=namespace)),
    )
    monkeypatch.setattr(
        step4_kernels,
        "envs",
        SimpleNamespace(VLLM_STEP_CC_LEVEL=cc_level),
    )

    assert step4_kernels._get_stepfun_qknorm_cache_op() is selected_op


@pytest.mark.parametrize("cc_level", [0, 3])
def test_qknorm_cache_selector_returns_none_when_extension_op_is_missing(
    monkeypatch,
    cc_level,
):
    monkeypatch.setattr(
        step4_kernels,
        "torch",
        SimpleNamespace(ops=SimpleNamespace(_C=SimpleNamespace())),
    )
    monkeypatch.setattr(
        step4_kernels,
        "envs",
        SimpleNamespace(VLLM_STEP_CC_LEVEL=cc_level),
    )

    assert step4_kernels._get_stepfun_qknorm_cache_op() is None


@pytest.mark.parametrize(
    ("head_dim", "rotary_pairs", "supported"),
    [
        (64, 32, True),
        (128, 64, True),
        (192, 64, True),
        (192, 96, True),
        (256, 128, True),
        (80, 40, False),
    ],
)
def test_qknorm_cache_rotary_layout_support(
    head_dim,
    rotary_pairs,
    supported,
):
    assert is_supported_optimus_qknorm_cache_rotary(head_dim, rotary_pairs) is supported


def test_router_bias_lazy_imports_model_scoped_kernel(
    monkeypatch,
):
    calls: list[dict] = []
    imported_modules: list[str] = []
    expected = (object(), object())

    def router(*_args, **kwargs):
        calls.append(kwargs)
        return expected

    def import_module(name):
        imported_modules.append(name)
        assert name == "vllm.models.step4.nvidia.ops.triton.router_bias"
        return SimpleNamespace(router_bias_triton_func=router)

    monkeypatch.setattr(step4_kernels.importlib, "import_module", import_module)

    actual = step4_kernels.router_bias_func(
        object(),
        object(),
        topk=2,
        renormalize=True,
        router_bias=object(),
        indices_dtype=torch.int64,
    )

    assert actual is expected
    assert imported_modules == ["vllm.models.step4.nvidia.ops.triton.router_bias"]
    assert calls == [
        {
            "renormalize": True,
            "routed_scaling_factor": 1.0,
            "nan_row_i_out": 0,
            "indices_dtype": torch.int64,
        }
    ]
