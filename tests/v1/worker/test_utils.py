# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import torch

from vllm.v1.worker.utils import (
    bind_kv_cache,
    copy_kv_cache_side_storage_blocks,
    prepare_kv_cache_side_storage_for_memory_profiling,
    reset_kv_cache_side_storage_runtime_state,
)


def test_bind_kv_cache(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    ctx = {
        "layers.0.self_attn": Attention(32, 128, 0.1, prefix="layers.0.self_attn"),
        "layers.1.self_attn": Attention(32, 128, 0.1, prefix="layers.1.self_attn"),
        "layers.2.self_attn": Attention(32, 128, 0.1, prefix="layers.2.self_attn"),
        "layers.3.self_attn": Attention(32, 128, 0.1, prefix="layers.3.self_attn"),
    }
    kv_cache = {
        "layers.0.self_attn": torch.zeros((1,)),
        "layers.1.self_attn": torch.zeros((1,)),
        "layers.2.self_attn": torch.zeros((1,)),
        "layers.3.self_attn": torch.zeros((1,)),
    }
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)
    assert ctx["layers.0.self_attn"].kv_cache is kv_cache["layers.0.self_attn"]
    assert ctx["layers.1.self_attn"].kv_cache is kv_cache["layers.1.self_attn"]
    assert ctx["layers.2.self_attn"].kv_cache is kv_cache["layers.2.self_attn"]
    assert ctx["layers.3.self_attn"].kv_cache is kv_cache["layers.3.self_attn"]

    assert runner_kv_caches[0] is kv_cache["layers.0.self_attn"]
    assert runner_kv_caches[1] is kv_cache["layers.1.self_attn"]
    assert runner_kv_caches[2] is kv_cache["layers.2.self_attn"]
    assert runner_kv_caches[3] is kv_cache["layers.3.self_attn"]


def test_kv_cache_side_storage_lifecycle_hooks(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    owner_name = "layers.0.self_attn"
    owner = Attention(32, 128, 0.1, prefix=owner_name)
    events: list[object] = []

    class SideStorage:
        main_layer_name = owner_name

        def bind_kv_cache_side_storage(self, forward_context):
            assert forward_context[owner_name].kv_cache is kv_cache[owner_name]
            events.append("bind")

        def zero_kv_cache_side_storage(self, block_ids):
            events.append(("zero", block_ids))

        def copy_kv_cache_side_storage(self, copies, num_blocks):
            events.append(("copy", copies, num_blocks))

        def reset_kv_cache_side_storage_runtime_state(self):
            events.append("reset")
            return 1, 2

    side_storage = SideStorage()
    ctx = {
        owner_name: owner,
        "layers.0.side": side_storage,
        # An alias must not execute lifecycle hooks twice.
        "layers.0.side_alias": side_storage,
    }
    kv_cache = {owner_name: torch.zeros((1,))}
    runner_kv_caches: list[torch.Tensor] = []

    bind_kv_cache(kv_cache, ctx, runner_kv_caches)
    copies = [(0, 1)]
    copy_kv_cache_side_storage_blocks(
        ctx,
        [SimpleNamespace(layer_names=[owner_name]), SimpleNamespace(layer_names=[])],
        [copies, [(2, 3)]],
        num_blocks=4,
    )
    reset_result = reset_kv_cache_side_storage_runtime_state(ctx)

    assert events == ["bind", ("copy", copies, 4), "reset"]
    assert reset_result == (1, 2)


def test_kv_side_storage_memory_profile_deduplicates_shared_owner():
    events: list[tuple[str, torch.device]] = []
    shared_workspace = object()

    class SideStorageMemoryProfiler:
        def __init__(self, name: str):
            self.name = name

        def kv_cache_side_storage_memory_profile_key(self, _forward_context):
            return shared_workspace

        def prepare_kv_cache_side_storage_memory_profile(
            self,
            _forward_context,
            device,
        ):
            events.append((self.name, device))

    first = SideStorageMemoryProfiler("first")
    second = SideStorageMemoryProfiler("second")
    device = torch.device("cpu")
    num_prepared = prepare_kv_cache_side_storage_for_memory_profiling(
        {
            "first": first,
            "first_alias": first,
            "second": second,
            "unrelated": object(),
        },
        device,
    )

    assert num_prepared == 1
    assert events == [("first", device)]


def test_gpu_model_runner_wake_resets_side_storage(monkeypatch):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    events: list[str] = []

    class SideStorage:
        main_layer_name = "owner"

        def bind_kv_cache_side_storage(self, forward_context):
            raise AssertionError("not used")

        def zero_kv_cache_side_storage(self, block_ids):
            raise AssertionError("not used")

        def copy_kv_cache_side_storage(self, block_copies, num_blocks):
            raise AssertionError("not used")

        def reset_kv_cache_side_storage_runtime_state(self):
            events.append("reset")
            return 1, 0

    runner = object.__new__(GPUModelRunner)
    runner.compilation_config = SimpleNamespace(
        static_forward_context={"side": SideStorage()}
    )
    runner.init_fp8_kv_scales = lambda: events.append("fp8")
    monkeypatch.setattr(
        torch.accelerator,
        "synchronize",
        lambda: events.append("sync"),
    )

    runner.post_kv_cache_wake_up()

    assert events == ["fp8", "reset", "sync"]


def test_attention_side_storage_contract_survives_runner_spec_replace(
    default_vllm_config,
):
    from vllm.model_executor.layers.attention import Attention

    attention = Attention(32, 128, 0.1, prefix="layers.0.self_attn")
    attention.kv_cache_requires_zeroing = True
    attention.kv_cache_extra_budget_page_size_bytes = 4096
    attention.kv_cache_extra_budget_fixed_size_bytes = 8192
    attention.kv_cache_prefix_recompute_tokens = 64

    spec = attention.get_kv_cache_spec(default_vllm_config)
    assert spec is not None
    assert spec.requires_zeroing
    assert spec.extra_budget_page_size_bytes == 4096
    assert spec.extra_budget_fixed_size_bytes == 8192
    assert spec.prefix_cache_recompute_tokens == 64

    # Both GPU runner implementations clone attention specs this way after
    # querying the backend's block-stride indexing capability.
    runner_spec = replace(spec, indexes_kv_by_block_stride=True)
    assert runner_spec.requires_zeroing
    assert runner_spec.extra_budget_page_size_bytes == 4096
    assert runner_spec.extra_budget_fixed_size_bytes == 8192
    assert runner_spec.prefix_cache_recompute_tokens == 64


def test_bind_kv_cache_non_attention(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    # example from Jamba PP=2
    ctx = {
        "model.layers.20.attn": Attention(32, 128, 0.1, prefix="model.layers.20.attn"),
        "model.layers.28.attn": Attention(32, 128, 0.1, prefix="model.layers.28.attn"),
    }
    kv_cache = {
        "model.layers.20.attn": torch.zeros((1,)),
        "model.layers.28.attn": torch.zeros((1,)),
    }

    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)

    assert ctx["model.layers.20.attn"].kv_cache is kv_cache["model.layers.20.attn"]
    assert ctx["model.layers.28.attn"].kv_cache is kv_cache["model.layers.28.attn"]

    assert runner_kv_caches[0] is kv_cache["model.layers.20.attn"]
    assert runner_kv_caches[1] is kv_cache["model.layers.28.attn"]


def test_bind_kv_cache_draft_model(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    layer_names = [
        "model.layers.0.attn",
        "model.layers.1.attn",
        "draft_model.layers.0.attn",
        "draft_model.layers.1.attn",
    ]
    ctx = {
        layer_name: Attention(32, 128, 0.1, prefix=layer_name)
        for layer_name in layer_names
    }
    kv_cache = {layer_name: torch.zeros((1,)) for layer_name in layer_names}
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)

    assert ctx["model.layers.0.attn"].kv_cache is kv_cache["model.layers.0.attn"]
    assert ctx["model.layers.1.attn"].kv_cache is kv_cache["model.layers.1.attn"]
    assert (
        ctx["draft_model.layers.0.attn"].kv_cache
        is kv_cache["draft_model.layers.0.attn"]
    )
    assert (
        ctx["draft_model.layers.1.attn"].kv_cache
        is kv_cache["draft_model.layers.1.attn"]
    )

    # caches are ordered by layer_index, interleaving target and draft model
    assert runner_kv_caches[0] is kv_cache["model.layers.0.attn"]
    assert runner_kv_caches[1] is kv_cache["draft_model.layers.0.attn"]
    assert runner_kv_caches[2] is kv_cache["model.layers.1.attn"]
    assert runner_kv_caches[3] is kv_cache["draft_model.layers.1.attn"]
