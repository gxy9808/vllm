# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import vllm.models.step4.sparse_attention as step4_sparse_attention
import vllm.models.step4.sparse_attention_mtp as step4_sparse_attention_mtp
from vllm.config import CompilationConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.models.step4.sparse_attention import (
    Step4DSAAttentionBackend,
    Step4DSAAttentionImpl,
    Step4DSAMetadataBuilder,
    Step4DSARuntimeLayout,
    _step4_use_flattened_decode_path,
    _validate_step4_dsa_attention_contract,
)
from vllm.models.step4.sparse_attention_mtp import (
    Step4DSAMTP,
    Step4DSAMTPTransaction,
    _step4_dsa_mtp_partition_decode_validity_kernel,
)
from vllm.models.step4.sparse_summary_cache import (
    Step4DSAScratchWorkspace,
    Step4SparseSummaryCache,
    Step4SparseSummaryCacheConfig,
    Step4SparseSummaryCacheLayer,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder


def test_step4_csa_clustered_decode_count_writes_guard_slot_sentinel():
    source_path = (
        Path(__file__).parents[3]
        / "vllm"
        / "models"
        / "step4"
        / "nvidia"
        / "ops"
        / "cute_dsl"
        / "sparse_gqa"
        / "indexer_ops"
        / "csa_compact_update_sm90_gqa.py"
    )
    tree = ast.parse(source_path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_csa_compact_decode_update_with_slots_cluster_valid_row"
    )

    def _guards_found_slot(guard: ast.expr) -> bool:
        return any(
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "found_slot"
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Lt)
            and len(node.comparators) == 1
            and isinstance(node.comparators[0], ast.Name)
            and node.comparators[0].id == "active_capacity"
            for node in ast.walk(guard)
        )

    class CountWriteGuardVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.guards: list[ast.expr] = []
            self.guard_results: list[bool] = []

        def visit_If(self, node: ast.If) -> None:
            self.guards.append(node.test)
            for statement in node.body:
                self.visit(statement)
            self.guards.pop()
            for statement in node.orelse:
                self.visit(statement)

        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "mCount"
                    and isinstance(target.slice, ast.Tuple)
                    and isinstance(target.slice.elts[0], ast.Name)
                    and target.slice.elts[0].id == "found_slot"
                ):
                    self.guard_results.append(
                        any(_guards_found_slot(guard) for guard in self.guards)
                    )

    visitor = CountWriteGuardVisitor()
    visitor.visit(function)
    assert visitor.guard_results == [True, True]


def test_step4_csa_prefill_prewarm_is_exported_from_public_interfaces():
    from importlib import import_module

    from vllm.models.step4.nvidia.ops.cute_dsl import sparse_gqa
    from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa import indexer_ops

    module_root = "vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa.indexer_ops"
    csa_compact_update_sm90_gqa = import_module(
        f"{module_root}.csa_compact_update_sm90_gqa"
    )
    interface = import_module(f"{module_root}.interface")

    export_name = "prewarm_csa_compact_prefill_update_with_slots_sm90_gqa"
    for module in (
        csa_compact_update_sm90_gqa,
        interface,
        indexer_ops,
        sparse_gqa,
    ):
        assert export_name in module.__all__
        assert callable(getattr(module, export_name))


def test_step4_dsa_backend_does_not_inherit_unsupported_flash_capabilities():
    assert Step4DSAAttentionBackend.is_sparse()
    assert Step4DSAAttentionBackend.supports_batch_invariance()
    assert Step4DSAAttentionBackend.supports_dtype(torch.float16)
    assert Step4DSAAttentionBackend.supports_dtype(torch.bfloat16)
    assert not Step4DSAAttentionBackend.supports_dtype(torch.float32)
    assert Step4DSAAttentionBackend.supports_head_size(128)
    assert Step4DSAAttentionBackend.supports_head_size(192)
    assert not Step4DSAAttentionBackend.supports_head_size(64)
    assert not Step4DSAAttentionBackend.supports_head_size(256)
    assert not Step4DSAAttentionBackend.supports_sliding_window()
    assert not Step4DSAAttentionBackend.supports_non_causal()
    assert Step4DSAAttentionBackend.supports_attn_type(AttentionType.DECODER)
    assert not Step4DSAAttentionBackend.supports_attn_type(AttentionType.ENCODER)
    assert not Step4DSAAttentionBackend.supports_sink()
    assert not Step4DSAAttentionBackend.supports_mm_prefix()
    assert not Step4DSAAttentionBackend.supports_per_head_quant_scales()
    assert not Step4DSAAttentionBackend.supports_kv_connector()
    assert not Step4DSAAttentionBackend.supports_compute_capability(
        DeviceCapability(8, 9)
    )
    assert Step4DSAAttentionBackend.supports_compute_capability(DeviceCapability(9, 0))
    assert not Step4DSAAttentionBackend.supports_compute_capability(
        DeviceCapability(10, 0)
    )


def test_step4_dsa_attention_is_a_piecewise_compilation_split_op():
    split_op = "vllm::step4_dsa_attention_with_output"
    compilation_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
    )

    compilation_config.set_splitting_ops_for_v1(
        all2all_backend="allgather_reducescatter",
    )

    assert split_op in CompilationConfig._attention_ops
    assert split_op in compilation_config.splitting_ops
    assert compilation_config.splitting_ops_contain_attention()


@pytest.mark.parametrize(
    ("decode_query_len", "expected_mode"),
    [
        (1, CUDAGraphMode.FULL_AND_PIECEWISE),
        (4, CUDAGraphMode.PIECEWISE),
    ],
)
def test_step4_dsa_routes_multirow_verifiers_out_of_full_cudagraph(
    decode_query_len: int,
    expected_mode: CUDAGraphMode,
):
    support = Step4DSAMetadataBuilder.get_cudagraph_support(None, None)
    assert support == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    compilation_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        splitting_ops=list(CompilationConfig()._attention_ops),
    )
    resolved_mode = compilation_config.resolve_cudagraph_mode_and_sizes(
        support,
        Step4DSAAttentionBackend.get_name(),
        uniform_decode_query_len=decode_query_len,
        use_v2_model_runner=True,
    )

    assert resolved_mode == expected_mode


def _step4_phase_common(
    query_lens: list[int],
    is_prefilling: list[bool],
) -> SimpleNamespace:
    query_start_loc_cpu = torch.tensor(
        [0, *torch.tensor(query_lens, dtype=torch.int32).cumsum(0).tolist()],
        dtype=torch.int32,
    )
    return SimpleNamespace(
        num_reqs=len(query_lens),
        num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens, default=0),
        query_start_loc_cpu=query_start_loc_cpu,
        _seq_lens_cpu=torch.tensor(
            [query_len + 64 for query_len in query_lens],
            dtype=torch.int32,
        ),
        is_prefilling=torch.tensor(is_prefilling, dtype=torch.bool),
    )


def _build_step4_phase_metadata(
    monkeypatch,
    *,
    query_lens: list[int],
    is_prefilling: list[bool],
    mtp: bool,
    reorder_batch_threshold: int,
):
    """Exercise Step4's phase contract without constructing CUDA state."""
    monkeypatch.setattr(
        FlashAttentionMetadataBuilder,
        "build",
        lambda self, common_prefix_len, common_attn_metadata, fast_build=False: (
            SimpleNamespace(max_query_len=common_attn_metadata.max_query_len)
        ),
    )
    builder = object.__new__(Step4DSAMetadataBuilder)
    builder._dsa_valid_requests = torch.zeros(1, dtype=torch.int32)
    builder._dsa_valid_tokens = torch.zeros(1, dtype=torch.int32)
    builder._mtp_enabled = mtp
    builder.reorder_batch_threshold = reorder_batch_threshold
    return builder.build(
        common_prefix_len=0,
        common_attn_metadata=_step4_phase_common(query_lens, is_prefilling),
    )


@pytest.mark.parametrize(
    ("query_lens", "is_prefilling", "expected_decodes", "expected_prefills"),
    [
        pytest.param([1, 1], [False, False], 2, 0, id="q1-decode"),
        pytest.param([1, 2], [False, True], 1, 1, id="q1-plus-short-prefill"),
        pytest.param([2], [True], 0, 1, id="short-prefill"),
    ],
)
def test_step4_no_mtp_phase_metadata_matches_develop_boundary(
    monkeypatch,
    query_lens: list[int],
    is_prefilling: list[bool],
    expected_decodes: int,
    expected_prefills: int,
):
    metadata = _build_step4_phase_metadata(
        monkeypatch,
        query_lens=query_lens,
        is_prefilling=is_prefilling,
        mtp=False,
        reorder_batch_threshold=1,
    )

    assert metadata.num_decodes == expected_decodes
    assert metadata.num_prefills == expected_prefills
    assert metadata.dsa_short_decode_reqs == expected_decodes
    assert metadata.dsa_num_verifier_reqs == 0
    # Absence is intentional: Step4 uses it to fall back to num_decodes.
    assert not hasattr(metadata, "mtp_num_verifier_reqs")


def test_step4_empty_phase_metadata_is_well_defined(monkeypatch):
    metadata = _build_step4_phase_metadata(
        monkeypatch,
        query_lens=[],
        is_prefilling=[],
        mtp=False,
        reorder_batch_threshold=1,
    )

    assert metadata.num_decodes == 0
    assert metadata.num_prefills == 0
    assert metadata.dsa_short_decode_reqs == 0
    assert metadata.dsa_num_verifier_reqs == 0
    assert not hasattr(metadata, "mtp_num_verifier_reqs")


def test_step4_mtp_phase_metadata_keeps_verifier_prefix_separate(monkeypatch):
    metadata = _build_step4_phase_metadata(
        monkeypatch,
        query_lens=[1, 2],
        is_prefilling=[False, False],
        mtp=True,
        reorder_batch_threshold=4,
    )

    assert metadata.dsa_short_decode_reqs == 2
    assert metadata.dsa_num_verifier_reqs == 1
    assert metadata.mtp_num_verifier_reqs == 2
    # Step4 intentionally routes multi-row MTP through the mixed PIECEWISE path.
    assert metadata.num_decodes == 0


@pytest.mark.parametrize(
    (
        "batch_invariant",
        "max_query_len",
        "num_actual_tokens",
        "num_reqs",
        "num_decode_reqs",
        "num_verifier_reqs",
        "expected",
    ),
    [
        # A pure MTP verifier batch uses the bounded decode path.
        (False, 4, 8, 2, 0, 2, True),
        # A verifier prefix mixed with a prefill is split by request and token.
        (False, 4, 11, 3, 0, 2, False),
        # Ordinary q1 decode remains on the bounded decode fast path.
        (False, 1, 2, 2, 2, 0, True),
        # Non-MTP short decode behavior is unchanged.
        (False, 4, 8, 2, 2, 0, True),
        # A short prefill must preserve the prefill contract in every mode.
        (False, 1, 2, 2, 0, 0, False),
        (False, 4, 8, 2, 0, 0, False),
        # Mixed decode/prefill batches are split below.
        (False, 1, 2, 2, 1, 0, False),
        # Invariant mode keeps true decode and MTP verifier rows on decode.
        (True, 1, 2, 2, 2, 0, True),
        (True, 4, 8, 2, 0, 2, True),
        # A short ordinary prefill must preserve the prefill contract.
        (True, 1, 2, 2, 0, 0, False),
        (True, 4, 8, 2, 0, 0, False),
        # Mixed decode/prefill batches are split below.
        (True, 1, 2, 2, 1, 0, False),
    ],
)
def test_step4_dsa_flattened_decode_routing(
    monkeypatch,
    batch_invariant: bool,
    max_query_len: int,
    num_actual_tokens: int,
    num_reqs: int,
    num_decode_reqs: int,
    num_verifier_reqs: int,
    expected: bool,
):
    monkeypatch.setattr(
        step4_sparse_attention.envs,
        "VLLM_BATCH_INVARIANT",
        batch_invariant,
    )
    assert (
        _step4_use_flattened_decode_path(
            max_query_len=max_query_len,
            num_actual_tokens=num_actual_tokens,
            num_reqs=num_reqs,
            num_decode_reqs=num_decode_reqs,
            num_verifier_reqs=num_verifier_reqs,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("num_decode_reqs", "num_verifier_reqs"),
    [
        (-1, 0),
        (3, 0),
        (0, -1),
        (0, 3),
    ],
)
def test_step4_dsa_flattened_decode_routing_rejects_invalid_prefix(
    num_decode_reqs: int,
    num_verifier_reqs: int,
):
    with pytest.raises(RuntimeError, match="outside the live request range"):
        _step4_use_flattened_decode_path(
            max_query_len=4,
            num_actual_tokens=8,
            num_reqs=2,
            num_decode_reqs=num_decode_reqs,
            num_verifier_reqs=num_verifier_reqs,
        )


@pytest.mark.parametrize(
    ("batch_invariant", "num_decodes", "num_verifiers", "expected"),
    [
        (False, 0, None, False),
        (False, 2, None, True),
        (False, 0, 2, True),
        (False, 0, 1, False),
        (True, 0, None, False),
        (True, 2, None, True),
        (True, 0, 2, True),
        (True, 0, 1, False),
    ],
)
def test_step4_short_prefill_summary_update_matches_attention_routing(
    monkeypatch,
    batch_invariant: bool,
    num_decodes: int,
    num_verifiers: int | None,
    expected: bool,
):
    monkeypatch.setattr(
        step4_sparse_attention.envs,
        "VLLM_BATCH_INVARIANT",
        batch_invariant,
    )
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        max_query_len=1,
        num_decodes=num_decodes,
    )
    if num_verifiers is not None:
        metadata.mtp_num_verifier_reqs = num_verifiers

    assert (
        Step4DSAAttentionImpl._use_decode_summary_update(
            attn_metadata=metadata,
            num_actual_tokens=2,
        )
        is expected
    )


def test_step4_prefill_union_groups_accept_mtp_verifier_token_prefix():
    query_start_loc_cpu = torch.tensor(
        [0, 4, 8, 25, 56, 4153],
        dtype=torch.int32,
    )

    groups = Step4DSAAttentionImpl._prefill_tile_union_group_counts(
        query_start_loc_cpu=query_start_loc_cpu,
        num_decode_reqs=2,
        num_decode_tokens=8,
        num_prefill_reqs=3,
        total_prefill_tokens=4145,
        tile_capacity=4096,
        q_group=16,
    )

    assert groups == (257, 4)


def test_step4_mixed_verifier_prefill_uses_separate_request_and_token_prefixes(
    monkeypatch,
):
    impl = object.__new__(Step4DSAAttentionImpl)
    impl.sparse_region_block_size = 8
    decode_calls: list[dict[str, object]] = []
    prefill_calls: list[dict[str, object]] = []
    flatten_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        impl,
        "_sparse_gqa_kv_cache_for_kernel",
        lambda kv_cache: (kv_cache[0], kv_cache[1]),
    )

    def flatten(**kwargs):
        flatten_calls.append(kwargs)
        num_tokens = int(kwargs["num_tokens"])
        pages = int(kwargs["block_table"].shape[1])
        return (
            torch.arange(num_tokens + 1, dtype=torch.int32),
            torch.arange(num_tokens, dtype=torch.int32),
            torch.zeros((num_tokens, pages), dtype=torch.int32),
            torch.tensor([0, 1, 1, 1, 1], dtype=torch.int32),
        )

    monkeypatch.setattr(impl, "_flatten_decode_q_le4_metadata", flatten)
    monkeypatch.setattr(
        impl,
        "_forward_sparse_gqa_cutedsl_decode",
        lambda **kwargs: decode_calls.append(kwargs) or kwargs["output"],
    )
    monkeypatch.setattr(
        impl,
        "_forward_sparse_gqa_prefill_tiles",
        lambda **kwargs: prefill_calls.append(kwargs),
    )

    query_start_loc = torch.tensor([0, 1, 5, 8], dtype=torch.int32)
    metadata = SimpleNamespace(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        seq_lens=torch.tensor([17, 20, 3], dtype=torch.int32),
        block_table=torch.zeros((3, 2), dtype=torch.int32),
        slot_mapping=torch.arange(8),
        num_actual_reqs=3,
        num_actual_tokens=8,
        max_query_len=4,
        max_seq_len=20,
        num_decodes=0,
        mtp_num_verifier_reqs=2,
        dsa_valid_requests=torch.tensor([3], dtype=torch.int32),
        dsa_valid_tokens=torch.tensor([8], dtype=torch.int32),
    )
    query = torch.zeros((8, 1, 4))
    output = torch.empty_like(query)
    proxy_query = torch.zeros((8, 1, 1, 4))
    proxy_weights = torch.zeros((8, 1, 1))

    impl._forward_sparse_gqa_cutedsl(
        query=query,
        kv_cache=torch.zeros((2, 1)),
        attn_metadata=metadata,
        output=output,
        summary_cache=object(),
        proxy_query=proxy_query,
        proxy_weights=proxy_weights,
        step_metadata=step4_sparse_attention.Step4DSAStepMetadata(),
    )

    assert len(flatten_calls) == 1
    assert flatten_calls[0]["num_tokens"] == 5
    assert flatten_calls[0]["query_start_loc"].tolist() == [0, 1, 5]
    assert len(decode_calls) == 1
    assert decode_calls[0]["query"].shape[0] == 5
    assert decode_calls[0]["slot_mapping"].tolist() == [0, 1, 2, 3, 4]
    assert len(prefill_calls) == 1
    assert prefill_calls[0]["num_decode_reqs"] == 2
    assert prefill_calls[0]["num_decode_tokens"] == 5
    assert prefill_calls[0]["num_prefill_reqs"] == 1
    assert prefill_calls[0]["total_prefill_tokens"] == 3


@pytest.mark.parametrize("batch_invariant", [False, True])
@pytest.mark.parametrize("prefill_query_len", [3, 4])
def test_step4_mixed_q1_short_prefill_uses_prefill_suffix(
    monkeypatch,
    batch_invariant: bool,
    prefill_query_len: int,
):
    monkeypatch.setattr(
        step4_sparse_attention.envs,
        "VLLM_BATCH_INVARIANT",
        batch_invariant,
    )
    impl = object.__new__(Step4DSAAttentionImpl)
    impl.sparse_region_block_size = 8
    decode_calls: list[dict[str, object]] = []
    prefill_calls: list[dict[str, object]] = []
    flatten_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        impl,
        "_sparse_gqa_kv_cache_for_kernel",
        lambda kv_cache: (kv_cache[0], kv_cache[1]),
    )
    monkeypatch.setattr(
        impl,
        "_flatten_decode_q_le4_metadata",
        lambda **kwargs: flatten_calls.append(kwargs),
    )
    monkeypatch.setattr(
        impl,
        "_forward_sparse_gqa_cutedsl_decode",
        lambda **kwargs: decode_calls.append(kwargs) or kwargs["output"],
    )
    monkeypatch.setattr(
        impl,
        "_forward_sparse_gqa_prefill_tiles",
        lambda **kwargs: prefill_calls.append(kwargs),
    )

    num_actual_tokens = 1 + prefill_query_len
    query_start_loc = torch.tensor(
        [0, 1, num_actual_tokens],
        dtype=torch.int32,
    )
    metadata = SimpleNamespace(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        seq_lens=torch.tensor([17, prefill_query_len], dtype=torch.int32),
        block_table=torch.zeros((2, 2), dtype=torch.int32),
        slot_mapping=torch.arange(num_actual_tokens),
        num_actual_reqs=2,
        num_actual_tokens=num_actual_tokens,
        max_query_len=prefill_query_len,
        max_seq_len=17,
        num_decodes=0,
        mtp_num_verifier_reqs=1,
        dsa_valid_requests=torch.tensor([2], dtype=torch.int32),
        dsa_valid_tokens=torch.tensor([num_actual_tokens], dtype=torch.int32),
    )
    query = torch.arange(num_actual_tokens * 4, dtype=torch.float32).view(
        num_actual_tokens, 1, 4
    )
    output = torch.empty_like(query)
    proxy_query = torch.arange(
        num_actual_tokens * 4,
        dtype=torch.float32,
    ).view(num_actual_tokens, 1, 1, 4)
    proxy_weights = torch.arange(
        num_actual_tokens,
        dtype=torch.float32,
    ).view(num_actual_tokens, 1, 1)

    impl._forward_sparse_gqa_cutedsl(
        query=query,
        kv_cache=torch.zeros((2, 1)),
        attn_metadata=metadata,
        output=output,
        summary_cache=object(),
        proxy_query=proxy_query,
        proxy_weights=proxy_weights,
        step_metadata=step4_sparse_attention.Step4DSAStepMetadata(),
    )

    assert flatten_calls == []
    assert len(decode_calls) == 1
    decode = decode_calls[0]
    torch.testing.assert_close(decode["query"], query[:1])
    torch.testing.assert_close(decode["proxy_query"], proxy_query[:1])
    torch.testing.assert_close(decode["proxy_weights"], proxy_weights[:1])
    assert decode["slot_mapping"].tolist() == [0]
    assert decode["query_start_loc"].tolist() == [0, 1]

    assert len(prefill_calls) == 1
    prefill = prefill_calls[0]
    assert prefill["num_decode_reqs"] == 1
    assert prefill["num_decode_tokens"] == 1
    assert prefill["num_prefill_reqs"] == 1
    assert prefill["total_prefill_tokens"] == prefill_query_len


@pytest.mark.parametrize("batch_invariant", [False, True])
@pytest.mark.parametrize("prefill_query_len", [3, 4])
def test_step4_pure_short_prefill_uses_prefill_dispatch(
    monkeypatch,
    batch_invariant: bool,
    prefill_query_len: int,
):
    monkeypatch.setattr(
        step4_sparse_attention.envs,
        "VLLM_BATCH_INVARIANT",
        batch_invariant,
    )
    impl = object.__new__(Step4DSAAttentionImpl)
    impl.sparse_region_block_size = 8
    decode_calls: list[dict[str, object]] = []
    prefill_calls: list[dict[str, object]] = []
    flatten_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        impl,
        "_sparse_gqa_kv_cache_for_kernel",
        lambda kv_cache: (kv_cache[0], kv_cache[1]),
    )

    def flatten(**kwargs):
        flatten_calls.append(kwargs)
        num_tokens = int(kwargs["num_tokens"])
        pages = int(kwargs["block_table"].shape[1])
        return (
            torch.arange(num_tokens + 1, dtype=torch.int32),
            torch.arange(num_tokens, dtype=torch.int32),
            torch.zeros((num_tokens, pages), dtype=torch.int32),
            torch.zeros((num_tokens,), dtype=torch.int32),
        )

    monkeypatch.setattr(impl, "_flatten_decode_q_le4_metadata", flatten)
    monkeypatch.setattr(
        impl,
        "_forward_sparse_gqa_cutedsl_decode",
        lambda **kwargs: decode_calls.append(kwargs) or kwargs["output"],
    )
    monkeypatch.setattr(
        impl,
        "_forward_sparse_gqa_prefill_tiles",
        lambda **kwargs: prefill_calls.append(kwargs),
    )

    query_start_loc = torch.tensor(
        [0, prefill_query_len],
        dtype=torch.int32,
    )
    metadata = SimpleNamespace(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        seq_lens=torch.tensor([prefill_query_len], dtype=torch.int32),
        block_table=torch.zeros((1, 2), dtype=torch.int32),
        slot_mapping=torch.arange(prefill_query_len),
        num_actual_reqs=1,
        num_actual_tokens=prefill_query_len,
        max_query_len=prefill_query_len,
        max_seq_len=prefill_query_len,
        num_decodes=0,
        mtp_num_verifier_reqs=0,
        dsa_valid_requests=torch.tensor([1], dtype=torch.int32),
        dsa_valid_tokens=torch.tensor([prefill_query_len], dtype=torch.int32),
    )
    query = torch.zeros((prefill_query_len, 1, 4))
    output = torch.empty_like(query)
    proxy_query = torch.zeros((prefill_query_len, 1, 1, 4))
    proxy_weights = torch.zeros((prefill_query_len, 1, 1))

    impl._forward_sparse_gqa_cutedsl(
        query=query,
        kv_cache=torch.zeros((2, 1)),
        attn_metadata=metadata,
        output=output,
        summary_cache=object(),
        proxy_query=proxy_query,
        proxy_weights=proxy_weights,
        step_metadata=step4_sparse_attention.Step4DSAStepMetadata(),
    )

    assert flatten_calls == []
    assert decode_calls == []
    assert len(prefill_calls) == 1
    prefill = prefill_calls[0]
    assert prefill["num_decode_reqs"] == 0
    assert prefill["num_decode_tokens"] == 0
    assert prefill["num_prefill_reqs"] == 1
    assert prefill["total_prefill_tokens"] == prefill_query_len


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"dtype": torch.float32}, "float16 or bfloat16"),
        ({"head_size": 64}, "head_size"),
        ({"q_heads_per_kv": 2}, "q_heads_per_kv"),
        ({"alibi_slopes": torch.ones(1)}, "ALiBi"),
        ({"sliding_window": (511, 0)}, "sliding-window"),
        ({"logits_soft_cap": 10.0}, "soft cap"),
        ({"attn_type": AttentionType.ENCODER}, "causal decoder"),
        ({"kv_sharing_target_layer_name": "shared"}, "KV cache sharing"),
        ({"sinks": torch.ones(1)}, "attention sinks"),
    ],
)
def test_step4_dsa_attention_contract_fails_fast(override, error):
    kwargs = {
        "dtype": torch.bfloat16,
        "head_size": 192,
        "q_heads_per_kv": 16,
        "alibi_slopes": None,
        "sliding_window": (-1, -1),
        "logits_soft_cap": 0.0,
        "attn_type": AttentionType.DECODER,
        "kv_sharing_target_layer_name": None,
        "sinks": None,
    }
    kwargs.update(override)
    with pytest.raises((ValueError, NotImplementedError), match=error):
        _validate_step4_dsa_attention_contract(**kwargs)


def test_step4_dsa_runtime_state_reset_is_complete_and_in_place():
    proxy_dim = 4
    config = Step4SparseSummaryCacheConfig(
        num_pages=3,
        page_size=16,
        region_block_size=8,
        num_kv_heads=1,
        proxy_dim=proxy_dim,
    )
    cache = Step4SparseSummaryCache(
        config=config,
        sum_cache=torch.full((4, 1, 1, proxy_dim), 7.0),
        count_cache=torch.full((4, 1, 1), 7.0),
        mean_cache=torch.full(config.sum_shape, 7, dtype=torch.uint8),
    )
    cache._step4_csa_active_region_ids = torch.full((4,), 7, dtype=torch.long)
    cache._step4_csa_active_slot_by_region = torch.full((6,), 7, dtype=torch.int32)
    cache._step4_csa_allocation_success = torch.zeros((1,), dtype=torch.int32)
    cache._step4_csa_numerator_cache = torch.full((4, 1, proxy_dim), 7.0)
    cache._step4_csa_denominator_cache = torch.full((4, 1, proxy_dim), 7.0)
    cache._step4_csa_max_cache = torch.full((4, 1), 7.0)
    cache._step4_csa_active_token_k = torch.full((2, 8, 1, proxy_dim), 7.0)
    cache._step4_csa_active_token_z = torch.full_like(
        cache._step4_csa_active_token_k, 7.0
    )
    cache._step4_csa_active_token_valid = torch.ones((2, 8), dtype=torch.uint8)

    transaction = Step4DSAMTPTransaction(
        max_num_reqs=2,
        max_rows_per_req=4,
        num_kv_heads=1,
        proxy_dim=proxy_dim,
        source_map_rows=16,
        active_capacity=4,
        device=torch.device("cpu"),
    )
    for value in vars(transaction).values():
        if isinstance(value, torch.Tensor):
            value.fill_(7)
    cache._step4_mtp_transaction = transaction

    workspace = Step4DSAScratchWorkspace(
        tensor_buffers_by_engine={
            0: {
                "csa_prefill_scratch_region_ids": torch.full(
                    (4,), 7, dtype=torch.int32
                ),
                "csa_prefill_scratch_row_map": torch.full((8,), 7, dtype=torch.int32),
                "csa_prefill_scratch_reset_map": torch.full((4,), 7, dtype=torch.int32),
                "csa_padded_index_k": torch.full((8,), 7.0),
            }
        }
    )

    pointers = {
        "mean": cache.mean_cache.data_ptr(),
        "active": cache._step4_csa_active_region_ids.data_ptr(),
        "transaction": transaction.row_regions.data_ptr(),
        "scratch": workspace.tensor_buffers_by_engine[0][
            "csa_padded_index_k"
        ].data_ptr(),
    }

    cache.reset_runtime_state()
    assert workspace.reset_runtime_state() == 4

    assert not cache.sum_cache.any()
    assert not cache.count_cache.any()
    assert not cache.mean_cache.any()
    assert cache._step4_csa_active_region_ids.eq(-1).all()
    assert cache._step4_csa_active_slot_by_region.eq(-1).all()
    assert cache._step4_csa_allocation_success.eq(1).all()
    assert not cache._step4_csa_numerator_cache.any()
    assert not cache._step4_csa_denominator_cache.any()
    assert torch.isneginf(cache._step4_csa_max_cache).all()
    assert not cache._step4_csa_active_token_k.any()
    assert not cache._step4_csa_active_token_z.any()
    assert not cache._step4_csa_active_token_valid.any()

    for name in (
        "row_regions",
        "row_positions",
        "row_source",
        "row_owner_block",
        "row_owner_block_index",
        "source_to_transaction",
    ):
        assert getattr(transaction, name).eq(-1).all()
    assert not transaction.correction_action.any()
    assert not transaction.state_numerator.any()
    assert not transaction.state_denominator.any()
    assert torch.isneginf(transaction.state_max_logits).all()
    assert not transaction.correction_free_count.any()
    assert not transaction.correction_allocation_cursor.any()

    scratch = workspace.tensor_buffers_by_engine[0]
    assert scratch["csa_prefill_scratch_region_ids"].eq(-1).all()
    assert scratch["csa_prefill_scratch_row_map"].eq(-1).all()
    assert not scratch["csa_prefill_scratch_reset_map"].any()
    assert not scratch["csa_padded_index_k"].any()

    assert cache.mean_cache.data_ptr() == pointers["mean"]
    assert cache._step4_csa_active_region_ids.data_ptr() == pointers["active"]
    assert transaction.row_regions.data_ptr() == pointers["transaction"]
    assert scratch["csa_padded_index_k"].data_ptr() == pointers["scratch"]


def test_step4_dsa_order_tokens_are_fixed_and_reset_in_place():
    workspace = Step4DSAScratchWorkspace()

    token0 = workspace.get_order_token(0, device=torch.device("cpu"))
    token1 = workspace.get_order_token(1, device=torch.device("cpu"))
    assert token0.dtype is torch.int64
    assert token1.dtype is torch.int64
    token0.fill_(7)
    token1.fill_(9)
    workspace.allocations_locked = True

    assert workspace.get_order_token(0, device=torch.device("cpu")) is token0
    assert workspace.get_order_token(1, device=torch.device("cpu")) is token1
    with pytest.raises(RuntimeError, match="ordering-token capacity"):
        workspace.get_order_token(2, device=torch.device("cpu"))

    assert workspace.reset_runtime_state() == 0
    assert token0.item() == 0
    assert token1.item() == 0


def test_step4_csa_allocation_status_resets_and_fails_closed():
    cache = SimpleNamespace(
        sum_cache=torch.empty((1,), dtype=torch.float32),
        _step4_csa_allocation_success=torch.zeros((1,), dtype=torch.int32),
    )

    success = Step4DSAAttentionImpl._begin_csa_allocation_check(cache)

    assert success is cache._step4_csa_allocation_success
    assert success.eq(1).all()
    Step4DSAAttentionImpl._assert_csa_allocation_success(success)

    success.zero_()
    with pytest.raises(RuntimeError, match="active-slot capacity is exhausted"):
        Step4DSAAttentionImpl._assert_csa_allocation_success(success)


def test_step4_triton_fallback_allocation_status_is_optional_and_validated():
    from vllm.models.step4.sparse_summary_cache import (
        _prepare_csa_fallback_allocation_status,
    )

    status = torch.zeros((1,), dtype=torch.int32)
    prepared = _prepare_csa_fallback_allocation_status(
        status,
        device=torch.device("cpu"),
    )
    assert prepared is status
    assert status.item() == 1

    # Legacy custom-op callers may omit the new status argument.  The fallback
    # allocates a graph-compatible one-element device tensor in that case.
    prepared = _prepare_csa_fallback_allocation_status(
        None,
        device=torch.device("cpu"),
    )
    assert prepared.shape == (1,)
    assert prepared.dtype is torch.int32
    assert prepared.device.type == "cpu"
    assert prepared.item() == 1

    with pytest.raises(ValueError, match="allocation_success"):
        _prepare_csa_fallback_allocation_status(
            torch.zeros((2,), dtype=torch.int32),
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="allocation_success"):
        _prepare_csa_fallback_allocation_status(
            torch.zeros((1,), dtype=torch.float32),
            device=torch.device("cpu"),
        )


def test_step4_triton_fallback_allocator_publishes_capacity_failure():
    source_path = (
        Path(__file__).parents[3]
        / "vllm"
        / "models"
        / "step4"
        / "sparse_summary_cache.py"
    )
    tree = ast.parse(source_path.read_text())
    kernel_names = (
        "_step4_sparse_summary_cache_csa_compact_decode_update_kernel",
        "_step4_sparse_summary_cache_csa_compact_decode_update_fast_kernel",
        "_step4_sparse_summary_cache_csa_compact_update_ordered_kernel",
    )

    for kernel_name in kernel_names:
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == kernel_name
        )
        assert "allocation_success" in {arg.arg for arg in function.args.args}
        stores_status_failure = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tl"
            and node.func.attr == "store"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "allocation_success"
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == 0
            for node in ast.walk(function)
        )
        assert stores_status_failure, kernel_name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_step4_csa_allocation_status_is_cuda_graph_safe():
    cache = SimpleNamespace(
        sum_cache=torch.empty((1,), device="cuda", dtype=torch.float32),
        _step4_csa_allocation_success=torch.ones(
            (1,), device="cuda", dtype=torch.int32
        ),
    )
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        success = Step4DSAAttentionImpl._begin_csa_allocation_check(cache)
        Step4DSAAttentionImpl._assert_csa_allocation_success(success)

    graph.replay()
    torch.accelerator.synchronize()
    assert cache._step4_csa_allocation_success.eq(1).all()


def _make_cuda_csa_allocation_state(*, occupied: bool):
    device = torch.device("cuda")
    active_region_ids = torch.tensor(
        [7 if occupied else -1],
        device=device,
        dtype=torch.long,
    )
    active_slot_by_region = torch.full(
        (8,),
        -1,
        device=device,
        dtype=torch.int32,
    )
    if occupied:
        active_slot_by_region[7] = 0
    return SimpleNamespace(
        sum_cache=torch.zeros((1, 1, 1, 256), device=device),
        count_cache=torch.zeros((1, 1, 1), device=device),
        mean_cache=torch.zeros((1, 8, 1, 256), device=device, dtype=torch.uint8),
        active_region_ids=active_region_ids,
        active_slot_by_region=active_slot_by_region,
        allocation_success=torch.ones((1,), device=device, dtype=torch.int32),
        active_numerator=torch.zeros((1, 1, 256), device=device),
        denominator=torch.zeros((1, 1, 256), device=device),
        max_logits=torch.full((1, 1), float("-inf"), device=device),
        flat_slot=torch.zeros((1,), device=device, dtype=torch.long),
        reset_slots=torch.full((1,), -1, device=device, dtype=torch.long),
        token_valid=torch.ones((1,), device=device, dtype=torch.bool),
        token_positions=torch.zeros((1,), device=device, dtype=torch.long),
        index_k=torch.ones((1, 1, 256), device=device, dtype=torch.bfloat16),
        index_z=torch.zeros((1, 1, 256), device=device, dtype=torch.bfloat16),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("occupied", [False, True], ids=["success", "exhausted"])
@pytest.mark.parametrize("update_kind", ["prefill", "decode-stage"])
def test_step4_cutedsl_updates_publish_allocation_status(
    occupied: bool,
    update_kind: str,
):
    from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa import (
        csa_compact_decode_stage_flush_with_slots_prevalidated_sm90_gqa,
        csa_compact_prefill_update_with_slots_prevalidated_sm90_gqa,
    )

    state = _make_cuda_csa_allocation_state(occupied=occupied)
    if update_kind == "prefill":
        csa_compact_prefill_update_with_slots_prevalidated_sm90_gqa(
            state.sum_cache,
            state.count_cache,
            state.mean_cache,
            state.active_region_ids,
            state.active_slot_by_region,
            state.active_numerator,
            state.denominator,
            state.max_logits,
            state.flat_slot,
            state.reset_slots,
            state.token_valid,
            state.token_positions,
            torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            torch.tensor([1], device="cuda", dtype=torch.int32),
            torch.zeros((2,), device="cuda", dtype=torch.int32),
            state.index_k,
            state.index_z,
            8,
            state.allocation_success,
        )
    else:
        csa_compact_decode_stage_flush_with_slots_prevalidated_sm90_gqa(
            state.sum_cache,
            state.count_cache,
            state.mean_cache,
            state.active_region_ids,
            state.active_slot_by_region,
            state.active_numerator,
            state.denominator,
            state.max_logits,
            torch.zeros(
                (1, 8, 1, 256),
                device="cuda",
                dtype=torch.bfloat16,
            ),
            torch.zeros(
                (1, 8, 1, 256),
                device="cuda",
                dtype=torch.bfloat16,
            ),
            torch.zeros((1, 8), device="cuda", dtype=torch.uint8),
            state.flat_slot,
            state.reset_slots,
            state.token_valid,
            state.token_positions,
            state.index_k,
            state.index_z,
            8,
            state.allocation_success,
        )
    torch.accelerator.synchronize()

    assert state.allocation_success.item() == (0 if occupied else 1)
    if not occupied:
        assert state.active_region_ids.item() == 0
        assert state.active_slot_by_region[0].item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("occupied", [False, True], ids=["success", "exhausted"])
def test_step4_mtp_update_allocator_publishes_allocation_status(occupied: bool):
    from vllm.models.step4.sparse_attention_mtp import (
        _step4_allocate_mtp_update_slots_kernel,
    )

    state = _make_cuda_csa_allocation_state(occupied=occupied)
    scratch_region_ids = torch.zeros((1,), device="cuda", dtype=torch.int32)
    scratch_row_map = torch.full((1, 8), -1, device="cuda", dtype=torch.int32)
    scratch_row_map[0, 0] = 0
    free_slots = torch.zeros((1,), device="cuda", dtype=torch.int32)
    free_count = torch.tensor(
        [0 if occupied else 1],
        device="cuda",
        dtype=torch.int32,
    )
    allocation_cursor = torch.zeros((1,), device="cuda", dtype=torch.int32)
    active_token_valid = torch.ones((1, 8), device="cuda", dtype=torch.uint8)
    scratch_reset_map = torch.zeros((1,), device="cuda", dtype=torch.int32)

    _step4_allocate_mtp_update_slots_kernel[(1,)](
        state.active_region_ids,
        state.active_slot_by_region,
        state.allocation_success,
        active_token_valid,
        scratch_region_ids,
        scratch_row_map,
        scratch_reset_map,
        free_slots,
        free_count,
        allocation_cursor,
        region_block_size=8,
        active_capacity=1,
        active_token_valid_stride_slot=int(active_token_valid.stride(0)),
    )
    torch.accelerator.synchronize()

    assert state.allocation_success.item() == (0 if occupied else 1)
    if not occupied:
        assert state.active_region_ids.item() == 0
        assert state.active_slot_by_region[0].item() == 0
        assert scratch_reset_map.item() == 1
        assert not active_token_valid.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("occupied", [False, True], ids=["success", "exhausted"])
def test_step4_mtp_correction_allocator_publishes_allocation_status(
    occupied: bool,
):
    from vllm.models.step4.sparse_attention_mtp import (
        _MTP_ACTION_CLEAR,
        _MTP_ACTION_INSTALL,
        _MTP_ACTION_INSTALL_PRE,
        _MTP_ACTION_NONE,
        _step4_apply_mtp_correction_kernel,
    )

    state = _make_cuda_csa_allocation_state(occupied=occupied)
    row_shape = (1,)
    state_shape = (1, 1, 256)
    row_regions = torch.zeros(row_shape, device="cuda", dtype=torch.long)
    row_positions = torch.zeros(row_shape, device="cuda", dtype=torch.long)
    row_source = torch.zeros(row_shape, device="cuda", dtype=torch.long)
    row_owner_block = torch.zeros(row_shape, device="cuda", dtype=torch.long)
    row_owner_block_index = torch.zeros(row_shape, device="cuda", dtype=torch.long)
    correction_action = torch.full(
        row_shape,
        _MTP_ACTION_INSTALL,
        device="cuda",
        dtype=torch.int8,
    )
    saved_numerator = torch.ones(state_shape, device="cuda")
    saved_denominator = torch.ones(state_shape, device="cuda")
    saved_max = torch.zeros((1, 1), device="cuda")
    free_slots = torch.zeros((1,), device="cuda", dtype=torch.int32)
    free_count = torch.tensor(
        [0 if occupied else 1],
        device="cuda",
        dtype=torch.int32,
    )
    allocation_cursor = torch.zeros((1,), device="cuda", dtype=torch.int32)
    active_token_valid = torch.ones((1, 8), device="cuda", dtype=torch.uint8)

    _step4_apply_mtp_correction_kernel[(1,)](
        row_regions,
        row_positions,
        row_source,
        row_owner_block,
        row_owner_block_index,
        correction_action,
        saved_numerator,
        saved_denominator,
        saved_max,
        saved_numerator,
        saved_denominator,
        saved_max,
        state.mean_cache,
        state.active_region_ids,
        state.active_slot_by_region,
        state.allocation_success,
        state.active_numerator,
        state.denominator,
        state.max_logits,
        active_token_valid,
        free_slots,
        free_count,
        allocation_cursor,
        mean_stride_page=state.mean_cache.stride(0),
        total_regions=8,
        summaries_per_page=8,
        active_capacity=1,
        proxy_dim=256,
        active_token_valid_stride_slot=int(active_token_valid.stride(0)),
        region_block_size=8,
        ACTION_NONE=_MTP_ACTION_NONE,
        ACTION_CLEAR=_MTP_ACTION_CLEAR,
        ACTION_INSTALL=_MTP_ACTION_INSTALL,
        ACTION_INSTALL_PRE=_MTP_ACTION_INSTALL_PRE,
        APPLY_INSTALL=True,
        BLOCK_CAPACITY=1,
        BLOCK_D=256,
    )
    torch.accelerator.synchronize()

    assert state.allocation_success.item() == (0 if occupied else 1)
    if not occupied:
        assert state.active_region_ids.item() == 0
        assert state.active_slot_by_region[0].item() == 0
        assert not active_token_valid.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("action_name", "slot_path"),
    [
        ("clear", "mapped"),
        ("clear", "scan"),
        ("install_empty", "mapped"),
        ("install_empty", "free"),
    ],
)
def test_step4_mtp_correction_clears_released_slot_state(
    action_name: str,
    slot_path: str,
):
    from vllm.models.step4.sparse_attention_mtp import (
        _MTP_ACTION_CLEAR,
        _MTP_ACTION_INSTALL,
        _MTP_ACTION_INSTALL_PRE,
        _MTP_ACTION_NONE,
        _step4_apply_mtp_correction_kernel,
    )

    occupied = slot_path != "free"
    state = _make_cuda_csa_allocation_state(occupied=occupied)
    region = 7
    if slot_path == "scan":
        state.active_slot_by_region[region] = -1
    state.active_numerator.fill_(3.0)
    state.denominator.fill_(4.0)
    state.max_logits.fill_(5.0)
    active_token_valid = torch.ones((1, 8), device="cuda", dtype=torch.uint8)

    row_shape = (1,)
    state_shape = (1, 1, 256)
    row_regions = torch.full(row_shape, region, device="cuda", dtype=torch.long)
    row_positions = torch.zeros(row_shape, device="cuda", dtype=torch.long)
    row_source = torch.zeros(row_shape, device="cuda", dtype=torch.long)
    row_owner_block = torch.zeros(row_shape, device="cuda", dtype=torch.long)
    row_owner_block_index = torch.zeros(row_shape, device="cuda", dtype=torch.long)
    action = _MTP_ACTION_CLEAR if action_name == "clear" else _MTP_ACTION_INSTALL_PRE
    correction_action = torch.full(
        row_shape,
        action,
        device="cuda",
        dtype=torch.int8,
    )
    saved_numerator = torch.ones(state_shape, device="cuda")
    saved_denominator = torch.zeros(state_shape, device="cuda")
    saved_max = torch.full((1, 1), float("-inf"), device="cuda")
    free_slots = torch.zeros((1,), device="cuda", dtype=torch.int32)
    free_count = torch.tensor(
        [1 if slot_path == "free" else 0],
        device="cuda",
        dtype=torch.int32,
    )
    allocation_cursor = torch.zeros((1,), device="cuda", dtype=torch.int32)

    _step4_apply_mtp_correction_kernel[(1,)](
        row_regions,
        row_positions,
        row_source,
        row_owner_block,
        row_owner_block_index,
        correction_action,
        saved_numerator,
        saved_denominator,
        saved_max,
        saved_numerator,
        saved_denominator,
        saved_max,
        state.mean_cache,
        state.active_region_ids,
        state.active_slot_by_region,
        state.allocation_success,
        state.active_numerator,
        state.denominator,
        state.max_logits,
        active_token_valid,
        free_slots,
        free_count,
        allocation_cursor,
        mean_stride_page=state.mean_cache.stride(0),
        total_regions=8,
        summaries_per_page=8,
        active_capacity=1,
        proxy_dim=256,
        active_token_valid_stride_slot=int(active_token_valid.stride(0)),
        region_block_size=8,
        ACTION_NONE=_MTP_ACTION_NONE,
        ACTION_CLEAR=_MTP_ACTION_CLEAR,
        ACTION_INSTALL=_MTP_ACTION_INSTALL,
        ACTION_INSTALL_PRE=_MTP_ACTION_INSTALL_PRE,
        APPLY_INSTALL=action_name != "clear",
        BLOCK_CAPACITY=1,
        BLOCK_D=256,
    )
    torch.accelerator.synchronize()

    assert state.active_region_ids.item() == -1
    assert state.active_slot_by_region[region].item() == -1
    assert not state.active_numerator.any()
    assert not state.denominator.any()
    assert torch.isneginf(state.max_logits).all()
    assert not active_token_valid.any()


def test_step4_dsa_page_reset_clears_incomplete_decode_tail():
    config = Step4SparseSummaryCacheConfig(
        num_pages=2,
        page_size=16,
        region_block_size=8,
        num_kv_heads=1,
        proxy_dim=4,
    )
    cache = Step4SparseSummaryCache(
        config=config,
        sum_cache=torch.zeros((4, 1, 1, 4)),
        count_cache=torch.zeros((4, 1, 1)),
        mean_cache=torch.zeros(config.sum_shape, dtype=torch.uint8),
    )
    cache._step4_csa_active_region_ids = torch.tensor(
        [1, -1, -1, -1],
        dtype=torch.long,
    )
    cache._step4_csa_active_slot_by_region = torch.full(
        (config.num_pages * config.summaries_per_page,),
        -1,
        dtype=torch.int32,
    )
    cache._step4_csa_active_slot_by_region[1] = 0
    cache._step4_csa_numerator_cache = torch.ones((4, 1, 4))
    cache._step4_csa_denominator_cache = torch.ones((4, 1, 4))
    cache._step4_csa_max_cache = torch.ones((4, 1))
    cache._step4_csa_active_token_k = torch.ones((4, 8, 1, 4))
    cache._step4_csa_active_token_z = torch.ones((4, 8, 1, 4))
    cache._step4_csa_active_token_valid = torch.ones((4, 8), dtype=torch.uint8)

    cache.reset_blocks([0])

    assert not cache._step4_csa_active_token_k[0].any()
    assert not cache._step4_csa_active_token_z[0].any()
    assert not cache._step4_csa_active_token_valid[0].any()
    assert cache._step4_csa_active_token_k[1].all()


def test_profiled_dsa_scratch_reuses_capacity_and_fails_closed_on_growth():
    buffer = torch.empty((8,), dtype=torch.float32)
    workspace = Step4DSAScratchWorkspace(
        tensor_buffers_by_engine={0: {"fixed": buffer}},
        order_tokens_by_engine={
            0: torch.zeros((1,), dtype=torch.int64),
        },
        memory_profiled=True,
        allocations_locked=True,
    )
    impl = object.__new__(Step4DSAAttentionImpl)
    impl._dsa_scratch_workspace = workspace
    impl._dsa_scratch_bound = False

    reused = impl._get_dsa_tensor_buffer_at_least(
        "fixed",
        (2, 4),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert reused.data_ptr() == buffer.data_ptr()

    with pytest.raises(RuntimeError, match="capacity was exceeded"):
        impl._get_dsa_tensor_buffer_at_least(
            "fixed",
            (9,),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_profiled_dsa_scratch_survives_temporary_kv_cache_cleanup():
    buffer = torch.empty((8,), dtype=torch.float32)
    workspace = Step4DSAScratchWorkspace(
        tensor_buffers_by_engine={0: {"fixed": buffer}},
        order_tokens_by_engine={
            0: torch.zeros((1,), dtype=torch.int64),
        },
        memory_profiled=True,
        allocations_locked=True,
    )
    target_impl = SimpleNamespace(
        _dsa_scratch_workspace=workspace,
        _summary_cache=object(),
        _summary_cache_config=object(),
        _dsa_scratch_bound=True,
    )
    layer = object.__new__(Step4SparseSummaryCacheLayer)
    layer._dsa_scratch_workspace = workspace
    layer._owns_dsa_scratch_workspace = True
    layer._budget_only_cache = object()
    layer._bound_summary_config = object()
    layer._target_impl = target_impl

    layer._clear_summary_cache_binding()

    assert layer._dsa_scratch_workspace is workspace
    assert workspace.tensor_buffers_by_engine[0]["fixed"] is buffer
    assert target_impl._summary_cache is None
    assert target_impl._summary_cache_config is None
    assert not target_impl._dsa_scratch_bound


@pytest.mark.parametrize("blocks_per_scheduler_block", [1, 2])
def test_step4_dsa_side_storage_cow_maps_scheduler_block_ids(
    blocks_per_scheduler_block: int,
):
    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy

    layer = object.__new__(Step4SparseSummaryCacheLayer)
    num_scheduler_blocks = 4
    layer._kv_cache = torch.zeros(
        (num_scheduler_blocks * blocks_per_scheduler_block, 2),
        dtype=torch.uint8,
    )
    layer._budget_only_cache = SimpleNamespace(
        blocks_per_scheduler_block=blocks_per_scheduler_block
    )

    for row in range(layer._kv_cache.shape[0]):
        layer._kv_cache[row].fill_(row + 1)

    layer.copy_kv_cache_side_storage(
        [KVCacheBlockCopy(src_block_id=1, dst_block_id=3)],
        num_blocks=num_scheduler_blocks,
    )

    src_start = blocks_per_scheduler_block
    dst_start = 3 * blocks_per_scheduler_block
    for offset in range(blocks_per_scheduler_block):
        torch.testing.assert_close(
            layer._kv_cache[dst_start + offset],
            layer._kv_cache[src_start + offset],
        )
    # The row immediately before the destination remains untouched.
    assert layer._kv_cache[dst_start - 1, 0].item() == dst_start


def _make_step4_dsa_cow_state(
    *,
    num_pages: int = 4,
    active_capacity: int = 4,
):
    config = Step4SparseSummaryCacheConfig(
        num_pages=num_pages,
        page_size=16,
        region_block_size=8,
        num_kv_heads=1,
        proxy_dim=4,
    )
    cache = Step4SparseSummaryCache(
        config=config,
        sum_cache=torch.zeros((active_capacity, 1, 1, 4)),
        count_cache=torch.zeros((active_capacity, 1, 1)),
        mean_cache=torch.zeros(config.sum_shape, dtype=torch.uint8),
    )
    cache._step4_csa_active_region_ids = torch.full(
        (active_capacity,),
        -1,
        dtype=torch.long,
    )
    cache._step4_csa_active_slot_by_region = torch.full(
        (num_pages * config.summaries_per_page,),
        -1,
        dtype=torch.int32,
    )
    cache._step4_csa_numerator_cache = torch.zeros((active_capacity, 1, 4))
    cache._step4_csa_denominator_cache = torch.zeros((active_capacity, 1, 4))
    cache._step4_csa_max_cache = torch.full(
        (active_capacity, 1),
        float("-inf"),
    )
    cache._step4_csa_active_token_k = torch.zeros((active_capacity, 8, 1, 4))
    cache._step4_csa_active_token_z = torch.zeros((active_capacity, 8, 1, 4))
    cache._step4_csa_active_token_valid = torch.zeros(
        (active_capacity, 8),
        dtype=torch.uint8,
    )
    cache.blocks_per_scheduler_block = 1
    return config, cache


def test_step4_dsa_side_storage_cow_clones_active_tail_state():
    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy

    config, cache = _make_step4_dsa_cow_state()
    # Page 0, fragment 1 is an incomplete region in slot 2. Its destination
    # page 2 must receive fragment 1 in a newly allocated slot (region 5).
    source_region = 1
    destination_region = 2 * config.summaries_per_page + 1
    source_slot = 2
    cache._step4_csa_active_region_ids[source_slot] = source_region
    cache._step4_csa_active_slot_by_region[source_region] = source_slot
    cache._step4_csa_numerator_cache[source_slot].fill_(3.0)
    cache._step4_csa_denominator_cache[source_slot].fill_(4.0)
    cache._step4_csa_max_cache[source_slot].fill_(5.0)
    cache._step4_csa_active_token_k[source_slot].fill_(6.0)
    cache._step4_csa_active_token_z[source_slot].fill_(7.0)
    cache._step4_csa_active_token_valid[source_slot].fill_(1)
    cache.mean_cache[0].fill_(11)

    layer = object.__new__(Step4SparseSummaryCacheLayer)
    layer._kv_cache = torch.zeros((config.num_pages, 2), dtype=torch.uint8)
    layer._budget_only_cache = cache
    layer._kv_cache[0].fill_(11)

    layer.copy_kv_cache_side_storage(
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)],
        num_blocks=config.num_pages,
    )

    destination_slot = int(
        cache._step4_csa_active_slot_by_region[destination_region].item()
    )
    assert destination_slot >= 0
    assert destination_slot != source_slot
    assert cache._step4_csa_active_region_ids[source_slot].item() == source_region
    assert cache._step4_csa_active_slot_by_region[source_region].item() == source_slot
    assert (
        cache._step4_csa_active_region_ids[destination_slot].item()
        == destination_region
    )
    torch.testing.assert_close(
        cache._step4_csa_numerator_cache[destination_slot],
        cache._step4_csa_numerator_cache[source_slot],
    )
    torch.testing.assert_close(
        cache._step4_csa_denominator_cache[destination_slot],
        cache._step4_csa_denominator_cache[source_slot],
    )
    torch.testing.assert_close(
        cache._step4_csa_max_cache[destination_slot],
        cache._step4_csa_max_cache[source_slot],
    )
    torch.testing.assert_close(
        cache._step4_csa_active_token_k[destination_slot],
        cache._step4_csa_active_token_k[source_slot],
    )
    torch.testing.assert_close(
        cache._step4_csa_active_token_z[destination_slot],
        cache._step4_csa_active_token_z[source_slot],
    )
    torch.testing.assert_close(
        cache._step4_csa_active_token_valid[destination_slot],
        cache._step4_csa_active_token_valid[source_slot],
    )
    assert layer._kv_cache[2].eq(11).all()


def test_step4_dsa_side_storage_cow_fails_closed_when_active_slots_exhausted():
    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy

    config, cache = _make_step4_dsa_cow_state(
        num_pages=2,
        active_capacity=1,
    )
    source_region = 0
    cache._step4_csa_active_region_ids[0] = source_region
    cache._step4_csa_active_slot_by_region[source_region] = 0
    cache._step4_csa_numerator_cache[0].fill_(3.0)

    layer = object.__new__(Step4SparseSummaryCacheLayer)
    layer._kv_cache = torch.zeros((config.num_pages, 2), dtype=torch.uint8)
    layer._budget_only_cache = cache

    with pytest.raises(RuntimeError, match="active-slot capacity is exhausted"):
        layer.copy_kv_cache_side_storage(
            [KVCacheBlockCopy(src_block_id=0, dst_block_id=1)],
            num_blocks=config.num_pages,
        )

    # The source remains owned and the destination payload is not copied after
    # the fail-closed state-capacity check.
    assert cache._step4_csa_active_region_ids[0].item() == source_region
    assert not layer._kv_cache[1].any()


def test_step4_dsa_side_storage_cow_repairs_stale_destination_reverse_map():
    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy

    config, cache = _make_step4_dsa_cow_state(
        num_pages=2,
        active_capacity=2,
    )
    source_region = 0
    source_slot = 0
    destination_region = config.summaries_per_page
    stale_destination_slot = 1
    cache._step4_csa_active_region_ids[source_slot] = source_region
    cache._step4_csa_active_slot_by_region[source_region] = source_slot
    cache._step4_csa_numerator_cache[source_slot].fill_(3.0)
    # Simulate a stale reverse-map hint: the authoritative active-slot table
    # still owns the destination region, but its reverse entry was lost.
    cache._step4_csa_active_region_ids[stale_destination_slot] = destination_region
    cache._step4_csa_active_slot_by_region[destination_region] = -1
    cache._step4_csa_numerator_cache[stale_destination_slot].fill_(9.0)

    layer = object.__new__(Step4SparseSummaryCacheLayer)
    layer._kv_cache = torch.zeros((config.num_pages, 2), dtype=torch.uint8)
    layer._budget_only_cache = cache

    layer.copy_kv_cache_side_storage(
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=1)],
        num_blocks=config.num_pages,
    )

    destination_slot = int(
        cache._step4_csa_active_slot_by_region[destination_region].item()
    )
    assert destination_slot == stale_destination_slot
    assert cache._step4_csa_active_region_ids.tolist() == [
        source_region,
        destination_region,
    ]
    torch.testing.assert_close(
        cache._step4_csa_numerator_cache[destination_slot],
        cache._step4_csa_numerator_cache[source_slot],
    )


def test_step4_dsa_cow_capacity_failure_preserves_destination_state():
    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy

    config, cache = _make_step4_dsa_cow_state(
        num_pages=3,
        active_capacity=3,
    )
    # Two live source regions need two destination slots, while the target
    # page already owns the only slot that could be released.  Capacity is
    # still insufficient, so the fail-closed path must not clear that state.
    source_slots = (0, 1)
    for slot, region in zip(source_slots, (0, 1)):
        cache._step4_csa_active_region_ids[slot] = region
        cache._step4_csa_active_slot_by_region[region] = slot
        cache._step4_csa_numerator_cache[slot].fill_(3.0 + slot)
    destination_region = config.summaries_per_page
    destination_slot = 2
    cache._step4_csa_active_region_ids[destination_slot] = destination_region
    cache._step4_csa_active_slot_by_region[destination_region] = destination_slot
    cache._step4_csa_numerator_cache[destination_slot].fill_(9.0)

    layer = object.__new__(Step4SparseSummaryCacheLayer)
    layer._kv_cache = torch.zeros((config.num_pages, 2), dtype=torch.uint8)
    layer._budget_only_cache = cache

    with pytest.raises(RuntimeError, match="active-slot capacity is exhausted"):
        layer.copy_kv_cache_side_storage(
            [KVCacheBlockCopy(src_block_id=0, dst_block_id=1)],
            num_blocks=config.num_pages,
        )

    assert cache._step4_csa_active_region_ids.tolist() == [0, 1, destination_region]
    assert (
        cache._step4_csa_active_slot_by_region[destination_region].item()
        == destination_slot
    )
    assert cache._step4_csa_numerator_cache[destination_slot, 0, 0].item() == 9.0
    assert not layer._kv_cache[1].any()


def test_step4_dsa_cow_rejects_conflicting_destination_mappings():
    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy

    _, cache = _make_step4_dsa_cow_state(num_pages=3)
    layer = object.__new__(Step4SparseSummaryCacheLayer)
    layer._kv_cache = torch.zeros((3, 2), dtype=torch.uint8)
    layer._budget_only_cache = cache

    with pytest.raises(ValueError, match="conflicting sources"):
        layer.copy_kv_cache_side_storage(
            [
                KVCacheBlockCopy(src_block_id=0, dst_block_id=2),
                KVCacheBlockCopy(src_block_id=1, dst_block_id=2),
            ],
            num_blocks=3,
        )


def test_step4_mtp_transaction_storage_budget_matches_allocated_tensors():
    kwargs = {
        "max_num_reqs": 3,
        "max_rows_per_req": 4,
        "num_kv_heads": 1,
        "proxy_dim": 8,
        "source_map_rows": 64,
        "active_capacity": 32,
    }
    transaction = Step4DSAMTPTransaction(
        **kwargs,
        device=torch.device("cpu"),
    )

    actual_bytes = sum(
        value.numel() * value.element_size()
        for value in vars(transaction).values()
        if isinstance(value, torch.Tensor)
    )

    assert Step4DSAMTPTransaction.storage_size_bytes(**kwargs) == actual_bytes


def test_step4_mtp_preallocates_mixed_prefill_partition_scratch():
    requests: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}

    class Owner:
        max_num_seqs = 3
        max_num_batched_tokens = 64

        @staticmethod
        def _round_up(value: int, alignment: int) -> int:
            return (value + alignment - 1) // alignment * alignment

        @staticmethod
        def _get_dsa_tensor_buffer_at_least(
            name: str,
            shape: tuple[int, ...],
            *,
            device: torch.device,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            requests[name] = (shape, dtype)
            return torch.empty(shape, device=device, dtype=dtype)

    mtp = Step4DSAMTP(Owner(), num_speculative_tokens=3)
    mtp.prepare_scratch(
        SimpleNamespace(
            sum_cache=torch.empty((1,)),
            region_block_size=8,
        )
    )

    assert requests["csa_mtp_prefill_flat_slot"] == ((64,), torch.int64)
    assert requests["csa_mtp_prefill_token_positions"] == ((64,), torch.int64)
    assert requests["csa_mtp_prefill_reset_slots"] == ((64,), torch.int64)
    assert requests["csa_mtp_prefill_token_valid"] == ((64,), torch.bool)
    assert requests["csa_mtp_q1_token_valid"] == ((64,), torch.bool)
    assert requests["csa_mtp_transaction_token_valid"] == ((64,), torch.bool)
    assert requests["csa_mtp_prefill_query_start_loc"] == ((4,), torch.int32)
    assert requests["csa_mtp_prefill_seq_lens"] == ((3,), torch.int32)


def test_step4_mtp_uses_cpu_query_prefix_as_verifier_token_boundary(monkeypatch):
    mtp = Step4DSAMTP(SimpleNamespace(), num_speculative_tokens=3)
    captured: dict[str, object] = {}
    monkeypatch.setattr(mtp, "correct", lambda **kwargs: None)
    monkeypatch.setattr(
        mtp,
        "_update_summary_cache",
        lambda **kwargs: captured.update(kwargs),
    )
    query_start_loc = torch.tensor([0, 4, 10], dtype=torch.int32)
    metadata = SimpleNamespace(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        seq_lens=torch.tensor([20, 6], dtype=torch.int32),
        block_table=torch.zeros((2, 2), dtype=torch.int32),
        mtp_num_verifier_reqs=1,
        max_query_len=6,
        dsa_valid_requests=torch.tensor([2], dtype=torch.int32),
        dsa_valid_tokens=torch.tensor([10], dtype=torch.int32),
    )
    layout = SimpleNamespace(
        token_flat_slot=torch.arange(10),
        token_positions=torch.arange(10),
        reset_slots=torch.full((10,), -1),
        token_valid=torch.ones((10,), dtype=torch.bool),
    )

    mtp.update(
        summary_cache=object(),
        attn_metadata=metadata,
        layout=layout,
        index_k=torch.zeros((10, 1, 4)),
        index_z=torch.zeros((10, 1, 4)),
        num_actual_tokens=10,
        use_decode_update=False,
    )

    assert captured["num_verifier_requests"] == 1
    assert captured["num_verifier_tokens"] == 4
    assert captured["has_q1_decode"] is False
    assert captured["preserve_completed_slots"] is True


def test_step4_mtp_detects_q1_inside_mixed_decode_prefix(monkeypatch):
    mtp = Step4DSAMTP(SimpleNamespace(), num_speculative_tokens=3)
    captured: dict[str, object] = {}
    monkeypatch.setattr(mtp, "correct", lambda **kwargs: None)
    monkeypatch.setattr(
        mtp,
        "_update_summary_cache",
        lambda **kwargs: captured.update(kwargs),
    )
    query_start_loc = torch.tensor([0, 1, 5], dtype=torch.int32)
    metadata = SimpleNamespace(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        seq_lens=torch.tensor([20, 24], dtype=torch.int32),
        block_table=torch.zeros((2, 2), dtype=torch.int32),
        mtp_num_verifier_reqs=2,
        max_query_len=4,
        dsa_valid_requests=torch.tensor([2], dtype=torch.int32),
        dsa_valid_tokens=torch.tensor([5], dtype=torch.int32),
    )
    layout = SimpleNamespace(
        token_flat_slot=torch.arange(5),
        token_positions=torch.arange(5),
        reset_slots=torch.full((5,), -1),
        token_valid=torch.ones((5,), dtype=torch.bool),
    )

    mtp.update(
        summary_cache=object(),
        attn_metadata=metadata,
        layout=layout,
        index_k=torch.zeros((5, 1, 4)),
        index_z=torch.zeros((5, 1, 4)),
        num_actual_tokens=5,
        use_decode_update=True,
    )

    assert captured["num_verifier_requests"] == 2
    assert captured["num_verifier_tokens"] == 5
    assert captured["has_q1_decode"] is True
    assert captured["preserve_completed_slots"] is True


def test_step4_mtp_mixed_verifier_prefill_splits_summary_updates(monkeypatch):
    ordinary_calls: list[dict[str, object]] = []

    class Owner:
        @staticmethod
        def _get_dsa_tensor_buffer_at_least(
            name: str,
            shape: tuple[int, ...],
            *,
            device: torch.device,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            del name
            return torch.empty(shape, device=device, dtype=dtype)

        @staticmethod
        def _update_summary_cache_with_padded_layout(**kwargs: object) -> None:
            ordinary_calls.append(kwargs)

    mtp = Step4DSAMTP(Owner(), num_speculative_tokens=3)
    verifier_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        mtp,
        "_update_summary_cache",
        lambda **kwargs: verifier_calls.append(kwargs),
    )
    layout = SimpleNamespace(
        token_flat_slot=torch.arange(10, dtype=torch.int64) + 100,
        token_positions=torch.arange(10, dtype=torch.int64) + 200,
        reset_slots=torch.arange(10, dtype=torch.int64) + 300,
        token_valid=torch.tensor(
            [True, True, True, True, True, False, True, True, True, True]
        ),
    )
    index_k = torch.arange(40, dtype=torch.float32).view(10, 1, 4)
    index_z = index_k + 1000
    query_start_loc = torch.tensor([0, 4, 7, 10], dtype=torch.int32)
    seq_lens = torch.tensor([20, 3, 6], dtype=torch.int32)
    block_table = torch.zeros((3, 2), dtype=torch.int32)
    valid_requests = torch.tensor([3], dtype=torch.int32)
    valid_tokens = torch.tensor([10], dtype=torch.int32)

    Step4DSAMTP._update_summary_cache(
        mtp,
        summary_cache=SimpleNamespace(proxy_dim=4),
        layout=layout,
        index_k=index_k,
        index_z=index_z,
        num_actual_tokens=10,
        use_decode_update=True,
        preserve_completed_slots=True,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_table=block_table,
        num_verifier_requests=1,
        num_verifier_tokens=4,
        has_q1_decode=False,
        valid_requests=valid_requests,
        valid_tokens=valid_tokens,
        step_metadata=object(),
    )

    assert len(verifier_calls) == 1
    verifier = verifier_calls[0]
    assert verifier["num_actual_tokens"] == 4
    assert verifier["num_verifier_tokens"] == 4
    assert verifier["preserve_completed_slots"] is True
    torch.testing.assert_close(verifier["index_k"], index_k[:4])
    torch.testing.assert_close(verifier["index_z"], index_z[:4])
    torch.testing.assert_close(
        verifier["layout"].token_flat_slot,
        layout.token_flat_slot[:4],
    )

    assert len(ordinary_calls) == 1
    prefill = ordinary_calls[0]
    assert prefill["num_actual_tokens"] == 6
    assert prefill["use_decode_update"] is False
    assert prefill["step_metadata"] is None
    torch.testing.assert_close(prefill["index_k"], index_k[4:])
    torch.testing.assert_close(prefill["index_z"], index_z[4:])
    torch.testing.assert_close(
        prefill["layout"].token_flat_slot,
        layout.token_flat_slot[4:],
    )
    torch.testing.assert_close(
        prefill["layout"].token_positions,
        layout.token_positions[4:],
    )
    torch.testing.assert_close(
        prefill["layout"].reset_slots,
        layout.reset_slots[4:],
    )
    torch.testing.assert_close(
        prefill["layout"].token_valid,
        layout.token_valid[4:],
    )
    torch.testing.assert_close(
        prefill["query_start_loc"],
        torch.tensor([0, 3, 6], dtype=torch.int32),
    )
    torch.testing.assert_close(
        prefill["seq_lens"],
        torch.tensor([3, 6], dtype=torch.int32),
    )


def test_step4_mtp_q1_verifier_prefill_masks_transaction_suffix(monkeypatch):
    """Only q2+ verifier rows may enter the transaction update."""
    buffers: dict[str, torch.Tensor] = {}
    ordinary_calls: list[dict[str, object]] = []

    class Owner:
        max_num_seqs = 3

        @staticmethod
        def _get_dsa_tensor_buffer_at_least(
            name: str,
            shape: tuple[int, ...],
            *,
            device: torch.device,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            required = 1
            for dim in shape:
                required *= int(dim)
            buffer = buffers.get(name)
            if (
                buffer is None
                or buffer.device != device
                or buffer.dtype != dtype
                or int(buffer.numel()) < required
            ):
                buffer = torch.ones((required,), device=device, dtype=dtype)
                buffers[name] = buffer
            return buffer[:required].view(*shape)

        @staticmethod
        def _update_summary_cache_with_padded_layout(**kwargs: object) -> None:
            ordinary_calls.append(kwargs)

    class PartitionKernel:
        def __getitem__(self, _grid):
            def launch(
                _query_start_loc,
                _layout_valid,
                q1_valid,
                transaction_valid,
                _valid_requests,
                _valid_tokens,
                _num_verifier_requests,
                **_kwargs,
            ):
                # Emulate the production kernel: it writes only the verifier
                # prefix and leaves rows after that prefix untouched.
                q1_valid.zero_()
                q1_valid[0] = True
                transaction_valid[1:5] = True

            return launch

    monkeypatch.setattr(
        step4_sparse_attention_mtp,
        "_step4_dsa_mtp_partition_decode_validity_kernel",
        PartitionKernel(),
    )

    mtp = Step4DSAMTP(Owner(), num_speculative_tokens=3)
    summary_cache = SimpleNamespace(
        proxy_dim=4,
        _step4_mtp_transaction=SimpleNamespace(max_rows_per_req=4),
    )
    layout = SimpleNamespace(
        token_flat_slot=torch.arange(8, dtype=torch.int64),
        token_positions=torch.arange(8, dtype=torch.int64),
        reset_slots=torch.full((8,), -1, dtype=torch.int64),
        token_valid=torch.ones((8,), dtype=torch.bool),
    )
    query_start_loc = torch.tensor([0, 1, 5, 8], dtype=torch.int32)
    metadata = dict(
        summary_cache=summary_cache,
        layout=layout,
        index_k=torch.zeros((8, 1, 4)),
        index_z=torch.zeros((8, 1, 4)),
        num_actual_tokens=8,
        use_decode_update=False,
        preserve_completed_slots=True,
        query_start_loc=query_start_loc,
        seq_lens=torch.tensor([20, 24, 3], dtype=torch.int32),
        block_table=torch.zeros((3, 2), dtype=torch.int32),
        num_verifier_requests=2,
        num_verifier_tokens=5,
        has_q1_decode=True,
        valid_requests=torch.tensor([3], dtype=torch.int32),
        valid_tokens=torch.tensor([8], dtype=torch.int32),
        step_metadata=object(),
    )

    original_update = Step4DSAMTP._update_summary_cache
    calls: list[dict[str, object]] = []

    def wrapped_update(self, **kwargs: object) -> None:
        calls.append(kwargs)
        if len(calls) == 1:
            original_update(self, **kwargs)

    monkeypatch.setattr(Step4DSAMTP, "_update_summary_cache", wrapped_update)
    Step4DSAMTP._update_summary_cache(mtp, **metadata)

    assert len(calls) == 2
    transaction_layout = calls[1]["layout"]
    assert isinstance(transaction_layout, SimpleNamespace)
    assert transaction_layout.token_valid.tolist() == [
        False,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert len(ordinary_calls) == 1
    assert ordinary_calls[0]["layout"].token_valid.tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Step4 DSA MTP requires CUDA")
def test_step4_mtp_partition_decode_graph_replay_cuda():
    query_start_loc = torch.tensor([0, 1, 5], device="cuda", dtype=torch.int32)
    layout_valid = torch.ones(5, device="cuda", dtype=torch.bool)
    q1_valid = torch.zeros_like(layout_valid)
    transaction_valid = torch.zeros_like(layout_valid)
    valid_requests = torch.tensor([2], device="cuda", dtype=torch.int32)
    valid_tokens = torch.tensor([5], device="cuda", dtype=torch.int32)

    def launch() -> None:
        q1_valid.zero_()
        transaction_valid.zero_()
        _step4_dsa_mtp_partition_decode_validity_kernel[(2, 4)](
            query_start_loc,
            layout_valid,
            q1_valid,
            transaction_valid,
            valid_requests,
            valid_tokens,
            2,
            max_rows_per_req=4,
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    graph.replay()
    torch.cuda.synchronize()
    assert q1_valid.tolist() == [True, False, False, False, False]
    assert transaction_valid.tolist() == [False, True, True, True, True]

    query_start_loc.copy_(torch.tensor([0, 4, 5], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()
    assert q1_valid.tolist() == [False, False, False, False, True]
    assert transaction_valid.tolist() == [True, True, True, True, False]


def _make_step4_mtp_cuda_state():
    config = Step4SparseSummaryCacheConfig(
        num_pages=7,
        page_size=64,
        region_block_size=8,
        num_kv_heads=1,
        proxy_dim=256,
    )
    cache = Step4SparseSummaryCache(
        config=config,
        sum_cache=torch.zeros((6, 1, 1, 256), device="cuda"),
        count_cache=torch.zeros((6, 1, 1), device="cuda"),
        mean_cache=torch.zeros(config.sum_shape, device="cuda", dtype=torch.uint8),
    )
    owner = object.__new__(Step4DSAAttentionImpl)
    owner.max_num_seqs = 3
    owner.max_num_batched_tokens = 64
    owner._decode_out_dtype = torch.bfloat16
    owner._summary_transaction_dtype = torch.float32
    owner._dsa_scratch_workspace = Step4DSAScratchWorkspace()
    owner.sparse_region_block_size = config.region_block_size
    mtp = Step4DSAMTP(owner, num_speculative_tokens=3)
    owner._mtp = mtp
    owner._initialize_csa_summary_state(
        summary_cache=cache,
        capacity=2 * owner.max_num_seqs,
    )
    mtp.initialize(cache)
    return cache, owner, mtp


def _step4_mtp_cuda_layout(
    regions: list[int],
    positions: list[int],
) -> Step4DSARuntimeLayout:
    region_tensor = torch.tensor(regions, device="cuda", dtype=torch.long)
    position_tensor = torch.tensor(positions, device="cuda", dtype=torch.long)
    reset_slots = torch.where(
        position_tensor.remainder(8) == 0,
        region_tensor,
        torch.full_like(region_tensor, -1),
    )
    return Step4DSARuntimeLayout(
        token_flat_slot=region_tensor,
        token_positions=position_tensor,
        reset_slots=reset_slots,
        token_valid=torch.ones_like(region_tensor, dtype=torch.bool),
    )


def _step4_mtp_cuda_values(
    requests: list[int],
    positions: list[int],
) -> torch.Tensor:
    scalars = torch.tensor(
        [request * 100 + position for request, position in zip(requests, positions)],
        device="cuda",
        dtype=torch.float32,
    )
    return scalars[:, None, None].expand(-1, 1, 256).contiguous()


def _read_step4_mean_region(
    cache: Step4SparseSummaryCache,
    region: int,
) -> torch.Tensor:
    dims = torch.arange(256, device="cuda", dtype=torch.long)
    page = region // cache.summaries_per_page
    fragment = region % cache.summaries_per_page
    atom = dims // 128
    atom_dim = dims - atom * 128
    chunk = atom_dim // 16
    byte = atom_dim % 16
    offsets = (
        page * int(cache.mean_cache.stride(0))
        + (fragment // 8) * 2048
        + atom * 1024
        + (fragment % 8) * 128
        + ((chunk ^ (region % 8)) * 16)
        + byte
    )
    return cache.mean_cache.view(-1)[offsets].clone()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Step4 DSA MTP requires CUDA")
def test_step4_mtp_new_slot_discards_poisoned_accumulator_cuda():
    cache, _, mtp = _make_step4_mtp_cuda_state()
    cache._step4_csa_active_region_ids.fill_(-1)
    cache._step4_csa_active_slot_by_region.fill_(-1)
    cache._step4_csa_numerator_cache.fill_(100.0)
    cache._step4_csa_denominator_cache.fill_(1.0)
    cache._step4_csa_max_cache.zero_()
    cache._step4_csa_active_token_k.fill_(1000.0)
    cache._step4_csa_active_token_z.zero_()
    cache._step4_csa_active_token_valid.fill_(1)

    positions = [13, 14]
    values = _step4_mtp_cuda_values([0, 0], positions)
    mtp.update(
        summary_cache=cache,
        attn_metadata=SimpleNamespace(
            query_start_loc=torch.tensor([0, 2], device="cuda", dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
            seq_lens=torch.tensor([15], device="cuda", dtype=torch.int32),
            block_table=torch.tensor([[0]], device="cuda", dtype=torch.int32),
            mtp_num_verifier_reqs=1,
            dsa_valid_requests=torch.tensor([1], device="cuda", dtype=torch.int32),
            dsa_valid_tokens=torch.tensor([2], device="cuda", dtype=torch.int32),
            max_query_len=2,
        ),
        layout=_step4_mtp_cuda_layout([1, 1], positions),
        index_k=values,
        index_z=torch.zeros_like(values),
        num_actual_tokens=2,
        use_decode_update=True,
    )
    torch.cuda.synchronize()

    active_slot = int(cache._step4_csa_active_slot_by_region[1].item())
    assert active_slot >= 0
    torch.testing.assert_close(
        cache._step4_csa_numerator_cache[active_slot, 0],
        torch.full((256,), 27.0, device="cuda"),
    )
    torch.testing.assert_close(
        cache._step4_csa_denominator_cache[active_slot, 0],
        torch.full((256,), 2.0, device="cuda"),
    )
    torch.testing.assert_close(
        cache._step4_csa_max_cache[active_slot, 0],
        torch.tensor(0.0, device="cuda"),
    )
    assert not cache._step4_csa_active_token_valid[active_slot].any()


def _prepare_step4_staged_q1_transaction(
    *,
    trailing_staged_value: float | None = None,
    verifier_token_valid: list[bool] | None = None,
):
    cache, owner, mtp = _make_step4_mtp_cuda_state()
    context_positions = list(range(8, 13))
    context_values = _step4_mtp_cuda_values([0] * 5, context_positions)
    owner._update_summary_cache_with_padded_layout(
        summary_cache=cache,
        layout=_step4_mtp_cuda_layout([1] * 5, context_positions),
        index_k=context_values,
        num_actual_tokens=5,
        proxy_dim=256,
        use_decode_update=False,
        index_z=torch.zeros_like(context_values),
        query_start_loc=torch.tensor([0, 5], device="cuda", dtype=torch.int32),
        seq_lens=torch.tensor([13], device="cuda", dtype=torch.int32),
    )

    q1_values = _step4_mtp_cuda_values([0], [13])
    owner._update_summary_cache_with_padded_layout(
        summary_cache=cache,
        layout=_step4_mtp_cuda_layout([1], [13]),
        index_k=q1_values,
        num_actual_tokens=1,
        proxy_dim=256,
        use_decode_update=True,
        index_z=torch.zeros_like(q1_values),
        query_start_loc=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        seq_lens=torch.tensor([14], device="cuda", dtype=torch.int32),
    )
    torch.cuda.synchronize()
    staged_slot = int(cache._step4_csa_active_slot_by_region[1].item())
    assert staged_slot >= 0
    assert cache._step4_csa_active_token_valid[staged_slot].tolist() == [
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
    ]
    if trailing_staged_value is not None:
        cache._step4_csa_active_token_k[staged_slot, 7].fill_(trailing_staged_value)
        cache._step4_csa_active_token_z[staged_slot, 7].zero_()
        cache._step4_csa_active_token_valid[staged_slot, 7] = 1

    verifier_positions = [14, 15, 16, 17]
    verifier_values = _step4_mtp_cuda_values([0] * 4, verifier_positions)
    verifier_layout = _step4_mtp_cuda_layout([1, 1, 2, 2], verifier_positions)
    if verifier_token_valid is not None:
        verifier_valid = torch.tensor(
            verifier_token_valid,
            device="cuda",
            dtype=torch.bool,
        )
        verifier_layout.token_valid.copy_(verifier_valid)
        verifier_layout.reset_slots.masked_fill_(~verifier_valid, -1)
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 4], device="cuda", dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([18], device="cuda", dtype=torch.int32),
        block_table=torch.tensor([[0]], device="cuda", dtype=torch.int32),
        mtp_num_verifier_reqs=1,
        dsa_valid_requests=torch.tensor([1], device="cuda", dtype=torch.int32),
        dsa_valid_tokens=torch.tensor([4], device="cuda", dtype=torch.int32),
        max_query_len=4,
    )
    mtp.update(
        summary_cache=cache,
        attn_metadata=metadata,
        layout=verifier_layout,
        index_k=verifier_values,
        index_z=torch.zeros_like(verifier_values),
        num_actual_tokens=4,
        use_decode_update=True,
    )
    torch.cuda.synchronize()
    return cache, owner, mtp, staged_slot


def _make_step4_committed_summary_reference(
    accepted_rows: int,
) -> Step4SparseSummaryCache:
    cache, owner, _ = _make_step4_mtp_cuda_state()
    committed_positions = list(range(8, 14 + accepted_rows))
    committed_values = _step4_mtp_cuda_values(
        [0] * len(committed_positions),
        committed_positions,
    )
    owner._update_summary_cache_with_padded_layout(
        summary_cache=cache,
        layout=_step4_mtp_cuda_layout(
            [position // 8 for position in committed_positions],
            committed_positions,
        ),
        index_k=committed_values,
        num_actual_tokens=len(committed_positions),
        proxy_dim=256,
        use_decode_update=False,
        index_z=torch.zeros_like(committed_values),
        query_start_loc=torch.tensor(
            [0, len(committed_positions)],
            device="cuda",
            dtype=torch.int32,
        ),
        seq_lens=torch.tensor(
            [14 + accepted_rows],
            device="cuda",
            dtype=torch.int32,
        ),
    )
    torch.cuda.synchronize()
    return cache


def _assert_step4_logical_region_state_equal(
    actual: Step4SparseSummaryCache,
    expected: Step4SparseSummaryCache,
    region: int,
) -> None:
    assert torch.equal(
        _read_step4_mean_region(actual, region),
        _read_step4_mean_region(expected, region),
    )
    actual_slot = int(actual._step4_csa_active_slot_by_region[region].item())
    expected_slot = int(expected._step4_csa_active_slot_by_region[region].item())
    assert (actual_slot >= 0) == (expected_slot >= 0)
    if actual_slot < 0:
        return
    assert actual._step4_csa_active_region_ids[actual_slot].item() == region
    assert expected._step4_csa_active_region_ids[expected_slot].item() == region
    torch.testing.assert_close(
        actual._step4_csa_numerator_cache[actual_slot],
        expected._step4_csa_numerator_cache[expected_slot],
    )
    torch.testing.assert_close(
        actual._step4_csa_denominator_cache[actual_slot],
        expected._step4_csa_denominator_cache[expected_slot],
    )
    torch.testing.assert_close(
        actual._step4_csa_max_cache[actual_slot],
        expected._step4_csa_max_cache[expected_slot],
    )
    assert not actual._step4_csa_active_token_valid[actual_slot].any()


def _run_step4_q1_with_optional_verifier(
    *,
    include_verifier: bool,
) -> tuple[torch.Tensor, ...]:
    cache, owner, mtp = _make_step4_mtp_cuda_state()
    context_positions = list(range(8, 14))
    context_regions = [1] * 6
    context_requests = [0] * 6
    query_starts = [0, 6]
    context_seq_lens = [14]
    if include_verifier:
        context_regions += [9] * 6
        context_requests += [1] * 6
        query_starts.append(12)
        context_seq_lens.append(14)
    context_values = _step4_mtp_cuda_values(
        context_requests,
        context_positions * (1 + include_verifier),
    )
    owner._update_summary_cache_with_padded_layout(
        summary_cache=cache,
        layout=_step4_mtp_cuda_layout(
            context_regions,
            context_positions * (1 + include_verifier),
        ),
        index_k=context_values,
        num_actual_tokens=int(context_values.shape[0]),
        proxy_dim=int(context_values.shape[-1]),
        use_decode_update=False,
        index_z=torch.zeros_like(context_values),
        query_start_loc=torch.tensor(
            query_starts,
            device="cuda",
            dtype=torch.int32,
        ),
        seq_lens=torch.tensor(
            context_seq_lens,
            device="cuda",
            dtype=torch.int32,
        ),
    )

    positions = [14]
    regions = [1]
    requests = [0]
    query_starts = [0, 1]
    seq_lens = [15]
    block_table = [[0]]
    if include_verifier:
        positions += [14, 15, 16, 17]
        regions += [9, 9, 10, 10]
        requests += [1] * 4
        query_starts.append(5)
        seq_lens.append(18)
        block_table.append([1])
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor(
            query_starts,
            device="cuda",
            dtype=torch.int32,
        ),
        query_start_loc_cpu=torch.tensor(query_starts, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, device="cuda", dtype=torch.int32),
        block_table=torch.tensor(block_table, device="cuda", dtype=torch.int32),
        mtp_num_verifier_reqs=len(seq_lens),
        dsa_valid_requests=torch.tensor(
            [len(seq_lens)],
            device="cuda",
            dtype=torch.int32,
        ),
        dsa_valid_tokens=torch.tensor(
            [len(positions)],
            device="cuda",
            dtype=torch.int32,
        ),
        max_query_len=4 if include_verifier else 1,
    )
    values = _step4_mtp_cuda_values(requests, positions)
    mtp.update(
        summary_cache=cache,
        attn_metadata=metadata,
        layout=_step4_mtp_cuda_layout(regions, positions),
        index_k=values,
        index_z=torch.zeros_like(values),
        num_actual_tokens=len(positions),
        use_decode_update=True,
    )
    torch.cuda.synchronize()
    q1_region = 1
    active_slot = int(cache._step4_csa_active_slot_by_region[q1_region].item())
    assert active_slot >= 0
    return (
        _read_step4_mean_region(cache, q1_region),
        cache._step4_csa_active_region_ids[active_slot].clone(),
        cache._step4_csa_numerator_cache[active_slot].clone(),
        cache._step4_csa_denominator_cache[active_slot].clone(),
        cache._step4_csa_max_cache[active_slot].clone(),
        cache._step4_csa_active_token_k[active_slot].clone(),
        cache._step4_csa_active_token_z[active_slot].clone(),
        cache._step4_csa_active_token_valid[active_slot].clone(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Step4 DSA MTP requires CUDA")
def test_step4_mtp_q1_decode_is_verifier_neighbor_invariant_cuda():
    isolated = _run_step4_q1_with_optional_verifier(include_verifier=False)
    mixed = _run_step4_q1_with_optional_verifier(include_verifier=True)
    for isolated_state, mixed_state in zip(isolated, mixed):
        assert torch.equal(mixed_state, isolated_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Step4 DSA MTP requires CUDA")
def test_step4_mtp_transaction_merges_staged_q1_tail_cuda():
    cache, _, _, staged_slot = _prepare_step4_staged_q1_transaction()
    transaction = cache._step4_mtp_transaction
    assert transaction.row_positions[0].item() == 14
    torch.testing.assert_close(
        transaction.state_pre_numerator[0, 0],
        torch.full((256,), float(sum(range(8, 14))), device="cuda"),
    )
    torch.testing.assert_close(
        transaction.state_pre_denominator[0, 0],
        torch.full((256,), 6.0, device="cuda"),
    )
    torch.testing.assert_close(
        transaction.state_pre_max_logits[0, 0],
        torch.tensor(0.0, device="cuda"),
    )
    assert not cache._step4_csa_active_token_valid[staged_slot].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Step4 DSA MTP requires CUDA")
def test_step4_mtp_ignores_staged_tail_after_first_verifier_row_cuda():
    cache, _, _, staged_slot = _prepare_step4_staged_q1_transaction(
        trailing_staged_value=1000.0,
        verifier_token_valid=[True, False, True, True],
    )
    active_slot = int(cache._step4_csa_active_slot_by_region[1].item())

    assert active_slot == staged_slot
    torch.testing.assert_close(
        cache._step4_csa_numerator_cache[active_slot, 0],
        torch.full((256,), float(sum(range(8, 15))), device="cuda"),
    )
    torch.testing.assert_close(
        cache._step4_csa_denominator_cache[active_slot, 0],
        torch.full((256,), 7.0, device="cuda"),
    )
    assert not cache._step4_csa_active_token_valid[active_slot].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Step4 DSA MTP requires CUDA")
@pytest.mark.parametrize("accepted_rows", range(5))
def test_step4_mtp_staged_q1_correction_matches_committed_reference_cuda(
    accepted_rows: int,
):
    cache, _, mtp, _ = _prepare_step4_staged_q1_transaction()
    mtp.correct(
        summary_cache=cache,
        attn_metadata=SimpleNamespace(
            query_start_loc=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            seq_lens=torch.tensor(
                [15 + accepted_rows],
                device="cuda",
                dtype=torch.int32,
            ),
            block_table=torch.tensor([[0]], device="cuda", dtype=torch.int32),
            dsa_valid_requests=torch.tensor([1], device="cuda", dtype=torch.int32),
        ),
    )
    expected = _make_step4_committed_summary_reference(accepted_rows)
    torch.cuda.synchronize()

    _assert_step4_logical_region_state_equal(cache, expected, 1)
    _assert_step4_logical_region_state_equal(cache, expected, 2)
    assert cache._step4_mtp_transaction.row_regions.eq(-1).all()


def test_step4_csa_fixed_runtime_budget_matches_allocated_tensors():
    proxy_dim = 4
    config = Step4SparseSummaryCacheConfig(
        num_pages=3,
        page_size=16,
        region_block_size=8,
        num_kv_heads=1,
        proxy_dim=proxy_dim,
    )
    cache = Step4SparseSummaryCache(
        config=config,
        sum_cache=torch.zeros((24, 1, 1, proxy_dim)),
        count_cache=torch.zeros((24, 1, 1)),
    )
    impl = object.__new__(Step4DSAAttentionImpl)
    impl.max_num_seqs = 3
    impl.summary_cache_num_proxy_kv_heads = 1
    impl.sparse_config = SimpleNamespace(proxy_dim=proxy_dim)
    impl.sparse_region_block_size = config.region_block_size
    impl._decode_out_dtype = torch.bfloat16
    impl._mtp = None

    capacity = impl._csa_active_region_capacity()
    impl._initialize_csa_summary_state(cache, capacity=capacity)
    fixed_state_names = (
        "_step4_csa_active_region_ids",
        "_step4_csa_allocation_success",
        "_step4_csa_numerator_cache",
        "_step4_csa_denominator_cache",
        "_step4_csa_max_cache",
        "_step4_csa_active_token_k",
        "_step4_csa_active_token_z",
        "_step4_csa_active_token_valid",
    )
    actual_bytes = sum(
        getattr(cache, name).numel() * getattr(cache, name).element_size()
        for name in fixed_state_names
    )

    assert impl.csa_fixed_runtime_state_size_bytes() == actual_bytes


def test_step4_mtp_fixed_runtime_budget_matches_allocated_tensors():
    proxy_dim = 4
    config = Step4SparseSummaryCacheConfig(
        num_pages=3,
        page_size=16,
        region_block_size=8,
        num_kv_heads=1,
        proxy_dim=proxy_dim,
    )
    impl = object.__new__(Step4DSAAttentionImpl)
    impl.max_num_seqs = 3
    impl.max_num_batched_tokens = 64
    impl.summary_cache_num_proxy_kv_heads = 1
    impl.sparse_config = SimpleNamespace(proxy_dim=proxy_dim)
    impl.sparse_region_block_size = config.region_block_size
    impl._decode_out_dtype = torch.bfloat16
    impl._dsa_scratch_workspace = Step4DSAScratchWorkspace()
    impl._dsa_scratch_bound = False
    impl._mtp = Step4DSAMTP(impl, num_speculative_tokens=3)

    capacity = impl._csa_active_region_capacity()
    cache = Step4SparseSummaryCache(
        config=config,
        sum_cache=torch.zeros((capacity, 1, 1, proxy_dim)),
        count_cache=torch.zeros((capacity, 1, 1)),
    )
    impl._initialize_csa_summary_state(cache, capacity=capacity)
    impl._mtp.initialize(cache)

    fixed_state_names = (
        "_step4_csa_active_region_ids",
        "_step4_csa_allocation_success",
        "_step4_csa_numerator_cache",
        "_step4_csa_denominator_cache",
        "_step4_csa_max_cache",
        "_step4_csa_active_token_k",
        "_step4_csa_active_token_z",
        "_step4_csa_active_token_valid",
    )
    actual_bytes = sum(
        getattr(cache, name).numel() * getattr(cache, name).element_size()
        for name in fixed_state_names
    )
    actual_bytes += sum(
        value.numel() * value.element_size()
        for value in vars(cache._step4_mtp_transaction).values()
        if isinstance(value, torch.Tensor)
    )

    assert capacity == 8 * impl.max_num_seqs * (impl._mtp.num_speculative_tokens + 1)
    assert impl.csa_fixed_runtime_state_size_bytes() == actual_bytes


def test_step4_csa_prefill_prewarm_uses_bound_cache_geometry(monkeypatch):
    config = Step4SparseSummaryCacheConfig(
        num_pages=2,
        page_size=64,
        region_block_size=8,
        num_kv_heads=1,
        proxy_dim=256,
    )
    sum_cache = torch.empty_strided(
        (16, 1, 1, 256),
        (320, 256, 256, 1),
        dtype=torch.float32,
    )
    count_cache = torch.empty_strided(
        (16, 1, 1),
        (3, 1, 1),
        dtype=torch.float32,
    )
    cache = Step4SparseSummaryCache(
        config=config,
        sum_cache=sum_cache,
        count_cache=count_cache,
    )
    calls = []
    monkeypatch.setattr(
        step4_sparse_attention,
        "prewarm_csa_compact_prefill_update_with_slots_sm90_gqa",
        lambda **kwargs: calls.append(kwargs),
    )

    Step4DSAAttentionImpl._prewarm_csa_compact_prefill_update(
        cache,
        index_dtype=torch.bfloat16,
    )

    assert calls == [
        {
            "device": sum_cache.device,
            "index_dtype": torch.bfloat16,
            "head_dim": 256,
            "summaries_per_page": 8,
            "sum_page_stride": 320,
            "count_page_stride": 3,
            "region_block_size": 8,
        }
    ]


def test_step4_side_storage_budget_covers_reverse_map_and_fixed_state():
    target_impl = SimpleNamespace(
        sparse_region_block_size=8,
        summary_cache_num_proxy_kv_heads=1,
        csa_fixed_runtime_state_size_bytes=lambda: 4096,
    )
    layer = object.__new__(Step4SparseSummaryCacheLayer)
    layer._target_impl = target_impl
    layer._sparse_config = SimpleNamespace(proxy_dim=4)
    layer._kv_cache_block_size = None
    layer._budget_bytes_per_page = None
    layer._backing_bytes_per_page = None

    page_budget = layer._set_summary_cache_budget(block_size=16)

    assert layer._backing_bytes_per_page == 16
    assert page_budget == 24
    assert layer._fixed_runtime_state_budget(block_size=16) == 4096
