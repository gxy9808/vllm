# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path


def test_step4_prefill_union_records_external_launch_tensor_lifetimes():
    source_path = (
        Path(__file__).parents[3]
        / "vllm"
        / "models"
        / "step4"
        / "nvidia"
        / "ops"
        / "cute_dsl"
        / "sparse_gqa"
        / "token_sparse_attn"
        / "prefill_union_paged_sm90_gqa.py"
    )
    tree = ast.parse(source_path.read_text())
    kernel_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "TokenWiseFlashAttnFwdSm90ManualMbarrierMmaPrefillUnionGQA"
    )
    run = next(
        node
        for node in kernel_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    guarded_record = next(
        node
        for node in ast.walk(run)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "can_record_stream"
    )
    tensor_loop = next(
        node
        for node in guarded_record.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "tensor"
        and isinstance(node.iter, ast.Tuple)
    )
    recorded_tensors = {
        element.id for element in tensor_loop.iter.elts if isinstance(element, ast.Name)
    }

    assert recorded_tensors == {
        "q",
        "k_cache",
        "v_cache",
        "o",
        "lse",
        "union_phys_indices",
        "union_logical_indices",
        "union_counts",
        "exact_mask_bits",
        "work_q_global",
        "work_q_local",
        "work_q_len",
        "causal_limits",
    }
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tensor"
        and node.func.attr == "record_stream"
        for node in ast.walk(tensor_loop)
    )
