# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from utils import skip_unsupported

from vllm.model_executor.layers.batch_invariant import (
    linear_fp32_batch_invariant,
)


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_linear_fp32_output_is_batch_invariant(default_vllm_config, dtype):
    torch.manual_seed(8193)
    weight = torch.randn((352, 4096), device="cuda", dtype=dtype)
    batched_input = torch.randn((257, 4096), device="cuda", dtype=dtype)

    batched_output = linear_fp32_batch_invariant(batched_input, weight)

    assert batched_output.dtype == torch.float32
    for row in (0, 42, 128, 256):
        single_output = linear_fp32_batch_invariant(
            batched_input[row : row + 1], weight
        )
        torch.testing.assert_close(
            single_output[0], batched_output[row], rtol=0, atol=0
        )


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_linear_fp32_output_matches_reference(default_vllm_config, dtype):
    torch.manual_seed(8193)
    weight = torch.randn((352, 4096), device="cuda", dtype=dtype)
    input_tensor = torch.randn((17, 4096), device="cuda", dtype=dtype)

    output = linear_fp32_batch_invariant(input_tensor, weight)
    reference = torch.nn.functional.linear(input_tensor.float(), weight.float())

    torch.testing.assert_close(output, reference, rtol=2e-3, atol=2e-2)


@pytest.mark.cpu_test
def test_step4_router_keeps_rocm_on_fallback(monkeypatch):
    import vllm.models.step4.model as step4_model

    layer = SimpleNamespace(
        weight=torch.ones((3, 4), dtype=torch.float16),
        _use_optimus_matmul_fp32=False,
    )

    inputs = torch.ones((2, 4), dtype=torch.float16)
    monkeypatch.setattr(step4_model.envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(step4_model.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(step4_model.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(
        step4_model,
        "linear_fp32_batch_invariant",
        lambda *_args, **_kwargs: pytest.fail(
            "ROCm must not enter the CUDA persistent router kernel"
        ),
    )

    output, output_bias = step4_model.FP32ReplicatedLinear.forward(layer, inputs)

    reference = torch.nn.functional.linear(
        inputs.to(torch.float32), layer.weight.to(torch.float32)
    )
    torch.testing.assert_close(output, reference)
    assert output_bias is None
