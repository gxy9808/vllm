# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import pytest
import torch
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd

import vllm.config
from vllm.compilation.passes.fusion.optimus_rms_fusion import (
    OptimusRMSNormFusionPass,
)
from vllm.compilation.passes.fx_utils import find_op_nodes
from vllm.config import CompilationConfig, PassConfig, VllmConfig
from vllm.platforms import current_platform


class AddThenOptimusRMSNorm(torch.nn.Module):
    def __init__(self, norm: torch.nn.Module) -> None:
        super().__init__()
        self.norm = norm

    def forward(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.compiler.is_compiling():
            residual_out = input + residual
            return self.norm(residual_out), residual_out
        return self.norm(input, residual=residual)


@pytest.mark.parametrize("epsilon", [1e-5, 1e-6])
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Optimus is CUDA-only")
def test_optimus_add_rmsnorm_fusion(epsilon: float) -> None:
    from vllm.models.step4.layernorm import OptimusRMSNorm

    config = VllmConfig(
        compilation_config=CompilationConfig(
            pass_config=PassConfig(fuse_optimus_rms=True),
        ),
    )

    with vllm.config.set_current_vllm_config(config):
        torch.set_default_device("cuda")
        norm = OptimusRMSNorm(
            256,
            eps=epsilon,
            zero_centered=True,
            dtype=torch.float32,
        )
        model = AddThenOptimusRMSNorm(norm)
        input = torch.randn(17, 256, dtype=torch.float32)
        residual = torch.randn_like(input)
        expected = model(input, residual)

        fusion_pass = OptimusRMSNormFusionPass(config)
        op_counts: dict[str, int] = {}

        def fw_compiler(
            graph_module: torch.fx.GraphModule,
            _: list[object],
        ) -> Callable:
            graph = graph_module.graph
            op_counts["add_before"] = len(
                list(find_op_nodes(torch.ops.aten.add.Tensor, graph))
            )
            fusion_pass(graph)
            op_counts["add_after"] = len(
                list(find_op_nodes(torch.ops.aten.add.Tensor, graph))
            )
            op_counts["fused_after"] = len(
                list(
                    find_op_nodes(
                        torch.ops.vllm.optimus_fused_add_rms_norm.default,
                        graph,
                    )
                )
            )
            graph.lint()
            graph_module.recompile()
            return make_boxed_func(graph_module.forward)

        backend = aot_autograd(fw_compiler=fw_compiler)
        with torch._inductor.config.patch(enable_auto_functionalized_v2=False):
            actual = torch.compile(model, backend=backend)(input, residual)

    torch.testing.assert_close(actual, expected)
    assert fusion_pass.matched_count == 1
    assert op_counts == {
        "add_before": 1,
        "add_after": 0,
        "fused_after": 1,
    }
