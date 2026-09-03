# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig
from vllm.logger import init_logger

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)


class OptimusAddRMSNormPattern:
    """Match residual add followed by zero-centered Optimus RMSNorm."""

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> None:
        self.epsilon = epsilon
        self.dtype = dtype
        self.device = device

    def register(self, pm_pass: PatternMatcherPass) -> None:
        optimus_rms_norm = torch.ops.vllm.optimus_rms_norm.default
        optimus_fused_add_rms_norm = torch.ops.vllm.optimus_fused_add_rms_norm.default

        def pattern(
            input: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            residual_out = torch.ops.aten.add.Tensor(input, residual)
            output, _ = auto_functionalized(
                optimus_rms_norm,
                x=residual_out,
                weight=weight,
                variance_epsilon=self.epsilon,
                out=None,
                zero_centered=True,
            )
            return output, residual_out

        def replacement(
            input: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return optimus_fused_add_rms_norm(
                input,
                residual,
                weight,
                self.epsilon,
                zero_centered=True,
            )

        inputs = [
            torch.empty(5, 16, dtype=self.dtype, device=self.device),
            torch.empty(5, 16, dtype=self.dtype, device=self.device),
            torch.empty(16, dtype=torch.float32, device=self.device),
        ]
        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
        )


class OptimusRMSNormFusionPass(VllmPatternMatcherPass):
    """Fuse residual add and zero-centered Optimus RMSNorm."""

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.patterns = PatternMatcherPass(pass_name="optimus_rmsnorm_fusion_pass")

        if not (
            hasattr(torch.ops.vllm, "optimus_rms_norm")
            and hasattr(torch.ops.vllm, "optimus_fused_add_rms_norm")
        ):
            logger.debug("Optimus RMSNorm custom ops are unavailable; skipping pass")
            return

        dtype = config.model_config.dtype if config.model_config else torch.bfloat16
        device = config.device_config.device if config.device_config else "cuda"
        for epsilon in (1e-5, 1e-6):
            OptimusAddRMSNormPattern(epsilon, dtype, device).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        VllmPatternMatcherPass.match_table[self.pass_name] += self.matched_count

    def uuid(self) -> str:
        return self.hash_source(self, OptimusAddRMSNormPattern)
