# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step4 normalization layers and custom-op registrations."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm.kernels  # noqa: F401
from vllm import envs
from vllm.utils.torch_utils import direct_register_custom_op


def _has_optimus_rms_norm_op() -> bool:
    return hasattr(torch.ops, "Optimus") and hasattr(
        torch.ops.Optimus, "RMSNorm_forward"
    )


def _has_stepfun_fused_add_rms_norm_op() -> bool:
    return hasattr(torch.ops, "_C") and hasattr(
        torch.ops._C, "optimus_fused_add_rms_norm"
    )


def _optimus_rms_norm_native(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    zero_centered: bool,
) -> torch.Tensor:
    compute = x.float()
    variance = compute.pow(2).mean(dim=-1, keepdim=True)
    compute = compute * torch.rsqrt(variance + variance_epsilon)
    scale = weight.float()
    if zero_centered:
        scale = scale + 1.0
    return (compute * scale).to(x.dtype)


def apply_optimus_rms_norm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    out: torch.Tensor | None = None,
    zero_centered: bool = False,
) -> torch.Tensor:
    del weight, variance_epsilon, zero_centered
    return torch.empty_like(x) if out is None else out


def apply_optimus_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    out: torch.Tensor | None = None,
    zero_centered: bool = False,
) -> torch.Tensor:
    if envs.VLLM_STEP_CC_LEVEL >= 1 or not _has_optimus_rms_norm_op():
        result = _optimus_rms_norm_native(x, weight, variance_epsilon, zero_centered)
    else:
        result, _ = torch.ops.Optimus.RMSNorm_forward(
            x,
            weight,
            variance_epsilon,
            zero_centered=zero_centered,
        )
    if out is None:
        return result
    if out.shape != result.shape or out.dtype != result.dtype:
        raise ValueError(
            "Optimus RMSNorm output buffer must match the computed output "
            f"shape/dtype, got out={tuple(out.shape)}/{out.dtype}, "
            f"result={tuple(result.shape)}/{result.dtype}."
        )
    out.copy_(result)
    return out


direct_register_custom_op(
    op_name="optimus_rms_norm",
    op_func=apply_optimus_rms_norm,
    mutates_args=["out"],
    fake_impl=apply_optimus_rms_norm_fake,
)


def apply_optimus_fused_add_rms_norm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    zero_centered: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    del weight, variance_epsilon, zero_centered
    return torch.empty_like(x), torch.empty_like(residual)


def apply_optimus_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    zero_centered: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if envs.VLLM_STEP_CC_LEVEL < 1 and _has_optimus_rms_norm_op():
        output, _, residual_out = torch.ops.Optimus.RMSNorm_forward(
            x,
            weight,
            variance_epsilon,
            residual,
            zero_centered=zero_centered,
        )
        return output, residual_out

    if _has_stepfun_fused_add_rms_norm_op():
        output = torch.empty_like(x)
        residual_out = torch.empty_like(residual)
        torch.ops._C.optimus_fused_add_rms_norm(
            output,
            residual_out,
            x,
            residual,
            weight,
            variance_epsilon,
            zero_centered,
        )
        return output, residual_out

    orig_dtype = x.dtype
    residual_out = (
        (x.float() + residual.float()).to(residual.dtype)
        if orig_dtype == torch.float16
        else x + residual
    )
    output = _optimus_rms_norm_native(
        residual_out,
        weight,
        variance_epsilon,
        zero_centered,
    ).to(orig_dtype)
    return output, residual_out


direct_register_custom_op(
    op_name="optimus_fused_add_rms_norm",
    op_func=apply_optimus_fused_add_rms_norm,
    mutates_args=[],
    fake_impl=apply_optimus_fused_add_rms_norm_fake,
)


class OptimusRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        zero_centered: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps
        self.zero_centered = zero_centered

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
        fp16_out: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            if output is not None or fp16_out:
                raise ValueError(
                    "Residual OptimusRMSNorm does not support output buffers "
                    "or fp16_out."
                )
            if self.zero_centered:
                return torch.ops.vllm.optimus_fused_add_rms_norm(
                    x,
                    residual,
                    self.weight,
                    self.variance_epsilon,
                    zero_centered=True,
                )

            from vllm import _custom_ops as ops

            ops.fused_add_rms_norm(
                x,
                residual,
                self.weight.data,
                self.variance_epsilon,
            )
            return x, residual

        if fp16_out:
            raise ValueError("OptimusRMSNorm does not support fp16_out.")
        return torch.ops.vllm.optimus_rms_norm(
            x,
            self.weight,
            self.variance_epsilon,
            out=output,
            zero_centered=self.zero_centered,
        )


class OptimusLayerNorm(nn.Module):
    """Per-head LayerNorm for the Step4 sparse-attention indexer."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(
            x.unflatten(-1, (-1, self.hidden_size)),
            (self.hidden_size,),
            self.weight,
            self.bias,
            self.variance_epsilon,
        )
        return normalized.flatten(-2, -1)
