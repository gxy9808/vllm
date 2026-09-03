# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step4 model entry point.

The DSA attention backends live in this package too (``sparse_attention``,
``sparse_summary_cache``) because they are Step4-specific and are selected by
the model rather than by platform detection. There is no per-platform split:
the CuTeDSL sparse kernels are SM90-only.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Step4ForCausalLM
    from .mtp import Step4MTP

__all__ = ["Step4ForCausalLM", "Step4MTP"]


def __getattr__(name: str):
    if name == "Step4ForCausalLM":
        from .model import Step4ForCausalLM

        return Step4ForCausalLM
    if name == "Step4MTP":
        from .mtp import Step4MTP

        return Step4MTP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
