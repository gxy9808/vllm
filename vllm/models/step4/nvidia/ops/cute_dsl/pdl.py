# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import cutlass.cute as cute
from cutlass._mlir.dialects import llvm


@cute.jit
def launch_dependents() -> None:
    """Signal only after stores that a programmatic dependent may consume."""
    llvm.inline_asm(
        None,
        [],
        "griddepcontrol.launch_dependents;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def wait_for_dependencies() -> None:
    """Wait immediately before the first load of predecessor-produced data."""
    llvm.inline_asm(
        None,
        [],
        "griddepcontrol.wait;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


__all__ = ["launch_dependents", "wait_for_dependencies"]
