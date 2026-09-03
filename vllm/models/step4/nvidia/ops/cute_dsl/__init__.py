"""Step4 CuTeDSL inference kernels.

This package root is intentionally import-safe.  CUTLASS/CuTe modules and the
Step4 runtime compatibility patches are loaded only when an exported operator
is called, so importing the Step4 model registry does not initialize CUDA.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

__all__ = [
    "fused_indexer_norm_rope_forward_impl",
    "fused_qknorm_rope_forward_impl",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "fused_qknorm_rope_forward_impl": (
        "vllm.models.step4.nvidia.ops.cute_dsl.qknorm_rope",
        "fused_qknorm_rope_forward_impl",
    ),
    "fused_indexer_norm_rope_forward_impl": (
        "vllm.models.step4.nvidia.ops.cute_dsl.indexer_norm_rope",
        "fused_indexer_norm_rope_forward_impl",
    ),
}

_RUNTIME_READY = False
_CUTEDSL_AVAILABLE: bool | None = None


def _ensure_runtime_ready() -> None:
    global _RUNTIME_READY
    if _RUNTIME_READY:
        return

    from ._cutlass_compat import apply_patches

    apply_patches()
    _RUNTIME_READY = True


def _load_export(name: str) -> Any:
    _ensure_runtime_ready()
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def _make_deferred_callable(name: str) -> Callable[..., Any]:
    def _deferred(*args: Any, **kwargs: Any) -> Any:
        impl = _load_export(name)
        try:
            return impl(*args, **kwargs)
        finally:
            # A first-call CuTeDSL compile may retain the calling frame.
            # Clear tensor-bearing argument slots before returning.
            del args, kwargs

    _deferred.__name__ = name
    _deferred.__qualname__ = name
    _deferred.__module__ = __name__
    _deferred.__doc__ = (
        f"Deferred loader for {__name__}.{name}; the kernel and CUTLASS "
        "compatibility layer are loaded on first call."
    )
    return _deferred


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = _make_deferred_callable(name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])


def is_available() -> bool:
    """Return whether the CUTLASS Python DSL can be imported."""
    global _CUTEDSL_AVAILABLE
    if _CUTEDSL_AVAILABLE is not None:
        return _CUTEDSL_AVAILABLE
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
    except ImportError:
        _CUTEDSL_AVAILABLE = False
    else:
        _CUTEDSL_AVAILABLE = True
    return _CUTEDSL_AVAILABLE
