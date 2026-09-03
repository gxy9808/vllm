"""Lazy public surface for Step4 sparse-indexer kernels."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

__all__ = [
    "cutedsl_topk_selector_sm90",
    "cutedsl_topk_selector_sm90_multi_cta",
    "prewarm_cutedsl_topk_selector_sm90_compilation",
]

_EXPORTS = {
    name: (
        "vllm.models.step4.nvidia.ops.cute_dsl.indexer_ops.topk_selector_sm90",
        name,
    )
    for name in __all__
}


def _make_deferred_callable(name: str) -> Callable[..., Any]:
    def _deferred(*args: Any, **kwargs: Any) -> Any:
        from .._cutlass_compat import apply_patches

        apply_patches()
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value(*args, **kwargs)

    _deferred.__name__ = name
    _deferred.__qualname__ = name
    _deferred.__module__ = __name__
    return _deferred


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = _make_deferred_callable(name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])
