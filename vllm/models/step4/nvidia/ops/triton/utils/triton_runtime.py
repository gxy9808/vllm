"""Triton runtime helpers used across kernels."""

from __future__ import annotations

import re
from typing import Union

import triton

try:  # pragma: no cover - torch may be missing in limited environments
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from triton.language import extra as _triton_extra
except ImportError:  # pragma: no cover - depends on Triton build
    _triton_extra = None


def _parse_version_str(version_str: str | None):
    numbers = [int(x) for x in re.findall(r"\d+", version_str or "0.0.0")]
    while len(numbers) < 3:
        numbers.append(0)
    return tuple(numbers[:3])


TRITON_VERSION = _parse_version_str(getattr(triton, "__version__", "0.0.0"))
# Triton exposes tl.extra.cuda.* starting from 3.5.0
TRITON_SUPPORTS_GDC_HINTS = bool(_triton_extra) and TRITON_VERSION >= (3, 5, 0)


def _detect_pdl_support_once() -> bool:
    if torch is None or not TRITON_SUPPORTS_GDC_HINTS:
        return False
    if not torch.cuda.is_available():
        return False
    device = torch.device("cuda", torch.cuda.current_device())
    if device.type != "cuda":
        return False
    cap_major, _cap_minor = torch.cuda.get_device_capability(device)
    return cap_major >= 9


_PDL_SUPPORTED_DEFAULT = None


def _get_pdl_supported():
    global _PDL_SUPPORTED_DEFAULT
    if _PDL_SUPPORTED_DEFAULT is None:
        _PDL_SUPPORTED_DEFAULT = _detect_pdl_support_once()
    return _PDL_SUPPORTED_DEFAULT


def is_pdl_supported(device=None) -> bool:
    """Return cached PDL capability detected lazily."""
    return _get_pdl_supported()


def resolve_launch_pdl_flag() -> bool:
    """Return cached flag indicating whether PDL can be enabled."""
    return _get_pdl_supported()


__all__ = [
    "TRITON_SUPPORTS_GDC_HINTS",
    "TRITON_VERSION",
    "is_pdl_supported",
    "resolve_launch_pdl_flag",
]
