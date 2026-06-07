"""Re-identification inference."""

# Subpackages are loaded lazily via __getattr__; __all__ documents the public API.
# pyright: reportUnsupportedDunderAll=false

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "backends",
    "encoder",
    "model",
    "parameters",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
