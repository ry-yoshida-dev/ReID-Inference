"""Re-identification inference."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .backends import ReIDBackend
from .parameters import ReIDParameters, ReIDParametersT
from .protocol import ReIDEncoder

# Core symbols are imported eagerly; encoders and subpackages load lazily via __getattr__.
# pyright: reportUnsupportedDunderAll=false

__all__ = [
    "ReIDBackend",
    "BaseReIDEncoder",
    "ReIDEncoder",
    "ReIDParameters",
    "ReIDParametersT",
    "FastReID",
    "TorchReID",
    "TransReID",
    "TorchReIDParameters",
    "TransReIDParameters",
    "backends",
    "encoder",
    "model",
    "parameters",
]

_LAZY_FROM_ENCODER = frozenset({"BaseReIDEncoder"})
_LAZY_FROM_MODEL = frozenset(
    {
        "FastReID",
        "TorchReID",
        "TransReID",
        "TorchReIDParameters",
        "TransReIDParameters",
    }
)
_LAZY_SUBMODULES = frozenset({"backends", "encoder", "model", "parameters"})

if TYPE_CHECKING:
    from .encoder import BaseReIDEncoder
    from .model import (
        FastReID,
        TorchReID,
        TorchReIDParameters,
        TransReID,
        TransReIDParameters,
    )


def __getattr__(name: str) -> Any:
    if name in _LAZY_FROM_ENCODER:
        from .encoder import BaseReIDEncoder

        return BaseReIDEncoder
    if name in _LAZY_FROM_MODEL:
        model = importlib.import_module(f"{__name__}.model")
        return getattr(model, name)
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
