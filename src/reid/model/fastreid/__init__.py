from __future__ import annotations

# pyright: reportUnsupportedDunderAll=false

from .preprocess import DEFAULT_PREPROCESSOR_CONFIG, build_default_torch_preprocessor

__all__ = [
    "FastReID",
    "DEFAULT_PREPROCESSOR_CONFIG",
    "build_default_torch_preprocessor",
]


def __getattr__(name: str):
    if name == "FastReID":
        from .model import FastReID

        return FastReID
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
