from __future__ import annotations

# pyright: reportUnsupportedDunderAll=false

from .preprocess import DEFAULT_PREPROCESSOR_CONFIG, build_default_torch_preprocessor

__all__ = [
    "TransReID",
    "TransReIDParameters",
    "DEFAULT_PREPROCESSOR_CONFIG",
    "build_default_torch_preprocessor",
]


def __getattr__(name: str):
    if name == "TransReID":
        from .model import TransReID

        return TransReID
    if name == "TransReIDParameters":
        from .parameter import TransReIDParameters

        return TransReIDParameters
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
