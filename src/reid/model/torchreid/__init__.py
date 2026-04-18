from __future__ import annotations

# pyright: reportUnsupportedDunderAll=false

from .preprocess import DEFAULT_PREPROCESSOR_CONFIG, build_default_torch_preprocessor

__all__ = [
    "TorchReID",
    "TorchReIDParameters",
    "DEFAULT_PREPROCESSOR_CONFIG",
    "build_default_torch_preprocessor",
]


def __getattr__(name: str):
    if name == "TorchReID":
        from .model import TorchReID

        return TorchReID
    if name == "TorchReIDParameters":
        from .parameter import TorchReIDParameters

        return TorchReIDParameters
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
