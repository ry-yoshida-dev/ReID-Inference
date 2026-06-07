from __future__ import annotations

# Encoder classes are loaded lazily via __getattr__; __all__ is for API / import * .
# pyright: reportUnsupportedDunderAll=false

__all__ = [
    "FastReID",
    "TorchReID",
    "TorchReIDParameters",
    "TransReID",
    "TransReIDParameters",
]


def __getattr__(name: str):
    if name == "FastReID":
        from .fastreid import FastReID

        return FastReID
    if name == "TorchReID":
        from .torchreid import TorchReID

        return TorchReID
    if name == "TorchReIDParameters":
        from .torchreid import TorchReIDParameters

        return TorchReIDParameters
    if name == "TransReID":
        from .transreid import TransReID

        return TransReID
    if name == "TransReIDParameters":
        from .transreid import TransReIDParameters

        return TransReIDParameters
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
