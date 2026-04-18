"""Default TorchPreprocessor for Torchreid (in-repo DEFAULT_PREPROCESSOR_CONFIG)."""

from __future__ import annotations

from typing import Any

from torch_modules import TorchPreprocessor

DEFAULT_PREPROCESSOR_CONFIG: dict[str, Any] = {
    "transforms": [
        {"name": "Resize", "params": {"size": [256, 128]}},
        {"name": "ToImage"},
        {
            "name": "ToDtype",
            "params": {"dtype": "float32", "scale": True},
        },
        {
            "name": "Normalize",
            "params": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
    ]
}


def build_default_torch_preprocessor(*, is_v2_enabled: bool = True) -> TorchPreprocessor:
    return TorchPreprocessor(DEFAULT_PREPROCESSOR_CONFIG, is_v2_enabled=is_v2_enabled)
