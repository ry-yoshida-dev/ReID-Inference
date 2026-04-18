"""Default TorchPreprocessor for FastReID (in-repo DEFAULT_PREPROCESSOR_CONFIG)."""

from __future__ import annotations

from typing import Any

from torch_modules import TorchPreprocessor

DEFAULT_PREPROCESSOR_CONFIG: dict[str, Any] = {
    "transforms": [
        {"name": "Resize", "params": {"size": [384, 128]}},
        {"name": "ToImage"},
        {"name": "Lambda", "params": {"func_name": "bgr_conversion"}},
        {"name": "ToDtype", "params": {"dtype": "float32", "scale": False}},
    ]
}


def build_default_torch_preprocessor(*, is_v2_enabled: bool = True) -> TorchPreprocessor:
    return TorchPreprocessor(DEFAULT_PREPROCESSOR_CONFIG, is_v2_enabled=is_v2_enabled)
