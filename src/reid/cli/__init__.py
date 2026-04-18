"""Shared helpers for command-line feature extraction and similar tools."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .feature_extract import (
    IMAGE_EXTENSIONS,
    add_common_feature_extraction_args,
    list_image_paths,
    load_rgb_images,
    parse_int_pair,
    save_per_image_features,
)

__all__ = [
    "IMAGE_EXTENSIONS",
    "FastReidFeatureCliArgs",
    "TorchReidFeatureCliArgs",
    "TransReidFeatureCliArgs",
    "add_common_feature_extraction_args",
    "build_fastreid_feature_parser",
    "build_torchreid_feature_parser",
    "build_transreid_feature_parser",
    "list_image_paths",
    "load_rgb_images",
    "parse_fastreid_feature_args",
    "parse_int_pair",
    "parse_torchreid_feature_args",
    "parse_transreid_feature_args",
    "run_fastreid_feature_extract_main",
    "run_torchreid_feature_extract_main",
    "run_transreid_feature_extract_main",
    "save_per_image_features",
]

if TYPE_CHECKING:
    from .feature_parsers import (
        FastReidFeatureCliArgs,
        TorchReidFeatureCliArgs,
        TransReidFeatureCliArgs,
        build_fastreid_feature_parser,
        build_torchreid_feature_parser,
        build_transreid_feature_parser,
        parse_fastreid_feature_args,
        parse_torchreid_feature_args,
        parse_transreid_feature_args,
        run_fastreid_feature_extract_main,
        run_torchreid_feature_extract_main,
        run_transreid_feature_extract_main,
    )

_LAZY_FROM_FEATURE_PARSERS = frozenset(
    {
        "FastReidFeatureCliArgs",
        "TorchReidFeatureCliArgs",
        "TransReidFeatureCliArgs",
        "build_fastreid_feature_parser",
        "build_torchreid_feature_parser",
        "build_transreid_feature_parser",
        "parse_fastreid_feature_args",
        "parse_torchreid_feature_args",
        "parse_transreid_feature_args",
        "run_fastreid_feature_extract_main",
        "run_torchreid_feature_extract_main",
        "run_transreid_feature_extract_main",
    }
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_FROM_FEATURE_PARSERS:
        fp = importlib.import_module(f"{__name__}.feature_parsers")
        return getattr(fp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
