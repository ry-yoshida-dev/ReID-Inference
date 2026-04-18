from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from .feature_extract import (
    add_common_feature_extraction_args,
    list_image_paths,
    load_rgb_images,
    parse_int_pair,
    save_per_image_features,
)
from ..encoder import ReIDEncoder
from ..backends import ReIDBackend
from ..parameters import ReIDParameters


class BaseFeatureExtractCliArgs(argparse.Namespace):
    images_dir: Path
    output_dir: Path
    weights: Path
    batch_size: int
    no_fp16: bool


class TorchReidFeatureCliArgs(BaseFeatureExtractCliArgs):
    model_name: str
    num_classes: int


class FastReidFeatureCliArgs(BaseFeatureExtractCliArgs):
    model_name: str


class TransReidFeatureCliArgs(BaseFeatureExtractCliArgs):
    model_name: str
    input_size: list[int]
    stride: list[int]
    hw_ratio: float


def build_torchreid_feature_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract Torchreid ReID features and save as .npy per image."
    )
    add_common_feature_extraction_args(p)
    p.add_argument(
        "--model-name",
        type=str,
        default="osnet_ain_x1_0",
        help="torchreid build_model name (must match the checkpoint architecture).",
    )
    p.add_argument(
        "--num-classes",
        type=int,
        default=2510,
        help="Classifier size for build_model (should match checkpoint when weights include the head).",
    )
    return p


def build_fastreid_feature_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract FastReID features and save as .npy per image."
    )
    add_common_feature_extraction_args(p)
    p.add_argument(
        "--model-name",
        type=str,
        default="sbs_S50",
        help=(
            "YAML stem under fast_reid configs/MOT17 (e.g. sbs_S50 for "
            "configs/MOT17/sbs_S50.yml)."
        ),
    )
    return p


def build_transreid_feature_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract TransReID features and save as .npy per image."
    )
    add_common_feature_extraction_args(p)
    p.add_argument(
        "--model-name",
        type=str,
        default="vit_small_patch16_224_TransReID",
        help="TransReID backbone registered in TransReID._import_backbone.",
    )
    p.add_argument(
        "--input-size",
        type=parse_int_pair,
        default="256,128",
        help="Input resolution [H, W] as H,W (e.g. 256,128).",
    )
    p.add_argument(
        "--stride",
        type=parse_int_pair,
        default="16,16",
        help="Stride pair as two integers (e.g. 16,16).",
    )
    p.add_argument(
        "--hw-ratio",
        type=float,
        default=2.0,
        help="Aspect ratio hint for load_param positional embedding resize.",
    )
    return p


def parse_torchreid_feature_args(argv: list[str] | None = None) -> TorchReidFeatureCliArgs:
    return build_torchreid_feature_parser().parse_args(argv, namespace=TorchReidFeatureCliArgs())


def parse_fastreid_feature_args(argv: list[str] | None = None) -> FastReidFeatureCliArgs:
    return build_fastreid_feature_parser().parse_args(argv, namespace=FastReidFeatureCliArgs())


def parse_transreid_feature_args(argv: list[str] | None = None) -> TransReidFeatureCliArgs:
    return build_transreid_feature_parser().parse_args(argv, namespace=TransReidFeatureCliArgs())


def _startup_paths_and_images(args: BaseFeatureExtractCliArgs) -> tuple[Path, list[Path]]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        logging.error("Weights file not found: %s", weights)
        sys.exit(1)

    image_paths = list_image_paths(args.images_dir.expanduser().resolve())
    if not image_paths:
        logging.error("No images found in %s", args.images_dir)
        sys.exit(1)

    return weights, image_paths


def _write_features(
    args: BaseFeatureExtractCliArgs,
    encoder: ReIDEncoder[Any],
    image_paths: list[Path],
) -> None:
    images = load_rgb_images(image_paths)
    features = encoder.predict_from_images(images)
    save_per_image_features(args.output_dir.expanduser().resolve(), image_paths, features)
    logging.info("Wrote %d feature files under %s", len(image_paths), args.output_dir)


def run_torchreid_feature_extract_main() -> None:
    from ..model.torchreid.model import TorchReID
    from ..model.torchreid.parameter import TorchReIDParameters
    from ..model.torchreid.preprocess import build_default_torch_preprocessor

    args = parse_torchreid_feature_args()
    weights, image_paths = _startup_paths_and_images(args)

    params = TorchReIDParameters(
        weights_path=str(weights),
        batch_size=args.batch_size,
        is_half_precision_enabled=not args.no_fp16,
        backend=ReIDBackend.TORCHREID,
        model_name=args.model_name,
        num_classes=args.num_classes,
    )
    encoder = TorchReID(
        preprocessor=build_default_torch_preprocessor(),
        parameters=params,
    )
    _write_features(args, encoder, image_paths)


def run_fastreid_feature_extract_main() -> None:
    from ..model.fastreid.model import FastReID
    from ..model.fastreid.preprocess import build_default_torch_preprocessor

    args = parse_fastreid_feature_args()
    weights, image_paths = _startup_paths_and_images(args)

    params = ReIDParameters(
        weights_path=str(weights),
        batch_size=args.batch_size,
        is_half_precision_enabled=not args.no_fp16,
        backend=ReIDBackend.FASTREID,
        model_name=args.model_name,
    )
    encoder = FastReID(
        preprocessor=build_default_torch_preprocessor(),
        parameters=params,
    )
    _write_features(args, encoder, image_paths)


def run_transreid_feature_extract_main() -> None:
    from ..model.transreid.model import TransReID
    from ..model.transreid.parameter import TransReIDParameters
    from ..model.transreid.preprocess import build_default_torch_preprocessor

    args = parse_transreid_feature_args()
    weights, image_paths = _startup_paths_and_images(args)

    params = TransReIDParameters(
        weights_path=str(weights),
        batch_size=args.batch_size,
        is_half_precision_enabled=not args.no_fp16,
        backend=ReIDBackend.TRANSREID,
        model_name=args.model_name,
        input_size=args.input_size,
        stride=args.stride,
        hw_ratio=args.hw_ratio,
    )
    encoder = TransReID(
        preprocessor=build_default_torch_preprocessor(),
        parameters=params,
    )
    _write_features(args, encoder, image_paths)
