"""Command-line feature extraction for ReID backends."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from reid.array_types import FloatArray
from reid.backends import ReIDBackend
from reid.encoder import BaseReIDEncoder
from reid.parameters import ReIDParameters, ReIDParametersT

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)


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


def list_image_paths(images_dir: Path) -> list[Path]:
    """Return sorted image paths under images_dir (non-recursive)."""
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {images_dir}")
    paths = [
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=lambda p: p.name.lower())


def add_common_feature_extraction_args(parser: argparse.ArgumentParser) -> None:
    """Register flags shared by all backend feature extractors."""
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("image"),
        help="Directory containing input images (top level only).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result"),
        help="Directory to write <image_stem>.npy feature files.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the model checkpoint / weights file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for the encoder DataLoader.",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable float16 inference on GPU (use float32).",
    )


def parse_int_pair(value: str) -> list[int]:
    """Parse 'H,W' into [H, W] for argparse type=."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        msg = f"Expected two integers as H,W; got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    try:
        return [int(parts[0]), int(parts[1])]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid integer pair: {value!r}") from e


def load_rgb_images(paths: list[Path]) -> list[Image.Image]:
    return [Image.open(p).convert("RGB") for p in paths]


def save_per_image_features(
    output_dir: Path,
    image_paths: list[Path],
    features: FloatArray,
) -> None:
    """Save one .npy per image using path.stem as the file name."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features (N, D); got shape {features.shape}")
    if len(image_paths) != features.shape[0]:
        raise ValueError(
            f"len(image_paths)={len(image_paths)} != features.shape[0]={features.shape[0]}"
        )
    for path, row in zip(image_paths, features, strict=True):
        out_path = output_dir / f"{path.stem}.npy"
        np.save(out_path, row)


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
        help="TransReID backbone name (see transreid_pytorch.model.backbones.BACKBONE_FACTORY).",
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
    encoder: BaseReIDEncoder[ReIDParametersT],
    image_paths: list[Path],
) -> None:
    images = load_rgb_images(image_paths)
    features = encoder.predict_from_images(images)
    save_per_image_features(args.output_dir.expanduser().resolve(), image_paths, features)
    logging.info("Wrote %d feature files under %s", len(image_paths), args.output_dir)


def run_torchreid_feature_extract(args: TorchReidFeatureCliArgs) -> None:
    from reid.model.torchreid.model import TorchReID
    from reid.model.torchreid.parameter import TorchReIDParameters
    from reid.model.torchreid.preprocess import build_default_torch_preprocessor

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


def run_fastreid_feature_extract(args: FastReidFeatureCliArgs) -> None:
    from reid.model.fastreid.model import FastReID
    from reid.model.fastreid.preprocess import build_default_torch_preprocessor

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


def run_transreid_feature_extract(args: TransReidFeatureCliArgs) -> None:
    from reid.model.transreid.model import TransReID
    from reid.model.transreid.parameter import TransReIDParameters
    from reid.model.transreid.preprocess import build_default_torch_preprocessor

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


def _print_unified_help() -> None:
    print(
        "usage: reid-extract {torchreid,fastreid,transreid} [options]\n\n"
        "Backends:\n"
        "  torchreid   Torchreid (deep-person-reid)\n"
        "  fastreid    FastReID\n"
        "  transreid   TransReID\n\n"
        "Run with a backend name and --help for backend-specific flags, e.g.:\n"
        "  reid-extract torchreid --help"
    )


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        if argv and argv[0] in {"-h", "--help"}:
            _print_unified_help()
            sys.exit(0)
        _print_unified_help()
        sys.exit(2)

    match argv[0]:
        case "torchreid":
            args = build_torchreid_feature_parser().parse_args(
                argv[1:], namespace=TorchReidFeatureCliArgs()
            )
            run_torchreid_feature_extract(args)
        case "fastreid":
            args = build_fastreid_feature_parser().parse_args(
                argv[1:], namespace=FastReidFeatureCliArgs()
            )
            run_fastreid_feature_extract(args)
        case "transreid":
            args = build_transreid_feature_parser().parse_args(
                argv[1:], namespace=TransReidFeatureCliArgs()
            )
            run_transreid_feature_extract(args)
        case _:
            _print_unified_help()
            sys.exit(2)


if __name__ == "__main__":
    main()
