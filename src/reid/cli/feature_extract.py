from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)


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
    features: np.ndarray,
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
