"""Roots for vendored trees under reid.external (this directory)."""

from __future__ import annotations

import sys
from pathlib import Path

EXTERNAL_ROOT = Path(__file__).resolve().parent

FAST_REID_ROOT = EXTERNAL_ROOT / "fast_reid"
# Imports use from fast_reid.fastreid...; the fast_reid package lives under EXTERNAL_ROOT.
FAST_REID_IMPORT_ROOT = EXTERNAL_ROOT

DEEP_PERSON_REID_ROOT = EXTERNAL_ROOT / "deep-person-reid"

TRANSREID_PYTORCH_ROOT = EXTERNAL_ROOT / "TransReID-SSL" / "transreid_pytorch"


def ensure_syspath(directory: Path) -> None:
    """Prepend *directory* to sys.path if it is not already present."""
    resolved = str(directory.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
