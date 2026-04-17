"""Locations of vendored code under ``reid.external`` (single place to update paths)."""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent

EXTERNAL_ROOT = _PKG_DIR / "external"

FAST_REID_ROOT = EXTERNAL_ROOT / "fast_reid"
# Imports are ``from fast_reid.fastreid...``; the ``fast_reid`` package lives under ``EXTERNAL_ROOT``.
FAST_REID_IMPORT_ROOT = EXTERNAL_ROOT

DEEP_PERSON_REID_ROOT = EXTERNAL_ROOT / "deep-person-reid"

TRANSREID_PYTORCH_ROOT = EXTERNAL_ROOT / "TransReID-SSL" / "transreid_pytorch"


def ensure_syspath(directory: Path) -> None:
    """Prepend *directory* to ``sys.path`` if it is not already present."""
    resolved = str(directory.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
