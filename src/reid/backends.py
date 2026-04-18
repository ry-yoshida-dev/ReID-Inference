"""ReID inference backends: which vendored stack to load and how to build encoder + preprocessor."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from torch_modules import TorchPreprocessor

    from .model import FastReID, TorchReID, TransReID
    from .parameters import ReIDParameters


class ReIDBackend(Enum):
    """
    Vendored backend id (stable string value). Drives parameter type, encoder class, default preprocessor.

    Attributes:
    ----------
    TORCHREID : ReIDBackend
        "torchreid": TorchReID / TorchReIDParameters.
    FASTREID : ReIDBackend
        "fast_reid": FastReID / ReIDParameters (MOT17 YAML stems).
    TRANSREID : ReIDBackend
        "transreid": TransReID / TransReIDParameters.
    """

    TORCHREID = "torchreid"
    FASTREID = "fast_reid"
    TRANSREID = "transreid"

    @property
    def parameter_class(self) -> "type[ReIDParameters]":
        """Parameter dataclass for this member (TorchReIDParameters / ReIDParameters / TransReIDParameters)."""
        match self:
            case ReIDBackend.TORCHREID:
                from .model import TorchReIDParameters

                return cast("type[ReIDParameters]", TorchReIDParameters)
            case ReIDBackend.FASTREID:
                from .parameters import ReIDParameters

                return ReIDParameters
            case ReIDBackend.TRANSREID:
                from .model import TransReIDParameters

                return cast("type[ReIDParameters]", TransReIDParameters)

    @property
    def encoder_class(self) -> "type[TorchReID] | type[FastReID] | type[TransReID]":  # pyright: ignore
        """Encoder class for this member (TorchReID / FastReID / TransReID)."""
        match self:
            case ReIDBackend.TORCHREID:
                from .model import TorchReID

                return TorchReID
            case ReIDBackend.FASTREID:
                from .model import FastReID

                return FastReID
            case ReIDBackend.TRANSREID:
                from .model import TransReID

                return TransReID

    def build_default_preprocessor(self, *, is_v2_enabled: bool = True) -> TorchPreprocessor:
        """
        Default TorchPreprocessor from in-repo config (no YAML on disk).

        Parameters:
        ----------
        is_v2_enabled : bool, optional
            TorchPreprocessor schema flag (default True).

        Returns:
        -------
        TorchPreprocessor
        """
        match self:
            case ReIDBackend.FASTREID:
                from .model.fastreid.preprocess import build_default_torch_preprocessor

                return build_default_torch_preprocessor(is_v2_enabled=is_v2_enabled)
            case ReIDBackend.TORCHREID:
                from .model.torchreid.preprocess import build_default_torch_preprocessor

                return build_default_torch_preprocessor(is_v2_enabled=is_v2_enabled)
            case ReIDBackend.TRANSREID:
                from .model.transreid.preprocess import build_default_torch_preprocessor

                return build_default_torch_preprocessor(is_v2_enabled=is_v2_enabled)
