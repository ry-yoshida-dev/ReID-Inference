from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model.fastreid import FastReID
    from .model.torchreid import TorchReID
    from .model.transreid import TransReID
    from .parameters import ReIDParameters


class ReID_Library(Enum):
    """
    Enum defining the libraries for reid.

    Attributes:
    ----------
    TORCHREID: Torchreid
    FASTREID: FastReID
    TRANSREID: TransReID
    """

    TORCHREID = "torchreid"
    FASTREID = "fast_reid"
    TRANSREID = "transreid"

    @property
    def parameter_class(self) -> "type[ReIDParameters]":
        """Parameter dataclass used for this library (e.g. TransReIDParameters)."""
        match self:
            case ReID_Library.TORCHREID | ReID_Library.FASTREID:
                from .parameters import ReIDParameters
                return ReIDParameters
            case ReID_Library.TRANSREID:
                from .model.transreid import TransReIDParameters
                return TransReIDParameters

    @property
    def encoder_class(self) -> "type[TorchReID] | type[FastReID] | type[TransReID]":
        """Encoder class for this library (e.g. TransReID)."""
        match self:
            case ReID_Library.TORCHREID:
                from .model.torchreid import TorchReID
                return TorchReID
            case ReID_Library.FASTREID:
                from .model.fastreid import FastReID
                return FastReID
            case ReID_Library.TRANSREID:
                from .model.transreid import TransReID
                return TransReID

