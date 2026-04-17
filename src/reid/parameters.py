from dataclasses import dataclass
from typing import TypeVar

from .names import ReID_Library
from torch_modules import TorchParameters

# For Generic ReIDEncoder[TP]; subclasses (e.g. TransReIDParameters) use this bound.
ReIDParametersT = TypeVar("ReIDParametersT", bound="ReIDParameters")


@dataclass(kw_only=True)
class ReIDParameters(TorchParameters):
    """
    The parameters for the reid model.

    Parameters:
    ----------
    library_name: ReIDName
        The name of the reid library.
    model_name: str
        The name of the reid model.
    """
    library_name: ReID_Library
    model_name: str

