from dataclasses import dataclass
from typing import TypeVar

from .backends import ReIDBackend
from torch_modules import TorchParameters

# For Generic BaseReIDEncoder[TP]; subclasses (e.g. TransReIDParameters) use this bound.
ReIDParametersT = TypeVar("ReIDParametersT", bound="ReIDParameters")


@dataclass(kw_only=True)
class ReIDParameters(TorchParameters):
    """
    Shared ReID fields; subclasses add backend-specific options.

    Attributes:
    ----------
    backend : ReIDBackend
        Backend selector (must match encoder).
    model_name : str
        Torchreid name, FastReID YAML stem, or TransReID backbone key.
    """
    backend: ReIDBackend
    model_name: str

