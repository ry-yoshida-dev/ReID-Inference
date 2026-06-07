from dataclasses import dataclass

from ...parameters import ReIDParameters


@dataclass
class TorchReIDParameters(ReIDParameters):
    """
    Torchreid build_model / load_pretrained_weights.

    Attributes:
    ----------
    num_classes : int
        build_model width; must match checkpoint if the FC head is loaded. Other fields: ReIDParameters / TorchParameters.
    """

    num_classes: int = 2510
