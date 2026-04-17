from dataclasses import dataclass, field

from ...parameters import ReIDParameters


@dataclass
class TransReIDParameters(ReIDParameters):
    """
    Parameters for the TransReID model.
    Extends ReIDParameters with TransReID-specific options.

    Parameters:
    ----------
    input_size: list[int]
        The input size of the model.
    stride: list[int]
        The stride of the model.
    hw_ratio: float
        The HW ratio of the model.
    """
    input_size: list[int] = field(default_factory=lambda: [256, 128])
    stride: list[int] = field(default_factory=lambda: [16, 16])
    hw_ratio: float = 2.0

    def __post_init__(self):
        if self.hw_ratio <= 0:
            raise ValueError("hw_ratio must be a positive number")
