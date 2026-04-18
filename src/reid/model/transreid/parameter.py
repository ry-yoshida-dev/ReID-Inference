from dataclasses import dataclass, field

from ...parameters import ReIDParameters


@dataclass
class TransReIDParameters(ReIDParameters):
    """
    TransReID backbone + load_param.

    Attributes:
    ----------
    input_size : list[int]
        [H, W] → img_size.
    stride : list[int]
        → stride_size.
    hw_ratio : float
        load_param pos-embed resize; > 0. model_name: see TransReID._import_backbone.
    """
    input_size: list[int] = field(default_factory=lambda: [256, 128])
    stride: list[int] = field(default_factory=lambda: [16, 16])
    hw_ratio: float = 2.0

    def __post_init__(self) -> None:
        """Assert hw_ratio > 0."""
        if self.hw_ratio <= 0:
            raise ValueError("hw_ratio must be a positive number")
