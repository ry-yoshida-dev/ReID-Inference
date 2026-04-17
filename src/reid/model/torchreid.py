import sys
import logging
import torch
from typing import cast

from ..encoder import ReIDEncoder
from ..parameters import ReIDParameters

logger = logging.getLogger(__name__)


class TorchReID(ReIDEncoder[ReIDParameters]):
    """
    Torchreid

    Attributes:
    ----------
    model_name: str
        The name of the model.
    weights_path: str
        The path to the weights file.

    Returns:
    -------
    torch.nn.Module: The model.
    """

    def _load_model(self) -> torch.nn.Module:

        """
        Load the model.

        https://kaiyangzhou.github.io/deep-person-reid/user_guide
        """
        model_name = self.parameters.model_name
        weights_path = self.parameters.weights_path

        logger.info("%s from Torchreid is adopted as the ReID model.", model_name)
        sys.path.append("./src/external/deep-person-reid")

        from torchreid.models import build_model # type: ignore
        from torchreid.utils import load_pretrained_weights # type: ignore

        model: torch.nn.Module = cast(
            torch.nn.Module, 
            build_model(
                name=model_name, 
                num_classes=2510, 
                loss="softmax", 
                pretrained=False
            ))
        load_pretrained_weights(model, weights_path)
        return model

    def __str__(self) -> str:
        return f"TorchReID(model_name={self.parameters.model_name}, weights_path={self.parameters.weights_path})"
