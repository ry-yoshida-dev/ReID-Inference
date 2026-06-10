import logging
import torch

from torchreid.models import build_model
from torchreid.utils import load_pretrained_weights

from ...encoder import BaseReIDEncoder
from .parameter import TorchReIDParameters

logger = logging.getLogger(__name__)


class TorchReID(BaseReIDEncoder[TorchReIDParameters]):
    """
    Torchreid via the packaged deep-person-reid distribution (import name: torchreid).

    Attributes:
    ----------
    Same as BaseReIDEncoder; see TorchReIDParameters for model_name / num_classes / weights, etc.
    """

    def _load_model(self) -> torch.nn.Module:
        """
        Returns:
        -------
        torch.nn.Module
            After load_pretrained_weights. See https://kaiyangzhou.github.io/deep-person-reid/user_guide
        """
        model_name = self.parameters.model_name
        weights_path = self.parameters.weights_path

        logger.info("%s from packaged Torchreid is adopted as the ReID model.", model_name)

        model = build_model(
            name=model_name,
            num_classes=self.parameters.num_classes,
            loss="softmax",
            pretrained=False,
        )
        load_pretrained_weights(model, weights_path)
        return model

    def __str__(self) -> str:
        return f"TorchReID(model_name={self.parameters.model_name}, weights_path={self.parameters.weights_path})"
