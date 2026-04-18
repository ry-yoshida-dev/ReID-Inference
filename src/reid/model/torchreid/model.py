import logging
import torch
from typing import cast

from ...encoder import ReIDEncoder
from ...external import DEEP_PERSON_REID_ROOT, ensure_syspath
from .parameter import TorchReIDParameters

logger = logging.getLogger(__name__)


class TorchReID(ReIDEncoder[TorchReIDParameters]):
    """
    Vendored Torchreid: build_model + load_pretrained_weights (sys.path adjusted).

    Attributes:
    ----------
    Same as ReIDEncoder; see TorchReIDParameters for model_name / num_classes / weights, etc.
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

        logger.info("%s from Torchreid is adopted as the ReID model.", model_name)
        ensure_syspath(DEEP_PERSON_REID_ROOT)

        from torchreid.models import build_model  # type: ignore
        from torchreid.utils import load_pretrained_weights  # type: ignore

        model: torch.nn.Module = cast(
            torch.nn.Module,
            build_model(
                name=model_name,
                num_classes=self.parameters.num_classes,
                loss="softmax",
                pretrained=False,
            ),
        )
        load_pretrained_weights(model, weights_path)
        return model

    def __str__(self) -> str:
        return f"TorchReID(model_name={self.parameters.model_name}, weights_path={self.parameters.weights_path})"


if __name__ == "__main__":
    from ...cli.feature_parsers import run_torchreid_feature_extract_main

    run_torchreid_feature_extract_main()
