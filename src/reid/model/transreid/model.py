from __future__ import annotations

import logging
from typing import Callable, Protocol, cast

import torch

from transreid_pytorch.model.backbones import get_backbone

from .parameter import TransReIDParameters
from ...encoder import ReIDEncoder

logger = logging.getLogger(__name__)


class _VitLikeLoadParam(Protocol):
    """ViT/DeiT TransReID backbones (vit_pytorch)."""

    def load_param(self, model_path: str, hw_ratio: float, /) -> None: ...


class _SwinLoadParam(Protocol):
    """Swin backbones: load_param takes checkpoint path only."""

    def load_param(self, model_path: str, /) -> None: ...


class TransReID(ReIDEncoder[TransReIDParameters]):
    """
    TransReID via the packaged transreid-ssl distribution.

    Attributes:
    ----------
    Same as ReIDEncoder; see TransReIDParameters and get_backbone for model_name keys.
    """

    def _load_model(self) -> torch.nn.Module:
        """
        Returns:
        -------
        torch.nn.Module
            Backbone after load_param.
        """
        backbone = get_backbone(self.parameters.model_name)
        logger.info("Using model_name: %s as backbone", self.parameters.model_name)

        model: torch.nn.Module = backbone(
            img_size=self.parameters.input_size,
            camera=0,
            view=0,
            stride_size=self.parameters.stride,
        )

        weights_path = self.parameters.weights_path
        if self.parameters.model_name.startswith("swin_"):
            cast(_SwinLoadParam, model).load_param(weights_path)
        else:
            cast(_VitLikeLoadParam, model).load_param(
                weights_path, self.parameters.hw_ratio
            )
        logger.info("Loaded weights from %s", weights_path)
        return model


if __name__ == "__main__":
    from ...cli.feature_parsers import run_transreid_feature_extract_main

    run_transreid_feature_extract_main()
