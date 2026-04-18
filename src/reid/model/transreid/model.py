from __future__ import annotations

import logging
from typing import Callable, Protocol, cast

import torch

from .parameter import TransReIDParameters
from ...encoder import ReIDEncoder

logger = logging.getLogger(__name__)


class _VitLikeLoadParam(Protocol):
    """Vendored ViT/DeiT TransReID backbones (vit_pytorch)."""

    def load_param(self, model_path: str, hw_ratio: float, /) -> None: ...


class _SwinLoadParam(Protocol):
    """Vendored Swin backbones: load_param takes checkpoint path only."""

    def load_param(self, model_path: str, /) -> None: ...


class TransReID(ReIDEncoder[TransReIDParameters]):
    """
    Vendored TransReID-SSL backbone + load_param (hw_ratio for pos-embed resize).

    Attributes:
    ----------
    Same as ReIDEncoder; see TransReIDParameters and _import_backbone for model_name keys.
    """

    def _load_model(self) -> torch.nn.Module:
        """
        Returns:
        -------
        torch.nn.Module
            Backbone after load_param.
        """
        backbone = self._import_backbone(self.parameters.model_name)
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

    def _import_backbone(self, model_name: str) -> Callable[..., torch.nn.Module]:
        """
        Parameters:
        ----------
        model_name : str
            ViT/DeiT/Swin key (see match cases below).

        Returns:
        -------
        Callable[..., torch.nn.Module]
            Called with img_size, camera, view, stride_size.

        Raises
        ------
        ValueError
            Unknown model_name.
        """
        from ...external import TRANSREID_PYTORCH_ROOT, ensure_syspath

        ensure_syspath(TRANSREID_PYTORCH_ROOT)

        match model_name:
            case 'vit_base_patch16_224_TransReID' | 'deit_base_patch16_224_TransReID':
                from model.backbones.vit_pytorch import vit_base_patch16_224_TransReID as backbone # type: ignore
            case 'vit_small_patch16_224_TransReID' | 'deit_small_patch16_224_TransReID':
                from model.backbones.vit_pytorch import vit_small_patch16_224_TransReID as backbone # type: ignore
            case 'swin_base_patch4_window7_224':
                from model.backbones.swin_transformer import swin_base_patch4_window7_224 as backbone # type: ignore
            case 'swin_small_patch4_window7_224':
                from model.backbones.swin_transformer import swin_small_patch4_window7_224 as backbone # type: ignore
            case _:
                raise ValueError(f'Unknown model_name: {model_name}')
        return backbone # type: ignore


if __name__ == "__main__":
    from ...cli.feature_parsers import run_transreid_feature_extract_main

    run_transreid_feature_extract_main()
