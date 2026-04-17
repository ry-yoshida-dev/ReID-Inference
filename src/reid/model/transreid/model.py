import logging
import torch

from .parameter import TransReIDParameters
from ...encoder import ReIDEncoder
from torch_modules import TorchPreprocessor

logger = logging.getLogger(__name__)


class TransReID(ReIDEncoder[TransReIDParameters]):
    def __init__(
        self,
        preprocessor: TorchPreprocessor,
        parameters: TransReIDParameters,
        ):
        super().__init__(
            preprocessor=preprocessor,
            parameters=parameters
            )

    def _load_model(self) -> torch.nn.Module:
        backbone = self._import_backbone(self.parameters.model_name)
        logger.info("Using model_name: %s as backbone", self.parameters.model_name)

        model = backbone(
            img_size=self.parameters.input_size,
            camera=0,
            view=0,
            stride_size=self.parameters.stride,
        )

        model.load_param(self.parameters.weights_path, hw_ratio=self.parameters.hw_ratio)
        logger.info("Loaded weights from %s", self.parameters.weights_path)
        return model

    def _import_backbone(
        self,
        model_name: str
        ) -> torch.nn.Module:
        """
        Import the backbone of the model.

        Parameters:
        ----------
        model_name: str
            The name of the model.

        Returns:
        ----------
        torch.nn.Module:
            The backbone of the model.
        """
        import sys
        sys.path.append("src/external/TransReID-SSL/transreid_pytorch")

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
