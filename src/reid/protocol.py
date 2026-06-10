from __future__ import annotations

from typing import Protocol

import torch
from .array_types import FloatArray, NumericArray
from PIL import Image
from torch_modules import TorchInferenceManager, TorchPreprocessor

from .parameters import ReIDParameters


class ReIDEncoder(Protocol):
    """
    Public ReID encoder interface for inference and configuration access.

    Backend implementations subclass ``BaseReIDEncoder``. Callers that receive an
    encoder from ``ReIDBackend.encoder_class`` should annotate values with this
    protocol.
    """

    preprocessor: TorchPreprocessor

    @property
    def parameters(self) -> ReIDParameters: ...

    torch_inference_manager: TorchInferenceManager
    model: torch.nn.Module

    @property
    def batch_size(self) -> int: ...

    @property
    def is_gpu_available(self) -> bool: ...

    def predict_from_image(
        self,
        image: Image.Image,
        xyxy_bboxes: NumericArray,
    ) -> FloatArray: ...

    def predict_from_images(
        self,
        images: list[Image.Image],
    ) -> FloatArray: ...
