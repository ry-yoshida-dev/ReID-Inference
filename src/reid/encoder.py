from __future__ import annotations

import torch
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic
from torch.utils.data import DataLoader

from .array_types import FloatArray, NumericArray

from .parameters import ReIDParametersT
from torch_modules import (
    TorchTensorDataset,
    ImageDataset,
    ImagesDataset,
    TorchPreprocessor,
    TorchInferenceManager
    )


@dataclass
class BaseReIDEncoder(ABC, Generic[ReIDParametersT]):
    """
    Abstract base class for ReID encoder backend implementations.

    Attributes:
    ----------
    preprocessor: TorchPreprocessor
        TorchModules dict; typical build: parameters.backend.build_default_preprocessor() or the chosen backend's preprocess.DEFAULT_PREPROCESSOR_CONFIG.
    parameters: ReIDParametersT
        The parameters for the encoder (ReIDParameters or a subclass).
    torch_inference_manager: TorchInferenceManager
        The torch inference manager for the encoder.
    model: torch.nn.Module
        The model for the encoder.
    """

    preprocessor: TorchPreprocessor
    parameters: ReIDParametersT
    torch_inference_manager: TorchInferenceManager = field(init=False, repr=False)
    model: torch.nn.Module = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.torch_inference_manager = TorchInferenceManager(
            is_half_precision_enabled=self.parameters.is_half_precision_enabled,
        )
        self.model = self._load_model()
        self.torch_inference_manager.setup_model(self.model)

    @abstractmethod
    def _load_model(self) -> torch.nn.Module:
        """
        Load the model.

        Returns:
        -------
        torch.nn.Module: The model.
        """

    def predict_from_image(
        self,
        image: Image.Image,
        xyxy_bboxes: NumericArray,
    ) -> FloatArray:
        """
        Run the model on crops defined by xyxy_bboxes in pixel coordinates.

        Parameters:
        ----------
        image: Image.Image
            Full RGB image the boxes refer to.
        xyxy_bboxes: NumericArray
            Bounding boxes (x1, y1, x2, y2) in the same coordinate system as image.

        Returns:
        -------
        FloatArray: Feature rows of shape (N, D) for N boxes, or empty when N is 0.
        """
        if len(xyxy_bboxes) == 0:
            return np.array([])
        dataset: TorchTensorDataset = ImageDataset(
            image=image,
            xyxy_bboxes=xyxy_bboxes,
            preprocessor=self.preprocessor
            )
        return self._predict_from_dataset(dataset=dataset)

    def predict_from_images(
        self,
        images: list[Image.Image],
    ) -> FloatArray:
        """
        One feature row per image (N, D).

        Parameters:
        ----------
        images : list[Image.Image]

        Returns:
        -------
        FloatArray
        """
        dataset: TorchTensorDataset = ImagesDataset(
            images=images,
            preprocessor=self.preprocessor
            )
        return self._predict_from_dataset(dataset=dataset)

    @torch.no_grad()
    def _predict_from_dataset(
        self,
        dataset: TorchTensorDataset,
    ) -> FloatArray:
        """
        Predict the output of the model from the dataset.

        WARNING:
        A large memory is required if the number of samples is large.

        Parameters:
        ----------
        dataset: TorchTensorDataset
            The dataset to predict from.

        Returns:
        -------
        FloatArray: The output tensor of the model.
            The shape is (N, D), where N is the number of samples in the dataset and D is the dimension of the output.
        """
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False
            )

        outputs: list[torch.Tensor] = []
        for _, batch in data_loader:
            batch = self.torch_inference_manager.preprocess_input(input_=batch)
            output = self.model(batch)
            outputs.append(output)
        combined = torch.cat(outputs, dim=0).detach().cpu().numpy()
        if self.parameters.is_half_precision_enabled and self.torch_inference_manager.is_gpu_available:
            combined = combined.astype(np.float16)
        return combined

    @property
    def batch_size(self) -> int:
        return self.parameters.batch_size

    @property
    def is_gpu_available(self) -> bool:
        return self.torch_inference_manager.is_gpu_available

