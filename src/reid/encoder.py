import torch
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic
from torch.utils.data import DataLoader

from .parameters import ReIDParametersT
from torch_modules import (
    TorchTensorDataset,
    ImageDataset,
    ImagesDataset,
    TorchPreprocessor,
    TorchInferenceManager
    )


@dataclass
class ReIDEncoder(ABC, Generic[ReIDParametersT]):
    """
    Re-identification encoder: preprocesses inputs and runs inference.

    Attributes:
    ----------
    preprocessor: TorchPreprocessor
        The preprocessor for the ReIDEncoder.
    parameters: ReIDParametersT
        The parameters for the ReIDEncoder (ReIDParameters or a subclass).
    torch_inference_manager: TorchInferenceManager
        The torch inference manager for the ReIDEncoder.
    model: torch.nn.Module
        The model for the ReIDEncoder.
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
        xyxy_bboxes: np.ndarray
        ) -> np.ndarray:
        """
        Predict the output of the model from the cropped image.

        Parameters:
        ----------
        image: Image.Image
            The cropped image to predict the output of the model.
        xyxy_bboxes: np.ndarray
            The bounding boxes of the objects with (x1, y1, x2, y2) format in the image.

        Returns:
        -------
        np.ndarray: The output tensor of the model.
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
        images: list[Image.Image]
        ) -> np.ndarray:
        """
        Predict the output of the model from the list of images.
        """
        dataset: TorchTensorDataset = ImagesDataset(
            images=images,
            preprocessor=self.preprocessor
            )
        return self._predict_from_dataset(dataset=dataset)

    @torch.no_grad()
    def _predict_from_dataset(
        self,
        dataset: TorchTensorDataset
        ) -> np.ndarray:
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
        np.ndarray: The output tensor of the model.
            The shape is (N, D), where N is the number of samples in the dataset and D is the dimension of the output.
        """
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False
            )

        outputs: list[torch.Tensor] = []
        for _, input in data_loader:
            input = self.torch_inference_manager.preprocess_input(input_=input)
            output = self.model(input)
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


