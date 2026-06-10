# transreid

## Overview

This subtree integrates **TransReID** via the pip package [`transreid-ssl`](https://github.com/ry-yoshida-dev/TransReID-SSL) (`transreid_pytorch.model.backbones.get_backbone`). Weights are loaded with backbone-specific `load_param` (including positional embedding resize controlled by `hw_ratio`).

## CLI

From the repository root (after `pip install .`):

```bash
reid-extract transreid \
  --weights weights/vit_small_cfs_lup.pth \
  --images-dir image \
  --output-dir result
```

Without installing: `PYTHONPATH=src python3 main.py transreid ...` (same flags).

### Common arguments

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--images-dir` | `image` | Directory of input images (top level only; sorted by name). |
| `--output-dir` | `result` | Output directory for `<stem>.npy` feature files. |
| `--weights` | *(required)* | Checkpoint path passed to `load_param`. |
| `--batch-size` | `16` | DataLoader batch size. |
| `--no-fp16` | off | Disable FP16 on GPU. |

### TransReID-specific arguments

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--model-name` | `vit_small_patch16_224_TransReID` | Backbone key in `BACKBONE_FACTORY` (ViT/DeiT/Swin variants). |
| `--input-size` | `256,128` | Input `H,W` as two integers. |
| `--stride` | `16,16` | Stride pair. |
| `--hw-ratio` | `2.0` | Hint for positional embedding resize in `load_param`. |

## Sample code (Python API)

Run with the package importable (for example `PYTHONPATH=src` from the repo root, or an editable install). `input_size` / `stride` / `hw_ratio` should match how the weights were trained when applicable.

[`ReIDBackend`](../../backends.py) picks the parameter dataclass, encoder class, and default preprocessor.

```python
from pathlib import Path

import numpy as np
from PIL import Image

from reid.backends import ReIDBackend

backend = ReIDBackend.TRANSREID
params = backend.parameter_class(
    weights_path=str(Path("weights/vit_small_cfs_lup.pth")),
    batch_size=16,
    is_half_precision_enabled=True,
    backend=backend,
    model_name="vit_small_patch16_224_TransReID",
    input_size=[256, 128],
    stride=[16, 16],
    hw_ratio=2.0,
)
encoder = backend.encoder_class(
    preprocessor=backend.build_default_preprocessor(),
    parameters=params,
)

images = [Image.open("image/person.jpg").convert("RGB")]
features = encoder.predict_from_images(images)
print(features.shape)

xyxy = np.array([[10.0, 20.0, 200.0, 400.0]], dtype=np.float32)
crop_features = encoder.predict_from_image(images[0], xyxy)
print(crop_features.shape)
```

## Components

| Component | Description |
| --------- | ----------- |
| [`model.py`](model.py) | `TransReID` encoder. |
| [`parameter.py`](parameter.py) | `TransReIDParameters` (input size, stride, `hw_ratio`, etc.). |
| [`preprocess.py`](preprocess.py) | Default `TorchPreprocessor` builder for this backend. |
