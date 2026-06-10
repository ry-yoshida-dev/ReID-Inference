# torchreid

## Overview

This subtree integrates **[Torchreid](https://kaiyangzhou.github.io/deep-person-reid/)** via the pip package [`deep-person-reid`](https://github.com/ry-yoshida-dev/deep-person-reid) (`import torchreid`): `build_model` + `load_pretrained_weights`. `num_classes` must be consistent with the checkpoint when the classifier head is present.

## CLI

From the repository root (after `pip install .`):

```bash
reid-extract torchreid \
  --weights osnet_ain_ms_d_c.pth \
  --images-dir image \
  --output-dir result
```

Without installing: `PYTHONPATH=src python3 main.py torchreid ...` (same flags).

### Common arguments

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--images-dir` | `image` | Directory of input images (top level only; sorted by name). |
| `--output-dir` | `result` | Output directory for `<stem>.npy` feature files. |
| `--weights` | *(required)* | Path to pretrained weights for `load_pretrained_weights`. |
| `--batch-size` | `16` | DataLoader batch size. |
| `--no-fp16` | off | Disable FP16 on GPU. |

### Torchreid-specific arguments

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--model-name` | `osnet_ain_x1_0` | `torchreid.models.build_model` name (must match checkpoint). |
| `--num-classes` | `2510` | Classifier width for `build_model` (match training when head is loaded). |

## Sample code (Python API)

Run with the package importable (for example `PYTHONPATH=src` from the repo root, or an editable install). `model_name` and `num_classes` must match the checkpoint when the classifier head is present.

[`ReIDBackend`](../../backends.py) picks the parameter dataclass, encoder class, and default preprocessor.

```python
from pathlib import Path

import numpy as np
from PIL import Image

from reid.backends import ReIDBackend

backend = ReIDBackend.TORCHREID
params = backend.parameter_class(
    weights_path=str(Path("osnet_ain_ms_d_c.pth")),
    batch_size=16,
    is_half_precision_enabled=True,
    backend=backend,
    model_name="osnet_ain_x1_0",
    num_classes=2510,
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
| [`model.py`](model.py) | `TorchReID` encoder. |
| [`parameter.py`](parameter.py) | `TorchReIDParameters` (`model_name`, `num_classes`, …). |
| [`preprocess.py`](preprocess.py) | Default `TorchPreprocessor` builder for this backend. |
