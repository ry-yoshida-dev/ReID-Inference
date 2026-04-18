# fastreid

## Overview

This subtree integrates **[FastReID](https://github.com/JDAI-CV/fast-reid)** from the vendored `fast_reid` package: YACS config under `configs/MOT17/<model_name>.yml`, `build_model`, and `Checkpointer` for weights. `model_name` is the **YAML stem** (e.g. `sbs_S50` for `configs/MOT17/sbs_S50.yml`).

## CLI

From the repository root, with `src` on `PYTHONPATH`:

```bash
PYTHONPATH=src python3 -m reid.model.fastreid.model \
  --weights weights/mot17_sbs_S50.pth \
  --images-dir image \
  --output-dir result
```

### Common arguments

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--images-dir` | `image` | Directory of input images (top level only; sorted by name). |
| `--output-dir` | `result` | Output directory for `<stem>.npy` feature files. |
| `--weights` | *(required)* | Checkpoint path for `Checkpointer.load`. |
| `--batch-size` | `16` | DataLoader batch size. |
| `--no-fp16` | off | Disable FP16 on GPU. |

### FastReID-specific arguments

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--model-name` | `sbs_S50` | Config stem under vendored `fast_reid/configs/MOT17/`. |

## Sample code (Python API)

Run with the package importable (for example `PYTHONPATH=src` from the repo root, or an editable install). Paths are examples; point `weights_path` at your checkpoint.

[`ReIDBackend`](../../backends.py) picks the parameter dataclass, encoder class, and default preprocessor; swap the enum member to try another backend (adjust backend-specific kwargs accordingly).

```python
from pathlib import Path

import numpy as np
from PIL import Image

from reid.backends import ReIDBackend

backend = ReIDBackend.FASTREID
params = backend.parameter_class(
    weights_path=str(Path("weights/mot17_sbs_S50.pth")),
    batch_size=16,
    is_half_precision_enabled=True,
    backend=backend,
    model_name="sbs_S50",
)
encoder = backend.encoder_class(
    preprocessor=backend.build_default_preprocessor(),
    parameters=params,
)

# One row per full image (N, D)
images = [Image.open("image/person.jpg").convert("RGB")]
features = encoder.predict_from_images(images)
print(features.shape)

# Crops from xyxy boxes on one image (same pixel coords as the PIL image)
xyxy = np.array([[10.0, 20.0, 200.0, 400.0]], dtype=np.float32)
crop_features = encoder.predict_from_image(images[0], xyxy)
print(crop_features.shape)
```

## Components

| Component | Description |
| --------- | ----------- |
| [`model.py`](model.py) | `FastReID` encoder, YACS `setup_cfg`, and `python -m` CLI entry. |
| [`preprocess.py`](preprocess.py) | Default `TorchPreprocessor` builder for this backend. |
