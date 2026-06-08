# reid

## Overview

This package implements **ReID encoders** on top of [TorchModules](https://github.com/ry-yoshida-private/TorchModules): preprocessing, optional half precision, and batched inference. Three backends are wired in as pip packages—**Torchreid** ([`deep-person-reid`](https://github.com/ry-yoshida-dev/deep-person-reid), `import torchreid`), **FastReID** ([`fast-reid`](https://github.com/ry-yoshida-dev/fast-reid), `fast_reid.fastreid`), and **TransReID** ([`transreid-ssl`](https://github.com/ry-yoshida-dev/TransReID-SSL), `transreid_pytorch`).

## Public API

Import the stable surface from the package root; encoder classes load lazily to avoid pulling every backend at import time.

```python
from reid import ReIDBackend, ReIDEncoder, ReIDParameters, TorchReID, TorchReIDParameters
```

| Symbol | Role |
| ------ | ---- |
| `ReIDBackend` | Backend enum (`TORCHREID`, `FASTREID`, `TRANSREID`). |
| `ReIDParameters` | Shared parameter dataclass; FastReID uses this type directly. |
| `ReIDParametersT` | TypeVar bound for `ReIDEncoder[TP]`. |
| `ReIDEncoder` | Abstract encoder base class. |
| `TorchReID`, `FastReID`, `TransReID` | Concrete encoder implementations. |
| `TorchReIDParameters`, `TransReIDParameters` | Backend-specific parameter subclasses. |

Subpackages (`reid.backends`, `reid.encoder`, `reid.model`, `reid.parameters`) remain available for deeper imports.

## Backends

| Backend    | Encoder class | Typical use |
| ---------- | ------------- | ----------- |
| **torchreid** | [`TorchReID`](model/torchreid/model.py) | OSNet and other models from `torchreid.models.build_model`. |
| **fast_reid** | [`FastReID`](model/fastreid/model.py) | MOT17-style YAML configs under vendored `fast_reid/configs/MOT17/`. |
| **transreid** | [`TransReID`](model/transreid/model.py) | ViT / Swin TransReID backbones from vendored `transreid_pytorch`. |

Library selection and defaults are exposed via [`ReIDBackend`](backends.py) (`TORCHREID`, `FASTREID`, `TRANSREID`).

## CLI feature extraction

Each backend’s `model` module is runnable as `__main__` and writes **one `.npy` per input image** (stem = image file stem). Common flags: `--images-dir`, `--output-dir`, `--weights` (required), `--batch-size`, `--no-fp16`. Backend-specific flags are documented under:

- [model/torchreid/README.md](model/torchreid/README.md)
- [model/fastreid/README.md](model/fastreid/README.md)
- [model/transreid/README.md](model/transreid/README.md)

## Components

| Component | Description |
| --------- | ----------- |
| [`encoder.py`](encoder.py) | Abstract `ReIDEncoder`: loads weights, wraps `TorchInferenceManager`, `predict_from_image(s)`. |
| [`parameters.py`](parameters.py) | Shared `ReIDParameters` (extends TorchModules `TorchParameters`). |
| [`backends.py`](backends.py) | `ReIDBackend` enum: encoder class, parameter class, default preprocessor builder. |
| [`cli/feature_extract.py`](cli/feature_extract.py) | Shared CLI helpers: image listing, batching, `.npy` output. |
| [`cli/feature_parsers.py`](cli/feature_parsers.py) | Argparse entrypoints for each backend’s feature script. |
| [model/torchreid/](model/torchreid/README.md) | Torchreid integration. |
| [model/fastreid/](model/fastreid/README.md) | FastReID integration. |
| [model/transreid/](model/transreid/README.md) | TransReID integration. |
