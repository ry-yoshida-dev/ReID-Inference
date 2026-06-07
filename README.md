# ReID-Inference

## Overview

ReID-Inference is a Python project for **person re-identification (ReID)** inference: it loads checkpoints from several popular codebases, runs batched feature extraction on RGB images, and writes one NumPy file per image.

For package layout, backends, and module map, see [src/reid/README.md](src/reid/README.md).



## Installation

From the repository root (the directory containing `pyproject.toml`):

```bash
pip install .
```

For development, install in editable mode:

```bash
pip install -e ".[dev]"
```

Dependencies are declared in `pyproject.toml` and installed with the package. To install only libraries without the package (for example in a constrained environment), you can still use:

```bash
pip install -r requirements.txt
```

`torch_modules` is pulled from Git; ensure network access and credentials if that URL is private.

## Example

After `pip install .`, run a backend-specific feature extractor from any directory (see also [src/reid/model/torchreid/README.md](src/reid/model/torchreid/README.md), [src/reid/model/fastreid/README.md](src/reid/model/fastreid/README.md), [src/reid/model/transreid/README.md](src/reid/model/transreid/README.md)):

```bash
python3 -m reid.model.torchreid.model \
  --weights osnet_ain_ms_d_c.pth \
  --images-dir image \
  --output-dir result
```

From a checkout without installing the package, use `PYTHONPATH=src` instead:

```bash
cd /path/to/ReID-Inference
export PYTHONPATH=src
python3 -m reid.model.torchreid.model \
  --weights osnet_ain_ms_d_c.pth \
  --images-dir image \
  --output-dir result
```

Weights must match the architecture (`--model-name` and related flags differ per backend).

## NOTICE

ReID model codebases are consumed as **pip dependencies** (see [`pyproject.toml`](pyproject.toml)). The wrapper in `src/reid/` is under the root [`LICENSE`](LICENSE); upstream licenses apply to each dependency repository.

### Third-party software (pip dependencies)

| Component | Declared in | Upstream | License |
| --------- | ----------- | -------- | ------- |
| deep-person-reid (`torchreid`) | [`pyproject.toml`](pyproject.toml) | [ry-yoshida-dev/deep-person-reid](https://github.com/ry-yoshida-dev/deep-person-reid) | MIT |
| fast-reid (`fast_reid`) | [`pyproject.toml`](pyproject.toml) | [ry-yoshida-dev/fast-reid](https://github.com/ry-yoshida-dev/fast-reid) | Apache-2.0 |
| transreid-ssl (`transreid_pytorch`) | [`pyproject.toml`](pyproject.toml) | [ry-yoshida-dev/TransReID-SSL](https://github.com/ry-yoshida-dev/TransReID-SSL) | MIT |
| torch_modules | [`pyproject.toml`](pyproject.toml) | [ry-yoshida-private/TorchModules](https://github.com/ry-yoshida-private/TorchModules) | *(see upstream)* |