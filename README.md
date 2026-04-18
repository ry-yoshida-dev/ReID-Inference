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
  --weights weights/osnet_ain_ms_d_c.pth \
  --images-dir image \
  --output-dir result
```

From a checkout without installing the package, use `PYTHONPATH=src` instead:

```bash
cd /path/to/ReID-Inference
export PYTHONPATH=src
python3 -m reid.model.torchreid.model \
  --weights weights/osnet_ain_ms_d_c.pth \
  --images-dir image \
  --output-dir result
```

Weights must match the architecture (`--model-name` and related flags differ per backend).

## NOTICE

This repository includes **vendored copies** of third-party open-source projects under [`src/reid/external/`](src/reid/external/). Those trees are **not necessarily under the same license as** the wrapper code in `src/reid/` (see the root [`LICENSE`](LICENSE) for this project’s own files). Original copyright and license texts are kept **in place** next to the corresponding code (typically files named `LICENSE`).

The table below summarizes vendored components and the license identified in the copy shipped here. If a row points to a subtree, open the `LICENSE` file in that path for the full terms.

### Third-party software (vendored)

| Component | Path in this repository | Upstream | License (see `LICENSE` in path) |
| --------- | ------------------------- | -------- | -------------------------------- |
| deep-person-reid | [`src/reid/external/deep-person-reid/`](src/reid/external/deep-person-reid/) | [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) | MIT |
| fast-reid | [`src/reid/external/fast_reid/`](src/reid/external/fast_reid/) | [JDAI-CV/fast-reid](https://github.com/JDAI-CV/fast-reid) | Apache-2.0 |
| TransReID-SSL | [`src/reid/external/TransReID-SSL/`](src/reid/external/TransReID-SSL/) | [damo-cv/TransReID-SSL](https://github.com/damo-cv/TransReID-SSL) | MIT |
| transreid_pytorch (bundled) | [`src/reid/external/TransReID-SSL/transreid_pytorch/`](src/reid/external/TransReID-SSL/transreid_pytorch/) | Vendored with TransReID-SSL | MIT |
| DINO (bundled) | [`src/reid/external/TransReID-SSL/dino/`](src/reid/external/TransReID-SSL/dino/) | [facebookresearch/dino](https://github.com/facebookresearch/dino) | Apache-2.0 |
| cluster-contrast-reid (bundled) | [`src/reid/external/TransReID-SSL/cluster-contrast-reid/`](src/reid/external/TransReID-SSL/cluster-contrast-reid/) | [alibaba/cluster-contrast-reid](https://github.com/alibaba/cluster-contrast-reid) | MIT |
| cnpy (bundled) | [`src/reid/external/fast_reid/projects/FastRT/third_party/cnpy/`](src/reid/external/fast_reid/projects/FastRT/third_party/cnpy/) | [rogersce/cnpy](https://github.com/rogersce/cnpy) | MIT |