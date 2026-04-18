# external

## Overview

This directory holds **vendored copies** of upstream ReID projects. They are used for imports and reference implementations only; the thin integration layer lives in [`reid/model/`](../model/) and [`reid/external/external_paths.py`](external_paths.py) (canonical roots and `ensure_syspath`).

Prefer upgrading these trees deliberately (merge/cherry-pick) rather than editing them ad hoc, so diffs stay traceable to upstream.

## Components

| Component | Upstream | Description |
| --------- | -------- | ----------- |
| [deep-person-reid/](https://github.com/KaiyangZhou/deep-person-reid) | KaiyangZhou/deep-person-reid | Torchreid: `build_model`, pretrained weight loading. |
| [fast_reid/](https://github.com/JDAI-CV/fast-reid) | JDAI-CV/fast-reid | FastReID: YACS configs, `build_model`, checkpoints. |
| [TransReID-SSL/](https://github.com/damo-cv/TransReID-SSL) | damo-cv/TransReID-SSL | TransReID PyTorch backbones under `transreid_pytorch/`. |
