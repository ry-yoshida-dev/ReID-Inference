import logging
from pathlib import Path
from typing import cast

import torch

from ..encoder import ReIDEncoder
from ..external_paths import FAST_REID_IMPORT_ROOT, FAST_REID_ROOT, ensure_syspath
from ..parameters import ReIDParameters

ensure_syspath(FAST_REID_IMPORT_ROOT)
from fast_reid.fastreid.config import get_cfg # type: ignore
from fast_reid.fastreid.modeling.meta_arch import build_model # type: ignore

logger = logging.getLogger(__name__)

def setup_cfg( # type: ignore
    config_file: str, 
    opts: list[str], 
    force_cpu: bool
    ):
    cfg = get_cfg() # type: ignore
    cfg.merge_from_file(config_file) # type: ignore
    cfg.merge_from_list(opts) # type: ignore
    cfg.MODEL.BACKBONE.PRETRAIN = False # type: ignore
    if force_cpu:
        cfg.MODEL.DEVICE = "cpu" # type: ignore
    cfg.freeze() # type: ignore
    return cfg # type: ignore


class FastReID(ReIDEncoder[ReIDParameters]):
    """
    FastReid
    """
    
    def _load_model(self) -> torch.nn.Module:
        from fast_reid.fastreid.utils.checkpoint import Checkpointer # type: ignore

        model_name = self.parameters.model_name
        weights_path = self.parameters.weights_path

        logger.info("%s from fast-reid is adopted as the ReID model.", model_name)
        
        config_file = str(FAST_REID_ROOT / "configs" / "MOT17" / f"{model_name}.yml")  # TODO
        if not Path(config_file).is_file():
            raise FileNotFoundError(f"config_file not found in {config_file}. {model_name} might not be a supported model for FastReID.")
        cfg = setup_cfg( # type: ignore
            config_file,
            ["MODEL.WEIGHTS", weights_path],
            force_cpu=not torch.cuda.is_available(),
            )

        model: torch.nn.Module = cast(torch.nn.Module, build_model(cfg)) # type: ignore
        Checkpointer(model).load(weights_path) # type: ignore

        return model # type: ignore