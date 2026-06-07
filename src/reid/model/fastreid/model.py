import logging
from pathlib import Path
from typing import cast

import torch

from fast_reid import CONFIG_ROOT
from fast_reid.fastreid.config import get_cfg
from fast_reid.fastreid.modeling.meta_arch import build_model
from fast_reid.fastreid.utils.checkpoint import Checkpointer

from ...encoder import ReIDEncoder
from ...parameters import ReIDParameters

logger = logging.getLogger(__name__)


def setup_cfg(
    config_file: str,
    opts: list[str],
    force_cpu: bool,
):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False
    if force_cpu:
        cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    return cfg


class FastReID(ReIDEncoder[ReIDParameters]):
    """
    FastReID via the packaged fast-reid distribution (import: fast_reid.fastreid).

    Attributes:
    ----------
    Same as ReIDEncoder; model_name is YAML stem under bundled configs/MOT17.
    """

    def _load_model(self) -> torch.nn.Module:
        """
        Returns:
        -------
        torch.nn.Module
            build_model(cfg) with Checkpointer-loaded weights.
        """
        model_name = self.parameters.model_name
        weights_path = self.parameters.weights_path

        logger.info("%s from packaged FastReID is adopted as the ReID model.", model_name)

        config_file = str(CONFIG_ROOT / "MOT17" / f"{model_name}.yml")
        if not Path(config_file).is_file():
            raise FileNotFoundError(
                f"config_file not found in {config_file}. "
                f"{model_name} might not be a supported model for FastReID."
            )
        cfg = setup_cfg(
            config_file,
            ["MODEL.WEIGHTS", weights_path],
            force_cpu=not torch.cuda.is_available(),
        )

        model: torch.nn.Module = cast(torch.nn.Module, build_model(cfg))
        Checkpointer(model).load(weights_path)

        return model


if __name__ == "__main__":
    from ...cli.feature_parsers import run_fastreid_feature_extract_main

    run_fastreid_feature_extract_main()
