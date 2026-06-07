"""
SOLIDER 連携メモ（未実装）。

            case "SOLIDER":
                sys.path.append("SOLIDER-REID")
                from model import make_model
                from config import cfg
                config_file = os.path.join("SOLIDER-REID", "configs", "market", f"{self.ModelParams.model_name}.yml") #model_name -> swin_base
                cfg.defrost()
                cfg.merge_from_file(config_file)
                cfg.MODEL.SEMANTIC_WEIGH = 0.2

                model = make_model(cfg, num_class=1, camera_num=1, view_num=1, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
                model.load_param(model_path)

            case "TransReID":
                NotImplementedError
"""
