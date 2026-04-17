import torch
import torch.nn as nn

class TransReIDBuilder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        print(f'[Info] Using Transformer_type: {cfg.transformer_type} as backbone')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


        backbone = self._import_backbone(cfg.transformer_type)

        self.model = backbone(
            img_size=cfg.input_size,
            camera=0,
            view=0,
            stride_size=cfg.stride,
        )

        self.model.load_param(cfg.weight_path, hw_ratio=cfg.hw_ratio)
        print(f'[Info] Loaded weights from {cfg.weight_path}')
        self.model.to(self.device)

        self.eval()

    def _import_backbone(self, transformer_type: str):
        match transformer_type:
            case 'vit_base_patch16_224_TransReID' | 'deit_base_patch16_224_TransReID':
                from model.backbones.vit_pytorch import vit_base_patch16_224_TransReID as backbone
            case 'vit_small_patch16_224_TransReID' | 'deit_small_patch16_224_TransReID':
                from model.backbones.vit_pytorch import vit_small_patch16_224_TransReID as backbone
            case 'swin_base_patch4_window7_224':
                from model.backbones.swin_transformer import swin_base_patch4_window7_224 as backbone
            case 'swin_small_patch4_window7_224':
                from model.backbones.swin_transformer import swin_small_patch4_window7_224 as backbone
            case _:
                raise ValueError(f'Unknown transformer_type: {transformer_type}')
        return backbone

    @torch.no_grad()
    def predict(self, x, cam_label=None, view_label=None):
        return self.model(x)

    def load_param(self, weight_path):
        param_dict = torch.load(weight_path, map_location='cpu')
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']

        for name in param_dict:
            if 'classifier' in name or 'fc' in name:
                continue
            try:
                self.state_dict()[name.replace('module.', '')].copy_(param_dict[name])
            except:
                raise ValueError(f'Failed to load parameter: {name}')
        print(f'Loaded pretrained model from {weight_path} (feature extraction only)')