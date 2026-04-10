import torchvision.models as models
import torch.nn as nn
import torch.hub


class VisionEncoder(nn.Module):
    def __init__(self, cache_dir=".cache/torch", pretrained=True):
        super().__init__()
        torch.hub.set_dir(cache_dir)
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        try:
            self.vit = models.vit_b_16(weights=weights)
        except Exception:
            self.vit = models.vit_b_16(weights=None)
        self.vit.heads = nn.Identity()  # head 분리 (classification layer 제거)

    def forward(self, x):
        return self.vit(x)
