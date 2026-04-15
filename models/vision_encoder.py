import torchvision.models as models
import torch.nn as nn
import torch.hub
"""
return _vision_transformer(
        patch_size=16, patch 크기 = 16 x 16
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=weights,
        progress=progress,
        **kwargs,
"""

# 입력 dim: (B, 3, H, W)
# 출력 dim: (B, 768) - ViT-B/16의 hidden_dim
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
