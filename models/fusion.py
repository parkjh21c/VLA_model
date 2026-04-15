import torch
import torch.nn as nn

class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, vision_feat1, vision_feat2, lang_feat, state_feat):
        # 단순 concatenation
        fused_feat = torch.cat([vision_feat1, vision_feat2, lang_feat, state_feat], dim=-1)
        return fused_feat