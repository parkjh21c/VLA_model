import torch
import torch.nn as nn

from models.vision_encoder import VisionEncoder
from models.language_encoder import LanguageEncoder
from models.fusion import Fusion

class Policy(nn.Module):
    def __init__(self, action_dim=6, cache_root=".cache"):
        super().__init__()

        # encoder
        self.vision_encoder = VisionEncoder(cache_dir=f"{cache_root}/torch")
        self.language_encoder = LanguageEncoder(cache_dir=f"{cache_root}/huggingface")

        # fusion
        self.fusion = Fusion()

        # feature dimension
        vision_dim = 768  # ViT-B/16
        lang_dim = 768  # DistilBERT
        fused_dim = vision_dim + lang_dim

        # action head (MLP)
        self.action_head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, image, text):
        # encoder
        vision_feat = self.vision_encoder(image)
        lang_feat = self.language_encoder(text)

        # fusion
        fused_feat = self.fusion(vision_feat, lang_feat)

        # action prediction
        action = self.action_head(fused_feat)

        return action
