import torch
import torch.nn as nn

from models.vision_encoder import VisionEncoder
from models.language_encoder import LanguageEncoder
from models.fusion import Fusion
from models.state import State

class Policy(nn.Module):
    def __init__(self, action_dim=6, cache_root=".cache"):
        super().__init__()

        # encoder
        self.vision_encoder = VisionEncoder(cache_dir=f"{cache_root}/torch")
        self.language_encoder = LanguageEncoder(cache_dir=f"{cache_root}/huggingface")
        self.state_encoder = State(state_dim=6)

        # fusion
        self.fusion = Fusion()

        # feature dimension
        vision_dim = 768  # ViT-B/16
        lang_dim = 768  # DistilBERT
        state_dim = 512
        fused_dim = vision_dim + vision_dim + lang_dim + state_dim

        # action head (MLP)
        self.action_head = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
    
    def forward(self, image1, image2, text, state):
        # encoder
        vision_feat1 = self.vision_encoder(image1)
        vision_feat2 = self.vision_encoder(image2)
        lang_feat = self.language_encoder(text)
        state_feat = self.state_encoder(state)

        # fusion
        fused_feat = self.fusion(vision_feat1, vision_feat2, lang_feat, state_feat)

        # action prediction
        action = self.action_head(fused_feat)

        return action
