import torch
import torch.nn as nn

class State(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.state_head = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )
        
    
    def forward(self, x):
        return self.state_head(x)