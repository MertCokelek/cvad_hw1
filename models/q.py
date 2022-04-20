import torch.nn as nn


class MultiLayerQ(nn.Module):
    """Q network consisting of an MLP."""
    def __init__(self, config):
        super().__init__()
        self.silinecek_mlp = nn.Linear(10, 20)

    def forward(self, features, actions):
        return self.silinecek_mlp(features)
