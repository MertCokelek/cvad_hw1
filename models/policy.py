import torch.nn as nn


class MultiLayerPolicy(nn.Module):
    """An MLP based policy network"""
    def __init__(self):
        super().__init__()
        self.silinecek = nn.Linear(10, 20)

    def forward(self, features):
        return self.silinecek(features)
