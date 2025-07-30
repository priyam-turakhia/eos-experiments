import torch
from torch import nn

class MSELoss(nn.Module):
    def forward(self, input, target):
        return 0.5 * ((input - target) ** 2).sum()