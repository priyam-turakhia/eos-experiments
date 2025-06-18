from torch import nn
from typing import List
import math

class FCNN(nn.Module):

    def __init__(self, activation: str, hidden_sizes: List[int], init_width: float = 1.0):
        super().__init__()
        self.init_width = init_width
        
        act = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}[activation]
        layers: List[nn.Module] = [nn.Flatten()]
        in_size = 28 * 28
        
        for hs in hidden_sizes:
            linear = nn.Linear(in_size, hs)
            self._init_linear(linear)
            layers.append(linear)
            layers.append(act)
            in_size = hs
            
        output_layer = nn.Linear(in_size, 10)
        self._init_linear(output_layer)
        layers.append(output_layer)
        
        self.model = nn.Sequential(*layers)
    
    def _init_linear(self, layer):
        fan_in = layer.in_features
        bound = self.init_width / math.sqrt(fan_in)
        layer.weight.data.uniform_(-bound, bound)
        if layer.bias is not None:
            layer.bias.data.uniform_(-bound, bound)
        
    def forward(self, x):
        return self.model(x)