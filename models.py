from torch import nn
from typing import List
import math

class FCNN(nn.Module):

    def __init__(self, activation: str, hidden_sizes: List[int], input_dim: int, num_classes: int, init_width: float = 1.0):

        super().__init__()
        self.init_width = init_width
        
        act = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}[activation]
        layers: List[nn.Module] = [nn.Flatten()]
        in_size = input_dim
        
        for hs in hidden_sizes:
            linear = nn.Linear(in_size, hs)
            self._init_linear(linear)
            layers.append(linear)
            layers.append(act)
            in_size = hs
            
        output_layer = nn.Linear(in_size, num_classes)
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

class CNN(nn.Module):
    def __init__(self, dataset: str, widths: List[int], fc_sizes: List[int], activation: str = 'relu', init_width: float = 1.0):
        super().__init__()
        self.init_width = init_width
        
        channels = {'mnist': 1, 'cifar10': 3, 'cifar100': 3}[dataset]
        num_classes = {'mnist': 10, 'cifar10': 10, 'cifar100': 100}[dataset]
        size = {'mnist': 28, 'cifar10': 32, 'cifar100': 32}[dataset]
        
        act = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}[activation]
        
        modules = []
        for i, width in enumerate(widths):
            prev_width = widths[i-1] if i > 0 else channels
            conv = nn.Conv2d(prev_width, width, kernel_size=5, padding=2)
            self._init_conv(conv)
            modules.extend([conv, act, nn.AvgPool2d(2)])
            size //= 2
            
        modules.append(nn.Flatten())
        in_features = widths[-1] * size * size
        for fc_size in fc_sizes:
            linear = nn.Linear(in_features, fc_size)
            self._init_linear(linear)
            modules.extend([linear, act])
            in_features = fc_size
            
        output_layer = nn.Linear(in_features, num_classes)
        self._init_linear(output_layer)
        modules.append(output_layer)
        
        self.model = nn.Sequential(*modules)
    
    def _init_conv(self, layer):
        fan_in = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
        bound = self.init_width / math.sqrt(fan_in)
        layer.weight.data.uniform_(-bound, bound)
        if layer.bias is not None:
            layer.bias.data.uniform_(-bound, bound)
            
    def _init_linear(self, layer):
        fan_in = layer.in_features
        bound = self.init_width / math.sqrt(fan_in)
        layer.weight.data.uniform_(-bound, bound)
        if layer.bias is not None:
            layer.bias.data.uniform_(-bound, bound)
    
    def forward(self, x):
        return self.model(x)