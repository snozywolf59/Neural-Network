import torch
import torch.nn as nn
import copy

class Encoder(nn.Module):
    def __init__(self, layer: nn.Module, N):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        
    def forward(self,x ):
        for layer in self.layers:
            x = layer(x)
        return x