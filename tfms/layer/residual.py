import torch
import torch.nn as nn

from norm import Norm

class ResidualLayer(nn.Module):
    def __init__(self, d_in, dropout):
        self.norm = Norm(d_in)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x, net):
        return x + self.dropout(self.norm(net(x)))
        
