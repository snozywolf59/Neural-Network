import torch
import torch.nn as nn
class Norm(nn.Module):
    def __init__(self, d_in, eps=1e-6):
        self.a = nn.Parameter(torch.ones(d_in))
        self.b = nn.Parameter(torch.zeros(d_in))
        self.eps = eps
    
    def forward(self, x: torch.Tensor):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b