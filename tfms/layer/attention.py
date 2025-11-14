import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, d_in, d_k, dropout):
        super.__init__()
        self.q_proj = nn.Linear(d_in, d_k)
        self.k_proj = nn.Linear(d_in, d_k)
        self.v_proj = nn.Linear(d_in, d_k)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        