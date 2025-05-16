import torch
import torch.nn as nn
from torch.nn import functional as F

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super.__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        
        self.head_dim = config.n_embd // config.n_head
        self.n_head = config.n_head
        
        # Linear layers for query, key, and value projections
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=True)
        
        # Output Projection
        
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, mask=None):
        B,T,C = x.size()
        
        # Comput Q,K,V matrices
        Q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)
        K = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)
        V = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)
        
        # Compute attention scores
        attn_scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask==0, float('-inf'))
            
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = (attn_weights @ V).transpose(1,2).contiguous.view(B, T, C)
        
        return self.proj(attn_output)
        
