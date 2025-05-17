import torch
import torch.nn as nn
from .transformer import Transformer
from .utils import causal_mask

class miniGPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embd = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embd = nn.Embedding(config.max_position_embeddings, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([Transformer(config) for _ in range(config.n_layer) ])
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Output projection tied with token embeddings
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = self.token_embd.weight
        
    def forward(self, idx):
        B,T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        
        # Compute token and position embeddings
        tok_emb = self.token_embd(idx)
        pos_emb = self.pos_embd(pos)
        x = self.drop(tok_emb + pos_emb)
        
        mask = causal_mask(T, idx)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
            
        # Apply final layer normalization and output projection
        x = self.ln_f(x)
        
        return self.head(x)