import torch.nn as nn
from .attention import CasualSelfAttention

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        # FFN
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=True),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=True),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x, mask):
        # Apply attention with residual connection
        x = x + self.attn(self.ln1(x), mask)
        
        # Apply FFN with residual connection
        x = x + self.mlp(self.ln2(x))
        
        return x
        
        