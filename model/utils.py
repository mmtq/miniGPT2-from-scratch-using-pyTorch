# Provide utility functions, such as generating causal masks.

import torch

def causal_mask(T, idx):
    # Create a lower triangular matrix to mask future tokens
    return torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)
