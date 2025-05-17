import torch.nn as nn

def gpt2_loss(logits, targets):
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))