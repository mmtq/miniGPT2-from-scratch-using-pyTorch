# Prepare the dataset for training.

import torch
from torch.utils.data import Dataset

class GPT2Dataset(Dataset):
    def __init__(self, tokenizer, texts, block_size):
        self.data = []
        for text in texts:
            tokens = tokenizer.encode(text)
            for i in range(0, len(tokens) - block_size):
                self.data.append(tokens[i:i+block_size+1])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = torch.tensor(self.data[index])
        return item[:-1], item[1:]