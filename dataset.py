# Prepare the dataset for training.

import torch
from torch.utils.data import Dataset

class GPT2Dataset(Dataset):
    def __init__(self, tokenizer, texts, block_size):
        self.data = []
        for i, text in enumerate(texts):
            tokens = tokenizer.encode(text)
            if tokens is None:
                print(f"Warning: tokens is None for text at index {i}")
            else:
                self.data.append(tokens)
        
        print(f"Total samples: {len(self.data)}")
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = torch.tensor(self.data[index])
        return item[:-1], item[1:]
