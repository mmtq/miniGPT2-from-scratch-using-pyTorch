class GPT2Config:
    vocab_size = 50257  # GPT-2 vocabulary size
    max_position_embeddings = 512  # Maximum sequence length
    n_layer = 6  # Number of transformer blocks
    n_head = 8  # Number of attention heads
    n_embd = 512  # Embedding size
    dropout = 0.1  # Dropout rate

# class GPT2Config:
#     vocab_size = 50257
#     max_position_embeddings = 64   
#     n_layer = 2                    
#     n_head = 2                     
#     n_embd = 128                   
#     dropout = 0.0