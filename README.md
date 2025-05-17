# 🧠 MiniGPT-2: Transformer-based Language Model from Scratch

A minimalist implementation of the GPT-2 architecture built entirely from scratch using PyTorch. This project is a learning-oriented reimplementation based on the original **GPT-2** and **Attention Is All You Need** papers, with clean code and detailed comments.

---

## 📚 Research Papers

This implementation is based on the following foundational papers:

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners (GPT-2 by OpenAI, 2019)](https://openai.com/research/language-unsupervised)

---

## 📂 Project Structure

```bash
minigpt2/
├── config.py                  # Hyperparameter definitions
├── tokenizer.py               # Tokenizer (uses pre-trained GPT-2 tokenizer)
├── model/
│   ├── __init__.py
│   ├── attention.py           # Multi-head causal self-attention
│   ├── transformer_block.py   # Transformer block (LN + Attention + MLP)
│   ├── gpt2.py                # Full GPT-2 model
│   └── utils.py               # Causal mask utility
├── dataset.py                 # Dataset preparation class
├── train.py                   # Training script
├── generate.py                # Text generation from trained model
├── losses.py                  # Loss function (CrossEntropy)
└── README.md                  # You are here
