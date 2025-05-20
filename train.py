import torch
from torch.utils.data import DataLoader
from config import GPT2Config
from model.gpt2 import miniGPT2
from tokenizer import tokenizer
from dataset import GPT2Dataset
from losses import gpt2_loss
import torch.optim as optim

# Load and prepare data
texts = []

with open('./texts/facts.txt', 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(texts)} non-empty lines.")

dataset = GPT2Dataset(tokenizer, texts, block_size=32)
print(f"Number of training samples: {len(dataset)}")

loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = GPT2Config()
model = miniGPT2(config).to(device)

# Load existing checkpoint if available
checkpoint_path = "./minigpt2.pth"
try:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Checkpoint loaded. Continuing training...")
except FileNotFoundError:
    print("No checkpoint found. Starting from scratch.")
optimizer = optim.Adam(model.parameters(), lr=3e-4)

#Training loop
for epoch in range(10):
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = gpt2_loss(logits, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "minigpt2.pth")