import pickle
import torch
import torch.nn as nn
import torch.optim as optim

print("Loading dataset...")
# Load dataset
with open("dataset.pkl", "rb") as f:
    data = pickle.load(f)

dataset = data["dataset"]
vocab_size = data["vocab_size"]
word2idx = data["word2idx"]
idx2word = data["idx2word"]
print(f"Dataset loaded with {len(dataset)} entries")

print(dataset[0])
print(dataset[1])
print(dataset[2])
print(dataset[3])

# Train model