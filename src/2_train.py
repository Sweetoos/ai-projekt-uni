import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from lib.model import LSTMWordPredictor
import os, psutil
from lib.constants import embedding_dim, hidden_dim

def mem():
    print(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} MB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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

class WordDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

batch_size = 50000
dataloader = DataLoader(WordDataset(dataset), batch_size=batch_size, shuffle=True)

# Train model
model = LSTMWordPredictor(vocab_size, embedding_dim, hidden_dim).to(device)
print(f"Model created")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
mem()

n_epochs = 10
for epoch in range(n_epochs):
    print(f"Epoch {epoch+1}/{n_epochs}")
    mem()
    epoch_loss = 0.0
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}/{len(dataloader)}")
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)

    avg_loss = epoch_loss / len(dataset)
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Persist
print("Saving model...")
torch.save(model.state_dict(), "lstm_word_predictor.pth")