import torch
import torch.nn as nn
import torch.optim as optim

class LSTMWordPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        print(f"Initializing model with vocab_size={vocab_size}, embedding_dim={embedding_dim}, hidden_dim={hidden_dim}")
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        out = self.fc(out[:, -1, :])  # Use output from last timestep
        return out