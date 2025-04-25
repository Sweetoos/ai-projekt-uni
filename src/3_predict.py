import pickle
import numpy as np
from lib.model import LSTMWordPredictor
from lib.constants import embedding_dim, hidden_dim
import torch

# Load vocab
word2idx, idx2word, vocab_size = None, None, None
with open("vocab.pkl", "rb") as f:
    word2idx, idx2word, vocab_size = pickle.load(f)

# Load model
model = LSTMWordPredictor(vocab_size, embedding_dim, hidden_dim)
model.load_state_dict(torch.load("lstm_word_predictor.pth"))
model.eval()

# Predict
def predict_next_word(model, context):
    model.eval()
    context_idxs = torch.tensor([[word2idx[w] for w in context]], dtype=torch.long)
    output = model(context_idxs)
    probs = torch.softmax(output, dim=1).detach().numpy()[0]
    predicted_idx = np.argmax(probs)
    return idx2word[predicted_idx]

context_size = 5
def predict_n_words(model, initial_context, n_words):
    model.eval()
    context = initial_context[:]

    for _ in range(n_words):
        # Get last context_size words
        context_input = context[-context_size:]
        input_tensor = torch.tensor([[word2idx[w] for w in context_input]], dtype=torch.long)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).numpy()[0]
            predicted_idx = np.argmax(probs)  # or use np.random.choice for sampling
            predicted_word = idx2word[predicted_idx]

        context.append(predicted_word)

    return context

test_context = ["we", "are", "going", "to"]
predicted = predict_next_word(model, test_context)
print(f"Input: {' '.join(test_context)} --> {predicted}")

predicted_few = predict_n_words(model, test_context, 3)
print(f"Input: {' '.join(test_context)} --> {' '.join(predicted_few)}")
