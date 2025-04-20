# Read parquet file and loop through each entry
# Remove punctuation and special characters
# Remove entries with less than 3 words
# Remove entries with numbers
# Create a dataset and persist to disk

import pandas as pd
import re
import random
import string
import pickle
from collections import Counter
from pathlib import Path

# Config
INPUT_PARQUET = "text_messages/data/train-00001-of-00002-889c5bcac2961f1b.parquet"
TEXT_COLUMN = "text"
OUTPUT_DATASET = "dataset.pkl"
CONTEXT_SIZE = 3

# Make a dataset of 10% for training
DROP_RATE = 0.9

# Load
df = pd.read_parquet(INPUT_PARQUET)
print("Parquet file loaded")
sentences = df[TEXT_COLUMN].dropna().astype(str).tolist()
print("Sentences extracted")

# Data cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = re.sub(r"\s+", " ", text).strip()  # clean up extra spaces
    return text

cleaned_sentences = []
for sentence in sentences:
    cleaned = clean_text(sentence)
    if len(cleaned.split()) >= 3:
        if random.random() > DROP_RATE:
          cleaned_sentences.append(cleaned)
print(f"Cleaned sentences: {len(cleaned_sentences)}")

# Vocabulary creation
words = " ".join(cleaned_sentences).split()
word_counts = Counter(words)
vocab = sorted(word_counts)
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# Dataset creation
def make_dataset(sentences, word2idx, context_size=3):
    data = []
    i = 0
    for sentence in sentences:
        i += 1
        if i % 50000 == 0:
            print(f"Processing sentence {i}/{len(sentences)}")

        tokens = sentence.split()
        if len(tokens) < context_size + 1:
            continue
        for i in range(len(tokens) - context_size):
            context = tokens[i:i+context_size]
            target = tokens[i+context_size]
            try:
                context_idx = [word2idx[w] for w in context]
                target_idx = word2idx[target]
                data.append((context_idx, target_idx))
            except KeyError:
                continue  # skip if any word is not in vocab
    return data

dataset = make_dataset(cleaned_sentences, word2idx, CONTEXT_SIZE)
print("Dataset created")

# Save to disk
output = {
    "dataset": dataset,
    "word2idx": word2idx,
    "idx2word": idx2word,
    "vocab_size": vocab_size
}

with open(OUTPUT_DATASET, "wb") as f:
    pickle.dump(output, f)

print(f"Saved dataset with {len(dataset)} entries to '{OUTPUT_DATASET}'")
