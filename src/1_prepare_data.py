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
OUTPUT_DATASET_META = "dataset_meta.pkl"
OUTPUT_VOCAB = "vocab.pkl"
CONTEXT_SIZE = 3

# Make a dataset of 10% for training
DROP_RATE = 0

# Load
df = pd.read_parquet(INPUT_PARQUET)
print("Parquet file loaded")
sentences = df[TEXT_COLUMN].dropna().astype(str).tolist()
print("Sentences extracted")

# Data cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)  # TODO: instead of removing just numbers, remove the entry entirely
    text = re.sub(r"\s+", " ", text).strip()  # clean up extra spaces
    return text

VOCAB_LIMIT = 10000
UNK_TOKEN = "<UNK>"

cleaned_sentences = []
for sentence in sentences:
    cleaned = clean_text(sentence)
    if len(cleaned.split()) >= 3:
        if random.random() > DROP_RATE:
            cleaned_sentences.append(cleaned)

print(f"Cleaned sentences: {len(cleaned_sentences)}")

words = " ".join(cleaned_sentences).split()
word_counts = Counter(words)

most_common_words = word_counts.most_common(VOCAB_LIMIT)
vocab = [word for word, _ in most_common_words]

vocab.append(UNK_TOKEN)

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
                context_idx = [word2idx.get(w, word2idx[UNK_TOKEN]) for w in context]
                target_idx = word2idx[target]
                data.append({ "x": context_idx, "y": target_idx })
            except KeyError:
                continue  # skip if any word is not in vocab
    return data

dataset = make_dataset(cleaned_sentences, word2idx, CONTEXT_SIZE)
print("Dataset created")

# Save to disk
dataset_meta = {
    "word2idx": word2idx,
    "idx2word": idx2word,
    "vocab_size": vocab_size
}

with open(OUTPUT_DATASET_META, "wb") as f:
    pickle.dump(dataset_meta, f)
print(f"Saved dataset meta to '{OUTPUT_DATASET_META}'")

df = pd.DataFrame(dataset)
df.to_parquet("dataset.parquet", index=False)
print(f"Saved dataset")

with open(OUTPUT_VOCAB, "wb") as f:
    pickle.dump((word2idx, idx2word, vocab_size), f)
print(f"Saved vocab of length {vocab_size} to '{OUTPUT_VOCAB}'")
