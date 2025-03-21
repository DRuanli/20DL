"""
Data processing for sentiment analysis - No NLTK required
This version uses basic string splitting instead of NLTK tokenization.
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
import os
import re


def simple_tokenize(text):
    """Simple tokenizer that doesn't require NLTK"""
    # Convert to lowercase
    text = text.lower()
    # Replace punctuation with spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Split on whitespace and filter out empty strings
    return [token for token in text.split() if token]


# Load the data
print("Loading sentiment data...")
if not os.path.exists('sentiment_data.csv'):
    print("sentiment_data.csv not found. Please run data_generator.py first.")
    exit(1)

data = pd.read_csv('sentiment_data.csv').dropna()
texts = data['text'].tolist()
contexts = data['context'].tolist()
labels = data['label'].map({'Positive': 0, 'Negative': 1, 'Neutral': 2}).tolist()

print(f"Loaded {len(texts)} samples with labels: {data['label'].value_counts().to_dict()}")

# Tokenize texts and contexts with our simple tokenizer
print("Tokenizing text and context...")
tokenized_texts = [simple_tokenize(t) for t in texts]
tokenized_contexts = [simple_tokenize(c) for c in contexts]

# Build vocabulary from all words in texts and contexts
all_words = [w for txt in (tokenized_texts + tokenized_contexts) for w in txt]
most_common = Counter(all_words).most_common(4998)  # Reserve 0 for <PAD> and 1 for <UNK>
vocab = {'<PAD>': 0, '<UNK>': 1}
for i, (w, _) in enumerate(most_common, 2):
    vocab[w] = i
vocab_size = len(vocab)

print(f"Vocabulary size: {vocab_size}")


def to_indices(tokens, max_len):
    """Convert tokens to indices with padding/truncation to max_len"""
    idxs = [vocab.get(t, 1) for t in tokens][:max_len]  # Truncate if needed
    # Pad if necessary
    if len(idxs) < max_len:
        idxs = idxs + [0] * (max_len - len(idxs))  # 0 is <PAD>
    return idxs


# Set maximum lengths based on requirements
max_len_text, max_len_context = 50, 20

# Convert tokenized texts and contexts to indices
print("Converting tokens to indices...")
text_indices = [to_indices(t, max_len_text) for t in tokenized_texts]
context_indices = [to_indices(c, max_len_context) for c in tokenized_contexts]

# Split data into train and test sets (80% train, 20% test)
print("Splitting data into train/test sets...")
train_texts, test_texts, train_contexts, test_contexts, train_labels, test_labels = train_test_split(
    text_indices, context_indices, labels, test_size=0.2, random_state=42
)

print(f"Training set: {len(train_texts)} samples")
print(f"Test set: {len(test_texts)} samples")


class SentimentDataset(Dataset):
    def __init__(self, texts, contexts, labels):
        self.texts = torch.tensor(texts)
        self.contexts = torch.tensor(contexts)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.contexts[idx], self.labels[idx]


# Create datasets
train_dataset = SentimentDataset(train_texts, train_contexts, train_labels)
test_dataset = SentimentDataset(test_texts, test_contexts, test_labels)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Data processing complete.")