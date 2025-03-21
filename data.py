import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import nltk
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


def load_and_process_data(csv_path='sentiment_data.csv', max_len_text=50, max_len_context=20):
    """
    Load and process the sentiment data from CSV.

    Args:
        csv_path: Path to the CSV file
        max_len_text: Maximum length for text sequences
        max_len_context: Maximum length for context sequences

    Returns:
        vocab: Vocabulary dictionary mapping words to indices
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
    """
    print(f"Loading data from {csv_path}...")

    # Load the CSV file and drop any rows with missing values
    data = pd.read_csv(csv_path).dropna()
    print(f"Loaded {len(data)} samples")

    # Map sentiment labels to numerical values
    label_map = {'Positive': 0, 'Negative': 1, 'Neutral': 2}

    # Extract texts, contexts, and labels
    texts = data['text'].tolist()
    contexts = data['context'].tolist()
    labels = data['label'].map(label_map).tolist()

    # Tokenize all texts and contexts
    print("Tokenizing text and context...")
    tokenized_texts = [word_tokenize(t.lower()) for t in texts]
    tokenized_contexts = [word_tokenize(c.lower()) for c in contexts]

    # Build vocabulary from all words
    print("Building vocabulary...")
    all_words = []
    for tokens in tokenized_texts + tokenized_contexts:
        all_words.extend(tokens)

    # Count word frequencies
    word_counts = Counter(all_words)
    print(f"Total unique words: {len(word_counts)}")

    # Take most common words, leaving room for <PAD> and <UNK> tokens
    # We'll use the top ~5000 words (adjust if needed)
    vocab_size = min(5000, len(word_counts) + 2)
    most_common = word_counts.most_common(vocab_size - 2)

    # Create vocabulary mapping
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for i, (word, _) in enumerate(most_common, 2):
        vocab[word] = i

    print(f"Final vocabulary size: {len(vocab)}")

    # Convert tokens to indices and pad/truncate to max length
    def to_indices(tokens, max_len):
        """Convert tokens to indices and pad/truncate to max_len"""
        indices = [vocab.get(token, 1) for token in tokens]  # Use <UNK> (index 1) for unknown words

        # Truncate if too long
        if len(indices) > max_len:
            indices = indices[:max_len]

        # Pad if too short
        elif len(indices) < max_len:
            indices += [0] * (max_len - len(indices))  # Use <PAD> (index 0) for padding

        return indices

    # Convert all texts and contexts to indices
    print("Converting tokens to indices...")
    text_indices = [to_indices(tokens, max_len_text) for tokens in tokenized_texts]
    context_indices = [to_indices(tokens, max_len_context) for tokens in tokenized_contexts]

    # Split data into training and testing sets (80% train, 20% test)
    print("Splitting into train and test sets...")
    X_text_train, X_text_test, X_context_train, X_context_test, y_train, y_test = train_test_split(
        text_indices, context_indices, labels, test_size=0.2, random_state=42
    )

    # Create datasets and dataloaders
    train_dataset = SentimentDataset(X_text_train, X_context_train, y_train)
    test_dataset = SentimentDataset(X_text_test, X_context_test, y_test)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

    # Create DataLoaders with batch size 32
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return vocab, train_loader, test_loader


class SentimentDataset(Dataset):
    """
    Dataset class for sentiment analysis data.
    """

    def __init__(self, texts, contexts, labels):
        """
        Initialize the dataset.

        Args:
            texts: List of text sequences (as indices)
            contexts: List of context sequences (as indices)
            labels: List of sentiment labels
        """
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.contexts = torch.tensor(contexts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.contexts[idx], self.labels[idx]


# Execute when run directly
if __name__ == "__main__":
    # Load and process the data
    vocab, train_loader, test_loader = load_and_process_data()

    # Print sample batch
    for texts, contexts, labels in train_loader:
        print(f"Sample batch shapes:")
        print(f"Texts: {texts.shape}")
        print(f"Contexts: {contexts.shape}")
        print(f"Labels: {labels.shape}")
        break

    print("Data processing complete!")