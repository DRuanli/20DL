import torch
import torch.nn as nn
import numpy as np
import os


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128,
                 output_dim=3, pretrained=False):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Load pretrained GloVe embeddings if specified
        if pretrained:
            # We'll implement a custom GloVe loader instead of using torchtext
            self.load_glove_embeddings(vocab_size, embedding_dim)

        # RNN layers for text and context
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def load_glove_embeddings(self, vocab_size, embedding_dim):
        """
        Load GloVe embeddings into the model's embedding layer.
        If the GloVe file doesn't exist, will use random embeddings instead.
        """
        from data import vocab  # Import vocab from data.py

        # Path to GloVe embeddings - adjust as needed
        glove_path = f'glove.6B.{embedding_dim}d.txt'

        # If GloVe file doesn't exist, download it
        if not os.path.exists(glove_path):
            print(f"GloVe embeddings not found at {glove_path}.")
            print("Using random embeddings instead of GloVe.")
            print("To use GloVe embeddings, download them manually from:")
            print("https://nlp.stanford.edu/data/glove.6B.zip")
            print("Then extract and place the glove.6B.100d.txt file in your project directory.")
            return

        print(f"Loading GloVe embeddings from {glove_path}...")

        # Initialize embeddings matrix
        embeddings = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
        embeddings[0] = np.zeros((embedding_dim,))  # <PAD> token

        # Load GloVe vectors
        word_to_idx = {word: idx for idx, word in enumerate(vocab.keys())}

        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]

                # Only load if word is in our vocabulary
                if word in word_to_idx and word_to_idx[word] < vocab_size:
                    vector = np.array(values[1:], dtype='float32')
                    embeddings[word_to_idx[word]] = vector

        # Load into embedding layer
        self.embedding.weight.data.copy_(torch.FloatTensor(embeddings))
        print("GloVe embeddings loaded successfully!")

    def forward(self, text, context):
        # Get embeddings for text and context
        text_embed = self.embedding(text)
        context_embed = self.embedding(context)

        # Process text and context through RNN
        _, text_hidden = self.rnn(text_embed)
        _, context_hidden = self.rnn(context_embed)

        # Concatenate the final hidden states
        combined = torch.cat((text_hidden.squeeze(0), context_hidden.squeeze(0)), dim=1)

        # Pass through fully connected layer
        return self.fc(combined)