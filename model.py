import torch
import torch.nn as nn
import torchtext.vocab as vocab
import os
import time
import sys


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128,
                 output_dim=3, pretrained=False):
        """
        Initialize the RNN model for sentiment analysis.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings (default: 100)
            hidden_dim: Number of hidden units in RNN (default: 128)
            output_dim: Number of output classes (default: 3 for Positive/Negative/Neutral)
            pretrained: Whether to use pretrained GloVe embeddings (default: False)
        """
        super().__init__()
        self.pretrained = pretrained

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Initialize with pretrained GloVe embeddings if specified
        if pretrained:
            try:
                print("Loading GloVe embeddings...")

                # Set cache directory to current folder
                # This helps avoid permission issues and makes download location explicit
                cache_dir = os.path.join(os.getcwd(), '.vector_cache')
                os.makedirs(cache_dir, exist_ok=True)
                print(f"GloVe vectors will be stored in: {cache_dir}")

                # Show downloading progress
                print("This may take a few minutes the first time (downloading ~400MB)")
                print("Progress: ", end="", flush=True)

                start_time = time.time()

                # Download/load the vectors with progress indicator
                def progress_hook(t):
                    # Only show progress every 5 seconds to avoid console spam
                    elapsed = time.time() - progress_hook.last_time
                    if elapsed > 5:
                        progress_hook.last_time = time.time()
                        print(".", end="", flush=True)

                progress_hook.last_time = time.time()

                # Load GloVe with timeout
                glove = vocab.GloVe(name='6B', dim=embedding_dim, cache=cache_dir)

                elapsed = time.time() - start_time
                print(f"\nGloVe loaded in {elapsed:.2f} seconds")

                # Check if vocab_size exceeds GloVe vocabulary size
                if vocab_size > glove.vectors.shape[0]:
                    raise ValueError(f"vocab_size exceeds GloVe vocabulary size!")

                # Copy the first vocab_size vectors from GloVe
                self.embedding.weight.data.copy_(glove.vectors[:vocab_size])
                print(f"Successfully loaded pretrained GloVe embeddings.")

            except KeyboardInterrupt:
                print("\nGloVe download interrupted by user.")
                print("Falling back to randomly initialized embeddings.")

            except Exception as e:
                print(f"\nError loading GloVe embeddings: {e}")
                print("Falling back to randomly initialized embeddings.")

        # RNN layer for processing text and context
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        # Final classification layer (combining text and context)
        # Hidden states from text and context RNNs are concatenated
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text, context):
        """
        Forward pass through the RNN model.

        Args:
            text: Batch of tokenized texts [batch_size, seq_len]
            context: Batch of tokenized contexts [batch_size, seq_len]

        Returns:
            output: Sentiment predictions [batch_size, output_dim]
        """
        # Process text through embedding and RNN
        text_embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]
        _, text_hidden = self.rnn(text_embedded)
        # text_hidden shape: [1, batch_size, hidden_dim]
        text_hidden = text_hidden.squeeze(0)  # [batch_size, hidden_dim]

        # Process context through embedding and RNN
        context_embedded = self.embedding(context)  # [batch_size, seq_len, embedding_dim]
        _, context_hidden = self.rnn(context_embedded)
        # context_hidden shape: [1, batch_size, hidden_dim]
        context_hidden = context_hidden.squeeze(0)  # [batch_size, hidden_dim]

        # Concatenate the final hidden states from text and context
        combined = torch.cat((text_hidden, context_hidden), dim=1)
        # combined shape: [batch_size, hidden_dim*2]

        # Pass through final linear layer to get class predictions
        return self.fc(combined)  # [batch_size, output_dim]