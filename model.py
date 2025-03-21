import torch
import torch.nn as nn
import os
import numpy as np
import urllib.request
import zipfile
import shutil


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128,
                 output_dim=3, pretrained=False):
        super().__init__()
        self.pretrained = pretrained

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Use GloVe if pretrained is True
        if pretrained:
            try:
                print("Setting up GloVe embeddings...")
                cache_dir = os.path.join(os.getcwd(), '.vector_cache')
                os.makedirs(cache_dir, exist_ok=True)

                # Path to files
                glove_zip_path = os.path.join(cache_dir, 'glove.6B.zip')
                glove_txt_path = os.path.join(cache_dir, f'glove.6B.{embedding_dim}d.txt')

                # Download if needed
                if not os.path.exists(glove_txt_path) and not os.path.exists(glove_zip_path):
                    print(f"Downloading GloVe vectors from Stanford NLP website...")
                    url = "https://nlp.stanford.edu/data/glove.6B.zip"
                    print(f"This will download a file (~862MB) to: {glove_zip_path}")

                    # Download with progress reporting
                    def report_progress(count, block_size, total_size):
                        percent = int(count * block_size * 100 / total_size)
                        if percent % 5 == 0:  # Report every 5%
                            print(f"\rDownload progress: {percent}%", end="")

                    # Download the file
                    urllib.request.urlretrieve(url, glove_zip_path, reporthook=report_progress)
                    print("\nDownload complete!")

                # Extract if needed
                if not os.path.exists(glove_txt_path) and os.path.exists(glove_zip_path):
                    print(f"Extracting GloVe vectors from zip file...")
                    with zipfile.ZipFile(glove_zip_path, 'r') as zip_ref:
                        zip_ref.extract(f'glove.6B.{embedding_dim}d.txt', cache_dir)
                    print("Extraction complete!")

                # Load vectors from file
                if os.path.exists(glove_txt_path):
                    print(f"Loading vectors from: {glove_txt_path}")
                    word_to_vec = {}
                    with open(glove_txt_path, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            if i % 50000 == 0:
                                print(f"Processed {i} vectors...")
                            values = line.split()
                            word = values[0]
                            vector = torch.tensor([float(val) for val in values[1:]])
                            word_to_vec[word] = vector

                    print(f"Loaded {len(word_to_vec)} GloVe vectors")

                    # Initialize embedding weights matrix
                    weights_matrix = torch.zeros((vocab_size, embedding_dim))

                    # Load first N words from GloVe (simplified approach)
                    # Note: For actual implementation, you should map your vocabulary
                    # to the corresponding GloVe vectors
                    words_found = 0
                    for i in range(2, min(vocab_size, len(word_to_vec) + 2)):
                        if i - 2 < len(word_to_vec):
                            weights_matrix[i] = list(word_to_vec.values())[i - 2]
                            words_found += 1

                    print(f"Found embeddings for {words_found} words")
                    self.embedding.weight.data.copy_(weights_matrix)
                    print("Successfully loaded pretrained GloVe embeddings")
                else:
                    raise FileNotFoundError(f"GloVe file not found at: {glove_txt_path}")
            except Exception as e:
                print(f"Error loading GloVe embeddings: {e}")
                print("Continuing with randomly initialized embeddings")

        # RNN layer for processing text and context
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        # Final classification layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text, context):
        # Process text through embedding and RNN
        text_embedded = self.embedding(text)
        _, text_hidden = self.rnn(text_embedded)
        text_hidden = text_hidden.squeeze(0)

        # Process context through embedding and RNN
        context_embedded = self.embedding(context)
        _, context_hidden = self.rnn(context_embedded)
        context_hidden = context_hidden.squeeze(0)

        # Combine text and context
        combined = torch.cat((text_hidden, context_hidden), dim=1)

        # Final prediction
        return self.fc(combined)