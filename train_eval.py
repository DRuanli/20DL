from data import train_loader, test_loader, vocab_size
from model import RNNModel
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import json
import os


def train_and_evaluate(model, train_loader, test_loader, epochs=200, lr=0.001):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        for text, context, labels in train_loader:
            # Move data to device
            text, context, labels = text.to(device), context.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            preds = model(text, context)

            # Calculate loss
            loss = criterion(preds, labels)
            total_loss += loss.item()
            batch_count += 1

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for text, context, labels in test_loader:
            # Move data to device
            text, context, labels = text.to(device), context.to(device), labels.to(device)

            # Forward pass
            logits = model(text, context)
            preds = logits.argmax(dim=1)

            # Move to CPU for metric calculation
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Test Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
    return acc, f1


# Run experiments for both pretrained and from-scratch embeddings
results = {}

# Check if GloVe embeddings exist
glove_available = os.path.exists('glove.6B.100d.txt')

# Experiment 1: From-scratch embeddings (always run this)
print("\n=== Training with Embeddings from Scratch ===")
model_scratch = RNNModel(
    vocab_size=vocab_size,
    embedding_dim=100,
    hidden_dim=128,
    output_dim=3,
    pretrained=False
)
key = "RNN_Pretrained=False"
acc, f1 = train_and_evaluate(model_scratch, train_loader, test_loader)
results[key] = {"Accuracy": float(acc), "F1-score": float(f1)}
print(f"{key} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

# Experiment 2: Pretrained embeddings (if available)
print("\n=== Training with Pretrained Embeddings ===")
if glove_available:
    model_pretrained = RNNModel(
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_dim=128,
        output_dim=3,
        pretrained=True
    )
    key = "RNN_Pretrained=True"
    acc, f1 = train_and_evaluate(model_pretrained, train_loader, test_loader)
    results[key] = {"Accuracy": float(acc), "F1-score": float(f1)}
    print(f"{key} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
else:
    print("GloVe embeddings not found. To use pretrained embeddings:")
    print("1. Download GloVe from https://nlp.stanford.edu/data/glove.6B.zip")
    print("2. Extract and place glove.6B.100d.txt in your project directory")
    print("Adding placeholder results for pretrained model...")
    results["RNN_Pretrained=True"] = {"Accuracy": 0.0, "F1-score": 0.0,
                                      "Note": "GloVe embeddings not available"}

# Save results to JSON file
with open("results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to results.json")