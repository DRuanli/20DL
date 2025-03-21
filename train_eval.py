import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import json
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

# Import data processing and model
from data import load_and_process_data
from model import RNNModel


def train_and_evaluate(model, train_loader, test_loader, epochs=10, lr=0.001, device="cpu"):
    """
    Train and evaluate the sentiment analysis model.

    Args:
        model: The RNNModel instance
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to run training on ('cpu' or 'cuda')

    Returns:
        accuracy: Test set accuracy
        f1: Test set F1 score
    """
    # Move model to device
    model = model.to(device)

    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training metrics tracking
    train_losses = []
    train_accuracies = []

    print(f"\nTraining model with {'pretrained' if model.pretrained else 'scratch'} embeddings...")
    print(f"Running on device: {device}")

    # Training loop
    start_time = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0
        epoch_preds = []
        epoch_labels = []

        for batch_idx, (texts, contexts, labels) in enumerate(train_loader):
            # Move batch to device
            texts, contexts, labels = texts.to(device), contexts.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(texts, contexts)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            epoch_preds.extend(predicted.cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())

            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"\rEpoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}", end="")

        # Calculate epoch metrics
        epoch_loss /= len(train_loader)
        epoch_accuracy = accuracy_score(epoch_labels, epoch_preds)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\rEpoch {epoch + 1}/{epochs} completed in {epoch_time:.2f}s | "
              f"Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")

    # Evaluation
    print("\nEvaluating model on test set...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, contexts, labels in test_loader:
            # Move batch to device
            texts, contexts, labels = texts.to(device), contexts.to(device), labels.to(device)

            # Forward pass
            outputs = model(texts, contexts)

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)

            # Collect predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1-score: {f1:.4f}")

    # Plot training metrics
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title(f'Training Loss ({("Pretrained" if model.pretrained else "Scratch")} Embeddings)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies)
    plt.title(f'Training Accuracy ({("Pretrained" if model.pretrained else "Scratch")} Embeddings)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # Create output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')

    # Save the plot
    plt.tight_layout()
    plt.savefig(f'output/metrics_{("pretrained" if model.pretrained else "scratch")}.png')
    plt.close()

    return accuracy, f1


def main():
    """
    Main function to run the training and evaluation experiments.
    """
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine device (use GPU if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and process data
    vocab, train_loader, test_loader = load_and_process_data()
    vocab_size = len(vocab)

    # Set hyperparameters
    embedding_dim = 100  # 100D embeddings (try 50 if resources are limited)
    hidden_dim = 128  # 128 hidden units (reduce to 64 if necessary)
    output_dim = 3  # 3 classes: Positive, Negative, Neutral
    epochs = 100  # 10 epochs

    # Dictionary to store results
    results = {}

    # Experiment 1: Pretrained GloVe embeddings
    print("\n" + "=" * 50)
    print("Experiment 1: Using Pretrained GloVe Embeddings")
    print("=" * 50)

    model_pretrained = RNNModel(vocab_size, embedding_dim, hidden_dim, output_dim, pretrained=True)
    acc_pre, f1_pre = train_and_evaluate(model_pretrained, train_loader, test_loader,
                                         epochs=epochs, device=device)

    # Save pretrained model
    torch.save(model_pretrained.state_dict(), f'output/model_pretrained_{timestamp}.pt')

    # Store results
    results["RNN_Pretrained=True"] = {
        "Accuracy": float(acc_pre),
        "F1-score": float(f1_pre)
    }

    # Experiment 2: Embeddings trained from scratch
    print("\n" + "=" * 50)
    print("Experiment 2: Using Embeddings Trained from Scratch")
    print("=" * 50)

    model_scratch = RNNModel(vocab_size, embedding_dim, hidden_dim, output_dim, pretrained=False)
    acc_scratch, f1_scratch = train_and_evaluate(model_scratch, train_loader, test_loader,
                                                 epochs=epochs, device=device)

    # Save scratch model
    torch.save(model_scratch.state_dict(), f'output/model_scratch_{timestamp}.pt')

    # Store results
    results["RNN_Pretrained=False"] = {
        "Accuracy": float(acc_scratch),
        "F1-score": float(f1_scratch)
    }

    # Print results summary
    print("\n" + "=" * 50)
    print("Results Summary")
    print("=" * 50)

    print("\n| Experiment     | Accuracy | F1-score |")
    print("|----------------|----------|----------|")
    print(f"| Pretrained     | {acc_pre:.4f}   | {f1_pre:.4f}   |")
    print(f"| Scratch        | {acc_scratch:.4f}   | {f1_scratch:.4f}   |")

    # Add observations
    if acc_pre > acc_scratch:
        observation = "Pretrained outperforms Scratch due to pre-trained embeddings on large datasets."
    elif acc_scratch > acc_pre:
        observation = "Scratch outperforms Pretrained, possibly due to domain-specific learning."
    else:
        observation = "Both approaches perform similarly on this dataset."

    print(f"\nObservation: {observation}")

    # Save results to JSON
    results["observation"] = observation
    results["timestamp"] = timestamp

    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\nResults saved to results.json")
    print("Model checkpoints and plots saved to 'output' directory")


if __name__ == "__main__":
    main()