"""
Binary Classification with PyTorch
COMP 395 – Deep Learning

The same model as binary_classification.py (one sigmoid neuron, squared-error
loss, per-sample SGD, alpha = 0.01, 100 epochs, weights start at zero), written
with an nn.Module class, autograd, and a torch.optim optimizer.

Run:  python binary_classification_pytorch.py
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# =============================================================================
# PART 1: The Model (replaces sigmoid, forward, compute_loss, compute_gradients)
# =============================================================================

class BinaryClassifier(nn.Module):
    """One sigmoid neuron: y_hat = sigmoid(w . x + b)."""

    def __init__(self, n_features):
        super().__init__()
        # nn.Linear holds w (shape 1 x n) and b (shape 1)
        self.linear = nn.Linear(n_features, 1)
        # Match the lab: start from w = 0, b = 0
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        """x: (B, n) batch of samples -> (B,) predictions in (0, 1)."""
        z = self.linear(x)             # x @ w.T + b
        y_hat = torch.sigmoid(z)
        return y_hat.squeeze(-1)

    # Convenience accessors so results line up with the lab's (w, b)
    @property
    def w(self):
        return self.linear.weight.detach().squeeze(0)

    @property
    def b(self):
        return self.linear.bias.detach().squeeze(0)


# =============================================================================
# PART 2: Data Loading and Preprocessing (identical to the lab)
# =============================================================================

def load_data():
    """Load and preprocess the breast cancer dataset."""
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0)
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_test_norm, y_train, y_test, data.feature_names


# =============================================================================
# PART 3: Training Loop
# =============================================================================

def train(X_train, y_train, alpha=0.01, n_epochs=100, verbose=True):
    """
    Train with stochastic gradient descent (one sample per update).

    Returns:
        model: trained BinaryClassifier
        losses: list of average loss per epoch
    """
    model = BinaryClassifier(X_train.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=alpha)

    # batch_size=1 and shuffle=False reproduce the lab's loop exactly
    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=1, shuffle=False)

    losses = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for x_i, y_i in loader:
            # Forward pass
            y_hat = model(x_i)

            # Loss. The lab's hand-derived gradient uses (y_hat - y) without
            # the factor 2, i.e. it is the gradient of 0.5 * (y - y_hat)^2.
            # Scaling by 0.5 makes autograd produce the same updates.
            loss = 0.5 * loss_fn(y_hat, y_i)

            # Backward pass and update
            optimizer.zero_grad()   # gradients accumulate otherwise
            loss.backward()         # computes dL/dw and dL/db
            optimizer.step()        # w -= alpha * dw ; b -= alpha * db

            # Report 0.5 * (y - y_hat)^2, matching the lab's compute_loss.
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(y_train)
        losses.append(avg_loss)

        if verbose and epoch % 10 == 0:
            print(f'Epoch {epoch:3d}, Loss: {avg_loss:.4f}')

    return model, losses


# =============================================================================
# PART 4: Evaluation
# =============================================================================

@torch.no_grad()
def predict(model, X):
    """Predict 0/1 labels for every row of X at once."""
    model.eval()
    return (model(X) >= 0.5).float()


def accuracy(y_true, y_pred):
    """Compute classification accuracy."""
    return (y_true == y_pred).float().mean().item()


# =============================================================================
# PART 5: Main
# =============================================================================

if __name__ == "__main__":
    print("Loading data...")
    X_train, X_test, y_train, y_test, feature_names = load_data()
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Features: {X_train.shape[1]}")

    print("\nTesting sigmoid...")
    print(f"  sigmoid(0) = {torch.sigmoid(torch.tensor(0.0)):.4f} (should be 0.5)")
    print(f"  sigmoid(10) = {torch.sigmoid(torch.tensor(10.0)):.4f} (should be ~1.0)")

    print("\nTraining...")
    model, losses = train(X_train, y_train, alpha=0.01, n_epochs=100)

    print("\nEvaluating...")
    train_acc = accuracy(y_train, predict(model, X_train))
    test_acc = accuracy(y_test, predict(model, X_test))
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Average Loss')
    axes[0].set_title('Training Loss (PyTorch)')

    axes[1].bar(range(len(model.w)), model.w.numpy())
    axes[1].set_xlabel('Feature Index')
    axes[1].set_ylabel('Weight')
    axes[1].set_title('Learned Weights (PyTorch)')

    plt.tight_layout()
    plt.savefig('training_results_pytorch.png', dpi=150)
    print("\nPlot saved to training_results_pytorch.png")
    plt.show()
