"""
Binary Classification from Scratch
COMP 395 – Deep Learning

Replace the TODO placeholders to complete the implementation.
Run `python -m pytest test_binary_classification.py -v` to test your code.
"""

import torch
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# =============================================================================
# PART 1: Implement the Core Functions
# =============================================================================


def sigmoid(z):
    """
    Sigmoid activation function.
    
    Args:
        z: scalar input (torch.Tensor)
    
    Returns:
        scalar output in (0, 1)
    """
    return 1 / (1 + torch.exp(-z))


def forward(x, w, b):
    """
    Forward pass for one sample.
    
    Args:
        x: (n,) feature vector for one sample
        w: (n,) weight vector
        b: scalar bias
    
    Returns:
        scalar prediction in (0, 1)
    """
    z = torch.dot(w, x) + b
    return sigmoid(z)


def compute_loss(y, y_hat):
    """
    Mean squared error loss for one sample.
    
    Args:
        y: scalar true label (0 or 1)
        y_hat: scalar prediction
    
    Returns:
        scalar loss
    """
    return 0.5 * (y_hat - y) ** 2


def compute_gradients(x, y, y_hat):
    """
    Compute gradients for one sample using the chain rule.
    
    Args:
        x: (n,) input features for one sample
        y: scalar true label
        y_hat: scalar prediction
    
    Returns:
        dw: (n,) gradient for weights
        db: scalar gradient for bias
    """
    error = y_hat - y
    sigmoid_deriv = y_hat * (1 - y_hat)
    delta = error * sigmoid_deriv

    dw = delta * x
    db = delta

    return dw, db


# =============================================================================
# PART 2: Data Loading and Preprocessing (provided)
# =============================================================================

def load_data():
    """Load and preprocess the breast cancer dataset."""
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    # Normalize using training statistics
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
    Train the model using stochastic gradient descent.
    
    Args:
        X_train: (m, n) training features
        y_train: (m,) training labels
        alpha: learning rate
        n_epochs: number of passes through the data
        verbose: print progress
    
    Returns:
        w: learned weights
        b: learned bias
        losses: list of average loss per epoch
    """
    # Initialize parameters
    n_features = X_train.shape[1]
    w = torch.zeros(n_features)
    b = torch.tensor(0.0)
    
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        
        for i in range(len(y_train)):
            x_i = X_train[i]
            y_i = y_train[i]
            
            # Forward pass: compute prediction for this sample
            y_hat = forward(x_i, w, b)

            # Compute loss
            epoch_loss += compute_loss(y_i, y_hat).item()

            # Compute gradients
            dw, db = compute_gradients(x_i, y_i, y_hat)

            # Update parameters using gradient descent
            w = w - alpha * dw
            b = b - alpha * db
        
        avg_loss = epoch_loss / len(y_train)
        losses.append(avg_loss)
        
        if verbose and epoch % 10 == 0:
            print(f'Epoch {epoch:3d}, Loss: {avg_loss:.4f}')
    
    return w, b, losses


# =============================================================================
# PART 4: Evaluation (provided)
# =============================================================================

def predict(X, w, b):
    """Make predictions for multiple samples."""
    predictions = []
    for i in range(len(X)):
        y_hat = forward(X[i], w, b)
        predictions.append(1.0 if y_hat >= 0.5 else 0.0)
    return torch.tensor(predictions)


def accuracy(y_true, y_pred):
    """Compute classification accuracy."""
    return (y_true == y_pred).float().mean().item()


# =============================================================================
# PART 5: Main (run training and evaluation)
# =============================================================================

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test, feature_names = load_data()
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Features: {X_train.shape[1]}")
    
    # Test sigmoid
    print("\nTesting sigmoid...")
    print(f"  sigmoid(0) = {sigmoid(torch.tensor(0.0)):.4f} (should be 0.5)")
    print(f"  sigmoid(10) = {sigmoid(torch.tensor(10.0)):.4f} (should be ~1.0)")
    
    # Train
    print("\nTraining...")
    w, b, losses = train(X_train, y_train, alpha=0.01, n_epochs=100)
    
    # Evaluate
    print("\nEvaluating...")
    train_pred = predict(X_train, w, b)
    test_pred = predict(X_test, w, b)
    
    train_acc = accuracy(y_train, train_pred)
    test_acc = accuracy(y_test, test_pred)
    
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Average Loss')
    axes[0].set_title('Training Loss')
    
    axes[1].bar(range(len(w)), w.numpy())
    axes[1].set_xlabel('Feature Index')
    axes[1].set_ylabel('Weight')
    axes[1].set_title('Learned Weights')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    print("\nPlot saved to training_results.png")
    plt.show()