"""
Automated Tests for Binary Classification Lab
COMP 395 – Deep Learning

DO NOT MODIFY THIS FILE.

Run with: python -m pytest test_binary_classification.py -v
"""

import pytest
import torch

# Import student's implementation
from binary_classification import sigmoid, forward, compute_loss, compute_gradients, train, load_data, predict, accuracy


# =============================================================================
# Helper Functions
# =============================================================================

def approx_equal(a, b, tol=1e-5):
    """Check if two values are approximately equal."""
    if isinstance(a, torch.Tensor):
        a = a.item() if a.dim() == 0 else a
    if isinstance(b, torch.Tensor):
        b = b.item() if b.dim() == 0 else b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < tol
    return torch.allclose(torch.tensor(a).float(), torch.tensor(b).float(), atol=tol)


# =============================================================================
# Test: Sigmoid Function
# =============================================================================

class TestSigmoid:
    """Tests for the sigmoid function."""
    
    def test_sigmoid_zero(self):
        """sigmoid(0) should equal 0.5"""
        result = sigmoid(torch.tensor(0.0))
        assert approx_equal(result, 0.5), f"sigmoid(0) = {result}, expected 0.5"
    
    def test_sigmoid_positive_large(self):
        """sigmoid(10) should be very close to 1"""
        result = sigmoid(torch.tensor(10.0))
        assert result > 0.999, f"sigmoid(10) = {result}, expected > 0.999"
    
    def test_sigmoid_negative_large(self):
        """sigmoid(-10) should be very close to 0"""
        result = sigmoid(torch.tensor(-10.0))
        assert result < 0.001, f"sigmoid(-10) = {result}, expected < 0.001"
    
    def test_sigmoid_positive(self):
        """sigmoid(2) should equal 1/(1+e^-2) ≈ 0.8808"""
        result = sigmoid(torch.tensor(2.0))
        expected = 1 / (1 + torch.exp(torch.tensor(-2.0)))
        assert approx_equal(result, expected), f"sigmoid(2) = {result}, expected {expected}"
    
    def test_sigmoid_negative(self):
        """sigmoid(-1) should equal 1/(1+e^1) ≈ 0.2689"""
        result = sigmoid(torch.tensor(-1.0))
        expected = 1 / (1 + torch.exp(torch.tensor(1.0)))
        assert approx_equal(result, expected), f"sigmoid(-1) = {result}, expected {expected}"


# =============================================================================
# Test: Forward Pass
# =============================================================================

class TestForward:
    """Tests for the forward pass."""
    
    def test_forward_simple(self):
        """forward([1,1], [1,1], 0) should equal sigmoid(2)"""
        w = torch.tensor([1.0, 1.0])
        x = torch.tensor([1.0, 1.0])
        b = torch.tensor(0.0)
        result = forward(x, w, b)
        expected = sigmoid(torch.tensor(2.0))
        assert approx_equal(result, expected), f"forward result = {result}, expected {expected}"
    
    def test_forward_with_bias(self):
        """forward([3,4], [1,2], -5) should equal sigmoid(6)"""
        w = torch.tensor([1.0, 2.0])
        x = torch.tensor([3.0, 4.0])
        b = torch.tensor(-5.0)
        result = forward(x, w, b)
        expected = sigmoid(torch.tensor(6.0))
        assert approx_equal(result, expected), f"forward result = {result}, expected {expected}"
    
    def test_forward_zero_weights(self):
        """forward(any, [0,0,0], 1) should equal sigmoid(1)"""
        w = torch.tensor([0.0, 0.0, 0.0])
        x = torch.tensor([5.0, 10.0, 15.0])
        b = torch.tensor(1.0)
        result = forward(x, w, b)
        expected = sigmoid(torch.tensor(1.0))
        assert approx_equal(result, expected), f"forward result = {result}, expected {expected}"
    
    def test_forward_output_range(self):
        """forward output should stay in [0, 1], even for extreme inputs
        (float32 sigmoid saturates to exactly 1.0 for large z)"""
        w = torch.tensor([1.0, -2.0, 0.5])
        x = torch.tensor([100.0, -50.0, 25.0])
        b = torch.tensor(10.0)
        result = forward(x, w, b)
        assert 0 <= result <= 1, f"forward output {result} not in [0, 1]"


# =============================================================================
# Test: Loss Function
# =============================================================================

class TestLoss:
    """Tests for the loss function."""
    
    def test_loss_perfect_one(self):
        """loss(y=1, y_hat=1) should equal 0"""
        result = compute_loss(torch.tensor(1.0), torch.tensor(1.0))
        assert approx_equal(result, 0.0), f"loss = {result}, expected 0.0"
    
    def test_loss_perfect_zero(self):
        """loss(y=0, y_hat=0) should equal 0"""
        result = compute_loss(torch.tensor(0.0), torch.tensor(0.0))
        assert approx_equal(result, 0.0), f"loss = {result}, expected 0.0"
    
    def test_loss_wrong_prediction_high(self):
        """loss(y=0, y_hat=1) should equal 0.5"""
        result = compute_loss(torch.tensor(0.0), torch.tensor(1.0))
        assert approx_equal(result, 0.5), f"loss = {result}, expected 0.5"
    
    def test_loss_wrong_prediction_low(self):
        """loss(y=1, y_hat=0) should equal 0.5"""
        result = compute_loss(torch.tensor(1.0), torch.tensor(0.0))
        assert approx_equal(result, 0.5), f"loss = {result}, expected 0.5"
    
    def test_loss_partial_prediction(self):
        """loss(y=1, y_hat=0.5) should equal 0.125"""
        result = compute_loss(torch.tensor(1.0), torch.tensor(0.5))
        assert approx_equal(result, 0.125), f"loss = {result}, expected 0.125"
    
    def test_loss_another_partial(self):
        """loss(y=0, y_hat=0.2) should equal 0.02"""
        result = compute_loss(torch.tensor(0.0), torch.tensor(0.2))
        assert approx_equal(result, 0.02), f"loss = {result}, expected 0.02"


# =============================================================================
# Test: Gradients
# =============================================================================

class TestGradients:
    """Tests for gradient computation."""
    
    def test_gradients_basic(self):
        """Test gradient computation with known values"""
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor(1.0)
        y_hat = torch.tensor(0.8)
        
        dw, db = compute_gradients(x, y, y_hat)
        
        # error = 0.8 - 1 = -0.2
        # sigmoid_deriv = 0.8 * 0.2 = 0.16
        # delta = -0.2 * 0.16 = -0.032
        expected_delta = -0.032
        expected_dw = torch.tensor([expected_delta * 1.0, expected_delta * 2.0])
        expected_db = expected_delta
        
        assert approx_equal(dw, expected_dw), f"dw = {dw}, expected {expected_dw}"
        assert approx_equal(db, expected_db), f"db = {db}, expected {expected_db}"
    
    def test_gradients_positive_error(self):
        """Test gradient with positive error (prediction too high)"""
        x = torch.tensor([0.5, -1.0, 2.0])
        y = torch.tensor(0.0)
        y_hat = torch.tensor(0.3)
        
        dw, db = compute_gradients(x, y, y_hat)
        
        # error = 0.3 - 0 = 0.3
        # sigmoid_deriv = 0.3 * 0.7 = 0.21
        # delta = 0.3 * 0.21 = 0.063
        expected_delta = 0.063
        expected_dw = expected_delta * x
        expected_db = expected_delta
        
        assert approx_equal(dw, expected_dw), f"dw = {dw}, expected {expected_dw}"
        assert approx_equal(db, expected_db), f"db = {db}, expected {expected_db}"
    
    def test_gradients_zero_when_perfect(self):
        """Gradient should be zero when prediction equals label"""
        x = torch.tensor([5.0, 3.0])
        y = torch.tensor(1.0)
        y_hat = torch.tensor(1.0)
        
        dw, db = compute_gradients(x, y, y_hat)
        
        assert approx_equal(dw, torch.tensor([0.0, 0.0])), f"dw = {dw}, expected [0, 0]"
        assert approx_equal(db, 0.0), f"db = {db}, expected 0"
    
    def test_gradients_max_sigmoid_deriv(self):
        """Test gradient at y_hat=0.5 (maximum sigmoid derivative)"""
        x = torch.tensor([1.0, 1.0])
        y = torch.tensor(0.0)
        y_hat = torch.tensor(0.5)
        
        dw, db = compute_gradients(x, y, y_hat)
        
        # error = 0.5, sigmoid_deriv = 0.25, delta = 0.125
        expected_dw = torch.tensor([0.125, 0.125])
        expected_db = 0.125
        
        assert approx_equal(dw, expected_dw), f"dw = {dw}, expected {expected_dw}"
        assert approx_equal(db, expected_db), f"db = {db}, expected {expected_db}"
    
    def test_gradients_shape(self):
        """Gradient dw should have same shape as input x"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = torch.tensor(1.0)
        y_hat = torch.tensor(0.6)
        
        dw, db = compute_gradients(x, y, y_hat)
        
        assert dw.shape == x.shape, f"dw shape = {dw.shape}, expected {x.shape}"
    
    def test_gradients_direction(self):
        """Gradient should push prediction toward label"""
        x = torch.tensor([1.0, 1.0])
        
        # When y=1 and y_hat < 1, gradient should be negative (to increase weights)
        dw_low, db_low = compute_gradients(x, torch.tensor(1.0), torch.tensor(0.3))
        assert db_low < 0, "Gradient should be negative when y=1 and y_hat=0.3"
        
        # When y=0 and y_hat > 0, gradient should be positive (to decrease weights)
        dw_high, db_high = compute_gradients(x, torch.tensor(0.0), torch.tensor(0.7))
        assert db_high > 0, "Gradient should be positive when y=0 and y_hat=0.7"


# =============================================================================
# Test: Training
# =============================================================================

class TestTraining:
    """Tests for the complete training pipeline."""
    
    @pytest.fixture
    def data(self):
        """Load data once for all training tests."""
        return load_data()
    
    def test_training_loss_decreases(self, data):
        """Training loss should decrease over epochs"""
        X_train, X_test, y_train, y_test, _ = data
        
        _, _, losses = train(X_train, y_train, alpha=0.01, n_epochs=50, verbose=False)
        
        # First 10 epochs average should be higher than last 10 epochs average
        early_loss = sum(losses[:10]) / 10
        late_loss = sum(losses[-10:]) / 10
        
        assert late_loss < early_loss, f"Loss should decrease: early={early_loss:.4f}, late={late_loss:.4f}"
    
    def test_training_accuracy(self, data):
        """Trained model should achieve >90% accuracy"""
        X_train, X_test, y_train, y_test, _ = data
        
        w, b, _ = train(X_train, y_train, alpha=0.01, n_epochs=100, verbose=False)
        
        train_pred = predict(X_train, w, b)
        train_acc = accuracy(y_train, train_pred)
        
        assert train_acc > 0.90, f"Training accuracy = {train_acc:.4f}, expected > 0.90"
    
    def test_test_accuracy(self, data):
        """Model should generalize to test set with >85% accuracy"""
        X_train, X_test, y_train, y_test, _ = data
        
        w, b, _ = train(X_train, y_train, alpha=0.01, n_epochs=100, verbose=False)
        
        test_pred = predict(X_test, w, b)
        test_acc = accuracy(y_test, test_pred)
        
        assert test_acc > 0.85, f"Test accuracy = {test_acc:.4f}, expected > 0.85"
    
    def test_weights_learned(self, data):
        """Weights should be non-zero after training"""
        X_train, X_test, y_train, y_test, _ = data
        
        w, b, _ = train(X_train, y_train, alpha=0.01, n_epochs=50, verbose=False)
        
        assert not torch.allclose(w, torch.zeros_like(w)), "Weights should be non-zero after training"


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
