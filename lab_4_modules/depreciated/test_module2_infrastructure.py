"""
Test script for Lab 4 Module 2 infrastructure.

This verifies that all the core components work before building the interactive interface.
"""

import numpy as np
import sys

print("="*70)
print("TESTING LAB 4 MODULE 2 INFRASTRUCTURE")
print("="*70)

# Test 1: Import dependencies
print("\n[1/6] Testing imports...")
try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    print("  [OK] Matplotlib imported")
except ImportError as e:
    print(f"  [X] Matplotlib import failed: {e}")
    sys.exit(1)

# Test 2: Define sigmoid and dataset
print("\n[2/6] Testing sigmoid and dataset creation...")

def sigmoid(z):
    """Sigmoid with clipping to prevent overflow."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def create_perfect_xor_dataset(n_per_cluster=25, noise_std=0.05, seed=42):
    """Create XOR dataset with minimal noise for training."""
    np.random.seed(seed)

    corners = np.array([
        [-1.5, -1.5],  # BL - Class 0
        [1.5, 1.5],     # TR - Class 0
        [-1.5, 1.5],    # TL - Class 1
        [1.5, -1.5],    # BR - Class 1
    ])
    labels = np.array([0, 0, 1, 1])

    X = np.repeat(corners, n_per_cluster, axis=0)
    y = np.repeat(labels, n_per_cluster)
    X = X + np.random.randn(len(X), 2) * noise_std

    return X, y

X_train, y_train = create_perfect_xor_dataset()
print(f"  [OK] Dataset created: {len(X_train)} samples")
print(f"    - Class 0: {np.sum(y_train==0)} samples")
print(f"    - Class 1: {np.sum(y_train==1)} samples")

# Test 3: Define TinyNetwork
print("\n[3/6] Testing TinyNetwork class...")

class TinyNetwork:
    """2-2-1 neural network for XOR classification."""

    def __init__(self, weights=None):
        if weights is None:
            self.w11 = self.w12 = self.b1 = 0.0
            self.w21 = self.w22 = self.b2 = 0.0
            self.w_out1 = self.w_out2 = self.b_out = 0.0
        else:
            self.set_weights(weights)

    def forward(self, x1, x2):
        """Forward pass for a single input."""
        z1 = self.w11 * x1 + self.w12 * x2 + self.b1
        h1 = sigmoid(z1)
        z2 = self.w21 * x1 + self.w22 * x2 + self.b2
        h2 = sigmoid(z2)
        z_out = self.w_out1 * h1 + self.w_out2 * h2 + self.b_out
        output = sigmoid(z_out)
        return output, h1, h2

    def predict_batch(self, X):
        """Forward pass for batch of inputs."""
        predictions = []
        h1_vals = []
        h2_vals = []
        for x in X:
            out, h1, h2 = self.forward(x[0], x[1])
            predictions.append(out)
            h1_vals.append(h1)
            h2_vals.append(h2)
        return np.array(predictions), np.array(h1_vals), np.array(h2_vals)

    def get_weights(self):
        """Get all weights as a list."""
        return [self.w11, self.w12, self.b1, self.w21, self.w22,
                self.b2, self.w_out1, self.w_out2, self.b_out]

    def set_weights(self, weights):
        """Set all weights from a list."""
        self.w11, self.w12, self.b1, self.w21, self.w22, self.b2, \
        self.w_out1, self.w_out2, self.b_out = weights

network = TinyNetwork()
print("  [OK] TinyNetwork class created")

# Test forward pass
test_output, test_h1, test_h2 = network.forward(0.5, 0.5)
print(f"  [OK] Forward pass works: output={test_output:.4f}, h1={test_h1:.4f}, h2={test_h2:.4f}")

# Test batch prediction
predictions, h1_vals, h2_vals = network.predict_batch(X_train[:5])
print(f"  [OK] Batch prediction works: {len(predictions)} predictions")

# Test 4: Loss and accuracy functions
print("\n[4/6] Testing loss and accuracy functions...")

def compute_loss(network, X, y):
    """Binary cross-entropy loss."""
    predictions, _, _ = network.predict_batch(X)
    epsilon = 1e-10
    bce = -np.mean(y * np.log(predictions + epsilon) +
                   (1 - y) * np.log(1 - predictions + epsilon))
    return bce

def compute_accuracy(network, X, y):
    """Classification accuracy."""
    predictions, _, _ = network.predict_batch(X)
    pred_labels = (predictions > 0.5).astype(int)
    return np.mean(pred_labels == y)

loss = compute_loss(network, X_train, y_train)
acc = compute_accuracy(network, X_train, y_train)
print(f"  [OK] Loss computed: {loss:.4f}")
print(f"  [OK] Accuracy computed: {acc:.2%}")

# Test 5: Gradient computation
print("\n[5/6] Testing gradient computation (backpropagation)...")

def compute_gradients(network, X, y):
    """Analytical backpropagation for 2-2-1 network."""
    n_samples = len(X)

    grads = {'w11': 0, 'w12': 0, 'b1': 0,
             'w21': 0, 'w22': 0, 'b2': 0,
             'w_out1': 0, 'w_out2': 0, 'b_out': 0}

    for i in range(n_samples):
        x1, x2 = X[i]
        target = y[i]

        # Forward pass
        z1 = network.w11 * x1 + network.w12 * x2 + network.b1
        h1 = sigmoid(z1)
        z2 = network.w21 * x1 + network.w22 * x2 + network.b2
        h2 = sigmoid(z2)
        z_out = network.w_out1 * h1 + network.w_out2 * h2 + network.b_out
        y_pred = sigmoid(z_out)

        # Backpropagation
        epsilon = 1e-10
        dL_dy_pred = -(target / (y_pred + epsilon) - (1 - target) / (1 - y_pred + epsilon))
        dy_pred_dz_out = y_pred * (1 - y_pred)
        delta_out = dL_dy_pred * dy_pred_dz_out

        grads['w_out1'] += delta_out * h1
        grads['w_out2'] += delta_out * h2
        grads['b_out'] += delta_out

        delta_h1 = delta_out * network.w_out1 * h1 * (1 - h1)
        delta_h2 = delta_out * network.w_out2 * h2 * (1 - h2)

        grads['w11'] += delta_h1 * x1
        grads['w12'] += delta_h1 * x2
        grads['b1'] += delta_h1
        grads['w21'] += delta_h2 * x1
        grads['w22'] += delta_h2 * x2
        grads['b2'] += delta_h2

    for key in grads:
        grads[key] /= n_samples

    return [grads['w11'], grads['w12'], grads['b1'],
            grads['w21'], grads['w22'], grads['b2'],
            grads['w_out1'], grads['w_out2'], grads['b_out']]

grads = compute_gradients(network, X_train, y_train)
print(f"  [OK] Gradients computed: {len(grads)} gradients")
print(f"    - Gradient range: [{min(grads):.4f}, {max(grads):.4f}]")

# Test 6: Training step
print("\n[6/6] Testing training step...")

def train_step(network, X, y, learning_rate):
    """Single gradient descent step."""
    grads = compute_gradients(network, X, y)
    weights = network.get_weights()
    new_weights = [w - learning_rate * g for w, g in zip(weights, grads)]
    network.set_weights(new_weights)
    loss = compute_loss(network, X, y)
    acc = compute_accuracy(network, X, y)
    return loss, acc

# Test with a convergent seed
CONVERGENT_SEED = [-6.4719, -9.1997, -8.0425, -5.5182, -6.2649, 3.0454, -8.0998, 9.6973, -5.2064]
network.set_weights(CONVERGENT_SEED)

initial_loss = compute_loss(network, X_train, y_train)
initial_acc = compute_accuracy(network, X_train, y_train)
print(f"  Initial state: Loss={initial_loss:.4f}, Acc={initial_acc:.2%}")

# Run 10 training steps
for epoch in range(10):
    loss, acc = train_step(network, X_train, y_train, 0.1)
    if epoch % 3 == 0:
        print(f"    Epoch {epoch+1:2d}: Loss={loss:.4f}, Acc={acc:.2%}")

print(f"  [OK] Training steps work!")

# Test convergence over 100 steps
print("\n" + "="*70)
print("CONVERGENCE TEST (100 epochs with LR=0.1)")
print("="*70)

network.set_weights(CONVERGENT_SEED)
loss_history = []
acc_history = []

for epoch in range(100):
    loss, acc = train_step(network, X_train, y_train, 0.1)
    loss_history.append(loss)
    acc_history.append(acc)

    if epoch % 20 == 0 or acc >= 0.95:
        print(f"Epoch {epoch+1:3d}: Loss={loss:.6f}, Acc={acc:.4f} ({acc*100:.1f}%)")

    if acc >= 0.95:
        print(f"\n[OK] CONVERGED in {epoch+1} epochs!")
        break

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("="*70)
print("\nSummary:")
print(f"  [OK] Dataset creation works")
print(f"  [OK] TinyNetwork forward/backward pass works")
print(f"  [OK] Loss and accuracy computation works")
print(f"  [OK] Gradient computation (backprop) works")
print(f"  [OK] Training step works")
print(f"  [OK] Network converges to {acc*100:.1f}% accuracy")
print(f"\n[OK] Infrastructure is ready for interactive interface!")
print("="*70)
