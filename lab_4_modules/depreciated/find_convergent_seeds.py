"""
Find and test convergent weight initializations for Lab 4 Module 2.

This script searches for random seeds that reliably converge when training
a 2-2-1 neural network on perfect XOR data with LR=0.1.

Goal: Find 10 seeds that converge to >95% accuracy within 200 epochs.
"""

import numpy as np

def sigmoid(z):
    """Sigmoid activation function with clipping to prevent overflow."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def create_perfect_xor_dataset(n_per_cluster=25, noise_std=0.05, seed=42):
    """Create perfect XOR dataset with minimal noise."""
    np.random.seed(seed)

    # Four corner points
    corners = np.array([
        [-1.5, -1.5],  # BL - Class 0
        [1.5, 1.5],     # TR - Class 0
        [-1.5, 1.5],    # TL - Class 1
        [1.5, -1.5],    # BR - Class 1
    ])
    labels = np.array([0, 0, 1, 1])

    # Replicate and add noise
    X = np.repeat(corners, n_per_cluster, axis=0)
    y = np.repeat(labels, n_per_cluster)
    X = X + np.random.randn(len(X), 2) * noise_std

    return X, y

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

def compute_gradients(network, X, y):
    """Analytical backpropagation for 2-2-1 network."""
    n_samples = len(X)

    # Initialize gradient accumulators
    grads = {
        'w11': 0, 'w12': 0, 'b1': 0,
        'w21': 0, 'w22': 0, 'b2': 0,
        'w_out1': 0, 'w_out2': 0, 'b_out': 0
    }

    # Accumulate gradients over all samples
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
        # Output layer
        epsilon = 1e-10
        dL_dy_pred = -(target / (y_pred + epsilon) - (1 - target) / (1 - y_pred + epsilon))
        dy_pred_dz_out = y_pred * (1 - y_pred)
        delta_out = dL_dy_pred * dy_pred_dz_out

        grads['w_out1'] += delta_out * h1
        grads['w_out2'] += delta_out * h2
        grads['b_out'] += delta_out

        # Hidden layer
        delta_h1 = delta_out * network.w_out1 * h1 * (1 - h1)
        delta_h2 = delta_out * network.w_out2 * h2 * (1 - h2)

        grads['w11'] += delta_h1 * x1
        grads['w12'] += delta_h1 * x2
        grads['b1'] += delta_h1
        grads['w21'] += delta_h2 * x1
        grads['w22'] += delta_h2 * x2
        grads['b2'] += delta_h2

    # Average gradients
    for key in grads:
        grads[key] /= n_samples

    return [grads['w11'], grads['w12'], grads['b1'],
            grads['w21'], grads['w22'], grads['b2'],
            grads['w_out1'], grads['w_out2'], grads['b_out']]

def train_step(network, X, y, learning_rate):
    """Single gradient descent step."""
    grads = compute_gradients(network, X, y)
    weights = network.get_weights()
    new_weights = [w - learning_rate * g for w, g in zip(weights, grads)]
    network.set_weights(new_weights)
    loss = compute_loss(network, X, y)
    acc = compute_accuracy(network, X, y)
    return loss, acc

def test_seed(seed_idx, init_scale=0.5, lr=0.1, max_epochs=200, target_acc=0.95):
    """Test if a random seed leads to convergence."""
    np.random.seed(seed_idx)

    # Create dataset
    X, y = create_perfect_xor_dataset()

    # Initialize network with random weights
    init_weights = np.random.randn(9) * init_scale
    network = TinyNetwork(init_weights)

    # Track training
    loss_history = []
    acc_history = []

    # Initial state
    loss = compute_loss(network, X, y)
    acc = compute_accuracy(network, X, y)
    loss_history.append(loss)
    acc_history.append(acc)

    # Training loop
    converged = False
    for epoch in range(max_epochs):
        loss, acc = train_step(network, X, y, lr)
        loss_history.append(loss)
        acc_history.append(acc)

        # Check convergence
        if acc >= target_acc:
            converged = True
            break

        # Check for divergence (NaN or exploding loss)
        if not np.isfinite(loss) or loss > 100:
            break

    return {
        'seed': seed_idx,
        'converged': converged,
        'final_acc': acc,
        'final_loss': loss,
        'epochs': epoch + 1,
        'init_weights': init_weights.tolist(),
        'loss_history': loss_history,
        'acc_history': acc_history
    }

def find_convergent_seeds(n_seeds_to_find=10, n_seeds_to_test=100):
    """Search for convergent seeds."""
    print(f"Searching for {n_seeds_to_find} convergent seeds...")
    print(f"Testing up to {n_seeds_to_test} random initializations")
    print("="*70)

    convergent_seeds = []

    for seed_idx in range(n_seeds_to_test):
        result = test_seed(seed_idx)

        if result['converged']:
            convergent_seeds.append(result)
            print(f"[OK] Seed {seed_idx:3d}: Converged in {result['epochs']:3d} epochs "
                  f"(acc={result['final_acc']:.3f}, loss={result['final_loss']:.4f})")

            if len(convergent_seeds) >= n_seeds_to_find:
                break
        else:
            if seed_idx % 10 == 0:
                print(f"[ ] Seed {seed_idx:3d}: Did not converge (acc={result['final_acc']:.3f})")

    print("="*70)
    print(f"\nFound {len(convergent_seeds)} convergent seeds out of {seed_idx + 1} tested")

    return convergent_seeds

if __name__ == "__main__":
    # Find seeds
    seeds = find_convergent_seeds(n_seeds_to_find=10, n_seeds_to_test=100)

    # Print summary
    print("\n" + "="*70)
    print("CONVERGENT SEEDS SUMMARY")
    print("="*70)
    print(f"{'Seed':<6} {'Epochs':<8} {'Final Acc':<12} {'Final Loss':<12}")
    print("-"*70)
    for s in seeds:
        print(f"{s['seed']:<6} {s['epochs']:<8} {s['final_acc']:<12.4f} {s['final_loss']:<12.6f}")

    # Export as Python list
    print("\n" + "="*70)
    print("COPY THIS TO NOTEBOOK:")
    print("="*70)
    print("\nCONVERGENT_SEEDS = [")
    for i, s in enumerate(seeds):
        weights_str = "[" + ", ".join([f"{w:.6f}" for w in s['init_weights']]) + "]"
        print(f"    {weights_str},  # Seed {s['seed']} - converges in {s['epochs']} epochs")
    print("]")

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
