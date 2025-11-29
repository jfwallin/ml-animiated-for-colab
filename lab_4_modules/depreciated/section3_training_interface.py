"""
Section 3: Watch Learning Happen - Interactive Training Interface

This is a standalone test of the training interface before adding to notebook.
Run this to verify the 2-panel visualization and training buttons work.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Copy infrastructure from test
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def create_perfect_xor_dataset(n_per_cluster=25, noise_std=0.05, seed=42):
    np.random.seed(seed)
    corners = np.array([[-1.5, -1.5], [1.5, 1.5], [-1.5, 1.5], [1.5, -1.5]])
    labels = np.array([0, 0, 1, 1])
    X = np.repeat(corners, n_per_cluster, axis=0)
    y = np.repeat(labels, n_per_cluster)
    X = X + np.random.randn(len(X), 2) * noise_std
    return X, y

class TinyNetwork:
    def __init__(self, weights=None):
        if weights is None:
            self.w11 = self.w12 = self.b1 = 0.0
            self.w21 = self.w22 = self.b2 = 0.0
            self.w_out1 = self.w_out2 = self.b_out = 0.0
        else:
            self.set_weights(weights)

    def forward(self, x1, x2):
        z1 = self.w11 * x1 + self.w12 * x2 + self.b1
        h1 = sigmoid(z1)
        z2 = self.w21 * x1 + self.w22 * x2 + self.b2
        h2 = sigmoid(z2)
        z_out = self.w_out1 * h1 + self.w_out2 * h2 + self.b_out
        output = sigmoid(z_out)
        return output, h1, h2

    def predict_batch(self, X):
        predictions, h1_vals, h2_vals = [], [], []
        for x in X:
            out, h1, h2 = self.forward(x[0], x[1])
            predictions.append(out)
            h1_vals.append(h1)
            h2_vals.append(h2)
        return np.array(predictions), np.array(h1_vals), np.array(h2_vals)

    def get_weights(self):
        return [self.w11, self.w12, self.b1, self.w21, self.w22,
                self.b2, self.w_out1, self.w_out2, self.b_out]

    def set_weights(self, weights):
        self.w11, self.w12, self.b1, self.w21, self.w22, self.b2, \
        self.w_out1, self.w_out2, self.b_out = weights

def compute_loss(network, X, y):
    predictions, _, _ = network.predict_batch(X)
    epsilon = 1e-10
    bce = -np.mean(y * np.log(predictions + epsilon) +
                   (1 - y) * np.log(1 - predictions + epsilon))
    return bce

def compute_accuracy(network, X, y):
    predictions, _, _ = network.predict_batch(X)
    pred_labels = (predictions > 0.5).astype(int)
    return np.mean(pred_labels == y)

def compute_gradients(network, X, y):
    n_samples = len(X)
    grads = {'w11': 0, 'w12': 0, 'b1': 0, 'w21': 0, 'w22': 0, 'b2': 0,
             'w_out1': 0, 'w_out2': 0, 'b_out': 0}

    for i in range(n_samples):
        x1, x2 = X[i]
        target = y[i]

        z1 = network.w11 * x1 + network.w12 * x2 + network.b1
        h1 = sigmoid(z1)
        z2 = network.w21 * x1 + network.w22 * x2 + network.b2
        h2 = sigmoid(z2)
        z_out = network.w_out1 * h1 + network.w_out2 * h2 + network.b_out
        y_pred = sigmoid(z_out)

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

def train_step(network, X, y, learning_rate):
    grads = compute_gradients(network, X, y)
    weights = network.get_weights()
    new_weights = [w - learning_rate * g for w, g in zip(weights, grads)]
    network.set_weights(new_weights)
    loss = compute_loss(network, X, y)
    acc = compute_accuracy(network, X, y)
    return loss, acc

# NEW: 2-Panel Visualization
def plot_training_state(network, X, y, loss_history, epoch, show_log_scale=True):
    """
    2-panel visualization: Input space boundary + Loss curve
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Decision boundary in input space
    x_min, x_max = -2.5, 2.5
    y_min, y_max = -2.5, 2.5
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    Z_out, _, _ = network.predict_batch(mesh_points)
    Z_out = Z_out.reshape(xx.shape)

    ax1.contourf(xx, yy, Z_out, levels=20, alpha=0.4, cmap='RdBu_r')
    ax1.contour(xx, yy, Z_out, levels=[0.5], colors='green', linewidths=3)
    ax1.scatter(X[y==0, 0], X[y==0, 1], c='blue', s=50, alpha=0.7,
               edgecolors='k', linewidths=1, label='Class 0')
    ax1.scatter(X[y==1, 0], X[y==1, 1], c='red', s=50, alpha=0.7,
               edgecolors='k', linewidths=1, label='Class 1')
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel('x1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('x2', fontsize=12, fontweight='bold')
    ax1.set_title(f'Decision Boundary (Epoch {epoch})', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Right panel: Loss curve
    if len(loss_history) > 0:
        ax2.plot(loss_history, 'b-', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Loss (Binary Cross-Entropy)', fontsize=12, fontweight='bold')
        ax2.set_title('Training Loss Curve', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        if show_log_scale and len(loss_history) > 5:
            ax2.set_yscale('log')
            ax2.set_ylabel('Loss (log scale)', fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No training yet...', ha='center', va='center',
                fontsize=14, transform=ax2.transAxes)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)

    plt.tight_layout()
    return fig

# Test the visualization
if __name__ == "__main__":
    print("="*70)
    print("TESTING SECTION 3: INTERACTIVE TRAINING INTERFACE")
    print("="*70)

    # Create dataset
    X_train, y_train = create_perfect_xor_dataset()
    print(f"\n[1/4] Dataset created: {len(X_train)} samples")

    # Initialize network with convergent seed
    CONVERGENT_SEED = [-6.4719, -9.1997, -8.0425, -5.5182, -6.2649, 3.0454,
                       -8.0998, 9.6973, -5.2064]
    network = TinyNetwork(CONVERGENT_SEED)
    print(f"[2/4] Network initialized with convergent seed")

    # Test visualization at epoch 0
    print(f"[3/4] Testing initial visualization...")
    loss_history = [compute_loss(network, X_train, y_train)]
    fig = plot_training_state(network, X_train, y_train, loss_history, epoch=0)
    plt.savefig('section3_test_epoch0.png', dpi=100)
    plt.close()
    print(f"      Saved: section3_test_epoch0.png")

    # Simulate training for 20 steps
    print(f"[4/4] Simulating 20 training steps...")
    for epoch in range(1, 21):
        loss, acc = train_step(network, X_train, y_train, 0.1)
        loss_history.append(loss)

        if epoch % 5 == 0:
            print(f"      Epoch {epoch:2d}: Loss={loss:.6f}, Acc={acc:.2%}")

    # Test visualization after training
    fig = plot_training_state(network, X_train, y_train, loss_history, epoch=20)
    plt.savefig('section3_test_epoch20.png', dpi=100)
    plt.close()
    print(f"      Saved: section3_test_epoch20.png")

    print("\n" + "="*70)
    print("SECTION 3 VISUALIZATION TEST COMPLETE!")
    print("="*70)
    print("\nCheck the generated images:")
    print("  - section3_test_epoch0.png  (initial state)")
    print("  - section3_test_epoch20.png (after 20 epochs)")
    print("\nBoth should show:")
    print("  Left: Decision boundary evolving")
    print("  Right: Loss decreasing over time")
