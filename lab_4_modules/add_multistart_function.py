"""
Add initialize_network_multistart() function to the notebook.
This will be inserted as a new code cell after Cell 7 (compute_gradients).
"""

import json

# Read notebook
with open('lab_4_module_2_training_neural_network-v3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Adding initialize_network_multistart() function...")

# Create new code cell with the multi-start initialization function
multistart_code = '''def initialize_network_multistart(X, y, n_trials=10, scale=0.5, verbose=True):
    """
    Try N random initializations and return the one with lowest initial loss.

    Parameters:
    - X, y: Training data
    - n_trials: Number of random initializations to try (default 10)
    - scale: Standard deviation for weight initialization (default 0.5)
    - verbose: Print progress (default True)

    Returns:
    - best_network: TinyNetwork with best initial weights
    - best_loss: Initial loss of best network
    """
    best_network = None
    best_loss = np.inf

    if verbose:
        print(f"Trying {n_trials} random initializations...")

    for trial in range(n_trials):
        # Create network with random init
        network = TinyNetwork(scale=scale)

        # Compute initial loss
        loss = compute_loss(network, X, y)

        if verbose and trial < 5:  # Show first few trials
            print(f"  Trial {trial+1}: Initial loss = {loss:.4f}")

        # Keep if best so far
        if loss < best_loss:
            best_loss = loss
            best_network = network

    if verbose:
        print(f"[OK] Best initialization: Loss = {best_loss:.4f}")

    return best_network, best_loss

print("Multi-start initialization function ready!")
'''

# Create new cell
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": multistart_code.split('\n')
}

# Add newlines to source lines (except last)
new_cell['source'] = [line + '\n' if i < len(new_cell['source']) - 1 else line
                      for i, line in enumerate(new_cell['source'])]

# Insert after Cell 7 (compute_gradients)
# This will become Cell 8, shifting everything else down
nb['cells'].insert(8, new_cell)

# Save
with open('lab_4_module_2_training_neural_network-v3.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[OK] Added initialize_network_multistart() as new Cell 8")
print(f"Total cells now: {len(nb['cells'])}")
