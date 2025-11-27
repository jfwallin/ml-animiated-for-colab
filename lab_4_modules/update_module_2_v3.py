"""
Script to update lab_4_module_2_training_neural_network-v3.ipynb with all improvements.

Changes:
1. Remove duplicate cells 15-18 (causing TypeError)
2. Add momentum training function
3. Add algorithm dropdown
4. Integrate all 5 datasets from Module 1
5. Add learning rate dropdown
6. Update narrative
"""

import json
import sys

# Read Module 1 v2 to get dataset function
print("Reading Module 1 v2 for dataset function...")
with open('lab_4_module_1_anatomy_tiny_nn-v2.ipynb', 'r', encoding='utf-8') as f:
    module1_nb = json.load(f)

# Extract dataset creation function from Cell 3 of Module 1
module1_cell3_source = ''.join(module1_nb['cells'][3]['source'])

# Read Module 2 v3
print("Reading Module 2 v3...")
with open('lab_4_module_2_training_neural_network-v3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Original notebook has {len(nb['cells'])} cells")

# ========== STEP 1: Remove duplicate cells 15-18 ==========
print("\n[Step 1/6] Removing duplicate cells 15-18...")
# Cells 15-18 are duplicates of cells 11-14 with old code
# We want to keep cells 0-14 and remove 15-18
nb['cells'] = nb['cells'][:15] + nb['cells'][19:]
print(f"   [OK] Removed 4 duplicate cells. Now have {len(nb['cells'])} cells")

# ========== STEP 2: Add train_step_momentum function ==========
print("\n[Step 2/6] Adding train_step_momentum function...")

momentum_cell_source = '''def train_step_momentum(network, X, y, base_lr, training_state):
    """Gradient descent with momentum and line search."""
    grads = compute_gradients(network, X, y)
    w = np.array(network.get_weights(), dtype=float)

    # Momentum parameter (typical value: 0.9)
    beta = 0.9

    # Initialize velocity if first iteration
    if 'velocity' not in training_state:
        training_state['velocity'] = np.zeros_like(grads)

    # Update velocity: v = beta * v + (1-beta) * gradient
    training_state['velocity'] = beta * training_state['velocity'] + (1 - beta) * np.array(grads)

    # Direction is now based on velocity instead of raw gradient
    direction = -training_state['velocity']

    # Line search: try 3 learning rates
    lrs = [base_lr, base_lr / 2, base_lr * 2]
    best_loss = np.inf
    best_w = w

    for eta in lrs:
        cand_w = w + eta * direction
        network.set_weights(cand_w.tolist())
        loss = compute_loss(network, X, y)
        if loss < best_loss:
            best_loss = loss
            best_w = cand_w

    # Commit to the best candidate
    network.set_weights(best_w.tolist())
    loss = best_loss
    acc = compute_accuracy(network, X, y)
    return loss, acc

print("Momentum training function ready!")'''

momentum_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": momentum_cell_source.split('\n')
}

# Add newlines to source
momentum_cell['source'] = [line + '\n' if i < len(momentum_cell['source']) - 1 else line
                          for i, line in enumerate(momentum_cell['source'])]

# Insert after Cell 8 (train_step function)
nb['cells'].insert(9, momentum_cell)
print("   [OK] Added train_step_momentum function after cell 8")

# ========== STEP 3: Replace dataset creation with multi-dataset function ==========
print("\n[Step 3/6] Replacing dataset creation with multi-dataset function...")

# Extract create_xor_dataset from Module 1 Cell 3
# This contains all 5 dataset types
create_dataset_source = module1_cell3_source

# Replace Cell 4 with new dataset creation
new_dataset_cell_source = '''def create_xor_dataset(dataset_type='clean', n_per_cluster=25, noise_std=0.25, seed=42):
    """
    Create XOR-style datasets with different difficulty levels.

    Parameters:
    - dataset_type: 'corner', 'corner_noisy', 'clean', 'noisy', or 'perfect'
    - n_per_cluster: number of samples per cluster (default 25)
    - noise_std: standard deviation of Gaussian noise (default 0.25)
    - seed: random seed for reproducibility

    Returns:
    - X: features (n_samples, 2)
    - y: labels (n_samples,)
    """
    np.random.seed(seed)

    if dataset_type == 'corner':
        # One corner vs other three corners (easier - linearly separable)
        corners = np.array([
            [-1.5, -1.5],  # BL - Class 0
            [-1.5, 1.5],   # TL - Class 1
            [1.5, -1.5],   # BR - Class 1
            [1.5, 1.5],    # TR - Class 1
        ])
        labels = np.array([0, 1, 1, 1])
        X = np.repeat(corners, n_per_cluster, axis=0)
        y = np.repeat(labels, n_per_cluster)
        X = X + np.random.randn(len(X), 2) * noise_std

    elif dataset_type == 'corner_noisy':
        # One corner vs three with more noise
        corners = np.array([
            [-1.5, -1.5],  # BL - Class 0
            [-1.5, 1.5],   # TL - Class 1
            [1.5, -1.5],   # BR - Class 1
            [1.5, 1.5],    # TR - Class 1
        ])
        labels = np.array([0, 1, 1, 1])
        X = np.repeat(corners, n_per_cluster, axis=0)
        y = np.repeat(labels, n_per_cluster)
        X = X + np.random.randn(len(X), 2) * (noise_std * 1.5)  # More noise

    elif dataset_type == 'clean':
        # Standard XOR (moderate difficulty)
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

    elif dataset_type == 'noisy':
        # XOR with more overlap (harder)
        corners = np.array([
            [-1.5, -1.5],  # BL - Class 0
            [1.5, 1.5],     # TR - Class 0
            [-1.5, 1.5],    # TL - Class 1
            [1.5, -1.5],    # BR - Class 1
        ])
        labels = np.array([0, 0, 1, 1])
        X = np.repeat(corners, n_per_cluster, axis=0)
        y = np.repeat(labels, n_per_cluster)
        X = X + np.random.randn(len(X), 2) * (noise_std * 2.0)  # Much more noise

    elif dataset_type == 'perfect':
        # Minimal noise XOR (easiest)
        corners = np.array([
            [-1.5, -1.5],  # BL - Class 0
            [1.5, 1.5],     # TR - Class 0
            [-1.5, 1.5],    # TL - Class 1
            [1.5, -1.5],    # BR - Class 1
        ])
        labels = np.array([0, 0, 1, 1])
        X = np.repeat(corners, n_per_cluster, axis=0)
        y = np.repeat(labels, n_per_cluster)
        X = X + np.random.randn(len(X), 2) * 0.05  # Minimal noise

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    return X, y

# Create initial dataset (will be updated by dropdown)
current_dataset_type = 'clean'
X_train, y_train = create_xor_dataset(dataset_type=current_dataset_type)

print(f"Dataset created: {len(X_train)} samples")
print(f"Class 0: {np.sum(y_train==0)} samples")
print(f"Class 1: {np.sum(y_train==1)} samples")
print(f"Current dataset: {current_dataset_type}")

# Visualize
plt.figure(figsize=(6, 6))
for label in [0, 1]:
    mask = y_train == label
    color = 'blue' if label == 0 else 'red'
    plt.scatter(X_train[mask, 0], X_train[mask, 1], c=color, s=30, alpha=0.6, edgecolors='k', linewidths=0.5)
plt.xlabel('x₁', fontsize=12)
plt.ylabel('x₂', fontsize=12)
plt.title(f'Dataset: {current_dataset_type}', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()'''

# Replace Cell 4
nb['cells'][4]['source'] = new_dataset_cell_source.split('\n')
nb['cells'][4]['source'] = [line + '\n' if i < len(nb['cells'][4]['source']) - 1 else line
                            for i, line in enumerate(nb['cells'][4]['source'])]
print("   [OK] Replaced Cell 4 with multi-dataset creation function")

# ========== STEP 4: Update training_state initialization and add dropdowns ==========
print("\n[Step 4/6] Adding algorithm and learning rate dropdowns...")

# Find the cell with training state (Cell 13 after adding momentum function, so now Cell 14)
cell_idx = 14  # Cell 13 becomes Cell 14 after insert

cell14_source = ''.join(nb['cells'][cell_idx]['source'])

# Add algorithm and learning rate to training_state
old_training_state_init = """training_state = {
    'network': TinyNetwork(),  # random init
    'epoch': 0,
    'loss_history': [],
    'learning_rate': 0.3,
    'current_seed_idx': None,  # we don't need seed list any more
}"""

new_training_state_init = """training_state = {
    'network': TinyNetwork(),  # random init
    'epoch': 0,
    'loss_history': [],
    'learning_rate': 0.3,
    'algorithm': 'basic',  # 'basic' or 'momentum'
    'current_seed_idx': None,
}"""

cell14_source = cell14_source.replace(old_training_state_init, new_training_state_init)

# Add algorithm dropdown before buttons
algorithm_dropdown_code = '''
# Algorithm selector dropdown
from ipywidgets import Dropdown

algorithm_dropdown = Dropdown(
    options=[
        ('Basic Gradient Descent', 'basic'),
        ('Gradient Descent + Momentum', 'momentum')
    ],
    value='basic',
    description='Algorithm:',
    layout=Layout(width='300px')
)

def on_algorithm_change(change):
    training_state['algorithm'] = change['new']
    # Reset velocity when switching algorithms
    if 'velocity' in training_state:
        del training_state['velocity']

algorithm_dropdown.observe(on_algorithm_change, names='value')

# Learning rate dropdown
lr_dropdown = Dropdown(
    options=[
        ('Slow (0.1)', 0.1),
        ('Moderate (0.3) - Default', 0.3),
        ('Fast (0.5)', 0.5),
        ('Very Fast (1.0)', 1.0)
    ],
    value=0.3,
    description='Learning Rate:',
    layout=Layout(width='300px')
)

def on_lr_change(change):
    training_state['learning_rate'] = change['new']

lr_dropdown.observe(on_lr_change, names='value')

# Dataset selector dropdown
dataset_dropdown = Dropdown(
    options=[
        ('Corner (easier)', 'corner'),
        ('Corner Noisy (moderate)', 'corner_noisy'),
        ('Clean XOR (moderate)', 'clean'),
        ('Noisy XOR (harder)', 'noisy'),
        ('Perfect XOR (easiest)', 'perfect')
    ],
    value='clean',
    description='Dataset:',
    layout=Layout(width='300px')
)

def on_dataset_change(change):
    global X_train, y_train, current_dataset_type
    current_dataset_type = change['new']
    X_train, y_train = create_xor_dataset(dataset_type=current_dataset_type)
    # Reset training when dataset changes
    training_state['network'] = TinyNetwork()
    training_state['epoch'] = 0
    training_state['loss_history'] = []
    if 'velocity' in training_state:
        del training_state['velocity']
    update_display()

dataset_dropdown.observe(on_dataset_change, names='value')
'''

# Insert dropdown code before buttons
insertion_point = "# Buttons"
cell14_source = cell14_source.replace(insertion_point, algorithm_dropdown_code + "\n" + insertion_point)

# Update train_n_steps to use algorithm selector
old_train_n_steps = """def train_n_steps(n):
    \"\"\"Train for n steps.\"\"\"
    for _ in range(n):
        #loss, acc = train_step(training_state['network'], X_train, y_train,
        #                      training_state['learning_rate'])
        loss, acc = train_step(training_state['network'], X_train, y_train,
                       training_state['learning_rate'], training_state)

        training_state['loss_history'].append(loss)
        training_state['epoch'] += 1

        # Early stopping if converged
        if acc >= 0.99:
            break

    update_display()"""

new_train_n_steps = """def train_n_steps(n):
    \"\"\"Train for n steps using selected algorithm.\"\"\"
    algorithm = training_state.get('algorithm', 'basic')

    for _ in range(n):
        # Dispatch to correct training function
        if algorithm == 'momentum':
            loss, acc = train_step_momentum(training_state['network'], X_train, y_train,
                                          training_state['learning_rate'], training_state)
        else:
            loss, acc = train_step(training_state['network'], X_train, y_train,
                                 training_state['learning_rate'], training_state)

        training_state['loss_history'].append(loss)
        training_state['epoch'] += 1

        # Early stopping if converged
        if acc >= 0.99:
            break

    update_display()"""

cell14_source = cell14_source.replace(old_train_n_steps, new_train_n_steps)

# Update on_train_converge similarly
old_train_converge = """def on_train_converge(btn):
    \"\"\"Train until convergence or max 200 epochs.\"\"\"
    max_epochs = 500
    while training_state['epoch'] < max_epochs:
        #loss, acc = train_step(training_state['network'], X_train, y_train,
        #                      training_state['learning_rate'])
        loss, acc = train_step(training_state['network'], X_train, y_train,
                       training_state['learning_rate'], training_state)

        training_state['loss_history'].append(loss)
        training_state['epoch'] += 1

        if acc >= 0.99 or loss < 0.01:
            break

    update_display()"""

new_train_converge = """def on_train_converge(btn):
    \"\"\"Train until convergence or max 500 epochs using selected algorithm.\"\"\"
    algorithm = training_state.get('algorithm', 'basic')
    max_epochs = 500

    while training_state['epoch'] < max_epochs:
        # Dispatch to correct training function
        if algorithm == 'momentum':
            loss, acc = train_step_momentum(training_state['network'], X_train, y_train,
                                          training_state['learning_rate'], training_state)
        else:
            loss, acc = train_step(training_state['network'], X_train, y_train,
                                 training_state['learning_rate'], training_state)

        training_state['loss_history'].append(loss)
        training_state['epoch'] += 1

        if acc >= 0.99 or loss < 0.01:
            break

    update_display()"""

cell14_source = cell14_source.replace(old_train_converge, new_train_converge)

# Update on_reset to clear velocity
old_reset = """def on_reset(btn):
    \"\"\"Reset to fresh random initialization.\"\"\"
    training_state['network'] = TinyNetwork()
    training_state['epoch'] = 0
    training_state['loss_history'] = []
    update_display()"""

new_reset = """def on_reset(btn):
    \"\"\"Reset to fresh random initialization.\"\"\"
    training_state['network'] = TinyNetwork()
    training_state['epoch'] = 0
    training_state['loss_history'] = []
    # Clear momentum velocity
    if 'velocity' in training_state:
        del training_state['velocity']
    update_display()"""

cell14_source = cell14_source.replace(old_reset, new_reset)

# Save updated cell 14
nb['cells'][cell_idx]['source'] = cell14_source.split('\n')
nb['cells'][cell_idx]['source'] = [line + '\n' if i < len(nb['cells'][cell_idx]['source']) - 1 else line
                                   for i, line in enumerate(nb['cells'][cell_idx]['source'])]

print("   [OK] Updated training state and added dropdowns")

# ========== STEP 5: Update display cell to show dropdowns ==========
print("\n[Step 5/6] Updating display cell to show dropdowns...")

# Cell 15 (after insert, was Cell 14)
cell15_idx = 15
cell15_source = ''.join(nb['cells'][cell15_idx]['source'])

# Update display to include dropdowns
old_display = """display(status_html)
display(HBox([train_1_btn, train_10_btn, train_converge_btn, reset_btn]))
display(plot_output)"""

new_display = """display(status_html)
display(HTML("<h4>Training Configuration:</h4>"))
display(HBox([dataset_dropdown, algorithm_dropdown, lr_dropdown]))
display(HTML("<h4>Training Controls:</h4>"))
display(HBox([train_1_btn, train_10_btn, train_converge_btn, reset_btn]))
display(plot_output)"""

cell15_source = cell15_source.replace(old_display, new_display)

nb['cells'][cell15_idx]['source'] = cell15_source.split('\n')
nb['cells'][cell15_idx]['source'] = [line + '\n' if i < len(nb['cells'][cell15_idx]['source']) - 1 else line
                                     for i, line in enumerate(nb['cells'][cell15_idx]['source'])]

print("   [OK] Updated display to show configuration dropdowns")

# ========== STEP 6: Update narrative cells ==========
print("\n[Step 6/6] Updating narrative cells...")

# Update Cell 1 to mention datasets and algorithms
cell1_source = ''.join(nb['cells'][1]['source'])

new_intro_paragraph = """

### This Module: Compare Algorithms and Datasets

In this module, you'll:
- **Try different datasets**: See how problem difficulty affects convergence
- **Compare algorithms**: Basic gradient descent vs. gradient descent with momentum
- **Experiment with learning rates**: Find the sweet spot between speed and stability

You'll discover that **better algorithms converge more reliably** across different datasets!
"""

# Add after "Enter: Gradient Descent" section
insertion_point_intro = "In this module, you'll **watch this process happen in real time**!"
cell1_source = cell1_source.replace(insertion_point_intro, insertion_point_intro + new_intro_paragraph)

nb['cells'][1]['source'] = cell1_source.split('\n')
nb['cells'][1]['source'] = [line + '\n' if i < len(nb['cells'][1]['source']) - 1 else line
                            for i, line in enumerate(nb['cells'][1]['source'])]

# Add guidance markdown cell before Section 3
guidance_cell_source = """---

## Understanding Your Controls

Before you start training, here's what each control does:

### Dataset Selector
- **Corner / Corner Noisy**: Easier problem (one corner vs. three corners)
- **Clean XOR**: Moderate difficulty (standard XOR problem)
- **Noisy XOR**: Harder (lots of overlap between classes)
- **Perfect XOR**: Easiest (minimal noise)

**Try different datasets to see how problem difficulty affects learning!**

### Algorithm Selector
- **Basic Gradient Descent**: Uses line search to pick best step size
- **Gradient Descent + Momentum**: Adds "velocity" to help escape local minima and speed up convergence

**Momentum often works better on harder problems!**

### Learning Rate
- **Slow (0.1)**: Safe but may take many steps
- **Moderate (0.3)**: Good default balance
- **Fast (0.5)**: Converges quickly but may be unstable
- **Very Fast (1.0)**: May overshoot or diverge

**Experiment to see the trade-off between speed and stability!**"""

guidance_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": guidance_cell_source.split('\n')
}

guidance_cell['source'] = [line + '\n' if i < len(guidance_cell['source']) - 1 else line
                          for i, line in enumerate(guidance_cell['source'])]

# Insert before Cell 11 (Section 3 header, which is now at index 11 after earlier insert)
nb['cells'].insert(11, guidance_cell)

print("   [OK] Updated narrative and added guidance cell")

# ========== SAVE ==========
print(f"\nFinal notebook has {len(nb['cells'])} cells")

with open('lab_4_module_2_training_neural_network-v3.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n" + "="*70)
print("UPDATE COMPLETE!")
print("="*70)
print("\nChanges made:")
print("  1. [OK] Removed duplicate cells 15-18")
print("  2. [OK] Added train_step_momentum function")
print("  3. [OK] Added algorithm dropdown (basic vs momentum)")
print("  4. [OK] Added learning rate dropdown (0.1, 0.3, 0.5, 1.0)")
print("  5. [OK] Added dataset dropdown (5 dataset types)")
print("  6. [OK] Updated training functions to use algorithm selector")
print("  7. [OK] Updated narrative and added guidance")
print(f"\nTotal cells: {len(nb['cells'])}")
print("="*70)
