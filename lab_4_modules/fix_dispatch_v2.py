"""
Fix algorithm dispatch by directly modifying the cell source lines.
"""

import json

# Read notebook
with open('lab_4_module_2_training_neural_network-v3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 15 has the training functions
cell15 = nb['cells'][15]
src_lines = cell15['source']

# Find and replace train_n_steps function
train_n_steps_start = next(i for i, line in enumerate(src_lines) if 'def train_n_steps' in line)
train_n_steps_end = next(i for i in range(train_n_steps_start + 1, len(src_lines)) if src_lines[i].startswith('def '))

# New train_n_steps implementation
new_train_n_steps = '''def train_n_steps(n):
    """Train for n steps using selected algorithm."""
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

    update_display()

'''

# Replace the function
new_lines = new_train_n_steps.split('\n')
new_lines = [line + '\n' for line in new_lines[:-1]] + [new_lines[-1]]
src_lines[train_n_steps_start:train_n_steps_end] = new_lines

# Find and replace on_train_converge function
converge_start = next(i for i, line in enumerate(src_lines) if 'def on_train_converge' in line)
# Find the end (next function starting with 'def ')
try:
    converge_end = next(i for i in range(converge_start + 1, len(src_lines)) if src_lines[i].startswith('def ') or (src_lines[i].startswith("'''") and i > converge_start + 5))
except StopIteration:
    # If there's a comment block, find it
    converge_end = next(i for i in range(converge_start + 1, len(src_lines)) if "'''def on_reset" in src_lines[i])

# New on_train_converge implementation
new_on_train_converge = '''def on_train_converge(btn):
    """Train until convergence or max 500 epochs using selected algorithm."""
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

    update_display()

'''

# Replace the function
new_converge_lines = new_on_train_converge.split('\n')
new_converge_lines = [line + '\n' for line in new_converge_lines[:-1]] + [new_converge_lines[-1]]
src_lines[converge_start:converge_end] = new_converge_lines

# Update cell
cell15['source'] = src_lines

# Write back
with open('lab_4_module_2_training_neural_network-v3.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[OK] Fixed algorithm dispatch")
print(f"  - Updated train_n_steps() at line {train_n_steps_start}")
print(f"  - Updated on_train_converge() at line {converge_start}")
print("Both functions now dispatch to basic or momentum algorithm based on dropdown")
