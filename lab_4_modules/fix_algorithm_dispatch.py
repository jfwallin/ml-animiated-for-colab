"""
Fix the train_n_steps and on_train_converge functions to dispatch to correct algorithm.
"""

import json

# Read notebook
with open('lab_4_module_2_training_neural_network-v3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 15 has the training functions
cell15_source = ''.join(nb['cells'][15]['source'])

# Fix train_n_steps
old_train_n_steps = '''def train_n_steps(n):
    """Train for n steps."""
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

    update_display()'''

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

    update_display()'''

cell15_source = cell15_source.replace(old_train_n_steps, new_train_n_steps)

# Fix on_train_converge
old_train_converge = '''def on_train_converge(btn):
    """Train until convergence or max 200 epochs."""
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

    update_display()'''

new_train_converge = '''def on_train_converge(btn):
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

    update_display()'''

cell15_source = cell15_source.replace(old_train_converge, new_train_converge)

# Save updated cell
nb['cells'][15]['source'] = cell15_source.split('\n')
nb['cells'][15]['source'] = [line + '\n' if i < len(nb['cells'][15]['source']) - 1 else line
                            for i, line in enumerate(nb['cells'][15]['source'])]

# Write back
with open('lab_4_module_2_training_neural_network-v3.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[OK] Fixed algorithm dispatch in train_n_steps and on_train_converge")
print("Both functions now correctly dispatch to basic or momentum algorithm based on dropdown")
