"""
Update on_reset() function to use multi-start initialization.
Also update initial network creation and on_dataset_change to use multi-start.
"""

import json

# Read notebook
with open('lab_4_module_2_training_neural_network-v3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Updating on_reset() and initial network creation to use multi-start...")

# Cell 16 has on_reset and initial training_state
cell16 = nb['cells'][16]
cell16_source = ''.join(cell16['source']) if isinstance(cell16['source'], list) else cell16['source']

# Replace on_reset function
old_on_reset = '''def on_reset(btn):
    """Reset to fresh random initialization."""
    training_state['network'] = TinyNetwork()
    training_state['epoch'] = 0
    training_state['loss_history'] = []
    # Clear momentum velocity
    if 'velocity' in training_state:
        del training_state['velocity']
    update_display()'''

new_on_reset = '''def on_reset(btn):
    """Reset with multi-start initialization (tries multiple random starts)."""
    global X_train, y_train

    # Multi-start: try 10 random inits, pick best
    status_html.value = "<h3>&#128260; Finding good starting point (10 random trials)...</h3>"

    best_network, best_loss = initialize_network_multistart(
        X_train, y_train,
        n_trials=10,
        scale=0.5,
        verbose=False  # Don't clutter output
    )

    # Update state
    training_state['network'] = best_network
    training_state['epoch'] = 0
    training_state['loss_history'] = []

    # Clear momentum velocity
    if 'velocity' in training_state:
        del training_state['velocity']

    # Show result
    acc = compute_accuracy(best_network, X_train, y_train)
    status_html.value = f"<h3>&#10024; Reset! Best init: Loss={best_loss:.4f}, Acc={acc:.2%}</h3>"

    update_display()'''

cell16_source = cell16_source.replace(old_on_reset, new_on_reset)

# Replace initial network creation in training_state
old_init = "    'network': TinyNetwork(),  # random init"
new_init = "    'network': None,  # Will be initialized with multi-start below"

cell16_source = cell16_source.replace(old_init, new_init)

# Replace network creation in on_dataset_change
old_dataset_reset = "    training_state['network'] = TinyNetwork()"
new_dataset_reset = "    best_network, _ = initialize_network_multistart(X_train, y_train, n_trials=10, verbose=False)\n    training_state['network'] = best_network"

cell16_source = cell16_source.replace(old_dataset_reset, new_dataset_reset)

# Convert back to list format
cell16['source'] = cell16_source.split('\n')
cell16['source'] = [line + '\n' if i < len(cell16['source']) - 1 else line
                   for i, line in enumerate(cell16['source'])]

# Save
with open('lab_4_module_2_training_neural_network-v3.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[OK] Updated on_reset() to use multi-start (10 trials)")
print("[OK] Updated on_dataset_change() to use multi-start")
print("[OK] Initial network will be created with multi-start in next cell")
