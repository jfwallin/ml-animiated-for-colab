"""
Add a new cell after the dataset creation to initialize the network with multi-start.
This cell will need to come after Cell 8 (initialize_network_multistart function) is defined.
Actually, we'll add it as part of Cell 16 before the training interface is displayed.
"""

import json

# Read notebook
with open('lab_4_module_2_training_neural_network-v3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Adding initial network creation with multi-start...")

# Cell 16 has training_state initialization
# We need to add code at the END of Cell 16 to initialize the network

cell16 = nb['cells'][16]
cell16_source = ''.join(cell16['source']) if isinstance(cell16['source'], list) else cell16['source']

# Add initialization code at the end (before the print statement)
init_code = '''
# Initialize network with multi-start (try 10 random inits, pick best)
print("Initializing network with multi-start strategy...")
training_state['network'], initial_loss = initialize_network_multistart(
    X_train, y_train,
    n_trials=10,
    scale=0.5,
    verbose=True
)
print(f"Network initialized! Starting loss: {initial_loss:.4f}\\n")
'''

# Insert before the final print statement
cell16_source = cell16_source.replace(
    'print("Training interface ready!")',
    init_code + '\nprint("Training interface ready!")'
)

# Convert back to list format
cell16['source'] = cell16_source.split('\n')
cell16['source'] = [line + '\n' if i < len(cell16['source']) - 1 else line
                   for i, line in enumerate(cell16['source'])]

# Save
with open('lab_4_module_2_training_neural_network-v3.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[OK] Added initial network creation with multi-start to Cell 16")
print("Network will be initialized with 10 random trials, picking the best")
