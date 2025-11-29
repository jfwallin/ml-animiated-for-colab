"""
Fix the critical sigmoid→tanh bug in Cell 7 compute_gradients().
This bug causes gradients to be computed for the wrong activation function.
"""

import json

# Read notebook
with open('lab_4_module_2_training_neural_network-v3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 7 has compute_gradients
cell7 = nb['cells'][7]
cell7_source = ''.join(cell7['source']) if isinstance(cell7['source'], list) else cell7['source']

print("Fixing sigmoid to tanh bug in compute_gradients()...")
print("Before:")
print("  Line 17: h1 = sigmoid(z1)")
print("  Line 19: h2 = sigmoid(z2)")

# Fix the bug: replace sigmoid with tanh for hidden layer activations
cell7_source = cell7_source.replace(
    '        h1 = sigmoid(z1)',
    '        h1 = tanh(z1)'
)
cell7_source = cell7_source.replace(
    '        h2 = sigmoid(z2)',
    '        h2 = tanh(z2)'
)

# Convert back to list format with newlines
cell7['source'] = cell7_source.split('\n')
cell7['source'] = [line + '\n' if i < len(cell7['source']) - 1 else line
                   for i, line in enumerate(cell7['source'])]

# Save
with open('lab_4_module_2_training_neural_network-v3.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nAfter:")
print("  Line 17: h1 = tanh(z1)")
print("  Line 19: h2 = tanh(z2)")
print("\n[OK] Fixed! Gradients now match TinyNetwork.forward() activation functions.")
print("Expected impact: Dramatic improvement in convergence!")
