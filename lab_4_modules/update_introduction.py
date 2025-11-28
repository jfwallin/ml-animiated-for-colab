"""
Update Cell 1 introduction to mention convergence robustness and smart initialization.
"""

import json

# Read notebook
with open('lab_4_module_2_training_neural_network-v3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Updating introduction to mention convergence robustness...")

# Cell 1 has the introduction
cell1 = nb['cells'][1]
cell1_source = ''.join(cell1['source']) if isinstance(cell1['source'], list) else cell1['source']

# Add new paragraph before "This Module: Compare Algorithms and Datasets"
new_paragraph = '''### Making Training Reliable

In this module, we'll see how gradient descent can automatically find good solutions - even for tricky problems like XOR. We'll use:
- **Smart initialization**: Trying multiple random starting points and picking the best one
- **Momentum**: A better gradient descent algorithm that helps escape local minima
- **Flexible learning rates**: Adjusting the step size to balance speed and stability

These techniques make training **robust and reliable** across different datasets!

'''

# Insert the new paragraph before "### This Module: Compare Algorithms and Datasets"
cell1_source = cell1_source.replace(
    '### This Module: Compare Algorithms and Datasets',
    new_paragraph + '### This Module: Compare Algorithms and Datasets'
)

# Convert back to list format
cell1['source'] = cell1_source.split('\n')
cell1['source'] = [line + '\n' if i < len(cell1['source']) - 1 else line
                   for i, line in enumerate(cell1['source'])]

# Save
with open('lab_4_module_2_training_neural_network-v3.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[OK] Updated introduction with convergence robustness paragraph")
print("New section added: 'Making Training Reliable'")
