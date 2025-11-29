"""
Add educational markdown cell explaining multi-start initialization.
Insert after Cell 14 (Section 3 header), before Cell 15 (plotting function).
"""

import json

# Read notebook
with open('lab_4_module_2_training_neural_network-v3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Adding educational narrative about multi-start initialization...")

# Create new markdown cell
multistart_narrative = '''### Smart Initialization Strategy

When you click **"Reset Network"**, the system tries **10 different random starting points** and automatically picks the one with the lowest initial loss. This is called **multi-start initialization**.

**Why this helps:**
- Some random starting points are better than others for gradient descent
- Finding a good starting point increases the chance of successful convergence
- This could be done automatically (detect when stuck, restart), but we're doing it manually so you can see when restarts help

**Real-world machine learning:**
- Training often uses smart initialization strategies (Xavier, He initialization)
- Advanced optimizers (Adam, RMSprop) adapt learning rates automatically
- Sometimes training is restarted multiple times to find the best solution

**Try this:**
1. Train until convergence or until stuck (loss stops decreasing)
2. If stuck, click "Reset Network" to get a fresh starting point
3. Notice how the initial loss varies - some starting points are much better!
'''

# Create new cell
new_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": multistart_narrative.split('\n')
}

# Add newlines to source lines (except last)
new_cell['source'] = [line + '\n' if i < len(new_cell['source']) - 1 else line
                      for i, line in enumerate(new_cell['source'])]

# Insert after Cell 14 (Section 3 header)
# This will become Cell 15, shifting everything else down
nb['cells'].insert(15, new_cell)

# Save
with open('lab_4_module_2_training_neural_network-v3.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[OK] Added multi-start narrative as new Cell 15")
print(f"Total cells now: {len(nb['cells'])}")
