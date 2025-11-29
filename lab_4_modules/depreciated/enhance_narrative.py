"""
Enhance the narrative to:
1. Remove outdated "pre-tested starting points" text
2. Add clear explanation of the gradient descent algorithm
3. Explain why multiple starting points help
4. Clarify that complex code follows simple rules
"""

import json

# Read notebook
with open('lab_4_module_2_training_neural_network-v3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Enhancing narrative...")

# ===== Cell 11: Replace "Pre-tested Starting Points" with algorithm explanation =====
cell11 = nb['cells'][11]
cell11_source = ''.join(cell11['source'])

# Replace the entire cell with better explanation
new_cell11_content = '''### How Gradient Descent Works: Simple Rules, Powerful Results

The code you see below looks complex, but it follows a few **simple rules**:

**The Algorithm (in plain English):**
1. **Forward pass**: Calculate prediction from current weights
2. **Compute error**: How wrong is the prediction?
3. **Backpropagation**: Calculate which direction to adjust each weight
4. **Update weights**: Take a small step in that direction
5. **Repeat** until error is small enough

**The math behind it:**
- **Derivatives** tell us which direction makes error smaller
- **Chain rule** connects output error back to every weight
- **Learning rate** controls step size (too big = unstable, too small = slow)

Despite ~100 lines of code, it's just these 5 steps repeated!

### Why Multiple Starting Points?

The loss landscape has **hills and valleys**. Gradient descent rolls downhill from wherever it starts:
- **Good starting point**: Near a deep valley → converges quickly
- **Bad starting point**: On a plateau or shallow valley → gets stuck

**Multi-start strategy**: Try 10 random starts, pick the one with lowest initial loss. This gives us the best "head start" down the hill!
'''

cell11['source'] = new_cell11_content.split('\n')
cell11['source'] = [line + '\n' if i < len(cell11['source']) - 1 else line
                    for i, line in enumerate(cell11['source'])]

# ===== Cell 15: Enhance multi-start explanation with more context =====
cell15 = nb['cells'][15]
cell15_source = ''.join(cell15['source'])

new_cell15_content = '''### Smart Initialization Strategy

When you click **"Reset Network"**, the system tries **10 different random starting points** and automatically picks the one with the lowest initial loss. This is called **multi-start initialization**.

**Why this helps:**
- The loss landscape is like a bumpy terrain with many hills and valleys
- Some random starting points are near deep valleys (good!) → fast convergence
- Others are on plateaus or shallow valleys (bad!) → slow or no convergence
- By trying multiple starts, we increase the odds of finding a good valley

**The intuition:**
- Imagine dropping 10 balls randomly on a hilly landscape
- Each ball rolls to a different local low point
- We pick the ball that found the **deepest valley** to start from
- This gives gradient descent the best chance to find a great solution!

**Real-world machine learning:**
- Training often uses smart initialization strategies (Xavier, He initialization)
- Advanced optimizers (Adam, RMSprop) adapt learning rates automatically
- Large models sometimes run multiple training sessions and pick the best result

**Try this:**
1. Train until convergence or until stuck (loss stops decreasing)
2. If stuck, click "Reset Network" to get a fresh starting point
3. Notice how the initial loss varies - some starting points are much better!
4. Compare: Does momentum help escape bad starting points?
'''

cell15['source'] = new_cell15_content.split('\n')
cell15['source'] = [line + '\n' if i < len(cell15['source']) - 1 else line
                    for i, line in enumerate(cell15['source'])]

# ===== Cell 1: Enhance "Enter: Gradient Descent" section =====
cell1 = nb['cells'][1]
cell1_source = ''.join(cell1['source'])

# Add a paragraph about algorithm simplicity
enhanced_gd_section = '''### Enter: Gradient Descent

Think of the loss function as a landscape:
- **High points** = bad weights (high error)
- **Low points** = good weights (low error)
- **Goal**: Roll downhill to find the lowest point

Gradient descent is like **rolling a ball downhill** in this loss landscape:
1. Start at a random location (random weights)
2. Look around and find which direction is steepest downward (compute gradients)
3. Take a small step in that direction (update weights)
4. Repeat until you reach a valley bottom (converged!)

**The key insight**: Even though the code looks complex (calculating derivatives, chain rule, etc.), it's just these 4 steps over and over. The math handles **9 weights simultaneously**, but the idea is simple: always move downhill!

In this module, you'll **watch this process happen in real time**!
'''

# Replace the old "Enter: Gradient Descent" section
old_gd_section = '''### Enter: Gradient Descent

Think of the loss function as a landscape:
- **High points** = bad weights (high error)
- **Low points** = good weights (low error)
- **Goal**: Roll downhill to find the lowest point

Gradient descent is like **rolling a ball downhill** in this loss landscape:
1. Start at a random location (random weights)
2. Look around and find which direction is steepest downward
3. Take a step in that direction
4. Repeat until you reach the bottom

In this module, you'll **watch this process happen in real time**!
'''

cell1_source = cell1_source.replace(old_gd_section, enhanced_gd_section)

cell1['source'] = cell1_source.split('\n')
cell1['source'] = [line + '\n' if i < len(cell1['source']) - 1 else line
                   for i, line in enumerate(cell1['source'])]

# Save
with open('lab_4_module_2_training_neural_network-v3.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[OK] Enhanced Cell 1: Added algorithm simplicity explanation")
print("[OK] Updated Cell 11: Explained algorithm steps and multiple starting points")
print("[OK] Enhanced Cell 15: Better multi-start intuition with landscape analogy")
print("\nKey improvements:")
print("  - Emphasized simple 4-step algorithm despite complex code")
print("  - Explained loss landscape and why some starts are better")
print("  - Added intuitive ball-rolling analogy for multi-start")
print("  - Connected to real-world ML practices")
