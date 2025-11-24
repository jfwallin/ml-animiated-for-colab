"""
Script to split lab_1_attempt_3.ipynb into modular notebooks
"""

import json
import sys
import os

# Fix Unicode output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def create_notebook(cells):
    """Create a notebook structure from a list of cells."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

def create_markdown_cell(text):
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text if isinstance(text, list) else [text]
    }

def create_code_cell(code):
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code if isinstance(code, list) else [code]
    }

# Read the original notebook
print("Reading original notebook...")
with open(r'c:\Users\jfwal\OneDrive\python-experiments-2023\ml-animated\ml_animated\lab_1_attempt_3.ipynb', 'r', encoding='utf-8') as f:
    original_nb = json.load(f)

print(f"Original notebook has {len(original_nb['cells'])} cells")

# Create output directory
output_dir = r'c:\Users\jfwal\OneDrive\python-experiments-2023\ml-animated\ml_animated\lab_1_modules'
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# ============================================================================
# MODULE 0: Setup and Group Code
# ============================================================================

print("\n" + "="*70)
print("Creating Module 0: Setup and Group Code")
print("="*70)

module_0_cells = []

# Title
module_0_cells.append(create_markdown_cell([
    "# Lab 1 - Module 0: Setup and Group Code\n",
    "\n",
    "**Run this module first!**\n",
    "\n",
    "This module:\n",
    "1. Sets up your group code\n",
    "2. Generates your group's unique data parameters\n",
    "3. Creates a file you'll use in other modules\n",
    "\n",
    "**Time:** ~3 minutes"
]))

# Getting started info (cell 3 from original)
module_0_cells.append(original_nb['cells'][3])

# Test Colab (cell 4)
module_0_cells.append(original_nb['cells'][4])

# Variables intro (cell 5)
module_0_cells.append(original_nb['cells'][5])

# Simple variables (cell 7)
module_0_cells.append(original_nb['cells'][7])

# Lists and loops (cell 8)
module_0_cells.append(original_nb['cells'][8])

# Plotting example (cell 9)
module_0_cells.append(original_nb['cells'][9])

# Group code explanation (cell 10)
module_0_cells.append(original_nb['cells'][10])

# Group code input (cell 11)
module_0_cells.append(original_nb['cells'][11])

# Generate all parameters and save
module_0_cells.append(create_code_cell([
    "import numpy as np\n",
    "import json\n",
    "from datetime import datetime\n",
    "\n",
    "# Generate all parameters for this group\n",
    "np.random.seed(group_code)\n",
    "\n",
    "# Line fitting parameters\n",
    "true_m = np.random.uniform(-3, 3)\n",
    "true_b = np.random.uniform(-5, 5)\n",
    "\n",
    "# Hidden function parameters\n",
    "hidden_a = np.random.uniform(0.5, 2.0)\n",
    "hidden_b = np.random.uniform(-4, 4)\n",
    "hidden_c = np.random.uniform(-10, 10)\n",
    "\n",
    "# Mountain landscape parameters\n",
    "num_peaks = np.random.randint(3, 6)\n",
    "\n",
    "# Store all parameters\n",
    "group_data = {\n",
    "    \"group_code\": group_code,\n",
    "    \"created_at\": datetime.now().isoformat(),\n",
    "    \"parameters\": {\n",
    "        \"line_slope\": float(true_m),\n",
    "        \"line_intercept\": float(true_b),\n",
    "        \"hidden_func_a\": float(hidden_a),\n",
    "        \"hidden_func_b\": float(hidden_b),\n",
    "        \"hidden_func_c\": float(hidden_c),\n",
    "        \"num_mountain_peaks\": int(num_peaks)\n",
    "    }\n",
    "}\n",
    "\n",
    "# Save to file\n",
    "filename = f\"lab1_group_{group_code}_params.json\"\n",
    "with open(filename, \"w\") as f:\n",
    "    json.dump(group_data, f, indent=2)\n",
    "\n",
    "print(\"=\" * 60)\n",
    "print(\"✓ Setup Complete!\")\n",
    "print(\"=\" * 60)\n",
    "print(f\"Group Code: {group_code}\")\n",
    "print(f\"Parameters file: {filename}\")\n",
    "print()\n",
    "print(\"IMPORTANT: You'll need to enter this same group code\")\n",
    "print(\"in each of the other lab modules!\")\n",
    "print(\"=\" * 60)"
]))

# Instructions
module_0_cells.append(create_markdown_cell([
    "## Next Steps\n",
    "\n",
    "1. **Remember your group code:** " + str(1234) + " (example)\n",
    "2. **Return to the LMS** and continue to Module 1\n",
    "3. **Enter the same group code** in each module you open\n",
    "\n",
    "The group code ensures all modules use the same data for your group!"
]))

# Save Module 0
module_0_nb = create_notebook(module_0_cells)
module_0_path = os.path.join(output_dir, "lab_1_module_0_setup.ipynb")
with open(module_0_path, 'w', encoding='utf-8') as f:
    json.dump(module_0_nb, f, indent=1, ensure_ascii=False)

print(f"✓ Created: {module_0_path}")
print(f"  Cells: {len(module_0_cells)}")

# ============================================================================
# MODULE 1: Global Error Visualization
# ============================================================================

print("\n" + "="*70)
print("Creating Module 1: Global Error Visualization")
print("="*70)

module_1_cells = []

# Title
module_1_cells.append(create_markdown_cell([
    "# Lab 1 - Module 1: Understanding Global Error\n",
    "\n",
    "**Learning Objectives:**\n",
    "- Understand what \"global error\" or \"loss\" means\n",
    "- See how error is measured across all data points\n",
    "- Visualize the relationship between data and predictions\n",
    "\n",
    "**Time:** ~5 minutes\n",
    "\n",
    "---\n",
    "\n",
    "**IMPORTANT:** Enter the same group code you used in Module 0!"
]))

# Setup code
module_1_cells.append(create_code_cell([
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Enter your group code\n",
    "group_code = int(input(\"Enter your group code: \"))\n",
    "np.random.seed(group_code)\n",
    "\n",
    "# Generate data (same as Module 0)\n",
    "true_m = np.random.uniform(-3, 3)\n",
    "true_b = np.random.uniform(-5, 5)\n",
    "x = np.linspace(-5, 5, 25)\n",
    "noise = np.random.normal(0, 1.0, size=len(x))\n",
    "y = true_m * x + true_b + noise\n",
    "\n",
    "def sse(y_true, y_pred):\n",
    "    \"\"\"Sum of squared errors (global loss).\"\"\"\n",
    "    return np.sum((y_true - y_pred)**2)\n",
    "\n",
    "print(\"✓ Data generated for your group\")"
]))

# Add "What is a model" explanation (cells 13-14)
module_1_cells.append(original_nb['cells'][13])
module_1_cells.append(original_nb['cells'][14])

# Section 2 header and plot (cells 16-17)
module_1_cells.append(original_nb['cells'][16])
module_1_cells.append(original_nb['cells'][17])

# Return to LMS
module_1_cells.append(create_markdown_cell([
    "## Next Steps\n",
    "\n",
    "Now that you've seen the global error visualization:\n",
    "\n",
    "1. **Return to the LMS**\n",
    "2. **Answer Questions 1-2** about global error\n",
    "3. **Continue to Module 2** for interactive line fitting"
]))

# Save Module 1
module_1_nb = create_notebook(module_1_cells)
module_1_path = os.path.join(output_dir, "lab_1_module_1_global_error.ipynb")
with open(module_1_path, 'w', encoding='utf-8') as f:
    json.dump(module_1_nb, f, indent=1, ensure_ascii=False)

print(f"✓ Created: {module_1_path}")
print(f"  Cells: {len(module_1_cells)}")

# ============================================================================
# MODULE 2: Interactive Line Fitting
# ============================================================================

print("\n" + "="*70)
print("Creating Module 2: Interactive Line Fitting")
print("="*70)

module_2_cells = []

# Title
module_2_cells.append(create_markdown_cell([
    "# Lab 1 - Module 2: Interactive Line Fitting\n",
    "\n",
    "**Learning Objectives:**\n",
    "- Explore how changing parameters affects error\n",
    "- Understand local vs. global error\n",
    "- Practice minimizing error through experimentation\n",
    "\n",
    "**Time:** ~10-15 minutes\n",
    "\n",
    "---\n",
    "\n",
    "**IMPORTANT:** Enter the same group code you used before!"
]))

# Setup
module_2_cells.append(create_code_cell([
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "from ipywidgets import interact, FloatSlider, Checkbox\n",
    "\n",
    "# Enter your group code\n",
    "group_code = int(input(\"Enter your group code: \"))\n",
    "np.random.seed(group_code)\n",
    "\n",
    "# Generate data\n",
    "true_m = np.random.uniform(-3, 3)\n",
    "true_b = np.random.uniform(-5, 5)\n",
    "x = np.linspace(-5, 5, 25)\n",
    "noise = np.random.normal(0, 1.0, size=len(x))\n",
    "y = true_m * x + true_b + noise\n",
    "\n",
    "def sse(y_true, y_pred):\n",
    "    return np.sum((y_true - y_pred)**2)\n",
    "\n",
    "print(\"✓ Data generated - ready for interactive fitting!\")"
]))

# Instructions
module_2_cells.append(create_markdown_cell([
    "## Instructions\n",
    "\n",
    "Use the sliders below to:\n",
    "1. Adjust the **slope (m)** and **intercept (b)** of the line\n",
    "2. Try to **minimize the Global Error (SSE)**\n",
    "3. Pay attention to the **warm/cold feedback**\n",
    "4. Observe how **residual lines** show local errors\n",
    "\n",
    "Take your time and experiment!"
]))

# Add Section 3 cells (cells 18-19)
module_2_cells.append(original_nb['cells'][18])
module_2_cells.append(original_nb['cells'][19])

# Summary
module_2_cells.append(create_code_cell([
    "# Summary of your attempts\n",
    "print(f\"Total attempts: {len(attempt_history)}\")\n",
    "print(f\"Best error achieved: {min([a['loss'] for a in attempt_history]):.2f}\")\n",
    "print()\n",
    "print(\"Remember these numbers for the LMS questions!\")"
]))

# Return to LMS
module_2_cells.append(create_markdown_cell([
    "## Next Steps\n",
    "\n",
    "1. **Return to the LMS**\n",
    "2. **Answer Questions 3-5** about local vs. global error\n",
    "3. **Continue to Module 3** for parameter space optimization"
]))

# Save Module 2
module_2_nb = create_notebook(module_2_cells)
module_2_path = os.path.join(output_dir, "lab_1_module_2_line_fitting.ipynb")
with open(module_2_path, 'w', encoding='utf-8') as f:
    json.dump(module_2_nb, f, indent=1, ensure_ascii=False)

print(f"✓ Created: {module_2_path}")
print(f"  Cells: {len(module_2_cells)}")

# ============================================================================
# MODULE 3: Parameter Space
# ============================================================================

print("\n" + "="*70)
print("Creating Module 3: Parameter Space Optimization")
print("="*70)

module_3_cells = []

# Title
module_3_cells.append(create_markdown_cell([
    "# Lab 1 - Module 3: Parameter Space Optimization\n",
    "\n",
    "**Learning Objectives:**\n",
    "- Optimize using only global error feedback\n",
    "- Explore parameter space systematically\n",
    "- Compare to gradient-based optimization\n",
    "\n",
    "**Time:** ~15-20 minutes\n",
    "\n",
    "---\n",
    "\n",
    "**IMPORTANT:** Enter the same group code!"
]))

# Setup
module_3_cells.append(create_code_cell([
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "from ipywidgets import FloatSlider, Button, Output, HBox, VBox\n",
    "from IPython.display import display\n",
    "\n",
    "group_code = int(input(\"Enter your group code: \"))\n",
    "np.random.seed(group_code)\n",
    "\n",
    "# Generate data\n",
    "true_m = np.random.uniform(-3, 3)\n",
    "true_b = np.random.uniform(-5, 5)\n",
    "x = np.linspace(-5, 5, 25)\n",
    "noise = np.random.normal(0, 1.0, size=len(x))\n",
    "y = true_m * x + true_b + noise\n",
    "\n",
    "print(\"✓ Data generated - ready for parameter space game!\")"
]))

# Add Section 3.2 explanation and game (cells 21-22)
module_3_cells.append(original_nb['cells'][21])
module_3_cells.append(original_nb['cells'][22])

# Summary
module_3_cells.append(create_code_cell([
    "# Summary\n",
    "if mse_history:\n",
    "    print(f\"Total guesses: {len(mse_history)}\")\n",
    "    best_idx = min(range(len(mse_history)), key=lambda i: mse_history[i]['MSE'])\n",
    "    print(f\"Best MSE: {mse_history[best_idx]['MSE']:.4f}\")\n",
    "    print(f\"Best (m, b): ({mse_history[best_idx]['m']:.2f}, {mse_history[best_idx]['b']:.2f})\")\n",
    "else:\n",
    "    print(\"No guesses made yet!\")"
]))

# Return to LMS
module_3_cells.append(create_markdown_cell([
    "## Next Steps\n",
    "\n",
    "1. **Return to the LMS**\n",
    "2. **Answer Questions 6-8** about parameter space optimization\n",
    "3. **Continue to Module 4** for hidden function optimization"
]))

# Save Module 3
module_3_nb = create_notebook(module_3_cells)
module_3_path = os.path.join(output_dir, "lab_1_module_3_parameter_space.ipynb")
with open(module_3_path, 'w', encoding='utf-8') as f:
    json.dump(module_3_nb, f, indent=1, ensure_ascii=False)

print(f"✓ Created: {module_3_path}")
print(f"  Cells: {len(module_3_cells)}")

# ============================================================================
# MODULE 4: Hidden Function
# ============================================================================

print("\n" + "="*70)
print("Creating Module 4: Hidden Function Optimization")
print("="*70)

module_4_cells = []

# Title
module_4_cells.append(create_markdown_cell([
    "# Lab 1 - Module 4: Hidden Function Optimization\n",
    "\n",
    "**Learning Objectives:**\n",
    "- Optimize a 1D function without seeing it\n",
    "- Use warm/cold feedback strategically\n",
    "- Develop search strategies\n",
    "\n",
    "**Time:** ~10-15 minutes\n",
    "\n",
    "---\n",
    "\n",
    "**IMPORTANT:** Enter the same group code!"
]))

# Setup
module_4_cells.append(create_code_cell([
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "from ipywidgets import FloatSlider, Button, Output, VBox\n",
    "from IPython.display import display\n",
    "\n",
    "group_code = int(input(\"Enter your group code: \"))\n",
    "np.random.seed(group_code)\n",
    "\n",
    "# Generate hidden function parameters\n",
    "a = np.random.uniform(0.5, 2.0)\n",
    "b_param = np.random.uniform(-4, 4)\n",
    "c_param = np.random.uniform(-10, 10)\n",
    "\n",
    "def hidden_func(x_val):\n",
    "    return a * (x_val - b_param)**2 + c_param\n",
    "\n",
    "print(\"✓ Hidden function created - ready to search!\")"
]))

# Add Section 4 cells (cells 25-27)
module_4_cells.append(original_nb['cells'][25])
module_4_cells.append(original_nb['cells'][26])
module_4_cells.append(original_nb['cells'][27])

# Summary
module_4_cells.append(create_code_cell([
    "# Summary\n",
    "if opt_history:\n",
    "    print(f\"Total attempts: {len(opt_history)}\")\n",
    "    best_idx = min(range(len(opt_history)), key=lambda i: opt_history[i]['f(x)'])\n",
    "    print(f\"Best f(x): {opt_history[best_idx]['f(x)']:.4f}\")\n",
    "    print(f\"Best x: {opt_history[best_idx]['x']:.2f}\")\n",
    "else:\n",
    "    print(\"No attempts made yet!\")"
]))

# Return to LMS
module_4_cells.append(create_markdown_cell([
    "## Next Steps\n",
    "\n",
    "1. **Return to the LMS**\n",
    "2. **Answer Questions 9-11** about optimization strategies\n",
    "3. **Continue to Module 5** for the final challenge!"
]))

# Save Module 4
module_4_nb = create_notebook(module_4_cells)
module_4_path = os.path.join(output_dir, "lab_1_module_4_hidden_function.ipynb")
with open(module_4_path, 'w', encoding='utf-8') as f:
    json.dump(module_4_nb, f, indent=1, ensure_ascii=False)

print(f"✓ Created: {module_4_path}")
print(f"  Cells: {len(module_4_cells)}")

# ============================================================================
# MODULE 5: Mountain Landscape
# ============================================================================

print("\n" + "="*70)
print("Creating Module 5: Mountain Landscape Search")
print("="*70)

module_5_cells = []

# Title
module_5_cells.append(create_markdown_cell([
    "# Lab 1 - Module 5: Mountain Landscape Search\n",
    "\n",
    "**Learning Objectives:**\n",
    "- Search in 2D space for optimal point\n",
    "- Understand local vs. global maxima\n",
    "- Connect to ML loss landscapes\n",
    "\n",
    "**Time:** ~15-20 minutes\n",
    "\n",
    "---\n",
    "\n",
    "**IMPORTANT:** Enter the same group code!"
]))

# Setup
module_5_cells.append(create_code_cell([
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "from ipywidgets import FloatSlider, Button, Output, HBox, VBox\n",
    "from IPython.display import display\n",
    "\n",
    "group_code = int(input(\"Enter your group code: \"))\n",
    "np.random.seed(group_code)\n",
    "\n",
    "# Generate mountain landscape\n",
    "num_peaks = np.random.randint(3, 6)\n",
    "peak_centers = []\n",
    "peak_heights = []\n",
    "peak_widths = []\n",
    "\n",
    "for _ in range(num_peaks):\n",
    "    cx = np.random.uniform(-3.0, 3.0)\n",
    "    cy = np.random.uniform(-3.0, 3.0)\n",
    "    height = np.random.uniform(1.0, 5.0)\n",
    "    width = np.random.uniform(0.6, 1.5)\n",
    "    peak_centers.append((cx, cy))\n",
    "    peak_heights.append(height)\n",
    "    peak_widths.append(width)\n",
    "\n",
    "def mountain_height(x, y):\n",
    "    x = np.asarray(x)\n",
    "    y = np.asarray(y)\n",
    "    z = np.zeros_like(x, dtype=float)\n",
    "    for (cx, cy), h, w in zip(peak_centers, peak_heights, peak_widths):\n",
    "        z += h * np.exp(-(((x - cx)**2 + (y - cy)**2) / (2 * w**2)))\n",
    "    return z\n",
    "\n",
    "print(f\"✓ Mountain landscape created with {num_peaks} peaks!\")"
]))

# Add Section 5 cells (cells 30-31)
module_5_cells.append(original_nb['cells'][30])
module_5_cells.append(original_nb['cells'][31])

# Summary
module_5_cells.append(create_code_cell([
    "# Summary\n",
    "if samples_2d:\n",
    "    print(f\"Total samples: {len(samples_2d)}\")\n",
    "    best_idx = max(range(len(samples_2d)), key=lambda i: samples_2d[i]['height'])\n",
    "    print(f\"Best height: {samples_2d[best_idx]['height']:.4f}\")\n",
    "    print(f\"Best (x, y): ({samples_2d[best_idx]['x']:.2f}, {samples_2d[best_idx]['y']:.2f})\")\n",
    "else:\n",
    "    print(\"No samples taken yet!\")"
]))

# Return to LMS
module_5_cells.append(create_markdown_cell([
    "## Next Steps\n",
    "\n",
    "1. **Return to the LMS**\n",
    "2. **Answer Questions 12-15** about the mountain search\n",
    "3. **Submit your lab** - You're done!\n",
    "\n",
    "Great work! 🎉"
]))

# Save Module 5
module_5_nb = create_notebook(module_5_cells)
module_5_path = os.path.join(output_dir, "lab_1_module_5_mountain.ipynb")
with open(module_5_path, 'w', encoding='utf-8') as f:
    json.dump(module_5_nb, f, indent=1, ensure_ascii=False)

print(f"✓ Created: {module_5_path}")
print(f"  Cells: {len(module_5_cells)}")

# ============================================================================
# NARRATIVE NOTEBOOK
# ============================================================================

print("\n" + "="*70)
print("Creating Narrative Notebook (for LMS text extraction)")
print("="*70)

narrative_cells = []

narrative_cells.append(create_markdown_cell([
    "# Lab 1 Narrative Content\n",
    "\n",
    "This notebook contains all the explanatory/narrative text from the original lab.\n",
    "Use this to extract text for LMS pages.\n",
    "\n",
    "This is NOT given to students - it's for instructor reference."
]))

# Collect all markdown cells from original
for i, cell in enumerate(original_nb['cells']):
    if cell['cell_type'] == 'markdown':
        narrative_cells.append(create_markdown_cell([f"---\n\n**Original Cell {i}:**\n\n"]))
        narrative_cells.append(cell)

# Save Narrative
narrative_nb = create_notebook(narrative_cells)
narrative_path = os.path.join(output_dir, "lab_1_narrative.ipynb")
with open(narrative_path, 'w', encoding='utf-8') as f:
    json.dump(narrative_nb, f, indent=1, ensure_ascii=False)

print(f"✓ Created: {narrative_path}")
print(f"  Cells: {len(narrative_cells)}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("✅ MODULE CREATION COMPLETE!")
print("="*70)
print()
print(f"Created {6} notebook modules in: {output_dir}")
print()
print("Modules created:")
print("  0. lab_1_module_0_setup.ipynb")
print("  1. lab_1_module_1_global_error.ipynb")
print("  2. lab_1_module_2_line_fitting.ipynb")
print("  3. lab_1_module_3_parameter_space.ipynb")
print("  4. lab_1_module_4_hidden_function.ipynb")
print("  5. lab_1_module_5_mountain.ipynb")
print("  +  lab_1_narrative.ipynb (for LMS text)")
print()
print("Next steps:")
print("  1. Review each module in Colab")
print("  2. Test the workflow (Module 0 → 1 → 2 → 3 → 4 → 5)")
print("  3. Upload to Google Drive or GitHub for Colab links")
print("  4. Create LMS page with links and question boxes")
print("="*70)
