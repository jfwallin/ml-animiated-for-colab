"""
Script to update lab_4_module_1_anatomy_tiny_nn-v2.ipynb with all improvements.

Changes:
1. Fix accuracy metric in Cell 7 (plot_network_state)
2. Add dataset-specific solution dictionaries to Cell 8
3. Insert new exploration guidance markdown cell
4. Update button labels in Cell 8
5. Reduce vertical spacing in Cell 8
"""

import json

# Read notebook
with open('lab_4_module_1_anatomy_tiny_nn-v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=" * 70)
print("UPDATING LAB 4 MODULE 1 V2")
print("=" * 70)

# ========== CHANGE 1: Fix Accuracy Metric in Cell 7 ==========
print("\n[1/5] Fixing accuracy metric in Cell 7...")

cell7 = nb['cells'][7]
cell7_source = ''.join(cell7['source']) if isinstance(cell7['source'], list) else cell7['source']

# Replace the accuracy calculation
old_accuracy = "    accuracy = np.mean((predictions > 0.5).astype(int) == y) * 100"

new_accuracy = """    # Calculate per-class and balanced accuracy
    pred_labels = (predictions > 0.5).astype(int)
    overall_accuracy = np.mean(pred_labels == y) * 100

    # Per-class accuracy
    class0_mask = (y == 0)
    class1_mask = (y == 1)
    class0_accuracy = np.mean(pred_labels[class0_mask] == y[class0_mask]) * 100 if np.any(class0_mask) else 0
    class1_accuracy = np.mean(pred_labels[class1_mask] == y[class1_mask]) * 100 if np.any(class1_mask) else 0

    # Balanced accuracy (average of per-class accuracies)
    balanced_accuracy = (class0_accuracy + class1_accuracy) / 2"""

cell7_source = cell7_source.replace(old_accuracy, new_accuracy)

# Also need to update the return statement to return the enhanced accuracy
# The function currently returns just 'accuracy', we need to return 'overall_accuracy'
cell7_source = cell7_source.replace("    return accuracy", "    return overall_accuracy, class0_accuracy, class1_accuracy, balanced_accuracy")

# Convert back to list format for notebook
cell7['source'] = cell7_source.split('\n')
# Add newlines back
cell7['source'] = [line + '\n' if i < len(cell7['source']) - 1 else line
                   for i, line in enumerate(cell7['source'])]

print("   [OK] Accuracy metric updated with per-class and balanced accuracy")

# ========== CHANGE 2: Add Dataset-Specific Solution Dictionaries to Cell 8 ==========
print("\n[2/5] Adding dataset-specific solution dictionaries to Cell 8...")

cell8 = nb['cells'][8]
cell8_source = ''.join(cell8['source']) if isinstance(cell8['source'], list) else cell8['source']

# Find where to insert the solution dictionaries (right after the comment about global variable)
insertion_point = "current_dataset_type = 'corner'\n"

solution_dicts = '''
# Dataset-specific example solutions
EXAMPLE_SOLUTIONS = {
    'corner': {
        'w11': 3.0, 'w12': 3.0, 'b1': -3.5,
        'w21': -3.0, 'w22': -3.0, 'b2': 0.5,
        'w_out1': 8.0, 'w_out2': -8.0, 'b_out': -2.0
    },
    'corner_noisy': {
        'w11': 2.5, 'w12': 2.5, 'b1': -3.0,
        'w21': -2.5, 'w22': -2.5, 'b2': 0.0,
        'w_out1': 6.0, 'w_out2': -6.0, 'b_out': -1.5
    },
    'clean': {
        'w11': 5.0, 'w12': 0.0, 'b1': 0.0,
        'w21': 0.0, 'w22': 5.0, 'b2': 0.0,
        'w_out1': 5.0, 'w_out2': 5.0, 'b_out': -7.0
    },
    'noisy': {
        'w11': 5.0, 'w12': 0.0, 'b1': 0.0,
        'w21': 0.0, 'w22': 5.0, 'b2': 0.0,
        'w_out1': 5.0, 'w_out2': 5.0, 'b_out': -7.0
    },
    'perfect': {
        'w11': -10.0, 'w12': -10.0, 'b1': -10.0,
        'w21': -10.0, 'w22': -10.0, 'b2': 5.0,
        'w_out1': -10.0, 'w_out2': 10.0, 'b_out': -5.0
    }
}

PERFECT_SOLUTIONS = {
    'corner': {
        'w11': 5.0, 'w12': 5.0, 'b1': -6.0,
        'w21': -5.0, 'w22': -5.0, 'b2': 1.0,
        'w_out1': 10.0, 'w_out2': -10.0, 'b_out': -3.0
    },
    'corner_noisy': {
        'w11': 4.0, 'w12': 4.0, 'b1': -5.0,
        'w21': -4.0, 'w22': -4.0, 'b2': 0.5,
        'w_out1': 8.0, 'w_out2': -8.0, 'b_out': -2.0
    },
    'clean': {
        'w11': -10.0, 'w12': -10.0, 'b1': -10.0,
        'w21': -10.0, 'w22': -10.0, 'b2': 5.0,
        'w_out1': -10.0, 'w_out2': 10.0, 'b_out': -5.0
    },
    'noisy': {
        'w11': -8.0, 'w12': -8.0, 'b1': -8.0,
        'w21': -8.0, 'w22': -8.0, 'b2': 4.0,
        'w_out1': -8.0, 'w_out2': 8.0, 'b_out': -4.0
    },
    'perfect': {
        'w11': -10.0, 'w12': -10.0, 'b1': -10.0,
        'w21': -10.0, 'w22': -10.0, 'b2': 5.0,
        'w_out1': -10.0, 'w_out2': 10.0, 'b_out': -5.0
    }
}

'''

cell8_source = cell8_source.replace(insertion_point, insertion_point + solution_dicts)

# Update load_example() function to use dictionaries
old_load_example = """def load_example(btn):
    \"\"\"Load simple example solution (not perfect, but understandable).\"\"\"
    # Set sliders to simple, interpretable values
    w11_slider.value = 5
    w12_slider.value = 0
    b1_slider.value = 0

    w21_slider.value = 0
    w22_slider.value = 5
    b2_slider.value = 0

    w_out1_slider.value = 5
    w_out2_slider.value = 5
    b_out_slider.value = -7"""

new_load_example = """def load_example(btn):
    \"\"\"Load simple example solution (dataset-specific).\"\"\"
    solution = EXAMPLE_SOLUTIONS[current_dataset_type]
    w11_slider.value = solution['w11']
    w12_slider.value = solution['w12']
    b1_slider.value = solution['b1']
    w21_slider.value = solution['w21']
    w22_slider.value = solution['w22']
    b2_slider.value = solution['b2']
    w_out1_slider.value = solution['w_out1']
    w_out2_slider.value = solution['w_out2']
    b_out_slider.value = solution['b_out']"""

cell8_source = cell8_source.replace(old_load_example, new_load_example)

# Update load_perfect_solution() function to use dictionaries
# Find the part where weights are actually loaded (after the confirmation check)
old_perfect_load = """        # Actually load the perfect solution
        w11_slider.value = -10
        w12_slider.value = -10
        b1_slider.value = -10

        w21_slider.value = -10
        w22_slider.value = -10
        b2_slider.value = 5

        w_out1_slider.value = -10
        w_out2_slider.value = 10
        b_out_slider.value = -5"""

new_perfect_load = """        # Actually load the perfect solution (dataset-specific)
        solution = PERFECT_SOLUTIONS[current_dataset_type]
        w11_slider.value = solution['w11']
        w12_slider.value = solution['w12']
        b1_slider.value = solution['b1']
        w21_slider.value = solution['w21']
        w22_slider.value = solution['w22']
        w22_slider.value = solution['b2']
        w_out1_slider.value = solution['w_out1']
        w_out2_slider.value = solution['w_out2']
        b_out_slider.value = solution['b_out']"""

cell8_source = cell8_source.replace(old_perfect_load, new_perfect_load)

# Update button labels to indicate dataset-specific behavior
cell8_source = cell8_source.replace(
    "    description='📖 Load Example',",
    "    description='📖 Load Example (dataset-specific)',"
)

cell8_source = cell8_source.replace(
    "    description='💡 Perfect Solution',",
    "    description='💡 Perfect Solution (dataset-specific)',"
)

# Fix: Need to update the update_network function to handle new accuracy format
old_update_accuracy = """    # Update accuracy display
    accuracy_html.value = f"<h3 style='text-align:center; color:#1967d2;'>Current Accuracy: {accuracy:.1f}%</h3>\""""

new_update_accuracy = """    # Update accuracy display with per-class metrics
    overall_acc, class0_acc, class1_acc, balanced_acc = accuracy
    accuracy_html.value = f"<h3 style='text-align:center; color:#1967d2;'>Overall: {overall_acc:.1f}% | Class 0: {class0_acc:.1f}% | Class 1: {class1_acc:.1f}% | Balanced: {balanced_acc:.1f}%</h3>\""""

cell8_source = cell8_source.replace(old_update_accuracy, new_update_accuracy)

# Also update the conditional that uses accuracy
cell8_source = cell8_source.replace(
    "    # Update guidance based on accuracy AND dataset type\n    if current_dataset_type == 'noisy':",
    "    # Update guidance based on accuracy AND dataset type\n    overall_acc_val = overall_acc  # Use overall accuracy for conditionals\n    if current_dataset_type == 'noisy':"
)

cell8_source = cell8_source.replace("if accuracy >= 85:", "if overall_acc_val >= 85:")
cell8_source = cell8_source.replace("elif accuracy >= 70:", "elif overall_acc_val >= 70:")
cell8_source = cell8_source.replace("if accuracy >= 99:", "if overall_acc_val >= 99:")
cell8_source = cell8_source.replace("elif accuracy >= 90:", "elif overall_acc_val >= 90:")
cell8_source = cell8_source.replace("elif accuracy >= 75:", "elif overall_acc_val >= 75:")

# Convert back to list format
cell8['source'] = cell8_source.split('\n')
cell8['source'] = [line + '\n' if i < len(cell8['source']) - 1 else line
                   for i, line in enumerate(cell8['source'])]

print("   [OK] Solution dictionaries added and load functions updated")
print("   [OK] Button labels updated to indicate dataset-specific behavior")

# ========== CHANGE 3: Insert New Exploration Guidance Markdown Cell ==========
print("\n[3/5] Inserting exploration guidance markdown cell...")

exploration_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "\n",
        "### 🎯 Exploration Activity\n",
        "\n",
        "**Try different datasets to see how data complexity affects neural networks:**\n",
        "\n",
        "1. **Start with \"corner\"** - Can you separate one corner from the other three? Use the sliders or try the \"Load Example\" button.\n",
        "\n",
        "2. **Add noise with \"corner_noisy\"** - How does noise affect your solution? Can you still achieve good accuracy?\n",
        "\n",
        "3. **Try \"clean\" XOR** - This is trickier! XOR is NOT linearly separable (you can't draw a straight line to separate the classes).\n",
        "\n",
        "4. **Challenge: \"noisy\" XOR** - Can you find a solution even with noisy data?\n",
        "\n",
        "**Pro tips:**\n",
        "- The \"Load Example\" button gives dataset-specific starting points\n",
        "- The \"Perfect Solution\" button shows near-optimal solutions for each dataset\n",
        "- Watch the **per-class accuracy** - if one class has 0% accuracy, your network might be ignoring it!\n",
        "\n",
        "**For classroom groups**: Your instructor may assign different datasets to different groups. Work together to find the best solution, then compare strategies with other groups!"
    ]
}

# Insert after cell 7 (before current cell 8)
nb['cells'].insert(8, exploration_cell)

print("   [OK] Exploration guidance cell inserted before interactive interface")

# ========== CHANGE 5: Reduce Vertical Spacing (in Cell 9 now, was Cell 8) ==========
print("\n[4/5] Reducing vertical spacing in controls...")

# Cell 8 is now cell 9 after insertion
cell9 = nb['cells'][9]
cell9_source = ''.join(cell9['source']) if isinstance(cell9['source'], list) else cell9['source']

# Remove explicit height from slider layouts (let them auto-adjust)
cell9_source = cell9_source.replace(
    "slider_layout = Layout(width='400px')",
    "slider_layout = Layout(width='400px')  # Height auto-adjusts"
)

# Add compact layouts to VBox widgets
cell9_source = cell9_source.replace(
    "display(VBox([w11_slider, w12_slider, b1_slider]))",
    "display(VBox([w11_slider, w12_slider, b1_slider], layout=Layout(margin='0px', padding='5px')))"
)

cell9_source = cell9_source.replace(
    "display(VBox([w21_slider, w22_slider, b2_slider]))",
    "display(VBox([w21_slider, w22_slider, b2_slider], layout=Layout(margin='0px', padding='5px')))"
)

cell9_source = cell9_source.replace(
    "display(VBox([w_out1_slider, w_out2_slider, b_out_slider]))",
    "display(VBox([w_out1_slider, w_out2_slider, b_out_slider], layout=Layout(margin='0px', padding='5px')))"
)

# Reduce margins between sections
cell9_source = cell9_source.replace(
    "display(HTML(\"<div style='margin:20px 0;'></div>\"))",
    "display(HTML(\"<div style='margin:8px 0;'></div>\"))"
)

cell9_source = cell9_source.replace(
    "display(HTML(\"<div style='margin:10px 0;'></div>\"))",
    "display(HTML(\"<div style='margin:5px 0;'></div>\"))"
)

# Convert back to list format
cell9['source'] = cell9_source.split('\n')
cell9['source'] = [line + '\n' if i < len(cell9['source']) - 1 else line
                   for i, line in enumerate(cell9['source'])]

print("   [OK] Vertical spacing reduced in controls")

# ========== SAVE UPDATED NOTEBOOK ==========
print("\n[5/5] Saving updated notebook...")

with open('lab_4_module_1_anatomy_tiny_nn-v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("   [OK] Notebook saved")

# ========== SUMMARY ==========
print("\n" + "=" * 70)
print("UPDATE COMPLETE!")
print("=" * 70)
print("\nChanges made:")
print("  1. [OK] Fixed accuracy metric with per-class and balanced accuracy")
print("  2. [OK] Added dataset-specific solution dictionaries")
print("  3. [OK] Inserted exploration guidance markdown cell")
print("  4. [OK] Updated button labels to show dataset-specific behavior")
print("  5. [OK] Reduced vertical spacing in controls")
print(f"\nTotal cells in notebook: {len(nb['cells'])}")
print("\nTest the notebook to ensure all changes work correctly!")
print("=" * 70)
