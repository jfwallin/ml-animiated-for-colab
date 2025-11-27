"""
Fix the load_example and load_perfect_solution functions to use dataset-specific dictionaries.
"""

import json
import re

# Read notebook
with open('lab_4_module_1_anatomy_tiny_nn-v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Fixing load_example and load_perfect_solution functions...")

# Cell 9 (interactive interface)
cell9 = nb['cells'][9]
cell9_source = ''.join(cell9['source']) if isinstance(cell9['source'], list) else cell9['source']

# Fix load_example function - find and replace the entire function body
# Match from function def to the update_network call
pattern_load_example = r"(def load_example\(btn\):.*?\"\"\".*?\"\"\".*?)(# Set sliders.*?b_out_slider\.value = -7)"

replacement_load_example = r'''\1solution = EXAMPLE_SOLUTIONS[current_dataset_type]
    w11_slider.value = solution['w11']
    w12_slider.value = solution['w12']
    b1_slider.value = solution['b1']
    w21_slider.value = solution['w21']
    w22_slider.value = solution['w22']
    b2_slider.value = solution['b2']
    w_out1_slider.value = solution['w_out1']
    w_out2_slider.value = solution['w_out2']
    b_out_slider.value = solution['b_out']'''

cell9_source = re.sub(pattern_load_example, replacement_load_example, cell9_source, flags=re.DOTALL)

# Fix load_perfect_solution function - find the weight assignment part
pattern_perfect = r"(# Actually load the perfect solution.*?)(w11_slider\.value = -10.*?b_out_slider\.value = -5)"

replacement_perfect = r'''\1solution = PERFECT_SOLUTIONS[current_dataset_type]
        w11_slider.value = solution['w11']
        w12_slider.value = solution['w12']
        b1_slider.value = solution['b1']
        w21_slider.value = solution['w21']
        w22_slider.value = solution['w22']
        b2_slider.value = solution['b2']
        w_out1_slider.value = solution['w_out1']
        w_out2_slider.value = solution['w_out2']
        b_out_slider.value = solution['b_out']'''

cell9_source = re.sub(pattern_perfect, replacement_perfect, cell9_source, flags=re.DOTALL)

# Convert back to list format
cell9['source'] = cell9_source.split('\n')
cell9['source'] = [line + '\n' if i < len(cell9['source']) - 1 else line
                   for i, line in enumerate(cell9['source'])]

# Save
with open('lab_4_module_1_anatomy_tiny_nn-v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[OK] load_example() function updated to use EXAMPLE_SOLUTIONS[current_dataset_type]")
print("[OK] load_perfect_solution() function updated to use PERFECT_SOLUTIONS[current_dataset_type]")
print("\nVerifying changes...")

# Verify
cell9_check = ''.join(nb['cells'][9]['source'])
if 'solution = EXAMPLE_SOLUTIONS[current_dataset_type]' in cell9_check:
    print("[OK] load_example uses dataset-specific solution")
else:
    print("[ERROR] load_example still hardcoded")

if 'solution = PERFECT_SOLUTIONS[current_dataset_type]' in cell9_check:
    print("[OK] load_perfect_solution uses dataset-specific solution")
else:
    print("[ERROR] load_perfect_solution still hardcoded")

print("\nDone!")
