"""
Update corner dataset solutions with the working solution found by user.
Solution: w11=-5.5, w12=-3, b1=-2, w21=2.5, w22=-9.5, b2=-6, w_out1=-8, w_out2=8.5, b_out=-6
"""

import json

# Read notebook
with open('lab_4_module_1_anatomy_tiny_nn-v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Updating corner solutions with working solution...")

# Cell 9 (interactive interface)
cell9 = nb['cells'][9]
cell9_source = ''.join(cell9['source']) if isinstance(cell9['source'], list) else cell9['source']

# Replace corner solution in EXAMPLE_SOLUTIONS
old_corner_example = """    'corner': {
        'w11': 3.0, 'w12': 3.0, 'b1': -3.5,
        'w21': -3.0, 'w22': -3.0, 'b2': 0.5,
        'w_out1': 8.0, 'w_out2': -8.0, 'b_out': -2.0
    },"""

new_corner_example = """    'corner': {
        'w11': -5.5, 'w12': -3.0, 'b1': -2.0,
        'w21': 2.5, 'w22': -9.5, 'b2': -6.0,
        'w_out1': -8.0, 'w_out2': 8.5, 'b_out': -6.0
    },"""

cell9_source = cell9_source.replace(old_corner_example, new_corner_example)

# Replace corner_noisy solution in EXAMPLE_SOLUTIONS (use same solution)
old_corner_noisy_example = """    'corner_noisy': {
        'w11': 2.5, 'w12': 2.5, 'b1': -3.0,
        'w21': -2.5, 'w22': -2.5, 'b2': 0.0,
        'w_out1': 6.0, 'w_out2': -6.0, 'b_out': -1.5
    },"""

new_corner_noisy_example = """    'corner_noisy': {
        'w11': -5.5, 'w12': -3.0, 'b1': -2.0,
        'w21': 2.5, 'w22': -9.5, 'b2': -6.0,
        'w_out1': -8.0, 'w_out2': 8.5, 'b_out': -6.0
    },"""

cell9_source = cell9_source.replace(old_corner_noisy_example, new_corner_noisy_example)

# Replace corner solution in PERFECT_SOLUTIONS
old_corner_perfect = """    'corner': {
        'w11': 5.0, 'w12': 5.0, 'b1': -6.0,
        'w21': -5.0, 'w22': -5.0, 'b2': 1.0,
        'w_out1': 10.0, 'w_out2': -10.0, 'b_out': -3.0
    },"""

new_corner_perfect = """    'corner': {
        'w11': -5.5, 'w12': -3.0, 'b1': -2.0,
        'w21': 2.5, 'w22': -9.5, 'b2': -6.0,
        'w_out1': -8.0, 'w_out2': 8.5, 'b_out': -6.0
    },"""

cell9_source = cell9_source.replace(old_corner_perfect, new_corner_perfect)

# Replace corner_noisy solution in PERFECT_SOLUTIONS (use same solution)
old_corner_noisy_perfect = """    'corner_noisy': {
        'w11': 4.0, 'w12': 4.0, 'b1': -5.0,
        'w21': -4.0, 'w22': -4.0, 'b2': 0.5,
        'w_out1': 8.0, 'w_out2': -8.0, 'b_out': -2.0
    },"""

new_corner_noisy_perfect = """    'corner_noisy': {
        'w11': -5.5, 'w12': -3.0, 'b1': -2.0,
        'w21': 2.5, 'w22': -9.5, 'b2': -6.0,
        'w_out1': -8.0, 'w_out2': 8.5, 'b_out': -6.0
    },"""

cell9_source = cell9_source.replace(old_corner_noisy_perfect, new_corner_noisy_perfect)

# Convert back to list format
cell9['source'] = cell9_source.split('\n')
cell9['source'] = [line + '\n' if i < len(cell9['source']) - 1 else line
                   for i, line in enumerate(cell9['source'])]

# Save
with open('lab_4_module_1_anatomy_tiny_nn-v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[OK] Updated EXAMPLE_SOLUTIONS['corner'] with working solution")
print("[OK] Updated EXAMPLE_SOLUTIONS['corner_noisy'] with working solution")
print("[OK] Updated PERFECT_SOLUTIONS['corner'] with working solution")
print("[OK] Updated PERFECT_SOLUTIONS['corner_noisy'] with working solution")
print("\nSolution values:")
print("  w11=-5.5, w12=-3.0, b1=-2.0")
print("  w21=2.5, w22=-9.5, b2=-6.0")
print("  w_out1=-8.0, w_out2=8.5, b_out=-6.0")
print("\nDone!")
