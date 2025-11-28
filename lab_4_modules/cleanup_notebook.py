"""
Clean up the notebook:
1. Remove Cell 13 (old CONVERGENT_SEEDS - no longer used with multi-start)
2. Merge Cell 14 (Section 3 header) with Cell 15 (Smart Init) into one markdown cell
3. Remove Cell 19 if it's empty
"""

import json

# Read notebook
with open('lab_4_module_2_training_neural_network-v3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Before cleanup: {len(nb['cells'])} cells")

# Check Cell 19
cell19_content = ''.join(nb['cells'][19]['source']) if isinstance(nb['cells'][19]['source'], list) else nb['cells'][19]['source']
print(f"Cell 19 content: '{cell19_content.strip()}'")

# ===== Step 1: Remove Cell 13 (CONVERGENT_SEEDS) =====
print("\n[1] Removing Cell 13 (old CONVERGENT_SEEDS - no longer used)")
del nb['cells'][13]

# Now all indices shift down by 1
# Old Cell 14 is now Cell 13
# Old Cell 15 is now Cell 14

# ===== Step 2: Merge old Cell 14 and 15 into one markdown =====
print("[2] Merging Section 3 header with Smart Initialization")

# Get both cells (now at index 13 and 14)
section3_header = ''.join(nb['cells'][13]['source']) if isinstance(nb['cells'][13]['source'], list) else nb['cells'][13]['source']
smart_init = ''.join(nb['cells'][14]['source']) if isinstance(nb['cells'][14]['source'], list) else nb['cells'][14]['source']

# Combine them
merged_content = section3_header.rstrip() + '\n\n' + smart_init

# Update Cell 13 with merged content
nb['cells'][13]['source'] = merged_content.split('\n')
nb['cells'][13]['source'] = [line + '\n' if i < len(nb['cells'][13]['source']) - 1 else line
                              for i, line in enumerate(nb['cells'][13]['source'])]

# Delete old Cell 14 (Smart Init standalone)
del nb['cells'][14]

# ===== Step 3: Remove Cell 18 if empty (was Cell 19, now shifted) =====
# After two deletions, old Cell 19 is now Cell 17
if len(nb['cells']) > 17:
    cell_content = ''.join(nb['cells'][17]['source']) if isinstance(nb['cells'][17]['source'], list) else nb['cells'][17]['source']
    if not cell_content.strip():
        print("[3] Removing empty cell at end")
        del nb['cells'][17]
    else:
        print(f"[3] Last cell not empty, keeping it")

# Save
with open('lab_4_module_2_training_neural_network-v3.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nAfter cleanup: {len(nb['cells'])} cells")
print("\n[OK] Notebook cleaned up!")
print("  - Removed old CONVERGENT_SEEDS (Cell 13)")
print("  - Merged Section 3 header with Smart Initialization")
print("  - Removed empty cell at end (if present)")
