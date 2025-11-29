import json

nb = json.load(open('lab_4_module_2_training_neural_network-v3.ipynb', 'r', encoding='utf-8'))

# Check cells around Section 3
for i in range(12, 16):
    cell = nb['cells'][i]
    content = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    print(f"\n{'='*70}")
    print(f"CELL {i} ({cell['cell_type']})")
    print('='*70)
    print(content)
