import json

nb = json.load(open('lab_4_module_2_training_neural_network-v3.ipynb', 'r', encoding='utf-8'))

# Read all markdown cells
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        content = ''.join(cell['source'])
        print(f"\n{'='*70}")
        print(f"CELL {i}")
        print('='*70)
        print(content)
