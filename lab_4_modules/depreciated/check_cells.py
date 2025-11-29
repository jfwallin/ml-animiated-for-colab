import json

nb = json.load(open('lab_4_module_2_training_neural_network-v3.ipynb', 'r', encoding='utf-8'))

for i in range(13, 18):
    print(f'\n=== Cell {i} ({nb["cells"][i]["cell_type"]}) ===')
    src = ''.join(nb['cells'][i]['source']) if isinstance(nb['cells'][i]['source'], list) else nb['cells'][i]['source']
    print(src[:200])
