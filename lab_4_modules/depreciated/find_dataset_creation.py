import json

nb = json.load(open('lab_4_module_2_training_neural_network-v3.ipynb', 'r', encoding='utf-8'))

cell4 = ''.join(nb['cells'][4]['source'])

# Find where X_train, y_train are created
if 'X_train' in cell4:
    # Extract the relevant lines
    lines = cell4.split('\n')
    for i, line in enumerate(lines):
        if 'X_train' in line:
            print(f"Line {i}: {line}")
            # Show context (3 lines before and after)
            for j in range(max(0, i-3), min(len(lines), i+4)):
                print(f"  {j}: {lines[j]}")
            break
