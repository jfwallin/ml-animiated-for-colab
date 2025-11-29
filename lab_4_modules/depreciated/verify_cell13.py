import json

nb = json.load(open('lab_4_module_2_training_neural_network-v3.ipynb', 'r', encoding='utf-8'))

cell13 = ''.join(nb['cells'][13]['source'])

# Write to file to avoid encoding issues
with open('cell13_content.txt', 'w', encoding='utf-8') as f:
    f.write("CELL 13 CONTENT (After Cleanup)\n")
    f.write("="*70 + "\n\n")
    f.write(cell13)

print("Cell 13 content written to cell13_content.txt")
print(f"Cell 13 length: {len(cell13)} characters")
print(f"Contains 'Section 3': {'Section 3' in cell13}")
print(f"Contains 'Smart Initialization': {'Smart Initialization' in cell13}")
