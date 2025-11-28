import json

nb = json.load(open('lab_4_module_2_training_neural_network-v3.ipynb', 'r', encoding='utf-8'))

print(f"Total cells: {len(nb['cells'])}\n")
print("="*70)

# Show all cells with index, type, and first line
for i, cell in enumerate(nb['cells']):
    cell_type = cell['cell_type']
    if isinstance(cell['source'], list):
        first_line = cell['source'][0][:70] if cell['source'] else '(empty)'
    else:
        first_line = cell['source'][:70] if cell['source'] else '(empty)'

    print(f"Cell {i:2d} ({cell_type:8s}): {first_line}")

print("\n" + "="*70)
print("Looking for duplicates...")

# Check for duplicate content
seen_content = {}
for i, cell in enumerate(nb['cells']):
    content = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    content_hash = content[:100]  # First 100 chars as identifier

    if content_hash in seen_content:
        print(f"\n!!! Potential duplicate:")
        print(f"    Cell {seen_content[content_hash]} and Cell {i}")
        print(f"    Preview: {content_hash[:50]}...")
    else:
        seen_content[content_hash] = i
