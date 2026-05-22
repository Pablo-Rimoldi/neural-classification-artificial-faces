import json

with open('notebooks/dl_test_byclass.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
total = len(cells)
print(f'Total cells: {total}')
print()

# Show last 40 cells
for i in range(max(0, total-40), total):
    c = cells[i]
    src = ''.join(c['source'])
    preview = src[:200].replace('\n', '\\n')
    print(f'Cell {i} [{c["cell_type"]}]: {preview}')
    print('---')
