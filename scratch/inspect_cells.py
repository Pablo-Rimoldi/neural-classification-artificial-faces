import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/dl_test_byclass.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

for idx in [24, 26, 28, 30, 32, 34, 36]:
    src = ''.join(cells[idx]['source'])
    print(f'===== CELL {idx} [{cells[idx]["cell_type"]}] =====')
    print(src)
    print()
