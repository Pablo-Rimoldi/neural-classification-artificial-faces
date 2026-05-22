import json

with open('notebooks/dl_test_byclass.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "Subject" in source or "SEX" in source or "PCA" in source or "channels" in source or "19" in source or "data_path" in source:
            print(f"--- Cell {i} ---")
            print(source[:500])
