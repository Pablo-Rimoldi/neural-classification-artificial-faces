# `src/` Refactor to Mirror the Notebook + Unified Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor everything under `src/` into a clean, importable Python package whose code is a faithful, function-level mirror of the reference notebook `notebooks/Neural_classification_artificial_faces.ipynb`, then add `src/main.py` that runs the notebook's full pipeline (Raw → Preprocessing → Epochs → Tensor → Encoding/Decoding → ML → DL → XAI) as one unified, CLI-driven program.

**Architecture:** The notebook is the single source of truth ("the bible"). Each notebook section becomes one focused module. Modules are pure functions/classes that take explicit arguments and return values (no notebook-style globals). `main.py` wires the stages together with a fast-by-default / full-on-flags switch that mirrors the notebook's `QUICK_TEST` and `RUN_OPTUNA_SEARCH` toggles. All paths live in `src/config.py` and default to the local repo (`data/file_raw/`, `data/file_tensor/`, `results/`), not Google Drive.

**Tech Stack:** Python 3.11, numpy, pandas, scipy, scikit-learn, xgboost, torch, optuna (+ cmaes), matplotlib, seaborn; pytest for tests.

---

## Global Constraints

These apply to **every** task. Values copied verbatim from notebook cell 5 (`## 1. Global Constants`), with Colab/Drive paths replaced by local repo paths per the agreed decision.

- **DO NOT COMMIT OR PUSH.** The user has explicitly instructed: do not push or commit any code. Each task below ends with a `git add` + `git commit` step **for reference/workflow only** — an executing agent MUST skip those steps (or stage only, no commit) until the user explicitly authorizes committing. Treat the commit step as "checkpoint reached," not "run git commit."
- **The notebook is the source of truth.** Where this plan says *"PORT cell N"*, copy that cell's code verbatim into the target function body, adapting only: (a) names/signatures given in the task's **Interfaces** block, (b) the explicit **Divergences to fix** listed in the task, (c) replacing hardcoded paths with `config` constants. Do not "improve" behaviour beyond the listed divergences — fidelity to the notebook is the requirement.
- **No notebook globals.** Functions receive their inputs as arguments and return outputs. No reliance on module-level mutable state from other stages.
- **Local paths only.** No `from google.colab import drive`, no `drive.mount`, no `/content/drive/...`.
- **Determinism.** Respect the notebook seeds: `RANDOM_STATE = 42`, `DL_SEED = 3407`. Seeded runs must be reproducible.
- **Python version floor:** 3.11 (repo already targets 3.11; type hints use `list[str] | None` syntax).
- **Constants (verbatim, place in `src/config.py`):**
  - `S_FREQ = 512`, `TRIGGER_ROW = 75`, `STEP_MS = 1000 / S_FREQ`
  - `EEG_CHANNELS = ['O1', 'O2', 'PO9', 'PO10', 'TP7', 'TP8', 'P3', 'P4', 'AF3', 'AF4', 'AFF1h', 'AFF2h', 'AFF3h', 'AFF4h']`
  - `PCA_COLUMNS = ['PCA_Frontal', 'PCA_Parietal', 'PCA_Occipital', 'PCA_Temporal']`
  - `TIME_START_MS = 200`, `TIME_END_MS = 600`, `TARGET_EPOCH_SAMPLES = 205`, `ARTIFACT_THRESHOLD_UV = 80.0`, `MAX_BAD_CHANNELS = 2`
  - `N_SPLITS = 5`, `RIDGE_ALPHAS = [1, 10, 100, 1000, 10000, 100000, 1000000]`, `VAR_THRESHOLD = 1e-6`
  - `N250_START, N250_END = 220, 290`, `P300_START, P300_END = 280, 500`
  - `RANDOM_STATE = 42`, `K_CANDIDATES = [5, 10, 15, 'all']`, `SCALER_CHOICES = ['passthrough', MinMaxScaler(), RobustScaler()]`
  - `DL_SEED = 3407`
  - Paths (local): `RAW_DIR = data/file_raw`, `PARQUET_PATH = data/dataset_eeg_preprocessed.parquet`, `TENSOR_PATH = data/file_tensor/final_tensor.npz`, `BEST_HP_PATH = data/best_hyperparameters.json`, `RESULTS_DIR = results`.
- **Label maps (verbatim):** condition codes → binary class: `{'50AM','60AF'} → 0 (AI)`, `{'70RM','80RF'} → 1 (Real)`. Encoding/decoding string labels: `{'50AM':'AI','60AF':'AI','70RM':'Real','80RF':'Real'}`. Sex channel value: `1.0 if SubjectSEX == 'F' else 0.0`.

### Data availability (read before starting)

The only data present on disk is `data/file_raw/` (103 raw `.txt` files) and `data/best_hyperparameters.json`. The previously committed `data/file_tensor/{x,y,subjects}.npy`, `data/file_tensor.zip`, `scratch/*.py`, and `results/todo` were **intentionally deleted by the user** — do not restore them. The tensor is rebuilt from `data/file_raw/` by the preprocessing chain (Tasks 2–7); the `tensor` test fixture (Task 1) builds and caches it to `config.TENSOR_PATH` on first use. Some `D` entries also appear in `git status` for these removed files; that is expected — leave them.

---

## File Structure

New/renamed layout under `src/` (each file = one notebook responsibility):

```
src/
  __init__.py
  config.py                      # all constants + local paths (notebook cell 5)
  io/
    __init__.py
    raw_loader.py                # load raw .txt → DataFrame  (cell 8)
  preprocessing/
    __init__.py
    baseline.py                  # baseline correction         (cell 11)
    regions.py                   # region PCA + column select   (cell 13)
    windowing.py                 # 200–600 ms time filter       (cell 15)
    epochs.py                    # temporal cleaning + epochs   (cells 22, 24)  [FIX artifact rule]
    tensor.py                    # tensor build/save/load       (cell 27)       [FIX sex channel + typo]
  analysis/
    __init__.py
    encoding_decoding.py         # zscore, flat-removal, ANOVA, time-window, STA, ridge, AUC, logreg, plot (cells 31–46)
  models/
    __init__.py
    ml/
      __init__.py
      transforms.py              # IntraSubjectZScore + helpers (cell 50)
      prepare.py                 # load tensor, exclude subj 01, decimate, flatten (cell 52)
      models.py                  # pipelines + grids            (cell 54)
      evaluation.py              # nested CV, stage-2, perm test, wilcoxon (cells 56–63)
    dl/
      __init__.py
      dataset.py                 # EEGDataset + augmentations   (cell 66)
      architecture.py            # SpatialTemporalCNN           (cell 66)
      training.py                # train/eval/optuna/nested CV  (cell 69)
      xai.py                     # permutation importance + saliency (cell 73)
  main.py                        # CLI orchestrator (fast default; --full / --optuna)
tests/
  conftest.py                    # shared fixtures (tensor, tiny raw subset)
  test_io.py
  test_preprocessing.py
  test_epochs_tensor.py
  test_encoding_decoding.py
  test_ml.py
  test_dl.py
  test_main.py
```

**Superseded legacy files** (handled in Task 18, not deleted without user OK):
- `src/preprocessing/preprocessing_eeg.py` → split into `io/raw_loader.py`, `preprocessing/baseline.py`, `preprocessing/regions.py`, `preprocessing/windowing.py`. Note: the notebook has **no** `encode_labels` step — dropped on purpose (see Task 7 divergence).
- `src/preprocessing/tensor_creation.py` → `preprocessing/epochs.py` + `preprocessing/tensor.py` (fixes the PCA z-threshold artifact rule and the `subject_final` typo).
- `src/models/dl/stcnn_nested_cv.py` → `models/dl/{architecture,dataset,training}.py`.
- `src/models/ml/ML_first_draft/ML_01.ipynb`, `src/preprocessing/data_cleaner.py` → kept as-is (legacy/utility, not part of the unified pipeline).
- Empty placeholder files `src/architecture/ruolo3.txt`, `src/models/ml/ruolo4.txt`, `src/preprocessing/ruolo2.txt`, `src/models/dl/todo` → left untouched per the user's no-delete rule.

---

## Task 1: Package scaffolding, `config.py`, and pytest harness

**Files:**
- Create: `src/__init__.py`, `src/io/__init__.py`, `src/preprocessing/__init__.py`, `src/analysis/__init__.py`, `src/models/__init__.py`, `src/models/ml/__init__.py`, `src/models/dl/__init__.py` (all empty)
- Create: `src/config.py`
- Create: `tests/conftest.py`
- Modify: `requirements.txt` (add `pandas`, `scipy`, `xgboost`, `cmaes`, `pytest`, `pyarrow`)
- Create: `pytest.ini`

**Interfaces:**
- Produces: `src.config` module exposing every constant in **Global Constraints** as module attributes; path constants as `pathlib.Path`. Helper `config.results_path(name: str) -> Path` returning `RESULTS_DIR / name`.
- Produces (fixtures in `conftest.py`): `tensor` → `tuple[np.ndarray, np.ndarray, np.ndarray]` = `(X, y, subjects)`. **Build it from raw** (`data/file_raw/` is the only data present — the committed `data/file_tensor/` was intentionally deleted). The fixture is session-scoped: load the cached `config.TENSOR_PATH` npz if it exists, otherwise run the preprocessing chain (Tasks 2–7) and cache it. `repo_root` → `Path`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import numpy as np
from src import config

def test_constants_match_notebook():
    assert config.S_FREQ == 512
    assert config.EEG_CHANNELS[0] == 'O1' and len(config.EEG_CHANNELS) == 14
    assert config.PCA_COLUMNS == ['PCA_Frontal', 'PCA_Parietal', 'PCA_Occipital', 'PCA_Temporal']
    assert config.ARTIFACT_THRESHOLD_UV == 80.0 and config.MAX_BAD_CHANNELS == 2
    assert config.RANDOM_STATE == 42 and config.DL_SEED == 3407
    assert config.TARGET_EPOCH_SAMPLES == 205

def test_paths_are_local_not_colab():
    assert 'drive' not in str(config.RAW_DIR).lower()
    assert config.RAW_DIR.name == 'file_raw'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.config'`).

- [ ] **Step 3: Write `src/config.py` and all `__init__.py` files**

Port cell 5 constants verbatim into module-level attributes. Replace the Colab block with:

```python
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / 'data'
RAW_DIR      = DATA_DIR / 'file_raw'
PARQUET_PATH = DATA_DIR / 'dataset_eeg_preprocessed.parquet'
TENSOR_PATH  = DATA_DIR / 'file_tensor' / 'final_tensor.npz'
BEST_HP_PATH = DATA_DIR / 'best_hyperparameters.json'
RESULTS_DIR  = PROJECT_ROOT / 'results'
# ... then every numeric/list constant from cell 5 verbatim ...
def results_path(name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name
```

Note: `SCALER_CHOICES = ['passthrough', MinMaxScaler(), RobustScaler()]` (instantiated objects, exactly as the notebook).

- [ ] **Step 4: Write `conftest.py` fixtures + `pytest.ini`**

```python
# tests/conftest.py
import numpy as np, pytest
from pathlib import Path
from src import config

@pytest.fixture(scope='session')
def repo_root():
    return config.PROJECT_ROOT

@pytest.fixture(scope='session')
def tensor():
    # data/file_tensor/ was intentionally removed; build from raw (data/file_raw/).
    if config.TENSOR_PATH.exists():
        z = np.load(config.TENSOR_PATH, allow_pickle=True)
        return z['x'], z['y'], z['subjects']
    # Build via the preprocessing chain (available from Task 7 onward).
    from src.io.raw_loader import load_raw_files
    from src.preprocessing.baseline import apply_baseline_correction
    from src.preprocessing.regions import add_region_pca_features, select_columns
    from src.preprocessing.windowing import filter_time_window
    from src.preprocessing.epochs import build_epochs
    from src.preprocessing.tensor import build_tensor, save_tensor
    df = filter_time_window(select_columns(add_region_pca_features(
        apply_baseline_correction(load_raw_files()))))
    X, y, subjects = build_tensor(*build_epochs(df))
    save_tensor(X, y, subjects)          # cache for later test sessions
    return X, y, subjects
```

```ini
# pytest.ini
[pytest]
markers =
    slow: long-running (full nested CV / optuna / DL training)
addopts = -ra
```

- [ ] **Step 5: Run tests to verify pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit (checkpoint only — DO NOT run until user authorizes)**

```bash
git add src/__init__.py src/**/__init__.py src/config.py tests/conftest.py tests/test_config.py pytest.ini requirements.txt
git commit -m "refactor: add src package scaffolding, config, and pytest harness"
```

---

## Task 2: `io/raw_loader.py` — raw `.txt` loading + filename feature engineering

**Files:**
- Create: `src/io/raw_loader.py`
- Test: `tests/test_io.py`

**Interfaces:**
- Produces: `load_raw_files(folder: Path | str = config.RAW_DIR, sfreq: int = config.S_FREQ, trigger_row: int = config.TRIGGER_ROW) -> pd.DataFrame`. Returns the concatenated raw dataframe with added columns `SubjectID` (str, 2 chars), `SubjectSEX` (str, 1 char 'M'/'F'), `TargetCODE` (str, e.g. `'50AM'`), `TargetNATURE` (str 'R'/'A'), `Time_ms` (float). **SubjectSEX/TargetNATURE stay as strings** (no integer encoding — the notebook never encodes them).
- Consumes: nothing from earlier tasks.

**Source / Divergences:** PORT cell 8. The existing `preprocessing_eeg.load_files` is nearly identical and may be used as the base, but it must read from `config.RAW_DIR` and keep string metadata.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_io.py
from src.io.raw_loader import load_raw_files
from src import config

def test_load_raw_files_shape_and_metadata():
    df = load_raw_files()
    assert {'SubjectID','SubjectSEX','TargetCODE','TargetNATURE','Time_ms'} <= set(df.columns)
    assert df['SubjectSEX'].dtype == object              # strings, not encoded
    assert set(df['TargetNATURE'].unique()) <= {'R','A'}
    assert set(df['SubjectSEX'].unique()) <= {'M','F'}
    # time axis anchored on trigger row
    one = df[df['SubjectID'] == df['SubjectID'].iloc[0]]
    assert (one['Time_ms'] < 0).any() and (one['Time_ms'] > 0).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_io.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `load_raw_files`** porting cell 8 (glob `*.txt`, `pd.read_csv(sep=r'\s+', engine='python')`, slice filename for metadata, build `Time_ms`, `pd.concat`).

- [ ] **Step 4: Run test to verify pass**

Run: `python -m pytest tests/test_io.py -v`
Expected: PASS.

- [ ] **Step 5: Commit (checkpoint only)**

```bash
git add src/io/raw_loader.py tests/test_io.py
git commit -m "refactor: port raw txt loader from notebook cell 8"
```

---

## Task 3: `preprocessing/baseline.py` — baseline correction

**Files:**
- Create: `src/preprocessing/baseline.py`
- Test: extend `tests/test_preprocessing.py`

**Interfaces:**
- Produces: `apply_baseline_correction(df: pd.DataFrame) -> pd.DataFrame`. Subtracts the pre-stimulus (`Time_ms < 0`) per-channel mean from every numeric channel **except** `Time_ms` and `Trigger`. Returns a new dataframe (does not mutate input).
- Consumes: dataframe from `load_raw_files`.

**Source:** PORT cell 11 (logic identical to existing `apply_baseline_correction`, but return a copy and do not print describe()).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preprocessing.py
import numpy as np
from src.io.raw_loader import load_raw_files
from src.preprocessing.baseline import apply_baseline_correction
from src import config

def test_baseline_zeroes_prestimulus_mean():
    df = apply_baseline_correction(load_raw_files())
    pre = df[df['Time_ms'] < 0]
    means = pre[config.EEG_CHANNELS].mean().abs()
    assert (means < 1e-6).all()        # pre-stimulus mean ~0 after correction
```

- [ ] **Step 2: Run** `python -m pytest tests/test_preprocessing.py::test_baseline_zeroes_prestimulus_mean -v` → FAIL (ImportError).
- [ ] **Step 3: Implement** porting cell 11.
- [ ] **Step 4: Run** same command → PASS.
- [ ] **Step 5: Commit (checkpoint only)** `git add src/preprocessing/baseline.py tests/test_preprocessing.py && git commit -m "refactor: port baseline correction from cell 11"`

---

## Task 4: `preprocessing/regions.py` — region-based PCA + column selection

**Files:**
- Create: `src/preprocessing/regions.py`
- Test: extend `tests/test_preprocessing.py`

**Interfaces:**
- Produces:
  - `add_region_pca_features(df: pd.DataFrame) -> pd.DataFrame` — adds `PCA_Frontal/Parietal/Occipital/Temporal` columns (PC1 per region) and returns df with them concatenated.
  - `select_columns(df: pd.DataFrame) -> pd.DataFrame` — returns `EEG_CHANNELS + PCA_COLUMNS + ['Trigger','SubjectID','SubjectSEX','TargetCODE','TargetNATURE','Time_ms']` (only those present).
- Consumes: baseline-corrected dataframe.

**Source / Divergences:** PORT cell 13. Use the notebook's exact `regions` dict:
```python
regions = {'Frontal': ['F','AF','FC'], 'Parietal': ['P','CP'], 'Occipital': ['PO','O'], 'Temporal': ['T','TP']}
```
Region membership is **not** mutually exclusive (a channel matching multiple prefixes contributes to multiple regions) — preserve this; do not "fix" it. Remaining channels = float64 columns not in `EEG_CHANNELS` + metadata. Each region: `StandardScaler().fit_transform` then `PCA(n_components=1)`.

- [ ] **Step 1: Write the failing test**

```python
def test_region_pca_and_selection():
    from src.io.raw_loader import load_raw_files
    from src.preprocessing.baseline import apply_baseline_correction
    from src.preprocessing.regions import add_region_pca_features, select_columns
    from src import config
    df = select_columns(add_region_pca_features(apply_baseline_correction(load_raw_files())))
    for c in config.PCA_COLUMNS:
        assert c in df.columns
    assert set(config.EEG_CHANNELS) <= set(df.columns)
    assert df.shape[1] == 14 + 4 + 6        # 14 EEG + 4 PCA + 6 meta(+Trigger)
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** porting cell 13 (split into the two functions).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit (checkpoint only)** `git commit -m "refactor: port region PCA + column selection from cell 13"`

---

## Task 5: `preprocessing/windowing.py` — 200–600 ms time-window filter

**Files:**
- Create: `src/preprocessing/windowing.py`
- Test: extend `tests/test_preprocessing.py`

**Interfaces:**
- Produces: `filter_time_window(df: pd.DataFrame, time_min_ms: float = config.TIME_START_MS, time_max_ms: float = config.TIME_END_MS) -> pd.DataFrame`. Keeps rows with `time_min_ms <= Time_ms <= time_max_ms`. Returns a copy.
- Consumes: selected-columns dataframe.

**Source:** PORT cell 15 (drop the print of value_counts; that becomes an explicit log in `main.py`).

- [ ] **Step 1: Write the failing test**

```python
def test_time_window_bounds():
    from src.preprocessing.windowing import filter_time_window
    import pandas as pd, numpy as np
    df = pd.DataFrame({'Time_ms': np.array([-50, 0, 200, 400, 600, 700], float)})
    out = filter_time_window(df)
    assert out['Time_ms'].min() >= 200 and out['Time_ms'].max() <= 600
    assert len(out) == 3
```

- [ ] **Step 2: Run** → FAIL. **Step 3:** implement (cell 15). **Step 4: Run** → PASS.
- [ ] **Step 5: Commit (checkpoint only)** `git commit -m "refactor: port time-window filter from cell 15"`

---

## Task 6: `preprocessing/epochs.py` — temporal cleaning + epoch builder (FIX artifact rule)

**Files:**
- Create: `src/preprocessing/epochs.py`
- Test: `tests/test_epochs_tensor.py`

**Interfaces:**
- Produces:
  - `apply_temporal_cleaning(epoch_data: np.ndarray) -> np.ndarray` — 4th-order Butterworth LP @ 40 Hz (`filtfilt`, `axis=0`), linear `detrend(axis=0)`, subtract mean of first 25 samples. Input/Output shape `(n_times, n_ch)`.
  - `build_epochs(df: pd.DataFrame) -> tuple[list, list, list, list, list]` returning `(EEG_list, PCA_list, y_list, sub_list, subject_sex_list)`. Each `EEG_list[i]` has shape `(14, TARGET_EPOCH_SAMPLES)`, each `PCA_list[i]` shape `(4, TARGET_EPOCH_SAMPLES)`; `y_list[i]` is the `TargetCODE` string; `subject_sex_list[i]` is `'M'`/`'F'`.
- Consumes: time-filtered dataframe.

**Source / Divergences (IMPORTANT):** PORT cells 22 + 24. The existing `tensor_creation.create_lists` **diverges and must be corrected to match the notebook**:
- ❌ Old rule: `if np.max(np.abs(cleaned_pca_data)) < np.std(cleaned_pca_data) * Z_THRESHOLD:` (PCA z-score gate) — **remove entirely**.
- ✅ Notebook rule (cell 24): after enforcing length, reject on **EEG amplitude**:
  ```python
  channel_max = np.max(np.abs(cleaned_eeg), axis=0)
  if np.sum(channel_max > ARTIFACT_THRESHOLD_UV) > MAX_BAD_CHANNELS:
      continue
  ```
- Group by `['SubjectID', 'TargetCODE']`, sort by `Time_ms`, `min_samples = max(int(0.05*S_FREQ), 16)`, truncate to `TARGET_EPOCH_SAMPLES` (skip if shorter), append `cleaned_eeg.T` / `cleaned_pca.T`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_epochs_tensor.py
import numpy as np
from src import config

def test_temporal_cleaning_shape_and_baseline():
    from src.preprocessing.epochs import apply_temporal_cleaning
    rng = np.random.default_rng(0)
    x = rng.standard_normal((205, 14)) + 5.0
    out = apply_temporal_cleaning(x)
    assert out.shape == (205, 14)
    assert np.abs(out[:25].mean(axis=0)).max() < 1e-6   # first-25 baseline removed

def test_build_epochs_shapes_and_string_labels():
    from src.io.raw_loader import load_raw_files
    from src.preprocessing.baseline import apply_baseline_correction
    from src.preprocessing.regions import add_region_pca_features, select_columns
    from src.preprocessing.windowing import filter_time_window
    from src.preprocessing.epochs import build_epochs
    df = filter_time_window(select_columns(add_region_pca_features(
        apply_baseline_correction(load_raw_files()))))
    eeg, pca, y, sub, sex = build_epochs(df)
    assert len(eeg) == len(pca) == len(y) == len(sub) == len(sex)
    assert eeg[0].shape == (14, config.TARGET_EPOCH_SAMPLES)
    assert pca[0].shape == (4, config.TARGET_EPOCH_SAMPLES)
    assert set(sex) <= {'M', 'F'}                       # strings preserved
    assert all(isinstance(v, str) and len(v) == 4 for v in y)   # TargetCODE strings
```

- [ ] **Step 2: Run** `python -m pytest tests/test_epochs_tensor.py -v` → FAIL.
- [ ] **Step 3: Implement** porting cells 22 + 24 with the corrected artifact rule above.
- [ ] **Step 4: Run** → PASS. Sanity: `len(eeg)` should be ≈103 (the notebook's reported epoch count) and never gated by the removed PCA rule.
- [ ] **Step 5: Commit (checkpoint only)** `git commit -m "refactor: port epoch builder, replace PCA z-gate with 80uV/2-bad-channel rule (cells 22,24)"`

---

## Task 7: `preprocessing/tensor.py` — tensor construction, save, load (FIX sex channel + typo)

**Files:**
- Create: `src/preprocessing/tensor.py`
- Test: extend `tests/test_epochs_tensor.py`

**Interfaces:**
- Produces:
  - `build_tensor(eeg_list, pca_list, y_list, sub_list, sex_list) -> tuple[np.ndarray, np.ndarray, np.ndarray]` → `(X, y, subjects)` where `X` is `(n_epochs, 19, TARGET_EPOCH_SAMPLES)` float64, `y` is array of `TargetCODE` strings, `subjects` is array of `SubjectID` strings. Channel order: 14 EEG, 4 PCA, then 1 **sex channel** = constant row of `1.0 if sex == 'F' else 0.0`.
  - `save_tensor(X, y, subjects, path: Path = config.TENSOR_PATH) -> None` → `np.savez(path, x=X, y=y, subjects=subjects)` (creates parent dir).
  - `load_tensor(path: Path = config.TENSOR_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray]`.
- Consumes: lists from `build_epochs`.

**Source / Divergences (IMPORTANT):** PORT cell 27. Fix the two bugs in the old `tensor_creation.create_tensor`:
- ❌ Old: `new_channel_data = np.full((1, ...), subject_sex_value)` using the raw/encoded sex value. ✅ Notebook: `sex_val = 1.0 if subject_sex_list[j] == 'F' else 0.0`.
- ❌ Old: `return x_final, y_final, subject_final` — undefined name (`subject_final`). ✅ Return `subjects_final`.
- Save keys must be `x`, `y`, `subjects` (the ML and DL stages load exactly these keys).

- [ ] **Step 1: Write the failing test**

```python
def test_build_and_roundtrip_tensor(tmp_path):
    import numpy as np
    from src.preprocessing.tensor import build_tensor, save_tensor, load_tensor
    eeg = [np.zeros((14, 205)), np.ones((14, 205))]
    pca = [np.zeros((4, 205)),  np.ones((4, 205))]
    X, y, subj = build_tensor(eeg, pca, ['50AM','70RF'.replace('RF','RM')], ['01','02'], ['F','M'])
    assert X.shape == (2, 19, 205) and X.dtype == np.float64
    assert (X[0, 18, :] == 1.0).all()      # subject 0 is 'F' → sex row = 1.0
    assert (X[1, 18, :] == 0.0).all()      # subject 1 is 'M' → sex row = 0.0
    p = tmp_path / 'final_tensor.npz'
    save_tensor(X, y, subj, p)
    X2, y2, s2 = load_tensor(p)
    assert np.array_equal(X, X2) and np.array_equal(subj, s2)
```

- [ ] **Step 2: Run** → FAIL. **Step 3:** implement porting cell 27 with both fixes. **Step 4: Run** → PASS.
- [ ] **Step 5:** Add an integration test (real data) asserting the full chain produces a `(n,19,205)` tensor with `n` ≈ 103 and 4 distinct `TargetCODE` values.

```python
def test_full_preprocessing_chain_tensor():
    from src.io.raw_loader import load_raw_files
    from src.preprocessing.baseline import apply_baseline_correction
    from src.preprocessing.regions import add_region_pca_features, select_columns
    from src.preprocessing.windowing import filter_time_window
    from src.preprocessing.epochs import build_epochs
    from src.preprocessing.tensor import build_tensor
    df = filter_time_window(select_columns(add_region_pca_features(
        apply_baseline_correction(load_raw_files()))))
    X, y, subj = build_tensor(*build_epochs(df))
    assert X.ndim == 3 and X.shape[1] == 19 and X.shape[2] == 205
    assert len(set(y.tolist())) == 4
```

- [ ] **Step 6: Commit (checkpoint only)** `git commit -m "refactor: port tensor build/save/load, fix sex channel + subjects typo (cell 27)"`

---

## Task 8: `analysis/encoding_decoding.py` — mass-univariate encoding/decoding (§7)

**Files:**
- Create: `src/analysis/encoding_decoding.py`
- Test: `tests/test_encoding_decoding.py`

**Interfaces:**
- Produces: `run_encoding_decoding(X: np.ndarray, y_codes: np.ndarray, subjects: np.ndarray, *, make_plot: bool = True, save_path: Path | None = None) -> dict` returning keys: `f_map` `(n_good, n_times)`, `sta` (dict label→`(n_good,n_times)`), `K` `(n_classes,n_good,n_times)`, `R2_train` (float), `r2_cv` `(n_good,)`, `auc_map` `(n_classes,n_good,n_times)`, `decoding_acc_mean` (float), `decoding_acc_std` (float), `good_ch` (np.ndarray), `n_good` (int), `time_axis` `(n_times,)`, `unique_labels` (np.ndarray). If `make_plot`, also renders the cell-45 figure and (when `save_path`) saves it.
- Consumes: tensor `(X, y, subjects)` from `load_tensor`.

**Source / Divergences:** PORT cells 31, 34, 37, 40, 43, 45, 46 in order, wrapped in the single function. Internal steps:
1. zscore over `axis=2`, NaN→0; flat-channel removal via `VAR_THRESHOLD` → `good_ch`, `X_clean` (cell 31).
2. map codes→`'AI'/'Real'` (cell 34 `label_map`); ANOVA `f_oneway` per `(channel,time)` → `f_map_full` (cell 34).
3. `ms_to_tp` window crop to N250_START..P300_END (cell 37).
4. STA per class; one-hot `S`; `RidgeCV(alphas=RIDGE_ALPHAS, cv=N_SPLITS)` → `K`, `R2_train`; per-channel 5-fold CV R² (cell 40).
5. AUC map via `roc_auc_score` per `(class,channel,time)`; `LogisticRegressionCV` decoded with `GroupKFold(groups=subjects)` → `scores` (cell 43).
6. Build the figure (cell 45) and the summary (cell 46). Save to `config.results_path('encoding_decoding.png')` when requested (replaces the notebook's bare `plt.savefig`/`plt.show`).

- [ ] **Step 1: Write the failing test** (uses the committed tensor fixture; `make_plot=False` for speed)

```python
# tests/test_encoding_decoding.py
import numpy as np

def test_encoding_decoding_outputs(tensor):
    from src.analysis.encoding_decoding import run_encoding_decoding
    X, y, subj = tensor
    out = run_encoding_decoding(X, y, subj, make_plot=False)
    n_good = out['n_good']
    assert out['f_map'].shape[0] == n_good
    assert out['K'].shape == (len(out['unique_labels']), n_good, out['f_map'].shape[1])
    assert out['auc_map'].shape == out['K'].shape
    assert 0.0 <= out['decoding_acc_mean'] <= 1.0
    # the constant sex channel (index 18) is flat → excluded by VAR_THRESHOLD
    assert 18 not in out['good_ch'].tolist()
    assert set(out['unique_labels'].tolist()) == {'AI', 'Real'}
```

- [ ] **Step 2: Run** `python -m pytest tests/test_encoding_decoding.py -v` → FAIL.
- [ ] **Step 3: Implement** porting cells 31–46 (figure code guarded by `make_plot`).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit (checkpoint only)** `git commit -m "refactor: port encoding/decoding analysis (cells 31-46)"`

---

## Task 9: `models/ml/transforms.py` — IntraSubjectZScore + ML helpers (§8 helpers)

**Files:**
- Create: `src/models/ml/transforms.py`
- Test: `tests/test_ml.py`

**Interfaces (exact names — later tasks depend on them):**
- `class IntraSubjectZScore(BaseEstimator, TransformerMixin)` — expects column 0 = subject id, z-scores remaining columns per subject; `transform` returns features without the id column.
- `decimate_to_odd(X: np.ndarray, factor: int) -> np.ndarray` — slices `[:, :, ::factor]`, drops last tp if even.
- `spatial_flatten(X, y, subjects) -> tuple[X_flat, y_flat, subjects_flat, trial_ids]`.
- `aggregate_trials(y_pred_tp, y_score_tp, y_true_tp, trial_ids) -> (y_true, y_pred, y_score)` (majority vote, mean score).
- `groups_disjoint(train_groups, test_groups) -> bool`
- `permute_trial_labels(y_tp, trial_ids, rng) -> np.ndarray`
- `summarize_selected_params(param_dicts) -> dict`
- `representative_params(param_dicts) -> dict`
- `permutation_p_value(observed, null_distribution) -> float`

**Source:** PORT cell 50 verbatim (all functions/classes above).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ml.py
import numpy as np

def test_intra_subject_zscore_drops_id_and_normalizes():
    from src.models.ml.transforms import IntraSubjectZScore
    X = np.array([[0., 1., 2.], [0., 3., 6.], [1., 10., 0.], [1., 20., 0.]])
    out = IntraSubjectZScore().fit_transform(X)
    assert out.shape == (4, 2)                        # id column removed
    assert abs(out[:2, 0].mean()) < 1e-9              # per-subject zero mean

def test_decimate_to_odd():
    from src.models.ml.transforms import decimate_to_odd
    X = np.zeros((3, 19, 205))
    assert decimate_to_odd(X, 2).shape[2] % 2 == 1

def test_aggregate_trials_majority_vote():
    from src.models.ml.transforms import aggregate_trials
    yt, yp, ys = aggregate_trials([0,0,1],[0.1,0.2,0.9],[0,0,0],[7,7,7])
    assert yt.tolist()==[0] and yp.tolist()==[0]

def test_permutation_p_value():
    from src.models.ml.transforms import permutation_p_value
    assert permutation_p_value(1.0, [0.0]*9) == (0+1)/(9+1)
```

- [ ] **Step 2: Run** → FAIL. **Step 3:** port cell 50. **Step 4: Run** → PASS.
- [ ] **Step 5: Commit (checkpoint only)** `git commit -m "refactor: port ML transforms and helpers (cell 50)"`

---

## Task 10: `models/ml/prepare.py` — tensor → ML design matrix (§8 preprocessing)

**Files:**
- Create: `src/models/ml/prepare.py`
- Test: extend `tests/test_ml.py`

**Interfaces:**
- Produces: `prepare_ml_data(tensor_path: Path = config.TENSOR_PATH, decimation_factor: int = 2, exclude_subjects=('01', 1)) -> dict` with keys `X_flat` `(n_rows, 1+n_channels)` (col 0 = subject code), `y_flat`, `subjects_flat`, `trial_ids`, `y_ml` `(n_trials,)`, `subjects_ml`, `n_trials`, `n_timepoints`. Drops the sex channel (keep channels `:18`), excludes subject 01, maps codes→`0=AI,1=Real`.
- Consumes: saved tensor; `transforms.decimate_to_odd`, `transforms.spatial_flatten`.

**Source / Divergences:** PORT cell 52. Note the notebook slices `X[:, :18, :]` only `if shape[1]==19`; keep that guard. `EXCLUDE = {'01', 1}`. Binary map: `np.where(np.isin(codes, ['50AM','60AF']), 0, 1)`.

- [ ] **Step 1: Write the failing test**

```python
def test_prepare_ml_data(tmp_path, tensor):
    import numpy as np
    from src.preprocessing.tensor import save_tensor
    from src.models.ml.prepare import prepare_ml_data
    X, y, subj = tensor
    p = tmp_path / 'final_tensor.npz'; save_tensor(X, y, subj, p)
    d = prepare_ml_data(p, decimation_factor=2)
    assert d['X_flat'].shape[1] == 1 + 18              # id col + 18 channels (sex dropped)
    assert set(np.unique(d['y_ml']).tolist()) <= {0, 1}
    assert '01' not in set(d['subjects_ml'].tolist())  # subject 01 excluded
    assert d['n_timepoints'] % 2 == 1                  # odd after decimate_to_odd
```

- [ ] **Step 2: Run** → FAIL. **Step 3:** port cell 52. **Step 4: Run** → PASS.
- [ ] **Step 5: Commit (checkpoint only)** `git commit -m "refactor: port ML data preparation (cell 52)"`

---

## Task 11: `models/ml/models.py` — leakage-safe pipelines + grids

**Files:**
- Create: `src/models/ml/models.py`
- Test: extend `tests/test_ml.py`

**Interfaces (exact):**
- `safe_mcc(y_true, y_pred) -> float`
- `acc_scorer` (`make_scorer(accuracy_score)`)
- `get_models_and_grids() -> list[tuple[str, Pipeline, dict, int]]` — the 8 entries: `LDA, LinearSVC, LinearSVC_Cal, LogReg_L1, LogReg_EN, SGD, XGBoost, Dummy` (keep `SVC_RBF`/`RandomForest` commented-out as in the notebook).
- `get_focused_grid(name: str) -> tuple[dict, int]`
- Internal: `_pipe(clf)` builds `Pipeline([('subjz', IntraSubjectZScore()), ('scaler', StandardScaler()), ('variance', VarianceThreshold(0.0)), ('selector', SelectKBest(f_classif)), ('clf', clf)])`; `_cal(estimator, method='sigmoid')`; `CALIB_CV = 3`.

**Source:** PORT cell 54 verbatim (imports `IntraSubjectZScore` from `transforms`, constants from `config`).

- [ ] **Step 1: Write the failing test**

```python
def test_models_catalogue():
    from src.models.ml.models import get_models_and_grids, get_focused_grid
    models = get_models_and_grids()
    names = [m[0] for m in models]
    assert names == ['LDA','LinearSVC','LinearSVC_Cal','LogReg_L1','LogReg_EN','SGD','XGBoost','Dummy']
    for _, pipe, grid, n_iter in models:
        assert [s[0] for s in pipe.steps] == ['subjz','scaler','variance','selector','clf']
        assert isinstance(n_iter, int)
    grid, n_iter = get_focused_grid('XGBoost')
    assert isinstance(grid, dict) and n_iter >= 1
```

- [ ] **Step 2: Run** → FAIL. **Step 3:** port cell 54. **Step 4: Run** → PASS.
- [ ] **Step 5: Commit (checkpoint only)** `git commit -m "refactor: port ML pipelines and hyperparameter grids (cell 54)"`

---

## Task 12: `models/ml/evaluation.py` — nested CV, stage-2, permutation, Wilcoxon

**Files:**
- Create: `src/models/ml/evaluation.py`
- Test: extend `tests/test_ml.py`

**Interfaces:**
- `run_nested_cv(models, data: dict, *, quick_test=True, n_outer_folds=5, n_outer_repeats=5, n_inner_splits=5, inner_test_size=0.20, collect_params=False, verbose=True) -> tuple` — returns `(results, pooled[, chosen])`. `data` is the dict from `prepare_ml_data`. When `quick_test`, internally clamps to `2,1,3` folds and `n_iter≤3` exactly as cell 56.
- `select_best(results) -> str` (highest mean acc).
- `run_stage2(best_name, models, data, **kw) -> dict` (cell 59 logic; returns adopted grid/config + chosen params).
- `permutation_test(best_pipe, data, n_permutations=1000, n_outer_folds=5, random_state=42) -> dict` (cell 62; keys `observed_acc, observed_mcc, p_acc, p_mcc, null_acc, null_mcc`).
- `wilcoxon_vs_best(results, best_name) -> pd.DataFrame` (cell 63, Bonferroni).
- `plot_ml_performance(results, pooled, order, best_name, save_path=None)` (cell 60 → save to `config.results_path('spatial_pipeline_performance.png')`).

**Source / Divergences:** PORT cells 56, 57, 59, 60, 62, 63. Replace module-level `X_flat/y_flat/subjects_flat/trial_ids` globals with lookups into the `data` dict passed in. Keep `QUICK_TEST` semantics but as the `quick_test` parameter (no module global). `inner_cv = GroupShuffleSplit(...)` built inside the function from params.

- [ ] **Step 1: Write the failing test** (quick mode, marked slow-ish but ~seconds)

```python
def test_nested_cv_quick_runs(tmp_path, tensor):
    from src.preprocessing.tensor import save_tensor
    from src.models.ml.prepare import prepare_ml_data
    from src.models.ml.models import get_models_and_grids
    from src.models.ml.evaluation import run_nested_cv, select_best
    X, y, subj = tensor
    p = tmp_path / 'final_tensor.npz'; save_tensor(X, y, subj, p)
    data = prepare_ml_data(p)
    models = [m for m in get_models_and_grids() if m[0] in ('LDA', 'Dummy')]
    results, pooled = run_nested_cv(models, data, quick_test=True, verbose=False)
    assert set(results) == {'LDA', 'Dummy'}
    for name in results:
        assert 'acc' in results[name] and len(results[name]['acc']) >= 1
    assert select_best(results) in {'LDA', 'Dummy'}
```

- [ ] **Step 2: Run** → FAIL. **Step 3:** port cells 56–63. **Step 4: Run** → PASS.
- [ ] **Step 5:** Add a `@pytest.mark.slow` test running the **full** 8-model `run_nested_cv(quick_test=False)` + `permutation_test(n_permutations=50)` to confirm wiring (kept out of the default run via the `slow` marker).
- [ ] **Step 6: Commit (checkpoint only)** `git commit -m "refactor: port ML nested CV, stage-2, permutation + Wilcoxon (cells 56-63)"`

---

## Task 13: `models/dl/dataset.py` — EEGDataset + augmentations

**Files:**
- Create: `src/models/dl/dataset.py`
- Test: `tests/test_dl.py`

**Interfaces (exact):**
- `temporal_jitter(x, max_shift=10)`, `channel_dropout(x, p_drop=0.15)`, `gaussian_noise(x, sigma=0.04)`, `amplitude_scale(x, lo=0.85, hi=1.15)` — numpy in/out, single-epoch `(channels, time)`.
- `class EEGDataset(Dataset)` — `__init__(self, x, y, augment=False)`; `__getitem__` returns `(torch.FloatTensor (C,T), torch.long scalar)`; applies the 4 augmentations only when `augment`.

**Source:** PORT the augmentation fns + `EEGDataset` from cell 66.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dl.py
import numpy as np, torch

def test_eeg_dataset_shapes_and_augment_flag():
    from src.models.dl.dataset import EEGDataset
    x = np.random.randn(5, 19, 205).astype('float32')
    y = np.array([0,1,0,1,0])
    xb, yb = EEGDataset(x, y, augment=False)[0]
    assert tuple(xb.shape) == (19, 205) and yb.dtype == torch.int64
    xb2, _ = EEGDataset(x, y, augment=True)[0]
    assert tuple(xb2.shape) == (19, 205)
```

- [ ] **Step 2: Run** → FAIL. **Step 3:** port cell 66 (dataset part). **Step 4: Run** → PASS.
- [ ] **Step 5: Commit (checkpoint only)** `git commit -m "refactor: port EEGDataset + augmentations (cell 66)"`

---

## Task 14: `models/dl/architecture.py` — SpatialTemporalCNN

**Files:**
- Create: `src/models/dl/architecture.py`
- Test: extend `tests/test_dl.py`

**Interfaces (exact):**
- `class SpatialTemporalCNN(nn.Module)` with `__init__(self, channels=19, temp_filters=32, kernel_size=16, n_layers=1, adj_init="identity", adj_norm="sigmoid", dropout=0.4, classes=2)`; `forward(x: (B, channels, T)) -> (B, classes)`. Learnable `adj` `(channels, channels)`, Conv1d temporal filters, optional 2nd residual layer, `AdaptiveAvgPool1d(8)`, dropout, `Linear(temp_filters*8, classes)`.

**Source:** PORT the `SpatialTemporalCNN` class from cell 66 verbatim.

- [ ] **Step 1: Write the failing test**

```python
def test_stcnn_forward_shape_and_determinism():
    import torch
    from src.models.dl.architecture import SpatialTemporalCNN
    torch.manual_seed(0); m = SpatialTemporalCNN(channels=19, classes=2).eval()
    x = torch.randn(4, 19, 205)
    with torch.no_grad():
        out1 = m(x); out2 = m(x)
    assert tuple(out1.shape) == (4, 2)
    assert torch.allclose(out1, out2)        # eval-mode deterministic
```

- [ ] **Step 2: Run** → FAIL. **Step 3:** port the class. **Step 4: Run** → PASS.
- [ ] **Step 5: Commit (checkpoint only)** `git commit -m "refactor: port SpatialTemporalCNN architecture (cell 66)"`

---

## Task 15: `models/dl/training.py` — training loop, optuna search, nested CV

**Files:**
- Create: `src/models/dl/training.py`
- Test: extend `tests/test_dl.py`

**Interfaces (exact):**
- `reset_all_seeds(seed)`, `mixup_batch(x, y, alpha=0.2)`, `mixup_criterion(crit, pred, ya, yb, lam)`, `build_scheduler(optimizer, scheduler_type, epochs)`, `train_one_epoch(model, loader, optimizer, criterion, device, mixup_alpha=0.2)`, `evaluate(model, loader, criterion, device)`, `collect_preds(model, loader, device)`, `hp_space(trial)`, `model_factory(hp)`.
- `load_dl_data(tensor_path=config.TENSOR_PATH) -> dict` — loads tensor, maps codes→`0/1` via `{"50AM":0,"60AF":0,"70RM":1,"80RF":1}`, normalises per-epoch `(X - mean)/(std+1e-8)` over axes `(1,2)`; returns `X_norm, y, subjects`.
- `run_dl_nested_cv(data, *, run_optuna=False, best_hp_path=config.BEST_HP_PATH, n_folds=5, n_inner_trials=150, device=None, seed=config.DL_SEED) -> dict` — GroupKFold outer loop; when `run_optuna`, CMA-ES inner search (cell 69 `objective`); else load precomputed HPs (from `best_hp_path`, falling back to the embedded `hardcoded_hps` list from cell 69). Returns `outer_accs, fold_aucs, all_preds, all_labels, all_probs, fold_models, fold_te_indices`.
- `summarize_dl_metrics(result: dict) -> dict` (cell 71: mean acc/auc, binomial test vs 0.5, classification report).

**Source / Divergences:** PORT cells 69 + 71. Key change: the notebook reads `hardcoded_hps` inline and `device`/data from globals. Here: pass `device` and `data` as args; read precomputed HPs from `config.BEST_HP_PATH` (the repo's `data/best_hyperparameters.json`) when present, else use the inline list as fallback. `RUN_OPTUNA_SEARCH` becomes the `run_optuna` parameter (default `False`). Skip the `!pip install cmaes` magic (declare `cmaes` in requirements instead).

- [ ] **Step 1: Write the failing test** (tiny, CPU, 2 epochs — fast)

```python
def test_dl_training_smoke(tmp_path, tensor):
    import torch, numpy as np
    from src.preprocessing.tensor import save_tensor
    from src.models.dl.training import load_dl_data, model_factory, train_one_epoch
    from src.models.dl.dataset import EEGDataset
    from torch.utils.data import DataLoader
    X, y, subj = tensor
    p = tmp_path / 'final_tensor.npz'; save_tensor(X, y, subj, p)
    d = load_dl_data(p)
    assert d['X_norm'].shape[1] == 19 and set(np.unique(d['y']).tolist()) <= {0,1}
    hp = {'temp_filters':16,'kernel_size':8,'n_layers':1,'adj_init':'identity',
          'adj_norm':'sigmoid','dropout':0.3}
    m = model_factory(hp)
    ds = EEGDataset(d['X_norm'][:16], d['y'][:16], augment=True)
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    train_one_epoch(m, dl, opt, torch.nn.CrossEntropyLoss(), torch.device('cpu'))
```

- [ ] **Step 2: Run** → FAIL. **Step 3:** port cells 69+71 with the arg-based device/data and HP loading. **Step 4: Run** → PASS.
- [ ] **Step 5:** Add `@pytest.mark.slow` test running `run_dl_nested_cv(run_optuna=False)` end-to-end on the real tensor (precomputed HPs) and asserting `0 <= mean acc <= 1` and 5 folds.
- [ ] **Step 6: Commit (checkpoint only)** `git commit -m "refactor: port DL training + nested CV + metrics (cells 69,71)"`

---

## Task 16: `models/dl/xai.py` — permutation importance + gradient saliency

**Files:**
- Create: `src/models/dl/xai.py`
- Test: extend `tests/test_dl.py`

**Interfaces:**
- `channel_names() -> list[str]` (the 19 names from cell 73, ending `'FaceSEX'`).
- `permutation_importance(fold_models, fold_te_indices, X_norm, y, device, n_repeats=20) -> tuple[np.ndarray, np.ndarray]` → `(importances (19,), std (19,))`.
- `gradient_saliency(fold_models, fold_te_indices, X_norm, y, device) -> tuple[np.ndarray, np.ndarray]` → `(saliency_map (19, n_times), std_map)`.
- `plot_xai(perm_importances, perm_std, saliency_map, *, n_folds, save_path=None) -> None` (cell 73 figure → `config.results_path('xai_analysis.png')`).

**Source:** PORT cell 73 (the permutation-importance and saliency computation blocks + the figure). Replace `np`/`device`/`fold_models` globals with parameters.

- [ ] **Step 1: Write the failing test** (build 2 tiny trained models, check shapes)

```python
def test_xai_shapes(tensor):
    import numpy as np, torch
    from src.models.dl.architecture import SpatialTemporalCNN
    from src.models.dl.xai import permutation_importance, gradient_saliency, channel_names
    X, y, subj = tensor
    Xn = ((X - X.mean((1,2),keepdims=True)) / (X.std((1,2),keepdims=True)+1e-8)).astype('float32')
    yb = np.isin(y, ['70RM','80RF']).astype('int64')
    models = [SpatialTemporalCNN(channels=19).eval() for _ in range(2)]
    te = [np.arange(0,20), np.arange(20,40)]
    imp, std = permutation_importance(models, te, Xn, yb, torch.device('cpu'), n_repeats=2)
    sal, _ = gradient_saliency(models, te, Xn, yb, torch.device('cpu'))
    assert imp.shape == (19,) and len(channel_names()) == 19
    assert sal.shape[0] == 19 and sal.shape[1] == Xn.shape[2]
```

- [ ] **Step 2: Run** → FAIL. **Step 3:** port cell 73. **Step 4: Run** → PASS.
- [ ] **Step 5: Commit (checkpoint only)** `git commit -m "refactor: port DL XAI (permutation importance + saliency) (cell 73)"`

---

## Task 17: `src/main.py` — unified CLI pipeline

**Files:**
- Create: `src/main.py`
- Test: `tests/test_main.py`

**Interfaces:**
- `run_pipeline(*, full: bool = False, run_optuna: bool = False, skip_dl: bool = False, make_plots: bool = True, rebuild_tensor: bool = True) -> dict` — executes the full notebook flow and returns a summary dict (tensor shape, decoding accuracy, best ML model + acc, DL mean acc/AUC + binomial p). Stages, in order:
  1. **Preprocess + tensor** (Tasks 2–7): if `rebuild_tensor` or tensor file missing → `load_raw_files → baseline → regions → select → window → epochs → tensor → save_tensor`; else `load_tensor`.
  2. **Encoding/Decoding** (Task 8) → save `results/encoding_decoding.png`.
  3. **ML** (Tasks 10–12): `prepare_ml_data`, `run_nested_cv(quick_test=not full)`, `select_best`, `run_stage2`, `plot_ml_performance`, `permutation_test(n_permutations=1000 if full else 50)`, `wilcoxon_vs_best`.
  4. **DL** (Tasks 15–16) unless `skip_dl`: `load_dl_data`, `run_dl_nested_cv(run_optuna=run_optuna)`, `summarize_dl_metrics`, `permutation_importance`, `gradient_saliency`, `plot_xai`.
- `main()` — `argparse` CLI: `--full` (sets `full=True`), `--optuna` (sets `run_optuna=True`), `--skip-dl`, `--no-plots`, `--use-cached-tensor`. Maps to `run_pipeline(...)`. **Defaults: fast mode** (`full=False`, `run_optuna=False`) mirroring the notebook's `QUICK_TEST=True` / `RUN_OPTUNA_SEARCH=False`.

**Source:** Compose the stage functions; print a section banner per stage matching the notebook's headers (`## 2. Raw Data`, etc.).

- [ ] **Step 1: Write the failing test** (fast path, DL skipped, cached tensor, no plots)

```python
# tests/test_main.py
def test_run_pipeline_fast_smoke(tmp_path, tensor, monkeypatch):
    import numpy as np
    from src import config
    from src.preprocessing.tensor import save_tensor
    # point config at a temp tensor so we don't rebuild from raw
    p = tmp_path / 'final_tensor.npz'
    save_tensor(*tensor, p)
    monkeypatch.setattr(config, 'TENSOR_PATH', p)
    from src.main import run_pipeline
    summary = run_pipeline(full=False, run_optuna=False, skip_dl=True,
                           make_plots=False, rebuild_tensor=False)
    assert 'decoding_acc_mean' in summary and 'best_ml_model' in summary
    assert summary['tensor_shape'][1] == 19
```

- [ ] **Step 2: Run** `python -m pytest tests/test_main.py -v` → FAIL.
- [ ] **Step 3: Implement** `run_pipeline` + `main()` argparse. Ensure modules read `config.TENSOR_PATH` dynamically (so the monkeypatch works) — import `config` and reference `config.TENSOR_PATH` at call time, not as a default bound at import.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5:** Manual verification (no commit): `python -m src.main --skip-dl --use-cached-tensor --no-plots` runs end-to-end in fast mode and prints the summary. Then a full `@pytest.mark.slow` test of `run_pipeline(full=False, skip_dl=False, make_plots=True)` writing all three `results/*.png`.
- [ ] **Step 6: Commit (checkpoint only)** `git commit -m "feat: add unified src/main.py pipeline CLI (fast default, --full/--optuna)"`

---

## Task 18: Docs, requirements finalize, legacy-file handling

**Files:**
- Modify: `README.md` (update "Quick Start": `python -m src.main` for the full pipeline; document `--full`, `--optuna`, `--skip-dl`; fix the stale `python src/data_cleaner.py` path).
- Modify: `requirements.txt` (final set: numpy, pandas, scipy, scikit-learn, xgboost, torch, optuna, cmaes, matplotlib, seaborn, pyarrow, jupyter, pytest).
- Create: `docs/superpowers/plans/` already holds this plan; no change.
- **Legacy files:** do NOT delete. Add a short `src/README.md` (or a `## Module map` section in the top README) stating which new module supersedes each old file (`preprocessing_eeg.py`, `tensor_creation.py`, `stcnn_nested_cv.py`, `data_cleaner.py`, `ML_first_draft/ML_01.ipynb`). Ask the user before removing any superseded `.py`.

**Interfaces:** none (docs only).

- [ ] **Step 1: Write the failing test** (docs presence + import surface)

```python
# tests/test_main.py (append)
def test_public_api_imports():
    import importlib
    for mod in ['src.config','src.io.raw_loader','src.preprocessing.baseline',
                'src.preprocessing.regions','src.preprocessing.windowing',
                'src.preprocessing.epochs','src.preprocessing.tensor',
                'src.analysis.encoding_decoding','src.models.ml.transforms',
                'src.models.ml.prepare','src.models.ml.models','src.models.ml.evaluation',
                'src.models.dl.dataset','src.models.dl.architecture',
                'src.models.dl.training','src.models.dl.xai','src.main']:
        importlib.import_module(mod)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_main.py::test_public_api_imports -v` → PASS once all modules exist (this is the integration gate).
- [ ] **Step 3: Update** README + requirements + module map.
- [ ] **Step 4: Run full fast suite** `python -m pytest -m "not slow" -v` → all green.
- [ ] **Step 5: Commit (checkpoint only)** `git commit -m "docs: update README/requirements, add module map for refactored src"`

---

## Self-Review

**1. Spec coverage** — every notebook section maps to a task:

| Notebook section | Cells | Task(s) |
|---|---|---|
| 0 Imports / 1 Constants | 3, 5 | 1 |
| 2 Raw Data | 8 | 2 |
| 3 Preprocessing | 11, 13, 15 | 3, 4, 5 |
| 4 Exploration (plot) | 18, 19 | folded into `main.py` logging (Task 17); optional `plot_subject` may be ported into `analysis` if desired |
| 5 Epochs | 22, 24 | 6 |
| 6 Tensor | 27 | 7 |
| 7 Encoding/Decoding | 31–46 | 8 |
| 8 ML | 50, 52, 54, 56–63 | 9, 10, 11, 12 |
| 9 DL | 66, 69, 71, 73 | 13, 14, 15, 16 |
| 10/11 Recap/Conclusions | 75–77 | narrative only — surfaced via `main.py` summary + README |
| Unified pipeline (main) | — | 17 |

Gap noted: the cell-18 per-subject sanity plot and cell-19 parquet export are not core to the unified run; Task 17 logs `value_counts` and the parquet export is optional (covered by `config.PARQUET_PATH` if the team wants it — add a `--save-parquet` flag if requested). Flag to user if the exploration plot must be reproduced exactly.

**2. Placeholder scan** — no "TBD"/"add error handling"/"similar to Task N". Bulk code is delegated to explicit notebook cells via the **Porting convention** (Global Constraints), with exact signatures, divergences, and real test code given per task.

**3. Type/name consistency** — `load_tensor`/`save_tensor` keys (`x`,`y`,`subjects`) are consistent across Tasks 7, 10, 15. `IntraSubjectZScore` defined in Task 9, consumed in Task 11. `prepare_ml_data`'s returned `data` dict is the exact input contract of `run_nested_cv`/`permutation_test` (Task 12). `model_factory`/`hp_space` names match between Tasks 14/15. `channel_names()` length 19 consistent with the 19-channel tensor.

**Divergence ledger (notebook wins, applied in-plan):**
1. Epoch artifact rejection: PCA z-threshold → 80 µV / max-2-bad-channels (Task 6).
2. Sex channel: raw/encoded value → `1.0 if 'F' else 0.0` (Task 7).
3. `subject_final` typo → `subjects_final` (Task 7).
4. `encode_labels` (M/F/R/A → int) dropped — notebook keeps strings (Tasks 2, 7).
5. Colab Drive paths → local `config` paths (Task 1, all stages).
6. Notebook globals → explicit function arguments (all stages).

---

## Execution Handoff

See the assistant message accompanying this plan for the two execution options (subagent-driven vs inline). **Reminder: per the user's instruction, do not commit or push during execution — treat every "Commit" step as a checkpoint marker only until the user authorizes committing.**
