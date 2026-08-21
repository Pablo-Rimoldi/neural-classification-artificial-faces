# Seeing the Uncanny: Neural Classification of Artificial Faces

## Project Overview

This project investigates the neural representation of AI-generated (GAN), hyper-realistic
faces compared to real human faces. Although the two categories are difficult to tell apart
behaviorally, event-related potentials (ERPs) recorded via EEG can carry a discriminative
signal. This repository implements the full pipeline used to test that hypothesis: from raw
EEG recordings to a tensor representation, through encoding/decoding analysis, classical
Machine Learning, and a Deep Learning model with explainability (XAI).

### Scientific Context

- **P.I.:** Prof. Alice Mado Proverbio (Univ. Milano-Bicocca)
- **Supervision:** Prof. Claudia Casellato (Univ. Pavia)
- **Team:** Pablo Rimoldi, Tommaso Godino, Andrea De Paola, Giacomo Colombo

### Technical Specifications

- **Data source:** 128-channel EEG (10-5 system), 512 Hz sampling rate.
- **Stimuli:** 440 male/female faces (Real vs. GAN-generated).
- **Signal processing:**
  - Bandpass filter: 0.01-70 Hz.
  - Notch filter: 50 Hz.
  - Epochs: -100 ms to 800 ms.
  - Reference: Common Average Reference (CAR).
  - Analysis window: **200-600 ms** post-stimulus.
- **Channels of interest (14):**
  `O1, O2, PO9, PO10, TP7, TP8, P3, P4, AF3, AF4, AFF1h, AFF2h, AFF3h, AFF4h`.

### Classification Task

EEG trials are labeled with one of 4 condition codes, which collapse into a binary
real-vs-AI task for modeling:

| Code | Condition       | Binary class |
|------|------------------|--------------|
| 50   | AI Male (AM)     | AI (0)       |
| 60   | AI Female (AF)   | AI (0)       |
| 70   | Real Male (RM)   | Real (1)     |
| 80   | Real Female (RF) | Real (1)     |

## Repository Layout

The pipeline is implemented as a Python package under `src/`, mirroring the reference
notebook section by section:

```
src/
  config.py                      Global constants and local filesystem paths
  io/
    raw_loader.py                Reads the raw per-subject EEG .txt files into a DataFrame
  preprocessing/
    baseline.py                  Baseline correction of the raw signal
    regions.py                   Region-level PCA features and column selection
    windowing.py                 Restricts samples to the 200-600 ms analysis window
    epochs.py                    Epoch construction and artifact rejection
    tensor.py                    Builds, saves, and loads the final (X, y, subjects) tensor
  analysis/
    encoding_decoding.py         Ridge encoding and decoding analysis on the tensor
  models/
    ml/
      transforms.py              Feature transforms (e.g. parameter representativeness)
      prepare.py                 Loads the tensor into the tabular ML data contract
      models.py                  Model/hyperparameter-grid factory for the ML pipeline
      evaluation.py              Nested CV, stage-2 search, permutation test, plots
    dl/
      dataset.py                 PyTorch Dataset/DataLoader construction
      architecture.py            The STCNN deep learning model definition
      training.py                Nested CV training loop and metric summary for the DL model
      xai.py                     Permutation importance, gradient saliency, XAI plots
  main.py                        Unified CLI entry point composing all stages
```

Legacy, pre-refactor files are kept for reference and are **not** removed; see
[Module map](#module-map) below for how they relate to the new package.

## Pipeline Overview

```
Raw EEG (.txt) -> Preprocessing (baseline, regions, windowing)
                -> Epochs (artifact rejection)
                -> Tensor (X, y, subjects)
                -> Encoding / Decoding analysis
                -> Machine Learning (nested CV over multiple models)
                -> Deep Learning (STCNN, nested CV) + XAI (saliency, permutation importance)
```

Each arrow corresponds to one or more modules under `src/`, composed end to end by
`run_pipeline()` in `src/main.py`.

## Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/Pablo-Rimoldi/neural-classification-artificial-faces
   cd neural-classification-artificial-faces
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the pipeline**

   ```bash
   python -m src.main
   ```

   By default this runs in **fast mode**: a quick-test configuration with reduced nested-CV
   folds/permutation counts, matching the reference notebook's default
   `QUICK_TEST = True`, `RUN_OPTUNA_SEARCH = False` settings. It rebuilds the tensor from the
   raw files in `data/file_raw/`, runs the encoding/decoding analysis, the ML pipeline, and
   the DL pipeline with XAI, saving figures to `results/`.

### CLI Flags

| Flag                 | Effect                                                                                           |
|-----------------------|---------------------------------------------------------------------------------------------------|
| `--full`             | Run the full (non quick-test) nested CV and permutation-test counts instead of the fast default. |
| `--optuna`           | Run the Optuna/CMA-ES hyperparameter search for the DL model instead of using the precomputed per-fold defaults in `data/best_hyperparameters.json`. |
| `--skip-dl`          | Skip the Deep Learning + XAI stage entirely (preprocessing, encoding/decoding, and ML still run). |
| `--no-plots`         | Do not generate or save any figures.                                                              |
| `--use-cached-tensor`| Load the cached tensor at `data/file_tensor/final_tensor.npz` instead of rebuilding it from raw files. |

Examples:

```bash
# Fast smoke run, ML + DL, no DL hyperparameter search
python -m src.main

# Full run (slow): full nested CV, full permutation counts
python -m src.main --full

# Full run including the Optuna/CMA-ES search for DL hyperparameters
python -m src.main --full --optuna

# Skip Deep Learning entirely (ML-only pipeline)
python -m src.main --skip-dl

# Reuse the tensor already saved on disk instead of rebuilding it from raw EEG files
python -m src.main --use-cached-tensor

# Headless run, no figures written to results/
python -m src.main --no-plots
```

**Fast vs. full mode:** fast mode (default) uses a quick-test configuration intended for
iterating on the code and verifying the pipeline runs end to end in a few minutes. Full mode
(`--full`) reproduces the actual experimental nested cross-validation and permutation-test
counts used for the reported results, and is correspondingly slower (the DL stage in
particular can take a long time on CPU). `--optuna` is independent of `--full` and only
controls whether DL hyperparameters are searched for or loaded from the cached file.

## Running Tests

The test suite mirrors the package layout (one test module per pipeline stage) and marks
long-running tests (full nested CV, Optuna search, DL training) with `@pytest.mark.slow`.

```bash
# Fast suite only (recommended for everyday development)
python -m pytest -m "not slow" -q

# Full suite, including slow tests
python -m pytest -q
```

## Key Experimental Results

The findings below reflect the full, executed benchmark from `notebooks/Neural_classification_artificial_faces.ipynb`:

### 1. Mass-Univariate Encoding & Linear Decoding (220–500 ms)
- **Ridge Encoding:** Best $\alpha = 100.0$, training $R^2 = 0.0052$, best cross-validated channel $R^2 = -0.0155$ (channel 17).
- **ANOVA Sensitivity Map:** Diffuse, peak $F = 8.57$ at channel 8 (AF3).
- **Signal Detection Theory (SDT) AUC:** Uniformly $\sim 0.500$ across individual channels.
- **Group-Aware Logistic Decoding:** Reached **$0.623 \pm 0.106$** accuracy across 5 folds (chance = 0.50), indicating that the AI-vs-real signal is distributed across multivariate patterns rather than localized in a single channel or timepoint.

### 2. Classical Machine Learning Benchmark (100 Trials, 25 Subjects)
Repeated group-aware nested cross-validation ($5 \times 5$ `StratifiedGroupKFold`, 25 outer estimates; inner `GroupShuffleSplit` with 5 splits and 20% holdout; intra-subject z-score $\to$ `SelectKBest` ANOVA-F $\to$ classifier):

| Model | Accuracy (mean & variance) | Macro-F1 (mean & variance) | ROC-AUC (mean & variance) |
|---|---|---|---|
| **LDA** | **0.700 (var 0.0048)** | **0.698 (var 0.0048)** | **0.783 (var 0.0090)** |
| LinearSVC_Cal | 0.700 (var 0.0062) | 0.698 (var 0.0064) | 0.778 (var 0.0098) |
| LinearSVC | 0.696 (var 0.0070) | 0.695 (var 0.0070) | 0.778 (var 0.0107) |
| LogReg_EN | 0.696 (var 0.0054) | 0.694 (var 0.0055) | 0.777 (var 0.0097) |
| LogReg_L1 | 0.694 (var 0.0075) | 0.692 (var 0.0076) | 0.778 (var 0.0121) |
| SGD | 0.688 (var 0.0099) | 0.678 (var 0.0126) | 0.778 (var 0.0131) |
| XGBoost | 0.678 (var 0.0092) | 0.673 (var 0.0101) | 0.774 (var 0.0122) |
| Dummy | 0.500 (var 0.0000) | 0.333 (var 0.0000) | 0.500 (var 0.0000) |

- **Stage 2 Focused Search on LDA ($n=80$ iterations):** Improved accuracy to **$0.720\ (\text{var } 0.0062)$**, macro-F1 to **$0.718$**, and ROC-AUC to **$0.780$** (**Adopted**).
  - *Most frequent hyperparameters across 25 folds:* `clf__shrinkage=0.05` (9/25), `clf__solver='lsqr'` (15/25), `scaler=RobustScaler()` (10/25), `selector__k=10` (15/25).
- **Permutation Test vs. Chance ($n=1000$ trial-level shuffles):**
  - Observed Accuracy: **$0.750$** vs. null mean $0.491\ (\text{var } 0.0038)$, **$p = 0.0010$** (chance = 0.50).
  - Observed MCC: **$+0.500$** vs. null mean $-0.018\ (\text{var } 0.0151)$, **$p = 0.0010$** $\implies$ **decoding statistically robust above chance**.
- **Wilcoxon Signed-Rank Tests (25 folds, Bonferroni-corrected):**
  - LDA significantly outperforms Dummy ($p_{\text{Bonf}} = 0.0001$).
  - Pairwise differences against competing linear models (LinearSVC_Cal $p_{\text{raw}}=0.0845$, LinearSVC $p_{\text{raw}}=0.0523$, LogReg_EN $p_{\text{raw}}=0.0557$, LogReg_L1 $p_{\text{raw}}=0.0394$, SGD $p_{\text{raw}}=0.0494$, XGBoost $p_{\text{raw}}=0.0318$) are non-significant after Bonferroni correction.

### 3. Deep Learning: SpatialTemporalCNN (103 Epochs, 5-Fold GroupKFold)
- **Architecture:** Learned $19 \times 19$ spatial mixing matrix $\to$ Conv1d temporal filters $\to$ Batch Normalization $\to$ ELU $\to$ optional residual layer $\to$ Adaptive Average Pooling ($T=8$) $\to$ Dropout $\to$ Linear logits (tuned via Optuna CMA-ES with 150 inner trials/fold).
- **Generalization Performance:**
  - Per-fold accuracies: `65.2%`, `60.0%`, `55.0%`, `70.0%`, `70.0%` (Per-fold: `['65%', '60%', '55%', '70%', '70%']`).
  - Per-fold AUCs: `0.652`, `0.570`, `0.750`, `0.680`, `0.700`.
  - **Mean Accuracy: $64.0\% \pm 5.8\%$**, **Mean AUC: $0.670$**.
  - **Binomial Test vs. Chance (50%):** **66/103 correct held-out predictions**, **$p = 0.0028$** (**statistically significant**, $p < 0.05$).
  - **Balanced Classification Report:**
    - AI ($50/60$): Precision = $0.63$, Recall = $0.65$, F1 = $0.64$ (support = 51)
    - Real ($70/80$): Precision = $0.65$, Recall = $0.63$, F1 = $0.64$ (support = 52)
    - Macro Avg F1 = $0.64$, Weighted Avg F1 = $0.64$.

### 4. Explainable AI (XAI) & Spatiotemporal Dynamics
- **Spatial Localization ("Where"):** Permutation importance ($N=20$ shuffles on held-out test sets across 5 folds) isolates **AF4, PO10, O2, TP8, and PCA_Parietal** as the primary discriminative features (a prominent right-lateralized posterior occipito-parietal cluster with frontal contribution).
- **Metadata Sanity Check:** Behavioral metadata feature `FaceSEX` had near-zero permutation importance ($\approx 0.008$), demonstrating that the network does not exploit metadata shortcuts.
- **Temporal Dynamics ("When"):** Discriminative saliency is late-stage rather than early sensory/perceptual, exhibiting prominent peaks at **$\sim 298\text{ ms}$, $\sim 355\text{ ms}$, and peaking sharply at $\sim 500\text{ ms}$** with sustained activity in the late LPP / P600 window ($530\text{–}600\text{ ms}$).
- **Spatial Filter Dissociation ("How"):** Contrasting permutation occlusion against gradient saliency distinguishes **bottleneck channels** (PO10, PCA_Frontal, PO9 with modest gradient but indispensable spatial information) from **redundant channels** (AFF3h with high local gradient but redundant spatial encoding).

---

## Reference Notebook

`notebooks/Neural_classification_artificial_faces.ipynb` is the reference notebook that the
`src/` package mirrors section by section (constants, raw loading, preprocessing, epoching,
tensor construction, encoding/decoding, ML, DL/XAI). It documents the original exploratory
analysis and is kept as the source of truth for the algorithmic steps implemented in `src/`.

## Module Map

The package under `src/` supersedes earlier, exploratory scripts and notebooks. Those files
are kept in the repository for reference and traceability but are no longer the active
implementation:

| Legacy file                                          | Superseded by                                                                                   |
|-------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `src/preprocessing/preprocessing_eeg.py`              | `src/io/raw_loader.py` + `src/preprocessing/baseline.py`, `regions.py`, `windowing.py`           |
| `src/preprocessing/tensor_creation.py`                | `src/preprocessing/epochs.py` + `src/preprocessing/tensor.py`                                    |
| `src/models/dl/stcnn_nested_cv.py`                    | `src/models/dl/architecture.py`, `dataset.py`, `training.py`                                     |
| `src/preprocessing/data_cleaner.py`                   | Kept as legacy; no direct replacement module (superseded in spirit by the preprocessing package) |
| `src/models/ml/ML_first_draft/ML_01.ipynb`            | Kept as legacy; superseded by `src/models/ml/{transforms,prepare,models,evaluation}.py`          |

These legacy files are intentionally left in place; do not delete them without explicit
confirmation. Placeholder text files (`ruolo*.txt`, `todo`) found alongside some modules are
internal team notes from the refactor and carry no runtime behavior.
