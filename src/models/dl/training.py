"""Training loop, hyperparameter search, and nested CV for the STCNN model.

Ported from notebook cells 66 (per-epoch normalisation / label mapping),
69 (training utilities + Optuna/CMA-ES inner search + outer nested-CV loop),
and 71 (metrics summary).

Divergences from the notebook: no module-level globals for data/device/model
state. Everything is passed explicitly via function arguments or the `data`
dict returned by `load_dl_data`. `RUN_OPTUNA_SEARCH` becomes the `run_optuna`
parameter of `run_dl_nested_cv` (default False). The `!pip install cmaes`
magic is dropped (declare `cmaes` as a regular dependency instead).
"""
import copy
import random

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from optuna.pruners import MedianPruner
from optuna.samplers import CmaEsSampler
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader

from src import config
from src.models.dl.architecture import SpatialTemporalCNN
from src.models.dl.dataset import EEGDataset

# Per-fold precomputed hyperparameters for the SpatialTemporalCNN (STCNN),
# taken verbatim from the ACTIVE `hardcoded_hps` list in notebook cell 69
# (the commented-out block in that cell is an older, unused variant and is
# intentionally not reproduced here).
DEFAULT_FOLD_HPS = [
    {'temp_filters': 128, 'kernel_size': 4, 'n_layers': 1, 'adj_init': 'uniform', 'adj_norm': 'softmax', 'dropout': 0.5875714865184984, 'lr': 0.00039944574656028425, 'wd': 9.3329893849496e-05, 'scheduler': 'step', 'batch_size': 16, 'epochs': 100, 'mixup_alpha': 0.12437494608574283},
    {'temp_filters': 128, 'kernel_size': 16, 'n_layers': 2, 'adj_init': 'random', 'adj_norm': 'sigmoid', 'dropout': 0.1931252250248428, 'lr': 0.00029249481849777224, 'wd': 0.00193796076277827, 'scheduler': 'none', 'batch_size': 8, 'epochs': 80, 'mixup_alpha': 0.1562916113503165},
    {'temp_filters': 64, 'kernel_size': 4, 'n_layers': 1, 'adj_init': 'uniform', 'adj_norm': 'sigmoid', 'dropout': 0.414347718713756, 'lr': 0.0008344540631504169, 'wd': 1.248983925829189e-05, 'scheduler': 'none', 'batch_size': 16, 'epochs': 120, 'mixup_alpha': 0.35237380732260076},
    {'temp_filters': 128, 'kernel_size': 4, 'n_layers': 1, 'adj_init': 'uniform', 'adj_norm': 'softmax', 'dropout': 0.3576234092040268, 'lr': 0.0024085114381778554, 'wd': 6.0016969669930806e-05, 'scheduler': 'step', 'batch_size': 8, 'epochs': 80, 'mixup_alpha': 0.16240200839794475},
    {'temp_filters': 128, 'kernel_size': 4, 'n_layers': 2, 'adj_init': 'random', 'adj_norm': 'sigmoid', 'dropout': 0.634267783778097, 'lr': 0.001336989948210231, 'wd': 1.7874482694064222e-05, 'scheduler': 'step', 'batch_size': 8, 'epochs': 80, 'mixup_alpha': 0.19622602414996604},
]


def reset_all_seeds(seed):
    """Seed Python's random, numpy, and torch (+ CUDA if available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mixup_batch(x, y, alpha=0.2):
    """Mix a batch with a randomly permuted copy of itself (mixup augmentation)."""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(crit, pred, ya, yb, lam):
    """Convex combination of the loss against both mixup targets."""
    return lam * crit(pred, ya) + (1 - lam) * crit(pred, yb)


def build_scheduler(optimizer, scheduler_type, epochs):
    """Build a LR scheduler ('cosine', 'step', or None passthrough)."""
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    return None


def train_one_epoch(model, loader, optimizer, criterion, device, mixup_alpha=0.2):
    """Run one training epoch, applying mixup to ~half of the batches."""
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        if random.random() < 0.5:
            xm, ya, yb_, lam = mixup_batch(xb, yb, alpha=mixup_alpha)
            loss = mixup_criterion(criterion, model(xm), ya, yb_, lam)
        else:
            loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Return the mean loss of `model` over `loader`."""
    model.eval()
    tot_loss, total = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        tot_loss += criterion(out, yb).item() * xb.size(0)
        total += xb.size(0)
    return tot_loss / total


@torch.no_grad()
def collect_preds(model, loader, device):
    """Collect (probs, preds, labels) for the positive class over `loader`."""
    model.eval()
    probs, preds, labels = [], [], []
    for xb, yb in loader:
        out = model(xb.to(device))
        probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(yb.numpy())
    return np.array(probs), np.array(preds), np.array(labels)


def hp_space(trial):
    """Optuna search space for the STCNN + training hyperparameters."""
    return dict(
        temp_filters=trial.suggest_categorical('temp_filters', [16, 32, 64, 128]),
        kernel_size=trial.suggest_categorical('kernel_size', [4, 8, 16, 32]),
        n_layers=trial.suggest_categorical('n_layers', [1, 2]),
        adj_init=trial.suggest_categorical('adj_init', ['uniform', 'identity', 'random']),
        adj_norm=trial.suggest_categorical('adj_norm', ['none', 'softmax', 'sigmoid']),
        dropout=trial.suggest_float('dropout', 0.1, 0.75),
        lr=trial.suggest_float('lr', 5e-5, 1e-2, log=True),
        wd=trial.suggest_float('wd', 1e-6, 5e-3, log=True),
        scheduler=trial.suggest_categorical('scheduler', ['none', 'cosine', 'step']),
        batch_size=trial.suggest_categorical('batch_size', [8, 16, 32]),
        epochs=trial.suggest_categorical('epochs', [60, 80, 100, 120]),
        mixup_alpha=trial.suggest_float('mixup_alpha', 0.05, 0.4),
    )


def model_factory(hp):
    """Build a SpatialTemporalCNN from a hyperparameter dict."""
    return SpatialTemporalCNN(
        temp_filters=hp['temp_filters'],
        kernel_size=hp['kernel_size'],
        n_layers=hp['n_layers'],
        adj_init=hp['adj_init'],
        adj_norm=hp['adj_norm'],
        dropout=hp['dropout'],
    )


def load_dl_data(tensor_path=config.TENSOR_PATH) -> dict:
    """Load the cached tensor and prepare it for the DL pipeline.

    Maps condition codes to binary labels via `config.CONDITION_TO_BINARY`
    and applies per-epoch normalisation (mean/std over the channel and time
    axes of each sample, i.e. axes (1, 2)), matching notebook cell 66.

    Returns:
        dict with keys 'X_norm' (float32 ndarray, N x C x T), 'y' (int64
        ndarray, N), and 'subjects' (ndarray, N).
    """
    raw = np.load(tensor_path, allow_pickle=True)
    X = raw['x'].astype(np.float32)
    y_raw = raw['y']
    subjects = raw['subjects']

    y = np.array([config.CONDITION_TO_BINARY.get(label, 0) for label in y_raw], dtype=np.int64)

    X_mean = X.mean(axis=(1, 2), keepdims=True)
    X_std = X.std(axis=(1, 2), keepdims=True)
    X_norm = (X - X_mean) / (X_std + 1e-8)

    return {'X_norm': X_norm, 'y': y, 'subjects': subjects}


def run_dl_nested_cv(
    data,
    *,
    run_optuna=False,
    n_folds=5,
    n_inner_trials=150,
    device=None,
    seed=config.DL_SEED,
) -> dict:
    """Run GroupKFold nested cross-validation for the STCNN model.

    Outer loop: GroupKFold over `n_folds` folds, grouped by subject. For each
    outer fold, hyperparameters are either searched via a CMA-ES Optuna study
    (when `run_optuna=True`, cell 69 `objective`) or taken from
    `DEFAULT_FOLD_HPS[fold]` (when `run_optuna=False`). The model is then
    retrained on the full outer-train split and evaluated on the outer-test
    split.

    Returns:
        dict with keys: 'outer_accs', 'fold_aucs', 'all_preds', 'all_labels',
        'all_probs', 'fold_models', 'fold_te_indices'.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Mirror the notebook's global determinism setup (cell 66): on CUDA, force
    # deterministic cuDNN kernels. No-op on CPU.
    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True

    X_norm, y, subjects = data['X_norm'], data['y'], data['subjects']

    reset_all_seeds(seed)

    n_folds = min(n_folds, len(np.unique(subjects)))
    outer_gkf = GroupKFold(n_splits=n_folds)
    criterion = nn.CrossEntropyLoss()

    outer_accs, fold_aucs = [], []
    all_preds, all_labels, all_probs = [], [], []
    fold_models = []
    fold_te_indices = []

    for outer_fold, (outer_tr_idx, outer_te_idx) in enumerate(
        outer_gkf.split(X_norm, y, groups=subjects)
    ):
        if run_optuna:
            def objective(trial):
                hp = hp_space(trial)
                gss = GroupShuffleSplit(n_splits=3, test_size=0.25, random_state=seed + outer_fold)
                split_losses = []
                for split_i, (lo_tr, lo_val) in enumerate(
                    gss.split(X_norm[outer_tr_idx], y[outer_tr_idx], groups=subjects[outer_tr_idx])
                ):
                    i_tr, i_val = outer_tr_idx[lo_tr], outer_tr_idx[lo_val]
                    tr_ds = EEGDataset(X_norm[i_tr], y[i_tr], augment=True)
                    val_ds = EEGDataset(X_norm[i_val], y[i_val], augment=False)
                    dl_drop = True if len(tr_ds) > hp['batch_size'] else False
                    tr_dl = DataLoader(tr_ds, batch_size=hp['batch_size'], shuffle=True, drop_last=dl_drop)
                    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

                    reset_all_seeds(seed + outer_fold * 100 + split_i)
                    m = model_factory(hp).to(device)
                    opt = torch.optim.Adam(m.parameters(), lr=hp['lr'], weight_decay=hp['wd'])
                    sch = build_scheduler(opt, hp['scheduler'], hp['epochs'])

                    best_val, patience, max_patience = float('inf'), 0, 20
                    for epoch in range(hp['epochs']):
                        train_one_epoch(m, tr_dl, opt, criterion, device, mixup_alpha=hp['mixup_alpha'])
                        if sch is not None:
                            sch.step()

                        if split_i == 0 and (epoch + 1) % 10 == 0:
                            val_loss = evaluate(m, val_dl, criterion, device)
                            trial.report(val_loss, step=epoch)
                            if trial.should_prune():
                                raise optuna.TrialPruned()

                        if (epoch + 1) % 10 == 0:
                            val_loss = evaluate(m, val_dl, criterion, device)
                            if val_loss < best_val:
                                best_val = val_loss
                                patience = 0
                            else:
                                patience += 1
                            if patience >= max_patience // 10:
                                break

                    split_losses.append(evaluate(m, val_dl, criterion, device))
                return np.mean(split_losses)

            sampler = CmaEsSampler(n_startup_trials=10, seed=seed + outer_fold, warn_independent_sampling=False)
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=20, interval_steps=10)
            study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(objective, n_trials=n_inner_trials, n_jobs=1)
            best_hp = study.best_trial.params
        else:
            best_hp = DEFAULT_FOLD_HPS[outer_fold]

        reset_all_seeds(seed + outer_fold * 999)

        model_final = model_factory(best_hp).to(device)
        opt_final = torch.optim.Adam(model_final.parameters(), lr=best_hp['lr'], weight_decay=best_hp['wd'])
        sch_final = build_scheduler(opt_final, best_hp['scheduler'], best_hp['epochs'])

        tr_ds_full = EEGDataset(X_norm[outer_tr_idx], y[outer_tr_idx], augment=True)
        te_ds_full = EEGDataset(X_norm[outer_te_idx], y[outer_te_idx], augment=False)

        drop_l = True if len(tr_ds_full) > best_hp['batch_size'] else False
        tr_dl_full = DataLoader(tr_ds_full, batch_size=best_hp['batch_size'], shuffle=True, drop_last=drop_l)
        te_dl_full = DataLoader(te_ds_full, batch_size=16, shuffle=False)

        for epoch in range(best_hp['epochs']):
            train_one_epoch(model_final, tr_dl_full, opt_final, criterion, device, mixup_alpha=best_hp['mixup_alpha'])
            if sch_final is not None:
                sch_final.step()

        fold_models.append(copy.deepcopy(model_final))
        fold_te_indices.append(outer_te_idx)

        probs, preds, labels = collect_preds(model_final, te_dl_full, device)
        acc = accuracy_score(labels, preds)
        try:
            fold_auc = roc_auc_score(labels, probs)
        except ValueError:
            fold_auc = 0.5

        outer_accs.append(acc)
        fold_aucs.append(fold_auc)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels)

    return {
        'outer_accs': outer_accs,
        'fold_aucs': fold_aucs,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs,
        'fold_models': fold_models,
        'fold_te_indices': fold_te_indices,
    }


def summarize_dl_metrics(result: dict) -> dict:
    """Summarise nested-CV results: mean acc/AUC, binomial test vs chance, report.

    Ported from notebook cell 71.
    """
    outer_accs_np = np.array(result['outer_accs'])
    fold_aucs_np = np.array(result['fold_aucs'])
    all_labels_np = np.array(result['all_labels'])
    all_preds_np = np.array(result['all_preds'])

    n_total = len(all_labels_np)
    n_correct = int((all_preds_np == all_labels_np).sum())
    binom_result = binomtest(n_correct, n=n_total, p=0.5, alternative='greater')

    report = classification_report(
        all_labels_np, all_preds_np, target_names=["AI (50/60)", "Human (70/80)"]
    )

    return {
        'mean_acc': float(outer_accs_np.mean()),
        'std_acc': float(outer_accs_np.std()),
        'mean_auc': float(fold_aucs_np.mean()),
        'per_fold_acc': outer_accs_np.tolist(),
        'n_correct': n_correct,
        'n_total': n_total,
        'p_value': float(binom_result.pvalue),
        'significant': bool(binom_result.pvalue < 0.05),
        'classification_report': report,
    }
