import json
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import optuna
from optuna.samplers import CmaEsSampler
from optuna.pruners import MedianPruner

# ── Assumes these are defined by the caller / preceding cells ─────────────
# X_dl        : np.ndarray  (N, 19, 205)  raw tensor (NOT pre-normalised)
# y_dl        : np.ndarray  (N,)          int64 labels
# subjects_dl : np.ndarray  (N,)          subject IDs for GroupKFold
# N_FOLDS_DL  : int
# N_INNER_TRIALS : int
# DL_SEED     : int
# device      : torch.device
# SpatialTemporalCNN : nn.Module class (defined in the preceding cell)
# EEGDataset  : torch.utils.data.Dataset
# ─────────────────────────────────────────────────────────────────────────

RUN_OPTUNA_SEARCH = True

# ── Search budget (override the value set in the preceding cell) ──────────
N_INNER_TRIALS = 150   # was 50 — more trials = better CMA-ES convergence
N_INNER_SPLITS = 5     # was 3 — more stable inner loss estimate


def reset_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mixup_batch(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(crit, pred, ya, yb, lam):
    return lam * crit(pred, ya) + (1 - lam) * crit(pred, yb)


def build_scheduler(optimizer, scheduler_type, epochs):
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6)
    if scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    return None


def train_one_epoch(model, loader, optimizer, criterion, mixup_alpha=0.2):
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
def evaluate(model, loader, criterion):
    model.eval()
    tot_loss, total = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        tot_loss += criterion(out, yb).item() * xb.size(0)
        total += xb.size(0)
    return tot_loss / total


@torch.no_grad()
def collect_preds(model, loader):
    model.eval()
    probs, preds, labels = [], [], []
    for xb, yb in loader:
        out = model(xb.to(device))
        probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(yb.numpy())
    return np.array(probs), np.array(preds), np.array(labels)


def model_factory(hp):
    return SpatialTemporalCNN(
        temp_filters=hp['temp_filters'],
        kernel_size=hp['kernel_size'],
        n_layers=hp['n_layers'],
        adj_init=hp['adj_init'],
        adj_norm=hp['adj_norm'],
        dropout=hp['dropout'],
    )


def hp_space(trial):
    return dict(
        # architecture — added 8 and 256 filters, kernel 64, third layer option
        temp_filters=trial.suggest_categorical('temp_filters', [8, 16, 32, 64, 128, 256]),
        kernel_size=trial.suggest_categorical('kernel_size', [4, 8, 16, 32, 64]),
        n_layers=trial.suggest_categorical('n_layers', [1, 2, 3]),
        adj_init=trial.suggest_categorical('adj_init', ['uniform', 'identity', 'random']),
        adj_norm=trial.suggest_categorical('adj_norm', ['none', 'softmax', 'sigmoid']),
        # regularisation — wider dropout range
        dropout=trial.suggest_float('dropout', 0.05, 0.80),
        # optimiser — one extra decade on both ends
        lr=trial.suggest_float('lr', 1e-5, 2e-2, log=True),
        wd=trial.suggest_float('wd', 1e-7, 1e-2, log=True),
        scheduler=trial.suggest_categorical('scheduler', ['none', 'cosine', 'step']),
        batch_size=trial.suggest_categorical('batch_size', [8, 16, 32]),
        # training length — added 150 and 200
        epochs=trial.suggest_categorical('epochs', [80, 100, 120, 150, 200]),
        mixup_alpha=trial.suggest_float('mixup_alpha', 0.0, 0.5),
    )


hardcoded_hps = [
    {'temp_filters': 32,  'kernel_size': 16, 'n_layers': 1, 'adj_init': 'identity', 'adj_norm': 'sigmoid', 'dropout': 0.430385, 'lr': 0.000886, 'wd': 9.445e-06, 'scheduler': 'cosine', 'batch_size': 32, 'epochs': 120, 'mixup_alpha': 0.3743},
    {'temp_filters': 64,  'kernel_size': 16, 'n_layers': 1, 'adj_init': 'random',   'adj_norm': 'softmax', 'dropout': 0.252020, 'lr': 0.000718, 'wd': 7.672e-05, 'scheduler': 'cosine', 'batch_size': 32, 'epochs': 120, 'mixup_alpha': 0.2786},
    {'temp_filters': 128, 'kernel_size': 4,  'n_layers': 2, 'adj_init': 'uniform',  'adj_norm': 'softmax', 'dropout': 0.349319, 'lr': 0.000134, 'wd': 0.001161,  'scheduler': 'none',   'batch_size': 32, 'epochs': 60,  'mixup_alpha': 0.0630},
    {'temp_filters': 32,  'kernel_size': 8,  'n_layers': 1, 'adj_init': 'identity', 'adj_norm': 'sigmoid', 'dropout': 0.395922, 'lr': 0.000453, 'wd': 0.000848,  'scheduler': 'none',   'batch_size': 32, 'epochs': 100, 'mixup_alpha': 0.2787},
    {'temp_filters': 128, 'kernel_size': 32, 'n_layers': 2, 'adj_init': 'random',   'adj_norm': 'softmax', 'dropout': 0.466126, 'lr': 0.000378, 'wd': 0.000111,  'scheduler': 'none',   'batch_size': 32, 'epochs': 120, 'mixup_alpha': 0.2111},
]

# ── Run ───────────────────────────────────────────────────────────────────

outer_gkf = GroupKFold(n_splits=N_FOLDS_DL)
criterion_gl = nn.CrossEntropyLoss()

outer_accs, fold_aucs = [], []
all_preds, all_labels, all_probs = [], [], []
fold_models = []
fold_te_indices = []
best_hps_found = []   # collects the best HP per fold for later reuse

if RUN_OPTUNA_SEARCH:
    print("Running STCNN — Nested CV with Optuna (this will take a while)...")
else:
    print("Running STCNN — Fast Mode (pre-computed hyperparameters)...")

for outer_fold, (outer_tr_idx, outer_te_idx) in enumerate(
        outer_gkf.split(X_dl, y_dl, groups=subjects_dl)):

    print(f"\n[Fold {outer_fold + 1}/{N_FOLDS_DL}]")

    # Channel-wise normalization computed on training fold only.
    # Preserves amplitude differences between samples (unlike per-sample norm).
    tr_mean = X_dl[outer_tr_idx].mean(axis=(0, 2), keepdims=True)  # (1, 19, 1)
    tr_std  = X_dl[outer_tr_idx].std(axis=(0, 2), keepdims=True)
    X_fold  = (X_dl - tr_mean) / (tr_std + 1e-8)

    if RUN_OPTUNA_SEARCH:
        def objective(trial):
            hp = hp_space(trial)
            gss = GroupShuffleSplit(n_splits=N_INNER_SPLITS, test_size=0.25,
                                    random_state=DL_SEED + outer_fold)
            split_losses = []
            for split_i, (lo_tr, lo_val) in enumerate(
                    gss.split(X_fold[outer_tr_idx], y_dl[outer_tr_idx],
                              groups=subjects_dl[outer_tr_idx])):

                i_tr, i_val = outer_tr_idx[lo_tr], outer_tr_idx[lo_val]
                tr_ds = EEGDataset(X_fold[i_tr], y_dl[i_tr], augment=True)
                val_ds = EEGDataset(X_fold[i_val], y_dl[i_val], augment=False)
                dl_drop = len(tr_ds) > hp['batch_size']
                tr_dl = DataLoader(tr_ds, batch_size=hp['batch_size'],
                                   shuffle=True, drop_last=dl_drop)
                val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

                reset_all_seeds(DL_SEED + outer_fold * 100 + split_i)
                m = model_factory(hp).to(device)
                opt = torch.optim.Adam(m.parameters(),
                                       lr=hp['lr'], weight_decay=hp['wd'])
                sch = build_scheduler(opt, hp['scheduler'], hp['epochs'])

                # patience counts consecutive 10-epoch windows with no improvement.
                # threshold=5 means 50 epochs of no improvement before stopping.
                best_val, patience, patience_threshold = float('inf'), 0, 5
                for epoch in range(hp['epochs']):
                    train_one_epoch(m, tr_dl, opt, criterion_gl,
                                    mixup_alpha=hp['mixup_alpha'])
                    if sch is not None:
                        sch.step()

                    if (epoch + 1) % 10 == 0:
                        val_loss = evaluate(m, val_dl, criterion_gl)
                        if split_i == 0:
                            trial.report(val_loss, step=epoch)
                            if trial.should_prune():
                                raise optuna.TrialPruned()
                        if val_loss < best_val:
                            best_val = val_loss
                            patience = 0
                        else:
                            patience += 1
                        if patience >= patience_threshold:
                            break

                split_losses.append(evaluate(m, val_dl, criterion_gl))
            return np.mean(split_losses)

        sampler = CmaEsSampler(n_startup_trials=20,
                               seed=DL_SEED + outer_fold,
                               warn_independent_sampling=False)
        pruner = MedianPruner(n_startup_trials=10,
                              n_warmup_steps=30, interval_steps=10)
        study = optuna.create_study(direction='minimize',
                                    sampler=sampler, pruner=pruner)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=N_INNER_TRIALS, n_jobs=1)
        best_hp = study.best_trial.params
        print(f"  Best HP: {best_hp}")
    else:
        best_hp = hardcoded_hps[outer_fold]

    best_hps_found.append(best_hp)

    # ── Final retraining on full outer_train ─────────────────────────────
    reset_all_seeds(DL_SEED + outer_fold * 999)

    model_final = model_factory(best_hp).to(device)
    opt_final = torch.optim.Adam(model_final.parameters(),
                                  lr=best_hp['lr'], weight_decay=best_hp['wd'])
    sch_final = build_scheduler(opt_final, best_hp['scheduler'], best_hp['epochs'])

    tr_ds_full = EEGDataset(X_fold[outer_tr_idx], y_dl[outer_tr_idx], augment=True)
    te_ds_full = EEGDataset(X_fold[outer_te_idx], y_dl[outer_te_idx], augment=False)
    drop_l = len(tr_ds_full) > best_hp['batch_size']
    tr_dl_full = DataLoader(tr_ds_full, batch_size=best_hp['batch_size'],
                             shuffle=True, drop_last=drop_l)
    te_dl_full = DataLoader(te_ds_full, batch_size=16, shuffle=False)

    for epoch in range(best_hp['epochs']):
        train_one_epoch(model_final, tr_dl_full, opt_final, criterion_gl,
                        mixup_alpha=best_hp['mixup_alpha'])
        if sch_final is not None:
            sch_final.step()

    fold_models.append(copy.deepcopy(model_final))
    fold_te_indices.append(outer_te_idx)

    probs, preds, labels = collect_preds(model_final, te_dl_full)
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

    print(f"  --> Acc: {acc * 100:.1f}% | AUC: {fold_auc:.3f}")

# ── Summary ───────────────────────────────────────────────────────────────

best_fold_idx = int(np.argmax(fold_aucs))
print(f"\nBest fold for XAI: Fold {best_fold_idx + 1} "
      f"(AUC={fold_aucs[best_fold_idx]:.3f})")

if RUN_OPTUNA_SEARCH:
    print("\n" + "=" * 60)
    print("BEST HYPERPARAMETERS FOUND — copy into hardcoded_hps")
    print("=" * 60)
    print("hardcoded_hps = [")
    for hp in best_hps_found:
        print(f"    {hp},")
    print("]")

    hp_path = "data/best_hyperparameters.json"
    with open(hp_path, "w") as f:
        json.dump(best_hps_found, f, indent=2)
    print(f"\nSaved to {hp_path}")
