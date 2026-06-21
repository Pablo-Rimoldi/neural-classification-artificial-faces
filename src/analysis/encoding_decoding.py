"""Mass-univariate encoding/decoding analysis on the EEG tensor (notebook sec. 7).

Ported verbatim from notebook cells 31, 34, 37, 40, 43, 45, 46 ("Setup:
Z-score Normalization & Flat Channel Removal" through "Plot of the
results"/"SUMMARY"), wrapped into a single function. The cell-45 figure and
cell-46 printed summary are guarded behind ``make_plot`` and, when requested,
saved to ``save_path`` instead of (only) shown interactively.
"""
import matplotlib
matplotlib.use('Agg')  # noqa: E402  (headless-safe backend; must precede pyplot import)
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f_oneway, zscore
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder

from src import config


def run_encoding_decoding(X, y_codes, subjects, *, make_plot=True, save_path=None):
    """Run the mass-univariate encoding/decoding analysis on an EEG tensor.

    Args:
        X: ndarray (n_epochs, n_channels, n_times_full) EEG tensor.
        y_codes: ndarray of TargetCODE strings (e.g. '50AM', '70RM', ...).
        subjects: ndarray of SubjectID strings/labels, one per epoch.
        make_plot: if True, render the cell-45 summary figure (and print the
            cell-46 textual summary). If False, no plotting is done.
        save_path: if given (and make_plot), save the figure to this path via
            ``plt.savefig`` in addition to the default backend behaviour.

    Returns:
        dict with keys: f_map (n_good, n_times), sta (dict label ->
        (n_good, n_times)), K (n_classes, n_good, n_times), R2_train (float),
        r2_cv (n_good,), auc_map (n_classes, n_good, n_times),
        decoding_acc_mean (float), decoding_acc_std (float),
        good_ch (ndarray), n_good (int), time_axis (n_times,),
        unique_labels (ndarray).
    """
    X_final = X
    y_final = y_codes
    subjects_final = subjects

    # --- Cell 31: Z-score normalisation & flat-channel removal -------------
    n_epochs, n_channels, n_times_full = X_final.shape

    X_norm = zscore(X_final, axis=2)
    X_norm[np.isnan(X_norm)] = 0

    channel_var = X_final.var(axis=2).mean(axis=0)
    good_ch = np.where(channel_var > config.VAR_THRESHOLD)[0]
    bad_ch = np.where(channel_var <= config.VAR_THRESHOLD)[0]
    X_clean = X_norm[:, good_ch, :]
    n_good = len(good_ch)
    print(f"Good channels: {n_good}/{n_channels}  |  Excluded: {bad_ch.tolist()}")

    # --- Cell 34: label mapping + mass-univariate ANOVA F-map --------------
    label_map = config.CONDITION_TO_LABEL
    y_labels = y_final.copy()
    y_labels = np.array([label_map[lbl] for lbl in y_labels])

    unique_labels = np.unique(y_labels)
    n_classes = len(unique_labels)
    f_map_full = np.zeros((n_good, n_times_full))

    for c in range(n_good):
        for t in range(n_times_full):
            groups = [X_clean[y_labels == lbl, c, t] for lbl in unique_labels]
            f_stat, _ = f_oneway(*groups)
            f_map_full[c, t] = f_stat if np.isfinite(f_stat) else 0

    # --- Cell 37: time window selection (N250 + P300) -----------------------
    ms_per_tp = (config.TIME_END_MS - config.TIME_START_MS) / n_times_full

    def ms_to_tp(ms):
        return int((ms - config.TIME_START_MS) / ms_per_tp)

    t_start = ms_to_tp(config.N250_START)
    t_end = min(n_times_full, ms_to_tp(config.P300_END))

    X_clean = X_clean[:, :, t_start:t_end]
    f_map = f_map_full[:, t_start:t_end]
    n_times = X_clean.shape[2]

    TIME_WIN_START = config.N250_START
    TIME_WIN_END = config.P300_END
    time_axis = np.linspace(TIME_WIN_START, TIME_WIN_END, n_times)
    ext = [TIME_WIN_START, TIME_WIN_END, n_good, 0]

    print(f"N250   : {config.N250_START}-{config.N250_END} ms")
    print(f"P300   : {config.P300_START}-{config.P300_END} ms")
    print(f"Window : {TIME_WIN_START}-{TIME_WIN_END} ms  ->  {n_times} timepoints  "
          f"({ms_per_tp:.2f} ms/tp)")

    # --- Cell 40: STA per class + Ridge encoding model ----------------------
    sta = {lbl: np.mean(X_clean[y_labels == lbl], axis=0) for lbl in unique_labels}

    ohe = OneHotEncoder(sparse_output=False)
    S = ohe.fit_transform(y_labels.reshape(-1, 1))
    Y_enc = X_clean.reshape(n_epochs, -1)

    ridge = RidgeCV(alphas=config.RIDGE_ALPHAS, cv=config.N_SPLITS)
    ridge.fit(S, Y_enc)
    K = ridge.coef_.T.reshape(n_classes, n_good, n_times)
    R2_train = r2_score(Y_enc, ridge.predict(S), multioutput='variance_weighted')
    print(f"Best alpha: {ridge.alpha_:.3f}  |  Training R^2: {R2_train:.4f}")

    kf = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=42)
    r2_cv = np.zeros(n_good)
    for ch in range(n_good):
        y_ch = X_clean[:, ch, :]
        preds, trues = [], []
        for tr, te in kf.split(S):
            m = RidgeCV(alphas=config.RIDGE_ALPHAS)
            m.fit(S[tr], y_ch[tr])
            preds.append(m.predict(S[te]))
            trues.append(y_ch[te])
        r2_cv[ch] = r2_score(np.vstack(trues), np.vstack(preds), multioutput='variance_weighted')

    # --- Cell 43: AUC discriminability map + logistic regression decoding --
    auc_map = np.zeros((n_classes, n_good, n_times))
    for cls_idx, lbl in enumerate(unique_labels):
        y_bin = (y_labels == lbl).astype(int)
        for c in range(n_good):
            for t in range(n_times):
                try:
                    auc_map[cls_idx, c, t] = roc_auc_score(y_bin, X_clean[:, c, t])
                except Exception:
                    auc_map[cls_idx, c, t] = 0.5

    X_2d = X_clean.reshape(n_epochs, -1)
    clf = LogisticRegressionCV(cv=config.N_SPLITS, max_iter=1000, random_state=42)
    gkf = GroupKFold(n_splits=config.N_SPLITS)
    scores = cross_val_score(clf, X_2d, y_labels, cv=gkf, groups=subjects_final)
    chance = 1 / n_classes
    print(f"Decoding accuracy: {scores.mean():.3f} +/- {scores.std():.3f}  "
          f"(chance = {chance:.2f})")

    decoding_acc_mean = float(scores.mean())
    decoding_acc_std = float(scores.std())

    # --- Cells 45-46: figure + printed summary (guarded by make_plot) ------
    if make_plot:
        n250_peak_ms = (config.N250_START + config.N250_END) / 2
        p300_peak_ms = (config.P300_START + config.P300_END) / 2

        fig = plt.figure(figsize=(20, 24))
        fig.suptitle('EEG Full Pipeline: Encoding + SDT + Decoding\n'
                     f'Time window: {TIME_WIN_START}-{TIME_WIN_END} ms  (N250 + P300)',
                     fontsize=15, fontweight='bold')
        gs = fig.add_gridspec(5, 4, hspace=0.5, wspace=0.35)

        ax = fig.add_subplot(gs[0, :2])
        im = ax.imshow(f_map, aspect='auto', cmap='hot', extent=ext)
        plt.colorbar(im, ax=ax, label='F-statistic')
        ax.axvline(n250_peak_ms, color='cyan', linewidth=1.2, linestyle='--', label='N250')
        ax.axvline(p300_peak_ms, color='yellow', linewidth=1.2, linestyle='--', label='P300')
        ax.legend(fontsize=8, loc='upper right')
        ax.set_title('ANOVA Encoding Map (N250 + P300 window)')
        ax.set_xlabel('Time (ms)'); ax.set_ylabel('Channel')

        ax = fig.add_subplot(gs[0, 2:])
        sta_mat = np.array([sta[lbl].mean(axis=0) for lbl in unique_labels])
        im = ax.imshow(sta_mat, aspect='auto', cmap='RdBu_r',
                       extent=[TIME_WIN_START, TIME_WIN_END, n_classes, 0])
        plt.colorbar(im, ax=ax, label='Z-score')
        ax.set_yticks(np.arange(n_classes) + 0.5)
        ax.set_yticklabels(unique_labels)
        ax.axvline(n250_peak_ms, color='black', linewidth=1.2, linestyle='--', label='N250')
        ax.axvline(p300_peak_ms, color='grey', linewidth=1.2, linestyle='--', label='P300')
        ax.legend(fontsize=8, loc='upper right')
        ax.set_title('STA per Class (channel-averaged)')
        ax.set_xlabel('Time (ms)')

        for i, lbl in enumerate(unique_labels):
            ax = fig.add_subplot(gs[1, i])
            vmax = np.abs(K[i]).max()
            im = ax.imshow(K[i], aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=ext)
            plt.colorbar(im, ax=ax, label='Weight')
            ax.axvline(n250_peak_ms, color='cyan', linewidth=1, linestyle='--')
            ax.axvline(p300_peak_ms, color='yellow', linewidth=1, linestyle='--')
            ax.set_title(f'K - {lbl}', fontsize=10)
            ax.set_xlabel('Time (ms)'); ax.set_ylabel('Channel')

        for i, lbl in enumerate(unique_labels):
            ax = fig.add_subplot(gs[2, i])
            im = ax.imshow(auc_map[i], aspect='auto', cmap='RdBu_r', vmin=0, vmax=1, extent=ext)
            plt.colorbar(im, ax=ax, label='AUC')
            ax.axvline(n250_peak_ms, color='cyan', linewidth=1, linestyle='--')
            ax.axvline(p300_peak_ms, color='yellow', linewidth=1, linestyle='--')
            ax.set_title(f'AUC - {lbl}', fontsize=10)
            ax.set_xlabel('Time (ms)'); ax.set_ylabel('Channel')

        ax = fig.add_subplot(gs[3, :2])
        im = ax.imshow(auc_map.mean(axis=0), aspect='auto', cmap='RdBu_r', vmin=0, vmax=1, extent=ext)
        plt.colorbar(im, ax=ax, label='Mean AUC')
        ax.axvline(n250_peak_ms, color='cyan', linewidth=1.2, linestyle='--', label='N250')
        ax.axvline(p300_peak_ms, color='yellow', linewidth=1.2, linestyle='--', label='P300')
        ax.legend(fontsize=8, loc='upper right')
        ax.set_title('Mean AUC Map (across classes)')
        ax.set_xlabel('Time (ms)'); ax.set_ylabel('Channel')

        ax = fig.add_subplot(gs[3, 2:])
        colors = ['#e74c3c' if r > 0 else '#95a5a6' for r in r2_cv]
        ax.bar(np.arange(n_good), r2_cv, color=colors, edgecolor='white')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel('Channel'); ax.set_ylabel('CV R^2')
        ax.set_title(f'Encoding R^2 per Channel ({config.N_SPLITS}-fold CV)')
        ax.set_xticks(np.arange(n_good))

        ax = fig.add_subplot(gs[4, 1:3])
        ax.bar(np.arange(config.N_SPLITS), scores, color='steelblue', edgecolor='white', label='Fold accuracy')
        ax.axhline(scores.mean(), color='navy', linestyle='-', linewidth=2,
                   label=f'Mean = {scores.mean():.3f}')
        ax.axhline(chance, color='tomato', linestyle='--', linewidth=2,
                   label=f'Chance = {chance:.2f}')
        ax.set_ylim(0, 1)
        ax.set_xlabel('CV Fold'); ax.set_ylabel('Accuracy')
        ax.set_title('Decoding Accuracy (Logistic Regression)')
        ax.set_xticks(np.arange(config.N_SPLITS))
        ax.set_xticklabels([f'Fold {i+1}' for i in range(config.N_SPLITS)])
        ax.legend()

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close(fig)

        # Cell 46: printed summary.
        print("\n" + "=" * 55)
        print("SUMMARY")
        print("=" * 55)
        print(f"  Data shape           : {X_final.shape}  [Epochs x Channels x Time]")
        print(f"  Good channels        : {n_good}/{n_channels}")
        print(f"  Time window          : {TIME_WIN_START}-{TIME_WIN_END} ms  "
              f"({n_times} timepoints,  {ms_per_tp:.2f} ms/tp)")
        print(f"  Classes              : {list(unique_labels)}")
        print(f"  Encoding weights K   : {K.shape}  [Classes x Channels x Time]")
        print(f"  Best alpha (Ridge)   : {ridge.alpha_:.1f}")
        print(f"  Best ANOVA ch        : ch={np.argmax(f_map.max(axis=1))}  "
              f"peak F={f_map.max():.2f}")
        print(f"  Best CV R^2 channel  : ch={np.argmax(r2_cv)}  R^2={r2_cv.max():.4f}")
        print(f"  Best mean AUC ch     : ch={np.argmax(auc_map.mean(axis=0).max(axis=1))}  "
              f"AUC={auc_map.mean(axis=0).max():.3f}")
        print(f"  Decoding accuracy    : {scores.mean():.3f} +/- {scores.std():.3f}  "
              f"(chance = {chance:.2f})")

    return {
        'f_map': f_map,
        'sta': sta,
        'K': K,
        'R2_train': R2_train,
        'r2_cv': r2_cv,
        'auc_map': auc_map,
        'decoding_acc_mean': decoding_acc_mean,
        'decoding_acc_std': decoding_acc_std,
        'good_ch': good_ch,
        'n_good': n_good,
        'time_axis': time_axis,
        'unique_labels': unique_labels,
    }
