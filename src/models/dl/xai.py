"""Explainability for the STCNN model: permutation importance and gradient saliency.

Ported from notebook cell 73 ("Permutation Importance" + "Saliency aggregated
over all folds" + the XAI figure). Divergences from the notebook: no module
globals for `np`/`device`/`fold_models`/`fold_te_indices`/`X_norm_dl`/`y_dl`/
`N_FOLDS_DL`/`time_ms` -- everything is passed explicitly as function
parameters, and `time_ms` is derived inside `gradient_saliency`/`plot_xai`
from `X_norm.shape[2]` (`np.linspace(200, 600, n_times)`).
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import find_peaks, savgol_filter
from sklearn.metrics import accuracy_score

from src import config

# ERP time windows highlighted on the heatmap / temporal-dynamics plots.
ERP_WINDOWS = [
    (170, 230, '#00bcd4', 'N170'),
    (280, 350, '#8bc34a', 'N2/P3a'),
    (380, 500, '#ff9800', 'LPP'),
    (530, 600, '#f44336', 'P600'),
]


def channel_names() -> list:
    """Return the 19 channel/feature names, in tensor-column order."""
    return [
        'O1', 'O2', 'PO9', 'PO10', 'TP7', 'TP8', 'P3', 'P4',
        'AF3', 'AF4', 'AFF1h', 'AFF2h', 'AFF3h', 'AFF4h',
        'PCA_Frontal', 'PCA_Parietal', 'PCA_Occipital', 'PCA_Temporal', 'FaceSEX'
    ]


def _get_acc(model, X_data, y_data, device):
    """Accuracy of `model` on (X_data, y_data), run on `device`."""
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X_data, dtype=torch.float32).to(device))
        preds = out.argmax(1).cpu().numpy()
    return accuracy_score(y_data, preds)


def permutation_importance(fold_models, fold_te_indices, X_norm, y, device, n_repeats=20):
    """Permutation importance aggregated over folds (test sets only).

    For each channel, shuffles that channel's values (within each fold's
    test set) `n_repeats` times and measures the resulting accuracy drop,
    averaged across folds. Returns (importances, std) of shape (19,).
    """
    n_channels = len(channel_names())

    base_accs = [
        _get_acc(m, X_norm[te], y[te], device)
        for m, te in zip(fold_models, fold_te_indices)
    ]

    perm_matrix = np.zeros((n_channels, n_repeats))

    for c in range(n_channels):
        for r in range(n_repeats):
            drops = []
            for k, (model_k, te_idx) in enumerate(zip(fold_models, fold_te_indices)):
                X_te, y_te = X_norm[te_idx], y[te_idx]
                X_sh = X_te.copy()
                np.random.shuffle(X_sh[:, c, :])
                drops.append(base_accs[k] - _get_acc(model_k, X_sh, y_te, device))
            perm_matrix[c, r] = np.mean(drops)

    perm_importances = perm_matrix.mean(axis=1)
    perm_std = perm_matrix.std(axis=1)
    return perm_importances, perm_std


def gradient_saliency(fold_models, fold_te_indices, X_norm, y, device):
    """Gradient saliency aggregated over all folds (test sets only).

    For each fold, computes |d(correct-class logit)/d(input)| averaged over
    the fold's test samples, yielding a (channels, timesteps) map per fold.
    Returns the mean and std over folds, each of shape (19, n_times).
    """
    fold_saliency_maps = []

    for model_k, te_idx in zip(fold_models, fold_te_indices):
        model_k.eval()
        X_te, y_te = X_norm[te_idx], y[te_idx]
        X_tensor_k = torch.tensor(X_te, dtype=torch.float32, requires_grad=True, device=device)
        Y_tensor_k = torch.tensor(y_te, device=device)

        model_k.zero_grad()
        logits_k = model_k(X_tensor_k)
        correct_logits_k = logits_k[torch.arange(len(y_te)), Y_tensor_k]
        correct_logits_k.sum().backward()

        sal_k = X_tensor_k.grad.abs().mean(dim=0).cpu().numpy()  # (channels, timesteps)
        fold_saliency_maps.append(sal_k)

    saliency_map = np.mean(fold_saliency_maps, axis=0)
    saliency_std_map = np.std(fold_saliency_maps, axis=0)
    return saliency_map, saliency_std_map


def _ch_color_perm(name, val):
    if name == 'FaceSEX':
        return '#ff9800'
    if val <= 0:
        return '#9e9e9e'
    return '#d32f2f'


def _ch_color_sal(name):
    if name == 'FaceSEX':
        return '#ff9800'
    if name.startswith('PCA'):
        return '#1565c0'
    return '#1976d2'


def _ch_color_scatter(name):
    if name == 'FaceSEX':
        return '#ff9800'
    if name.startswith('PCA'):
        return '#1565c0'
    return '#d32f2f'


def _minmax(v):
    return (v - v.min()) / (v.max() - v.min() + 1e-10)


def plot_xai(perm_importances, perm_std, saliency_map, *, n_folds, save_path=None):
    """Render the 5-panel XAI figure (cell 73) from precomputed XAI arrays.

    `saliency_map` has shape (19, n_times); `n_times` drives the time axis
    (`np.linspace(200, 600, n_times)`). If `save_path` is given, the figure
    is written there (e.g. via `config.results_path('xai_analysis.png')`);
    otherwise the figure is returned without being saved. Headless (Agg
    backend) -- no display is required.
    """
    names = channel_names()
    n_times = saliency_map.shape[1]
    time_ms = np.linspace(200, 600, n_times)

    channel_saliency = saliency_map.mean(axis=1)
    temporal_saliency = saliency_map.mean(axis=0)
    temporal_smooth = savgol_filter(temporal_saliency, window_length=11, polyorder=3)

    sal_perrow = np.zeros_like(saliency_map)
    for i in range(saliency_map.shape[0]):
        row = saliency_map[i]
        rmin, rmax = row.min(), row.max()
        sal_perrow[i] = (row - rmin) / (rmax - rmin + 1e-10)

    peaks_idx, props = find_peaks(
        temporal_smooth,
        prominence=temporal_smooth.std() * 0.6,
        distance=max(1, int(len(time_ms) * 0.05)),
    )
    top5 = peaks_idx[np.argsort(props['prominences'])[-5:]] if len(peaks_idx) > 5 else peaks_idx

    perm_n = _minmax(perm_importances)
    sal_n = _minmax(channel_saliency)

    fig = plt.figure(figsize=(22, 20))
    gs = gridspec.GridSpec(
        3, 2,
        height_ratios=[1.2, 1.2, 0.9],
        width_ratios=[1, 1.5],
        hspace=0.42, wspace=0.28
    )

    # Permutation Importance (aggregated over folds)
    ax0 = fig.add_subplot(gs[0, 0])
    sort_idx = np.argsort(perm_importances)
    s_names_a = np.array(names)[sort_idx]
    s_imp = perm_importances[sort_idx]
    s_std = perm_std[sort_idx]
    colors_a = [_ch_color_perm(n, v) for n, v in zip(s_names_a, s_imp)]

    ax0.barh(s_names_a, s_imp, color=colors_a, edgecolor='black', height=0.72, zorder=3)
    ax0.errorbar(s_imp, np.arange(len(s_names_a)), xerr=s_std, fmt='none',
                 color='#212121', capsize=3.5, linewidth=1.6, zorder=4)
    ax0.axvline(0, color='black', linewidth=1)
    ax0.set_xlabel('Accuracy Drop (Delta)', fontsize=12)
    ax0.set_title(f'A. Permutation Importance\n(Aggregated over {n_folds} folds, test-set, N=20)', fontsize=13, fontweight='bold')
    ax0.grid(axis='x', alpha=0.3, zorder=0)
    ax0.legend(handles=[
        mpatches.Patch(color='#d32f2f', label='EEG / PCA'),
        mpatches.Patch(color='#ff9800', label='FaceSEX (Behavioral)'),
        mpatches.Patch(color='#9e9e9e', label='Delta <= 0 (Irrelevant)')
    ], fontsize=8.5, loc='lower right')

    # Channel Saliency (mean over all folds)
    ax1 = fig.add_subplot(gs[1, 0])
    sort_idx_b = np.argsort(channel_saliency)
    s_names_b = np.array(names)[sort_idx_b]
    s_sal = channel_saliency[sort_idx_b]
    colors_b = [_ch_color_sal(n) for n in s_names_b]

    ax1.barh(s_names_b, s_sal, color=colors_b, edgecolor='black', height=0.72)
    ax1.set_xlabel('Mean Gradient Magnitude', fontsize=12)
    ax1.set_title(f'B. Channel Saliency\n(Mean over {n_folds} folds - more generalizable)', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.legend(handles=[
        mpatches.Patch(color='#1976d2', label='Raw EEG Channels'),
        mpatches.Patch(color='#1565c0', label='PCA Features'),
        mpatches.Patch(color='#ff9800', label='FaceSEX'),
    ], fontsize=8.5, loc='lower right')

    # Scatter - Spatial Filter Dissociation
    ax4 = fig.add_subplot(gs[2, 0])
    sc_colors = [_ch_color_scatter(n) for n in names]
    ax4.scatter(perm_n, sal_n, c=sc_colors, s=80, edgecolors='black', linewidths=0.8, zorder=5)

    divergence = np.abs(perm_n - sal_n)
    for i, (x, y_pt, name) in enumerate(zip(perm_n, sal_n, names)):
        if divergence[i] > 0.28 or name == 'FaceSEX':
            ax4.annotate(name, (x, y_pt), fontsize=7.5,
                         xytext=(5, 4), textcoords='offset points',
                         arrowprops=dict(arrowstyle='-', color='gray', lw=0.8))

    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.35, linewidth=1.5, label='Perfect Agreement')
    ax4.fill_betweenx([0, 1], [0.4, 0.4], [1, 1], alpha=0.04, color='red')
    ax4.fill_betweenx([0, 1], [0, 0], [0.6, 0.6], alpha=0.04, color='blue')
    ax4.text(0.72, 0.08, 'Bottleneck\nChannel', fontsize=8, color='#b71c1c', ha='center')
    ax4.text(0.15, 0.75, 'Redundant\nChannel', fontsize=8, color='#0d47a1', ha='center')

    ax4.set_xlabel('Permutation Importance (Norm)', fontsize=11)
    ax4.set_ylabel('Gradient Saliency (Norm)', fontsize=11)
    ax4.set_title('E. Spatial Filter Dissociation\n(Bottleneck vs. Redundant Channels)', fontsize=13, fontweight='bold')
    ax4.grid(alpha=0.3)
    ax4.legend(fontsize=9)
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)

    # Spatiotemporal Heatmap (aggregated saliency)
    ax2 = fig.add_subplot(gs[0:2, 1])
    im = ax2.imshow(
        sal_perrow,
        aspect='auto',
        cmap='magma',
        extent=[time_ms[0], time_ms[-1], -0.5, len(names) - 0.5],
        origin='lower',
        interpolation='nearest',
        vmin=0, vmax=1
    )
    cbar = plt.colorbar(im, ax=ax2, pad=0.02)
    cbar.set_label('Saliency (Row-normalized)', fontsize=11)

    ax2.set_yticks(np.arange(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('EEG Channels / Features', fontsize=12)
    ax2.set_title(f'C. Spatiotemporal Heatmap (Mean {n_folds} folds, Row-normalized)\nwith ERP Windows', fontsize=13, fontweight='bold')

    for (t1, t2, col, label) in ERP_WINDOWS:
        if t1 >= time_ms[0]:
            ax2.axvspan(t1, t2, color=col, alpha=0.12, zorder=0)
            ax2.text((t1 + t2) / 2, len(names) - 0.3, label,
                     ha='center', va='top', fontsize=8.5, color=col, fontweight='bold')

    # Temporal Dynamics with inter-fold confidence band
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.plot(time_ms, temporal_saliency, color='#e65100', linewidth=1.0, alpha=0.35, label='Raw (fold mean)')
    ax3.plot(time_ms, temporal_smooth, color='#e65100', linewidth=2.5, label='Smoothed (Savitzky-Golay)')

    for (t1, t2, col, label) in ERP_WINDOWS:
        if t1 >= time_ms[0]:
            ax3.axvspan(t1, t2, color=col, alpha=0.12)
            ax3.text((t1 + t2) / 2, temporal_smooth.max() * 0.97, label,
                     ha='center', va='top', fontsize=8.5, color=col, fontweight='bold')

    for p in top5:
        ax3.axvline(time_ms[p], color='#212121', linestyle='--', alpha=0.7, linewidth=1.5)
        ax3.text(time_ms[p] + 3, temporal_smooth[p] * 1.03, f"{time_ms[p]:.0f} ms",
                 fontweight='bold', fontsize=9.5)

    ax3.set_xlabel('Time (ms)', fontsize=12)
    ax3.set_ylabel('Global Saliency', fontsize=12)
    ax3.set_title(f'D. Temporal Dynamics - Mean {n_folds} folds\n(Prominence-based Top 5 Peaks)', fontsize=13, fontweight='bold')
    ax3.grid(alpha=0.4)
    ax3.legend(fontsize=10, loc='upper left')

    plt.suptitle(f"XAI - SpatialTemporalCNN (STCNN): Saliency aggregated over {n_folds} folds",
                 fontsize=18, fontweight='bold', y=0.995)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
