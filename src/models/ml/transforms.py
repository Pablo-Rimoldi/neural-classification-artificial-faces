import numpy as np
from collections import Counter
from scipy.stats import mode
from sklearn.base import BaseEstimator, TransformerMixin


class IntraSubjectZScore(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        ids, feats = X[:, 0], X[:, 1:].copy()
        for s in np.unique(ids):
            m = ids == s
            mu = feats[m].mean(axis=0)
            sd = feats[m].std(axis=0)
            sd = np.where(sd == 0.0, 1.0, sd)
            feats[m] = (feats[m] - mu) / sd
        return feats


def decimate_to_odd(X, factor):
    Xd = np.asarray(X)[:, :, ::factor]
    if Xd.shape[2] % 2 == 0:
        Xd = Xd[:, :, :-1]
    return Xd


def spatial_flatten(X, y, subjects):
    X = np.asarray(X)
    n_trials, n_channels, n_tp = X.shape
    X_flat       = X.transpose(0, 2, 1).reshape(-1, n_channels)
    y_flat       = np.repeat(np.asarray(y), n_tp)
    subjects_flat = np.repeat(np.asarray(subjects), n_tp)
    trial_ids    = np.repeat(np.arange(n_trials), n_tp)
    return X_flat, y_flat, subjects_flat, trial_ids


def aggregate_trials(y_pred_tp, y_score_tp, y_true_tp, trial_ids):
    y_pred_tp  = np.asarray(y_pred_tp)
    y_score_tp = np.asarray(y_score_tp)
    y_true_tp  = np.asarray(y_true_tp)
    trial_ids  = np.asarray(trial_ids)
    y_true, y_pred, y_score = [], [], []
    for t_id in np.unique(trial_ids):
        m = trial_ids == t_id
        y_true.append(y_true_tp[m][0])
        vote, _ = mode(y_pred_tp[m], keepdims=False)
        y_pred.append(vote)
        y_score.append(y_score_tp[m].mean())
    return np.array(y_true), np.array(y_pred), np.array(y_score)


def groups_disjoint(train_groups, test_groups):
    return len(set(np.asarray(train_groups).tolist()) &
               set(np.asarray(test_groups).tolist())) == 0


def permute_trial_labels(y_tp, trial_ids, rng):
    y_tp      = np.asarray(y_tp)
    trial_ids = np.asarray(trial_ids)
    uniq = np.unique(trial_ids)
    trial_label = np.array([y_tp[trial_ids == t][0] for t in uniq])
    permuted    = trial_label[rng.permutation(len(uniq))]
    mapping     = dict(zip(uniq.tolist(), permuted))
    return np.array([mapping[t] for t in trial_ids.tolist()])


def summarize_selected_params(param_dicts):
    keys = set()
    for d in param_dicts:
        keys.update(d.keys())
    summary = {}
    for key in sorted(keys):
        counts = Counter()
        for d in param_dicts:
            if key not in d:
                continue
            value = d[key]
            label = (type(value).__name__
                     if hasattr(value, 'get_params') and
                     not isinstance(value, (int, float, str, bool))
                     else value)
            counts[label] += 1
        summary[key] = counts.most_common()
    return summary


def representative_params(param_dicts):
    def is_number(v):
        return (isinstance(v, (int, float, np.integer, np.floating))
                and not isinstance(v, bool))
    def all_int(vals):
        return all(isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                   for v in vals)
    def signature(v):
        if hasattr(v, 'get_params') and not isinstance(v, (int, float, str, bool)):
            return type(v).__name__
        return v
    keys = set().union(*(d.keys() for d in param_dicts)) if param_dicts else set()
    rep = {}
    for key in keys:
        vals = [d[key] for d in param_dicts if key in d]
        if vals and all(is_number(v) for v in vals):
            med = np.median(vals)
            rep[key] = int(round(med)) if all_int(vals) else float(med)
        else:
            top_sig = Counter(signature(v) for v in vals).most_common(1)[0][0]
            rep[key] = next(v for v in vals if signature(v) == top_sig)
    return rep


def permutation_p_value(observed, null_distribution):
    null = np.asarray(null_distribution, dtype=float)
    return (np.sum(null >= observed) + 1) / (null.size + 1)
