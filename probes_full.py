"""Full-hidden-state baselines + CV single-feature scoring, all on the same StratifiedKFold.

  diff_of_means_auc / logreg_auc : CV AUC of the two probes on a condition's states (any layer).
  probe_by_layer                 : those two probes at EVERY layer -> depth curves.
  cv_single_feature_auc          : CV AUC of ONE scalar feature (direction chosen on train).
  logreg_direction               : fit a single logreg probe on full data, return its h-space
                                   weight vector (for projecting onto the eigenbasis).
  sparse_probe_curve             : L1-logreg at several strengths -> (n_nonzero, CV AUC) for the
                                   localized-vs-distributed test.
"""
from __future__ import annotations
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def _folds(y, n_folds, seed):
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)


def diff_of_means_auc(X, y, n_folds=5, seed=42):
    X = np.asarray(X, float); y = np.asarray(y).astype(int); aucs = []
    for tr, va in _folds(y, n_folds, seed).split(X, y):
        w = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
        if len(np.unique(y[va])) > 1:
            aucs.append(roc_auc_score(y[va], X[va] @ w))
    return float(np.nanmean(aucs)) if aucs else float("nan")


def logreg_auc(X, y, n_folds=5, seed=42, C=1.0):
    X = np.asarray(X, float); y = np.asarray(y).astype(int); aucs = []
    for tr, va in _folds(y, n_folds, seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[va]))[:, 1]
        if len(np.unique(y[va])) > 1:
            aucs.append(roc_auc_score(y[va], p))
    return float(np.nanmean(aucs)) if aucs else float("nan")


def cv_single_feature_auc(x, y, n_folds=5, seed=42):
    """CV ROC-AUC of one scalar feature; direction (sign) chosen on the train fold only."""
    x = np.asarray(x, float); y = np.asarray(y).astype(int)
    if np.allclose(np.nanstd(x), 0):
        return float("nan")
    aucs = []
    for tr, va in _folds(y, n_folds, seed).split(x.reshape(-1, 1), y):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        a_tr = roc_auc_score(y[tr], x[tr])
        sign = 1.0 if a_tr >= 0.5 else -1.0            # pick orientation on train
        aucs.append(roc_auc_score(y[va], sign * x[va]))
    return float(np.nanmean(aucs)) if aucs else float("nan")


def probe_by_layer(states, layers, y, conditions=("neutral", "pressured", "shift"),
                   n_folds=5, seed=42, C=1.0):
    """states: dict L{L}_neutral / L{L}_pressured. Returns {layer: {cond: {diff_means, logreg}}}."""
    out = {}
    for L in layers:
        n = states[f"L{L}_neutral"]; p = states[f"L{L}_pressured"]
        out[int(L)] = {}
        for cond in conditions:
            H = n if cond == "neutral" else p if cond == "pressured" else (p - n)
            out[int(L)][cond] = {"diff_means_auc": diff_of_means_auc(H, y, n_folds, seed),
                                 "logreg_auc": logreg_auc(H, y, n_folds, seed, C)}
    return out


def logreg_direction(X, y, C=1.0):
    """Fit one logreg on all rows (standardized); return the direction in ORIGINAL h-space."""
    X = np.asarray(X, float)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(sc.transform(X), np.asarray(y).astype(int))
    w = clf.coef_.ravel() / sc.scale_                  # undo standardization -> h-space direction
    return w / (np.linalg.norm(w) + 1e-30)


def sparse_probe_curve(X, y, Cs=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0), n_folds=5, seed=42):
    """L1-logreg at several strengths. Returns list of {C, n_nonzero_median, auc}."""
    X = np.asarray(X, float); y = np.asarray(y).astype(int)
    rows = []
    for C in Cs:
        aucs, nnz = [], []
        for tr, va in _folds(y, n_folds, seed).split(X, y):
            if len(np.unique(y[tr])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            # penalty="l1" + liblinear works across sklearn versions; silence the >=1.8 deprecation note.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = LogisticRegression(C=C, penalty="l1", solver="liblinear", max_iter=2000)
                clf.fit(sc.transform(X[tr]), y[tr])
            nnz.append(int((clf.coef_.ravel() != 0).sum()))
            if len(np.unique(y[va])) > 1:
                aucs.append(roc_auc_score(y[va], clf.predict_proba(sc.transform(X[va]))[:, 1]))
        rows.append({"C": C, "n_nonzero_median": int(np.median(nnz)) if nnz else 0,
                     "auc": float(np.nanmean(aucs)) if aucs else float("nan")})
    return rows
