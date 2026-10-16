"""Full-hidden-state baselines, scored by cross-validated ROC-AUC so they drop straight
into the same table as the spectral single-feature / combined AUCs (SAME folds via seed).

  diff_of_means_auc : train-free honesty-lit standard. Direction w = mean(h|y=1) - mean(h|y=0)
                      fit on the TRAIN fold, score = h . w on the held-out fold.
  logreg_auc        : L2 logistic regression on full h (the linear-probe capacity ceiling),
                      StandardScaler fit on train only.

Both return the mean held-out AUC over StratifiedKFold(n_folds, seed). Give them the raw
last-layer states for a condition (neutral / pressured / shift) to get one AUC per condition.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def _cv(X, y, n_folds, seed):
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y).astype(int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return X, y, skf


def diff_of_means_auc(X, y, n_folds=5, seed=42):
    X, y, skf = _cv(X, y, n_folds, seed)
    aucs = []
    for tr, va in skf.split(X, y):
        mu1 = X[tr][y[tr] == 1].mean(axis=0)
        mu0 = X[tr][y[tr] == 0].mean(axis=0)
        w = mu1 - mu0
        score = X[va] @ w
        if len(np.unique(y[va])) > 1:
            aucs.append(roc_auc_score(y[va], score))
    return float(np.nanmean(aucs)) if aucs else float("nan")


def logreg_auc(X, y, n_folds=5, seed=42, C=1.0):
    X, y, skf = _cv(X, y, n_folds, seed)
    aucs = []
    for tr, va in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[va]))[:, 1]
        if len(np.unique(y[va])) > 1:
            aucs.append(roc_auc_score(y[va], p))
    return float(np.nanmean(aucs)) if aucs else float("nan")


def probe_table(states_last, y, conditions=("neutral", "pressured", "shift"), n_folds=5, seed=42, C=1.0):
    """states_last: dict with 'neutral','pressured' (n,d). Returns {cond: {dom_auc, logreg_auc}}."""
    out = {}
    for cond in conditions:
        if cond == "neutral":   H = states_last["neutral"]
        elif cond == "pressured": H = states_last["pressured"]
        elif cond == "shift":   H = states_last["pressured"] - states_last["neutral"]
        else: continue
        out[cond] = {"diff_means_auc": diff_of_means_auc(H, y, n_folds, seed),
                     "logreg_auc":     logreg_auc(H, y, n_folds, seed, C)}
    return out
