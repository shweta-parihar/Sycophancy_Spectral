"""§3.6 -- Classification.

3.6.1 Single-feature threshold classifiers.
      For each (measure, basis, condition, layer) = 3 x 3 x 3 x 5 = 135 combinations,
      treat the scalar as a one-feature classifier: ROC-AUC, then the Youden-J-optimal
      threshold (J = sensitivity + specificity - 1). Both directions (above/below) are
      tested and the better is reported.

3.6.2 Combined logistic regression.
      L2-regularized LR (C = 1.0), 5-fold stratified CV, features standardized WITHIN
      each fold (fit on train, applied to val) so no scaling information leaks. We
      report mean/std AUC across folds plus precision/recall/F1/accuracy from the
      pooled out-of-fold predictions, and standardized coefficients from a full-data
      refit for feature importance.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .utils import log


# ---------------------------------------------------------------------------
# 3.6.1 single-feature threshold classifiers
# ---------------------------------------------------------------------------
def threshold_classifier(x: np.ndarray, y: np.ndarray) -> dict:
    """ROC-AUC + Youden-optimal operating point, testing both directions."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2 or np.allclose(x.std(), 0):
        return {"auc": float("nan"), "threshold": float("nan"), "direction": "above",
                "precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "accuracy": float("nan")}

    best = None
    for direction, score in (("above", x), ("below", -x)):
        auc = roc_auc_score(y, score)
        fpr, tpr, thr = roc_curve(y, score)
        j = tpr - fpr
        i = int(np.argmax(j))
        pred = (score >= thr[i]).astype(int)
        cand = {
            "auc": float(auc),
            "threshold": float(thr[i] if direction == "above" else -thr[i]),
            "direction": direction,
            "youden_j": float(j[i]),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
        }
        if best is None or cand["auc"] > best["auc"]:
            best = cand
    return best


def sweep_single_features(feature_df: pd.DataFrame, labels: np.ndarray,
                          measures: Sequence[str] = ("l1_l2", "gini", "alpha")) -> pd.DataFrame:
    """Run 3.6.1 over the full (measure, basis, condition, layer) grid.

    Reads columns named "{measure}__L{layer}__{basis}__{condition}" from the wide
    feature frame written by the `measures` stage.
    """
    rows = []
    for col in feature_df.columns:
        parts = col.split("__")
        if len(parts) != 4 or parts[0] not in measures:
            continue
        mname, layer, basis, cond = parts
        r = threshold_classifier(feature_df[col].to_numpy(), labels)
        r.update({"layer": int(layer[1:]), "basis": basis, "condition": cond,
                  "measure": mname, "feature": col})
        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    cols = ["measure", "basis", "condition", "layer", "auc", "threshold", "direction",
            "precision", "recall", "f1", "accuracy", "feature"]
    df = df[[c for c in cols if c in df.columns]].sort_values("auc", ascending=False)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3.6.2 cross-validated logistic regression
# ---------------------------------------------------------------------------
def cv_logistic(X: np.ndarray, y: np.ndarray, C: float = 1.0, n_folds: int = 5,
                seed: int = 0) -> dict:
    """Stratified CV with within-fold standardization. Returns metrics + OOF predictions."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    aucs, oof = [], np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])                     # fit on TRAIN only
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")  # L2 is the default penalty
        clf.fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[va]))[:, 1]
        oof[va] = p
        aucs.append(roc_auc_score(y[va], p) if len(np.unique(y[va])) > 1 else np.nan)

    pred = (oof >= 0.5).astype(int)
    return {
        "auc_mean": float(np.nanmean(aucs)),
        "auc_std": float(np.nanstd(aucs)),
        "auc_folds": [float(a) for a in aucs],
        "auc_oof": float(roc_auc_score(y, oof)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "n_features": int(X.shape[1]),
        "oof_pred": oof,
    }


def feature_importance(X: np.ndarray, y: np.ndarray, names: Sequence[str],
                       C: float = 1.0) -> pd.DataFrame:
    """Refit on the full dataset; standardized coefs are comparable across features."""
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")  # L2 is the default penalty
    clf.fit(sc.transform(X), y.astype(int))
    coef = clf.coef_[0]
    df = pd.DataFrame({"feature": list(names), "coef": coef, "abs_coef": np.abs(coef)})
    return df.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def evaluate_feature_sets(feature_df: pd.DataFrame, labels: np.ndarray,
                          feature_sets: Dict[str, List[str]], cfg) -> pd.DataFrame:
    """Run 3.6.2 across named feature sets of increasing complexity."""
    rows = []
    for name, cols in feature_sets.items():
        cols = [c for c in cols if c in feature_df.columns]
        if not cols:
            log.warning(f"feature set '{name}': no columns present, skipping")
            continue
        X = feature_df[cols].to_numpy(dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        r = cv_logistic(X, labels, cfg.lr_C, cfg.n_folds, cfg.cv_seed)
        r.pop("oof_pred")
        r.update({"feature_set": name, "features": ", ".join(cols)})
        rows.append(r)
        log.info(f"  {name:<38} n_feat={r['n_features']:<3} "
                 f"AUC={r['auc_mean']:.3f}±{r['auc_std']:.3f}  F1={r['f1']:.3f}")
    df = pd.DataFrame(rows)
    cols = ["feature_set", "n_features", "auc_mean", "auc_std", "auc_oof",
            "precision", "recall", "f1", "accuracy", "features"]
    return df[[c for c in cols if c in df.columns]].sort_values("auc_mean", ascending=False)
