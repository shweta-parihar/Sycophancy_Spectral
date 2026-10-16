"""§3.7 -- Interpretive analysis.

What does an eigenmode actually encode? Three complementary views:

  1. Pearson correlations (with p-values) between a mode's signed projection and
     interpretable properties, reported separately for the pressured projection,
     the neutral projection, and the shift.
  2. Quartile analysis: split experiments into four equal groups by the mode's
     projection and report sycophancy rate / correct overlap / agreement rate /
     truthfulness rate per quartile. Nonparametric and easy to read.
  3. Category analysis: mean projection for sycophantic vs non-sycophantic
     experiments within each TruthfulQA category -- is the signature domain-general
     or domain-specific?
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .utils import log

# §3.7 interpretable properties. Columns come from the labeled-responses frame plus
# geometry features computed in measures.geometry_features.
PROPERTY_COLUMNS = [
    "sycophantic",
    "correct_overlap",
    "wrong_overlap",
    "agree_pressured",
    "agree_neutral",
    "truthful",
    "response_len",
    "hedge_count",
    "cos_neutral_pressured",
    "shift_l2",
    "question_difficulty",
]


def add_derived_properties(df: pd.DataFrame, geom: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Attach geometry features and a question-difficulty proxy."""
    out = df.copy()
    for k, v in geom.items():
        out[k] = v
    # Difficulty proxy: how many distinct incorrect answers the question attracts,
    # normalized. TruthfulQA has no difficulty field; this is a documented stand-in.
    if "incorrect_answers" in out.columns:
        n_wrong = out["incorrect_answers"].apply(lambda a: len(a) if isinstance(a, (list, tuple)) else 0)
        out["question_difficulty"] = (n_wrong - n_wrong.mean()) / (n_wrong.std() + 1e-9)
    return out


def correlate_mode(proj: Dict[str, np.ndarray], mode: int, props: pd.DataFrame,
                   columns: Optional[List[str]] = None) -> pd.DataFrame:
    """§3.7 -- Pearson r + p between one mode's projection and each property,
    for each of the pressured / neutral / shift conditions."""
    columns = [c for c in (columns or PROPERTY_COLUMNS) if c in props.columns]
    rows = []
    for cond, H in proj.items():
        x = np.asarray(H[:, mode], dtype=np.float64)
        for c in columns:
            y = pd.to_numeric(props[c], errors="coerce").to_numpy(dtype=np.float64)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 3 or np.std(y[ok]) == 0:
                r, p = np.nan, np.nan
            else:
                r, p = stats.pearsonr(x[ok], y[ok])
            rows.append({"mode": mode, "condition": cond, "property": c,
                         "pearson_r": float(r), "p_value": float(p), "n": int(ok.sum())})
    df = pd.DataFrame(rows)
    return df.reindex(df["pearson_r"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def correlate_modes(proj: Dict[str, np.ndarray], modes: List[int],
                    props: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([correlate_mode(proj, m, props) for m in modes], ignore_index=True)


def quartile_analysis(proj: np.ndarray, mode: int, props: pd.DataFrame) -> pd.DataFrame:
    """§3.7 -- split into 4 equal groups by projection value; summarize each."""
    x = np.asarray(proj[:, mode], dtype=np.float64)
    q = pd.qcut(pd.Series(x).rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    df = props.copy()
    df["_q"] = q.values
    df["_x"] = x

    agg = {"_x": "mean"}
    for c in ("sycophantic", "correct_overlap", "wrong_overlap", "agree_pressured", "truthful",
              "response_len", "hedge_count"):
        if c in df.columns:
            agg[c] = "mean"
    out = df.groupby("_q", observed=True).agg(agg).reset_index()
    out = out.rename(columns={"_q": "quartile", "_x": "mean_projection",
                              "sycophantic": "sycophancy_rate",
                              "agree_pressured": "agreement_rate",
                              "truthful": "truthfulness_rate"})
    out.insert(0, "mode", mode)
    out["n"] = df.groupby("_q", observed=True).size().values
    return out


def category_analysis(proj: np.ndarray, mode: int, props: pd.DataFrame,
                      min_n: int = 5) -> pd.DataFrame:
    """§3.7 -- mean projection per category, split by sycophancy label."""
    x = np.asarray(proj[:, mode], dtype=np.float64)
    df = props[["category", "sycophantic"]].copy()
    df["proj"] = x

    rows = []
    for cat, g in df.groupby("category"):
        syc, non = g[g["sycophantic"] == True], g[g["sycophantic"] == False]  # noqa: E712
        if len(g) < min_n:
            continue
        rows.append({
            "mode": mode, "category": cat, "n": len(g),
            "n_syc": len(syc), "n_non": len(non),
            "syc_rate": float(g["sycophantic"].mean()),
            "mean_proj_syc": float(syc["proj"].mean()) if len(syc) else np.nan,
            "mean_proj_non": float(non["proj"].mean()) if len(non) else np.nan,
            "separation": (float(syc["proj"].mean() - non["proj"].mean())
                           if len(syc) and len(non) else np.nan),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("n", ascending=False).reset_index(drop=True) if len(out) else out
