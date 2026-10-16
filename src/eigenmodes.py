"""§3.5 -- Eigenmode analysis.

For each eigenmode k we ask, separately for sycophantic and non-sycophantic states:
  3.5.1 energy |h_hat_k|^2  -- two-sample t-test + Cohen's d, Bonferroni-corrected
        across all d modes. "gain modes" = sycophantic states carry MORE energy,
        "loss modes" = LESS. We then look at where those modes sit in the eigenvalue
        (graph-frequency) spectrum.
  3.5.2 signed projections h_hat_k = phi_k^T h -- sign carries information that energy
        discards (same energy, opposite direction along the eigenvector).
  3.5.3 condition decomposition -- pressured / neutral / shift.
        A mode significant in pressured+shift but NOT neutral = "capitulation mode"
        (encodes the response to pressure).
        A mode significant in neutral                        = "vulnerability mode"
        (susceptibility is encoded before any pressure is applied).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .utils import log


def cohens_d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pooled-SD effect size, vectorized over the mode axis. a, b: (n_a, d), (n_b, d)."""
    na, nb = a.shape[0], b.shape[0]
    va, vb = a.var(axis=0, ddof=1), b.var(axis=0, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    return (a.mean(axis=0) - b.mean(axis=0)) / (pooled + 1e-30)


def mode_table(Hhat: np.ndarray, labels: np.ndarray, eigenvalues: np.ndarray,
               alpha: float = 0.05, use_energy: bool = True) -> pd.DataFrame:
    """One row per eigenmode: group means, Cohen's d, t, p, Bonferroni significance.

    Hhat   : (n, d) projections for ONE layer/basis/condition
    labels : (n,) boolean sycophancy labels
    """
    V = np.asarray(Hhat, dtype=np.float64)
    V = V ** 2 if use_energy else V
    syc, non = V[labels], V[~labels]
    d = V.shape[1]

    t, p = stats.ttest_ind(syc, non, axis=0, equal_var=False)
    dd = cohens_d(syc, non)
    thresh = alpha / d  # Bonferroni across all modes

    return pd.DataFrame({
        "mode": np.arange(d),
        "eigenvalue": np.asarray(eigenvalues, dtype=np.float64),
        "mean_syc": syc.mean(axis=0),
        "mean_non": non.mean(axis=0),
        "delta": syc.mean(axis=0) - non.mean(axis=0),
        "cohens_d": dd,
        "t_stat": t,
        "p_value": p,
        "significant": p < thresh,
        "bonferroni_threshold": thresh,
        "direction": np.where(dd > 0, "gain", "loss"),
    })


def energy_redistribution(Hhat: np.ndarray, labels: np.ndarray, eigenvalues: np.ndarray,
                          alpha: float = 0.05) -> dict:
    """§3.5.1 -- summary of where energy moves, plus total-energy comparison."""
    tbl = mode_table(Hhat, labels, eigenvalues, alpha, use_energy=True)
    sig = tbl[tbl["significant"]]
    gain = sig[sig["direction"] == "gain"]
    loss = sig[sig["direction"] == "loss"]

    total = (np.asarray(Hhat, dtype=np.float64) ** 2).sum(axis=1)
    t_tot, p_tot = stats.ttest_ind(total[labels], total[~labels], equal_var=False)

    summary = {
        "n_modes": int(len(tbl)),
        "bonferroni_threshold": float(tbl["bonferroni_threshold"].iloc[0]),
        "n_significant": int(len(sig)),
        "n_gain_modes": int(len(gain)),
        "n_loss_modes": int(len(loss)),
        "gain_eigenvalue_mean": float(gain["eigenvalue"].mean()) if len(gain) else None,
        "loss_eigenvalue_mean": float(loss["eigenvalue"].mean()) if len(loss) else None,
        "total_energy_syc_mean": float(total[labels].mean()),
        "total_energy_non_mean": float(total[~labels].mean()),
        "total_energy_t": float(t_tot),
        "total_energy_p": float(p_tot),
        "max_abs_cohens_d": float(tbl["cohens_d"].abs().max()),
        "top_modes_by_abs_d": tbl.reindex(tbl["cohens_d"].abs().sort_values(ascending=False).index)
                                 .head(20)[["mode", "eigenvalue", "cohens_d", "p_value"]]
                                 .to_dict("records"),
    }
    return {"table": tbl, "summary": summary}


def condition_decomposition(proj: Dict[str, np.ndarray], labels: np.ndarray,
                            eigenvalues: np.ndarray, alpha: float = 0.05,
                            use_energy: bool = False) -> pd.DataFrame:
    """§3.5.3 -- per-mode significance under pressured / neutral / shift, then classify.

    proj: {"pressured": (n,d), "neutral": (n,d), "shift": (n,d)} in ONE basis at ONE layer.
    Returns one row per mode with per-condition d/p and a mode_type label.
    """
    out = {}
    for cond in ("pressured", "neutral", "shift"):
        t = mode_table(proj[cond], labels, eigenvalues, alpha, use_energy=use_energy)
        out[cond] = t

    df = pd.DataFrame({"mode": out["pressured"]["mode"], "eigenvalue": out["pressured"]["eigenvalue"]})
    for cond in ("pressured", "neutral", "shift"):
        df[f"d_{cond}"] = out[cond]["cohens_d"].values
        df[f"p_{cond}"] = out[cond]["p_value"].values
        df[f"sig_{cond}"] = out[cond]["significant"].values

    def classify(r):
        if r["sig_neutral"]:
            return "vulnerability"          # susceptibility present before pressure
        if r["sig_pressured"] and r["sig_shift"]:
            return "capitulation"           # pure response to pressure
        if r["sig_pressured"] or r["sig_shift"]:
            return "partial"
        return "none"

    df["mode_type"] = df.apply(classify, axis=1)
    counts = df["mode_type"].value_counts().to_dict()
    log.info(f"  mode types: {counts}")
    return df


def top_modes(df: pd.DataFrame, condition: str = "pressured", k: int = 20) -> List[int]:
    """The k modes with the largest |Cohen's d| in a given condition."""
    col = f"d_{condition}" if f"d_{condition}" in df.columns else "cohens_d"
    return df.reindex(df[col].abs().sort_values(ascending=False).index)["mode"].head(k).astype(int).tolist()


def signed_projection_frame(proj: Dict[str, np.ndarray], modes: List[int]) -> pd.DataFrame:
    """§3.5.2 -- extract signed projections for selected modes into a tidy frame."""
    cols = {}
    for cond, H in proj.items():
        for m in modes:
            cols[f"mode{m}_{cond}"] = np.asarray(H[:, m], dtype=np.float64)
    return pd.DataFrame(cols)
