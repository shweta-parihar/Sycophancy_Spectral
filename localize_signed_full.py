"""Localization ("where does sycophancy live") on the SIGNED-Laplacian eigenbasis.

Detection asks *can we predict* caving; localization asks *which part of the representation
carries it*. All four run at the last layer on the signed basis (chosen on principle: it is the
only non-degenerate construction, so its eigenvalues are real graph frequencies).

  mode_frequency_profile : per-eigenmode Cohen's d + CV single-mode AUC, paired with the mode's
                           eigenvalue (graph frequency). -> "is the signal in low or high modes?"
  band_energy_profile    : split modes into frequency bands; compare per-item band energy between
                           caved and resisted. -> "do caved states move energy between bands?"
  probe_direction_profile: fit a full-h logreg probe, rotate its direction into the eigenbasis,
                           report |w_k|^2 vs frequency (+ its spectral centroid). -> "what
                           frequencies does the discriminative direction use?"
  sparse_probe_profile    : L1 probe at increasing strength -> (n_nonzero dims, AUC). -> "localized
                           (few dims reach the ceiling) or distributed?"

Inputs per cell (last layer):
  proj   : {"neutral": (n,d), "pressured": (n,d)}  states projected onto the signed eigenbasis
  lam    : (d,)  signed-Laplacian eigenvalues, ascending (graph frequency)
  basis  : (d,d) eigenvectors Phi (columns), for rotating a h-space direction into mode space
  Hraw   : {"neutral": (n,d), "pressured": (n,d)}  raw hidden states (for probe direction + L1)
  y      : (n,) bool, caved (1) vs resisted (0)
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import probes_full as PRB


def _cohens_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / max(na + nb - 2, 1))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def _energy_fracs(H):
    """Per-row fraction of energy in each coordinate: p_ik = H_ik^2 / sum_k H_ik^2."""
    P = np.asarray(H, float) ** 2
    return P / (P.sum(1, keepdims=True) + 1e-30)


def mode_frequency_profile(proj, lam, y, condition="pressured", n_folds=5, seed=42):
    """Per signed-eigenmode: Cohen's d (caved vs resisted) and CV single-mode AUC, vs eigenvalue."""
    H = proj["pressured"] if condition == "pressured" else (
        proj["neutral"] if condition == "neutral" else proj["pressured"] - proj["neutral"])
    y = np.asarray(y).astype(bool)
    lam = np.asarray(lam, float)
    rows = []
    for k in range(H.shape[1]):
        x = H[:, k]
        rows.append({"mode": int(k), "eigenvalue": float(lam[k]),
                     "cohen_d": _cohens_d(x[y], x[~y]),
                     "abs_cohen_d": abs(_cohens_d(x[y], x[~y])),
                     "auc": PRB.cv_single_feature_auc(x, y, n_folds, seed)})
    return rows


def band_energy_profile(proj, lam, y, condition="pressured", n_bands=5):
    """Split modes into n_bands equal-count frequency bands; per item, energy fraction in each band.
    Compare caved vs resisted (Cohen's d on the band-energy fraction). Band 0 = lowest frequency."""
    H = proj["pressured"] if condition == "pressured" else (
        proj["neutral"] if condition == "neutral" else proj["pressured"] - proj["neutral"])
    y = np.asarray(y).astype(bool)
    lam = np.asarray(lam, float)
    order = np.argsort(lam)                      # ascending frequency
    bands = np.array_split(order, n_bands)
    P = _energy_fracs(H)                          # (n,d)
    rows = []
    for bi, idx in enumerate(bands):
        be = P[:, idx].sum(1)                     # per-row energy fraction in this band
        rows.append({"band": bi, "n_modes": int(len(idx)),
                     "lam_lo": float(lam[idx].min()), "lam_hi": float(lam[idx].max()),
                     "energy_caved": float(be[y].mean()), "energy_resisted": float(be[~y].mean()),
                     "cohen_d": _cohens_d(be[y], be[~y])})
    return rows


def probe_direction_profile(Hraw, basis, lam, y, condition="shift", C=1.0, n_bins=10):
    """Fit a full-h logreg probe, rotate its h-space direction into the signed eigenbasis, and
    report how its weight energy |w_k|^2 is distributed over graph frequency (+ spectral centroid)."""
    H = Hraw["pressured"] if condition == "pressured" else (
        Hraw["neutral"] if condition == "neutral" else Hraw["pressured"] - Hraw["neutral"])
    w_h = PRB.logreg_direction(H, y, C=C)               # unit direction in h-space
    w_mode = basis.T @ w_h                               # rotate into eigenbasis: w_k
    e = w_mode ** 2; e = e / (e.sum() + 1e-30)           # energy per mode (sums to 1)
    lam = np.asarray(lam, float)
    centroid = float((lam * e).sum())                    # spectral centroid of the probe direction
    # cumulative energy vs ascending frequency + coarse per-bin energy for a bar chart
    order = np.argsort(lam)
    cum = np.cumsum(e[order])
    bins = np.array_split(order, n_bins)
    binned = [{"bin": i, "lam_lo": float(lam[ix].min()), "lam_hi": float(lam[ix].max()),
               "energy": float(e[ix].sum())} for i, ix in enumerate(bins)]
    per_mode = [{"mode": int(k), "eigenvalue": float(lam[k]), "w_energy": float(e[k])}
                for k in range(len(e))]
    return {"spectral_centroid": centroid, "lam_mean": float(lam.mean()),
            "energy_below_median_freq": float(e[lam <= np.median(lam)].sum()),
            "binned_energy": binned, "cum_energy_sorted": cum.tolist(), "per_mode": per_mode}


def sparse_probe_profile(Hraw, y, condition="shift", Cs=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0),
                         n_folds=5, seed=42):
    """L1 probe on full h at several strengths -> (n_nonzero dims, CV AUC). Also the full-h logreg
    ceiling for reference, so the report can say 'd* dims reach X% of the ceiling'."""
    H = Hraw["pressured"] if condition == "pressured" else (
        Hraw["neutral"] if condition == "neutral" else Hraw["pressured"] - Hraw["neutral"])
    curve = PRB.sparse_probe_curve(H, y, Cs=Cs, n_folds=n_folds, seed=seed)
    ceiling = PRB.logreg_auc(H, y, n_folds=n_folds, seed=seed)
    d = H.shape[1]
    return {"dim": int(d), "full_logreg_auc": ceiling, "curve": curve}


def run_localization(proj_signed, lam_signed, basis_signed, Hraw, y,
                     cond_mode="pressured", cond_probe="shift", n_bands=5, C=1.0,
                     n_folds=5, seed=42):
    """All four analyses on the signed basis. Returns a dict ready to save as JSON."""
    y = np.asarray(y).astype(bool)
    if y.sum() < 3 or (~y).sum() < 3:
        return {"skipped": "too few caved/resisted for localization",
                "n_caved": int(y.sum()), "n_resisted": int((~y).sum())}
    return {
        "basis": "signed", "cond_mode": cond_mode, "cond_probe": cond_probe,
        "n_caved": int(y.sum()), "n_resisted": int((~y).sum()),
        "mode_frequency": mode_frequency_profile(proj_signed, lam_signed, y, cond_mode, n_folds, seed),
        "band_energy": band_energy_profile(proj_signed, lam_signed, y, cond_mode, n_bands),
        "probe_direction": probe_direction_profile(Hraw, basis_signed, lam_signed, y, cond_probe, C),
        "sparse_probe": sparse_probe_profile(Hraw, y, cond_probe, n_folds=n_folds, seed=seed),
    }
