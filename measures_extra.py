"""Extra per-item spectral measures beyond l1/l2, Gini, alpha.

All take Hhat = Phi^T H  (n, d) projected coefficients, return (n,) one scalar per item.
Two families:
  CONCENTRATION (how spread the energy is; same question as l1/l2, different math):
      spec_entropy, part_ratio, topk_energy, kurtosis
  FREQUENCY-AWARE (WHERE the energy sits on the graph-frequency axis; needs eigenvalues):
      mean_freq (= normalized Dirichlet energy h^T L h / h^T h), spec_spread, rolloff, bandpow
  CROSS-CONDITION:
      spectral_flux(Hhat_neutral, Hhat_pressured)  -- per-item change in the energy spectrum

Frequency measures are only meaningful for bases with meaningful eigenvalues (the graph
Laplacians and PCA); the driver gates them to those bases.
"""
from __future__ import annotations
import numpy as np

_EPS = 1e-30


def _energy_fracs(Hhat):
    H = np.asarray(Hhat, dtype=np.float64)
    e = H * H
    tot = e.sum(axis=1, keepdims=True) + _EPS
    return e / tot                                  # (n, d) p_k per item


# ---------- concentration ----------
def spec_entropy(Hhat):
    """Shannon entropy of the energy spectrum. High = diffuse. Range [0, ln d]."""
    p = _energy_fracs(Hhat)
    return -(p * np.log(p + _EPS)).sum(axis=1)


def part_ratio(Hhat):
    """Participation ratio = (sum h^2)^2 / sum h^4 = 1/sum p_k^2. Effective #active modes."""
    H = np.asarray(Hhat, dtype=np.float64)
    e = H * H
    return (e.sum(axis=1) ** 2) / ((e ** 2).sum(axis=1) + _EPS)


def topk_energy(Hhat, frac=0.01):
    """Energy fraction captured by the top ceil(frac*d) modes. High = concentrated."""
    p = _energy_fracs(Hhat)
    d = p.shape[1]
    k = max(1, int(np.ceil(frac * d)))
    part = np.partition(p, d - k, axis=1)[:, d - k:]
    return part.sum(axis=1)


def kurtosis(Hhat):
    """Excess kurtosis of the coefficient distribution per item. High = heavy-tailed (few big coeffs)."""
    H = np.asarray(Hhat, dtype=np.float64)
    mu = H.mean(axis=1, keepdims=True)
    s2 = H.var(axis=1) + _EPS
    m4 = ((H - mu) ** 4).mean(axis=1)
    return m4 / (s2 ** 2) - 3.0


# ---------- frequency-aware (need eigenvalues lam, shape (d,)) ----------
def mean_freq(Hhat, lam):
    """Energy-weighted mean graph frequency c = sum lam_k p_k  (= normalized Dirichlet energy)."""
    p = _energy_fracs(Hhat)
    return p @ np.asarray(lam, dtype=np.float64)


def spec_spread(Hhat, lam):
    """Std of the graph-frequency distribution (bandwidth)."""
    p = _energy_fracs(Hhat); lam = np.asarray(lam, float)
    c = p @ lam
    var = p @ (lam ** 2) - c ** 2
    return np.sqrt(np.clip(var, 0, None))


def rolloff(Hhat, lam, q=0.85):
    """Graph frequency below which q of the energy lies (modes sorted by ascending lam)."""
    lam = np.asarray(lam, float)
    order = np.argsort(lam)
    p = _energy_fracs(Hhat)[:, order]
    cum = np.cumsum(p, axis=1)
    idx = (cum < q).sum(axis=1)                      # first index reaching q
    idx = np.clip(idx, 0, len(order) - 1)
    return lam[order][idx]


def bandpow(Hhat, lam):
    """Low-band / high-band energy ratio (split at the median eigenvalue). >1 = energy is low-freq."""
    lam = np.asarray(lam, float)
    med = np.median(lam)
    p = _energy_fracs(Hhat)
    low = p[:, lam <= med].sum(axis=1)
    high = p[:, lam > med].sum(axis=1) + _EPS
    return low / high


# ---------- cross-condition ----------
def spectral_flux(Hhat_a, Hhat_b):
    """Sum of squared change in the energy spectrum between two conditions (same basis)."""
    pa = _energy_fracs(Hhat_a); pb = _energy_fracs(Hhat_b)
    return ((pb - pa) ** 2).sum(axis=1)


CONCENTRATION = {"spec_entropy": spec_entropy, "part_ratio": part_ratio,
                 "topk_energy": topk_energy, "kurtosis": kurtosis}
FREQUENCY = {"mean_freq": mean_freq, "spec_spread": spec_spread,
             "rolloff": rolloff, "bandpow": bandpow}


def compute_concentration(Hhat):
    return {m: f(Hhat) for m, f in CONCENTRATION.items()}


def compute_frequency(Hhat, lam):
    return {m: f(Hhat, lam) for m, f in FREQUENCY.items()}
