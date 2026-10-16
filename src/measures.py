"""§3.4 -- Spectral structure measures on h_hat = Phi^T h.

l1_l2  : ||h_hat||_1 / ||h_hat||_2, in [1, sqrt(d)]. 1 = all energy in one component
         (maximally sparse); sqrt(d) ~= 59.9 for d=3584 = perfectly uniform (maximally
         diffuse). Since Phi is orthonormal, ||h_hat||_2 = ||h||_2 (Parseval).
gini   : inequality of the energy distribution |h_hat_k|^2. 0 = uniform, 1 = one spike.
alpha  : negative slope of log |h_hat|^2_(k) vs log k over descending-sorted energies.
         Higher alpha = faster decay = energy concentrated in the top components.

All three are computed row-wise and vectorized over the experiment axis.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def l1_l2_ratio(Hhat: np.ndarray) -> np.ndarray:
    """(n, d) -> (n,). Basis-dependent diffuseness."""
    H = np.asarray(Hhat, dtype=np.float64)
    l1 = np.abs(H).sum(axis=1)
    l2 = np.linalg.norm(H, axis=1) + 1e-30
    return l1 / l2


def gini(Hhat: np.ndarray) -> np.ndarray:
    """Gini coefficient of the squared-magnitude distribution. (n, d) -> (n,)."""
    E = np.sort(np.asarray(Hhat, dtype=np.float64) ** 2, axis=1)  # ascending
    n, d = E.shape
    idx = np.arange(1, d + 1, dtype=np.float64)[None, :]
    total = E.sum(axis=1) + 1e-30
    # G = (2 * sum(i * x_(i)) / (d * sum(x))) - (d + 1) / d
    return (2.0 * (idx * E).sum(axis=1) / (d * total)) - (d + 1.0) / d


def powerlaw_alpha(Hhat: np.ndarray, floor_rel: float = 1e-12) -> np.ndarray:
    """§3.4.3 -- fit log|h_hat|^2_(k) = c - alpha * log k over all d ranks. (n, d) -> (n,).

    Energies are clipped at max*floor_rel before the log so that exact/near zeros
    (e.g. PCA null-space directions) cannot dominate the fit with -inf.
    Because the regressor log k is shared across rows, the OLS slope has a closed form.
    """
    E = np.sort(np.asarray(Hhat, dtype=np.float64) ** 2, axis=1)[:, ::-1]  # descending
    floor = np.maximum(E[:, :1] * floor_rel, 1e-300)
    E = np.maximum(E, floor)

    n, d = E.shape
    x = np.log(np.arange(1, d + 1, dtype=np.float64))
    y = np.log(E)
    xc = x - x.mean()
    yc = y - y.mean(axis=1, keepdims=True)
    slope = (yc @ xc) / (xc @ xc)
    return -slope  # alpha = negative slope


MEASURES = {
    "l1_l2": l1_l2_ratio,
    "gini": gini,
    "alpha": powerlaw_alpha,
}


def compute_measures(Hhat: np.ndarray, cfg=None) -> Dict[str, np.ndarray]:
    floor = getattr(cfg, "alpha_floor_rel", 1e-12) if cfg is not None else 1e-12
    return {
        "l1_l2": l1_l2_ratio(Hhat),
        "gini": gini(Hhat),
        "alpha": powerlaw_alpha(Hhat, floor),
    }


def geometry_features(neutral: np.ndarray, pressured: np.ndarray) -> Dict[str, np.ndarray]:
    """Condition-geometry features used in interpretation and the 11-feature set (§3.6.2)."""
    diff = pressured - neutral
    num = (neutral * pressured).sum(1)
    den = np.linalg.norm(neutral, axis=1) * np.linalg.norm(pressured, axis=1) + 1e-30
    return {
        "cos_neutral_pressured": (num / den).astype(np.float64),
        "shift_l2": np.linalg.norm(diff, axis=1).astype(np.float64),
    }
