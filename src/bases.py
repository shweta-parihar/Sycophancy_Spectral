"""§3.3 -- Spectral basis construction.

Three orthonormal bases of R^d, all built from POOLED neutral+pressured states
(1634 x 3584 at each layer) and therefore blind to the sycophancy labels:

  identity   -- standard coordinate basis (no change of basis; the baseline)
  pca        -- eigenvectors of the covariance of the pooled states, eigenvalue-descending
  laplacian  -- eigenvectors of the symmetric normalized Laplacian of the
                dimension-correlation graph, eigenvalue-ascending

The Laplacian construction (§3.3.1-3.3.2):
    X in R^{1634 x 3584}; each COLUMN is one dimension's profile across experiments
    C = corr(columns of X)               in [-1, 1]^{3584 x 3584}
    W = (C + 1) / 2, diag(W) = 0         non-negative similarity, no self-loops
    D = diag(W 1)
    L = I - D^{-1/2} W D^{-1/2}          symmetric, PSD, eigenvalues in [0, 2]
    L = Phi Lambda Phi^T                 orthonormal eigenbasis, ascending eigenvalues

Low graph frequency (small lambda) = smooth across highly-correlated dimensions.
High graph frequency (large lambda) = oscillating across uncorrelated dimensions.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .utils import log, timed


def dimension_correlation(X: np.ndarray) -> np.ndarray:
    """Pearson correlation between dimension profiles (columns of X). Returns (d, d)."""
    X = np.asarray(X, dtype=np.float64)
    sd = X.std(axis=0)
    dead = sd < 1e-12
    if dead.any():
        log.warning(f"{int(dead.sum())} dimensions have ~zero variance; "
                    f"their correlations are set to 0 (self-correlation 1).")
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    if dead.any():
        C[dead, :] = 0.0
        C[:, dead] = 0.0
        C[dead, dead] = 1.0
    np.clip(C, -1.0, 1.0, out=C)
    return C


def normalized_laplacian(C: np.ndarray) -> np.ndarray:
    """§3.3.2 -- W = (C+1)/2 with zero diagonal, then L = I - D^-1/2 W D^-1/2."""
    W = (C + 1.0) / 2.0
    np.fill_diagonal(W, 0.0)
    deg = W.sum(axis=1)
    if (deg <= 0).any():
        log.warning(f"{int((deg <= 0).sum())} isolated dimensions; degree floored at 1e-12.")
        deg = np.maximum(deg, 1e-12)
    dinv = 1.0 / np.sqrt(deg)
    L = -(W * dinv[:, None]) * dinv[None, :]
    np.fill_diagonal(L, 1.0 + np.diag(L))  # add I
    L = (L + L.T) / 2.0                    # enforce exact symmetry against fp drift
    return L


def _eigh(M: np.ndarray, backend: str) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric eigendecomposition, ascending eigenvalues."""
    if backend == "torch":
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        t = torch.from_numpy(np.ascontiguousarray(M)).to(dev, torch.float64)
        w, v = torch.linalg.eigh(t)
        return w.cpu().numpy(), v.cpu().numpy()
    from scipy.linalg import eigh  # §3.3.2 specifies scipy.linalg.eigh

    return eigh(M)


def build_laplacian_basis(X: np.ndarray, backend: str = "scipy") -> Dict[str, np.ndarray]:
    with timed(f"laplacian basis (d={X.shape[1]}, backend={backend})"):
        C = dimension_correlation(X)
        L = normalized_laplacian(C)
        w, Phi = _eigh(L, backend)  # ascending
    log.info(f"  eigenvalue range [{w.min():.3f}, {w.max():.3f}]  "
             f"(normalized Laplacian bound: [0, 2])")
    return {"eigenvalues": w.astype(np.float64), "basis": Phi.astype(np.float32)}


def build_pca_basis(X: np.ndarray, backend: str = "scipy") -> Dict[str, np.ndarray]:
    """§3.3.3 -- eigenvectors of the covariance matrix, sorted by DECREASING eigenvalue.

    Note: with n=1634 samples in d=3584 dimensions the covariance has rank <= n-1,
    so ~1951 trailing eigenvectors span the null space and their eigenvalues are ~0.
    They are still orthonormal, so the basis is valid; see README note D3.
    """
    with timed(f"pca basis (n={X.shape[0]}, d={X.shape[1]})"):
        Xc = np.asarray(X, dtype=np.float64)
        cov = np.cov(Xc, rowvar=False)
        w, V = _eigh(cov, backend)
        order = np.argsort(w)[::-1]  # descending
        w, V = w[order], V[:, order]
    rank = int((w > max(w.max(), 1e-30) * 1e-10).sum())
    log.info(f"  numerical rank ~{rank}/{X.shape[1]}; "
             f"top-10 var explained {100 * w[:10].sum() / w.sum():.1f}%")
    return {"eigenvalues": w.astype(np.float64), "basis": V.astype(np.float32),
            "mean": Xc.mean(axis=0).astype(np.float32)}


def build_identity_basis(d: int) -> Dict[str, np.ndarray]:
    return {"eigenvalues": np.arange(d, dtype=np.float64), "basis": None}  # None => no-op


def project(H: np.ndarray, basis: Optional[np.ndarray]) -> np.ndarray:
    """h_hat = Phi^T h, row-wise: (n, d) @ (d, d) -> (n, d). basis=None => identity."""
    if basis is None:
        return np.asarray(H, dtype=np.float32)
    return (np.asarray(H, dtype=np.float32) @ basis).astype(np.float32)


def build_all_bases(cfg, states: Dict[str, np.ndarray], layer: int) -> Dict[str, dict]:
    """Build every configured basis at one layer from the pooled 1634 states."""
    X = np.concatenate([states[f"L{layer}_neutral"], states[f"L{layer}_pressured"]], axis=0)
    log.info(f"Layer {layer}: pooled matrix X {X.shape} (neutral + pressured, label-blind)")
    out: Dict[str, dict] = {}
    for name in cfg.bases:
        if name == "identity":
            out[name] = build_identity_basis(X.shape[1])
        elif name == "pca":
            out[name] = build_pca_basis(X, cfg.eig_backend)
        elif name == "laplacian":
            out[name] = build_laplacian_basis(X, cfg.eig_backend)
        else:
            raise ValueError(f"unknown basis: {name}")
    return out


def condition_states(states: Dict[str, np.ndarray], layer: int, condition: str) -> np.ndarray:
    """§3.5.3 -- the three conditions analyzed per layer."""
    p = states[f"L{layer}_pressured"]
    n = states[f"L{layer}_neutral"]
    if condition == "pressured":
        return p
    if condition == "neutral":
        return n
    if condition == "shift":
        return p - n
    raise ValueError(f"unknown condition: {condition}")
