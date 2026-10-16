"""Signed-Laplacian basis (Kunegis et al. 2010) for the dimension-correlation graph.

Difference from the existing (C+1)/2 Laplacian (src.bases.build_laplacian_basis):
  - adjacency keeps the SIGN of the correlation:            A = C   (diag 0)
    (instead of W = (C+1)/2, which sends anti-correlations toward 0 and
     uncorrelated pairs to 0.5, producing the near-complete, degenerate graph)
  - degree uses the ABSOLUTE weights (signed degree):       D_bar = diag(sum_j |A_ij|)
  - symmetric normalized signed Laplacian:                  L = I - D_bar^-1/2 A D_bar^-1/2

L is real-symmetric and positive-semidefinite (x^T L x = 1/2 sum |A_ij| (x_i - sign(A_ij) x_j)^2 >= 0),
so the eigenbasis is orthonormal and the whole downstream pipeline (projection, l1/l2, Gini,
measures, AUC) is unchanged. Optionally k-NN-sparsify on |C| first (keep each node's k
strongest-magnitude edges, symmetric union) to break residual density.

Reuses src.bases.dimension_correlation so the correlation computation matches exactly.
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np

from src.bases import dimension_correlation, _eigh   # same corr + eigensolver as the (C+1)/2 path
from src.utils import log, timed


def knn_sparsify(A: np.ndarray, k: int) -> np.ndarray:
    """Keep each node's k strongest-|weight| edges; symmetric union; signs preserved."""
    A = A.copy()
    np.fill_diagonal(A, 0.0)
    d = A.shape[0]
    k = int(min(k, d - 1))
    mag = np.abs(A)
    idx = np.argpartition(-mag, kth=k - 1, axis=1)[:, :k]      # top-k per row
    keep = np.zeros_like(A, dtype=bool)
    rows = np.arange(d)[:, None]
    keep[rows, idx] = True
    keep = keep | keep.T                                       # symmetric union
    return np.where(keep, A, 0.0)


def signed_normalized_laplacian(A: np.ndarray) -> np.ndarray:
    """A is a signed adjacency with zero diagonal. L = I - D_bar^-1/2 A D_bar^-1/2, D_bar=diag(sum|A|)."""
    A = np.asarray(A, dtype=np.float64).copy()
    np.fill_diagonal(A, 0.0)
    deg = np.abs(A).sum(axis=1)
    if (deg <= 0).any():
        log.warning(f"{int((deg <= 0).sum())} isolated dimensions; signed degree floored at 1e-12.")
        deg = np.maximum(deg, 1e-12)
    dinv = 1.0 / np.sqrt(deg)
    L = -(A * dinv[:, None]) * dinv[None, :]
    np.fill_diagonal(L, 1.0 + np.diag(L))      # add I
    L = (L + L.T) / 2.0                         # enforce exact symmetry
    return L


def build_signed_laplacian_basis(X: np.ndarray, backend: str = "scipy",
                                 knn: Optional[int] = None) -> Dict[str, np.ndarray]:
    tag = f"signed laplacian (d={X.shape[1]}, backend={backend}" + (f", knn={knn})" if knn else ", dense)")
    with timed(tag):
        C = dimension_correlation(X)
        A = C.copy(); np.fill_diagonal(A, 0.0)
        if knn:
            A = knn_sparsify(A, knn)
        L = signed_normalized_laplacian(A)
        w, Phi = _eigh(L, backend)             # ascending
    log.info(f"  signed eigenvalue range [{w.min():.3f}, {w.max():.3f}]  (bound [0, 2])")
    return {"eigenvalues": w.astype(np.float64), "basis": Phi.astype(np.float32)}
