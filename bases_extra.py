"""Two more UNSIGNED graph constructions on the dimension-correlation graph.

  posrelu : W = max(C, 0)   -- keep only positive (co-activation) edges, drop anti-correlation
  abscorr : W = |C|         -- strength of the linear relationship, sign-blind

Both are non-negative, so they use the standard symmetric normalized Laplacian
L = I - D^-1/2 W D^-1/2 (same form as the (C+1)/2 path), just with a different W.
Reuses src.bases.dimension_correlation + _eigh so correlation + eigensolver match exactly.
"""
from __future__ import annotations
from typing import Dict
import numpy as np

from src.bases import dimension_correlation, _eigh
from src.utils import log, timed


def _unsigned_norm_laplacian(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64).copy()
    np.fill_diagonal(W, 0.0)
    deg = W.sum(axis=1)
    deg = np.maximum(deg, 1e-12)
    dinv = 1.0 / np.sqrt(deg)
    L = -(W * dinv[:, None]) * dinv[None, :]
    np.fill_diagonal(L, 1.0 + np.diag(L))
    return (L + L.T) / 2.0


def _build(X, backend, Wfn, name):
    with timed(f"{name} basis (d={X.shape[1]}, backend={backend})"):
        C = dimension_correlation(X)
        W = Wfn(C)
        L = _unsigned_norm_laplacian(W)
        w, Phi = _eigh(L, backend)
    log.info(f"  {name} eigenvalue range [{w.min():.3f}, {w.max():.3f}]")
    return {"eigenvalues": w.astype(np.float64), "basis": Phi.astype(np.float32)}


def build_posrelu_basis(X, backend="scipy"):
    return _build(X, backend, lambda C: np.maximum(C, 0.0), "posrelu")


def build_abscorr_basis(X, backend="scipy"):
    return _build(X, backend, lambda C: np.abs(C), "abscorr")
