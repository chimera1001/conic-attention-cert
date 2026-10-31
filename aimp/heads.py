"""Exact float64 forward pass of one head at one query position, and its analytic gradient.

``Z`` is a (T x d) array of token inputs (row t = token t). Scores ``s_j = z_i^T M z_j`` for
the keys ``j`` (in key order), pattern ``a = softmax(s)``, output ``y = sum_j a_j N z_j``,
objective ``phi = c^T y``.
"""
from __future__ import annotations

import numpy as np

from aimp.types import Instance


def softmax(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, float)
    e = np.exp(s - s.max())
    return e / e.sum()


def logsumexp(s: np.ndarray) -> float:
    s = np.asarray(s, float)
    m = s.max()
    return float(m + np.log(np.exp(s - m).sum()))


def forward(inst: Instance, Z: np.ndarray) -> dict:
    """Nominal quantities at ``Z``: ``s, a, y, phi`` (float64)."""
    M, N = inst.head.M, inst.head.N
    Zk = Z[inst.keys]  # (n, d)
    q = Z[inst.query]
    s = Zk @ (M.T @ q)  # s_j = q^T M z_j
    a = softmax(s)
    V = Zk @ N.T  # rows N z_j
    y = a @ V
    return {"s": s, "a": a, "y": y, "phi": float(inst.c @ y)}


def phi(inst: Instance, Z: np.ndarray) -> float:
    return forward(inst, Z)["phi"]


def grad_phi(inst: Instance, Z: np.ndarray) -> tuple[float, np.ndarray]:
    """``phi`` and ``d phi / d Z`` (T x d).

    ``d phi/d z_j |_direct = a_j N^T c``; ``g_j = c^T N z_j``; ``d phi/d s = a * (g - a.g)``;
    ``d s_j / d z_j = M^T z_i``; ``d s_j / d z_i = M z_j`` (both apply when ``i in J``).
    """
    M, N, c = inst.head.M, inst.head.N, inst.c
    keys, i = inst.keys, inst.query
    out = forward(inst, Z)
    a = out["a"]
    Zk = Z[keys]
    g = Zk @ (N.T @ c)
    dphi_ds = a * (g - a @ g)
    G = np.zeros_like(Z)
    Ntc = N.T @ c
    Mtq = M.T @ Z[i]
    for idx, j in enumerate(keys):
        G[j] += a[idx] * Ntc + dphi_ds[idx] * Mtq
    G[i] += (dphi_ds[:, None] * (Zk @ M.T)).sum(axis=0)  # sum_j dphi/ds_j * M z_j
    return out["phi"], G


def nominal(inst: Instance) -> dict:
    """Forward pass at the set centers."""
    return forward(inst, inst.centers())
