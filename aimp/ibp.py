"""Baseline ``r_0``: pure interval bound propagation through the head, and the interval pattern bounds.

Interval scores (from a method of :func:`aimp.bounds.score_intervals`) -> attention intervals ``[a_lo, a_hi]``
(Lemma 2 of the paper) ->
coordinatewise interval mixing ``y_k in sum_j [a_lo_j, a_hi_j] * [v_lo_jk, v_hi_jk]`` with
``v_j = N z_j`` (exact coordinate bounds over ``Z_j``) -> ``c^T y`` by interval arithmetic.
The simplex constraint ``sum a = 1`` and the coupling between coordinates are not used — this is
what IBP does.
"""
from __future__ import annotations

import time

import numpy as np

from aimp import sets as S
from aimp.bounds import controls as _controls
from aimp.types import Instance, SolveResult


def solve_ibp(inst: Instance, ctrl: dict | None = None, method: str = "auto") -> SolveResult:
    t0 = time.perf_counter()
    ctrl = ctrl or _controls(inst, method)
    a_lo, a_hi = ctrl["a_lo"], ctrl["a_hi"]
    N = inst.head.N
    y_lo = np.zeros(inst.d)
    y_hi = np.zeros(inst.d)
    for idx, j in enumerate(inst.keys):
        v_lo, v_hi = S.coord_bounds(inst.sets[j], N)
        prods = np.stack([a_lo[idx] * v_lo, a_lo[idx] * v_hi, a_hi[idx] * v_lo, a_hi[idx] * v_hi])
        y_lo += prods.min(axis=0)
        y_hi += prods.max(axis=0)
    c = inst.c
    value = float(np.sum(np.maximum(c * y_lo, c * y_hi)))
    return SolveResult(
        method="ibp", status="optimal", value=value,
        diagnostics={"y_lo": y_lo, "y_hi": y_hi, "score_method": ctrl["score_intervals"]["method"]},
        params={"score_method": ctrl["score_intervals"]["method"]},
        wall_time_s=time.perf_counter() - t0,
    )


def solve_ibp_obj(inst: Instance, ctrl: dict | None = None, method: str = "auto") -> SolveResult:
    """``ibp_obj``, the interval pattern bounds of the paper's experiments: attention intervals ``[a_lo, a_hi]``
    (Lemma 2), then the exact support of the
    objective per key, ``g_j = max_{z in Z_j} c^T N z``, combined by interval arithmetic (no simplex):
    ``r = sum_j max(a_lo_j g_j, a_hi_j g_j)``. The simplex-constrained value (an LP over
    ``a in [a_lo, a_hi]``, ``sum a = 1``) is recorded in ``diagnostics["value_simplex"]``."""
    t0 = time.perf_counter()
    ctrl = ctrl or _controls(inst, method)
    a_lo, a_hi = np.maximum(ctrl["a_lo"], 0.0), np.minimum(ctrl["a_hi"], 1.0)
    w = inst.head.N.T @ inst.c
    g = np.array([S.support(inst.sets[j], w) for j in inst.keys])
    value = float(np.sum(np.maximum(a_lo * g, a_hi * g)))
    # simplex version: greedy fractional knapsack (exact LP solution)
    a = a_lo.copy()
    room = 1.0 - a.sum()
    for k in np.argsort(-g):
        add = min(a_hi[k] - a[k], max(room, 0.0))
        a[k] += add
        room -= add
    v_simplex = float(a @ g) if abs(room) <= 1e-12 else float("nan")
    return SolveResult(method="ibp_obj", status="optimal", value=value,
                       diagnostics={"g": g, "value_simplex": v_simplex, "score_method": ctrl["score_intervals"]["method"]},
                       params={"score_method": ctrl["score_intervals"]["method"]},
                       wall_time_s=time.perf_counter() - t0)
