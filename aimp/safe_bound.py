"""The safe dual bound (Proposition 16 of the paper): a rigorous upper bound from an approximate dual solution, in the
spirit of the error bounds of Jansson et al. (VSDP), for problems modelled in CVXPY.

For ``max f(x)`` over ``x`` with constraints ``g_k(x) in K_k`` and any multipliers ``lambda_k in K_k^*`` (dual cones),
weak duality gives, for every feasible ``x``, ``f(x) <= l(x) := f(x) + sum_k <lambda_k, g_k(x)>``. When every ``g_k`` is
affine, ``l`` is affine: ``l(x) = l(0) + <r, x>`` with the *dual residual* ``r``. Hence

    f*  <=  l(0) + sup_{x in B} <r, x>

for any set ``B`` containing the feasible set. This is the familiar form ``b^T y + sum_i |r_i| U_i``: ``l(0) = b^T y``
is the dual objective at the (cone-projected) dual point, ``|x_i| <= U_i`` the a-priori bounds. The PSD moment matrix
``X`` is not bounded entrywise but through its own cone (Jansson, VSDP): over ``{X >= 0, X_kk <= D_k}``,
``<R, X> <= lambda_max^+(S R S) * sum_k X_kk / D_k`` with ``S = diag(sqrt(D_k))`` (and ``<= sum |R+_kl| sqrt(D_k D_l)``
with ``R+`` the PSD part of ``R``; the smaller one is used). Grouped diagonal budgets (a ball block ``tr U <= rho^2``)
replace ``sum_k X_kk / D_k <= #coords`` by ``1`` per group.

Multipliers are CVXPY's dual values, mapped to the dual cones: ``Inequality`` (``expr <= 0``): ``lambda = max(dual, 0)``,
term ``-lambda expr``; ``Equality``: free, term ``-nu expr``; ``SOC(t, x)``: ``(lambda_t, lambda_x)`` with ``lambda_t``
raised to ``||lambda_x||`` if needed, term ``+<lambda, (t, x)>``; ``ExpCone(x, y, z)``: ``(u, v, w)`` with ``w`` raised to
``-u exp(v/u - 1)`` (``u < 0``) or ``(0, max(v,0), max(w,0))`` (``u >= 0``), term ``+<lambda, (x, y, z)>``. A non-affine
``Inequality`` (convex ``expr``; CVXPY's DCP rules) is replaced by its tangent at the returned point
(``expr(x) >= expr(xh) + <grad, x - xh>``), which keeps ``l`` an upper bound of ``f`` on the feasible set.

Every number is evaluated in float64 (no interval arithmetic); a rounding allowance ``64 u (|l|(0) + sum |r|_abs U)``
computed from the absolute values of all terms is added (``u`` = unit roundoff). The bound is rigorous up to that.
"""
from __future__ import annotations

import math
import time

import numpy as np
import scipy.sparse as sp

U_ROUND = np.finfo(float).eps / 2


def _flat(x) -> np.ndarray:
    return np.asarray(x, float).flatten(order="F")


def _grad_mats(expr, variables) -> dict:
    """``{var.id: sparse (var.size x expr.size)}``: CVXPY's gradient of ``expr`` at the current variable values (exact
    coefficients for affine ``expr``). Missing variables -> no entry."""
    g = expr.grad
    out = {}
    for v in variables:
        J = g.get(v) if g is not None else None
        if J is None:
            continue
        if not sp.issparse(J):
            J = sp.csc_matrix(np.asarray(J, float).reshape(v.size, -1, order="F"))
        out[v.id] = J
    return out


def _expr_zero_value(expr, variables) -> np.ndarray:
    saved = [v.value for v in variables]
    try:  # save_value: no attribute validation (the solver's X is PSD only to solver precision)
        for v in variables:
            v.save_value(np.zeros(v.shape))
        return _flat(expr.value)
    finally:
        for v, val in zip(variables, saved):
            v.save_value(val)


class _Acc:
    """Accumulates ``l(x) = c0 + <g, x>`` and its absolute-value companion (for the rounding allowance)."""

    def __init__(self, variables):
        self.vars = variables
        self.c0 = 0.0
        self.c0_abs = 0.0
        self.g = {v.id: np.zeros(v.size) for v in variables}
        self.g_abs = {v.id: np.zeros(v.size) for v in variables}

    def add(self, expr, lam: np.ndarray, sign: float, linearize: bool = False):
        """Add ``sign * <lam, expr(x)>`` (affine ``expr``), or of its tangent at the current point (``linearize``)."""
        lam = _flat(lam)
        if not np.any(lam):
            return
        J = _grad_mats(expr, self.vars)
        if linearize:  # tangent: e(xh) + <J, x - xh>
            e0 = _flat(expr.value)
            for v in self.vars:
                if v.id in J:
                    e0 = e0 - J[v.id].T @ _flat(v.value)
        else:
            e0 = _expr_zero_value(expr, self.vars)
        self.c0 += sign * float(lam @ e0)
        self.c0_abs += float(np.abs(lam) @ np.abs(e0))
        for vid, Jv in J.items():
            self.g[vid] += sign * (Jv @ lam)
            self.g_abs[vid] += abs(Jv) @ np.abs(lam)


def _soc_fix(lam_t: np.ndarray, lam_x: np.ndarray, axis: int) -> tuple[np.ndarray, int]:
    """Raise ``lambda_t`` to ``||lambda_x||`` (columnwise per CVXPY's ``axis``); returns (lambda_t, #fixed)."""
    lam_t = np.atleast_1d(np.asarray(lam_t, float)).copy()
    X = np.asarray(lam_x, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1) if lam_t.size == 1 else X.reshape(1, -1)
    nrm = np.linalg.norm(X, axis=0) if axis == 0 else np.linalg.norm(X, axis=1)
    nrm = np.atleast_1d(nrm)
    bad = lam_t < nrm
    lam_t[bad] = nrm[bad]
    return lam_t, int(bad.sum())


def _exp_fix(u, v, w) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Map ``(u, v, w)`` into ``K_exp^* = {u < 0, -u e^{v/u} <= e w} U {u = 0, v >= 0, w >= 0}`` by raising ``w``
    (``u < 0``) or clipping (``u >= 0``)."""
    u, v, w = (np.atleast_1d(np.asarray(x, float)).copy() for x in (u, v, w))
    nfix = 0
    for k in range(u.size):
        if u[k] < 0:
            lw = math.log(-u[k]) + v[k] / u[k] - 1.0  # log of the smallest admissible w
            need = math.exp(lw) if lw < 700 else math.inf
            if w[k] < need:
                w[k] = need * (1 + 4 * U_ROUND)
                nfix += 1
        else:
            if u[k] != 0 or v[k] < 0 or w[k] < 0:
                nfix += 1
            u[k], v[k], w[k] = 0.0, max(v[k], 0.0), max(w[k], 0.0)
    return u, v, w, nfix


def safe_dual_bound(prob, X, D: np.ndarray, groups: list[tuple[list[int], float]] | None = None,
                    var_bounds: dict | None = None) -> dict:
    """Rigorous upper bound on ``max`` of the solved CVXPY problem ``prob`` (the safe dual bound, Proposition 16).

    ``X``: the PSD moment-matrix variable (``PSD=True``), ``D[k] >= X_kk`` on the feasible set (``D[0] = 1``),
    ``groups``: optional ``[(indices, budget)]`` with ``sum_{k in idx} X_kk / D_k <= budget`` (coordinates not in any
    group get budget 1 each). ``var_bounds``: ``{var.id: U}`` with ``|x| <= U`` (scalar or array), or ``{var.id: (lo, hi)}``
    (a box; needed for variables with a ``nonneg`` attribute, whose multiplier is not a CVXPY constraint) for every
    other variable. Returns ``{value_safe, l0, psd_term, box_term, rounding, ...}``; ``value_safe = None`` if some multiplier or
    bound is missing."""
    import cvxpy as cp
    from cvxpy.constraints import SOC, Equality, ExpCone, Inequality

    t0 = time.perf_counter()
    var_bounds = var_bounds or {}
    variables = prob.variables()
    out = {"value_safe": None}
    if any(v.value is None for v in variables):
        out["reason"] = "no primal values"
        return out
    for v in variables:
        if v.id != X.id and v.id not in var_bounds:
            out["reason"] = f"no bound for variable {v.name()}"
            return out
    acc = _Acc(variables)
    if not isinstance(prob.objective, cp.Maximize):
        raise ValueError("safe_dual_bound expects a Maximize problem")
    acc.add(prob.objective.expr, np.ones(1), +1.0)
    nfix = {"soc": 0, "exp": 0, "ineq": 0, "missing_dual": 0, "linearized": 0}
    for con in prob.constraints:
        dv = con.dual_value
        if dv is None:
            nfix["missing_dual"] += 1  # multiplier 0: valid
            continue
        if isinstance(con, Equality):
            if not con.expr.is_affine():
                out["reason"] = "non-affine equality"
                return out
            acc.add(con.expr, dv, -1.0)
        elif isinstance(con, Inequality):
            lam = _flat(dv)
            nfix["ineq"] += int(np.sum(lam < 0))
            lam = np.maximum(lam, 0.0)
            aff = con.expr.is_affine()
            if not aff:
                nfix["linearized"] += 1
            acc.add(con.expr, lam, -1.0, linearize=not aff)
        elif isinstance(con, SOC):
            lam_t, k = _soc_fix(dv[0], dv[1], con.axis)
            nfix["soc"] += k
            acc.add(con.args[0], lam_t, +1.0)
            acc.add(con.args[1], np.asarray(dv[1], float), +1.0)
        elif isinstance(con, ExpCone):
            u, v_, w, k = _exp_fix(*dv)
            nfix["exp"] += k
            for arg, lam in zip(con.args, (u, v_, w)):
                acc.add(arg, lam, +1.0)
        else:
            out["reason"] = f"unsupported constraint {type(con).__name__}"
            return out
    # --- sup of <r, x> over the a-priori set
    m = X.shape[0]
    D = np.maximum(np.asarray(D, float), 0.0)
    R = acc.g[X.id].reshape((m, m), order="F")
    R = (R + R.T) / 2
    Rabs = acc.g_abs[X.id].reshape((m, m), order="F")
    Rabs = (Rabs + Rabs.T) / 2
    sD = np.sqrt(D)
    budget = 0.0
    in_group = np.zeros(m, bool)
    for idx, b in groups or []:
        idx = [k for k in idx if D[k] > 0]
        if idx:
            budget += float(b)
            in_group[idx] = True
    budget += float(np.sum((D > 0) & ~in_group))
    SRS = R * np.outer(sD, sD)
    lam_max = float(np.linalg.eigvalsh(SRS)[-1])
    eig_err = 8 * m * U_ROUND * float(np.linalg.norm(SRS, "fro"))
    psd_eig = (max(lam_max, 0.0) + eig_err) * budget
    ev, V = np.linalg.eigh(R)
    Rp = (V * np.maximum(ev, 0.0)) @ V.T
    psd_ent = float(np.sum(np.abs(Rp) * np.outer(sD, sD))) + 8 * m * U_ROUND * float(np.linalg.norm(R, "fro")) * float(sD.sum()) ** 2
    psd_term = min(psd_eig, psd_ent)
    box_term = 0.0
    box_abs = float(np.sum(Rabs * np.outer(sD, sD)))
    for v in variables:
        if v.id == X.id:
            continue
        b = var_bounds[v.id]
        if isinstance(b, tuple):  # box [lo, hi]
            lo = _flat(np.broadcast_to(np.asarray(b[0], float), v.shape))
            hi = _flat(np.broadcast_to(np.asarray(b[1], float), v.shape))
        else:  # |x| <= U (the term sum |r_i| U_i)
            hi = _flat(np.broadcast_to(np.asarray(b, float), v.shape))
            lo = -hi
        r = acc.g[v.id]
        box_term += float(np.sum(np.maximum(r * lo, r * hi)))
        box_abs += float(acc.g_abs[v.id] @ np.maximum(np.abs(lo), np.abs(hi)))
    rounding = 64 * U_ROUND * (acc.c0_abs + box_abs)
    val = acc.c0 + psd_term + box_term + rounding
    resid_free = max((float(np.max(np.abs(acc.g[v.id]))) for v in variables if v.id != X.id and v.size), default=0.0)
    out.update({"value_safe": float(val), "l0": acc.c0, "psd_term": psd_term, "psd_term_eig": psd_eig,
                "psd_term_entrywise": psd_ent, "lam_max_SRS": lam_max, "psd_budget": budget, "box_term": box_term,
                "rounding": rounding, "resid_free_max": resid_free, "fixes": nfix,
                "time_s": time.perf_counter() - t0})
    return out
