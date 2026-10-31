"""The pattern polytope certificate ``r_pp`` (Definition 2 of the paper; ``r_ps`` in the code), solved as a convex
relaxation (Gurobi LP/SOCP and CVXPY+MOSEK backends) or by the greedy algorithm of Proposition 8.

In base coordinates the perspective variable of key ``j`` is ``w_j = a_j c_j + B_j g_j`` with
``g_j in a_j U_j``: box ``|g| <= a_j h``; ball/ellipsoid ``||g|| <= a_j rho``; hull
(``c = 0``, ``B = points^T``) ``g >= 0, sum g = a_j``. This is exactly ``w_j in a_j Z_j``: the relaxation of the
head certification problem in which each product ``a_j z_j`` becomes a variable in the scaled set ``a_j Z_j``
(Section 4), whose value is ``r_pp``.
"""
from __future__ import annotations

import math
import time

import numpy as np

from aimp import sets as S
from aimp.bounds import DEFAULT_CUT, bound_general, choose_cut, controls as _controls
from aimp.heads import forward
from aimp.types import Instance, SolveResult

RECOVER_TOL = 1e-9


def solve_ps(inst: Instance, ctrl: dict | None = None, cut: str = "auto", backend: str = "gurobi",
             interval_bounds: bool = True, threads: int | None = None, time_limit: float = 60.0) -> SolveResult:
    """``r_ps = max c^T y`` over the pattern polytope relaxation. ``cut`` in {auto, none, B, C} (the certified pair of
    :func:`aimp.bounds.choose_cut`); ``interval_bounds=False`` drops ``a_lo <= a <= a_hi`` (used by a regression test on
    the two-key instance :func:`aimp.instances.prop9`)."""
    t0 = time.perf_counter()
    ctrl = ctrl or _controls(inst)
    cutinfo = choose_cut(ctrl, inst.n, cut)
    reps = [S.affine(inst.sets[j]) for j in inst.keys]
    if backend == "gurobi":
        status, value, a, G, diag, params = _gurobi(inst, ctrl, cutinfo, reps, interval_bounds, threads, time_limit)
        if status != "optimal":  # a non-optimal barrier value is not a valid bound: re-solve with MOSEK
            g_status, g_value = status, value
            status, value, a, G, diag, params = _cvxpy(inst, ctrl, cutinfo, reps, interval_bounds, threads)
            diag = {**diag, "gurobi_status": g_status, "gurobi_value": g_value}
            backend = f"cvxpy (fallback: gurobi {g_status})"
    elif backend == "cvxpy":
        status, value, a, G, diag, params = _cvxpy(inst, ctrl, cutinfo, reps, interval_bounds, threads)
    else:
        raise ValueError(backend)
    name = f"ps_{cutinfo['cut_used'] if cut != 'auto' else 'auto'}" + ("" if interval_bounds else "_noib")
    res = SolveResult(method=name, status=status, value=value, diagnostics=diag,
                      params={**params, "backend": backend, "cut": cut, "interval_bounds": interval_bounds},
                      wall_time_s=0.0)
    res.diagnostics.update({"cut_used": cutinfo["cut_used"], "eps": cutinfo["eps"],
                            "ahat": cutinfo["ahat"]})
    if a is not None:
        _constructive_check(inst, ctrl, cutinfo, reps, a, G, res)
    res.wall_time_s = time.perf_counter() - t0
    return res


def _constructive_check(inst, ctrl, cutinfo, reps, a, G, res):
    """Recover ``ztilde_j = w_j / a_j`` (center if ``a_j <= tol``), evaluate the true head there."""
    Z = inst.centers().copy()
    for idx, j in enumerate(inst.keys):
        rep = reps[idx]
        if a[idx] > RECOVER_TOL and rep.r:
            u = S.base_project(rep, G[idx] / a[idx])  # projection only removes solver-tolerance noise
            Z[j] = S.to_z(rep, u)
    if not inst.query_in_keys:
        Z[inst.query] = inst.sets[inst.query].center
    val = forward(inst, Z)["phi"]
    bound = bound_general(cutinfo["eps_for_bound"], ctrl["rad"], ctrl["c_norm"])
    gap = res.value - val
    res.recovered_point = Z
    res.recovered_value = float(val)
    res.diagnostics.update({
        "a": a, "ps_recovered_value": float(val), "ps_gap_constructive": float(gap),
        "constructive_bound": float(bound),
        "constructive_ok": bool(gap <= bound + 1e-6),
    })
    if not gap <= bound + 1e-6:
        raise AssertionError(f"pattern polytope constructive gap {gap} > 2 eps rad ||c|| = {bound} "
                             f"({inst.meta.get('instance_id')})")


def _gurobi(inst, ctrl, cutinfo, reps, interval_bounds, threads, time_limit):
    import gurobipy as gp
    from gurobipy import GRB

    from aimp.solvers import gurobi_model, gurobi_status

    m, used = gurobi_model("ps", threads=threads, time_limit=time_limit,
                           BarQCPConvTol=1e-8, BarConvTol=1e-9)
    n, d, N = inst.n, inst.d, inst.head.N
    lb = np.maximum(ctrl["a_lo"], 0) if interval_bounds else np.zeros(n)
    ub = np.minimum(ctrl["a_hi"], 1) if interval_bounds else np.ones(n)
    a = m.addMVar(n, lb=lb, ub=ub, name="a")
    m.addConstr(a.sum() == 1)
    if cutinfo["eps"] is not None:
        p = m.addMVar(n, lb=0, name="p")
        q = m.addMVar(n, lb=0, name="q")
        m.addConstr(a - cutinfo["ahat"] == p - q)
        m.addConstr(p.sum() + q.sum() <= cutinfo["eps"])
    G = []
    y_expr = 0
    for idx, j in enumerate(inst.keys):
        rep = reps[idx]
        y_expr = y_expr + (N @ rep.c) * a[idx] if rep.base != "simplex" else y_expr
        if rep.r == 0:
            G.append(None)
            continue
        g = m.addMVar(rep.r, lb=-GRB.INFINITY, name=f"g{j}")
        if rep.base == "box":
            m.addConstr(g <= rep.h * a[idx])
            m.addConstr(g >= -rep.h * a[idx])
        elif rep.base == "ball":
            r = m.addVar(lb=0, name=f"r{j}")
            m.addConstr(r == rep.rho * a[idx])
            m.addConstr(g @ g <= r * r)
        elif rep.base == "simplex":
            m.addConstr(g >= 0)
            m.addConstr(g.sum() == a[idx])
        y_expr = y_expr + (N @ rep.B) @ g
        G.append(g)
    y = m.addMVar(N.shape[0], lb=-GRB.INFINITY, name="y")
    m.addConstr(y == y_expr)
    m.setObjective(inst.c @ y, GRB.MAXIMIZE)
    m.optimize()
    status = gurobi_status(m)
    diag = {"runtime": m.Runtime, "bar_iter": int(m.BarIterCount) if m.BarIterCount else 0}
    if status not in ("optimal", "suboptimal") or m.SolCount == 0:
        m.dispose()
        return status, float("nan"), None, None, diag, used
    value = float(m.ObjVal)
    av = a.X.copy()
    Gv = [None if g is None else g.X.copy() for g in G]
    m.dispose()
    return status, value, av, Gv, diag, used


def _cvxpy(inst, ctrl, cutinfo, reps, interval_bounds, threads):
    import cvxpy as cp

    from aimp.solvers import cvx_solve

    n, N = inst.n, inst.head.N
    a = cp.Variable(n)
    cons = [a >= 0, cp.sum(a) == 1]
    if interval_bounds:
        cons += [a >= ctrl["a_lo"], a <= ctrl["a_hi"]]
    if cutinfo["eps"] is not None:
        p = cp.Variable(n, nonneg=True)
        q = cp.Variable(n, nonneg=True)
        cons += [a - cutinfo["ahat"] == p - q, cp.sum(p + q) <= cutinfo["eps"]]
    G = []
    y = 0
    for idx, j in enumerate(inst.keys):
        rep = reps[idx]
        if rep.base != "simplex":
            y = y + a[idx] * (N @ rep.c)
        if rep.r == 0:
            G.append(None)
            continue
        g = cp.Variable(rep.r)
        if rep.base == "box":
            cons += [g <= a[idx] * rep.h, g >= -a[idx] * rep.h]
        elif rep.base == "ball":
            cons += [cp.norm(g) <= rep.rho * a[idx]]
        elif rep.base == "simplex":
            cons += [g >= 0, cp.sum(g) == a[idx]]
        y = y + (N @ rep.B) @ g
        G.append(g)
    prob = cp.Problem(cp.Maximize(inst.c @ y), cons)
    status, info = cvx_solve(prob, threads)
    if status not in ("optimal", "suboptimal"):
        return status, float("nan"), None, None, info, info["mosek_params"]
    Gv = [None if g is None else np.asarray(g.value).copy() for g in G]
    return status, float(prob.value), np.asarray(a.value).copy(), Gv, info, info["mosek_params"]


# ----------------------------------------------------------------------------- greedy fast path (Proposition 8)
def ps_greedy(h: np.ndarray, lo: np.ndarray, hi: np.ndarray, ahat: np.ndarray | None = None,
              eps: float | None = None, tol: float = 1e-12) -> tuple[float, np.ndarray]:
    """``max {h^T a : a in simplex, lo <= a <= hi, ||a - ahat||_1 <= eps}`` exactly (the two-phase transfer of
    Proposition 8), in
    ``O(n log n)``. ``eps = None``: no l1 cut.

    Reduction: with ``b = clip(ahat, lo, hi)``, every box-feasible ``a`` has ``|a_j - ahat_j| = |a_j - b_j| +
    |b_j - ahat_j|``, so the cut is ``||a - b||_1 <= eps' = eps - ||b - ahat||_1``. From ``b``, first restore
    ``sum a = 1`` (add the missing mass to the largest ``h`` / remove the excess from the smallest ``h``; each unit costs
    one unit of budget), then transfer mass from the smallest to the largest ``h`` (two units of budget per unit
    moved) while ``h_top > h_bottom``, capacities and budget allow (fractional knapsack)."""
    h = np.asarray(h, float)
    n = h.shape[0]
    lo = np.clip(np.asarray(lo, float), 0.0, 1.0)
    hi = np.clip(np.asarray(hi, float), 0.0, 1.0)
    if ahat is None or eps is None:
        b, budget = lo.copy(), math.inf
    else:
        ahat = np.asarray(ahat, float)
        b = np.clip(ahat, lo, hi)
        budget = float(eps) - float(np.abs(b - ahat).sum())
    if budget < -tol:
        raise ValueError(f"pattern polytope empty: box-clipping ahat already costs {eps - budget} > eps = {eps}")
    a = b.copy()
    up = np.argsort(-h, kind="stable")
    down = np.argsort(h, kind="stable")
    delta = 1.0 - float(a.sum())
    if delta > 0:
        for j in up:
            m = min(hi[j] - a[j], delta)
            if m > 0:
                a[j] += m
                delta -= m
                budget -= m
            if delta <= tol:
                break
    elif delta < 0:
        for k in down:
            m = min(a[k] - lo[k], -delta)
            if m > 0:
                a[k] -= m
                delta += m
                budget -= m
            if delta >= -tol:
                break
    if abs(delta) > 1e-9 or budget < -1e-9:
        raise ValueError(f"pattern polytope empty (residual mass {delta}, budget {budget})")
    t = max(budget, 0.0) / 2  # mass that may still be transferred
    iu = idn = 0
    while t > 0 and iu < n and idn < n:
        j, k = up[iu], down[idn]
        if h[j] <= h[k]:
            break
        cu, cd = hi[j] - a[j], a[k] - lo[k]
        if cu <= 0:
            iu += 1
            continue
        if cd <= 0:
            idn += 1
            continue
        m = min(cu, cd, t)
        a[j] += m
        a[k] -= m
        t -= m
    return float(h @ a), a


def ps_lp(h, lo, hi, ahat=None, eps=None, threads: int | None = None) -> float:
    """The same LP solved by Gurobi's dual simplex (reference for the greedy)."""
    from gurobipy import GRB

    from aimp.solvers import gurobi_model

    n = len(h)
    m, _ = gurobi_model("ps_lp", threads=threads, Method=1)
    a = m.addMVar(n, lb=np.clip(lo, 0, 1), ub=np.clip(hi, 0, 1))
    m.addConstr(a.sum() == 1)
    if ahat is not None and eps is not None:
        p = m.addMVar(n, lb=0)
        q = m.addMVar(n, lb=0)
        m.addConstr(a - np.asarray(ahat, float) == p - q)
        m.addConstr(p.sum() + q.sum() <= float(eps))
    m.setObjective(np.asarray(h, float) @ a, GRB.MAXIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"ps_lp status {m.Status}")
    v = float(m.ObjVal)
    m.dispose()
    return v


def solve_ps_greedy(inst: Instance, ctrl: dict | None = None, cut: str = DEFAULT_CUT) -> SolveResult:
    """``r_ps`` (= ``r_pp``) by the greedy transfer over the pattern polytope with the support values
    ``h_j = max_{Z_j} c^T N z`` (Proposition 8).
    Constructive check as for :func:`solve_ps`: the maximizers of ``h_j`` give a feasible point."""
    from aimp.theory import support_values

    t0 = time.perf_counter()
    c0 = time.process_time()
    ctrl = ctrl or _controls(inst)
    cutinfo = choose_cut(ctrl, inst.n, cut)
    t1 = time.perf_counter()
    h = support_values(inst)
    t2 = time.perf_counter()
    value, a = ps_greedy(h, ctrl["a_lo"], ctrl["a_hi"], cutinfo["ahat"], cutinfo["eps"])
    t3 = time.perf_counter()
    g = inst.head.N.T @ inst.c
    Z = inst.centers().copy()
    for j in inst.keys:
        Z[j] = S.support_argmax(inst.sets[j], g)
    if not inst.query_in_keys:
        Z[inst.query] = inst.sets[inst.query].center
    val = forward(inst, Z)["phi"]
    res = SolveResult(method=f"ps_greedy_{cut}", status="optimal", value=value,
                      params={"cut": cut, "backend": "greedy"},
                      diagnostics={"cut_used": cutinfo["cut_used"], "eps": cutinfo["eps"], "a": a, "h": h,
                                   "time_support_s": t2 - t1, "time_greedy_s": t3 - t2},
                      recovered_point=Z, recovered_value=float(val))
    res.wall_time_s = time.perf_counter() - t0
    res.cpu_time_s = time.process_time() - c0
    return res
