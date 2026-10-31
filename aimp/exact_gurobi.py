"""Exact reference: the head certification problem as a nonconvex program for Gurobi 13 (NonConvex=2, exact exp via
FuncNonlinear=1), solved by spatial branch-and-bound (the exact references of the synthetic experiments, Appendix B).

Token variables are written in base coordinates ``z_t = c_t + B_t u_t`` (sets.AffineRep), with
``u_t`` constrained by the base set (box bounds / ``u^T u <= rho^2`` / simplex). Auxiliary
linear variables ``k_j = M z_j`` and ``v_j = N z_j`` keep the bilinear terms at ``O(n d)``.

* ``form="log"`` (default): ``t_j = s_j - tau``, ``a_j = exp(t_j)``, ``sum a = 1``.
* ``form="algebraic"``: ``e_j = exp(s_j - s_ref)``, ``S = sum e``, ``a_j S = e_j``; the shift
  ``s_ref = max_j s_hi_j`` is an exact reformulation (softmax is shift invariant) that avoids
  overflow of ``exp(s)``.
* ``exp_impl="genexp"`` uses ``addGenConstrExp`` + ``FuncNonlinear=1``; ``"nl"`` uses
  ``addGenConstrNL(a, nlfunc.exp(t))``.
"""
from __future__ import annotations

import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from aimp import sets as S
from aimp.bounds import _gurobi_base_vars, controls as _controls
from aimp.heads import forward
from aimp.solvers import gurobi_model, gurobi_status
from aimp.types import Instance, SolveResult


def _interval_mul(alo, ahi, blo, bhi):
    p = np.stack([alo * blo, alo * bhi, ahi * blo, ahi * bhi])
    return p.min(axis=0), p.max(axis=0)


def solve_exact(inst: Instance, ctrl: dict | None = None, form: str = "log", exp_impl: str = "genexp",
                time_limit: float = 600.0, warm_start: np.ndarray | None = None,
                threads: int | None = None, strengthen: bool = False, stop_if_ub_below: float | None = None,
                stop_min_runtime: float = 20.0, **gurobi_params) -> SolveResult:
    """``stop_if_ub_below``: terminate the B&B as soon as the best bound is ``<=`` this value and at least
    ``stop_min_runtime`` seconds have passed (status ``"interrupted"``; ``value`` = the valid ObjBound). Useful in
    sweeps to stop once the reference is accurate enough (the instance is resolved); never set by default."""
    t0 = time.perf_counter()
    ctrl = ctrl or _controls(inst)
    params = {"FuncNonlinear": 1}
    params.update(gurobi_params)
    m, used = gurobi_model(f"exact_{form}", threads=threads, time_limit=time_limit, **params)
    M, N, c = inst.head.M, inst.head.N, inst.c
    n, d = inst.n, inst.d
    dv = N.shape[0]  # value / output dimension (= d unless the head is in reduced coordinates, Proposition 15)
    s_lo, s_hi = ctrl["s_lo"], ctrl["s_hi"]
    a_lo, a_hi = ctrl["a_lo"], ctrl["a_hi"]
    tau_lo, tau_hi = ctrl["tau_lo"], ctrl["tau_hi"]

    reps = {t: S.affine(inst.sets[t]) for t in inst.tokens}
    U, Zv = {}, {}
    for t in inst.tokens:
        rep = reps[t]
        zlo, zhi = S.coord_bounds(inst.sets[t])
        Zv[t] = m.addMVar(d, lb=zlo, ub=zhi, name=f"z{t}")
        if rep.r:
            U[t] = _gurobi_base_vars(m, rep, f"u{t}")
            m.addConstr(Zv[t] == rep.c + rep.B @ U[t])
    q = Zv[inst.query]
    s = m.addMVar(n, lb=s_lo, ub=s_hi, name="s")
    K, V = [], []
    for idx, j in enumerate(inst.keys):
        klo, khi = S.coord_bounds(inst.sets[j], M)
        vlo, vhi = S.coord_bounds(inst.sets[j], N)
        k = m.addMVar(d, lb=klo, ub=khi, name=f"k{j}")
        v = m.addMVar(dv, lb=vlo, ub=vhi, name=f"v{j}")
        m.addConstr(k == M @ Zv[j])
        m.addConstr(v == N @ Zv[j])
        m.addConstr(s[idx] == q @ k)
        K.append((klo, khi))
        V.append((vlo, vhi, v))

    a_lb = np.maximum(a_lo, 0.0)
    a_ub = np.minimum(a_hi, 1.0)
    a = m.addMVar(n, lb=a_lb, ub=a_ub, name="a")
    extra = {}
    if form == "log":
        tau = m.addVar(lb=tau_lo, ub=tau_hi, name="tau")
        t_lo, t_hi = s_lo - tau_hi, s_hi - tau_lo
        tv = m.addMVar(n, lb=t_lo, ub=t_hi, name="t")
        for idx in range(n):
            m.addConstr(tv[idx] == s[idx] - tau)
            _add_exp(m, tv[idx], a[idx], exp_impl)
        m.addConstr(a.sum() == 1)
        extra.update(tau=tau, t=tv)
    elif form == "algebraic":
        s_ref = float(np.max(s_hi))
        sh = m.addMVar(n, lb=s_lo - s_ref, ub=s_hi - s_ref, name="sh")
        e = m.addMVar(n, lb=np.exp(s_lo - s_ref), ub=np.exp(s_hi - s_ref), name="e")
        Ssum = m.addVar(lb=float(np.exp(s_lo - s_ref).sum()), ub=float(np.exp(s_hi - s_ref).sum()), name="S")
        for idx in range(n):
            m.addConstr(sh[idx] == s[idx] - s_ref)
            _add_exp(m, sh[idx], e[idx], exp_impl)
            m.addConstr(a[idx] * Ssum == e[idx])
        m.addConstr(Ssum == e.sum())
        m.addConstr(a.sum() == 1)  # implied by a_j S = e_j and S = sum e; kept as a valid redundant constraint
        extra.update(sh=sh, e=e, S=Ssum, s_ref=s_ref)
    else:
        raise ValueError(form)

    ylo, yhi = np.zeros(dv), np.zeros(dv)
    for idx in range(n):
        plo, phi_ = _interval_mul(a_lb[idx], a_ub[idx], V[idx][0], V[idx][1])
        ylo += plo
        yhi += phi_
    y = m.addMVar(dv, lb=ylo, ub=yhi, name="y")
    for k in range(dv):
        m.addConstr(y[k] == gp.quicksum(a[idx] * V[idx][2][k] for idx in range(n)))
    if strengthen:
        _add_redundant(m, inst, ctrl, reps, U, a, y)
    m.setObjective(c @ y, GRB.MAXIMIZE)

    if warm_start is not None:
        m.update()
        _set_start(m, inst, reps, warm_start, Zv, U, s, a, y, V, extra, form)

    try:
        if stop_if_ub_below is not None:
            thr = float(stop_if_ub_below)

            def _cb(model, where):
                if (where == GRB.Callback.MIP and model.cbGet(GRB.Callback.MIP_OBJBND) <= thr
                        and model.cbGet(GRB.Callback.RUNTIME) >= stop_min_runtime):
                    model.terminate()

            m.optimize(_cb)
        else:
            m.optimize()
        status = gurobi_status(m)
    except gp.GurobiError as err:
        return SolveResult(method=f"exact_{form}", status="error", value=float("nan"),
                           diagnostics={"message": str(err)}, params=used,
                           wall_time_s=time.perf_counter() - t0)
    diag = {"mip_gap": _attr(m, "MIPGap"), "node_count": _attr(m, "NodeCount"),
            "sol_count": m.SolCount, "runtime": m.Runtime, "form": form, "exp_impl": exp_impl,
            "stop_if_ub_below": stop_if_ub_below, "stop_min_runtime": stop_min_runtime}
    value_lb = value_lb_ver = None
    Zinc = None
    if m.SolCount > 0:
        value_lb = float(m.ObjVal)
        Zinc = inst.centers().copy()
        for t in inst.tokens:
            # recompute z from u (exact affine map) and project for robustness to tolerances
            zt = reps[t].c + reps[t].B @ U[t].X if reps[t].r else reps[t].c.copy()
            Zinc[t] = S.project(inst.sets[t], zt)
        value_lb_ver = forward(inst, Zinc)["phi"]
    value = (float(m.ObjBound) if status in ("optimal", "time_limit", "suboptimal", "interrupted", "mem_limit")
             else float("nan"))
    approx = [] if status == "optimal" else [status]
    mname = f"exact_{form}" + ("" if exp_impl == "genexp" else "_nl") + ("_strong" if strengthen else "")
    res = SolveResult(method=mname, status=status,
                      value=value, value_lb=value_lb, value_lb_verified=value_lb_ver,
                      recovered_point=Zinc, recovered_value=value_lb_ver, diagnostics=diag,
                      approximations=approx, params={**used, "form": form, "exp_impl": exp_impl, "strengthen": strengthen},
                      wall_time_s=time.perf_counter() - t0)
    m.dispose()
    return res


def _add_redundant(m, inst, ctrl, reps, U, a, y):
    """Valid redundant constraints (do not change the feasible set): ``g_j = a_j u_j`` (bilinear)
    with the perspective constraints on ``g_j`` and on ``u_j - g_j`` (the products of ``a_j`` and ``1 - a_j`` with the
    set, as in (R5) of Definition 3), ``y = sum_j N (a_j c_j + B_j g_j)``, and the certified l1 cut of the pattern
    polytope (32)."""
    from aimp.bounds import choose_cut

    N = inst.head.N
    y_expr = 0
    for idx, j in enumerate(inst.keys):
        rep = reps[j]
        y_expr = y_expr + (N @ rep.c) * a[idx]
        if not rep.r or rep.base == "simplex":
            if rep.r:
                return  # hulls: skip (not supported here)
            continue
        lim = rep.h if rep.base == "box" else np.full(rep.r, rep.rho)
        g = m.addMVar(rep.r, lb=-lim, ub=lim, name=f"g{j}")
        for k in range(rep.r):
            m.addConstr(g[k] == a[idx] * U[j][k])
        if rep.base == "box":
            m.addConstr(g <= rep.h * a[idx])
            m.addConstr(g >= -rep.h * a[idx])
            m.addConstr(U[j] - g <= rep.h * (1 - a[idx]))
            m.addConstr(U[j] - g >= -rep.h * (1 - a[idx]))
        else:
            r1 = m.addVar(lb=0, name=f"r1_{j}")
            r2 = m.addVar(lb=0, name=f"r2_{j}")
            h = m.addMVar(rep.r, lb=-2 * rep.rho, ub=2 * rep.rho, name=f"h{j}")
            m.addConstr(r1 == rep.rho * a[idx])
            m.addConstr(r2 == rep.rho * (1 - a[idx]))
            m.addConstr(h == U[j] - g)
            m.addConstr(g @ g <= r1 * r1)
            m.addConstr(h @ h <= r2 * r2)
        y_expr = y_expr + (N @ rep.B) @ g
    m.addConstr(y == y_expr)
    cutinfo = choose_cut(ctrl, inst.n, "auto")
    p = m.addMVar(inst.n, lb=0, name="cut_p")
    q = m.addMVar(inst.n, lb=0, name="cut_q")
    m.addConstr(a - cutinfo["ahat"] == p - q)
    m.addConstr(p.sum() + q.sum() <= cutinfo["eps"])


def _attr(m, name):
    try:
        return float(getattr(m, name))
    except (gp.GurobiError, AttributeError):
        return None


def _add_exp(m, x, y, exp_impl):
    if exp_impl == "genexp":
        m.addGenConstrExp(x, y)  # exact with FuncNonlinear=1
    elif exp_impl == "nl":
        m.addGenConstrNL(y, gp.nlfunc.exp(x))
    else:
        raise ValueError(exp_impl)


def _set_start(m, inst, reps, Z0, Zv, U, s, a, y, V, extra, form):
    out = forward(inst, Z0)
    for t in inst.tokens:
        Zv[t].Start = Z0[t]
        if reps[t].r and reps[t].base != "simplex":
            u, *_ = np.linalg.lstsq(reps[t].B, Z0[t] - reps[t].c, rcond=None)
            u = S.base_project(reps[t], u)
            U[t].Start = u
            if reps[t].base == "ball":
                for k in range(reps[t].r):
                    m.getVarByName(f"w_u{t}[{k}]").Start = u[k] / reps[t].rho
        elif reps[t].r and reps[t].base == "simplex":  # hull: the weights of the start point
            U[t].Start = S._project_hull_weights(inst.sets[t].points, Z0[t])
    s.Start = out["s"]
    a.Start = out["a"]
    y.Start = out["y"]
    for idx, j in enumerate(inst.keys):
        V[idx][2].Start = inst.head.N @ Z0[j]
    if form == "log":
        tau = float(np.log(np.exp(out["s"] - out["s"].max()).sum()) + out["s"].max())
        extra["tau"].Start = tau
        extra["t"].Start = out["s"] - tau
    else:
        e = np.exp(out["s"] - extra["s_ref"])
        extra["sh"].Start = out["s"] - extra["s_ref"]
        extra["e"].Start = e
        extra["S"].Start = float(e.sum())
