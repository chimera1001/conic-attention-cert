"""Thin wrappers around Gurobi and CVXPY/MOSEK with the package's default parameters and thread budget."""
from __future__ import annotations

import time

import gurobipy as gp
from gurobipy import GRB

from aimp import compute

import os

#: Gurobi's hard memory limit per model (GB; ``MemLimit``, status ``mem_limit`` when hit) and the node-file threshold
#: every model gets both, so that a runaway nonconvex solve cannot exhaust a shared machine's memory.
#: ``AIMP_GUROBI_MEMLIMIT_GB`` overrides.
GUROBI_MEMLIMIT_GB = float(os.environ.get("AIMP_GUROBI_MEMLIMIT_GB", "20"))

GUROBI_DEFAULTS = {
    "NonConvex": 2,
    "MIPGap": 1e-6,
    "MIPGapAbs": 1e-7,  # absolute MIP gap, alongside the relative MIPGap
    "FeasibilityTol": 1e-8,
    "OptimalityTol": 1e-8,
    "IntFeasTol": 1e-9,
    "NodefileStart": 16.0,
    "MemLimit": GUROBI_MEMLIMIT_GB,
}

_GRB_STATUS = {
    GRB.OPTIMAL: "optimal",
    GRB.TIME_LIMIT: "time_limit",
    GRB.SUBOPTIMAL: "suboptimal",
    GRB.INFEASIBLE: "infeasible",
    GRB.INF_OR_UNBD: "infeasible",
    GRB.UNBOUNDED: "unbounded",
    GRB.NUMERIC: "error",
    GRB.INTERRUPTED: "interrupted",
    GRB.MEM_LIMIT: "mem_limit",
}


def gurobi_version() -> str:
    return ".".join(map(str, gp.gurobi.version()))


def mosek_version() -> str:
    import mosek

    return ".".join(map(str, mosek.Env().getversion()))


def cvxpy_version() -> str:
    import cvxpy

    return cvxpy.__version__


def gurobi_model(name: str, threads: int | None = None, time_limit: float | None = None,
                 **params) -> tuple[gp.Model, dict]:
    """A silent Gurobi model with the package defaults, the thread budget and ``params`` overrides.
    Returns (model, the parameter dict actually set)."""
    env = _quiet_env()
    m = gp.Model(name, env=env)
    used = dict(GUROBI_DEFAULTS)
    used.update(params)
    used["NodefileStart"] = min(float(used["NodefileStart"]), float(used["MemLimit"]))  # spill before the hard limit
    if time_limit is not None:
        used["TimeLimit"] = float(time_limit)
    for k, v in used.items():
        m.setParam(k, v)
    used["Threads"] = compute.configure_gurobi(m, threads)
    return m, used


_ENV = None


def _quiet_env() -> gp.Env:
    global _ENV
    if _ENV is None:
        _ENV = gp.Env(empty=True)
        _ENV.setParam("OutputFlag", 0)
        _ENV.start()
    return _ENV


def gurobi_status(m: gp.Model) -> str:
    return _GRB_STATUS.get(m.Status, f"error_status_{m.Status}")


_MOSEK_LAST: dict = {}


def _install_mosek_capture() -> None:
    """Wrap CVXPY's MOSEK ``invert`` so that the MOSEK task's own solution info is read before CVXPY deletes the
    task (needed by :func:`certified_value`). CVXPY *dualizes* conic problems for MOSEK: the task is
    ``(D) max{-b^T y : A^T y = c, y in K*}``, the dual of CVXPY's ``(P) min{c^T x : A x + b in K}``. Our problems
    are ``max f = -min(-f)``, so (weak duality) ``f* <= -pobj_task`` whenever the task's primal point ``y`` is
    feasible: ``value_dual = -pobj``, certified by the task's *primal* residuals; ``value_primal = -dobj``
    (= CVXPY's ``prob.value``, the objective at the recovered ``X``)."""
    from cvxpy.reductions.solvers.conic_solvers.mosek_conif import MOSEK

    if getattr(MOSEK, "_aimp_capture", False):
        return
    orig = MOSEK.invert

    def invert(self, solver_output, inverse_data):
        _MOSEK_LAST.clear()
        task = solver_output.get("task") if isinstance(solver_output, dict) else None
        if task is not None:
            try:
                import mosek

                st = mosek.soltype.itr
                if task.solutiondef(st):
                    (pobj, pviolcon, pviolvar, pviolbarvar, pviolcone, _pviolitg, dobj, dviolcon, dviolvar,
                     dviolbarvar, dviolcone) = task.getsolutioninfo(st)
                    from cvxpy.settings import OBJ_OFFSET

                    off = inverse_data.get(OBJ_OFFSET, 0.0) if isinstance(inverse_data, dict) else 0.0
                    _MOSEK_LAST.update({
                        "task_pobj": pobj, "task_dobj": dobj, "obj_offset": float(off or 0.0),
                        "task_solsta": str(task.getsolsta(st)), "task_prosta": str(task.getprosta(st)),
                        # residuals of OUR dual (= the task's primal) and of OUR primal (= the task's dual)
                        "dual_resid": float(max(pviolcon, pviolvar, pviolbarvar, pviolcone)),
                        "dual_resid_parts": [pviolcon, pviolvar, pviolbarvar, pviolcone],
                        "primal_resid": float(max(dviolcon, dviolvar, dviolbarvar, dviolcone)),
                    })
            except Exception as e:  # never fatal: the value then falls back to the primal value
                _MOSEK_LAST["capture_error"] = repr(e)
        return orig(self, solver_output, inverse_data)

    MOSEK.invert = invert
    MOSEK._aimp_capture = True


def cvx_solve(prob, threads: int | None = None, **mosek_extra) -> tuple[str, dict]:
    """Solve a CVXPY problem with MOSEK under the thread budget; returns (status, info).

    ``OPTIMAL_INACCURATE`` maps to ``suboptimal``. ``info["mosek_task"]`` holds MOSEK's own solution info
    (see :func:`_install_mosek_capture`)."""
    import cvxpy as cp

    _install_mosek_capture()
    _MOSEK_LAST.clear()
    params = compute.mosek_params(threads, **mosek_extra)
    info = {"mosek_params": params}
    t_call = time.perf_counter()
    try:
        prob.solve(solver=cp.MOSEK, mosek_params=params, verbose=False)
    except cp.error.SolverError as e:
        return "error", {**info, "message": str(e), "mosek_task": dict(_MOSEK_LAST)}
    except Exception as e:  # e.g. mosek.Error on ill-scaled data (err_huge_c): a failed solve, not a crash
        if type(e).__module__.split(".")[0] != "mosek":
            raise
        return "error", {**info, "message": f"mosek.Error: {e}", "mosek_task": dict(_MOSEK_LAST)}
    info["mosek_task"] = dict(_MOSEK_LAST)
    info["solve_call_s"] = time.perf_counter() - t_call  # CVXPY compile + MOSEK + inversion
    info["compilation_time"] = getattr(prob, "compilation_time", None)
    st = prob.status
    status = {
        cp.OPTIMAL: "optimal",
        cp.OPTIMAL_INACCURATE: "suboptimal",
        cp.INFEASIBLE: "infeasible",
        cp.INFEASIBLE_INACCURATE: "infeasible",
        cp.UNBOUNDED: "unbounded",
        cp.UNBOUNDED_INACCURATE: "unbounded",
    }.get(st, "error")
    stats = prob.solver_stats
    info.update({
        "cvxpy_status": st,
        "solve_time": getattr(stats, "solve_time", None),
        "num_iters": getattr(stats, "num_iters", None),
    })
    try:  # primal/dual objective from MOSEK's own solution info
        extra = stats.extra_stats
        if extra is not None and hasattr(extra, "getdouinf"):
            import mosek

            info["primal_obj"] = extra.getdouinf(mosek.dinfitem.intpnt_primal_obj)
            info["dual_obj"] = extra.getdouinf(mosek.dinfitem.intpnt_dual_obj)
    except Exception:
        pass
    return status, info


DUAL_RESID_TOL = 1e-8  # dual residual below which MOSEK's dual objective is taken as the value (certified_value)
DUAL_RESID_RELATIVE = True  # True: dual_resid <= 1e-8 max(1, |value|); False: the absolute rule dual_resid <= 1e-8


def certified_value(status: str, value_primal: float | None, info: dict) -> dict:
    """Certified value of a *maximization* solved by :func:`cvx_solve`: ``value := value_dual`` when MOSEK's dual
    residual is ``<= 1e-8 max(1, |value_dual|)`` (relative; see ``DUAL_RESID_RELATIVE``); otherwise
    ``max(value_primal, value_dual)`` flagged ``unreliable``. A solve that CVXPY reports as failed
    (``status == "error"``) still gets a value when its dual point is certified (``"dual"``); otherwise it keeps no
    value. Returns ``{value, value_primal, value_dual, dual_resid, primal_resid, value_source, unreliable}``."""
    import math

    t = info.get("mosek_task") or {}
    vp = value_primal if (value_primal is not None and math.isfinite(value_primal)) else None
    out = {"value_primal": vp, "value_dual": None, "dual_resid": t.get("dual_resid"),
           "primal_resid": t.get("primal_resid"), "value_source": "primal", "unreliable": False,
           "value": vp if status in ("optimal", "suboptimal") else None}
    if "task_pobj" not in t:
        return out
    vd = -(t["task_pobj"] + t["obj_offset"])
    out["value_dual"] = vd
    out["value_primal_task"] = -(t["task_dobj"] + t["obj_offset"])
    if t["dual_resid"] <= DUAL_RESID_TOL * (max(1.0, abs(vd)) if DUAL_RESID_RELATIVE else 1.0):
        out.update(value=vd, value_source="dual")
    elif status in ("optimal", "suboptimal"):
        cand = [v for v in (vp, vd) if v is not None]
        out.update(value=max(cand), value_source="max_unreliable", unreliable=True)
    else:
        out.update(value=None, value_source="none_unreliable", unreliable=True)
    return out
