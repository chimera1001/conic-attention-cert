"""The re-solve on the rescaled formulation and the classification of the theorem checks used for gpt-oss-20b.

On gpt-oss heads a single key often holds almost all of the attention and the others have weights of 1e-5 or less. The
interior-point solver then satisfies the interval diagonal of those keys, ``X[a, a] <= (a_lo + a_hi) a - a_lo a_hi``, only to its
absolute feasibility tolerance (~1e-8) while the entries themselves are ~1e-10. Through the square root in the covariance term this
can raise the objective by ~1e-5. ``value_safe`` stays a rigorous upper bound on the relaxation, but a loose one. The remedy is a
second solve of the same relaxation on the equivalent formulation with every attention weight divided by its upper bound
(``solve_sdp1(scale_a=True)``), keeping the smaller of the two safe values.

* :func:`residual_cov` -- the covariance that the residuals of a solver's point allow: per key, the Schur bound
  ``sqrt(var_a var_u)`` at the point minus its value with the interval diagonal and the ball constraint exact at the same ``a``.
* :func:`apply_rescaled` -- the trigger (that covariance exceeds 1% of ``cert`` minus the best feasible value, or a theorem
  check is not satisfied) and the rescaled re-solve at the 1e-9 tolerances.
* :func:`classify` -- the checks: satisfied if ``value - lb <= bound + 1e-6``; unresolved if the excess is within the floor
  ``5e-8 max|h| + slack`` (the solver's measured accuracy on equivalent formulations) or below the residual-allowed covariance of the
  final solve; violated if ``value - ub - bound`` exceeds all of them; else inconclusive.
"""
from __future__ import annotations

import math
import time

import numpy as np

REL = 0.01
FLOOR_REL = 5e-8


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def residual_cov(inst, ctrl: dict, X: np.ndarray, pos_a, upos: dict) -> float:
    from aimp import sets as S

    gN = inst.head.N.T @ np.asarray(inst.c, float)
    a_lo, a_hi = np.clip(np.asarray(ctrl["a_lo"], float), 0, 1), np.clip(np.asarray(ctrl["a_hi"], float), 0, 1)
    tot = 0.0
    for idx, j in enumerate(inst.keys):
        rep = S.affine(inst.sets[j])
        if not rep.r:
            continue
        pa, up = int(pos_a[idx]), upos[j]
        g = rep.B.T @ gN
        ab = float(X[0, pa])
        va = float(X[pa, pa] - ab ** 2)
        vu = float(g @ (X[np.ix_(up, up)] - np.outer(X[0, up], X[0, up])) @ g)
        cap = (ab - a_lo[idx]) * (a_hi[idx] - ab)
        tot += max(0.0, math.sqrt(max(va, 0.0) * max(vu, 0.0)) - math.sqrt(max(cap, 0.0)) * float(rep.rho)
                   * float(np.linalg.norm(g)))
    return tot


def resolve(inst, ctrl: dict, threads: int | None, scale_a: bool) -> dict:
    """``sdp_1`` at the 1e-9 tolerances (the re-solve of Appendix B), on the original or the rescaled formulation, with the
    residual-allowed covariance of its point."""
    from aimp.sdp_mosek import solve_sdp1

    t0 = time.perf_counter()
    r = solve_sdp1(inst, ctrl, threads=threads, time_limit=None, tol=None, cut="auto", keep_X=True, scale_a=scale_a)
    dg = r.diagnostics
    vs = dg.get("value_safe")
    return {"value_safe": float(vs) if _finite(vs) else None, "value_dual": dg.get("value_dual"),
            "value_primal": dg.get("value_primal"), "status": r.status, "solsta": dg.get("solsta"),
            "safe_slack": (vs - dg["value_dual"]) if (_finite(vs) and _finite(dg.get("value_dual"))) else None,
            "primal_max_violation": dg.get("primal_max_violation"), "num_iters": dg.get("num_iters"),
            "C_res": residual_cov(inst, ctrl, r.X, r.pos_a, r.upos) if getattr(r, "X", None) is not None else None,
            "wall_s": time.perf_counter() - t0}


def apply_rescaled(inst, ctrl: dict, rec: dict, C0: float | None, lb_best: float, flagged: bool, threads: int | None) -> dict:
    """The trigger and the rescaled re-solve; lowers ``rec['value_safe']`` and ``rec['cert']`` when the rescaled safe value is
    smaller. Returns the record's ``option_b`` field."""
    vs_pre, cert_pre = rec.get("value_safe"), rec["cert"]
    gap = cert_pre - lb_best
    trig = bool(_finite(C0) and C0 > REL * max(gap, 0.0))
    info = {"C_res_orig": C0, "gap_pre": gap, "value_safe_pre": vs_pre, "cert_pre": cert_pre, "trigger_cov": trig,
            "trigger_flagged": bool(flagged), "triggered": bool(trig or flagged), "rescaled": None, "lowered": False,
            "final_solve": "original", "C_res_final": C0}
    if info["triggered"] and vs_pre is not None:
        rb = resolve(inst, ctrl, threads, scale_a=True)
        info["rescaled"] = rb
        if rb["value_safe"] is not None and rb["value_safe"] < vs_pre:
            rec["value_safe"] = rb["value_safe"]
            rec["cert"] = min(rec["r_greedy"], rb["value_safe"])
            info.update({"lowered": True, "final_solve": "rescaled", "C_res_final": rb["C_res"]})
    info.update({"cert_final": rec["cert"], "cert_decrease": cert_pre - rec["cert"],
                 "cert_decrease_over_gap": (cert_pre - rec["cert"]) / gap if gap > 0 else None})
    return info


def classify(value, bound, lb, ub, floor: float, cres: float = 0.0) -> str | None:
    if value is None or bound is None or not _finite(value) or not _finite(bound):
        return None
    c = cres if _finite(cres) else 0.0
    if value - lb <= bound + 1e-6:
        return "satisfied"
    if value - lb - bound <= max(floor, c):
        return "unresolved"
    if ub is not None and value - ub - bound > max(1e-6, floor, c):
        return "violated"
    return "inconclusive"


def h_scale(inst) -> float:
    from aimp.theory import directional_constants

    dc = directional_constants(inst)
    return float(np.max(np.abs(np.r_[dc["h_sup"], dc["h_inf"]])))
