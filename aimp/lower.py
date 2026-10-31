"""The strong lower bound ``lb_strong`` on ``p_star`` (and the feasible ``phi_min_strong``): the best feasible value
from multistart local ascent, with the starting points listed in Appendix B of the paper.

Starts (each a feasible point ``Z``; every start type is recorded):

* ``center``  — the token centres;
* ``greedy``  — the greedy's support-maximizer point (``argmax_{Z_j} c^T N z`` per key; for ``-c`` the one of ``-c``);
* ``sdp_ubar`` — the SDP's first-moment point ``X[0, u]`` (projected into the sets);
* ``sdp_eig1..3`` — the top-3 eigenvector roundings of the moment matrix: ``v_k / v_k[0]`` restricted to the ``u`` blocks
  (sign with ``v_k[0] >= 0``; ``v_k`` itself when ``|v_k[0]| < 1e-8``), each scaled radially into the sets;
* ``bb_leaf`` — every branch-and-bound leaf's ``ubar`` (:func:`aimp.certify.branch_and_bound`, when present);
* ``random`` — 24 seeded points uniform on the sets' boundaries (every token on the sphere of its base ball);
* ``extra``   — any further caller-supplied points (e.g. the recovered point of an earlier SDP solve).

Local ascent: L-BFGS-B in base coordinates with the ball constraints by radial projection (the parametrization of
:class:`aimp.lower_bounds._Param`, analytic gradients), ``maxiter = 2000``, ``ftol = 1e-12``, ``gtol = 1e-9``.
Every value is a numpy re-evaluation of the true head at a feasible point, so ``lb_strong <= p_star`` and
``phi_min_strong >= phi_min``. ``floor`` (e.g. ``lb_old``) is a value known to be attained at a feasible point; if it beats
every ascent by more than ``FLOOR_TOL`` (relative) it is reported as the winner (``"floor"``) — useful for units whose
moment matrix was not stored.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from aimp import sets as S
from aimp.heads import phi
from aimp.lower_bounds import _Param
from aimp.types import Instance

MAXITER = 2000
FTOL = 1e-12
GTOL = 1e-9
N_RANDOM = 24
FLOOR_TOL = 1e-9


def boundary_points(inst: Instance, rng: np.random.Generator, k: int) -> list[np.ndarray]:
    """``k`` points with every free token uniform on the boundary of its base set (ball: sphere; box: a random vertex
    face point)."""
    out = []
    reps = {t: S.affine(inst.sets[t]) for t in inst.tokens}
    for _ in range(k):
        Z = inst.centers().copy()
        for t in inst.tokens:
            rep = reps[t]
            if not rep.r:
                continue
            if rep.base == "ball":
                g = rng.standard_normal(rep.r)
                u = rep.rho * g / np.linalg.norm(g)
            elif rep.base == "box":
                u = rng.uniform(-1, 1, rep.r) * rep.h
                f = rng.integers(rep.r)
                u[f] = rep.h[f] * (1 if rng.random() < 0.5 else -1)
            else:
                u = S.base_sample(rep, rng, 1)[0]
            Z[t] = S.to_z(rep, u)
        out.append(Z)
    return out


def sdp_starts(inst: Instance, X: np.ndarray, upos: dict, n_eig: int = 3) -> dict[str, np.ndarray]:
    """``sdp_ubar`` and ``sdp_eig{k}`` points from a moment matrix ``X`` with the ``u``-block positions ``upos[t]``."""
    reps = {t: S.affine(inst.sets[t]) for t in inst.tokens}

    def point(vec):
        Z = inst.centers().copy()
        for t in inst.tokens:
            if reps[t].r:
                Z[t] = S.to_z(reps[t], S.base_project(reps[t], np.asarray(vec[upos[t]], float)))
        return Z

    out = {"sdp_ubar": point(X[0])}
    ev, V = np.linalg.eigh(X)
    for k in range(min(n_eig, X.shape[0])):
        v = V[:, -1 - k]
        if v[0] < 0:
            v = -v
        out[f"sdp_eig{k + 1}"] = point(v / v[0] if abs(v[0]) >= 1e-8 else v)
    return out


def ascend(inst: Instance, P: _Param, Z0: np.ndarray, maxiter: int = MAXITER) -> tuple[float, np.ndarray, dict]:
    p0 = P.p_of_Z(Z0)
    if P.dim == 0:
        return float(phi(inst, Z0)), Z0, {"nit": 0, "success": True}
    res = minimize(P.fun_grad, p0, jac=True, method="L-BFGS-B", bounds=P.bounds(),
                   options={"maxiter": maxiter, "ftol": FTOL, "gtol": GTOL})
    Z = P.Z_of(res.x)
    return float(phi(inst, Z)), Z, {"nit": int(res.nit), "success": bool(res.success)}


def strong_lb(inst: Instance, seed: int, greedy_Z: np.ndarray | None = None,
              starts: dict[str, np.ndarray] | None = None, leaves: list[np.ndarray] | None = None,
              n_random: int = N_RANDOM, maxiter: int = MAXITER, floor: float | None = None) -> dict:
    """Best local ascent over the starts of the module docstring. ``starts``: named points (``sdp_*``, ``extra*``);
    ``leaves``: branch-and-bound leaf points. Returns ``value``, ``Z``, ``winner`` (start type), per-start values."""
    t0 = time.perf_counter()
    P = _Param(inst)
    rng = np.random.default_rng(seed)
    named: list[tuple[str, np.ndarray]] = [("center", inst.centers())]
    if greedy_Z is not None:
        named.append(("greedy", np.asarray(greedy_Z, float)))
    for k, Z in (starts or {}).items():
        named.append((k, np.asarray(Z, float)))
    for Z in leaves or []:
        named.append(("bb_leaf", np.asarray(Z, float)))
    named += [("random", Z) for Z in boundary_points(inst, rng, n_random)]
    best_v, best_Z, best_k = -np.inf, inst.centers(), None
    per, nits, nok = [], [], 0
    for k, Z0 in named:
        v, Z, info = ascend(inst, P, Z0, maxiter)
        per.append((k, v))
        nits.append(info["nit"])
        nok += info["success"]
        if v > best_v:
            best_v, best_Z, best_k = v, Z, k
    winner = best_k  # the floor is credited only when it beats every ascent by more than FLOOR_TOL (relative)
    if floor is not None and np.isfinite(floor) and floor > best_v + FLOOR_TOL * max(1.0, abs(best_v)):
        winner = "floor"
    by_type: dict[str, float] = {}
    for k, v in per:
        by_type[k] = max(by_type.get(k, -np.inf), v)
    return {"value": float(max(best_v, floor if (floor is not None and np.isfinite(floor)) else -np.inf)),
            "value_ascent": float(best_v), "Z": best_Z, "winner": winner, "best_by_type": by_type,
            "n_starts": len(named), "n_converged": nok, "nit_median": float(np.median(nits)) if nits else 0.0,
            "nit_max": int(max(nits)) if nits else 0, "wall_s": time.perf_counter() - t0,
            "params": {"maxiter": maxiter, "ftol": FTOL, "gtol": GTOL, "n_random": n_random, "seed": seed}}


def neg_instance(inst: Instance) -> Instance:
    return Instance(head=inst.head, keys=inst.keys, query=inst.query, sets=inst.sets, c=-np.asarray(inst.c, float),
                    meta=dict(inst.meta))


def greedy_point(inst: Instance) -> np.ndarray:
    """The greedy's support-maximizer point (``relax_ps.solve_ps_greedy``'s recovered point)."""
    g = inst.head.N.T @ inst.c
    Z = inst.centers().copy()
    for j in inst.keys:
        Z[j] = S.support_argmax(inst.sets[j], g)
    if not inst.query_in_keys:
        Z[inst.query] = inst.sets[inst.query].center
    return Z


def strong_bracket(inst: Instance, seed: int, starts: dict | None = None, leaves: list | None = None,
                   lb_floor: float | None = None, phi_min_floor: float | None = None,
                   n_random: int = N_RANDOM, maxiter: int = MAXITER) -> dict:
    """``lb_strong`` (on ``+c``) and ``phi_min_strong`` (``-`` the same on ``-c``; feasible, so ``>= phi_min``). The
    ``-c`` run uses the same SDP / leaf points as starts and the ``-c`` greedy point. ``phi_min_floor``: a known feasible
    value of ``phi`` (e.g. an earlier multistart minimum ``phi_min_ms``), which caps ``phi_min_strong`` from above."""
    up = strong_lb(inst, seed, greedy_point(inst), starts, leaves, n_random, maxiter, lb_floor)
    neg = neg_instance(inst)
    dn = strong_lb(neg, seed + 7919, greedy_point(neg), starts, leaves, n_random, maxiter,
                   None if phi_min_floor is None else -phi_min_floor)
    return {"lb_strong": up["value"], "lb_strong_ascent": up["value_ascent"], "lb_winner": up["winner"],
            "lb_best_by_type": up["best_by_type"], "phi_min_strong": -dn["value"], "phi_min_winner": dn["winner"],
            "phi_min_best_by_type": {k: -v for k, v in dn["best_by_type"].items()},
            "lb_Z": up["Z"], "phi_min_Z": dn["Z"],
            "lb_info": {k: up[k] for k in ("n_starts", "n_converged", "nit_median", "nit_max", "wall_s", "params")},
            "phi_min_info": {k: dn[k] for k in ("n_starts", "n_converged", "nit_median", "nit_max", "wall_s")},
            "wall_s": up["wall_s"] + dn["wall_s"]}


# ------------------------------------------------------------------------------------------------ alternating ascent
ALT_ROUNDS = 20
ALT_GAIN_REL = 1e-12


def fix_query_at(inst: Instance, z: np.ndarray) -> Instance:
    """The instance with the query token (and its own key and value) fixed at the point ``z`` of its set."""
    st = inst.sets[inst.query]
    pt = S.ellipsoid(np.asarray(z, float), st.E, 0.0) if st.kind == "ellipsoid" else S.point(np.asarray(z, float))
    sets = dict(inst.sets)
    sets[inst.query] = pt
    return Instance(head=inst.head, keys=list(inst.keys), query=inst.query, sets=sets, c=np.asarray(inst.c, float),
                    meta={**inst.meta, "fixed_query": True})


def admissible_excess(inst: Instance, Z: np.ndarray) -> float:
    """Largest relative excess of ``Z`` over the token sets (``<= 1e-9``: admissible); ellipsoids by their base radius."""
    worst = -np.inf
    for t in inst.tokens:
        rep = S.affine(inst.sets[t])
        if not rep.r:
            worst = max(worst, float(np.linalg.norm(Z[t] - rep.c)) / max(1.0, float(np.linalg.norm(rep.c))))
            continue
        w, *_ = np.linalg.lstsq(rep.B, Z[t] - rep.c, rcond=None)
        worst = max(worst, float(np.linalg.norm(w)) / rep.rho - 1.0,
                    float(np.linalg.norm(rep.B @ w - (Z[t] - rep.c))) / max(1.0, float(np.linalg.norm(rep.c))))
    return worst


def alternating_ascent(inst: Instance, Z0: np.ndarray, spread: float, rounds: int = ALT_ROUNDS,
                       gain_rel: float = ALT_GAIN_REL) -> dict:
    """Block-coordinate ascent from ``Z0`` (Section 7, "stronger ascents"): (a) the keys := the exact maximizers of
    Theorem 7 for the current query point (with the query's own key and value fixed with it); (b) the query := local
    ascent with the other tokens fixed. Stops when a round gains less than ``gain_rel x spread`` or after ``rounds``
    rounds. Every value is a forward pass at an admissible input."""
    from aimp.exact_fq import fq_problem, solve_max, worst_input
    from aimp.heads import forward

    i = inst.query
    Z = np.asarray(Z0, float).copy()
    v_prev = float(forward(inst, Z)["phi"])
    trace, statuses = [{"round": 0, "value": v_prev}], []
    best_v, best_Z = v_prev, Z.copy()
    stop = "rounds"
    for k in range(1, rounds + 1):
        fqi = fix_query_at(inst, Z[i])
        P = fq_problem(fqi)
        mx = solve_max(P)
        statuses.append(mx["status"])
        Za = worst_input(P, fqi, mx["t"]) if mx["t"] is not None else fqi.centers()
        va = float(forward(inst, Za)["phi"])
        sets_q = {t: (inst.sets[t] if t == i else S.point(Za[t])) for t in inst.tokens}
        iq = Instance(head=inst.head, keys=list(inst.keys), query=i, sets=sets_q, c=np.asarray(inst.c, float))
        vb, Zb, info = ascend(iq, _Param(iq), Za)
        Zb = np.asarray(Zb, float)
        vb = float(forward(inst, Zb)["phi"])
        v_round, Z_round = max([(va, Za), (vb, Zb)], key=lambda x: x[0])
        trace.append({"round": k, "value_keys": va, "value_query": vb, "p_fq_lo": mx["p_lo"], "p_fq_hi": mx["p_hi"],
                      "exact_status": mx["status"], "query_nit": info["nit"]})
        if v_round > best_v:
            best_v, best_Z = v_round, Z_round.copy()
        gain = v_round - v_prev
        v_prev = max(v_prev, v_round)
        Z = Z_round
        if gain < gain_rel * spread:
            stop = "gain"
            break
    return {"value": best_v, "Z": best_Z, "rounds": len(trace) - 1, "stop": stop, "trace": trace,
            "exact_unresolved": int(sum(s != "exact" for s in statuses)), "admissible_excess": admissible_excess(inst, best_Z)}
