"""The certificate of Definition 4 (Algorithm 1), branch-and-bound over the scores, and the certified radius.

``certify(inst, ctrl, eps_target)``::

    r_greedy = r_pp by the greedy algorithm (Proposition 8; cut = the width pair by default, cut="auto" the pair with
               the smaller radius; attention intervals of Lemma 2)
    if bound_C <= eps_target:  cert = r_greedy                       route "margin"
    else:                      cert = min(r_greedy, value_safe(sdp_1)) route "min"

``sdp_1`` = ``persp + simplex_x_z`` (the level-one relaxation of Definition 3) with the log-attention intervals (14),
cone form, and ``value := value_safe`` (the safe dual bound of Proposition 16) **always**; the primal / dual /
dual-residual-certified values are kept as diagnostics. The second-order gap bound is checked in two norm-based forms
(``F0t`` = the ``eta_exp(w_t)`` constant; ``F0s`` = ``2S/(1-S)``, diagnostic; bound (39) itself is
:func:`aimp.theory.f0v_final`). Without an exact reference the check uses the best known lower bound ``lb`` and upper
bound ``ub`` on ``p_star``: ``value_safe - lb <= bound`` proves it (``"satisfied"``), ``value_safe - ub > bound``
refutes it (``"violated"``; the true gap exceeds the bound), anything in between is ``"inconclusive"``.

``rho_star(gap_fn, eps, lo, hi)`` — first crossing of ``gap_fn(rho) <= eps``: half-decade grid scan, then bisection in ``log rho``.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Optional

import numpy as np

from aimp.types import Instance

SDP_1_RLT = ("persp", "simplex_x_z")
CHECK_TOL = 1e-6  # as theory.THEOREM_TOL


def theorem_check(value: Optional[float], bound: Optional[float], lb: Optional[float], ub: Optional[float],
                  tol: float = CHECK_TOL) -> dict:
    """Three-valued check of ``value - p_star <= bound`` from ``lb <= p_star <= ub``."""
    if value is None or bound is None or not math.isfinite(value):
        return {"status": None, "gap_lb": None, "excess_lb": None}
    gap_lb = None if lb is None else value - lb
    if gap_lb is not None and gap_lb <= bound + tol:
        st = "satisfied"
    elif ub is not None and value - ub > bound + tol:
        st = "violated"
    else:
        st = "inconclusive"
    return {"status": st, "gap_lb": gap_lb, "excess_lb": None if gap_lb is None else gap_lb - bound,
            "ratio_lb": None if (gap_lb is None or bound <= 0) else gap_lb / bound}


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _task_api_ok(inst: Instance) -> bool:
    from aimp import sets as S

    return all(S.affine(inst.sets[t]).base in ("ball", "point") for t in inst.tokens)


def certify(inst: Instance, ctrl: dict | None = None, eps_target: float = 1e-3, threads: int | None = None,
            time_limit: float | None = None, run_sdp: bool = True,
            lb: float | None = None, ub: float | None = None, nom: dict | None = None,
            score_method: str = "auto", sdp_kw: dict | None = None, sdp_tol: dict | None = None,
            return_sdp: bool = False, cut: str = "B") -> dict:
    """The certificate of Definition 4 on one instance (Algorithm 1 without branching; the margin route when the margin
    bound ``bound_C`` meets ``eps_target``). Returns a flat JSON-able record.

    ``run_sdp=False`` stops after the greedy (route ``"greedy_only"``; the first-order certificate only, for sweeps of
    the greedy). ``lb`` / ``ub``: externally known bounds on ``p_star`` (e.g. a multistart value, an exact
    ``ObjBound``) used by the bound checks. ``cut``: the l1 pattern cut of the greedy and of ``sdp_1`` (default the
    width pair ``"B"``; ``"auto"``, the pair with the smaller radius, is the choice of the paper's experiments)."""
    from aimp import bounds
    from aimp.heads import nominal
    from aimp.relax_ps import solve_ps_greedy
    from aimp.theory import sandwich_S

    t0 = time.perf_counter()
    times: dict[str, float] = {}
    if ctrl is None:
        ctrl = bounds.controls(inst, score_method, eps_target=eps_target, threads=threads)
        times["controls_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    g = solve_ps_greedy(inst, ctrl, cut)
    times["greedy_s"] = time.perf_counter() - t1
    nomv = nominal(inst)
    ahat_nom = nomv["a"]
    n = inst.n
    bC = ctrl.get("bound_C")
    rec: dict = {
        "n": n, "eps_target": eps_target, "phi_nominal": nomv["phi"],
        "S": sandwich_S(ctrl), "w_t": ctrl["w_t"], "w_t_loose": ctrl.get("w_t_loose"),
        "theta": float(1.0 - ahat_nom @ ahat_nom), "theta_mid": float(1.0 - np.asarray(ctrl["ahat"]) @ ctrl["ahat"]),
        "gamma": ctrl["gamma"], "Delta": ctrl["Delta"], "rad": ctrl["rad"], "c_norm": ctrl["c_norm"],
        "bound_B": ctrl["bound_B"], "bound_C": bC, "regime": ctrl["regime"],
        "r_greedy": g.value, "greedy_cut": g.diagnostics["cut_used"], "greedy_recovered": g.recovered_value,
    }
    lbs = [v for v in (lb, g.recovered_value) if _finite(v)]
    ubs = [v for v in (ub, g.value) if _finite(v)]
    sdp_rec: dict = {}
    if bC is not None and bC <= eps_target:
        route, cert = "margin", g.value
    elif not run_sdp:
        route, cert = "greedy_only", g.value
    else:
        from aimp.relax_sdp import PRUNE_TOL, solve_sdp
        from aimp.theory import bound_F0s, bound_F0t, nominal_decomposition, support_values

        kw = {"t_interval": "tight", "cone_form": True, "safe": "always", **(sdp_kw or {})}
        t2 = time.perf_counter()
        if _task_api_ok(inst) and not sdp_kw:
            from aimp.sdp_mosek import solve_sdp1  # same relaxation, MOSEK task API

            r = solve_sdp1(inst, ctrl, threads=threads, time_limit=time_limit, tol=sdp_tol, keep_X=return_sdp, cut=cut)
            vs = r.diagnostics.get("value_safe")
            vs = float(vs) if _finite(vs) else None
        else:
            r = solve_sdp(inst, ctrl, rlt=SDP_1_RLT, threads=threads, time_limit=time_limit, cut=cut, **kw)
            vs = r.diagnostics.get("value_safe")
            vs = float(vs) if _finite(vs) else None
        times["sdp_total_s"] = time.perf_counter() - t2
        dg = r.diagnostics
        times.update({"sdp_build_s": dg.get("time_build_s"), "sdp_solve_call_s": dg.get("solve_call_s"),
                      "mosek_solve_s": dg.get("solve_time"), "cvxpy_compile_s": dg.get("compilation_time"),
                      "value_safe_s": (dg.get("safe_info") or {}).get("time_s"), "safe_stage_s": dg.get("time_safe_s")})
        if _finite(r.recovered_value):
            lbs.append(r.recovered_value)
        if vs is not None:
            route, cert = "min", min(g.value, vs)
        else:
            route, cert = "min_sdp_failed", g.value
        sdp_rec = {
            "sdp_status": r.status, "value_safe": vs, "sdp_value_certified": r.value if _finite(r.value) else None,
            "sdp_value_source": dg.get("value_source"), "sdp_value_primal": dg.get("value_primal"),
            "sdp_value_dual": dg.get("value_dual"), "sdp_dual_resid": dg.get("dual_resid"),
            "sdp_primal_resid": dg.get("primal_resid"), "sdp_unreliable": dg.get("unreliable"),
            "safe_info": dg.get("safe_info"), "sdp_recovered": r.recovered_value,
            "sdp_gap_constructive": None if (vs is None or not _finite(r.recovered_value)) else vs - r.recovered_value,
            "rank1_ratio": dg.get("rank1_ratio"), "X_min_eig": dg.get("X_min_eig"), "top_eigs": dg.get("top_eigs"),
            "primal_max_violation": dg.get("primal_max_violation"), "D": r.params.get("D"),
            "pruned_keys": dg.get("pruned_keys"), "pruning_slack": dg.get("pruning_slack", 0.0),
            "w_t_used": dg.get("w_t_used"), "num_iters": dg.get("num_iters"), "threads": r.params.get("MSK_IPAR_NUM_THREADS"),
            "sdp_approximations": list(r.approximations),
            "sdp_solver": r.params.get("solver", "cvxpy"), "sdp_nrows": dg.get("nrows"),
            "sdp_tol": {k: r.params.get(k) for k in ("MSK_DPAR_INTPNT_CO_TOL_PFEAS", "MSK_DPAR_INTPNT_CO_TOL_DFEAS",
                                                      "MSK_DPAR_INTPNT_CO_TOL_REL_GAP")},
            "sdp_solsta": dg.get("solsta"), "safe_slack": None if (vs is None or dg.get("value_dual") is None)
            else vs - dg["value_dual"],
        }
        # --- norm-based second-order bound checks, both forms
        t3 = time.perf_counter()
        nom = nom or nominal_decomposition(inst)
        a_hi = np.asarray(ctrl["a_hi"], float)
        hv = support_values(inst)
        slack = float(sum(a_hi[k] * (hv[k] - hv.min()) for k in range(n) if a_hi[k] <= PRUNE_TOL))
        f0t = bound_F0t(inst, ctrl, nom, mode="tight")
        fs = bound_F0s(inst, ctrl, nom, slack)
        lb_best, ub_best = max(lbs) if lbs else None, min(ubs) if ubs else None
        bt = f0t["bound_F0t"] + slack
        sdp_rec.update({
            "bound_F0t": bt, "F0t_eta_exp": f0t["F0t_eta_exp"], "F0t_term_sandwich": f0t["F0t_term_sandwich"],
            "F0_term_cov": f0t["F0t_term_cov"], "F0_eta": nom["eta"], "F0_nu": nom["nu"],
            "F0t_check": theorem_check(vs, bt, lb_best, ub_best),
            "bound_F0s": fs["bound_F0s"], "F0s_applies": fs["F0s_applies"],
            "F0s_check": theorem_check(vs, fs["bound_F0s"], lb_best, ub_best) if fs["F0s_applies"] else
            {"status": None},
        })
        times["f0_checks_s"] = time.perf_counter() - t3
    lb_best = max(lbs) if lbs else None
    rec.update(sdp_rec)
    if return_sdp:  # not JSON: the caller pops these (starts of the strong lower bound, aimp.lower)
        rec["_greedy"] = g
        rec["_sdp"] = r if sdp_rec else None
    rec.update({
        "route": route, "cert": cert, "lb_best": lb_best,
        "gap_vs_nominal": cert - nomv["phi"], "gap_vs_lb": None if lb_best is None else cert - lb_best,
        "greedy_gap_vs_nominal": g.value - nomv["phi"],
        "times": {**times, "total_s": time.perf_counter() - t0},
    })
    return rec


def rho_star(gap_fn: Callable[[float], float], eps: float, lo: float, hi: float, steps: int = 6,
             cache: dict | None = None, per_decade: int = 2) -> dict:
    """The certified radius at tolerance ``eps`` as the *first crossing* (the tightness radius of Section 4 when
    ``gap_fn`` is the gap): the largest ``rho`` such that ``gap_fn(rho') <= eps`` at every evaluated ``rho' <= rho``.
    ``gap_fn`` is ``cert(rho) - phi_nominal`` or ``cert(rho) - lb(rho)``; the latter is not monotone in ``rho`` (it
    returns to ~0 when the balls are so large that every pattern is reachable), so a plain bisection on ``[lo, hi]``
    would report the large-radius end. Search: scan upward on the geometric grid ``lo * 10^(k/per_decade)`` up to ``hi``
    until the first failure, then ``steps`` bisections in ``log rho`` between the last passing and the first failing
    grid point. ``cache`` (rho -> gap) is shared across calls.

    Returns ``rho_star`` (largest verified rho), ``rho_upper`` (first rho seen to fail; ``None`` if none),
    ``censored`` in {``None``, ``"above"`` (every grid point up to ``hi`` passes), ``"below"`` (``lo`` fails;
    ``rho_star = None``)}, ``evals`` and ``monotone_evals``."""
    cache = {} if cache is None else cache

    def f(r):
        if r not in cache:
            cache[r] = float(gap_fn(r))
        return cache[r]

    nk = int(round(math.log10(hi / lo) * per_decade))
    grid = [lo * 10 ** (k / per_decade) for k in range(nk)] + [hi]
    evals, last_pass, first_fail = [], None, None
    for r in grid:
        g = f(r)
        evals.append((r, g))
        if g > eps:
            first_fail = r
            break
        last_pass = r
    if first_fail is None:
        return {"rho_star": hi, "rho_upper": None, "censored": "above", "evals": evals, "eps": eps,
                "monotone_evals": _monotone(evals)}
    if last_pass is None:
        return {"rho_star": None, "rho_upper": lo, "censored": "below", "evals": evals, "eps": eps,
                "monotone_evals": _monotone(evals)}
    a, b = last_pass, first_fail
    for _ in range(steps):
        m = math.sqrt(a * b)
        gm = f(m)
        evals.append((m, gm))
        if gm <= eps:
            a = m
        else:
            b = m
    return {"rho_star": a, "rho_upper": b, "censored": None, "evals": evals, "eps": eps,
            "monotone_evals": _monotone(evals)}


def _monotone(evals) -> bool:
    """Are the evaluated gaps nondecreasing in rho (up to 1e-9 relative)?"""
    pts = sorted(evals)
    return all(pts[k + 1][1] >= pts[k][1] - 1e-9 * max(1.0, abs(pts[k][1])) for k in range(len(pts) - 1))


# ------------------------------------------------------------------------------- re-solve rule, branch-and-bound
GUARD_REL = 0.01  # re-solve rule of Appendix B: re-solve at 1e-9 if safe_slack > 0.01 (cert - lb)
BB_K = 24
BB_CLOSE_REL = 1e-3
BB_RECORD_AT = (1, 2, 4, 8, 16, 24)


def needs_retol(safe_slack: float | None, gap: float | None, rel: float = GUARD_REL) -> bool:
    """The re-solve rule of Appendix B: ``safe_slack > 0.01 (cert - lb)``, with ``safe_slack`` the safe value minus the
    solver's dual objective (a non-positive gap always triggers)."""
    if safe_slack is None or gap is None or not (_finite(safe_slack) and _finite(gap)):
        return False
    return safe_slack > rel * gap


def sdp_retol(inst: Instance, ctrl: dict, threads: int | None = None, time_limit: float | None = None,
              cut: str = "B") -> dict:
    """The re-solve of the rule: ``sdp_1`` at the package's default MOSEK tolerances (1e-9). Returns its ``value_safe``,
    ``value_dual``, status, solve time."""
    from aimp.sdp_mosek import solve_sdp1

    t0 = time.perf_counter()
    r = solve_sdp1(inst, ctrl, threads=threads, time_limit=time_limit, tol=None, cut=cut)
    dg = r.diagnostics
    vs = dg.get("value_safe")
    return {"value_safe": float(vs) if _finite(vs) else None, "value_dual": dg.get("value_dual"), "status": r.status,
            "solsta": dg.get("solsta"), "safe_slack": (vs - dg["value_dual"]) if (_finite(vs) and
                                                                                  _finite(dg.get("value_dual"))) else None,
            "mosek_solve_s": dg.get("solve_time"), "wall_s": time.perf_counter() - t0, "num_iters": dg.get("num_iters")}


def split_scores(ctrl: dict) -> np.ndarray:
    """``S_j = a_hi_j w_j^2 / 8`` at a node (the weighted secant slack of key ``j``, split score of ``rule="S"``;
    ``w_j`` the widths of the log-attention intervals (14), ``a_hi`` clipped to [0, 1])."""
    from aimp.bounds import t_intervals

    t_lo, t_hi = t_intervals(ctrl["s_lo"], ctrl["s_hi"], "tight")
    return np.clip(np.asarray(ctrl["a_hi"], float), 0.0, 1.0) * (t_hi - t_lo) ** 2 / 8


def split_box(s_lo: np.ndarray, s_hi: np.ndarray, j: int) -> tuple[tuple, tuple]:
    """The two children of a node: key ``j``'s score interval cut at its midpoint (their union is the parent box)."""
    mid = 0.5 * (float(s_lo[j]) + float(s_hi[j]))
    lo1, hi1 = np.array(s_lo, float), np.array(s_hi, float)
    lo2, hi2 = lo1.copy(), hi1.copy()
    hi1[j] = mid
    lo2[j] = mid
    return (lo1, hi1), (lo2, hi2)


def _guard_mode(guard) -> str:
    """``True`` / ``"node"``: the re-solve rule on every solved node; ``"termination"``: the rule after the search, on
    the root and the open leaves near the upper bound; ``False`` / ``"off"``: none."""
    if guard is True or guard == "node":
        return "node"
    if guard is False or guard is None or guard == "off":
        return "off"
    if guard == "termination":
        return "termination"
    raise ValueError(f"guard {guard!r}")


def default_f0v_inputs(inst: Instance, ctrl: dict, cut: str = "B") -> dict:
    """The unit-level inputs of bound (39) for the bound-lookahead rule, computed from the instance: ``h_max`` and
    ``nu_{c,j}`` (exact support values), ``rad_c`` (directional constants), ``eta`` (the score covariance (18),
    ``max_j ||B_i^T M B_j|| rho_i rho_j``), ``phi_min_cert = -r_greedy(-c)`` with the root controls."""
    from aimp.lower import neg_instance
    from aimp.relax_ps import solve_ps_greedy
    from aimp.theory import directional_constants, nominal_decomposition

    dc = directional_constants(inst)
    return {"h_max": dc["h_sup"], "nu_c": dc["rho_g"], "rad_c": dc["rad_c"], "eta": nominal_decomposition(inst)["eta"],
            "phi_min_cert": -solve_ps_greedy(neg_instance(inst), ctrl, cut).value}


def branch_and_bound(inst: Instance, ctrl: dict, lb: float, phi_min: float | None = None, K: int = BB_K,
                     threads: int | None = None, tol: dict | None = None, time_limit: float | None = None,
                     close_rel: float = BB_CLOSE_REL, r_greedy: float | None = None, guard="termination",
                     ascent_maxiter: int = 2000, leaf_points: list | None = None, rule: str = "lookahead",
                     h_max: np.ndarray | None = None, f0v_inputs: dict | None = None, cut: str = "B") -> dict:
    """Best-first branch-and-bound over the scores (Section 5, Appendix B): a bracket on ``p_star`` in the native
    task-API path.

    Node = the unit with its score box ``[s_lo, s_hi]`` (the root's = the certified intervals of ``ctrl``); node bound =
    ``value_safe(sdp_1)`` with the node's box as the score constraints and the controls (attention and log-attention
    intervals) recomputed from it (:func:`aimp.bounds.child_controls`), capped by the parent's bound (a child is a
    restriction of its parent). Best-first on the node bound. ``UB`` = max over the leaves' bounds (an unsolved child carries its parent's bound; a child is
    discarded only on a MOSEK primal-infeasibility certificate, or pruned when its bound ``<= LB``); ``LB`` = ``lb`` updated
    with a local ascent from every solved node's ``ubar`` point. Stops when ``UB - LB <= close_rel x spread``
    (``spread = LB - phi_min``) or after ``K`` SDP solves (guard re-solves are not counted in ``K``; they are recorded).

    Split key (at the midpoint of its score interval): ``rule="S"`` — ``argmax_j S_j`` (the largest weighted secant
    slack); ``rule="value"`` — ``argmax_j P_j``, ``P_j = S_j (h_j^max - phi_min)_+`` with the node's own ``S_j``, the unit's ``h_j^max`` (``h_max``,
    default the support values of the sets) and ``phi_min``. Both rules fall back to ``argmax_j S_j`` when every
    score of the rule is 0, and to the widest score interval when every ``S_j`` is 0. Every split records the predicted gain
    of both rules at the split key, ``S / (S - 0.75 S_j)`` and ``Pi_v / (Pi_v - 0.75 P_j)``, and the other rule's key.

    ``rule="lookahead"`` (the bound-lookahead rule of Section 5): for every key with ``D_j > 0`` the two children (its
    score interval halved) get their controls (:func:`aimp.bounds.child_controls`) and their bounds (39) with the unit's
    ``h_j^max``, ``phi_min_cert``, ``eta``, ``rad_c``, ``nu_{c,j}`` (``f0v_inputs``, computed by :func:`default_f0v_inputs`
    when not given; plus each child's pruning slack); split
    ``argmin_j max(F^-_j, F^+_j)``, ties broken by the larger ``D_j chi_j``; predicted gain ``F_node / min_j max(F^-, F^+)``.
    Every split also records the keys the S and value rules would have chosen.

    Defaults: ``guard="termination"``, ``rule="lookahead"``. The branching runs of Section 7 split by ``rule="S"``; on
    72 heads the paper compares it with ``rule="lookahead"``.

    ``guard``: ``"node"`` (``True``) — the re-solve rule of Appendix B (``safe_slack > 0.01 (value_safe - LB)`` -> 1e-9
    re-solve, the smaller ``value_safe`` kept) on every solved node; ``"termination"`` — only after the search, on
    the root (the reported root number) and on the open leaves whose bound is within ``0.01 (UB - LB)`` of ``UB``; a leaf's
    bound comes from the nearest solved node on its path whose ``value_safe`` attains it (an unsolved child: its parent),
    which is the node re-solved; a re-solved node's value caps every leaf below it (they are restrictions of it); the leaf
    pass repeats until the leaf attaining ``UB`` has been evaluated; ``"off"`` — no guard.

    Returns the per-solve trace, ``UB_at[k]`` / ``LB_at[k]`` for ``k`` in ``BB_RECORD_AT`` (carried forward after an early
    stop; the search values, before a termination guard), ``UB_at_guarded`` (the same with the guarded root at ``k = 1``
    and the guarded final ``UB`` from ``k = n_solves`` on), the nodes and splits (observed ratio against the final ``LB``
    from the search bounds), ``cert_bb = min(r_greedy, UB)``. ``leaf_points``: the solved nodes' ``ubar`` points."""
    from aimp.bounds import child_controls
    from aimp.lower import ascend
    from aimp.lower_bounds import _Param
    from aimp.sdp_mosek import solve_sdp1
    from aimp.theory import sandwich_S, value_scores

    if rule not in ("S", "value", "lookahead"):
        raise ValueError(f"rule {rule!r}")
    if rule == "lookahead" and f0v_inputs is None:
        f0v_inputs = default_f0v_inputs(inst, ctrl, cut)
    gmode = _guard_mode(guard)
    t0 = time.perf_counter()
    n = inst.n
    P = _Param(inst)
    LB = float(lb)
    LB_Z = None
    nodes: list[dict] = []
    trace: list[dict] = []
    n_solves = 0
    n_retol = 0
    if h_max is None:
        from aimp.theory import support_values

        h_max = support_values(inst)
    h_max = np.asarray(h_max, float)
    phi_rule = None if phi_min is None else float(phi_min)
    if rule in ("value", "lookahead") and phi_rule is None:
        raise ValueError(f"rule={rule!r} needs phi_min")

    def spread() -> float | None:
        return None if phi_min is None else LB - float(phi_min)

    def leaf_UB() -> float:
        return max(nd["bound"] for nd in nodes if nd["leaf"])

    def solve(node: dict, nctrl: dict) -> None:
        nonlocal LB, LB_Z, n_solves, n_retol
        ts = time.perf_counter()
        r = solve_sdp1(inst, nctrl, threads=threads, time_limit=time_limit, tol=tol, cut=cut)
        n_solves += 1
        dg = r.diagnostics
        vs = dg.get("value_safe")
        vs = float(vs) if _finite(vs) else None
        node.update({"solved": True, "status": r.status, "solsta": dg.get("solsta"), "value_safe_raw": vs,
                     "value_dual": dg.get("value_dual"), "mosek_solve_s": dg.get("solve_time"),
                     "num_iters": dg.get("num_iters"), "pruned_keys": dg.get("pruned_keys"), "solve_index": n_solves})
        if str(dg.get("solsta")) == "solsta.prim_infeas_cer":
            node.update({"bound": -math.inf, "discarded": "primal_infeasible", "vs_eff": None})
        else:
            if vs is None:  # failed solve: the node keeps its parent's bound (valid, uninformative)
                node["failed"] = True
            else:
                slack = None if not _finite(dg.get("value_dual")) else vs - dg["value_dual"]
                node["safe_slack"] = slack
                if gmode == "node" and needs_retol(slack, vs - LB):
                    rt = sdp_retol(inst, nctrl, threads, time_limit, cut)
                    n_retol += 1
                    node["retol"] = rt
                    if rt["value_safe"] is not None and rt["value_safe"] < vs:
                        vs = rt["value_safe"]
            node["vs_eff"] = vs
            node["bound"] = min(node["parent_bound"], vs) if vs is not None else node["parent_bound"]
            if r.recovered_point is not None:
                v0 = float(r.recovered_value)
                v, Z, info = ascend(inst, P, r.recovered_point, ascent_maxiter)
                node.update({"ubar_value": v0, "ubar_ascent_value": v, "ubar_ascent_nit": info["nit"]})
                if leaf_points is not None:
                    leaf_points.append(r.recovered_point)
                if v > LB:
                    LB, LB_Z = v, Z
        node["t_s"] = time.perf_counter() - ts
        trace.append({"k": n_solves, "UB": leaf_UB(), "LB": LB, "node": node["id"]})

    def f0v_node(c_: dict) -> tuple[float, dict]:
        from aimp.bounds import t_intervals
        from aimp.theory import f0v_final

        fi = f0v_inputs
        t_lo, t_hi = t_intervals(c_["s_lo"], c_["s_hi"], "tight")
        hm = np.asarray(fi["h_max"], float)
        a_hi_raw = np.asarray(c_["a_hi"], float)
        slack = float(sum(a_hi_raw[k] * (hm[k] - hm.min()) for k in range(n) if a_hi_raw[k] <= 1e-6))
        ff = f0v_final(t_lo, t_hi, hm, c_["a_lo"], c_["a_hi"], fi["nu_c"], phi_min_cert=fi["phi_min_cert"],
                       phi_min_strong=None, eta=fi["eta"], rad_c=fi["rad_c"], pruning_slack=slack)
        return float(ff["bound_F0v_final"]), ff

    def lookahead(nctrl: dict) -> dict:
        F_node, ffn = f0v_node(nctrl)
        Dj, chi = ffn["D_j"], ffn["chi_j"]
        cand = []
        for jj in range(n):
            if not Dj[jj] > 0:
                continue
            (l1, h1), (l2, h2) = split_box(nctrl["s_lo"], nctrl["s_hi"], jj)
            Fm = f0v_node(child_controls(nctrl, l1, h1, n))[0]
            Fp = f0v_node(child_controls(nctrl, l2, h2, n))[0]
            cand.append((max(Fm, Fp), -float(Dj[jj] * chi[jj]), jj, Fm, Fp))
        if not cand:
            return {"key": None, "F_node": F_node}
        best = min(cand)
        return {"key": best[2], "F_node": F_node, "F_minus": best[3], "F_plus": best[4], "F_child_max": best[0],
                "predicted_gain_lookahead": (F_node / best[0]) if best[0] > 0 else None, "n_candidates": len(cand),
                "D_chi": -best[1]}

    def choose(nctrl: dict) -> tuple[int, dict]:
        Sj = split_scores(nctrl)
        Pj = value_scores(Sj, h_max, phi_rule) if phi_rule is not None else None
        Sn = float(np.sum(Sj))
        Pn = float(np.sum(Pj)) if Pj is not None else None
        jS = int(np.argmax(Sj)) if np.any(Sj > 0) else None
        jP = int(np.argmax(Pj)) if (Pj is not None and np.any(Pj > 0)) else None
        la = lookahead(nctrl) if rule == "lookahead" else None
        if jS is None and not (la and la["key"] is not None):  # no secant slack left: split the widest score interval
            j, how = int(np.argmax(np.asarray(nctrl["s_hi"], float) - np.asarray(nctrl["s_lo"], float))), "widest"
        elif rule == "lookahead" and la["key"] is not None:
            j, how = la["key"], "lookahead"
        elif rule == "value" and jP is not None:
            j, how = jP, "value"
        else:
            j, how = jS, "S" if rule == "S" else "S_fallback"
        jalt = (jP if rule == "S" else jS)

        def g(jj):
            if jj is None:
                return None, None
            gs = Sn / (Sn - 0.75 * float(Sj[jj])) if Sn > 0 else None
            gv = (Pn / (Pn - 0.75 * float(Pj[jj]))) if (Pn is not None and Pn > 0) else None
            return gs, gv

        gs, gv = g(j)
        ags, agv = g(jalt)
        info = {"key": j, "how": how, "S": Sn, "S_j": float(Sj[j]), "Pi_v": Pn,
                "P_j": None if Pj is None else float(Pj[j]), "predicted_gain_S": gs, "predicted_gain_v": gv,
                "predicted_gain": gv if (rule == "value" and how == "value") else gs,
                "alt_key": jalt, "alt_predicted_gain_S": ags, "alt_predicted_gain_v": agv,
                "rules_agree": bool(jalt is None or jalt == j), "key_S": jS, "key_value": jP}
        if la is not None:
            info.update({k: v for k, v in la.items() if k != "key"})
            info["key_lookahead"] = la["key"]
            if how == "lookahead":
                info["predicted_gain"] = la["predicted_gain_lookahead"]
        return j, info

    root = {"id": 0, "parent": None, "depth": 0, "s_lo": np.asarray(ctrl["s_lo"], float).tolist(),
            "s_hi": np.asarray(ctrl["s_hi"], float).tolist(), "parent_bound": math.inf, "bound": math.inf,
            "leaf": True, "solved": False, "S": sandwich_S(ctrl), "w_t": float(ctrl["w_t"])}
    nodes.append(root)
    ctrls = {0: ctrl}
    solve(root, ctrl)
    splits: list[dict] = []
    stop = None
    while True:
        UB = leaf_UB()
        sp = spread()
        if sp is not None and UB - LB <= close_rel * sp:
            stop = "closed"
            break
        if n_solves >= K:
            stop = "K"
            break
        opn = [nd for nd in nodes if nd["leaf"] and nd["solved"] and nd["bound"] > LB]
        if not opn:
            stop = "all_pruned"
            break
        node = max(opn, key=lambda nd: nd["bound"])
        nctrl = ctrls[node["id"]]
        j, sinfo = choose(nctrl)
        (lo1, hi1), (lo2, hi2) = split_box(nctrl["s_lo"], nctrl["s_hi"], j)
        node["leaf"] = False
        kids = []
        for lo_, hi_ in ((lo1, hi1), (lo2, hi2)):
            cid = len(nodes)
            cc = child_controls(nctrl, lo_, hi_, n)
            ctrls[cid] = cc
            kid = {"id": cid, "parent": node["id"], "depth": node["depth"] + 1, "s_lo": lo_.tolist(), "s_hi": hi_.tolist(),
                   "split_key": j, "parent_bound": node["bound"], "bound": node["bound"], "leaf": True, "solved": False,
                   "S": sandwich_S(cc), "w_t": float(cc["w_t"])}
            nodes.append(kid)
            kids.append(kid)
        splits.append({"parent": node["id"], **sinfo, "children": [k["id"] for k in kids], "at_solve": n_solves})
        for kid in kids:
            if n_solves >= K:
                break
            solve(kid, ctrls[kid["id"]])
            # prune: leaves whose bound <= LB are closed (kept for UB; never split)
    UB_search = leaf_UB()
    for nd in nodes:
        nd["bound_search"] = nd["bound"]
    for s in splits:  # observed ratio of the parent bound gap to the max child bound gap, against the final LB
        par = nodes[s["parent"]]
        ch = [nodes[c] for c in s["children"]]
        if all(c["solved"] for c in ch):
            pg = par["bound"] - LB
            cg = max(c["bound"] for c in ch) - LB
            s["parent_gap"], s["max_child_gap"] = pg, cg
            s["observed_gain"] = pg / cg if cg > 0 else None
            s["children_closed"] = bool(cg <= 0)
        else:
            s["observed_gain"] = None
            s["incomplete"] = True
    # ---- the re-solve rule at termination (root + the open leaves near UB)
    guard_log: list[dict] = []
    root_bound_search = root["bound"]
    if gmode == "termination":
        def recompute_bounds() -> None:  # parents have smaller ids than their children
            for nd in nodes:
                if nd.get("discarded"):
                    continue
                pb = math.inf if nd["parent"] is None else nodes[nd["parent"]]["bound"]
                nd["parent_bound"] = pb
                v = nd.get("vs_eff") if nd["solved"] else None
                nd["bound"] = min(pb, v) if v is not None else pb

        def source_of(nd: dict) -> dict | None:
            x = nd
            while x is not None:
                if x["solved"] and x.get("vs_eff") is not None and x["vs_eff"] <= nd["bound"]:
                    return x
                x = None if x["parent"] is None else nodes[x["parent"]]
            return None

        evaluated: set[int] = set()

        def guard_node(src: dict, gap: float, why: str) -> None:
            nonlocal n_retol
            evaluated.add(src["id"])
            fire = needs_retol(src.get("safe_slack"), gap)
            ent = {"node": src["id"], "why": why, "safe_slack": src.get("safe_slack"), "gap": gap, "fired": bool(fire),
                   "value_safe_before": src.get("vs_eff")}
            if fire:
                rt = sdp_retol(inst, ctrls[src["id"]], threads, time_limit, cut)
                n_retol += 1
                src["retol"] = rt
                ent.update({"value_safe_1e9": rt["value_safe"], "wall_s": rt["wall_s"], "solsta": rt["solsta"]})
                if rt["value_safe"] is not None and rt["value_safe"] < src["vs_eff"]:
                    src["vs_eff"] = rt["value_safe"]
                    recompute_bounds()
            guard_log.append(ent)

        if root.get("vs_eff") is not None:
            guard_node(root, root["vs_eff"] - LB, "root")
        passes = 0
        while True:
            UB = leaf_UB()
            win = GUARD_REL * max(UB - LB, 0.0)
            cand = []
            for nd in nodes:
                if not nd["leaf"] or nd.get("discarded") or nd["bound"] <= LB or nd["bound"] < UB - win:
                    continue
                src = source_of(nd)
                if src is not None and src["id"] not in evaluated and all(src is not c for c, _ in cand):
                    cand.append((src, nd))
            if not cand:
                break
            passes += 1
            for src, nd in cand:
                guard_node(src, nd["bound"] - LB, f"leaf {nd['id']}")
        guard_passes = passes
    else:
        guard_passes = None
    UB = leaf_UB()
    UB_at, LB_at = {}, {}
    for k in BB_RECORD_AT:
        rows = [t for t in trace if t["k"] <= k]
        UB_at[k], LB_at[k] = rows[-1]["UB"], rows[-1]["LB"]
    UB_at_g = {}
    for k in BB_RECORD_AT:
        if k >= n_solves:
            UB_at_g[k] = UB
        elif k == 1:
            UB_at_g[k] = min(UB_at[k], root["vs_eff"]) if root.get("vs_eff") is not None else UB_at[k]
        else:
            UB_at_g[k] = UB_at[k]
    sp = spread()
    out = {"UB": UB, "LB": LB, "stop": stop, "n_solves": n_solves, "n_retol": n_retol, "trace": trace,
           "UB_at": UB_at, "LB_at": LB_at, "UB_at_guarded": UB_at_g, "UB_search": UB_search,
           "root_bound_search": root_bound_search, "root_bound_guarded": root["bound"],
           "nodes": nodes, "splits": splits, "spread_final": sp, "rule": rule, "guard": gmode,
           "guard_log": guard_log, "guard_passes": guard_passes,
           "gap_rel_final": None if not sp else (UB - LB) / sp, "close_rel": close_rel, "K": K,
           "cert_bb": UB if r_greedy is None else min(float(r_greedy), UB), "r_greedy": r_greedy,
           "valid": bool(UB >= LB - 1e-7 * max(1.0, abs(LB))), "wall_s": time.perf_counter() - t0,
           "LB_Z": LB_Z, "ub_monotone": all(trace[k + 1]["UB"] <= trace[k]["UB"] + 1e-12 * max(1.0, abs(trace[k]["UB"]))
                                            for k in range(len(trace) - 1)) and UB <= UB_search + 1e-12 * max(1.0, abs(UB_search))}
    return out
