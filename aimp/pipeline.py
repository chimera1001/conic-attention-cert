"""The per-unit procedures of the GPT-2 experiments (Section 7 and Appendix B of the paper).

* :func:`evaluate_unit` — one unit with a varying query: Shor score intervals, controls, the first-order certificate
  ``r_pp`` (greedy pattern-polytope bound), the level-one relaxation in MOSEK's task API with its safe dual bound, the
  certificate ``min(r_pp, safe value)``, the re-solve rule of Appendix B, the best feasible value from multistart
  ascent (``lb_strong``) and the smallest output found (``phi_min_strong``), and the per-key quantities of the
  second-order bound (the value-weighted slack and the terms of bound (39)).
* :func:`evaluate_fq_exact` — one unit with a fixed query: the exact worst case of Theorem 7 (certified brackets of
  the maximum and of the minimum), the first-order certificate, and the maximizing / minimizing angle of every varying
  key in the canonical frame of :func:`canonical_frame` (the benchmark of ``benchmark/``).
* :func:`evaluate_fq_sdp` — the certificate on a fixed-query unit, measured against the exact worst case.

Every procedure takes the unit's specification ``spec`` (``"a"``, the paper's, by default; ``"lm"``, the logit-margin
specification of :func:`aimp.gpt2_units.make_instance`) and records the Vertex-Softmax bound ``r_VS`` on the unit's
score intervals (:func:`aimp.bounds.vertex_softmax`). Every reported semidefinite value is the safe dual bound of
Proposition 16 (``value_safe``).
"""
from __future__ import annotations

import math
import time

import numpy as np

from aimp import gpt2_units as U
from aimp import option_b as OB
from aimp.io import env_info, git_commit, peak_rss_gb

#: interior-point tolerances on GPT-2 (Appendix B): primal / dual feasibility 1e-6, relative gap 1e-8; the safe dual
#: bound is valid from any dual point, so tighter tolerances only add iterations.
from aimp.sdp_mosek import PHASE3_TOL as SDP_TOL  # noqa: E402

VALID_TOL = 1e-7  # cert >= lb - 1e-7 max(1, |lb|)
#: a certified dominant key (the margin regime): the margin bound is at most 1% of the output's range of motion
MARGIN_REL = 0.01
MS_STARTS = 2  # the quick multistart (centre, greedy point and two random starts) whose value floors lb_strong


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _valid(v, lb) -> bool | None:
    if not (_finite(v) and _finite(lb)):
        return None
    return bool(v >= lb - VALID_TOL * max(1.0, abs(lb)))


def _option_b(inst, ctrl, rec: dict, sres, retol, lb_best: float, flagged: bool, threads) -> dict:
    """gpt-oss: the residual-allowed covariance of the final solve's point, the trigger and the rescaled re-solve
    (:func:`aimp.option_b.apply_rescaled`)."""
    if retol is not None and retol.get("value_safe") is not None and retol["value_safe"] == rec.get("value_safe"):
        C0 = retol.get("C_res")
    elif sres is not None and getattr(sres, "X", None) is not None:
        C0 = OB.residual_cov(inst, ctrl, sres.X, sres.pos_a, sres.upos)
    else:
        C0 = None
    return OB.apply_rescaled(inst, ctrl, rec, C0, lb_best, flagged, threads)


def _classify_checks(inst, vs, bounds_: dict, lb, ub, ob: dict, rec: dict, retol) -> dict:
    """The theorem checks with the floor ``5e-8 max|h| + slack`` and the residual-allowed covariance of the final solve."""
    if ob.get("final_solve") == "rescaled":
        slack = ob["rescaled"].get("safe_slack") or 0.0
    elif retol is not None and retol.get("value_safe") is not None and retol["value_safe"] == rec.get("value_safe"):
        slack = retol.get("safe_slack") or 0.0
    else:
        slack = rec.get("safe_slack") or 0.0
    floor = OB.FLOOR_REL * OB.h_scale(inst) + max(slack, 0.0)
    return {k: OB.classify(vs, b_, lb, ub, floor, ob.get("C_res_final") or 0.0) for k, b_ in bounds_.items() if b_ is not None}


def controls(inst, threads: int | None = None) -> tuple[dict, dict]:
    """Shor score intervals in the head's role-reduced coordinates and every control derived from them."""
    from aimp import bounds, gpt2

    si = gpt2.score_intervals_roles(inst, threads=threads)
    return si, bounds.controls(inst, intervals=si, eps_target=0.0, threads=threads)


def quick_multistart(inst, ctrl, seed: int) -> tuple[float, float]:
    """A cheap feasible value on ``+c`` and on ``-c`` (centre, greedy point, two random starts, 300 iterations)."""
    from aimp.lower import neg_instance
    from aimp.lower_bounds import multistart_lb
    from aimp.relax_ps import solve_ps_greedy

    g = solve_ps_greedy(inst, ctrl, "auto")
    up = multistart_lb(inst, MS_STARTS, seed=seed, extra_starts=[g.recovered_point], maxiter=300)
    neg = neg_instance(inst)
    gn = solve_ps_greedy(neg, ctrl, "auto")
    dn = multistart_lb(neg, MS_STARTS, seed=seed + 7, extra_starts=[gn.recovered_point], maxiter=300)
    return float(up.value), -float(dn.value)


def collapse_fields(inst, ctrl, *, phi_min_strong: float, eta: float, pruning_slack: float) -> dict:
    """The quantities of Figure 3 and of bound (39): the weighted secant slack ``S``, the value-weighted slack
    ``Pi_v = sum_j S_j (h_j^max - phi_min)_+``, the dispersion of the nominal pattern, and the terms of the second-order
    bound with the exact secant gaps, the slack budget and the certified lower bound on the minimum."""
    from aimp.lower import neg_instance
    from aimp.relax_ps import solve_ps_greedy
    from aimp.theory import directional_constants, f0v_final_fields, perkey_fields

    phi_min_cert = -float(solve_ps_greedy(neg_instance(inst), ctrl, "auto").value)
    dc = directional_constants(inst)
    pk = perkey_fields(inst, ctrl, phi_min_strong=phi_min_strong, phi_min_cert=phi_min_cert, eta=eta,
                       pruning_slack=pruning_slack, dc=dc)
    ff = f0v_final_fields(inst, ctrl, phi_min_cert=phi_min_cert, phi_min_strong=phi_min_strong, eta=eta,
                          pruning_slack=pruning_slack, dc=dc)
    keep = ("bound_F0v_final", "F0vf_term_knapsack", "F0vf_term_eta", "F0vf_term_cov", "F0vf_pruning_slack", "D_tot",
            "D_sum", "budget_binds", "Pi_v_star")
    return {"S": pk["S"], "Pi_v": pk["Pi_v"], "bound_F0v": pk["bound_F0v"], "phi_min_cert": phi_min_cert,
            "rad_c": dc["rad_c"], "perkey": pk["perkey"], **{k: ff[k] for k in keep if k in ff},
            "n_pruned": int(np.sum(np.clip(np.asarray(ctrl["a_hi"], float), 0, 1) <= 1e-6))}


def evaluate_unit(l: int, h: int, p: int, pos: int, rho: float, threads: int | None = None, spec: str = "a",
                  model: str = "gpt2") -> dict:
    """The certificate of one varying-query unit with the procedure of Appendix B (``model``: as
    :func:`aimp.gpt2_units.make_instance`)."""
    from aimp.certify import certify, needs_retol, sdp_retol, theorem_check
    from aimp.gpt2 import nominal_value_range
    from aimp.lower import sdp_starts, strong_bracket

    T: dict = {}
    t0 = time.perf_counter()
    inst = U.make_instance(l, h, p, pos, rho, spec=spec, model=model)
    si, ctrl = controls(inst, threads)
    T["controls_s"] = time.perf_counter() - t0
    rng_ = nominal_value_range(inst)
    seed = U.unit_seed(l, h, p, pos, rho)
    rec = certify(inst, ctrl, eps_target=0.0, threads=threads, sdp_tol=SDP_TOL, return_sdp=True, cut="auto")
    _, sres = rec.pop("_greedy"), rec.pop("_sdp")
    t3 = time.perf_counter()
    lb_ms, pm_ms = quick_multistart(inst, ctrl, seed)
    lb_floor = max(v for v in (lb_ms, rec["lb_best"]) if _finite(v))
    T["multistart_s"] = time.perf_counter() - t3
    t4 = time.perf_counter()  # multistart ascent with the starts of Appendix B, including the relaxation's
    starts = sdp_starts(inst, sres.X, sres.upos) if (sres is not None and getattr(sres, "X", None) is not None) else {}
    b = strong_bracket(inst, seed + 31, starts=starts, lb_floor=lb_floor, phi_min_floor=pm_ms)
    T["lb_strong_s"] = time.perf_counter() - t4
    lbs, pms = b["lb_strong"], b["phi_min_strong"]
    vs0 = rec.get("value_safe")
    retol = None  # Appendix B: re-solve at 1e-9 when value_safe - dual objective > 1% of (cert - best feasible value)
    if vs0 is not None and needs_retol(rec.get("safe_slack"), rec["cert"] - lbs):
        retol = sdp_retol(inst, ctrl, threads, cut="auto") if model == "gpt2" else OB.resolve(inst, ctrl, threads, False)
        if retol["value_safe"] is not None and retol["value_safe"] < vs0:
            rec["value_safe"] = retol["value_safe"]
            rec["cert"] = min(rec["r_greedy"], rec["value_safe"])
    vs = rec.get("value_safe")
    cf = collapse_fields(inst, ctrl, phi_min_strong=pms, eta=rec.get("F0_eta") or 0.0,
                         pruning_slack=rec.get("pruning_slack") or 0.0)
    ob = None
    if model != "gpt2":  # the rescaled re-solve (aimp/option_b.py)
        pre = [theorem_check(vs, b_, lbs, rec["cert"]) for b_ in (cf["bound_F0v_final"], rec.get("bound_F0t"),
                                                                  rec.get("bound_F0s") if rec.get("F0s_applies") else None)]
        ob = _option_b(inst, ctrl, rec, sres, retol, lbs, any(c["status"] not in (None, "satisfied") for c in pre), threads)
        vs = rec.get("value_safe")
    checks = {"F0v_final": theorem_check(vs, cf["bound_F0v_final"], lbs, rec["cert"]),
              "F0t": theorem_check(vs, rec.get("bound_F0t"), lbs, rec["cert"])}
    if rec.get("F0s_applies"):
        checks["F0s"] = theorem_check(vs, rec.get("bound_F0s"), lbs, rec["cert"])
    if ob is not None:
        rec["option_b"] = ob
        rec["checks_scaled"] = _classify_checks(inst, vs, {k: b_ for k, b_ in (("F0v_final", cf["bound_F0v_final"]),
                                                                              ("F0t", rec.get("bound_F0t")),
                                                                              ("F0s", rec.get("bound_F0s")
                                                                               if rec.get("F0s_applies") else None))},
                                                lbs, rec["cert"], ob, rec, retol)
    spread = lbs - pms
    gap = rec["cert"] - lbs
    g_ = ctrl["gamma"]  # margin bound with the directional constant: 4 (n - 1) e^{-gamma} rad_c (gamma > 0)
    bound_C_c = 4 * (inst.n - 1) * math.exp(-g_) * cf["rad_c"] if (_finite(g_) and g_ > 0) else None
    rec.update({
        "layer": l, "head": h, "prompt": p, "pos": pos, "rho_rel": rho, "spec": spec, "model": model, "range": rng_,
        "s_lo": ctrl["s_lo"], "s_hi": ctrl["s_hi"], "a_lo": ctrl["a_lo"], "a_hi": ctrl["a_hi"],
        "h_max": ctrl["h_max"], "r_VS": ctrl["r_VS"], "VS_k": ctrl["VS_k"], "c_norm": ctrl["c_norm"],
        "phi_nominal": rec["phi_nominal"], "lb_ms": lb_ms, "phi_min_ms": pm_ms, "lb_floor": lb_floor, "lb_strong": lbs, "phi_min_strong": pms,
        "lb_winner": b["lb_winner"], "phi_min_winner": b["phi_min_winner"], "lb_best_by_type": b["lb_best_by_type"],
        "phi_min_best_by_type": b["phi_min_best_by_type"], "lb_info": b["lb_info"], "sdp_starts": bool(starts),
        "retol": retol is not None, "retol_info": retol, "value_safe_first": vs0,
        **{k: v for k, v in cf.items() if k != "S"}, "perkey": cf["perkey"],
        "checks": {k: v["status"] for k, v in checks.items()}, "check_ratios": {k: v.get("ratio_lb") for k, v in checks.items()},
        "gap": gap, "greedy_gap": rec["r_greedy"] - lbs, "spread": spread, "rel_gap": gap / spread if spread > 0 else None,
        "rel_gap_greedy": (rec["r_greedy"] - lbs) / spread if spread > 0 else None,
        "gap_over_range": gap / rng_ if rng_ > 0 else None, "numerical_floor": bool(gap <= 1e-7 * rng_),
        "positive_score_margin": bool(_finite(ctrl["gamma"]) and ctrl["gamma"] > 0),
        "bound_C_c": bound_C_c,
        "margin_regime": bool(bound_C_c is not None and spread > 0 and bound_C_c <= MARGIN_REL * spread),
        "valid": _valid(rec["cert"], lbs), "valid_value_safe": _valid(vs, lbs), "valid_greedy": _valid(rec["r_greedy"], lbs),
        "valid_VS": _valid(ctrl["r_VS"], lbs),
        "threads": threads, "env": env_info(), "git": git_commit(), "peak_rss_gb": peak_rss_gb()})
    rec["times"].update({**T, "unit_total_s": time.perf_counter() - t0})
    return rec


# ------------------------------------------------------------------------------------------------ fixed query
def canonical_frame(l: int, h: int, p: int, pos: int, c: np.ndarray | None = None) -> dict:
    """The plane in which every varying key of a fixed-query unit moves, in the 768-dimensional post-normalization
    space: the score of key ``j`` changes by ``alpha . delta_j`` with ``alpha = W_K q_pos / sqrt(d_k)`` (``q_pos`` the
    fixed query) and its value along ``c`` by ``beta . delta_j`` with ``beta = W_V W_O c``. ``e1 = alpha / ||alpha||``,
    ``e2`` = the normalized component of ``beta`` orthogonal to ``e1``. A key at canonical angle ``theta`` is the token
    ``zbar_j + rho_j (cos theta e1 + sin theta e2)``. ``c``: the specification (default: the direction of the nominal
    head output)."""
    hw, _ = U.head_data(l, h)
    P = U.ctx()["P"][p]
    z = P["z"][l]
    q = z[pos] @ hw["W_Q"] + hw["b_Q"]
    if c is None:
        ybar = P["merged"][l][pos, h * 64:(h + 1) * 64] @ hw["W_O"]
        c = ybar / np.linalg.norm(ybar)
    alpha = hw["W_K"] @ q / math.sqrt(hw["W_K"].shape[1])
    beta = hw["W_V"] @ (hw["W_O"] @ c)
    e1 = alpha / np.linalg.norm(alpha)
    b2 = beta - (beta @ e1) * e1
    nb = float(np.linalg.norm(b2))
    e2 = b2 / nb if nb > 0 else np.zeros_like(b2)
    return {"e1": e1, "e2": e2, "alpha": alpha, "beta": beta, "c": c, "z": z, "beta_perp_norm": nb}


def angles_of(frame: dict, deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Canonical angles and norms of perturbations ``deltas`` (rows, 768-dimensional)."""
    a = deltas @ frame["e1"]
    b = deltas @ frame["e2"]
    return np.arctan2(b, a), np.linalg.norm(deltas, axis=1)


def evaluate_fq_exact(l: int, h: int, p: int, pos: int, rho: float, threads: int = 1, spec: str = "a",
                      model: str = "gpt2") -> dict:
    """The exact worst case of one fixed-query unit (Theorem 7; certified subdivision of each key's boundary angle in
    place of the closed form of part (iii)), the first-order certificate and the canonical angles of the maximizing
    and minimizing inputs (``model``: as :func:`aimp.gpt2_units.make_instance`; gpt-oss: one plane per key,
    :func:`aimp.gptoss.canonical_angles`)."""
    from aimp.exact_fq import exact_fq, fq_problem
    from aimp.gpt2 import nominal_value_range
    from aimp.heads import nominal
    from aimp.lower import neg_instance
    from aimp.relax_ps import solve_ps_greedy

    t0 = time.perf_counter()
    fq = U.make_instance(l, h, p, pos, rho, fixed_query=True, spec=spec, model=model)
    si, ctrl = controls(fq, threads)
    t1 = time.perf_counter()
    g = solve_ps_greedy(fq, ctrl, "auto")
    gn = solve_ps_greedy(neg_instance(fq), ctrl, "auto")
    t2 = time.perf_counter()
    ex = exact_fq(fq)
    t3 = time.perf_counter()
    rng_ = nominal_value_range(fq)
    p_lo, p_hi, m_lo, m_hi = ex["p_fq_lo"], ex["p_fq_hi"], ex["min_fq_lo"], ex["min_fq_hi"]
    exact = ex["status"] == "exact"
    p_mid, m_mid = (0.5 * (p_lo + p_hi), 0.5 * (m_lo + m_hi)) if exact else (None, None)
    spread = (p_mid - m_mid) if exact else None
    # canonical angles: the worst-case inputs in reduced coordinates -> perturbations of the normalized tokens
    Pq = fq_problem(fq)
    var_keys = [fq.keys[idx] for idx in Pq.var]
    ang, fr = {}, {"beta_perp_norm": None}
    if model != "gpt2":  # gpt-oss: the plane of key j depends on j through its rotary map
        from aimp import gptoss

        for tag, Zp in (("max", ex["Z_max"]), ("min", ex["Z_min"])):
            a = gptoss.canonical_angles(fq, var_keys, Zp)
            ang[tag] = {k: np.asarray(v, float) for k, v in a.items()}
    else:
        _, pr = U.head_data(l, h)
        fr = canonical_frame(l, h, p, pos, c=fq.c)
        for tag, Zp in (("max", ex["Z_max"]), ("min", ex["Z_min"])):
            D = np.zeros((len(var_keys), pr.Vt.shape[1]))
            for r, j in enumerate(var_keys):
                w, *_ = np.linalg.lstsq(pr.E, Zp[j] - fq.sets[j].center, rcond=None)
                D[r] = pr.Vt.T @ w
            th, nr = angles_of(fr, D)
            inplane = np.linalg.norm(D - np.outer(D @ fr["e1"], fr["e1"]) - np.outer(D @ fr["e2"], fr["e2"]), axis=1)
            ang[tag] = {"theta": th, "norm": nr, "off_plane": inplane}
    rho_j = np.array([fq.sets[j].rho for j in var_keys])
    nomv = nominal(fq)
    return {"layer": l, "head": h, "prompt": p, "pos": pos, "rho_rel": rho, "n": fq.n, "spec": spec, "model": model,
            "range": rng_, "phi_nominal": nomv["phi"], "h_max": ctrl["h_max"], "r_VS": ctrl["r_VS"], "VS_k": ctrl["VS_k"],
            "valid_VS": _valid(ctrl["r_VS"], p_lo), "c_norm": ctrl["c_norm"], "p_fq_lo": p_lo, "p_fq_hi": p_hi, "min_fq_lo": m_lo, "min_fq_hi": m_hi,
            "p_fq": p_mid, "min_fq": m_mid, "spread_fq": spread, "exact_status": ex["status"],
            "status_max": ex["status_max"], "status_min": ex["status_min"],
            "r_greedy": g.value, "greedy_cut": g.diagnostics["cut_used"], "greedy_recovered": g.recovered_value,
            "phi_min_cert": -float(gn.value),
            "greedy_gap": None if not exact else g.value - p_mid,
            "rel_gap_greedy": (g.value - p_mid) / spread if (exact and spread > 0) else None,
            "value_max_input": ex["value_max_input"], "value_min_input": ex["value_min_input"],
            "max_input_in_bracket": ex["max_input_in_bracket"], "min_input_in_bracket": ex["min_input_in_bracket"],
            "varying_keys": var_keys, "rho_j": rho_j,
            "theta_max": ang["max"]["theta"], "theta_min": ang["min"]["theta"],
            "norm_max_rel": ang["max"]["norm"] / np.maximum(rho_j, 1e-300),
            "norm_min_rel": ang["min"]["norm"] / np.maximum(rho_j, 1e-300),
            "off_plane_max": float(np.max(ang["max"]["off_plane"], initial=0.0)),
            "off_plane_min": float(np.max(ang["min"]["off_plane"], initial=0.0)),
            "beta_perp_norm": fr["beta_perp_norm"],
            "evals_max": ex["evals_max"], "evals_min": ex["evals_min"], "width_rel_max": ex["width_rel_max"],
            "width_rel_min": ex["width_rel_min"], "lo_cert_max": ex["lo_cert_max"], "hi_cert_max": ex["hi_cert_max"],
            "lo_cert_min": ex["lo_cert_min"], "hi_cert_min": ex["hi_cert_min"], "n_var": ex["n_var"],
            "s_lo": ctrl["s_lo"], "s_hi": ctrl["s_hi"], "w_t": ctrl["w_t"], "Delta": ctrl["Delta"], "gamma": ctrl["gamma"],
            "valid_greedy": _valid(g.value, p_lo),
            "times": {"controls_s": t1 - t0, "greedy_s": t2 - t1, "exact_s": t3 - t2, "total_s": time.perf_counter() - t0},
            "threads": threads, "git": git_commit(), "env": env_info(), "peak_rss_gb": peak_rss_gb()}


def evaluate_fq_sdp(l: int, h: int, p: int, pos: int, rho: float, xr: dict, threads: int | None = None,
                    spec: str = "a", model: str = "gpt2") -> dict:
    """The certificate on a fixed-query unit (the same procedure as :func:`evaluate_unit`), measured against the
    exact worst case ``xr`` of :func:`evaluate_fq_exact` (``model``: as :func:`aimp.gpt2_units.make_instance`)."""
    from aimp.certify import certify, needs_retol, sdp_retol, theorem_check
    from aimp.gpt2 import nominal_value_range
    from aimp.lower import sdp_starts, strong_bracket

    T: dict = {}
    t0 = time.perf_counter()
    fq = U.make_instance(l, h, p, pos, rho, fixed_query=True, spec=spec, model=model)
    si, ctrl = controls(fq, threads)
    T["controls_s"] = time.perf_counter() - t0
    rng_ = nominal_value_range(fq)
    xexact = xr.get("exact_status") == "exact"
    rec = certify(fq, ctrl, eps_target=0.0, threads=threads, sdp_tol=SDP_TOL, return_sdp=True, cut="auto")
    _, sres = rec.pop("_greedy"), rec.pop("_sdp")
    t4 = time.perf_counter()
    starts = sdp_starts(fq, sres.X, sres.upos) if (sres is not None and getattr(sres, "X", None) is not None) else {}
    b = strong_bracket(fq, U.unit_seed(l, h, p, pos, rho) + 41, starts=starts)
    T["lb_strong_s"] = time.perf_counter() - t4
    lbs, pms = b["lb_strong"], b["phi_min_strong"]
    lb_best = max(lbs, xr["p_fq_lo"])
    vs0 = rec.get("value_safe")
    retol = None
    if vs0 is not None and needs_retol(rec.get("safe_slack"), vs0 - lb_best):
        retol = sdp_retol(fq, ctrl, threads, cut="auto") if model == "gpt2" else OB.resolve(fq, ctrl, threads, False)
        if retol["value_safe"] is not None and retol["value_safe"] < vs0:
            rec["value_safe"] = retol["value_safe"]
    vs = rec.get("value_safe")
    rec["cert"] = min(rec["r_greedy"], vs) if vs is not None else rec["r_greedy"]
    cf = collapse_fields(fq, ctrl, phi_min_strong=(xr["min_fq"] if xexact else pms), eta=rec.get("F0_eta") or 0.0,
                         pruning_slack=rec.get("pruning_slack") or 0.0)
    lb_c, ub_c = xr["p_fq_lo"], xr["p_fq_hi"]
    ob = None
    if model != "gpt2":  # the rescaled re-solve (aimp/option_b.py)
        pre = [theorem_check(vs, b_, lb_c, ub_c) for b_ in (cf["bound_F0v_final"], rec.get("bound_F0t"),
                                                            rec.get("bound_F0s") if rec.get("F0s_applies") else None)]
        ob = _option_b(fq, ctrl, rec, sres, retol, lb_best, any(c["status"] not in (None, "satisfied") for c in pre), threads)
        vs = rec.get("value_safe")
    checks = {"F0v_final": theorem_check(vs, cf["bound_F0v_final"], lb_c, ub_c),
              "F0t": theorem_check(vs, rec.get("bound_F0t"), lb_c, ub_c)}
    if model != "gpt2" and rec.get("F0s_applies"):
        checks["F0s"] = theorem_check(vs, rec.get("bound_F0s"), lb_c, ub_c)
    if ob is not None:
        rec["option_b"] = ob
        rec["checks_scaled"] = _classify_checks(fq, vs, {k: b_ for k, b_ in (("F0v_final", cf["bound_F0v_final"]),
                                                                            ("F0t", rec.get("bound_F0t")),
                                                                            ("F0s", rec.get("bound_F0s")
                                                                             if rec.get("F0s_applies") else None))},
                                                lb_c, ub_c, ob, rec, retol)
    p_fq = xr["p_fq"] if xexact else None
    spread_fq = xr["spread_fq"] if xexact else None
    rec.update({
        "layer": l, "head": h, "prompt": p, "pos": pos, "rho_rel": rho, "spec": spec, "model": model, "range": rng_,
        "fixed_query": True,
        "s_lo": ctrl["s_lo"], "s_hi": ctrl["s_hi"], "a_lo": ctrl["a_lo"], "a_hi": ctrl["a_hi"],
        "h_max": ctrl["h_max"], "r_VS": ctrl["r_VS"], "VS_k": ctrl["VS_k"], "c_norm": ctrl["c_norm"],
        "phi_nominal": rec["phi_nominal"], "lb_strong": lbs, "phi_min_strong": pms, "lb_winner": b["lb_winner"], "lb_best_by_type": b["lb_best_by_type"],
        "sdp_starts": bool(starts), "retol": retol is not None, "retol_info": retol, "value_safe_first": vs0,
        "exact_status": xr.get("exact_status"), "p_fq_lo": xr["p_fq_lo"], "p_fq_hi": xr["p_fq_hi"], "p_fq": p_fq,
        "min_fq": xr.get("min_fq"), "spread_fq": spread_fq,
        **{k: v for k, v in cf.items() if k not in ("S", "perkey")},
        "checks": {k: v["status"] for k, v in checks.items()}, "check_ratios": {k: v.get("ratio_lb") for k, v in checks.items()},
        "gap_exact": None if p_fq is None else rec["cert"] - p_fq,
        "rel_gap_exact": None if not spread_fq else (rec["cert"] - p_fq) / spread_fq,
        "rel_gap_greedy_exact": None if not spread_fq else (rec["r_greedy"] - p_fq) / spread_fq,
        "gap_over_bound": (None if (p_fq is None or not cf["bound_F0v_final"]) else
                           (rec["cert"] - p_fq) / cf["bound_F0v_final"]),
        "lb_strong_gap_exact": None if not xexact else p_fq - lbs,
        "valid": _valid(rec["cert"], lb_c), "valid_value_safe": _valid(vs, lb_c), "valid_greedy": _valid(rec["r_greedy"], lb_c),
        "valid_VS": _valid(ctrl["r_VS"], lb_c),
        "threads": threads, "env": env_info(), "git": git_commit(), "peak_rss_gb": peak_rss_gb()})
    rec["times"].update({**T, "unit_total_s": time.perf_counter() - t0})
    return rec
