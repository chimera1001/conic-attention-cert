"""Gap bounds of the level-one relaxation and their constants: the second-order gap bound (bound (25) of Theorem 12 in
the paper; :func:`f0v_final`) with the exact secant gaps (Lemma 3) and the value-weighted knapsack (10), its norm-based
and total-variation variants (``bound_F0``, ``bound_F0t``, ``bound_F0s``), the directional constants ``rad_c``, ``nu_c``
(Section 2), the first-order bounds of Theorem 9 in directional form, a linearization bound (``bound_F``, for the
``lin_sandwich`` family of :mod:`aimp.relax_sdp`), and the nominal affine/bilinear decomposition of the scores."""
from __future__ import annotations

import math

import numpy as np

from aimp import sets as S
from aimp.bounds import _quad_objective
from aimp.heads import softmax
from aimp.types import Instance


def _exp(x: float) -> float:
    """``exp`` that returns ``inf`` instead of raising (score widths > 709 occur on GPT-2 heads at relative radii
    close to 1; the bound is then +inf, i.e. vacuous)."""
    return math.exp(x) if x < 709.0 else math.inf


def base_radius(rep: S.AffineRep) -> float:
    """Circumradius of the base set ``U`` (``||u|| <= rho`` for every ``u in U``): ball ``rho``, box ``||h||``."""
    if not rep.r:
        return 0.0
    if rep.base == "ball":
        return float(rep.rho)
    if rep.base == "box":
        return float(np.linalg.norm(rep.h))
    raise NotImplementedError(rep.base)


def chebyshev_center(V: np.ndarray, threads=None) -> tuple[np.ndarray, float]:
    """``min_m max_k ||V_k - m||`` (small SOCP); the radius is re-evaluated in numpy at the returned ``m`` (an upper
    bound for any ``m``, so every bound using it stays valid)."""
    import cvxpy as cp

    from aimp.solvers import cvx_solve

    m = cp.Variable(V.shape[1])
    t = cp.Variable()
    prob = cp.Problem(cp.Minimize(t), [cp.norm(V[k] - m) <= t for k in range(V.shape[0])])
    status, _ = cvx_solve(prob, threads)
    mv = m.value if status in ("optimal", "suboptimal") and m.value is not None else V.mean(axis=0)
    r = float(np.max(np.linalg.norm(V - mv, axis=1)))
    r_mean = float(np.max(np.linalg.norm(V - V.mean(axis=0), axis=1)))
    return (mv, r) if r <= r_mean else (V.mean(axis=0), r_mean)


def nominal_decomposition(inst: Instance) -> dict:
    """Nominal decomposition of the scores: ``s_j = shat_j + lin_i.u_i + lin_j.u_j + u_i^T Q_j u_j`` in base coordinates
    around the set centers; ``ahat = sigma(shat)``, ``G = J(shat)``; ``delta_j = ||lin_i|| rho_i + ||lin_j|| rho_j``
    (bounds the affine score shift), ``eta_j = ||Q_j|| rho_i rho_j`` (bounds the bilinear remainder),
    ``nu = max_j ||N B_j|| rho_j``, ``theta = 1 - ||ahat||^2``. For a query that is a key (shared variable) ``lin_i``
    holds both linear parts."""
    M, N = inst.head.M, inst.head.N
    i = inst.query
    reps = {t: S.affine(inst.sets[t]) for t in inst.tokens}
    rho = {t: base_radius(reps[t]) for t in inst.tokens}
    pieces, shat, delta, eta_j = [], [], [], []
    for j in inst.keys:
        same = j == i
        const, lin_i, lin_j, Q = _quad_objective(M, reps[i], reps[j], same)
        lin_i = lin_i if reps[i].r else np.zeros(0)
        lin_j = None if same else (lin_j if reps[j].r else np.zeros(0))
        dl = (np.linalg.norm(lin_i) * rho[i] if lin_i.size else 0.0) + \
             (np.linalg.norm(lin_j) * rho[j] if (lin_j is not None and lin_j.size) else 0.0)
        rj = rho[i] if same else rho[j]
        e = float(np.linalg.norm(Q, 2)) * rho[i] * rj if (Q is not None and Q.size and rho[i] and rj) else 0.0
        pieces.append({"key": j, "same": same, "shat": const, "lin_i": lin_i, "lin_j": lin_j, "Q": Q})
        shat.append(const)
        delta.append(float(dl))
        eta_j.append(e)
    shat = np.array(shat)
    ahat = softmax(shat)
    G = np.diag(ahat) - np.outer(ahat, ahat)
    nu = max((float(np.linalg.norm(N @ reps[j].B, 2)) * rho[j] if reps[j].r else 0.0) for j in inst.keys)
    V = np.stack([N @ reps[j].c for j in inst.keys])
    m0, rad0 = chebyshev_center(V)
    Dt = max(delta)
    eta = max(eta_j)
    return {"pieces": pieces, "shat": shat, "ahat": ahat, "G": G, "delta": np.array(delta), "Delta_tilde": Dt,
            "eta_j": np.array(eta_j), "eta": eta, "eps_lin": Dt**2 + eta, "nu": nu, "rad0": rad0, "m0": m0,
            "theta": float(1.0 - ahat @ ahat), "rho": rho}


def bound_F0(inst: Instance, ctrl: dict, nom: dict | None = None) -> dict:
    """Norm-based form of the second-order gap bound (a diagnostic; the paper's bound (25) is :func:`f0v_final`):
    ``r - p_star <= ||c|| [rad (eta_exp + eta) + nu W / 2]`` with ``W = sum_j (a_hi - a_lo)`` (attention intervals of
    Lemma 2, clipped to [0, 1]), ``eta_exp = Delta^2/8 e^Delta`` (secant slack constant on the score width),
    ``eta = max_j ||B_i^T M B_j|| rho_i rho_j`` (the score covariance (9)), ``nu = max_j ||N B_j|| rho_j``."""
    nom = nom or nominal_decomposition(inst)
    a_lo = np.clip(np.asarray(ctrl["a_lo"], float), 0, 1)
    a_hi = np.clip(np.asarray(ctrl["a_hi"], float), 0, 1)
    W = float(np.sum(a_hi - a_lo))
    D = float(ctrl["Delta"])
    eta_exp = D**2 / 8 * _exp(D)
    rad, cn = float(ctrl["rad"]), float(ctrl["c_norm"])
    term_exp = rad * (eta_exp + nom["eta"])
    term_cov = nom["nu"] * W / 2
    return {"bound_F0": cn * (term_exp + term_cov), "F0_W": W, "F0_eta_exp": eta_exp, "F0_eta": nom["eta"],
            "F0_nu": nom["nu"], "F0_term_sandwich": cn * term_exp, "F0_term_cov": cn * term_cov}


def t_width(ctrl: dict, mode: str = "loose") -> float:
    """Width of the interval on which the implemented log-form relaxation places its exp sandwich (secant).
    ``"loose"``: the wider intervals ``t_j = s_j - tau in [s_lo_j - tau_hi, s_hi_j - tau_lo]``, i.e.
    ``w_t = max_j (s_hi_j - s_lo_j) + (tau_hi - tau_lo)`` (<= 2 Delta, cf. eq. (7)). ``"tight"``: intersected with
    ``[log a_lo_j, log a_hi_j]``, the log-attention intervals (6)."""
    if mode == "loose":
        s_lo, s_hi = np.asarray(ctrl["s_lo"], float), np.asarray(ctrl["s_hi"], float)
        return float(np.max(s_hi - s_lo) + (float(ctrl["tau_hi"]) - float(ctrl["tau_lo"])))
    from aimp.bounds import t_width as tw

    return tw(ctrl["s_lo"], ctrl["s_hi"], mode)


def bound_F0t(inst: Instance, ctrl: dict, nom: dict | None = None, mode: str = "loose") -> dict:
    """:func:`bound_F0` with the secant constant evaluated on the secant's actual interval width ``w_t`` instead of
    ``Delta`` (``eta_exp = w_t^2/8 e^{w_t}``). ``mode`` must match the relaxation's ``t_interval`` (``"tight"`` for
    ``sdp_1``: the bound is stated for the relaxation whose secant lives on that interval)."""
    nom = nom or nominal_decomposition(inst)
    f0 = bound_F0(inst, ctrl, nom)
    w = t_width(ctrl, mode)
    eta_t = w**2 / 8 * _exp(w)
    cn, rad = float(ctrl["c_norm"]), float(ctrl["rad"])
    sand = cn * rad * (eta_t + nom["eta"])
    return {"bound_F0t": sand + f0["F0_term_cov"], "F0t_w_t": w, "F0t_eta_exp": eta_t, "F0t_term_sandwich": sand,
            "F0t_mode": mode, "F0t_term_cov": f0["F0_term_cov"]}


def sandwich_S(ctrl: dict) -> float:
    """The weighted secant slack of eq. (8): ``S = (1/8) sum_j a_hi_j w_j^2`` with ``w_j = t_hi_j - t_lo_j`` the widths
    of the log-attention intervals (6) and ``a_hi`` the upper attention bounds of Lemma 2 (clipped to [0, 1])."""
    from aimp.bounds import t_intervals

    t_lo, t_hi = t_intervals(ctrl["s_lo"], ctrl["s_hi"], "tight")
    a_hi = np.clip(np.asarray(ctrl["a_hi"], float), 0.0, 1.0)
    return float(np.sum(a_hi * (t_hi - t_lo) ** 2) / 8)


def bound_F0s(inst: Instance, ctrl: dict, nom: dict | None = None, pruning_slack: float = 0.0) -> dict:
    """Total-variation form of the second-order gap bound (diagnostic; the measure ``2S/(1-S)`` discussed after
    Theorem 12): ``||c|| [rad (2S/(1-S) + eta) + nu W/2] + pruning_slack``, defined when ``S < 1`` (``S`` =
    :func:`sandwich_S`). Should hold for ``sdp_1`` and be tighter than ``bound_F0t``."""
    nom = nom or nominal_decomposition(inst)
    f0 = bound_F0(inst, ctrl, nom)
    S = sandwich_S(ctrl)
    applies = S < 1
    cn, rad = float(ctrl["c_norm"]), float(ctrl["rad"])
    b = cn * (rad * (2 * S / (1 - S) + nom["eta"])) + f0["F0_term_cov"] + float(pruning_slack) if applies else None
    return {"bound_F0s": b, "F0s_S": S, "F0s_applies": bool(applies), "F0s_term_cov": f0["F0_term_cov"],
            "F0s_pruning_slack": float(pruning_slack)}


def bound_F(inst: Instance, ctrl: dict, nom: dict | None = None) -> dict:
    """Linearization gap bound (not in the paper; for the ``lin_sandwich`` family of :mod:`aimp.relax_sdp`):
    ``2||c|| [(rad_0 + nu)(Delta~^2 + eta) + 2 theta nu Delta~]`` with the constants of
    :func:`nominal_decomposition`."""
    nom = nom or nominal_decomposition(inst)
    cn = float(ctrl["c_norm"])
    t1 = (nom["rad0"] + nom["nu"]) * nom["eps_lin"]
    t2 = 2 * nom["theta"] * nom["nu"] * nom["Delta_tilde"]
    return {"bound_F": 2 * cn * (t1 + t2), "F_rad0": nom["rad0"], "F_Delta_tilde": nom["Delta_tilde"],
            "F_theta": nom["theta"], "F_term_sandwich": 2 * cn * t1, "F_term_cov": 2 * cn * t2}


def support_values(inst: Instance) -> np.ndarray:
    """``h_j = max_{z in Z_j} c^T N z`` (support function of ``N Z_j`` in direction ``c``; the support values
    ``h_j^max`` of the first-order certificate, Proposition 8)."""
    g = inst.head.N.T @ inst.c
    return np.array([S.support(inst.sets[j], g) for j in inst.keys])


THEOREM_TOL = 1e-6  # a bound check passes if r - p_ref <= bound + 1e-6
F0_METHODS_PHASE1 = ("sdp_1", "sdp_2", "sdp_3_score", "sdp_3_cut", "sdp_3_bxb", "sdp_3_ball")  # interval diag + persp
F0_INFO_METHODS_PHASE1 = ("sdp_0", "sdp_1_unitdiag")  # hypotheses not met (no persp / unit diag): informational


def theorem_fields(rec: dict, name: str, bound: float, slack: float = 0.0, applies: bool = True) -> dict:
    """Strong check ``gap = value - p_ref <= bound + slack + 1e-6`` written as ``bound_<name>``, ``ratio_<name>``,
    ``bound_<name>_satisfied`` (None if the hypotheses do not hold: ``applies=False`` stores the ratio only,
    under ``bound_<name>_satisfied_info``)."""
    gap = rec.get("gap")
    ok = None if gap is None else bool(gap <= bound + slack + THEOREM_TOL)
    rec[f"bound_{name}"] = bound
    rec[f"ratio_{name}"] = None if (gap is None or bound <= 0) else gap / bound
    if slack:
        rec[f"bound_{name}_slack"] = slack
    if applies:
        rec[f"bound_{name}_satisfied"] = ok
    else:
        rec[f"bound_{name}_satisfied"] = None
        rec[f"bound_{name}_satisfied_info"] = ok
    return rec


# --------------------------------------------------------------------------- directional constants and bounds
def directional_constants(inst: Instance) -> dict:
    """Directional constants of Section 2 (the half-range ``rad_c`` of the values) in the coordinates the certificate
    runs in: ``h_j(z) = c^T N z``, ``g_j = B_j^T N^T c``, ``hbar_j = c^T N c_j``. ``rad_c`` = half the range of
    ``c^T N z`` over the union of the sets,
    computed exactly from support functions (``max_j sup_{Z_j} h - min_j inf_{Z_j} h``)/2, which is the closed form
    ``max_j (hbar_j + rho_j ||g_j||) - min_j (hbar_j - rho_j ||g_j||)`` for balls / ellipsoids; ``nu_c = max_j rho_j ||g_j||``
    with ``rho_j`` the base radius (:func:`base_radius`; ellipsoids: the ball radius of the base, ``B_j = E``)."""
    gN = inst.head.N.T @ np.asarray(inst.c, float)
    hi = np.array([S.support(inst.sets[j], gN) for j in inst.keys])
    lo = np.array([-S.support(inst.sets[j], -gN) for j in inst.keys])
    hbar = np.array([float(gN @ inst.sets[j].center) for j in inst.keys])
    gnorm = []
    for j in inst.keys:
        rep = S.affine(inst.sets[j])
        if not rep.r:
            gnorm.append(0.0)
        elif rep.base in ("ball", "box"):
            gnorm.append(base_radius(rep) * float(np.linalg.norm(rep.B.T @ gN)))
        else:  # hull: no base radius; the half-width of h over the set
            gnorm.append(float((hi[len(gnorm)] - lo[len(gnorm)]) / 2))
    gnorm = np.array(gnorm)
    return {"rad_c": float((hi.max() - lo.min()) / 2), "nu_c": float(gnorm.max()), "h_sup": hi, "h_inf": lo,
            "hbar": hbar, "rho_g": gnorm, "m_c": float((hi.max() + lo.min()) / 2)}


def directional_bounds(rad_c: float, nu_c: float, *, n: int, Delta: float, gamma: float | None, S_: float | None,
                       eta: float | None = None, W: float | None = None, eta_exp_wt: float | None = None,
                       pruning_slack: float = 0.0, eps_cut: float | None = None) -> dict:
    """Directional bound fields from the directional constants and the record's controls. ``bound_B_c = Delta rad_c``
    (the width form of the first-order gap bound (20) of Theorem 9), ``bound_C_c = 4 (n-1) e^{-gamma} rad_c``
    (``gamma > 0``; the margin bound (21) without its denominator), ``bound_F0s_c = rad_c (2S/(1-S) + eta) + nu_c W/2``
    (+ the pruning slack of the pruned keys, as in the norm form; ``S < 1`` else ``None``), ``bound_F0t_c`` the same
    with ``eta_exp(w_t)``, ``Pi_c = rad_c min(2S/(1-S), 2)``; ``bound_ps_c = 2 eps rad_c`` with the greedy's cut radius
    ``eps`` (the general form (20) for the pair actually used, diagnostic)."""
    out = {"rad_c": rad_c, "nu_c": nu_c, "bound_B_c": Delta * rad_c,
           "bound_C_c": (4 * (n - 1) * _exp(-gamma) * rad_c) if (gamma is not None and gamma > 0) else None,
           "bound_ps_c": None if eps_cut is None else 2 * eps_cut * rad_c}
    if S_ is not None:
        q = 2 * S_ / (1 - S_) if S_ < 1 else math.inf
        out["Pi_c"] = rad_c * min(q, 2.0)
        cov = None if (W is None or nu_c is None) else nu_c * W / 2
        out["F0_term_cov_c"] = cov
        if eta is not None and cov is not None:
            out["bound_F0s_c"] = rad_c * (q + eta) + cov + pruning_slack if S_ < 1 else None
            out["bound_F0t_c"] = (rad_c * (eta_exp_wt + eta) + cov + pruning_slack) if eta_exp_wt is not None else None
    return out


def W_of(ctrl_or_alo, a_hi=None) -> float:
    """``W = sum_j (a_hi_j - a_lo_j)`` with both clipped to [0, 1] (as :func:`bound_F0`)."""
    if a_hi is None:
        a_lo, a_hi = ctrl_or_alo["a_lo"], ctrl_or_alo["a_hi"]
    else:
        a_lo = ctrl_or_alo
    return float(np.sum(np.clip(np.asarray(a_hi, float), 0, 1) - np.clip(np.asarray(a_lo, float), 0, 1)))


def directional_fields(inst: Instance, ctrl: dict, nom: dict | None = None, pruning_slack: float = 0.0,
                       cut: str = "B") -> dict:
    """All directional bound fields of one instance from its controls."""
    from aimp.bounds import choose_cut

    nom = nom or nominal_decomposition(inst)
    dc = directional_constants(inst)
    w = t_width(ctrl, "tight")
    Sv = sandwich_S(ctrl)
    out = directional_bounds(dc["rad_c"], dc["nu_c"], n=inst.n, Delta=float(ctrl["Delta"]), gamma=float(ctrl["gamma"]),
                             S_=Sv, eta=nom["eta"], W=W_of(ctrl), eta_exp_wt=w**2 / 8 * _exp(w),
                             pruning_slack=pruning_slack, eps_cut=choose_cut(ctrl, inst.n, cut)["eps_for_bound"])
    out.update({"m_c": dc["m_c"], "S": Sv, "w_t": w})
    return out


# --------------------------------------------------------------------------- value-weighted secant slack
def secant_widths(ctrl: dict) -> np.ndarray:
    """Per-key secant widths ``w_j = t_hi_j - t_lo_j`` on the log-attention intervals (6) (the widths of
    :func:`sandwich_S`)."""
    from aimp.bounds import t_intervals

    t_lo, t_hi = t_intervals(ctrl["s_lo"], ctrl["s_hi"], "tight")
    return np.asarray(t_hi - t_lo, float)


def perkey_S(ctrl: dict) -> np.ndarray:
    """``S_j = a_hi_j w_j^2 / 8`` (the per-key bound ``D_j <= S_j`` on the exact secant gap, Lemma 3; ``a_hi`` clipped
    to [0, 1]); ``sum_j S_j = S``."""
    return np.clip(np.asarray(ctrl["a_hi"], float), 0.0, 1.0) * secant_widths(ctrl) ** 2 / 8


def value_scores(S_j: np.ndarray, h_max: np.ndarray, phi_min: float) -> np.ndarray:
    """``P_j = S_j (h_j^max - phi_min)_+``: the slack that can move the objective (value-weighted sandwich identity,
    Proposition 4); ``sum_j P_j`` is the value-weighted slack of Figure 3."""
    return np.asarray(S_j, float) * np.maximum(np.asarray(h_max, float) - float(phi_min), 0.0)


def p7v_sides(e: np.ndarray, d: np.ndarray, h: np.ndarray) -> tuple[float, float]:
    """Both sides of the value-weighted sandwich identity of Proposition 4 for ``abar = e + d`` (``sum abar = 1``)
    and ``sigma = e / sum e``: ``sum_j (abar_j - sigma_j) h_j`` and ``sum_j d_j (h_j - hbar_sigma)``."""
    e, d, h = (np.asarray(x, float) for x in (e, d, h))
    sigma = e / e.sum()
    hs = float(sigma @ h)
    return float((e + d - sigma) @ h), float(d @ (h - hs))


def bound_F0v_terms(S_j, h_max, a_lo, a_hi, nu_cj, phi_min_cert: float, eta: float, rad_c: float,
                    pruning_slack: float = 0.0) -> dict:
    """The second-order gap bound with the per-key slack ``S_j >= D_j`` in place of the knapsack of bound (25):
    ``r - p_star <= sum_j S_j (h_j^max - phi_min_cert + eta rad_c)_+ + eta rad_c + (1/2) sum_j (a_hi_j - a_lo_j)
    nu_{c,j}`` (+ the pruning slack of the pruned keys, as in every second-order bound check here).
    ``a_lo``, ``a_hi`` clipped to [0, 1] as in ``W``. Returns the bound and its three terms."""
    S_j, h_max, nu_cj = (np.asarray(x, float) for x in (S_j, h_max, nu_cj))
    a_lo = np.clip(np.asarray(a_lo, float), 0.0, 1.0)
    a_hi = np.clip(np.asarray(a_hi, float), 0.0, 1.0)
    er = float(eta) * float(rad_c)
    sand = float(S_j @ np.maximum(h_max - float(phi_min_cert) + er, 0.0))
    cov = 0.5 * float((a_hi - a_lo) @ nu_cj)
    return {"bound_F0v": sand + er + cov + float(pruning_slack), "F0v_term_sandwich": sand, "F0v_term_eta": er,
            "F0v_term_cov": cov, "F0v_pruning_slack": float(pruning_slack)}


def perkey_fields(inst: Instance, ctrl: dict, *, phi_min_strong: float, phi_min_cert: float, eta: float,
                  pruning_slack: float = 0.0, a_nom: np.ndarray | None = None, dc: dict | None = None) -> dict:
    """Per-key arrays and unit fields of the value-weighted slack of one unit from its controls (no SDP).

    Per key: ``S_j``, ``w_j``, ``a_lo_j``, ``a_hi_j``, ``hbar_j = c^T N c_j``, ``nu_{c,j} = rho_j ||g_j||``,
    ``h_j^max = hbar_j + nu_{c,j}`` (the exact support value, :func:`directional_constants`), ``P_j`` (with
    ``phi_min_strong``). Unit: ``Pi_v = sum_j P_j``, ``bound_F0v`` (with the certified ``phi_min_cert``),
    ``P_max_share = max_j P_j / Pi_v``, ``S_max_share = max_j S_j / S``, ``dom_key_is_argmaxS`` (``argmax_j S_j`` is the
    key of largest nominal attention ``a_nom``), and both rules' split keys with their predicted gains
    (``Pi_v / (Pi_v - 0.75 P_j)``, ``S / (S - 0.75 S_j)``)."""
    from aimp.heads import nominal

    dc = dc or directional_constants(inst)
    w = secant_widths(ctrl)
    Sj = perkey_S(ctrl)
    h_max = np.asarray(dc["h_sup"], float)
    nu_cj = np.asarray(dc["rho_g"], float)
    hbar = np.asarray(dc["hbar"], float)
    Pj = value_scores(Sj, h_max, phi_min_strong)
    a_nom = np.asarray(nominal(inst)["a"] if a_nom is None else a_nom, float)
    Sv, Pv = float(Sj.sum()), float(Pj.sum())
    jS, jP, jA = int(np.argmax(Sj)), int(np.argmax(Pj)), int(np.argmax(a_nom))

    def gains(j):
        return {"S": Sv / (Sv - 0.75 * Sj[j]) if Sv > 0 else None,
                "v": Pv / (Pv - 0.75 * Pj[j]) if Pv > 0 else None}

    f0v = bound_F0v_terms(Sj, h_max, ctrl["a_lo"], ctrl["a_hi"], nu_cj, phi_min_cert, eta, dc["rad_c"], pruning_slack)
    gS, gP = gains(jS), gains(jP)
    return {
        "perkey": {"S_j": Sj, "w_j": w, "a_lo": np.asarray(ctrl["a_lo"], float), "a_hi": np.asarray(ctrl["a_hi"], float),
                   "hbar": hbar, "nu_c": nu_cj, "h_max": h_max, "P_j": Pj, "a_nom": a_nom},
        "S": Sv, "Pi_v": Pv, **f0v, "phi_min_cert": float(phi_min_cert), "phi_min_strong": float(phi_min_strong),
        "eta": float(eta), "rad_c": float(dc["rad_c"]), "nu_c": float(nu_cj.max()),
        "P_max_share": float(Pj.max() / Pv) if Pv > 0 else None, "S_max_share": float(Sj.max() / Sv) if Sv > 0 else None,
        "dom_key": jA, "argmax_S": jS, "argmax_P": jP, "dom_key_is_argmaxS": bool(jS == jA),
        "dom_key_is_argmaxP": bool(jP == jA), "rules_agree": bool(jS == jP),
        "gain_S_at_argmaxS": gS["S"], "gain_v_at_argmaxS": gS["v"], "gain_S_at_argmaxP": gP["S"],
        "gain_v_at_argmaxP": gP["v"],
    }


def saturated_slack(S_j, a_lo, a_hi) -> np.ndarray:
    """Saturating per-key slack: besides ``d_j <= S_j``, also ``d_j = abar_j - e^{tbar_j} <= a_hi_j - a_lo_j``, because
    the relaxation keeps ``abar_j <= a_hi_j`` and ``tbar_j >= t_lo_j >= log a_lo_j`` (log-attention intervals (6); cf.
    the saturation of the exact secant gap in Lemma 3). Returns ``D_j = min(S_j, a_hi_j - a_lo_j)`` (``a`` clipped to
    [0, 1]); ``D_j`` in place of ``S_j`` in ``Pi_v`` / ``bound_F0v`` gives their saturating forms (``Pi_v_sat``,
    ``bound_F0v_sat``)."""
    a_lo = np.clip(np.asarray(a_lo, float), 0.0, 1.0)
    a_hi = np.clip(np.asarray(a_hi, float), 0.0, 1.0)
    return np.minimum(np.asarray(S_j, float), np.maximum(a_hi - a_lo, 0.0))


# --------------------------------------------------------------------------- bound (25) with the exact secant gaps
_W_SERIES = 1e-2  # w below: series for y = expm1(w)/w - 1 (terms to w^6 / 5040)
_Y_SERIES = 1e-2  # y below: series for F(y) = (1+y) log1p(y) - y (terms to y^8 / 56)
_W_LARGE = 30.0  # w above: the m-form A + m (log(1+y) - 1) (A may underflow, e^w overflow)


def _F_of_y(y: float) -> float:
    """``F(y) = (1+y) log1p(y) - y`` (>= 0), with the series ``sum_{k>=2} (-1)^k y^k / (k(k-1))`` (k <= 8) for
    ``y < 1e-2``."""
    if y < _Y_SERIES:
        return y * y * (1 / 2 - y * (1 / 6 - y * (1 / 12 - y * (1 / 20 - y * (1 / 30 - y * (1 / 42 - y / 56))))))
    return (1.0 + y) * math.log1p(y) - y


def secant_gap(t_lo: float, t_hi: float) -> float:
    """The exact secant gap ``D = max_{t in [t_lo, t_hi]} (ell(t) - e^t)`` of Lemma 3, eq. (14) (``ell`` the secant of
    ``exp`` on the interval), evaluated stably (Appendix B): ``w = t_hi - t_lo``, ``A = e^{t_lo}``,
    ``y = expm1(w)/w - 1`` (series ``sum_{k=1..6} w^k / (k+1)!`` for ``w < 1e-2``), ``D = A F(y)`` (:func:`_F_of_y`);
    for ``w > 30`` the equivalent ``D = A + m (log(1+y) - 1)`` with ``m = (e^{t_hi} - A)/w`` and
    ``log(1+y) = w + log(-expm1(-w)) - log w`` (no overflow of ``e^w``, no ``0 x inf`` when ``A`` underflows). ``D = 0``
    if ``w <= 0``. The series have enough terms for full double precision at their switch points (a 3-term ``y`` series
    is only ~2e-11 relative at ``w = 1e-3``)."""
    t_lo, t_hi = float(t_lo), float(t_hi)
    w = t_hi - t_lo
    if not (w > 0.0):
        return 0.0
    if w > _W_LARGE:
        A = math.exp(t_lo)
        m = (math.exp(t_hi) - A) / w
        L = w + math.log(-math.expm1(-w)) - math.log(w)
        return max(A + m * (L - 1.0), 0.0)
    if w < _W_SERIES:
        y = w * (1 / 2 + w * (1 / 6 + w * (1 / 24 + w * (1 / 120 + w * (1 / 720 + w / 5040)))))
    else:
        y = math.expm1(w) / w - 1.0
    return math.exp(t_lo) * _F_of_y(y)


def secant_gaps(t_lo, t_hi) -> np.ndarray:
    """:func:`secant_gap` per key."""
    return np.array([secant_gap(a, b) for a, b in zip(np.asarray(t_lo, float), np.asarray(t_hi, float))])


def knapsack(chi, D, D_tot: float) -> tuple[float, np.ndarray]:
    """``KN(chi, D, D_tot) = max { sum_j d_j chi_j : 0 <= d_j <= D_j, sum_j d_j <= D_tot }`` (fractional knapsack: fill
    the keys in decreasing ``chi_j``, only those with ``chi_j > 0``). Returns the value and the maximizer ``d``."""
    chi = np.asarray(chi, float)
    D = np.maximum(np.asarray(D, float), 0.0)
    d = np.zeros_like(D)
    left = max(float(D_tot), 0.0)
    val = 0.0
    for j in np.argsort(-chi, kind="stable"):
        if chi[j] <= 0 or left <= 0:
            break
        take = min(D[j], left)
        d[j] = take
        left -= take
        val += take * chi[j]
    return float(val), d


def f0v_final(t_lo, t_hi, h_max, a_lo, a_hi, nu_cj, *, phi_min_cert: float, phi_min_strong: float | None, eta: float,
              rad_c: float, pruning_slack: float = 0.0) -> dict:
    """The second-order gap bound (25) of Theorem 12. Per key ``A_j = e^{t_lo_j}``, ``B_j = e^{t_hi_j}``, ``w_j``,
    ``D_j`` (:func:`secant_gap`), ``chi_j = (h_j^max - phi_min_cert + eta rad_c)_+``, ``chi*_j = (h_j^max -
    phi_min_strong)_+``; unit ``D_tot = 1 - sum_j A_j``,
    ``bound_F0v_final = KN(chi, D, D_tot) + eta rad_c + (1/2) sum_j (a_hi_j - a_lo_j) nu_{c,j}`` (+ the pruning slack
    of the keys pruned in the relaxation, cf. Appendix B; ``a`` clipped to [0, 1]), ``Pi_v_star = KN(chi*, D, D_tot)``.
    ``t_lo, t_hi``: the log-attention intervals (6) (:func:`aimp.bounds.t_intervals`, ``"tight"``; equal to
    ``[log a_lo_j, log a_hi_j]``)."""
    t_lo, t_hi = np.asarray(t_lo, float), np.asarray(t_hi, float)
    h_max, nu_cj = np.asarray(h_max, float), np.asarray(nu_cj, float)
    a_lo = np.clip(np.asarray(a_lo, float), 0.0, 1.0)
    a_hi = np.clip(np.asarray(a_hi, float), 0.0, 1.0)
    A = np.exp(t_lo)
    B = np.exp(t_hi)
    D = secant_gaps(t_lo, t_hi)
    D_tot = max(1.0 - float(A.sum()), 0.0)
    er = float(eta) * float(rad_c)
    chi = np.maximum(h_max - float(phi_min_cert) + er, 0.0)
    kn, d = knapsack(chi, D, D_tot)
    cov = 0.5 * float((a_hi - a_lo) @ nu_cj)
    out = {"D_j": D, "A_j": A, "B_j": B, "w_j": t_hi - t_lo, "chi_j": chi, "d_knapsack": d, "D_tot": D_tot,
           "D_sum": float(D.sum()), "budget_binds": bool(float(D[chi > 0].sum()) > D_tot),
           "F0vf_term_knapsack": kn, "F0vf_term_eta": er, "F0vf_term_cov": cov, "F0vf_pruning_slack": float(pruning_slack),
           "bound_F0v_final": kn + er + cov + float(pruning_slack)}
    if phi_min_strong is not None:
        chis = np.maximum(h_max - float(phi_min_strong), 0.0)
        out["chi_star_j"] = chis
        out["Pi_v_star"] = knapsack(chis, D, D_tot)[0]
    return out


def f0v_final_fields(inst: Instance, ctrl: dict, *, phi_min_cert: float, phi_min_strong: float | None, eta: float,
                     pruning_slack: float = 0.0, dc: dict | None = None) -> dict:
    """:func:`f0v_final` of one instance from its controls (log-attention intervals (6), exact support values
    ``h_j^max``, ``nu_{c,j} = rho_j ||g_j||``, ``rad_c`` of :func:`directional_constants`)."""
    from aimp.bounds import t_intervals

    dc = dc or directional_constants(inst)
    t_lo, t_hi = t_intervals(ctrl["s_lo"], ctrl["s_hi"], "tight")
    return f0v_final(t_lo, t_hi, dc["h_sup"], ctrl["a_lo"], ctrl["a_hi"], dc["rho_g"], phi_min_cert=phi_min_cert,
                     phi_min_strong=phi_min_strong, eta=eta, rad_c=dc["rad_c"], pruning_slack=pruning_slack)
