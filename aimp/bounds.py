"""Certified quantities: lemma constants (margin bound, Cauchy–Schwarz score half-width, LayerNorm radius, secant
gap), score intervals (eq. (5) of the paper), attention intervals (Lemma 2), the controlling quantities of Table 1 and
the regime labels."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from aimp import sets as S
from aimp.heads import logsumexp, softmax
from aimp.types import Instance

GROTHENDIECK_UPPER = 1.7822139781
SHOR_PAD_REL = 1e-7  # safety pad added to SDP-derived interval ends (solver tolerance)


# ------------------------------------------------------------------ reference constants
def grothendieck_upper() -> float:
    return GROTHENDIECK_UPPER


def eta(Delta: float) -> float:
    """``Delta^2/8 * exp(Delta)``: the interpolation bound of the secant gap (Lemma 3) on an interval of width
    ``Delta``, with ``e^Delta`` in place of ``e^{t_hi}`` (not the score covariance eta of eq. (9))."""
    return Delta**2 / 8 * (math.exp(Delta) if Delta < 709.0 else math.inf)


#: Default pattern cut (R6) of the greedy and of ``sdp_1``: the width pair ``(sigma(s_mid), Delta/2)``. The margin pair's
#: l1 cut is implied by the attention intervals (proof of Theorem 9; Appendix B), so ``cut="auto"``, which takes the pair
#: with the smaller radius as in the paper's experiments, amounts to dropping the cut when it picks the margin pair.
DEFAULT_CUT = "B"


def eps_B(Delta: float) -> float:
    return Delta / 2


def eps_C(n: int, gamma: float) -> float:
    if n <= 1:
        return 0.0
    x = (n - 1) * math.exp(-gamma)
    return 2 * x / (1 + x)


def bound_B(Delta: float, rad: float, c_norm: float) -> float:
    return Delta * rad * c_norm


def bound_C(n: int, gamma: float, rad: float, c_norm: float) -> float:
    return 4 * (n - 1) * math.exp(-gamma) * rad * c_norm


def bound_general(eps: float, rad: float, c_norm: float) -> float:
    return 2 * eps * rad * c_norm


# ----------------------------------------------------------------------- lemma quantities
def l4_onehot_bound(n: int, gamma: float) -> float:
    """Margin bound of Lemma 2(iii), eq. (13): ``||a - e_{j*}||_1 <= 2(n-1)e^{-gamma}/(1+(n-1)e^{-gamma})``."""
    return eps_C(n, gamma)


def l5_halfwidth(M_op: float, rho_i: float, rho_j: float, zi_norm: float, zj_norm: float) -> float:
    """Cauchy–Schwarz score half-width: ``|s_j - zbar_i^T M zbar_j| <= ||M|| (rho_i ||zbar_j|| + rho_j ||zbar_i|| +
    rho_i rho_j)``."""
    return M_op * (rho_i * zj_norm + rho_j * zi_norm + rho_i * rho_j)


def l6_rmsnorm_radius(d: int, g_inf: float, rho_x: float, xbar_norm: float) -> float:
    """RMSNorm radius: ``||z - zbar|| <= 2 sqrt(d) ||g||_inf rho_x / ||xbar||`` (requires ``rho_x < ||xbar||``)."""
    if rho_x >= xbar_norm:
        raise ValueError("the normalization radius bound requires rho_x < ||xbar||")
    return 2 * math.sqrt(d) * g_inf * rho_x / xbar_norm


def l6_layernorm_radius(g: np.ndarray, rho_x: float, xbar: np.ndarray) -> float:
    """The same radius for LayerNorm: replace ``xbar`` by ``P xbar`` (centering); bias is additive."""
    d = xbar.shape[0]
    Pxbar = xbar - xbar.mean()
    return l6_rmsnorm_radius(d, float(np.max(np.abs(g))), rho_x, float(np.linalg.norm(Pxbar)))


def p7_secant_error(s_lo: float, s_hi: float) -> float:
    """Interpolation bound of the secant gap (Lemma 3): ``max (secant - exp) <= Delta^2/8 e^{s_hi}``."""
    return (s_hi - s_lo) ** 2 / 8 * math.exp(s_hi)


def l3_max_vsigma(v: np.ndarray, s_lo: np.ndarray, s_hi: np.ndarray) -> tuple[float, np.ndarray]:
    """Scores in a box (Remark 1): ``max_{s in box} v^T softmax(s)`` exactly, over the ``n+1`` threshold vertices
    ``s_j = s_hi_j if v_j > theta else s_lo_j``. Returns (value, maximizing vertex)."""
    v = np.asarray(v, float)
    order = np.argsort(-v)
    best, best_s = -np.inf, None
    for k in range(v.shape[0] + 1):
        s = np.asarray(s_lo, float).copy()
        top = order[:k]
        s[top] = np.asarray(s_hi, float)[top]
        val = float(v @ softmax(s))
        if val > best:
            best, best_s = val, s
    return best, best_s


def vertex_softmax(h: np.ndarray, s_lo: np.ndarray, s_hi: np.ndarray) -> dict:
    """The Vertex-Softmax bound ``r_VS = max {sum_j softmax(s)_j h_j : s_lo <= s <= s_hi}``, computed exactly over the
    ``n + 1`` threshold vertices of the remark on scores in a box (``h`` sorted in decreasing order, a top segment of
    ``k`` keys at ``s_hi`` and the rest at ``s_lo``, ``k = 0..n``), with the softmax in log space
    (``softmax(s)_j = exp(s_j - logsumexp(s))``). With ``h`` the support values ``h_j^max`` this is the tightest bound
    that the score intervals allow with every value at its largest (Rezazadeh and Gholami Davoodi); it lies between
    ``p*`` and the first-order certificate ``r_pp``. Returns ``value``, the maximizing vertex ``s``, its pattern ``a``,
    the size ``k`` of its top segment and the values of all ``n + 1`` vertices."""
    h = np.asarray(h, float)
    lo = np.asarray(s_lo, float)
    hi = np.asarray(s_hi, float)
    order = np.argsort(-h, kind="stable")
    vals = np.empty(h.shape[0] + 1)
    best = None
    for k in range(h.shape[0] + 1):
        s = lo.copy()
        s[order[:k]] = hi[order[:k]]
        a = np.exp(s - logsumexp(s))
        vals[k] = float(a @ h)
        if best is None or vals[k] > best[0]:
            best = (vals[k], s, a, k)
    return {"value": best[0], "s": best[1], "a": best[2], "k": best[3], "vertex_values": vals}


# ------------------------------------------------------------------ attention intervals (Lemma 2)
def softmax_interval_bounds(s_lo: np.ndarray, s_hi: np.ndarray) -> dict:
    """``a_lo_j = 1/(1+sum_{l!=j} exp(s_hi_l - s_lo_j))``, ``a_hi_j = 1/(1+sum_{l!=j} exp(s_lo_l - s_hi_j))``,
    ``tau_lo = logsumexp(s_lo)``, ``tau_hi = logsumexp(s_hi)`` (computed in log space)."""
    s_lo = np.asarray(s_lo, float)
    s_hi = np.asarray(s_hi, float)
    n = s_lo.shape[0]
    a_lo = np.empty(n)
    a_hi = np.empty(n)
    for j in range(n):
        other_hi = np.delete(s_hi, j)
        other_lo = np.delete(s_lo, j)
        a_lo[j] = math.exp(s_lo[j] - logsumexp(np.append(other_hi, s_lo[j])))
        a_hi[j] = math.exp(s_hi[j] - logsumexp(np.append(other_lo, s_hi[j])))
    return {"a_lo": a_lo, "a_hi": a_hi, "tau_lo": logsumexp(s_lo), "tau_hi": logsumexp(s_hi)}


def log_softmax_bounds(s_lo: np.ndarray, s_hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``log a_lo_j = s_lo_j - lse(s_hi_{-j}, s_lo_j)``, ``log a_hi_j = s_hi_j - lse(s_lo_{-j}, s_hi_j)`` (the attention
    intervals of Lemma 2 in log space; no underflow)."""
    s_lo = np.asarray(s_lo, float)
    s_hi = np.asarray(s_hi, float)
    n = s_lo.shape[0]
    lo, hi = np.empty(n), np.empty(n)
    for j in range(n):
        lo[j] = s_lo[j] - logsumexp(np.append(np.delete(s_hi, j), s_lo[j]))
        hi[j] = s_hi[j] - logsumexp(np.append(np.delete(s_lo, j), s_hi[j]))
    return lo, hi


def t_intervals(s_lo: np.ndarray, s_hi: np.ndarray, mode: str = "tight") -> tuple[np.ndarray, np.ndarray]:
    """Interval of the log-attention ``t_j = s_j - tau = log a_j`` on which the SDP places its exp sandwich.
    ``"tight"`` (default; eq. (6) of the paper): ``[s_lo_j - tau_hi, s_hi_j - tau_lo]`` intersected with
    ``[log a_lo_j, log a_hi_j]`` (Lemma 2), which is always contained in the former, so ``tight = [log a_lo, log a_hi]``
    (the max/min is still taken literally). ``"loose"``: the wider intervals ``[s_lo_j - tau_hi, s_hi_j - tau_lo]``."""
    s_lo = np.asarray(s_lo, float)
    s_hi = np.asarray(s_hi, float)
    t_lo = s_lo - logsumexp(s_hi)
    t_hi = s_hi - logsumexp(s_lo)
    if mode == "loose":
        return t_lo, t_hi
    if mode != "tight":
        raise ValueError(mode)
    la_lo, la_hi = log_softmax_bounds(s_lo, s_hi)
    return np.maximum(t_lo, la_lo), np.minimum(t_hi, la_hi)


def t_width(s_lo, s_hi, mode: str = "tight") -> float:
    """``w_t = max_j (t_hi_j - t_lo_j)``: the secant width of eq. (6) (for ``mode="tight"``)."""
    t_lo, t_hi = t_intervals(s_lo, s_hi, mode)
    return float(np.max(t_hi - t_lo))


# -------------------------------------------------------------------- score intervals (eq. (5))
def _score_pieces(inst: Instance, j: int):
    """For key token ``j``: (rep_i, rep_j, same) with ``s = (c_i + B_i u_i)^T M (c_j + B_j u_j)``."""
    rep_i = S.affine(inst.sets[inst.query])
    same = j == inst.query
    rep_j = rep_i if same else S.affine(inst.sets[j])
    return rep_i, rep_j, same


def _linear_interval(inst: Instance, j: int) -> Optional[tuple[float, float]]:
    """Exact interval when the score is linear (query or key fixed) or constant."""
    M = inst.head.M
    Si, Sj = inst.sets[inst.query], inst.sets[j]
    rep_i, rep_j, same = _score_pieces(inst, j)
    if same:
        if rep_i.r == 0:
            v = float(rep_i.c @ M @ rep_i.c)
            return v, v
        return None
    if rep_i.r == 0:
        g = M.T @ rep_i.c
        return -S.support(Sj, -g), S.support(Sj, g)
    if rep_j.r == 0:
        g = M @ rep_j.c
        return -S.support(Si, -g), S.support(Si, g)
    return None


def _cs(inst: Instance, j: int) -> tuple[float, float]:
    M = inst.head.M
    Si, Sj = inst.sets[inst.query], inst.sets[j]
    zi, zj = Si.center, Sj.center
    mid = float(zi @ M @ zj)
    hw = l5_halfwidth(np.linalg.norm(M, 2), S.circumradius(Si), S.circumradius(Sj),
                      np.linalg.norm(zi), np.linalg.norm(zj))
    return mid - hw, mid + hw


def _interval(inst: Instance, j: int) -> tuple[float, float]:
    """Interval arithmetic: exact coordinate bounds of ``k = M z_j``, then interval dot with ``z_i``."""
    M = inst.head.M
    k_lo, k_hi = S.coord_bounds(inst.sets[j], M)
    q_lo, q_hi = S.coord_bounds(inst.sets[inst.query])
    prods = np.stack([q_lo * k_lo, q_lo * k_hi, q_hi * k_lo, q_hi * k_hi])
    return float(prods.min(axis=0).sum()), float(prods.max(axis=0).sum())


def _quad_objective(M, rep_i, rep_j, same):
    """``s = const + lin_i.u_i + lin_j.u_j + <Q, U_ij>`` (for ``same``: ``lin_j = 0``, ``Q`` on ``U_ii``)."""
    ci, Bi, cj, Bj = rep_i.c, rep_i.B, rep_j.c, rep_j.B
    const = float(ci @ M @ cj)
    if same:
        lin_i = Bi.T @ (M + M.T) @ ci
        return const, lin_i, None, Bi.T @ M @ Bi
    return const, Bi.T @ (M @ cj), Bj.T @ (M.T @ ci), Bi.T @ M @ Bj


def lifted_base_constraints(u, U, rep, cp):
    """Shor lifting of a base set (first moments ``u``, second-moment block ``U``):
    box ``|u| <= h`` and the diagonal products ``U_kk <= h_k^2``; ball ``||u|| <= rho``, ``tr U <= rho^2``."""
    if rep.base == "box":
        return [u <= rep.h, u >= -rep.h, cp.diag(U) <= rep.h**2]
    if rep.base == "ball":
        return [cp.norm(u) <= rep.rho, cp.trace(U) <= rep.rho**2]
    if rep.base == "simplex":
        return [u >= 0, cp.sum(u) == 1, U >= 0, cp.sum(U, axis=1) == u]
    raise ValueError(rep.base)


def box_x_box(u1, u2, U12, h1, h2, cp):
    """Full RLT products of ``-h1 <= u1 <= h1`` with ``-h2 <= u2 <= h2`` (four families)."""
    r1, r2 = h1.shape[0], h2.shape[0]
    A1 = cp.reshape(u1, (r1, 1), order="C") @ np.ones((1, r2))  # A1[k,l] = u1_k
    A2 = np.ones((r1, 1)) @ cp.reshape(u2, (1, r2), order="C")  # A2[k,l] = u2_l
    H = np.outer(h1, h2)
    H1 = np.outer(h1, np.ones(r2))
    H2 = np.outer(np.ones(r1), h2)
    return [
        H - cp.multiply(H1, A2) - cp.multiply(H2, A1) + U12 >= 0,  # (h1-u1)(h2-u2)
        H + cp.multiply(H1, A2) + cp.multiply(H2, A1) + U12 >= 0,  # (h1+u1)(h2+u2)
        H + cp.multiply(H1, A2) - cp.multiply(H2, A1) - U12 >= 0,  # (h1-u1)(h2+u2)
        H - cp.multiply(H1, A2) + cp.multiply(H2, A1) - U12 >= 0,  # (h1+u1)(h2-u2)
    ]


def _shor(inst: Instance, j: int, use_box_x_box: bool = False, threads=None) -> tuple[float, float, dict]:
    import cvxpy as cp

    from aimp.solvers import cvx_solve

    rep_i, rep_j, same = _score_pieces(inst, j)
    const, lin_i, lin_j, Q = _quad_objective(inst.head.M, rep_i, rep_j, same)
    if not (np.any(Q) or np.any(lin_i) or (lin_j is not None and np.any(lin_j))):
        return const, const, {"constant": True}
    ri = rep_i.r
    rj = 0 if same else rep_j.r
    D = ri + rj
    X = cp.Variable((D + 1, D + 1), PSD=True)
    ui = X[0, 1:1 + ri]
    cons = [X[0, 0] == 1]
    cons += lifted_base_constraints(ui, X[1:1 + ri, 1:1 + ri], rep_i, cp)
    if same:
        expr = const + lin_i @ ui + cp.sum(cp.multiply(Q, X[1:1 + ri, 1:1 + ri]))
        if use_box_x_box and rep_i.base == "box":
            cons += box_x_box(ui, ui, X[1:1 + ri, 1:1 + ri], rep_i.h, rep_i.h, cp)
    else:
        uj = X[0, 1 + ri:]
        cons += lifted_base_constraints(uj, X[1 + ri:, 1 + ri:], rep_j, cp)
        Uij = X[1:1 + ri, 1 + ri:]
        expr = const + lin_i @ ui + lin_j @ uj + cp.sum(cp.multiply(Q, Uij))
        if use_box_x_box and rep_i.base == "box" and rep_j.base == "box":
            cons += box_x_box(ui, ui, X[1:1 + ri, 1:1 + ri], rep_i.h, rep_i.h, cp)
            cons += box_x_box(uj, uj, X[1 + ri:, 1 + ri:], rep_j.h, rep_j.h, cp)
            cons += box_x_box(ui, uj, Uij, rep_i.h, rep_j.h, cp)
    out, info = [], {}
    for sense in ("min", "max"):
        prob = cp.Problem(cp.Maximize(expr) if sense == "max" else cp.Minimize(expr), cons)
        status, inf = cvx_solve(prob, threads)
        if status not in ("optimal", "suboptimal"):
            raise RuntimeError(f"shor score interval {sense} for key {j}: {status} {inf}")
        v = float(prob.value)
        pad = SHOR_PAD_REL * max(1.0, abs(v))
        out.append(v - pad if sense == "min" else v + pad)
        info[sense] = {"status": status, "value_raw": v,
                       "rank1_ratio": _rank1_ratio(X.value)}
    return out[0], out[1], info


def _rank1_ratio(X) -> float:
    if X is None:
        return float("nan")
    ev = np.linalg.eigvalsh((X + X.T) / 2)[::-1]
    return float(ev[1] / ev[0]) if ev.shape[0] > 1 and ev[0] > 0 else 0.0


def _exact(inst: Instance, j: int, time_limit: float = 60.0, threads=None) -> tuple[float, float, dict]:
    """Gurobi NonConvex on the bilinear program; interval ends are the ObjBounds (certified)."""
    import gurobipy as gp
    from gurobipy import GRB

    from aimp.solvers import gurobi_model, gurobi_status

    rep_i, rep_j, same = _score_pieces(inst, j)
    const, lin_i, lin_j, Q = _quad_objective(inst.head.M, rep_i, rep_j, same)
    res, info = [], {}
    for sense in ("min", "max"):
        m, _ = gurobi_model(f"score_exact_{j}_{sense}", threads=threads, time_limit=time_limit)
        ui = _gurobi_base_vars(m, rep_i, "ui")
        if same:
            obj = const + lin_i @ ui + ui @ Q @ ui
        else:
            uj = _gurobi_base_vars(m, rep_j, "uj")
            obj = const + lin_i @ ui + lin_j @ uj + ui @ Q @ uj
        m.setObjective(obj, GRB.MAXIMIZE if sense == "max" else GRB.MINIMIZE)
        m.optimize()
        st = gurobi_status(m)
        if m.SolCount == 0 and st != "optimal":
            raise RuntimeError(f"exact score interval {sense} key {j}: {st}")
        res.append(float(m.ObjBound))
        info[sense] = {"status": st, "obj_val": float(m.ObjVal), "obj_bound": float(m.ObjBound),
                       "mip_gap": float(m.MIPGap) if m.IsMIP else 0.0}
        m.dispose()
    return res[0], res[1], info


def _gurobi_base_vars(m, rep, name):
    """MVar ``u`` of a base set with its constraints (box/ball/simplex)."""
    from gurobipy import GRB

    if rep.base == "box":
        return m.addMVar(rep.r, lb=-rep.h, ub=rep.h, name=name)
    if rep.base == "ball":
        # normalized: u = rho w, ||w|| <= 1. Gurobi's absolute tolerance on ``u@u <= rho^2`` lets
        # ||u|| exceed rho by ~viol/(2 rho) (observed 4e-5 at rho=0.01); on ``w@w <= 1`` it is ~viol/2 * rho.
        u = m.addMVar(rep.r, lb=-rep.rho, ub=rep.rho, name=name)
        w = m.addMVar(rep.r, lb=-1.0, ub=1.0, name=f"w_{name}")
        m.addConstr(u == rep.rho * w)
        m.addConstr(w @ w <= 1.0)
        return u
    if rep.base == "simplex":
        u = m.addMVar(rep.r, lb=0.0, ub=1.0, name=name)
        m.addConstr(u.sum() == 1)
        return u
    if rep.base == "point":
        return m.addMVar(0, name=name)
    raise ValueError(rep.base)


def _dual_trust(inst: Instance, j: int, n_samples: int = 200, seed: int = 0, threads=None) -> tuple[float, float, dict]:
    """Balls, distinct tokens: ``max_{z_i} z_i^T M z_j = zbar_i^T M z_j + rho_i ||M z_j||``; maximize
    that over ``z_j`` by boundary sampling + local search (a lower bound on ``s_hi``); the certified
    interval is the Shor one (returned), the sampled values are in ``info``."""
    from scipy.optimize import minimize

    Si, Sj = inst.sets[inst.query], inst.sets[j]
    if Si.kind != "ball" or Sj.kind != "ball" or j == inst.query:
        raise ValueError("dual_trust needs two distinct ball tokens")
    M = inst.head.M
    qi, ri, cj, rj = Si.center, Si.rho, Sj.center, Sj.rho
    rng = np.random.default_rng(seed)

    def best(sign):
        def f(v):  # z_j = cj + rj * v/||v|| (boundary parametrization)
            nv = np.linalg.norm(v)
            z = cj + rj * v / max(nv, 1e-300)
            return -(sign * qi @ M @ z + ri * np.linalg.norm(M @ z))

        vals = []
        for v0 in rng.standard_normal((n_samples, cj.shape[0])):
            vals.append((f(v0), v0))
        vals.sort(key=lambda t: t[0])
        best_v = min(minimize(f, v0, method="L-BFGS-B").fun for _, v0 in vals[:5])
        return -best_v

    hi_lb = best(+1.0)  # lower bound on s_hi
    lo_ub = -best(-1.0)  # upper bound on s_lo
    lo, hi, info = _shor(inst, j, threads=threads)
    info.update({"s_hi_lower_bound_sampled": hi_lb, "s_lo_upper_bound_sampled": lo_ub})
    return lo, hi, info


def score_intervals(inst: Instance, method: str = "auto", threads=None, **kw) -> dict:
    """Certified ``[s_lo_j, s_hi_j]`` for every key (key order). ``method``:
    ``cs`` | ``interval`` | ``shor`` | ``shor_bxb`` | ``exact`` | ``dual_trust`` | ``auto``
    (auto = ``exact`` if any box token and ``d <= 16``, else ``shor``).

    For every method except ``cs`` (the Cauchy–Schwarz half-width of :func:`l5_halfwidth`), a score that is linear
    (query or key fixed) or constant is computed exactly by support functions."""
    if method == "auto":
        any_box = any(inst.sets[t].kind == "box" for t in inst.tokens)
        method = "exact" if (any_box and inst.d <= 16) else "shor"
    s_lo, s_hi, extra = [], [], {}
    for j in inst.keys:
        lin = None if method == "cs" else _linear_interval(inst, j)
        if lin is not None:
            lo, hi, inf = lin[0], lin[1], {"linear_exact": True}
        elif method == "cs":
            (lo, hi), inf = _cs(inst, j), {}
        elif method == "interval":
            (lo, hi), inf = _interval(inst, j), {}
        elif method == "shor":
            lo, hi, inf = _shor(inst, j, threads=threads)
        elif method == "shor_bxb":
            lo, hi, inf = _shor(inst, j, use_box_x_box=True, threads=threads)
        elif method == "exact":
            lo, hi, inf = _exact(inst, j, threads=threads, **kw)
        elif method == "dual_trust":
            lo, hi, inf = _dual_trust(inst, j, threads=threads, **kw)
        else:
            raise ValueError(method)
        s_lo.append(lo)
        s_hi.append(hi)
        extra[str(j)] = inf
    return {"method": method, "s_lo": np.array(s_lo), "s_hi": np.array(s_hi), "info": extra}


# ------------------------------------------------------------------- controlling quantities
def value_spread(inst: Instance, threads=None) -> dict:
    """``rad_ub = min_m max_j (||N zbar_j - m|| + dev_j)`` with ``dev_j >= max_{Z_j} ||N(z - zbar_j)||``: an upper bound
    on the Chebyshev radius of the union of the value sets ``N Z_j`` (``||c||`` times it bounds ``rad_c``, Section 2),
    minimized over ``m`` by an SOCP (MOSEK). ``is_upper_bound`` is True unless every
    ``dev_j`` is exact and the configuration is a single key."""
    import cvxpy as cp

    from aimp.solvers import cvx_solve

    N = inst.head.N
    V = np.stack([N @ inst.sets[j].center for j in inst.keys])
    devs, exact = zip(*(S.max_dev_norm(inst.sets[j], N) for j in inst.keys))
    devs = np.array(devs)
    m = cp.Variable(N.shape[0])
    t = cp.Variable()
    prob = cp.Problem(cp.Minimize(t), [cp.norm(V[k] - m) + devs[k] <= t for k in range(len(devs))])
    status, _ = cvx_solve(prob, threads)
    if status in ("optimal", "suboptimal"):
        mval = m.value
    else:
        mval = V.mean(axis=0)
    # evaluate the bound at the returned m in numpy (a valid upper bound for any m)
    rad = float(np.max(np.linalg.norm(V - mval, axis=1) + devs))
    rad_mean = float(np.max(np.linalg.norm(V - V.mean(axis=0), axis=1) + devs))
    if rad_mean < rad:
        rad, mval = rad_mean, V.mean(axis=0)
    return {"rad": rad, "m": mval, "devs": devs, "dev_exact": list(exact), "socp_status": status,
            "is_upper_bound": True}


def controls(inst: Instance, method: str = "auto", eps_target: float = 1e-3,
             intervals: Optional[dict] = None, threads=None) -> dict:
    """The controlling quantities that depend on the score intervals (Table 1: ``Delta``, attention intervals,
    ``tau`` interval, ``gamma``, ``w_t``), the norm value radius ``rad``, the first-order bounds of Theorem 9 in norm
    form (``bound_B``, ``bound_C``) and a regime label: ``SMALL_WIDTH`` / ``MARGIN`` if the width / margin bound meets
    ``eps_target``, else ``AMBIGUOUS``. Also the support values ``h_max`` and the Vertex-Softmax bound ``r_VS``
    (:func:`vertex_softmax`) on the same score intervals."""
    from aimp.theory import support_values

    si = intervals if intervals is not None else score_intervals(inst, method, threads=threads)
    s_lo, s_hi = si["s_lo"], si["s_hi"]
    n = inst.n
    sbar = (s_lo + s_hi) / 2
    Delta = float(np.max(s_hi - s_lo))
    sm = softmax_interval_bounds(s_lo, s_hi)
    vs = value_spread(inst, threads=threads)
    rad = vs["rad"]
    c_norm = float(np.linalg.norm(inst.c))
    R = max(S.max_norm(inst.sets[t]) for t in inst.tokens)
    if n == 1:
        gamma, j_star, top2 = math.inf, 0, None
    else:
        gs = [s_lo[k] - np.max(np.delete(s_hi, k)) for k in range(n)]
        j_star = int(np.argmax(gs))
        gamma = float(gs[j_star])
        o = np.argsort(-sbar)[:2]
        top2 = float(min(s_hi[o[0]], s_hi[o[1]]) - max(s_lo[o[0]], s_lo[o[1]]))
    eB = eps_B(Delta)
    eC = eps_C(n, gamma) if gamma > 0 else None
    bB = bound_B(Delta, rad, c_norm)
    bC = bound_C(n, gamma, rad, c_norm) if gamma > 0 else None
    if bB <= eps_target:
        regime = "SMALL_WIDTH"
    elif gamma > 0 and bC <= eps_target:
        regime = "MARGIN"
    else:
        regime = "AMBIGUOUS"
    h_max = support_values(inst)
    vx = vertex_softmax(h_max, s_lo, s_hi)
    return {
        "h_max": h_max, "r_VS": vx["value"], "VS_k": vx["k"],
        "score_intervals": si, "s_lo": s_lo, "s_hi": s_hi, "sbar": sbar, "Delta": Delta,
        "a_lo": sm["a_lo"], "a_hi": sm["a_hi"], "tau_lo": sm["tau_lo"], "tau_hi": sm["tau_hi"],
        "ahat": softmax(sbar), "rad": rad, "rad_is_upper_bound": vs["is_upper_bound"],
        "rad_info": vs, "R": R, "gamma": gamma, "j_star": j_star, "top2_overlap": top2,
        "eps_B": eB, "eps_C": eC, "bound_B": bB, "bound_C": bC, "c_norm": c_norm,
        "regime": regime, "top2_flag": (top2 is not None and top2 > 0), "eps_target": eps_target,
        "w_t_loose": t_width(s_lo, s_hi, "loose"), "w_t": t_width(s_lo, s_hi, "tight"),
    }


def child_controls(ctrl: dict, s_lo: np.ndarray, s_hi: np.ndarray, n: int) -> dict:
    """Controls of a branch-and-bound node: the same instance restricted to score box ``[s_lo, s_hi]`` (a sub-box of the
    parent's certified intervals). Everything that depends on the scores (attention intervals, ``tau``, ``Delta``,
    ``ahat``, ``gamma``, cut radii, ``w_t``, ``r_VS``) is recomputed; set-only quantities (``rad``, ``R``, ``c_norm``,
    ``h_max``) are inherited."""
    s_lo = np.asarray(s_lo, float)
    s_hi = np.asarray(s_hi, float)
    sbar = (s_lo + s_hi) / 2
    Delta = float(np.max(s_hi - s_lo))
    sm = softmax_interval_bounds(s_lo, s_hi)
    if n == 1:
        gamma, j_star = math.inf, 0
    else:
        gs = [s_lo[k] - np.max(np.delete(s_hi, k)) for k in range(n)]
        j_star = int(np.argmax(gs))
        gamma = float(gs[j_star])
    out = dict(ctrl)
    out.update({
        "s_lo": s_lo, "s_hi": s_hi, "sbar": sbar, "Delta": Delta, "a_lo": sm["a_lo"], "a_hi": sm["a_hi"],
        "tau_lo": sm["tau_lo"], "tau_hi": sm["tau_hi"], "ahat": softmax(sbar), "gamma": gamma, "j_star": j_star,
        "eps_B": eps_B(Delta), "eps_C": eps_C(n, gamma) if gamma > 0 else None,
        "bound_B": bound_B(Delta, ctrl["rad"], ctrl["c_norm"]),
        "bound_C": bound_C(n, gamma, ctrl["rad"], ctrl["c_norm"]) if gamma > 0 else None,
        "w_t_loose": t_width(s_lo, s_hi, "loose"), "w_t": t_width(s_lo, s_hi, "tight"),
        "score_intervals": {**ctrl.get("score_intervals", {}), "s_lo": s_lo, "s_hi": s_hi, "branch": True},
    })
    if ctrl.get("h_max") is not None:
        vs = vertex_softmax(ctrl["h_max"], s_lo, s_hi)
        out.update({"r_VS": vs["value"], "VS_k": vs["k"]})
    return out


def choose_cut(ctrl: dict, n: int, cut: str = "auto") -> dict:
    """The certified pair ``(ahat, eps)`` of the l1 cut of the pattern polytope (18): ``"B"`` the width pair, ``"C"``
    the margin pair (when ``gamma > 0``), ``"auto"`` the one with the smaller radius. Returns
    ``{"cut_used", "ahat", "eps", "eps_for_bound"}``; ``cut="none"`` gives
    ``eps = None`` (no cut; trivial eps = 1 for the bound since ``||a - a'||_1 <= 2``)."""
    if cut == "none":
        return {"cut_used": "none", "ahat": None, "eps": None, "eps_for_bound": 1.0}
    options = {"B": (ctrl["ahat"], ctrl["eps_B"])}
    if ctrl["gamma"] > 0:
        e = np.zeros(n)
        e[ctrl["j_star"]] = 1.0
        options["C"] = (e, ctrl["eps_C"])
    if cut == "auto":
        cut = min(options, key=lambda k: options[k][1])
    if cut not in options:
        raise ValueError(f"cut {cut!r} not applicable (gamma={ctrl['gamma']})")
    ahat, eps = options[cut]
    eps_bound = float(eps)
    if cut == "B" and "C" in options:  # the width-pair polytope (in box ∩ simplex) inherits the margin bound (21)
        eps_bound = min(eps_bound, float(options["C"][1]))
    return {"cut_used": cut, "ahat": ahat, "eps": float(eps), "eps_for_bound": eps_bound}
