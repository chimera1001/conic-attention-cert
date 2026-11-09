"""The level-one lifted relaxation (Definition 3 of the paper; CVXPY + MOSEK) with switchable RLT families.

Lifted vector ``xi = (u_i ; u_{j1} ; ... ; a_1 ; ... ; a_n ; [tau] ; [s_j ...] ; [p_j, q_j ...])`` in
base coordinates (sets.AffineRep; for ellipsoids the base coordinate ``w`` is lifted, not ``z``; tokens with
a degenerate set are not lifted). Moment matrix ``X = [[1, xi^T], [xi, Xi]] >= 0``. Index bookkeeping
lives in :class:`LiftIndex` (tests/test_lift_index.py).

**Affine forms.** Every linear inequality ``f(xi) >= 0`` used in an RLT product is a coefficient
vector ``f`` over ``(1, xi)`` (index 0 = constant). For a rank-1 ``X = (1,xi)(1,xi)^T``,
``f^T X g = f(xi) g(xi)``, so the lifted product of two nonnegative forms is ``f^T X g >= 0`` and a
batch of products is the matrix inequality ``F X G^T >= 0``. All RLT families below except
``persp`` / ``simplex_x_z`` / ``box_x_box`` (written out directly) are written this way.

Families (``rlt`` is a set of names):

* ``"diag"`` (always on unless ``diag=False``): for every lifted scalar with
  bounds ``l <= x <= u``, ``Xi[x,x] <= (l+u) x - l u``: box coordinates ``Xi[u_k,u_k] <= h_k^2``,
  ``a_j`` with the attention intervals ``[a_lo_j, a_hi_j]`` (the interval diagonal (R4)), and the lifted ``tau``,
  ``s_j``, ``p_j``, ``q_j``.
* ``"persp"``: products of ``a_j >= 0`` and ``1 - a_j >= 0`` with ``Z_j`` (the products with the sets of (R5), in
  perspective form).
* ``"simplex_x_z"``: ``sum_j Xi[a_j, xi_k] = xi_k`` for every lifted coordinate, ``Xi[a_j,a_l] >= 0`` (the sums and
  sign constraints of (R5)).
* ``"softmax_x_z"``: the secant ``g_j >= 0`` and ``K`` tangents ``h_j^(k) >= 0`` (linear in
  ``s_j, tau, a_j``) times the linear set bounds of the query and of key ``j`` (balls/ellipsoids:
  their base bounding box ``rho -+ u_k >= 0``), times ``a_l >= 0``, ``1 - a_l >= 0`` (all ``l``) and
  times ``tau - tau_lo >= 0``, ``tau_hi - tau >= 0``. ``tau`` is lifted (``D += 1``). If ``s_j`` is
  affine in the lifted ``u`` (query or key fixed) its form is that affine function;
  otherwise ``s_j`` is lifted (``D += 1`` per such key) with ``E[s_j] = <M, Xi[u_i,u_j]> + ...``.
* ``"softmax_x_ball"`` (an alternative to ``softmax_x_z``): the same secant/tangent forms
  ``f >= 0`` times the ball constraints themselves, as SOC products ``||Xi[f, u_t]|| <= rho_t E[f]``
  (the perspective of ``||u_t|| <= rho_t`` with the nonnegative scalar ``f``, exactly as ``persp``
  does with ``a_j``), for the query and key ``j`` when their base set is a ball.
* ``"score_rlt"``: ``(s_j - s_lo_j)`` and ``(s_hi_j - s_j)`` times ``a_l``, ``1 - a_l`` (lifts quadratic ``s_j``).
* ``"cut_x_z"``: lifts ``p, q`` of the l1 cut (``D += 2n``); the cut equality ``a - ahat - p + q = 0``
  times every lifted coordinate; ``p_j, q_j, eps - sum(p+q) >= 0`` times the linear set bounds of
  every lifted token (balls: bounding box, plus the SOC perspective products as in ``persp``).
* ``"box_x_box"``: full RLT between all pairs of box blocks.
* ``"lin_sandwich"`` (a sandwich linearized at the nominal scores; not used in the paper): (S1) lifted ``pl_j, ql_j``
  (``D += 2n``) with ``r_j(a,u) = a_j - ahat_j - G_j^T (stilde(u) - shat) = pl_j - ql_j``, ``pl, ql >= 0``,
  ``sum(pl + ql) <= eps_lin = Delta~^2 + eta`` (nominal ``shat``, ``ahat = sigma(shat)``, ``G = J(shat)``; aimp.theory);
  (S2) that identity times every lifted ``u`` coordinate; (S3) ``||Xi[pl_j, u_j]|| <= rho_j pl_j`` (same for ``ql``) for
  ball keys, the linear products with the box bounds for box keys.

**Log-attention intervals, cone form, safe value.** ``t_interval="tight"`` (default; eq. (6) of the paper): the secant
lives on ``[log a_lo_j, log a_hi_j]`` intersected with ``[s_lo_j - tau_hi, s_hi_j - tau_lo]`` (``"loose"``: the wider
intervals ``[s_lo_j - tau_hi, s_hi_j - tau_lo]``, with ``t_hi`` clipped at ``log a_hi`` above ``T_CLIP``); ``w_t``
before/after is recorded. ``cone_form=True`` (default) writes ``exp(t) <= a`` as ``ExpCone(t, 1, a)`` and every norm
bound as ``SOC`` (same feasible set; CVXPY then returns full cone multipliers, which the safe bound needs). ``safe``:
``"auto"`` computes ``value_safe`` (the safe dual bound of Proposition 16, :mod:`aimp.safe_bound`) whenever the dual
residual exceeds ``1e-8 max(1, |value|)`` (relative) and then uses it as ``value``; ``"always"`` computes it on every
solve (diagnostic); ``"never"``.

**Numerics.** (a) Exp-cone pruning (Appendix B): keys with ``a_hi_j <= prune_tol`` (default ``1e-6``, certified by the
attention intervals) lose their exp-cone and secant constraints and all their ``softmax_x_*`` products (a relaxation;
the loss is at most ``pruning_slack = sum_pruned a_hi_j (h_j - min_l h_l)``). (b) Certified value:
``value := value_dual`` (MOSEK's dual objective, see aimp.solvers.certified_value) when the dual residual is
``<= 1e-8``, else ``max(primal, dual)`` flagged ``unreliable``.
"""
from __future__ import annotations

import math
import time

import numpy as np

from aimp import sets as S
from aimp.bounds import (_quad_objective, box_x_box, choose_cut, controls as _controls,
                         lifted_base_constraints)
from aimp.heads import forward
from aimp.types import Instance, SolveResult

KNOWN_FAMILIES = {"diag", "persp", "simplex_x_z", "box_x_box", "softmax_x_z", "softmax_x_ball",
                  "score_rlt", "cut_x_z", "lin_sandwich"}
PRUNE_TOL = 1e-6  # exp-cone pruning threshold on a_hi (numerics (a) above; Appendix B)
T_CLIP = 18.0  # t_interval="loose": clip t_hi = s_hi - tau_lo at log(a_hi) only above this (exp(18) ~ 6.6e7)
SOFTMAX_K = 3  # tangent points per key (endpoints + midpoint)


class LiftIndex:
    """Positions of the lifted blocks inside the moment matrix ``X`` (row/col 0 is the constant 1).

    ``u(t)`` is the slice of token ``t``'s base coordinates; ``a(idx)`` the position of ``a_idx``
    (key order). Tokens: the query first, then the keys that differ from it (``Instance.tokens``);
    tokens whose base dimension is 0 get an empty slice. Optional extra lifted scalars (``tau``,
    ``s{idx}``, ``p{idx}``, ``q{idx}``) follow the ``a`` block in the order given; ``x(name)``.
    """

    def __init__(self, dims: dict[int, int], token_order: list[int], n: int, extras: tuple = ()):
        self.token_order = list(token_order)
        self._u: dict[int, slice] = {}
        off = 1
        for t in self.token_order:
            self._u[t] = slice(off, off + dims[t])
            off += dims[t]
        self._a = list(range(off, off + n))
        off += n
        self._x = {}
        for name in extras:
            if name in self._x:
                raise ValueError(f"duplicate lifted scalar {name}")
            self._x[name] = off
            off += 1
        self.D = off - 1
        self.size = self.D + 1
        self.n = n

    @staticmethod
    def for_instance(inst: Instance, extras: tuple = ()) -> "LiftIndex":
        dims = {t: S.affine(inst.sets[t]).r for t in inst.tokens}
        return LiftIndex(dims, inst.tokens, inst.n, extras)

    def u(self, t: int) -> slice:
        return self._u[t]

    def a(self, idx: int) -> int:
        return self._a[idx]

    def x(self, name: str) -> int:
        return self._x[name]

    def has(self, name: str) -> bool:
        return name in self._x

    @property
    def a_slice(self) -> slice:
        return slice(self._a[0], self._a[-1] + 1)

    def labels(self) -> list[str]:
        """Human-readable label of every row of ``X``."""
        lab = ["1"] * self.size
        for t, sl in self._u.items():
            for k, r in enumerate(range(sl.start, sl.stop)):
                lab[r] = f"u{t}[{k}]"
        for idx, r in enumerate(self._a):
            lab[r] = f"a{idx}"
        for name, r in self._x.items():
            lab[r] = name
        return lab

    # ---- affine forms over (1, xi)
    def e(self, pos: int, coef: float = 1.0) -> np.ndarray:
        f = np.zeros(self.size)
        f[pos] = coef
        return f

    def const(self, c: float) -> np.ndarray:
        return self.e(0, c)


def _set_bound_forms(L: LiftIndex, t: int, rep) -> list[np.ndarray]:
    """Linear set bounds of token ``t`` as nonnegative forms: box ``h_k -+ u_k``; ball/ellipsoid base
    bounding box ``rho -+ u_k`` (valid but weaker than the ball)."""
    out = []
    if not rep.r:
        return out
    lim = rep.h if rep.base == "box" else np.full(rep.r, rep.rho)
    for k, pos in enumerate(range(L.u(t).start, L.u(t).stop)):
        out.append(L.const(lim[k]) - L.e(pos))
        out.append(L.const(lim[k]) + L.e(pos))
    return out


def _rlt_geq(F: list[np.ndarray], G: list[np.ndarray], X, cp) -> list:
    """Lifted products ``F X G^T >= 0`` of two lists of nonnegative forms."""
    if not F or not G:
        return []
    return [np.vstack(F) @ X @ np.vstack(G).T >= 0]


def _soc_products(F: list[np.ndarray], L: LiftIndex, t: int, rep, X, cp, cone_form: bool = True) -> list:
    """For nonnegative forms ``f`` and a ball base set of token ``t``: ``||(X f)[u_t]|| <= rho E[f]``."""
    if not F or not rep.r or rep.base != "ball":
        return []
    cons = []
    for f in F:
        Xf = X @ f
        cons.append(cp.SOC(rep.rho * Xf[0], Xf[L.u(t)]) if cone_form else cp.norm(Xf[L.u(t)]) <= rep.rho * Xf[0])
    return cons


def solve_sdp(inst: Instance, ctrl: dict | None = None, rlt=("persp", "simplex_x_z"), cut: str = "auto",
              diag: bool = True, threads: int | None = None, name: str | None = None,
              softmax_K: int = SOFTMAX_K, diag_a: str = "interval",
              check_point: np.ndarray | None = None, time_limit: float | None = None,
              prune_tol: float | None = PRUNE_TOL, value_mode: str = "certified", t_interval: str = "tight",
              cone_form: bool = True, safe: str = "auto") -> SolveResult:
    """``diag_a="interval"`` (default; the interval diagonal (R4)): ``Xi[a_j,a_j] <= (a_lo+a_hi) a_j - a_lo a_hi``;
    ``diag_a="unit"``: the weaker unit diagonal ``Xi[a_j,a_j] <= a_j`` (the comparison of Figure 1 and Table 3).

    ``check_point`` (a feasible ``Z``, T x d): do not solve; instead set ``X`` to the rank-1 lift of
    the true configuration at ``Z`` (true ``a, tau, s, p, q``) and return the maximum violation of every
    constraint (status ``"checked"``, ``diagnostics["max_violation"]``) — a validity test of the
    formulation: every true point must be feasible, with objective ``c^T y(Z)``.

    ``prune_tol`` (numerics (a) of the module docstring; ``None`` disables pruning) and ``value_mode``
    (``"certified"``: numerics (b); ``"primal"``: CVXPY's value at the returned ``X``, not a certified bound).
    ``t_interval``, ``cone_form``, ``safe``: see the module docstring."""
    t0 = time.perf_counter()
    c0 = time.process_time()
    import cvxpy as cp

    from aimp.solvers import cvx_solve

    rlt = set(rlt)
    if diag:
        rlt.add("diag")
    unknown = rlt - KNOWN_FAMILIES
    if unknown:
        raise NotImplementedError(f"RLT families not implemented: {sorted(unknown)}")
    ctrl = ctrl or _controls(inst)
    cutinfo = choose_cut(ctrl, inst.n, cut)
    M, N, c = inst.head.M, inst.head.N, inst.c
    n = inst.n
    i = inst.query
    reps = {t: S.affine(inst.sets[t]) for t in inst.tokens}
    if any(rep.base == "simplex" for rep in reps.values()):
        raise NotImplementedError("hull sets are not supported in the SDP")
    notes = []

    # --- score structure: s_j = const + lin_i.u_i + lin_j.u_j + <Q, U_ij>
    pieces = []
    for j in inst.keys:
        same = j == i
        const, lin_i, lin_j, Q = _quad_objective(M, reps[i], reps[j], same)
        quad = bool(reps[i].r and (same or reps[j].r)) and bool(np.any(Q))
        pieces.append((j, same, const, lin_i, lin_j, Q, quad))

    need_forms = bool(rlt & {"softmax_x_z", "softmax_x_ball", "score_rlt"})
    lift_tau = bool(rlt & {"softmax_x_z", "softmax_x_ball"})
    use_cut = cutinfo["eps"] is not None
    if "cut_x_z" in rlt and not use_cut:
        notes.append("cut_x_z skipped: no l1 cut (cut='none')")
    lift_pq = "cut_x_z" in rlt and use_cut
    use_lin = "lin_sandwich" in rlt
    nom = None
    if use_lin:
        from aimp.theory import nominal_decomposition

        nom = nominal_decomposition(inst)
    a_hi_ctrl = np.asarray(ctrl["a_hi"], float)
    pruned = [idx for idx in range(n) if prune_tol is not None and a_hi_ctrl[idx] <= prune_tol]
    extras = []
    if lift_tau:
        extras.append("tau")
    if need_forms:
        extras += [f"s{idx}" for idx, pc in enumerate(pieces) if pc[6]]
    if lift_pq:
        extras += [f"p{idx}" for idx in range(n)] + [f"q{idx}" for idx in range(n)]
    if use_lin:
        extras += [f"pl{idx}" for idx in range(n)] + [f"ql{idx}" for idx in range(n)]
    L = LiftIndex.for_instance(inst, tuple(extras))
    X = cp.Variable((L.size, L.size), PSD=True)
    cons = [X[0, 0] == 1]
    uf = {t: X[0, L.u(t)] for t in inst.tokens}  # first moments
    a = X[0, L.a_slice]

    def blk(t1, t2):
        return X[L.u(t1), L.u(t2)]

    # --- lifted set constraints
    for t in inst.tokens:
        if reps[t].r:
            if reps[t].base == "box":
                cons += [uf[t] <= reps[t].h, uf[t] >= -reps[t].h]
                if "diag" in rlt:
                    cons += [cp.diag(blk(t, t)) <= reps[t].h ** 2]
            elif cone_form and reps[t].base == "ball":
                cons += [cp.SOC(cp.Constant(float(reps[t].rho)), uf[t]), cp.trace(blk(t, t)) <= reps[t].rho ** 2]
            else:
                cons += lifted_base_constraints(uf[t], blk(t, t), reps[t], cp)
    if "box_x_box" in rlt:
        box_tokens = [t for t in inst.tokens if reps[t].base == "box"]
        for x1, t1 in enumerate(box_tokens):
            for t2 in box_tokens[x1:]:
                cons += box_x_box(uf[t1], uf[t2], blk(t1, t2), reps[t1].h, reps[t2].h, cp)

    # --- scores s_j = <M, E[z_i z_j^T]> (first moments) and their forms
    s_lo, s_hi = ctrl["s_lo"], ctrl["s_hi"]
    s, s_form = [], []
    for idx, (j, same, const, lin_i, lin_j, Q, quad) in enumerate(pieces):
        e = const
        f = L.const(const)
        if reps[i].r:
            e = e + lin_i @ uf[i]
            f[L.u(i)] += lin_i
        if not same and reps[j].r:
            e = e + lin_j @ uf[j]
            f[L.u(j)] += lin_j
        if quad:
            e = e + cp.sum(cp.multiply(Q, blk(i, j)))
            if L.has(f"s{idx}"):
                cons += [X[0, L.x(f"s{idx}")] == e]
                e = X[0, L.x(f"s{idx}")]
                f = L.e(L.x(f"s{idx}"))
            else:
                f = None  # quadratic and not lifted: no affine form
        s.append(e)
        s_form.append(f)
    for idx in range(n):
        if isinstance(s[idx], float):
            continue
        cons += [s[idx] >= s_lo[idx], s[idx] <= s_hi[idx]]

    # --- softmax (log-form) with exp cone and secant
    tau_lo, tau_hi = ctrl["tau_lo"], ctrl["tau_hi"]
    if lift_tau:
        tau = X[0, L.x("tau")]
        tau_form = L.e(L.x("tau"))
    else:
        tau = cp.Variable()
        tau_form = None
    cons += [tau >= tau_lo, tau <= tau_hi]
    from aimp.bounds import t_intervals

    t_lo, t_hi = t_intervals(s_lo, s_hi, "loose")
    w_t_loose = float(np.max(t_hi - t_lo))
    a_lo, a_hi = np.maximum(ctrl["a_lo"], 0.0), np.minimum(ctrl["a_hi"], 1.0)
    if t_interval == "tight":  # eq. (6): t_j = log a_j in [log a_lo_j, log a_hi_j]
        t_lo, t_hi = t_intervals(s_lo, s_hi, "tight")
        clip_mask = np.zeros(n, bool)
    elif t_interval == "loose":
        # numerical safeguard: t_j = log a_j <= log a_hi_j is implied by exp(t_j) <= a_j <= a_hi_j; clip
        # the interval end only where exp(t_hi) would put coefficients ~ e^{18} or larger into the secant
        clip_mask = t_hi > T_CLIP
        if np.any(clip_mask):
            t_hi = np.where(clip_mask, np.minimum(t_hi, np.log(np.maximum(a_hi, 1e-300))), t_hi)
            notes.append(f"t_hi clipped to log(a_hi) for keys {np.flatnonzero(clip_mask).tolist()}")
    else:
        raise ValueError(t_interval)
    w_t_used = float(np.max(t_hi - t_lo))
    if diag_a == "unit":
        da_lo, da_hi = np.zeros(n), np.ones(n)
    elif diag_a == "interval":
        da_lo, da_hi = a_lo, a_hi
    else:
        raise ValueError(diag_a)
    softmax_forms = []  # per key: list of nonnegative forms (secant, tangents)
    for idx in range(n):
        tj = s[idx] - tau
        cons += [tj >= t_lo[idx], tj <= t_hi[idx]]
        if idx in pruned:  # pruning: no exp cone, secant or softmax products for a certified-negligible key
            softmax_forms.append([])
            continue
        cons += [cp.constraints.ExpCone(tj, cp.Constant(1.0), a[idx]) if cone_form else cp.exp(tj) <= a[idx]]
        w = t_hi[idx] - t_lo[idx]
        if w > 1e-12:
            slope = (np.exp(t_hi[idx]) - np.exp(t_lo[idx])) / w
            cons += [a[idx] <= np.exp(t_lo[idx]) + slope * (tj - t_lo[idx])]
        else:
            slope = None
            cons += [a[idx] <= np.exp(t_hi[idx])]
        forms = []
        if lift_tau and s_form[idx] is not None:
            ft = s_form[idx] - tau_form
            ea = L.e(L.a(idx))
            if slope is not None:
                forms.append(L.const(np.exp(t_lo[idx]) - slope * t_lo[idx]) + slope * ft - ea)  # secant g_j
            else:
                forms.append(L.const(np.exp(t_hi[idx])) - ea)
            for tk in np.linspace(t_lo[idx], t_hi[idx], softmax_K) if w > 1e-12 else [t_lo[idx]]:
                forms.append(ea - np.exp(tk) * (L.const(1.0 - tk) + ft))  # tangent h_j^(k)
        softmax_forms.append(forms)
    cons += [a >= 0, a <= 1, cp.sum(a) == 1, a >= ctrl["a_lo"], a <= ctrl["a_hi"]]
    if "diag" in rlt:
        cons += [X[L.a(idx), L.a(idx)] <= (da_lo[idx] + da_hi[idx]) * a[idx] - da_lo[idx] * da_hi[idx]
                 for idx in range(n)]
        if L.has("tau"):
            pt = L.x("tau")
            cons += [X[pt, pt] <= (tau_lo + tau_hi) * tau - tau_lo * tau_hi]
        for idx in range(n):
            if L.has(f"s{idx}"):
                ps_ = L.x(f"s{idx}")
                cons += [X[ps_, ps_] <= (s_lo[idx] + s_hi[idx]) * X[0, ps_] - s_lo[idx] * s_hi[idx]]
    if use_cut:
        if lift_pq:
            p = cp.hstack([X[0, L.x(f"p{idx}")] for idx in range(n)])
            q = cp.hstack([X[0, L.x(f"q{idx}")] for idx in range(n)])
            cons += [p >= 0, q >= 0]
        else:
            p = cp.Variable(n, nonneg=True)
            q = cp.Variable(n, nonneg=True)
        cons += [a - cutinfo["ahat"] == p - q, cp.sum(p + q) <= cutinfo["eps"]]

    # --- mixing: w_j = E[a_j z_j] = a_j c_j + B_j Xi[a_j, u_j]
    y = 0
    Gm = {}
    for idx, j in enumerate(inst.keys):
        rep = reps[j]
        y = y + a[idx] * (N @ rep.c)
        if rep.r:
            Gm[idx] = X[L.a(idx), L.u(j)]
            y = y + (N @ rep.B) @ Gm[idx]
    if "persp" in rlt:
        for idx, j in enumerate(inst.keys):
            rep = reps[j]
            if not rep.r:
                continue
            g, u = Gm[idx], uf[j]
            if rep.base == "box":
                cons += [g <= a[idx] * rep.h, g >= -a[idx] * rep.h,
                         u - g <= (1 - a[idx]) * rep.h, u - g >= -(1 - a[idx]) * rep.h]
            else:
                if cone_form:
                    cons += [cp.SOC(rep.rho * a[idx], g), cp.SOC(rep.rho * (1 - a[idx]), u - g)]
                else:
                    cons += [cp.norm(g) <= rep.rho * a[idx], cp.norm(u - g) <= rep.rho * (1 - a[idx])]
    if "simplex_x_z" in rlt:
        for t in inst.tokens:
            if reps[t].r:
                cons += [cp.sum(X[L.a_slice, L.u(t)], axis=0) == uf[t]]
        cons += [cp.sum(X[L.a_slice, L.a_slice], axis=0) == a, X[L.a_slice, L.a_slice] >= 0]

    # --- second-order coupling families (affine forms)
    a_forms = [L.e(L.a(l)) for l in range(n)] + [L.const(1.0) - L.e(L.a(l)) for l in range(n)]
    n_prod = 0
    if rlt & {"softmax_x_z", "softmax_x_ball"}:
        tau_forms = [tau_form - L.const(tau_lo), L.const(tau_hi) - tau_form]
        for idx, j in enumerate(inst.keys):
            F = softmax_forms[idx]
            if not F:
                if idx not in pruned:
                    notes.append(f"softmax products skipped for key {j}: score form unavailable")
                continue
            toks = [i] if j == i else [i, j]
            if "softmax_x_z" in rlt:
                G = [g for t in toks for g in _set_bound_forms(L, t, reps[t])] + a_forms + tau_forms
                cons += _rlt_geq(F, G, X, cp)
                n_prod += len(F) * len(G)
            if "softmax_x_ball" in rlt:
                for t in toks:
                    soc = _soc_products(F, L, t, reps[t], X, cp, cone_form)
                    cons += soc
                    n_prod += len(soc)
    if "score_rlt" in rlt:
        for idx in range(n):
            if s_form[idx] is None or isinstance(s[idx], float):
                continue
            F = [s_form[idx] - L.const(s_lo[idx]), L.const(s_hi[idx]) - s_form[idx]]
            cons += _rlt_geq(F, a_forms, X, cp)
            n_prod += len(F) * len(a_forms)
    if lift_pq:
        eps = cutinfo["eps"]
        pf = [L.e(L.x(f"p{idx}")) for idx in range(n)]
        qf = [L.e(L.x(f"q{idx}")) for idx in range(n)]
        slack = L.const(eps) - sum(pf) - sum(qf)
        eq = [L.e(L.a(idx)) - L.const(cutinfo["ahat"][idx]) - pf[idx] + qf[idx] for idx in range(n)]
        cons += [np.vstack(eq) @ X == 0]  # cut equality times every coordinate (and the constant)
        nonneg = pf + qf + [slack]
        G = [g for t in inst.tokens for g in _set_bound_forms(L, t, reps[t])]
        cons += _rlt_geq(nonneg, G, X, cp)
        for t in inst.tokens:
            cons += _soc_products(nonneg, L, t, reps[t], X, cp, cone_form)
        if "diag" in rlt:
            for f in pf + qf:
                pos = int(np.argmax(f))
                cons += [X[pos, pos] <= eps * X[0, pos]]
        n_prod += len(nonneg) * len(G) + len(eq) * L.size

    if use_lin:
        eps_l = nom["eps_lin"]
        plf = [L.e(L.x(f"pl{idx}")) for idx in range(n)]
        qlf = [L.e(L.x(f"ql{idx}")) for idx in range(n)]
        shift = []  # affine score shift stilde_l(u) - shat_l as forms (no constant)
        for idx, pc in enumerate(nom["pieces"]):
            f = np.zeros(L.size)
            if pc["lin_i"].size:
                f[L.u(i)] += pc["lin_i"]
            if pc["lin_j"] is not None and pc["lin_j"].size:
                f[L.u(pc["key"])] += pc["lin_j"]
            shift.append(f)
        eq = []
        for idx in range(n):
            r_j = L.e(L.a(idx)) - L.const(nom["ahat"][idx]) - sum(nom["G"][idx, l] * shift[l] for l in range(n))
            eq.append(r_j - plf[idx] + qlf[idx])
        ucols = [0] + [k for t in inst.tokens for k in range(L.u(t).start, L.u(t).stop)]
        cons += [np.vstack(eq) @ X[:, ucols] == 0]  # (S1) (column 0) and (S2) (u columns)
        cons += [X[0, L.x(f"pl{idx}")] >= 0 for idx in range(n)] + [X[0, L.x(f"ql{idx}")] >= 0 for idx in range(n)]
        cons += [sum(X[0, L.x(f"pl{idx}")] + X[0, L.x(f"ql{idx}")] for idx in range(n)) <= eps_l]
        for idx, j in enumerate(inst.keys):  # (S3)
            pq = [plf[idx], qlf[idx]]
            if reps[j].base == "ball":
                cons += _soc_products(pq, L, j, reps[j], X, cp, cone_form)
            elif reps[j].base == "box":
                cons += _rlt_geq(pq, _set_bound_forms(L, j, reps[j]), X, cp)
        if "diag" in rlt:
            for f in plf + qlf:
                pos = int(np.argmax(f))
                cons += [X[pos, pos] <= eps_l * X[0, pos]]
        n_prod += len(eq) * len(ucols)

    prob = cp.Problem(cp.Maximize(c @ y), cons)
    pruning_slack = 0.0
    if pruned:
        from aimp.theory import support_values

        hv = support_values(inst)
        pruning_slack = float(sum(a_hi_ctrl[idx] * (hv[idx] - hv.min()) for idx in pruned))
    if check_point is not None:
        return _check_point(inst, cutinfo, L, reps, X, None if lift_tau else tau,
                            (p, q) if (use_cut and not lift_pq) else None, prob, check_point, t0, nom)
    extra = {} if time_limit is None else {"MSK_DPAR_OPTIMIZER_MAX_TIME": float(time_limit)}
    t_build = time.perf_counter() - t0
    status, info = cvx_solve(prob, threads, **extra)
    t_safe0 = time.perf_counter()
    fams = sorted(rlt)
    method = name or "sdp[" + ",".join(f for f in fams if f != "diag") + "]"
    from aimp.solvers import certified_value

    vprim = float(prob.value) if status in ("optimal", "suboptimal") and prob.value is not None else None
    cv = certified_value(status, vprim, info)
    sb = None
    if value_mode == "primal":
        cv.update(value=vprim, value_source="primal", unreliable=False)
    elif status in ("optimal", "suboptimal") and (safe == "always" or (safe == "auto" and cv["unreliable"])):
        sb = _safe(prob, X, L, reps, inst, rlt, diag_a, a_hi, s_lo, s_hi, tau_lo, tau_hi, tau, cutinfo,
                   (p, q) if (use_cut and not lift_pq) else None, nom)
        if cv["unreliable"] and sb.get("value_safe") is not None and math.isfinite(sb["value_safe"]):
            cv.update(value=sb["value_safe"], value_source="safe", unreliable=False)
    t_safe = time.perf_counter() - t_safe0
    value = cv["value"] if cv["value"] is not None else float("nan")
    if status == "error" and cv["value"] is not None:
        status = "dual_certified"  # CVXPY gave up, but MOSEK's dual point is feasible to 1e-8: a valid bound
    res = SolveResult(method=method, status=status, value=value,
                      params={**info.get("mosek_params", {}), "rlt": fams, "cut": cut, "D": L.D,
                              "lifted_extras": list(extras), "softmax_K": softmax_K,
                              "diag_a": diag_a, "prune_tol": prune_tol, "value_mode": value_mode,
                              "t_interval": t_interval, "cone_form": cone_form, "safe": safe,
                              **({"eps_lin": nom["eps_lin"]} if use_lin else {})},
                      diagnostics={"cut_used": cutinfo["cut_used"], "eps": cutinfo["eps"], "notes": notes,
                                   "n_rlt_products": n_prod, "pruned_keys": pruned,
                                   "t_hi_clipped_keys": np.flatnonzero(clip_mask).tolist(),
                                   "pruning_slack": pruning_slack, "t_interval": t_interval, "w_t_loose": w_t_loose,
                                   "time_build_s": t_build, "time_safe_s": t_safe,
                                   "w_t_used": w_t_used, "cone_form": cone_form,
                                   "value_safe": None if sb is None else sb.get("value_safe"),
                                   "safe_info": None if sb is None else {k: v for k, v in sb.items() if k != "value_safe"},
                                   **{k: cv[k] for k in ("value_primal", "value_dual", "dual_resid", "primal_resid",
                                                         "value_source", "unreliable")},
                                   **{k: v for k, v in info.items() if k != "mosek_params"}},
                      approximations=([] if status == "optimal" else [status])
                      + (["value_unreliable"] if cv["unreliable"] and value_mode != "primal" else [])
                      + ([f"exp_pruned:{len(pruned)}"] if pruned else []))
    if X.value is not None and status in ("optimal", "suboptimal"):
        Xv = (X.value + X.value.T) / 2
        ev = np.linalg.eigvalsh(Xv)[::-1]
        res.diagnostics.update(_primal_violation(prob))
        res.diagnostics["X_min_eig"] = float(ev[-1])
        res.diagnostics.update({
            "top_eigs": ev[:5].tolist(), "rank1_ratio": float(ev[1] / ev[0]) if ev.size > 1 else 0.0,
            "a": np.asarray(a.value).copy(),
        })
        # recovery: first moments -> project onto the base sets -> true head
        Z = inst.centers().copy()
        for t in inst.tokens:
            if reps[t].r:
                Z[t] = S.to_z(reps[t], S.base_project(reps[t], Xv[0, L.u(t)]))
        val = forward(inst, Z)["phi"]
        res.recovered_point = Z
        res.recovered_value = float(val)
        res.diagnostics["sdp_recovered_value"] = float(val)
        res.diagnostics["sdp_gap_constructive"] = float(res.value - val)
        res.diagnostics["X"] = Xv
        res.diagnostics["labels"] = L.labels()
    res.wall_time_s = time.perf_counter() - t0
    res.cpu_time_s = time.process_time() - c0
    return res


def true_lift(inst: Instance, cutinfo: dict, L: LiftIndex, reps: dict, Z: np.ndarray,
              nom: dict | None = None) -> np.ndarray:
    """``(1, xi)`` of the true configuration ``Z``: base coordinates, ``a = softmax(s)``,
    ``tau = logsumexp(s)``, ``s_j``, and ``p = (a - ahat)_+``, ``q = (ahat - a)_+``; for ``lin_sandwich``
    ``pl = r_+``, ``ql = r_-`` with ``r_j = a_j - ahat_j - G_j^T (stilde(u) - shat)`` (nominal ``ahat``, ``G``)."""
    from aimp.heads import logsumexp

    out = forward(inst, Z)
    xi = np.zeros(L.size)
    xi[0] = 1.0
    for t in inst.tokens:
        if reps[t].r:
            u, *_ = np.linalg.lstsq(reps[t].B, Z[t] - reps[t].c, rcond=None)
            xi[L.u(t)] = u
    for idx in range(inst.n):
        xi[L.a(idx)] = out["a"][idx]
        if L.has(f"s{idx}"):
            xi[L.x(f"s{idx}")] = out["s"][idx]
        if L.has(f"p{idx}"):
            dlt = out["a"][idx] - cutinfo["ahat"][idx]
            xi[L.x(f"p{idx}")] = max(dlt, 0.0)
            xi[L.x(f"q{idx}")] = max(-dlt, 0.0)
    if L.has("tau"):
        xi[L.x("tau")] = logsumexp(out["s"])
    if L.has("pl0"):
        i = inst.query
        shift = np.zeros(inst.n)
        for idx, pc in enumerate(nom["pieces"]):
            if pc["lin_i"].size:
                shift[idx] += pc["lin_i"] @ xi[L.u(i)]
            if pc["lin_j"] is not None and pc["lin_j"].size:
                shift[idx] += pc["lin_j"] @ xi[L.u(pc["key"])]
        r = out["a"] - nom["ahat"] - nom["G"] @ shift
        for idx in range(inst.n):
            xi[L.x(f"pl{idx}")] = max(r[idx], 0.0)
            xi[L.x(f"ql{idx}")] = max(-r[idx], 0.0)
    return xi


def _check_point(inst, cutinfo, L, reps, X, tau_var, pq_vars, prob, Z, t0, nom=None) -> SolveResult:
    from aimp.heads import logsumexp

    xi = true_lift(inst, cutinfo, L, reps, Z, nom)
    X.value = np.outer(xi, xi)
    out = forward(inst, Z)
    if tau_var is not None:
        tau_var.value = logsumexp(out["s"])
    if pq_vars is not None:
        dlt = out["a"] - cutinfo["ahat"]
        pq_vars[0].value = np.maximum(dlt, 0.0)
        pq_vars[1].value = np.maximum(-dlt, 0.0)
    viol = [_violation(con) for con in prob.constraints]
    worst = int(np.argmax(viol)) if viol else -1
    return SolveResult(method="sdp_check", status="checked", value=float(prob.objective.value),
                       diagnostics={"max_violation": max(viol, default=0.0), "worst_constraint": worst,
                                    "worst_constraint_str": str(prob.constraints[worst])[:300] if worst >= 0 else "",
                                    "true_value": out["phi"], "n_constraints": len(viol)},
                       wall_time_s=time.perf_counter() - t0)


def _violation(con) -> float:
    """Max violation of one CVXPY constraint at the current values. ``ExpCone(x, y, z)`` is measured directly as
    ``max(0, y exp(x/y) - z)`` (CVXPY's projection-based residual is only ~1e-8 accurate on the cone's boundary, where
    every true point of the log-form softmax lies)."""
    from cvxpy.constraints import ExpCone

    if isinstance(con, ExpCone):
        x, y, z = (np.asarray(a.value, float).ravel() for a in con.args)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            v = np.where(y > 0, y * np.exp(x / np.where(y > 0, y, 1.0)) - z, np.where(x <= 0, -z, np.inf))
        return float(max(0.0, np.max(v))) if v.size else 0.0
    v = con.violation()
    return float(np.max(v)) if np.size(v) else 0.0


def _primal_violation(prob) -> dict:
    """Max violation of every CVXPY constraint at the returned solution (MOSEK's 'optimal' can carry primal
    violations on badly scaled instances)."""
    worst, idx = 0.0, -1
    for k, con in enumerate(prob.constraints):
        try:
            m = _violation(con)
        except ValueError:  # expression without a value
            continue
        if m > worst:
            worst, idx = m, k
    return {"primal_max_violation": worst,
            "primal_worst_constraint": str(prob.constraints[idx])[:160] if idx >= 0 else ""}


def _safe(prob, X, L: LiftIndex, reps: dict, inst: Instance, rlt: set, diag_a: str, a_hi, s_lo, s_hi, tau_lo, tau_hi,
          tau, cutinfo, pq, nom) -> dict:
    """A-priori bounds of the lifted coordinates for :func:`aimp.safe_bound.safe_dual_bound` (Proposition 16):
    ``X_kk <= D_k`` from the constraints of the relaxation (trace / box diagonal, the interval diagonal of ``a``,
    ``tau``, ``s``, ``p, q``, ``pl, ql`` — all part of ``"diag"``), the ball blocks as groups with budget 1
    (``tr U <= rho^2``), and ``|tau|``, ``p, q`` for the non-lifted scalars."""
    from aimp.safe_bound import safe_dual_bound

    if "diag" not in rlt:
        return {"value_safe": None, "reason": "no diagonal bounds (diag=False)"}
    D = np.zeros(L.size)
    D[0] = 1.0
    groups = []
    for t in inst.tokens:
        rep = reps[t]
        if not rep.r:
            continue
        idx = list(range(L.u(t).start, L.u(t).stop))
        if rep.base == "ball":
            D[idx] = rep.rho ** 2
            groups.append((idx, 1.0))
        elif rep.base == "box":
            D[idx] = rep.h ** 2
        else:
            return {"value_safe": None, "reason": f"base {rep.base}"}
    for k in range(inst.n):
        D[L.a(k)] = 1.0 if diag_a == "unit" else float(a_hi[k]) ** 2
        if L.has(f"s{k}"):
            D[L.x(f"s{k}")] = max(s_lo[k] ** 2, s_hi[k] ** 2)
        for nm in (f"p{k}", f"q{k}"):
            if L.has(nm):
                D[L.x(nm)] = cutinfo["eps"] ** 2
        for nm in (f"pl{k}", f"ql{k}"):
            if L.has(nm):
                D[L.x(nm)] = nom["eps_lin"] ** 2
    if L.has("tau"):
        D[L.x("tau")] = max(tau_lo ** 2, tau_hi ** 2)
    vb = {}
    if not L.has("tau"):
        vb[tau.id] = max(abs(tau_lo), abs(tau_hi))
    if pq is not None:
        vb[pq[0].id] = (0.0, cutinfo["eps"])  # nonneg attribute: no CVXPY multiplier, so the box must carry it
        vb[pq[1].id] = (0.0, cutinfo["eps"])
    try:
        return safe_dual_bound(prob, X, D, groups, vb)
    except Exception as e:  # never fatal: the value then follows the certified-value rule (numerics (b))
        return {"value_safe": None, "reason": repr(e)}
