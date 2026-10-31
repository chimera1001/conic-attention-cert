"""The exact worst case of one head with a fixed query (Theorem 7 of the paper), computed with a certified subdivision
of each key's boundary angle in place of the closed form of Theorem 7(iii) (as in Section 7).

Setting: the query token's set is a point; every other key ``u`` varies independently in a ball (or ellipsoid) of its own,
so ``s_u = sbar_u + a_u^T w``, ``g_u = gbar_u + b_u^T w`` (``||w|| <= rho_u``, base coordinates) and the pair ``(s_u, g_u)``
ranges over the planar ellipse ``C_u = (sbar_u, gbar_u) + rho_u L_u (cos t, sin t)`` (boundary; ``L_u L_u^T = [a_u b_u]^T
[a_u b_u]``, a segment when the Gram matrix has rank 1). With ``phi = sum_j e^{s_j} g_j / sum_j e^{s_j}``,

    ``H(lambda) = sum_{fixed j} e^{s_j} (g_j - lambda) + sum_u max_{(s, g) in C_u} e^s (g - lambda)``

is continuous and strictly decreasing and ``p_star = max phi`` is its unique root (Dinkelbach; exact by separability).

**Per-key maximum** ``mu_u(lambda) = max_t f_u(t)``, ``f_u(t) = e^{s(t) - s_ref} (g(t) - lambda)`` (a common shift
``s_ref`` only scales ``H``). Grid of 4 096 angles (cells of width ``2 pi / 4096`` centred on them), the best 3 refined by
Brent; ``mu_lo`` = best evaluated value. Certified upper bound per cell (both valid; the smaller is used):
``f(c) + Lip h/2`` with the global ``Lip = e^{s_max - s_ref} rho ||L|| (|g|_max + |lambda| + 1)``, and the second-order
``f(c) + |f'(c)| h/2 + K_c h^2/8`` with the cell's ``K_c = e^{s(c) + sigma_s h/2 - s_ref} [(sigma_s^2 + sigma_s) G_c +
2 sigma_s sigma_g + sigma_g] >= |f''|`` on the cell (``sigma_s = rho ||L[0]||``, ``sigma_g = rho ||L[1]||``,
``G_c = |g(c) - lambda| + sigma_g h/2``). Cells whose bound exceeds ``mu_lo`` are
refined x4 until ``mu_hi - mu_lo <= min(1e-13 (1 + |mu_lo|), tol_key)`` (the first-order bound alone would need ~1e13
angles) and the per-key tolerance ``tol_key = tol_rel x range / n_keys`` (the per-key widths add up in ``H``; it binds
only for large ``n``). Every computed ``f``
carries a pointwise rounding allowance (subtracted for ``mu_lo``, added to the cell bounds).

**Root.** ``H_lo = sum fixed + sum mu_lo``, ``H_hi`` likewise; the bracket keeps ``H_lo(lam_lo) > 0`` (so ``p_star >
lam_lo``) or ``lam_lo = phi(feasible point)``, and ``H_hi(lam_hi) < 0`` (so ``p_star < lam_hi``). Candidates: the
Dinkelbach point (the value ``phi`` of the current maximizers, a feasible value) and, once it stalls, ``lam_lo + tol/2``
and bisection; stop at ``lam_hi - lam_lo <= 1e-10 x range`` (a safeguarded bisection: the same certificates as a plain
bisection, with ~10x fewer ``H`` evaluations).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar

from aimp import sets as S
from aimp.types import Instance

U_ROUND = np.finfo(float).eps / 2
N0 = 4096
MU_TOL = 1e-13  # mu_hi - mu_lo <= MU_TOL (1 + |mu_lo|)
LAM_TOL = 1e-10  # lam_hi - lam_lo <= LAM_TOL x range
MAX_LEVELS = 24  # x4 refinement levels per key (h = 2 pi / (4096 4^24) ~ 5e-18: far below float resolution)
MAX_CELLS = 1 << 18  # cells alive at one level, per key (~40 MB of arrays); hitting it stops the refinement (still certified)
N_BRENT = 3


@dataclass
class FQProblem:
    """Fixed-query head in planar form (key order kept: ``fixed`` and ``var`` hold key indices)."""

    fixed: list[int]
    s_fix: np.ndarray
    g_fix: np.ndarray
    var: list[int]
    sbar: np.ndarray
    gbar: np.ndarray
    rho: np.ndarray
    L: np.ndarray  # (nu, 2, 2): boundary point = (sbar, gbar) + rho L (cos t, sin t)
    U: list = field(default_factory=list)  # per varying key (r_u, 2): base point w = rho U (cos t, sin t)
    range: float = 1.0
    n: int = 0

    def negated(self) -> "FQProblem":
        """The same head with ``-c`` (its maximum is ``-min phi``)."""
        L = self.L.copy()
        L[:, 1, :] *= -1
        return FQProblem(self.fixed, self.s_fix, -self.g_fix, self.var, self.sbar, -self.gbar, self.rho, L, self.U,
                         self.range, self.n)

    @property
    def g_bounds(self) -> tuple[float, float]:
        sg = self.rho * np.linalg.norm(self.L[:, 1, :], axis=1) if self.var else np.zeros(0)
        lo = [*(self.g_fix.tolist()), *((self.gbar - sg).tolist())]
        hi = [*(self.g_fix.tolist()), *((self.gbar + sg).tolist())]
        return float(min(lo)), float(max(hi))

    @property
    def s_max(self) -> float:
        ss = self.rho * np.linalg.norm(self.L[:, 0, :], axis=1) if self.var else np.zeros(0)
        return float(max([*(self.s_fix.tolist()), *((self.sbar + ss).tolist())]))


def fq_problem(inst: Instance) -> FQProblem:
    """The planar form of a fixed-query instance: the query's set must be a point (``rho = 0``); keys with a point set
    are fixed, the others must be balls / ellipsoids (base ``"ball"``)."""
    M, N, c = inst.head.M, inst.head.N, np.asarray(inst.c, float)
    rq = S.affine(inst.sets[inst.query])
    if rq.r:
        raise ValueError("exact_fq: the query's set must be a point (fixed query)")
    q = rq.c
    gN = N.T @ c
    fixed, s_fix, g_fix, var, sbar, gbar, rho, Ls, Us = [], [], [], [], [], [], [], [], []
    for idx, j in enumerate(inst.keys):
        rep = S.affine(inst.sets[j])
        s0 = float(q @ M @ rep.c)
        g0 = float(gN @ rep.c)
        if not rep.r:
            fixed.append(idx)
            s_fix.append(s0)
            g_fix.append(g0)
            continue
        if rep.base != "ball":
            raise NotImplementedError(f"exact_fq: base {rep.base!r} (boxes / hulls give polygons; not implemented)")
        A = np.stack([rep.B.T @ (M.T @ q), rep.B.T @ gN], axis=1)  # (r, 2): (s - sbar, g - gbar) = A^T w
        Uu, sig, Vt = np.linalg.svd(A, full_matrices=False)
        k = sig.size
        Lm = np.zeros((2, 2))
        Um = np.zeros((A.shape[0], 2))
        Lm[:, :k] = Vt.T * sig  # A^T U(c, s) = V Sigma (c, s)
        Um[:, :k] = Uu
        var.append(idx)
        sbar.append(s0)
        gbar.append(g0)
        rho.append(float(rep.rho))
        Ls.append(Lm)
        Us.append(Um)
    g_all = [*g_fix, *gbar]
    rng_ = float(max(g_all) - min(g_all)) if g_all else 0.0
    return FQProblem(fixed, np.array(s_fix, float), np.array(g_fix, float), var, np.array(sbar, float),
                     np.array(gbar, float), np.array(rho, float), np.array(Ls, float).reshape(-1, 2, 2), Us,
                     rng_ if rng_ > 0 else 1.0, inst.n)


# ------------------------------------------------------------------------------------------------ one key
def _sg(P: FQProblem, u: int, t):
    ct, st = np.cos(t), np.sin(t)
    L = P.L[u]
    r = P.rho[u]
    return P.sbar[u] + r * (L[0, 0] * ct + L[0, 1] * st), P.gbar[u] + r * (L[1, 0] * ct + L[1, 1] * st)


def _f_df(P: FQProblem, u: int, t, lam: float, s_ref: float, with_err: bool = False):
    """``f = e^{s - s_ref} (g - lam)`` and ``f'`` at angles ``t``; ``with_err``: also a bound on the rounding error of
    the computed ``f`` (``32 eps e^{s - s_ref} (1 + |s - s_ref| + |s|) (|g| + |lam| + 1)``: the exponent's absolute error
    and the product's relative error)."""
    ct, st = np.cos(t), np.sin(t)
    L = P.L[u]
    r = P.rho[u]
    s = P.sbar[u] + r * (L[0, 0] * ct + L[0, 1] * st)
    g = P.gbar[u] + r * (L[1, 0] * ct + L[1, 1] * st)
    ds = r * (-L[0, 0] * st + L[0, 1] * ct)
    dg = r * (-L[1, 0] * st + L[1, 1] * ct)
    E = np.exp(s - s_ref)
    f, df = E * (g - lam), E * (ds * (g - lam) + dg)
    if not with_err:
        return f, df
    amp = 32 * U_ROUND * E * (1 + np.abs(s - s_ref) + np.abs(s))
    err = amp * (np.abs(g) + abs(lam) + 1)
    err_df = amp * (np.abs(ds) * (np.abs(g) + abs(lam) + 1) + np.abs(dg) + 1)  # rounding of the computed f'
    return f, df, err, s, g, err_df


def key_max(P: FQProblem, u: int, lam: float, s_ref: float, n0: int = N0, tol: float = MU_TOL,
            max_levels: int = MAX_LEVELS, max_cells: int = MAX_CELLS, tol_abs: float | None = None) -> dict:
    """Certified ``[mu_lo, mu_hi]`` for ``max_t e^{s(t) - s_ref} (g(t) - lam)`` on the boundary of ``C_u`` (module
    docstring), the maximizing angle and the refinement statistics.

    Refinement stops when the *refinable* part of the gap (the cell bounds without the rounding allowance) is below
    ``min(tol (1 + |mu_lo|), tol_abs)`` (``tol_abs``: the per-key tolerance, absolute in the units of ``H``): the
    rounding allowance ``err`` is irreducible (it does not shrink with the cells), so it is added
    to the final bracket but never enters the stopping test — with it inside, ``mu_hi - mu_lo >= 2 err`` can exceed the
    tolerance forever and the surviving cells multiply x4 per level (unbounded memory growth). Hard caps:
    ``max_levels`` levels and ``max_cells`` cells per level (``stop`` records which ended it; the bracket is valid
    either way, only wider)."""
    r = P.rho[u]
    L = P.L[u]
    sig_s = r * float(np.linalg.norm(L[0]))
    sig_g = r * float(np.linalg.norm(L[1]))
    Emax = math.exp(min(P.sbar[u] + sig_s - s_ref, 700.0))
    gmax_abs = abs(P.gbar[u]) + sig_g
    Gl = max(abs(P.gbar[u] + sig_g - lam), abs(P.gbar[u] - sig_g - lam))
    lip = Emax * r * float(np.linalg.norm(L, 2)) * (gmax_abs + abs(lam) + 1.0)  # global Lipschitz constant of f

    def K_cell(s_c, g_c, hw):
        """``>= |f''|`` on the cell ``[c - hw, c + hw]``: ``e^{s - s_ref} <= e^{s(c) - s_ref + sigma_s hw}``,
        ``|g - lam| <= |g(c) - lam| + sigma_g hw`` on the cell (``|s'| <= sigma_s``, ``|g'| <= sigma_g``); capped by the
        global constant."""
        Ec = np.exp(np.minimum(s_c - s_ref + sig_s * hw, 700.0))
        Gc = np.minimum(np.abs(g_c - lam) + sig_g * hw, Gl)
        return np.minimum(Ec, Emax) * ((sig_s ** 2 + sig_s) * Gc + 2 * sig_s * sig_g + sig_g)

    h = 2 * math.pi / n0
    cen = np.arange(n0) * h
    f, df, err, s_c, g_c, edf = _f_df(P, u, cen, lam, s_ref, True)
    k_best = np.argsort(-f)[:N_BRENT]
    t_best = float(cen[k_best[0]])
    mu_lo = float(f[k_best[0]] - err[k_best[0]])  # certified lower bound: a computed value minus its rounding error
    for k in k_best:  # Brent on [c - h, c + h]
        c0 = float(cen[k])
        res = minimize_scalar(lambda t: -float(_f_df(P, u, np.array([t]), lam, s_ref)[0][0]), bounds=(c0 - h, c0 + h),
                              method="bounded", options={"xatol": 1e-14})
        fv, _, ev, _, _, _ = _f_df(P, u, np.array([float(res.x)]), lam, s_ref, True)
        v = float(fv[0] - ev[0])
        if v > mu_lo:
            mu_lo, t_best = v, float(res.x)
    levels = 0
    ncells = n0
    stop = None
    while True:
        ub0 = np.minimum(f + lip * h / 2,  # refinable part (bounds over the whole cell [c - h/2, c + h/2])
                         f + (np.abs(df) + edf) * h / 2 + K_cell(s_c, g_c, h / 2) * h * h / 8)
        ub = ub0 + err  # + the rounding allowance of the computed f
        mu_hi = max(float(ub.max()), mu_lo)
        f_best = float(f.max())
        thr = tol * (1 + abs(mu_lo))
        if tol_abs is not None:
            thr = min(thr, tol_abs)
        if float(ub0.max()) - f_best <= thr:
            stop = "tol"
            break
        if levels >= max_levels:
            stop = "max_levels"
            break
        keep = ub > mu_lo
        if 4 * int(keep.sum()) > max_cells:
            stop = "max_cells"
            break
        cen = cen[keep]
        h /= 4
        cen = (cen[:, None] + np.array([-1.5, -0.5, 0.5, 1.5])[None, :] * h).ravel()
        f, df, err, s_c, g_c, edf = _f_df(P, u, cen, lam, s_ref, True)
        k = int(np.argmax(f - err))
        if f[k] - err[k] > mu_lo:
            mu_lo, t_best = float(f[k] - err[k]), float(cen[k])
        levels += 1
        ncells += cen.size
    return {"mu_lo": mu_lo, "mu_hi": mu_hi, "t": t_best, "levels": levels, "cells": ncells, "stop": stop,
            "converged": stop == "tol"}


# ------------------------------------------------------------------------------------------------ H and the root
def H_eval(P: FQProblem, lam: float, s_ref: float, max_levels: int = MAX_LEVELS, max_cells: int = MAX_CELLS,
           tol_key: float | None = None) -> dict:
    """``H_lo``, ``H_hi`` at ``lam`` (shift ``s_ref``), the maximizing angles and the value ``phi`` of the maximizers
    (a feasible point: its value is a lower bound on ``p_star``). ``capped``: keys whose refinement hit a cap (their
    ``[mu_lo, mu_hi]`` is still a valid enclosure, only wider)."""
    Ef = np.exp(P.s_fix - s_ref)
    fix = Ef * (P.g_fix - lam)
    rnd = 32 * U_ROUND * float(np.sum(Ef * (1 + np.abs(P.s_fix - s_ref) + np.abs(P.s_fix)) * (np.abs(P.g_fix) + abs(lam) + 1)))
    lo = float(fix.sum()) - rnd
    hi = float(fix.sum()) + rnd
    ts, levels, capped = [], 0, 0
    for u in range(len(P.var)):
        km = key_max(P, u, lam, s_ref, max_levels=max_levels, max_cells=max_cells, tol_abs=tol_key)
        lo += km["mu_lo"]
        hi += km["mu_hi"]
        ts.append(km["t"])
        levels = max(levels, km["levels"])
        capped += km["stop"] != "tol"
    ts = np.array(ts, float)
    val = point_value(P, ts)
    return {"lam": lam, "H_lo": lo, "H_hi": hi, "t": ts, "value": val, "levels": levels, "capped": int(capped),
            "converged": capped == 0}


def point_value(P: FQProblem, ts: np.ndarray) -> float:
    """``phi`` at the configuration with the varying keys at angles ``ts`` (planar form; stable softmax)."""
    s, g = [*P.s_fix.tolist()], [*P.g_fix.tolist()]
    for u, t in enumerate(ts):
        a, b = _sg(P, u, float(t))
        s.append(float(a))
        g.append(float(b))
    s, g = np.array(s), np.array(g)
    e = np.exp(s - s.max())
    return float(e @ g / e.sum())


def _shift(P: FQProblem, ts: np.ndarray | None) -> float:
    """``log sum_j e^{s_j}`` at the current configuration (centres when ``ts`` is None), capped so that no key's
    ``e^{s - s_ref}`` overflows."""
    s = [*P.s_fix.tolist()]
    for u in range(len(P.var)):
        s.append(float(_sg(P, u, float(ts[u]))[0]) if ts is not None else float(P.sbar[u]))
    s = np.array(s)
    m = float(s.max())
    lse = m + math.log(float(np.exp(s - m).sum()))
    return max(lse, P.s_max - 600.0)


def solve_max(P: FQProblem, tol_rel: float = LAM_TOL, max_evals: int = 100, max_levels: int = MAX_LEVELS,
              max_cells: int = MAX_CELLS, per_key_tol: bool = True) -> dict:
    """A certified bracket ``p_lo <= p_star = max phi <= p_hi``, the maximizing angles of the best feasible point and its
    value. Every end is certified when it is set: ``p_lo`` by ``H_lo(p_lo) > 0`` (``lo_cert = "H_lo>0"``) or by a feasible
    input attaining ``p_lo`` up to a rounding allowance (``"feasible"``: Dinkelbach values; the input is returned), ``p_hi``
    by ``H_hi(p_hi) < 0`` (``"H_hi<0"``) or ``p_hi = max_j g_j^max`` (``"g_max"``, phi is an average). ``status = "exact"``
    only if ``p_hi - p_lo <= tol_rel x range`` **and** no per-key refinement hit a cap; otherwise ``"unresolved"`` with the
    (wider, still certified) bracket and ``width_rel`` — nothing narrower is ever returned.

    Candidates: Dinkelbach (``lam <- phi(maximizers at lam)``) while it moves by more than ``tol/4``; then ``lam_lo +
    2^k tol/2`` (``H_hi < 0`` there closes the bracket); bisection otherwise. ``per_key_tol``: each key's
    refinement also stops only at ``tol_rel x range / n_keys`` (absolute, units of ``H``; ``False``: only the relative
    per-key tolerance ``MU_TOL``)."""
    t0 = time.perf_counter()
    tol = tol_rel * P.range
    tol_key = tol / max(1, P.n) if per_key_tol else None
    g_lo, g_hi = P.g_bounds
    s0 = np.r_[P.s_fix, P.sbar]
    e0 = np.exp(s0 - s0.max())
    v_center = float(e0 @ np.r_[P.g_fix, P.gbar] / e0.sum())
    gabs = float(max(abs(g_lo), abs(g_hi)))

    def feas(v: float) -> float:  # a computed phi at a feasible input, minus its rounding allowance
        return v - 64 * U_ROUND * (abs(v) + gabs)

    if not P.var:  # nothing varies: phi is a constant
        return {"p_lo": v_center, "p_hi": v_center, "lo_cert": "constant", "hi_cert": "constant", "t": np.zeros(0),
                "value": v_center, "evals": 0, "stop": "constant", "status": "exact", "capped_evals": 0,
                "width_rel": 0.0, "wall_s": time.perf_counter() - t0, "trace": []}
    lam_lo, lo_cert = feas(v_center), "feasible"  # p_star >= phi(centres)
    lam_hi, hi_cert = g_hi + 1e-9 * P.range, "g_max"  # phi is an average of the values
    best_v, best_t = v_center, None
    s_ref = _shift(P, None)
    trace, evals, stop = [], 0, None
    phase, cand = "dinkelbach", v_center
    widen = 0  # after an ambiguous evaluation: try lam_lo + 2^widen tol / 2
    capped_evals = 0
    while True:
        if lam_hi - lam_lo <= tol:
            stop = "tol"
            break
        if evals >= max_evals:
            stop = "max_evals"
            break
        if phase == "close":
            cand = lam_lo + (2 ** widen) * tol / 2
        elif phase == "bisect" or not (lam_lo <= cand < lam_hi):
            cand = 0.5 * (lam_lo + lam_hi)
        ev = H_eval(P, cand, s_ref, max_levels, max_cells, tol_key)
        evals += 1
        capped_evals += ev["capped"] > 0
        trace.append({"phase": phase, **{k: ev[k] for k in ("lam", "H_lo", "H_hi", "value", "levels", "capped")}})
        if ev["value"] > best_v:
            best_v, best_t = ev["value"], ev["t"]
        if feas(ev["value"]) > lam_lo:
            lam_lo, lo_cert = feas(ev["value"]), "feasible"
        if ev["H_lo"] > 0:  # p_star > cand
            if cand > lam_lo:
                lam_lo, lo_cert = cand, "H_lo>0"
            step = ev["value"] - cand  # Dinkelbach progress
            s_ref = _shift(P, ev["t"])
            if phase == "dinkelbach" and step > tol / 4:
                cand = ev["value"]
            elif phase == "dinkelbach":
                phase = "close"
            elif phase == "close":
                phase, cand = "dinkelbach", ev["value"]
        elif ev["H_hi"] < 0:  # p_star < cand
            lam_hi, hi_cert = min(lam_hi, cand), "H_hi<0"
            phase = "close" if phase == "dinkelbach" else "bisect"
        else:  # H_lo <= 0 <= H_hi: cand is within the per-key resolution of the root
            if phase == "dinkelbach":
                phase = "close"
            elif phase == "close" and widen < 30 and lam_lo + (2 ** (widen + 1)) * tol / 2 < lam_hi:
                widen += 1  # step further above lam_lo until H_hi < 0 certifies an upper end
            else:
                stop = "resolution"
                break
    status = "exact" if (stop == "tol" and capped_evals == 0) else "unresolved"
    return {"p_lo": float(lam_lo), "p_hi": float(lam_hi), "lo_cert": lo_cert, "hi_cert": hi_cert,
            "t": best_t, "value": float(best_v), "evals": evals, "stop": stop, "status": status,
            "capped_evals": int(capped_evals), "width_rel": float((lam_hi - lam_lo) / P.range), "tol_key": tol_key,
            "wall_s": time.perf_counter() - t0, "trace": trace}


def worst_input(P: FQProblem, inst: Instance, ts: np.ndarray) -> np.ndarray:
    """The input ``Z`` (rows = tokens) of the configuration with the varying keys at angles ``ts``."""
    Z = inst.centers().copy()
    for u, idx in enumerate(P.var):
        j = inst.keys[idx]
        rep = S.affine(inst.sets[j])
        w = P.rho[u] * (P.U[u] @ np.array([math.cos(ts[u]), math.sin(ts[u])]))
        Z[j] = S.to_z(rep, w)
    return Z


def exact_fq(inst: Instance, tol_rel: float = LAM_TOL, max_levels: int = MAX_LEVELS, max_cells: int = MAX_CELLS,
             per_key_tol: bool = True) -> dict:
    """Theorem 7 on a fixed-query instance: the certified brackets of ``max phi`` and of ``min phi`` (``-c``), the
    worst-case inputs and their forward-pass values (``aimp.heads.forward``), which must lie in the brackets (to 1e-12
    relative). ``status = "exact"`` iff both brackets are exact (:func:`solve_max`); else ``"unresolved"`` (the brackets
    are still certified, only wider: ``width_rel_max`` / ``width_rel_min``)."""
    from aimp.heads import forward

    t0 = time.perf_counter()
    P = fq_problem(inst)
    mx = solve_max(P, tol_rel, max_levels=max_levels, max_cells=max_cells, per_key_tol=per_key_tol)
    mn = solve_max(P.negated(), tol_rel, max_levels=max_levels, max_cells=max_cells, per_key_tol=per_key_tol)
    Zmax = worst_input(P, inst, mx["t"]) if mx["t"] is not None else inst.centers()
    Zmin = worst_input(P, inst, mn["t"]) if mn["t"] is not None else inst.centers()
    vmax = forward(inst, Zmax)["phi"]
    vmin = forward(inst, Zmin)["phi"]
    tolv = 1e-12 * max(1.0, P.range, abs(mx["p_hi"]), abs(mn["p_hi"]))
    status = "exact" if (mx["status"] == "exact" and mn["status"] == "exact") else "unresolved"
    return {"p_fq_lo": mx["p_lo"], "p_fq_hi": mx["p_hi"], "min_fq_lo": -mn["p_hi"], "min_fq_hi": -mn["p_lo"],
            "status": status, "status_max": mx["status"], "status_min": mn["status"],
            "value_max_input": vmax, "value_min_input": vmin, "planar_value_max": mx["value"],
            "planar_value_min": -mn["value"], "t_max": mx["t"], "t_min": mn["t"], "Z_max": Zmax, "Z_min": Zmin,
            "max_input_in_bracket": bool(mx["p_lo"] - tolv <= vmax <= mx["p_hi"] + tolv),
            "min_input_in_bracket": bool(-mn["p_hi"] - tolv <= vmin <= -mn["p_lo"] + tolv),
            "evals_max": mx["evals"], "evals_min": mn["evals"], "stop_max": mx["stop"], "stop_min": mn["stop"],
            "capped_evals_max": mx["capped_evals"], "capped_evals_min": mn["capped_evals"],
            "width_rel_max": mx["width_rel"], "width_rel_min": mn["width_rel"], "lo_cert_max": mx["lo_cert"],
            "hi_cert_max": mx["hi_cert"], "lo_cert_min": mn["lo_cert"], "hi_cert_min": mn["hi_cert"],
            "range_planar": P.range, "n_var": len(P.var), "n_fixed": len(P.fixed),
            "trace_max": mx["trace"], "trace_min": mn["trace"], "wall_s": time.perf_counter() - t0}


def fix_query(inst: Instance) -> Instance:
    """The instance with the query token's set replaced by its centre (``Z_i = {zbar_i}``; for a query that is
    a key, key ``i`` and its value are fixed too). Ellipsoids keep their ``E`` (radius 0)."""
    st = inst.sets[inst.query]
    pt = S.ellipsoid(st.center, st.E, 0.0) if st.kind == "ellipsoid" else S.point(st.center)
    sets = dict(inst.sets)
    sets[inst.query] = pt
    return Instance(head=inst.head, keys=list(inst.keys), query=inst.query, sets=sets, c=np.asarray(inst.c, float),
                    meta={**inst.meta, "fixed_query": True})
