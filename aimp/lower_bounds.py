"""Multistart lower bound on ``p_star``: local ascent from several feasible starts.

Each token is parametrized in its base coordinates (sets.AffineRep): boxes by ``u`` with
L-BFGS-B bounds ``[-h, h]``; balls/ellipsoids by an unconstrained ``p`` mapped radially,
``u = p * min(1, rho/||p||)`` (the projection onto the ball, so every iterate is feasible);
hulls by ``u = softmax(p)``. Gradients are analytic (heads.grad_phi + the chain rule).
Every returned value is a numpy re-evaluation of the true head at a feasible point.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from aimp import sets as S
from aimp.heads import grad_phi, phi, softmax
from aimp.types import Instance, SolveResult


class _Param:
    def __init__(self, inst: Instance):
        self.inst = inst
        self.reps = {t: S.affine(inst.sets[t]) for t in range(inst.T)}
        self.free = [t for t in inst.tokens if self.reps[t].r > 0]
        self.sl, off = {}, 0
        for t in self.free:
            self.sl[t] = slice(off, off + self.reps[t].r)
            off += self.reps[t].r
        self.dim = off

    def bounds(self):
        b = []
        for t in self.free:
            rep = self.reps[t]
            if rep.base == "box":
                b += list(zip(-rep.h, rep.h))
            else:
                b += [(None, None)] * rep.r
        return b

    def u_of(self, t, p):
        rep = self.reps[t]
        x = p[self.sl[t]]
        if rep.base == "box":
            return np.clip(x, -rep.h, rep.h)
        if rep.base == "ball":
            nx = np.linalg.norm(x)
            return x if nx <= rep.rho else x * (rep.rho / nx)
        if rep.base == "simplex":
            return softmax(x)
        raise ValueError(rep.base)

    def Z_of(self, p):
        Z = self.inst.centers().copy()
        for t in self.inst.tokens:
            rep = self.reps[t]
            Z[t] = S.to_z(rep, self.u_of(t, p)) if rep.r else rep.c
        return Z

    def p_of_u(self, t, u):
        rep = self.reps[t]
        if rep.base == "simplex":
            return np.log(np.maximum(u, 1e-12))
        return u

    def p_of_Z(self, Z):
        """Base coordinates of a feasible ``Z`` (least squares; projected onto the base set)."""
        p = np.zeros(self.dim)
        for t in self.free:
            rep = self.reps[t]
            u, *_ = np.linalg.lstsq(rep.B, Z[t] - rep.c, rcond=None)
            p[self.sl[t]] = self.p_of_u(t, S.base_project(rep, u))
        return p

    def fun_grad(self, p):
        Z = self.Z_of(p)
        val, G = grad_phi(self.inst, Z)
        g = np.zeros(self.dim)
        for t in self.free:
            rep = self.reps[t]
            gu = rep.B.T @ G[t]
            x = p[self.sl[t]]
            if rep.base == "box":
                gp_ = gu * (np.abs(x) <= rep.h)
            elif rep.base == "ball":
                nx = np.linalg.norm(x)
                if nx <= rep.rho:
                    gp_ = gu
                else:
                    xh = x / nx
                    gp_ = (rep.rho / nx) * (gu - xh * (xh @ gu))
            else:  # softmax Jacobian
                s = softmax(x)
                gp_ = s * (gu - s @ gu)
            g[self.sl[t]] = gp_
        return -val, -g


def multistart_lb(inst: Instance, n_starts: int = 50, seed: int = 0,
                  extra_starts: list[np.ndarray] | None = None, maxiter: int = 500) -> SolveResult:
    """Best of ``n_starts`` random feasible starts + the centers + ``extra_starts`` (Z arrays)."""
    t0 = time.perf_counter()
    P = _Param(inst)
    rng = np.random.default_rng(seed)
    starts = [np.zeros(P.dim) if not any(P.reps[t].base == "simplex" for t in P.free)
              else P.p_of_Z(inst.centers())]
    for Z0 in extra_starts or []:
        starts.append(P.p_of_Z(Z0))
    for _ in range(n_starts):
        p = np.zeros(P.dim)
        for t in P.free:
            p[P.sl[t]] = P.p_of_u(t, S.base_sample(P.reps[t], rng, 1)[0])
        starts.append(p)
    best_val, best_Z, n_ok = -np.inf, inst.centers(), 0
    vals = []
    if P.dim == 0:
        best_val = phi(inst, best_Z)
        vals = [best_val]
    else:
        bnds = P.bounds()
        for p0 in starts:
            res = minimize(P.fun_grad, p0, jac=True, method="L-BFGS-B", bounds=bnds,
                           options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-12})
            Z = P.Z_of(res.x)
            v = phi(inst, Z)  # exact re-evaluation at a feasible point
            vals.append(v)
            n_ok += bool(res.success)
            if v > best_val:
                best_val, best_Z = v, Z
    return SolveResult(
        method="multistart", status="optimal", value=float(best_val),
        value_lb=float(best_val), value_lb_verified=float(best_val),
        recovered_point=best_Z, recovered_value=float(best_val),
        diagnostics={"n_starts_total": len(starts), "n_converged": n_ok,
                     "top_values": sorted(vals, reverse=True)[:5]},
        params={"n_starts": n_starts, "seed": seed, "maxiter": maxiter},
        wall_time_s=time.perf_counter() - t0,
    )
