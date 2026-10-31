"""Set geometry: constructors, affine (base) representation, support functions, projection,
radii, sampling and bounding balls/ellipsoids.

Every set is represented affinely as ``Z = {c + B u : u in U}`` with a *base set* ``U``:

* ``"box"``     — ``|u_k| <= h_k`` (only the coordinates with ``h_k > 0`` are kept);
* ``"ball"``    — ``||u|| <= rho`` (balls: ``B = I``; ellipsoids: ``B = E``);
* ``"simplex"`` — ``u >= 0, sum u = 1`` (hulls: ``c = 0``, ``B = points^T``);
* ``"point"``   — ``u`` has dimension 0 (degenerate sets, e.g. ``rho = 0``).

All formulations lift/optimize ``u``; this is an affine change of variables, under which the
Shor relaxation and the RLT products are invariant.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aimp.types import SetSpec


# ----------------------------------------------------------------------------- constructors
def box(l, u) -> SetSpec:
    l = np.asarray(l, float)
    u = np.asarray(u, float)
    if np.any(u < l):
        raise ValueError("box with u < l")
    return SetSpec(kind="box", center=(l + u) / 2, l=l, u=u)


def box_around(center, half_width) -> SetSpec:
    center = np.asarray(center, float)
    h = np.broadcast_to(np.asarray(half_width, float), center.shape)
    return box(center - h, center + h)


def ball(center, rho: float) -> SetSpec:
    return SetSpec(kind="ball", center=np.asarray(center, float), rho=float(rho))


def ellipsoid(center, E, rho: float) -> SetSpec:
    E = np.asarray(E, float)
    if E.ndim == 1:
        E = E[:, None]
    return SetSpec(kind="ellipsoid", center=np.asarray(center, float), E=E, rho=float(rho))


def hull(points) -> SetSpec:
    P = np.atleast_2d(np.asarray(points, float))
    return SetSpec(kind="hull", center=P.mean(axis=0), points=P)


def point(z) -> SetSpec:
    """A fixed token (ball of radius 0)."""
    return ball(z, 0.0)


# ----------------------------------------------------------------------- base representation
@dataclass
class AffineRep:
    """``Z = {c + B u : u in U}`` with base set ``U`` of kind ``base``."""

    c: np.ndarray  # (d,)
    B: np.ndarray  # (d, r)
    base: str  # box | ball | simplex | point
    h: np.ndarray | None = None  # box half-widths (r,)
    rho: float | None = None  # ball radius

    @property
    def r(self) -> int:
        return self.B.shape[1]


def affine(S: SetSpec) -> AffineRep:
    d = S.center.shape[0]
    if S.kind == "box":
        h = (S.u - S.l) / 2
        mask = h > 0
        if not mask.any():
            return AffineRep(c=S.center.copy(), B=np.zeros((d, 0)), base="point")
        return AffineRep(c=S.center.copy(), B=np.eye(d)[:, mask], base="box", h=h[mask])
    if S.kind in ("ball", "ellipsoid"):
        if S.rho == 0:
            return AffineRep(c=S.center.copy(), B=np.zeros((d, 0)), base="point")
        B = np.eye(d) if S.kind == "ball" else S.E
        return AffineRep(c=S.center.copy(), B=B.copy(), base="ball", rho=float(S.rho))
    if S.kind == "hull":
        if S.points.shape[0] == 1:
            return AffineRep(c=S.points[0].copy(), B=np.zeros((d, 0)), base="point")
        return AffineRep(c=np.zeros(d), B=S.points.T.copy(), base="simplex")
    raise ValueError(S.kind)


def base_project(rep: AffineRep, u: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the base set ``U``."""
    if rep.base == "point":
        return np.zeros(0)
    if rep.base == "box":
        return np.clip(u, -rep.h, rep.h)
    if rep.base == "ball":
        nu = np.linalg.norm(u)
        return u if nu <= rep.rho else u * (rep.rho / nu)
    if rep.base == "simplex":
        return project_simplex(u)
    raise ValueError(rep.base)


def base_sample(rep: AffineRep, rng: np.random.Generator, k: int = 1) -> np.ndarray:
    """``k`` points of the base set (uniform for box/ball, Dirichlet(1) for the simplex); (k, r)."""
    r = rep.r
    if rep.base == "point":
        return np.zeros((k, 0))
    if rep.base == "box":
        return rng.uniform(-1, 1, size=(k, r)) * rep.h
    if rep.base == "ball":
        g = rng.standard_normal((k, r))
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        rad = rep.rho * rng.uniform(0, 1, size=(k, 1)) ** (1.0 / r)
        return g * rad
    if rep.base == "simplex":
        return rng.dirichlet(np.ones(r), size=k)
    raise ValueError(rep.base)


def base_center(rep: AffineRep) -> np.ndarray:
    if rep.base == "simplex":
        return np.full(rep.r, 1.0 / rep.r)
    return np.zeros(rep.r)


def to_z(rep: AffineRep, u: np.ndarray) -> np.ndarray:
    return rep.c + rep.B @ u if rep.r else rep.c.copy()


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex (sort-based)."""
    n = v.shape[0]
    s = np.sort(v)[::-1]
    css = np.cumsum(s) - 1
    k = np.nonzero(s - css / np.arange(1, n + 1) > 0)[0][-1]
    theta = css[k] / (k + 1)
    return np.maximum(v - theta, 0)


# ---------------------------------------------------------------------- support / geometry
def support(S: SetSpec, g: np.ndarray) -> float:
    """``max_{z in S} g^T z`` (exact)."""
    g = np.asarray(g, float)
    if S.kind == "box":
        return float(g @ S.center + np.abs(g) @ ((S.u - S.l) / 2))
    if S.kind == "ball":
        return float(g @ S.center + S.rho * np.linalg.norm(g))
    if S.kind == "ellipsoid":
        return float(g @ S.center + S.rho * np.linalg.norm(S.E.T @ g))
    if S.kind == "hull":
        return float(np.max(S.points @ g))
    raise ValueError(S.kind)


def support_argmax(S: SetSpec, g: np.ndarray) -> np.ndarray:
    """A maximizer of ``g^T z`` over ``S``."""
    g = np.asarray(g, float)
    if S.kind == "box":
        return np.where(g >= 0, S.u, S.l)
    if S.kind == "ball":
        ng = np.linalg.norm(g)
        return S.center + (S.rho * g / ng if ng > 0 else 0)
    if S.kind == "ellipsoid":
        v = S.E.T @ g
        nv = np.linalg.norm(v)
        return S.center + (S.rho * S.E @ v / nv if nv > 0 else 0)
    if S.kind == "hull":
        return S.points[int(np.argmax(S.points @ g))].copy()
    raise ValueError(S.kind)


def coord_bounds(S: SetSpec, A: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Exact componentwise bounds of ``A z`` over ``S`` (``A = I`` if None)."""
    d = S.center.shape[0]
    A = np.eye(d) if A is None else np.asarray(A, float)
    hi = np.array([support(S, a) for a in A])
    lo = np.array([-support(S, -a) for a in A])
    return lo, hi


def circumradius(S: SetSpec) -> float:
    """``max_{z in S} ||z - center||`` (box: ``||u - l||/2``; ellipsoid: ``rho ||E||``; exact)."""
    if S.kind == "box":
        return float(np.linalg.norm(S.u - S.l) / 2)
    if S.kind == "ball":
        return float(S.rho)
    if S.kind == "ellipsoid":
        return float(S.rho * np.linalg.norm(S.E, 2))
    if S.kind == "hull":
        return float(np.max(np.linalg.norm(S.points - S.center, axis=1)))
    raise ValueError(S.kind)


def max_norm(S: SetSpec) -> float:
    """Certified upper bound on ``max_{z in S} ||z||`` (``||center|| + circumradius``)."""
    return float(np.linalg.norm(S.center) + circumradius(S))


def max_dev_norm(S: SetSpec, A: np.ndarray) -> tuple[float, bool]:
    """Upper bound on ``max_{z in S} ||A (z - center)||``; returns (value, is_exact).

    ball ``rho ||A||`` (exact); ellipsoid ``rho ||A E||`` (exact); box ``||A|| rho_box``
    (upper bound); hull: exact maximum over the points.
    """
    if S.kind == "ball":
        return float(S.rho * np.linalg.norm(A, 2)), True
    if S.kind == "ellipsoid":
        return float(S.rho * np.linalg.norm(A @ S.E, 2)), True
    if S.kind == "box":
        h = (S.u - S.l) / 2
        exact = np.count_nonzero(h) <= 1
        return float(np.linalg.norm(A, 2) * np.linalg.norm(h)), bool(exact)
    if S.kind == "hull":
        return float(np.max(np.linalg.norm((S.points - S.center) @ A.T, axis=1))), True
    raise ValueError(S.kind)


def contains(S: SetSpec, z: np.ndarray, tol: float = 1e-9) -> bool:
    z = np.asarray(z, float)
    if S.kind == "box":
        return bool(np.all(z >= S.l - tol) and np.all(z <= S.u + tol))
    if S.kind == "ball":
        return bool(np.linalg.norm(z - S.center) <= S.rho + tol)
    if S.kind == "ellipsoid":
        # least-squares preimage; z must lie in the range of E
        w, *_ = np.linalg.lstsq(S.E, z - S.center, rcond=None)
        resid = np.linalg.norm(S.E @ w - (z - S.center))
        return bool(resid <= tol * (1 + np.linalg.norm(z)) and np.linalg.norm(w) <= S.rho + tol)
    if S.kind == "hull":
        from scipy.optimize import linprog

        m = S.points.shape[0]
        res = linprog(np.zeros(m), A_eq=np.vstack([S.points.T, np.ones(m)]),
                      b_eq=np.concatenate([z, [1.0]]), bounds=[(0, None)] * m, method="highs")
        return bool(res.status == 0)
    raise ValueError(S.kind)


def project(S: SetSpec, z: np.ndarray) -> np.ndarray:
    """Euclidean projection of ``z`` onto ``S``."""
    z = np.asarray(z, float)
    if S.kind == "box":
        return np.clip(z, S.l, S.u)
    if S.kind == "ball":
        dz = z - S.center
        n = np.linalg.norm(dz)
        return z if n <= S.rho else S.center + dz * (S.rho / n)
    if S.kind == "ellipsoid":
        return S.center + S.E @ _trust_region(S.E, z - S.center, S.rho)
    if S.kind == "hull":
        return S.points.T @ _project_hull_weights(S.points, z)
    raise ValueError(S.kind)


def _trust_region(E: np.ndarray, r: np.ndarray, rho: float) -> np.ndarray:
    """``argmin_{||w|| <= rho} ||E w - r||`` via SVD and bisection on the multiplier."""
    U, sv, Vt = np.linalg.svd(E, full_matrices=False)
    b = U.T @ r
    keep = sv > 1e-14 * max(1.0, sv.max(initial=0))
    sv, b, Vt = sv[keep], b[keep], Vt[keep]
    if sv.size == 0:
        return np.zeros(E.shape[1])

    def w_of(lam):
        return Vt.T @ (sv * b / (sv**2 + lam))

    w = w_of(0.0)
    if np.linalg.norm(w) <= rho:
        return w
    lo, hi = 0.0, max(1.0, float(np.max(sv * np.abs(b))) / max(rho, 1e-300))
    while np.linalg.norm(w_of(hi)) > rho:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if np.linalg.norm(w_of(mid)) > rho:
            lo = mid
        else:
            hi = mid
    return w_of(hi)


def _project_hull_weights(P: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Weights ``lam`` on the simplex minimizing ``||P^T lam - z||`` (projected gradient, FISTA)."""
    m = P.shape[0]
    lam = np.full(m, 1.0 / m)
    G = P @ P.T
    L = max(np.linalg.eigvalsh(G).max(), 1e-12)
    y, t = lam.copy(), 1.0
    Pz = P @ z
    for _ in range(5000):
        lam_new = project_simplex(y - (G @ y - Pz) / L)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = lam_new + (t - 1) / t_new * (lam_new - lam)
        if np.linalg.norm(lam_new - lam) < 1e-14:
            lam = lam_new
            break
        lam, t = lam_new, t_new
    return lam


def bounding_ball(S: SetSpec) -> tuple[np.ndarray, float]:
    """A ball containing ``S``: (center, circumradius)."""
    return S.center.copy(), circumradius(S)


def bounding_ellipsoid(S: SetSpec) -> SetSpec:
    """An ellipsoid containing ``S``. Box: the minimum-volume one, ``E = sqrt(r) diag(h)`` on the
    non-degenerate coordinates; ellipsoid/ball: itself; hull: the bounding ball (not the minimum-volume
    Löwner–John ellipsoid)."""
    if S.kind in ("ball", "ellipsoid"):
        return S
    if S.kind == "box":
        h = (S.u - S.l) / 2
        mask = h > 0
        r = int(mask.sum())
        E = np.eye(S.center.shape[0])[:, mask] * h[mask] * np.sqrt(max(r, 1))
        return ellipsoid(S.center, E, 1.0)
    c, rad = bounding_ball(S)
    return ball(c, rad)
