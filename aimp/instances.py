"""Seeded instance generators: the synthetic head families of the paper's experiments (Section 7) and trivial
sanity-check instances."""
from __future__ import annotations

import numpy as np

from aimp import sets as S
from aimp.types import HeadSpec, Instance


def random_head(d: int, d_k: int, d_v: int, temp: float, rng: np.random.Generator, name="random") -> HeadSpec:
    """Gaussian ``W_Q, W_K, W_V, W_O`` scaled so that ``||M|| = temp`` and ``||N|| = 1``."""
    W_Q = rng.standard_normal((d, d_k))
    W_K = rng.standard_normal((d, d_k))
    W_V = rng.standard_normal((d, d_v))
    W_O = rng.standard_normal((d_v, d))
    h0 = HeadSpec.from_weights(W_Q, W_K, W_V, W_O)
    if temp > 0:
        f = np.sqrt(temp / np.linalg.norm(h0.M, 2))
        W_Q, W_K = W_Q * f, W_K * f
    else:
        W_Q = np.zeros_like(W_Q)
    W_V = W_V / np.linalg.norm(h0.N, 2)
    return HeadSpec.from_weights(W_Q, W_K, W_V, W_O, name=name)


def _make_sets(Zbar: np.ndarray, rho: float, set_type: str) -> dict:
    d = Zbar.shape[1]
    out = {}
    for t, z in enumerate(Zbar):
        if set_type == "ball":
            out[t] = S.ball(z, rho)
        elif set_type == "box":
            out[t] = S.box_around(z, rho / np.sqrt(d))  # circumradius ||u - l||/2 = rho
        else:
            raise ValueError(set_type)
    return out


def f_random(seed: int, d: int = 8, d_k: int = 4, d_v: int = 4, n: int = 3, temp: float = 1.0,
             rho: float = 0.1, set_type: str = "ball", query_in_keys: bool = True, phase: int = 0) -> Instance:
    """F_random: Gaussian head (``||M|| = temp``, ``||N|| = 1``), tokens with ``||zbar|| = sqrt(d)``,
    ``c`` a random unit vector. Query = key 0 if ``query_in_keys`` else an extra token ``n``."""
    rng = np.random.default_rng(seed)
    head = random_head(d, d_k, d_v, temp, rng)
    T = n if query_in_keys else n + 1
    Zbar = rng.standard_normal((T, d))
    Zbar *= np.sqrt(d) / np.linalg.norm(Zbar, axis=1, keepdims=True)
    c = rng.standard_normal(d)
    c /= np.linalg.norm(c)
    keys = list(range(n))
    query = 0 if query_in_keys else n
    iid = f"p{phase}_F_random_{set_type}_d{d}_n{n}_t{temp}_q{int(query_in_keys)}_s{seed}_rho{rho}"
    return Instance(head=head, keys=keys, query=query, sets=_make_sets(Zbar, rho, set_type), c=c,
                    meta={"instance_id": iid, "phase": phase, "family": "F_random", "seed": seed,
                          "d": d, "d_k": d_k, "d_v": d_v, "n": n, "temp": temp, "rho": rho,
                          "set_type": set_type, "query_in_keys": query_in_keys})


def prop9(kappa: float, nu_norm: float = 2.0, d: int = 3) -> Instance:
    """Two-key instance of the type of the sharpness instances of Proposition 10 (one key raises its score while it
    lowers its value along ``c``): ``M = kappa e1 e2^T``, ``N = ||nu|| e1 e2^T``, query fixed at ``e1``
    (not a key), key 1 the segment ``e3 + t e2`` (ellipsoid ``E = e2``, ``rho = 1``), key 2 fixed at
    ``e3``; ``c = -nu/||nu|| = -e1``. Then ``s_1 = kappa t``, ``s_2 = 0``, ``N z_1 = t nu``, ``N z_2 = 0``."""
    e = np.eye(d)
    M = kappa * np.outer(e[0], e[1])
    N = nu_norm * np.outer(e[0], e[1])
    head = HeadSpec(M=M, N=N, d=d, d_k=1, d_v=1, name=f"prop9_k{kappa}")
    sets = {0: S.ellipsoid(e[2], e[1], 1.0), 1: S.point(e[2]), 2: S.point(e[0])}
    return Instance(head=head, keys=[0, 1], query=2, sets=sets, c=-e[0],
                    meta={"instance_id": f"p0_prop9_k{kappa}", "phase": 0, "family": "F_prop9",
                          "kappa": kappa, "nu_norm": nu_norm, "d": d, "n": 2, "seed": None,
                          "set_type": "ellipsoid", "query_in_keys": False})


def trivial(kind: str, seed: int = 0, d: int = 4, n: int = 3, set_type: str = "ball",
            query_in_keys: bool = True) -> Instance:
    """Trivial sanity-check instances: ``rho0`` (all radii 0), ``n1`` (single key), ``M0`` (``M = 0``)."""
    if kind == "rho0":
        inst = f_random(seed, d=d, d_k=2, d_v=2, n=n, rho=0.0, set_type=set_type, query_in_keys=query_in_keys)
    elif kind == "n1":
        inst = f_random(seed, d=d, d_k=2, d_v=2, n=1, rho=0.5, set_type=set_type, query_in_keys=query_in_keys)
    elif kind == "M0":
        inst = f_random(seed, d=d, d_k=2, d_v=2, n=n, temp=0.0, rho=0.5, set_type=set_type,
                        query_in_keys=query_in_keys)
    else:
        raise ValueError(kind)
    inst.meta["family"] = f"trivial_{kind}"
    inst.meta["instance_id"] = f"p0_trivial_{kind}_{set_type}_d{d}_n{inst.n}_q{int(query_in_keys)}_s{seed}"
    return inst


# ============================================================================ synthetic families (Section 7)
FAMILIES = ("F_prop9", "F_random", "F_ambiguous", "F_margin", "F_rank1")


def _normalized(rng, T, d):
    Z = rng.standard_normal((T, d))
    return Z * (np.sqrt(d) / np.linalg.norm(Z, axis=1, keepdims=True))


def _unit(rng, d):
    c = rng.standard_normal(d)
    return c / np.linalg.norm(c)


def _p1_meta(family, set_type, d, d_k, d_v, n, cfg_key, cfg_val, qk, seed, rho, temp, **extra):
    iid = f"p1_{family}_{set_type}_d{d}_n{n}_{cfg_key}{cfg_val}_q{int(qk)}_s{seed}_rho{rho}"
    return {"instance_id": iid, "phase": 1, "family": family, "seed": seed, "d": d, "d_k": d_k, "d_v": d_v,
            "n": n, "temp": temp, "rho": rho, "set_type": set_type, "query_in_keys": qk,
            "config": f"{cfg_key}{cfg_val}", **extra}


def _null_projector(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Orthogonal projector onto null(A)."""
    _, sv, Vt = np.linalg.svd(A)
    rank = int(np.sum(sv > tol * max(1.0, sv.max(initial=0))))
    V0 = Vt[rank:].T
    return V0 @ V0.T


def p1_random(seed, n, rho, set_type, query_in_keys, temp=1.0, d=8, d_k=4, d_v=4, family="F_random"):
    """F_random: Gaussian head with ``||M|| = temp``, ``||N|| = 1``, ``||zbar_j|| = sqrt(d)``, random unit
    ``c``. F_rank1 is the same with ``d_k = 1``."""
    inst = f_random(seed, d=d, d_k=d_k, d_v=d_v, n=n, temp=temp, rho=rho, set_type=set_type,
                    query_in_keys=query_in_keys, phase=1)
    inst.meta = _p1_meta(family, set_type, d, d_k, d_v, n, "t", temp, query_in_keys, seed, rho, temp)
    return inst


def p1_margin(seed, n, rho, set_type, query_in_keys, gamma0=2.0, d=8, d_k=4, d_v=4, temp=1.0):
    """F_margin: F_random, then ``zbar_{j*}`` moved along ``g = M^T zbar_i`` so that
    ``sbar_{j*} - max_{j != j*} sbar_j = gamma0`` (nominal margin). ``j*`` is the first key that is not the query
    (so the move changes only ``sbar_{j*}``)."""
    inst = f_random(seed, d=d, d_k=d_k, d_v=d_v, n=n, temp=temp, rho=rho, set_type=set_type,
                    query_in_keys=query_in_keys, phase=1)
    M = inst.head.M
    Zbar = inst.centers()
    i = inst.query
    j_star = next(j for j in inst.keys if j != i)
    g = M.T @ Zbar[i]
    sbar = np.array([Zbar[i] @ M @ Zbar[j] for j in inst.keys])
    k_star = inst.keys.index(j_star)
    target = np.max(np.delete(sbar, k_star)) + gamma0
    Zbar[j_star] = Zbar[j_star] + (target - sbar[k_star]) / (g @ g) * g
    inst.sets = _make_sets(Zbar, rho, set_type)
    sbar_new = np.array([Zbar[i] @ M @ Zbar[j] for j in inst.keys])
    inst.meta = _p1_meta("F_margin", set_type, d, d_k, d_v, n, "g", gamma0, query_in_keys, seed, rho, temp,
                         j_star=j_star,
                         nominal_margin=float(sbar_new[k_star] - np.max(np.delete(sbar_new, k_star))),
                         zbar_jstar_norm=float(np.linalg.norm(Zbar[j_star])))
    return inst


def p1_ambiguous(seed, n, rho, set_type, query_in_keys, sep=1.0, d=8, d_k=4, d_v=4, temp=1.0, drop=4.0):
    """F_ambiguous (Conjecture 3): keys A=0, B=1 with ``sbar_A = sbar_B`` and value centers ``N zbar_B =
    (1 - 2 sep) N zbar_A`` (``sep = 1``: antipodal, ``+-N zbar_A``, so the pair spans the value set and
    ``||N(zbar_A - zbar_B)|| ~ 2 rad``); the remaining keys have value center at the pair's midpoint and scores
    ``sbar_A - drop - U(0,1)``; ``c`` is aligned with ``N(zbar_A - zbar_B)``.

    Construction (moves inside null(N) change scores but not values): ``g = M^T zbar_i``, ``P0`` = projector onto
    ``null(N)``, ``gn = P0 g``. ``zbar_B = zbar_A - 2 sep (I - P0) zbar_A + beta gn`` with ``beta`` solving
    ``sbar_B = sbar_A``; other keys ``(zbar_A + zbar_B)/2 + gamma_k gn + r_k``, ``r_k`` in null(N) orthogonal to ``gn``
    (score- and value-neutral), scaled towards ``||zbar_k|| = sqrt(d)``. If the query is a key it is key A (so
    ``sbar_A`` is the self score)."""
    rng = np.random.default_rng(seed)
    head = random_head(d, d_k, d_v, temp, rng)
    M, N = head.M, head.N
    T = n if query_in_keys else n + 1
    Z = _normalized(rng, T, d)
    i = 0 if query_in_keys else n
    zi, zA = Z[i], Z[0]
    P0 = _null_projector(N)
    g = M.T @ zi
    gn = P0 @ g
    gg = float(gn @ gn)
    if gg < 1e-12:
        raise ValueError("score gradient has no null(N) component")
    sA = float(zi @ M @ zA)
    base = zA - 2 * sep * ((np.eye(d) - P0) @ zA)
    zB = base + (sA - g @ base) / gg * gn  # g . gn = gn . gn
    Z[1] = zB
    mid = (zA + zB) / 2
    for k in range(2, n):
        target = sA - drop - rng.uniform(0, 1)
        core = mid + (target - g @ mid) / gg * gn
        r = P0 @ rng.standard_normal(d)
        r -= (r @ gn) / gg * gn
        need = d - float(core @ core)
        Z[k] = core + r / max(np.linalg.norm(r), 1e-300) * np.sqrt(max(need, 0.0))
    v = N @ (zA - zB)
    c = v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-12 else _unit(rng, d)
    inst = Instance(head=head, keys=list(range(n)), query=i, sets=_make_sets(Z, rho, set_type), c=c, meta={})
    sbar = np.array([Z[i] @ M @ Z[j] for j in inst.keys])
    inst.meta = _p1_meta("F_ambiguous", set_type, d, d_k, d_v, n, "sep", sep, query_in_keys, seed, rho, temp,
                         value_separation=float(np.linalg.norm(v)), sbar=sbar.tolist(),
                         score_tie=float(abs(sbar[0] - sbar[1])),
                         others_drop=float(sbar[0] - sbar[2:].max()) if n > 2 else None)
    return inst


def p1_prop9(seed, rho, set_type, d=8, kappa_M=1.0, nu_norm=1.0, n=2, query_in_keys=False):
    """F_prop9: the two-key instance of :func:`prop9` (of the type of Proposition 10) embedded in ``R^d`` by a seeded
    random rotation ``Q``: ``M = kappa_M Q e1 e2^T Q^T``, ``N = nu Q e1 e2^T Q^T``, query fixed at ``Q e1``, key 2 fixed
    at ``Q e3``, key 1 the ball (or the cube of half-width ``rho/sqrt(d)``) of radius ``rho`` around ``Q e3``,
    ``c = -Q e1``. Score ``s_1 = kappa_M (Q e2)^T u``, so for balls ``Delta = 2 kappa_M rho`` (the effective score slope
    ``kappa_M rho`` follows the rho sweep). ``n = 2`` only."""
    if n != 2 or query_in_keys:
        raise ValueError("F_prop9 is defined for n = 2 with a fixed query not in J")
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    e = Q.T  # rows e[k] = Q e_k
    M = kappa_M * np.outer(e[0], e[1])
    N = nu_norm * np.outer(e[0], e[1])
    head = HeadSpec(M=M, N=N, d=d, d_k=1, d_v=1, name="prop9_embedded")
    if set_type == "ball":
        k1 = S.ball(e[2], rho)
    elif set_type == "box":
        k1 = S.box_around(e[2], rho / np.sqrt(d))
    else:
        raise ValueError(set_type)
    sets = {0: k1, 1: S.point(e[2]), 2: S.point(e[0])}
    inst = Instance(head=head, keys=[0, 1], query=2, sets=sets, c=-e[0], meta={})
    inst.meta = _p1_meta("F_prop9", set_type, d, 1, 1, 2, "k", kappa_M, False, seed, rho, kappa_M,
                         kappa_M=kappa_M, nu_norm=nu_norm)
    return inst


def p1_cov(seed: int, n: int, kappa: float, anti: bool = False, rho: float = 1.0, d: int = 8,
           nu0: float = 1.0) -> Instance:
    """F_cov / F_cov_anti: score-value covariance families (values aligned / anti-aligned with the scores). Query fixed
    at ``q`` (``||q|| = sqrt(d)``, token ``n``, not a key); unit ``w``, unit ``c``; ``M = kappa q w^T / ||q||^2`` (so
    ``s_j = kappa w^T z_j``) and ``N = +-nu0 c w^T / ||c||^2`` (so ``c^T N z_j = +-nu0 w^T z_j``: value shift = +-score
    shift / kappa, ``x_l = +-y_l``). Key centers ``zbar_j`` orthogonal to ``w`` with ``||zbar_j|| = sqrt(d)`` (equal
    nominal scores 0, uniform ``ahat``), ball sets of radius ``rho``. ``phi = +-sum_j sigma_j(kappa t) t_j`` with
    ``t_j = w^T u_j in [-rho, rho]``."""
    rng = np.random.default_rng(seed)
    q = _normalized(rng, 1, d)[0]
    w = _unit(rng, d)
    c = _unit(rng, d)
    M = kappa * np.outer(q, w) / (q @ q)
    sign = -1.0 if anti else 1.0
    N = sign * nu0 * np.outer(c, w) / (c @ c)
    head = HeadSpec(M=M, N=N, d=d, d_k=1, d_v=1, name="F_cov_anti" if anti else "F_cov")
    Z = rng.standard_normal((n, d))
    Z -= np.outer(Z @ w, w)
    Z *= np.sqrt(d) / np.linalg.norm(Z, axis=1, keepdims=True)
    sets = {j: S.ball(Z[j], rho) for j in range(n)}
    sets[n] = S.point(q)
    fam = "F_cov_anti" if anti else "F_cov"
    inst = Instance(head=head, keys=list(range(n)), query=n, sets=sets, c=c, meta={})
    inst.meta = _p1_meta(fam, "ball", d, 1, 1, n, "k", kappa, False, seed, rho, kappa, kappa=kappa, nu0=nu0,
                         w=w.tolist(), sbar=[float(q @ M @ Z[j]) for j in range(n)])
    return inst


def phase1_instance(family: str, seed: int, n: int, rho: float, set_type: str, query_in_keys: bool,
                    cfg: float) -> Instance:
    """Dispatch on the family; ``cfg`` = temp (F_random, F_rank1), gamma0 (F_margin), sep (F_ambiguous),
    kappa_M (F_prop9), kappa (F_cov, F_cov_anti)."""
    if family == "F_random":
        return p1_random(seed, n, rho, set_type, query_in_keys, temp=cfg)
    if family == "F_rank1":
        return p1_random(seed, n, rho, set_type, query_in_keys, temp=cfg, d_k=1, family="F_rank1")
    if family == "F_margin":
        return p1_margin(seed, n, rho, set_type, query_in_keys, gamma0=cfg)
    if family == "F_ambiguous":
        return p1_ambiguous(seed, n, rho, set_type, query_in_keys, sep=cfg)
    if family == "F_prop9":
        return p1_prop9(seed, rho, set_type, kappa_M=cfg, n=n, query_in_keys=query_in_keys)
    if family in ("F_cov", "F_cov_anti"):  # cfg = kappa; query fixed, balls (set_type / query_in_keys ignored)
        return p1_cov(seed, n, kappa=cfg, anti=(family == "F_cov_anti"), rho=rho)
    raise ValueError(family)


# ============================================================================ secant-width and temperature sweeps
P2_D, P2_N, P2_RHO_A = 8, 3, 0.1


def p2_delta(seed: int, w_target: float, temp: float) -> Instance:
    """Secant-width sweep (part ``"2A"``): F_random, ``d = 8``, ``n = 3``, balls, ``rho = 0.1``, query = key 0, with the
    temperature ``temp`` chosen (by ``temp_for_w`` in ``experiments/synthetic.py``) so that the certified secant width
    ``w_t`` (eq. (6) of the paper) equals ``w_target``."""
    inst = p1_random(seed, P2_N, P2_RHO_A, "ball", True, temp=temp, d=P2_D)
    inst.meta = _p1_meta("F_random", "ball", P2_D, 4, 4, P2_N, "w", w_target, True, seed, P2_RHO_A, temp,
                         w_target=w_target, part="2A")
    inst.meta["instance_id"] = inst.meta["instance_id"].replace("p1_", "p2_", 1)
    inst.meta["phase"] = 2
    return inst


def p2_f1(family: str, seed: int, rho: float, kappa: float, query_in_keys: bool = True) -> Instance:
    """Temperature sweep at fixed rho (part ``"2B"``; the order of the gap in ``||M||``): F_random / F_ambiguous (sep 1)
    with ``||M|| = kappa``; ``F_degen`` = F_random whose key 1 (not the query) varies only in ``(N^T c)^perp`` (flat
    ellipsoid of radius ``rho``), so ``g_1 = B_1^T N^T c = 0`` — a degenerate control in which the varying key's value
    along ``c`` is fixed."""
    if family == "F_random":
        inst = p1_random(seed, P2_N, rho, "ball", query_in_keys, temp=kappa)
    elif family == "F_ambiguous":
        inst = p1_ambiguous(seed, P2_N, rho, "ball", query_in_keys, sep=1.0, temp=kappa)
    elif family == "F_ambiguous_ts":  # temperature-scaled F_ambiguous: built at temp 1, then M -> kappa M
        inst = p1_ambiguous(seed, P2_N, rho, "ball", query_in_keys, sep=1.0, temp=1.0)
        h = inst.head
        inst.head = HeadSpec(M=kappa * h.M, N=h.N, d=h.d, d_k=h.d_k, d_v=h.d_v, raw=None, name="F_ambiguous_ts")
    elif family == "F_degen":
        inst = p1_random(seed, P2_N, rho, "ball", query_in_keys, temp=kappa)
        g = inst.head.N.T @ inst.c
        _, _, Vt = np.linalg.svd(g[None, :])
        E = Vt[1:].T  # orthonormal basis of g^perp (d x (d-1))
        j = next(t for t in inst.keys if t != inst.query)
        inst.sets[j] = S.ellipsoid(inst.sets[j].center, E, rho)
    else:
        raise ValueError(family)
    inst.meta = _p1_meta(family, "ball", P2_D, inst.head.d_k, inst.head.d_v, P2_N, "k", kappa, query_in_keys, seed, rho,
                         kappa, kappa=kappa, part="2B")
    inst.meta["instance_id"] = inst.meta["instance_id"].replace("p1_", "p2_", 1)
    inst.meta["phase"] = 2
    return inst
