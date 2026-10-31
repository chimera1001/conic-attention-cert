"""Theorem 7 of the paper, ``aimp/exact_fq.py`` — the exact worst case of one head with a fixed query.

(1) Synthetic fixed-query heads (d = 8, n in {3, 4, 6} by seed, 50 seeds): the bracket ``[p_fq_lo, p_fq_hi]`` is inside
Gurobi's ``[incumbent, ObjBound]`` (up to Gurobi's tolerances) and contains its proven optimum; 10^5 random feasible
points never exceed ``p_fq_hi``; the recovered worst-case input's value is ``>= p_fq_lo - 1e-12``; the same for the
minimum. (2) The planar reduction: the ellipse of ``(s_u, g_u)`` is the image of the ball. (3) A GPT-2 unit: the projected
and the unprojected (768-dim balls, biases) planar forms give the same bracket, and the worst-case input lifted to 768
dimensions has the same head output as in projected coordinates."""
import functools
import math

import numpy as np
import pytest

from aimp import bounds, instances
from aimp import sets as S
from aimp.exact_fq import FQProblem, exact_fq, fix_query, fq_problem, solve_max
from aimp.exact_gurobi import solve_exact
from aimp.heads import forward

SEEDS = range(50)
GUROBI_T = 10.0  # max
GUROBI_T_MIN = 3.0  # min (also checked by sampling)


def _inst(seed: int):
    rng = np.random.default_rng(17_4 + seed)
    n = (3, 4, 6)[seed % 3]
    rho = float(rng.choice([0.05, 0.2, 0.5, 1.0]))
    temp = float(rng.choice([0.5, 1.0, 2.0, 4.0]))
    return fix_query(instances.f_random(seed, d=8, d_k=4, d_v=4, n=n, temp=temp, rho=rho, set_type="ball",
                                        query_in_keys=bool(seed % 2)))


@functools.lru_cache(maxsize=None)
def _unit(seed: int):
    inst = _inst(seed)
    ex = exact_fq(inst)
    ctrl = bounds.controls(inst, threads=1)
    gu = solve_exact(inst, ctrl, threads=1, time_limit=GUROBI_T)
    from aimp.lower import neg_instance

    gl = solve_exact(neg_instance(inst), ctrl, threads=1, time_limit=GUROBI_T_MIN)
    return inst, ex, gu, gl


def _random_max(inst, k: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    reps = {j: S.affine(inst.sets[j]) for j in inst.keys}
    M, N, c = inst.head.M, inst.head.N, inst.c
    q = inst.sets[inst.query].center
    s_all, g_all = [], []
    for j in inst.keys:
        rep = reps[j]
        U = S.base_sample(rep, rng, k) if rep.r else np.zeros((k, 0))
        Z = rep.c + U @ rep.B.T if rep.r else np.broadcast_to(rep.c, (k, rep.c.size))
        s_all.append(Z @ (M.T @ q))
        g_all.append(Z @ (N.T @ c))
    s, g = np.stack(s_all, 1), np.stack(g_all, 1)
    e = np.exp(s - s.max(axis=1, keepdims=True))
    v = (e * g).sum(1) / e.sum(1)
    return float(v.max()), float(v.min())


@pytest.mark.parametrize("seed", SEEDS)
def test_exact_fq_against_gurobi_and_sampling(seed):
    inst, ex, gu, gl = _unit(seed)
    rng_ = ex["range_planar"]
    lo, hi = ex["p_fq_lo"], ex["p_fq_hi"]
    mlo, mhi = ex["min_fq_lo"], ex["min_fq_hi"]
    assert ex["status"] == "exact" and ex["capped_evals_max"] == 0 and ex["capped_evals_min"] == 0
    assert hi - lo <= 1e-10 * rng_ + 1e-15 and ex["stop_max"] == "tol"
    assert mhi - mlo <= 1e-10 * rng_ + 1e-15 and ex["stop_min"] == "tol"
    assert ex["lo_cert_max"] in ("H_lo>0", "feasible") and ex["hi_cert_max"] in ("H_hi<0", "g_max")
    # worst-case inputs are feasible and attain the brackets (strict: float rounding only)
    ft = 1e-12 * max(1.0, abs(hi), abs(mlo))
    for Z in (ex["Z_max"], ex["Z_min"]):
        for t in inst.tokens:
            assert S.contains(inst.sets[t], Z[t], tol=1e-9)
    assert lo - ft <= ex["value_max_input"] <= hi + ft
    assert mlo - ft <= ex["value_min_input"] <= mhi + ft
    # Gurobi. Its verified incumbent is a numpy re-evaluation at a feasible point: compared strictly (1e-12). Its
    # ObjBound is valid only up to its own tolerances (FeasibilityTol 1e-8, MIPGap 1e-6, MIPGapAbs 1e-7): tg.
    tg = 1e-6 * max(1.0, abs(hi)) + 1e-7
    assert gu.value_lb_verified <= hi + ft, (seed, gu.value_lb_verified, hi)
    if math.isfinite(gu.value):
        assert lo <= gu.value + tg, (seed, lo, gu.value)
    if gu.status == "optimal":  # optimality: incumbent within MIPGap of p_star
        assert lo - gu.value_lb_verified <= tg, (seed, gu.value_lb_verified, lo)
    assert -gl.value_lb_verified >= mlo - ft  # a feasible value of phi is >= min phi
    if math.isfinite(gl.value):
        assert mhi >= -gl.value - tg
    # 10^5 random feasible points
    vmax, vmin = _random_max(inst, 100_000, seed)
    assert vmax <= hi + 1e-12 * max(1.0, abs(hi)), (seed, vmax, hi)
    assert vmin >= mlo - 1e-12 * max(1.0, abs(mlo)), (seed, vmin, mlo)


def test_gurobi_mostly_optimal():
    """The comparison is meaningful: Gurobi proves optimality on a good share of the fixed-query instances within 10 s
    (18 / 50 when first run; the others are checked against Gurobi's incumbent and ObjBound)."""
    st = [_unit(s)[2].status for s in SEEDS]
    assert sum(x == "optimal" for x in st) >= 15, st


def test_planar_ellipse_is_the_image():
    rng = np.random.default_rng(0)
    inst = _inst(7)
    P = fq_problem(inst)
    for u, idx in enumerate(P.var):
        j = inst.keys[idx]
        rep = S.affine(inst.sets[j])
        q = inst.sets[inst.query].center
        A = np.stack([rep.B.T @ (inst.head.M.T @ q), rep.B.T @ (inst.head.N.T @ inst.c)], 1)
        assert np.allclose(P.L[u] @ P.L[u].T, A.T @ A, rtol=1e-10, atol=1e-12)
        for t in rng.uniform(0, 2 * np.pi, 20):  # boundary points come from base points on the sphere
            w = P.rho[u] * (P.U[u] @ np.array([np.cos(t), np.sin(t)]))
            assert np.linalg.norm(w) == pytest.approx(P.rho[u], rel=1e-12)
            pt = np.array([P.sbar[u], P.gbar[u]]) + P.rho[u] * P.L[u] @ np.array([np.cos(t), np.sin(t)])
            assert np.allclose(A.T @ w + [P.sbar[u], P.gbar[u]], pt, rtol=1e-12, atol=1e-12)


def test_degenerate_segment_and_constant():
    """Rank-1 Gram (score constant in the key, e.g. M = 0) and a fully fixed head."""
    inst = _inst(4)
    head = inst.head
    from aimp.types import HeadSpec, Instance

    h0 = HeadSpec(M=np.zeros_like(head.M), N=head.N, d=head.d, d_k=head.d_k, d_v=head.d_v)
    i0 = Instance(head=h0, keys=inst.keys, query=inst.query, sets=inst.sets, c=inst.c)
    ex = exact_fq(i0)
    # with M = 0 the pattern is uniform: max phi = mean of the per-key maxima of g
    g = np.array([S.support(inst.sets[j], head.N.T @ inst.c) for j in inst.keys])
    assert ex["p_fq_lo"] <= g.mean() + 1e-12 and ex["p_fq_hi"] >= g.mean() - 1e-12
    fixed = Instance(head=head, keys=inst.keys, query=inst.query,
                     sets={t: S.point(inst.sets[t].center) for t in inst.sets}, c=inst.c)
    ex = exact_fq(fixed)
    assert ex["p_fq_lo"] == ex["p_fq_hi"] == pytest.approx(forward(fixed, fixed.centers())["phi"], abs=1e-14)


def test_gpt2_projected_vs_unprojected():
    from aimp import gpt2_units as U

    try:
        U.prepare()
    except Exception as e:  # no network and no cached weights
        pytest.skip(f"GPT-2 weights unavailable: {e!r}")
    l, h, p, pos, rho = 5, 1, 0, 8, 0.01
    inst = U.make_instance(l, h, p, pos, rho)
    fq = fix_query(inst)
    ex = exact_fq(fq)
    hw, pr = U.head_data(l, h)
    z = U.ctx()["P"][p]["z"][l]
    dk = 64
    q = z[pos] @ hw["W_Q"] + hw["b_Q"]
    gO = hw["W_O"] @ fq.c  # g = c^T W_O^T v, v = W_V^T z + b_V
    # unprojected planar form: keys j < pos vary in 768-dim balls rho ||z_j||, key pos fixed
    s_fix, g_fix, sbar, gbar, rr, Ls, Us = [], [], [], [], [], [], []
    for j in range(pos + 1):
        k = z[j] @ hw["W_K"] + hw["b_K"]
        v = z[j] @ hw["W_V"] + hw["b_V"]
        s0, g0 = float(q @ k) / math.sqrt(dk), float(gO @ v)
        if j == pos:
            s_fix.append(s0), g_fix.append(g0)
            continue
        A = np.stack([hw["W_K"] @ q / math.sqrt(dk), hw["W_V"] @ gO], 1)  # (768, 2)
        U, sig, Vt = np.linalg.svd(A, full_matrices=False)
        sbar.append(s0), gbar.append(g0), rr.append(rho * float(np.linalg.norm(z[j]))), Ls.append(Vt.T * sig), Us.append(U)
    Pu = FQProblem([pos], np.array(s_fix), np.array(g_fix), list(range(pos)), np.array(sbar), np.array(gbar),
                   np.array(rr), np.array(Ls), Us, float(max(g_fix + gbar) - min(g_fix + gbar)), pos + 1)
    mu = solve_max(Pu)
    tol = 2e-10 * ex["range_planar"]
    assert mu["p_lo"] == pytest.approx(ex["p_fq_lo"], abs=tol) and mu["p_hi"] == pytest.approx(ex["p_fq_hi"], abs=tol)
    # the worst-case input in 768 dimensions: z_j + rho_j U (cos, sin); the head's output there (numpy forward)
    Z = z[: pos + 1].copy()
    for u in range(pos):
        Z[u] = z[u] + rr[u] * (Us[u] @ np.array([math.cos(mu["t"][u]), math.sin(mu["t"][u])]))
    qq = Z[pos] @ hw["W_Q"] + hw["b_Q"]
    s = (Z @ hw["W_K"] + hw["b_K"]) @ qq / math.sqrt(dk)
    a = np.exp(s - s.max())
    a /= a.sum()
    y = a @ ((Z @ hw["W_V"] + hw["b_V"]) @ hw["W_O"])
    v768 = float(fq.c @ y)
    assert v768 == pytest.approx(mu["value"], abs=1e-9 * ex["range_planar"])
    assert ex["p_fq_lo"] - tol <= v768 <= ex["p_fq_hi"] + tol
    # and the projected worst-case input's forward pass equals its planar value
    assert ex["value_max_input"] == pytest.approx(ex["planar_value_max"], abs=1e-10 * ex["range_planar"])


def test_forced_cell_cap_stays_valid_and_unresolved():
    """With a tiny per-level cell cap the refinement stops early; every key's [mu_lo, mu_hi] is still a valid
    enclosure, so the (wider) brackets contain the uncapped exact ones and the sampled values, and the unit is reported
    ``unresolved`` with its width — never as exact."""
    for seed in (0, 5, 7):
        inst = _inst(seed)
        ex = _unit(seed)[1]
        cap = exact_fq(inst, max_cells=8)
        assert cap["status"] == "unresolved" and cap["capped_evals_max"] > 0
        assert cap["width_rel_max"] >= ex["width_rel_max"] and cap["width_rel_min"] >= ex["width_rel_min"]
        tol = 1e-12 * max(1.0, abs(ex["p_fq_hi"]))
        assert cap["p_fq_lo"] <= ex["p_fq_lo"] + tol and cap["p_fq_hi"] >= ex["p_fq_hi"] - tol
        assert cap["min_fq_lo"] <= ex["min_fq_lo"] + tol and cap["min_fq_hi"] >= ex["min_fq_hi"] - tol
        vmax, vmin = _random_max(inst, 20_000, seed + 1)
        assert vmax <= cap["p_fq_hi"] + tol and vmin >= cap["min_fq_lo"] - tol
        assert cap["max_input_in_bracket"] and cap["min_input_in_bracket"]
        # per key: the capped enclosure contains the converged one
        from aimp.exact_fq import _shift, key_max

        P = fq_problem(inst)
        lam = 0.5 * (ex["p_fq_lo"] + ex["p_fq_hi"])
        s_ref = _shift(P, None)
        for u in range(len(P.var)):
            a = key_max(P, u, lam, s_ref)
            b = key_max(P, u, lam, s_ref, max_cells=8)
            assert b["stop"] in ("max_cells", "tol")
            assert b["mu_lo"] <= a["mu_hi"] and b["mu_hi"] >= a["mu_lo"] and b["mu_hi"] >= a["mu_hi"] - 1e-15
