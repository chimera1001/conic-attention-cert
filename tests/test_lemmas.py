"""Numerical checks of the lemmas: the Lipschitz bound (Lemma 1), scores in a box (Remark 1), the attention intervals
and the margin bound (Lemma 2), the secant gap bound (Lemma 3), and the exactness of the scaled-set (perspective)
relaxation of the products ``a_j z_j`` (Section 4)."""
import itertools
import math

import numpy as np
import pytest

from aimp import bounds, sets as S
from aimp.heads import softmax

RNG = np.random.default_rng(12345)


def test_L2_lipschitz():
    rng = np.random.default_rng(0)
    for _ in range(5000):
        n = rng.integers(2, 9)
        scale = 10 ** rng.uniform(-3, 1.5)
        s = rng.standard_normal(n) * scale
        sp = s + rng.standard_normal(n) * scale * rng.uniform(0, 2)
        lhs = np.abs(softmax(s) - softmax(sp)).sum()
        rng_ = np.ptp(s - sp)
        assert lhs <= 0.5 * rng_ + 1e-12
        assert 0.5 * rng_ <= np.abs(s - sp).max() + 1e-12


def test_L3_threshold_vs_bruteforce():
    rng = np.random.default_rng(1)
    for _ in range(300):
        n = int(rng.integers(1, 9))
        v = rng.standard_normal(n)
        lo = rng.standard_normal(n) * 2
        hi = lo + rng.uniform(0, 3, n)
        best_thr, _ = bounds.l3_max_vsigma(v, lo, hi)
        brute = max(v @ softmax(np.where(np.array(bits), hi, lo)) for bits in itertools.product([0, 1], repeat=n))
        assert abs(best_thr - brute) <= 1e-12
        # random interior points never exceed it
        for s in rng.uniform(lo, hi, size=(50, n)):
            assert v @ softmax(s) <= best_thr + 1e-12


def test_L3_monotone_coordinate():
    rng = np.random.default_rng(2)
    for _ in range(200):
        n = int(rng.integers(2, 7))
        v, s = rng.standard_normal(n), rng.standard_normal(n)
        k = int(rng.integers(n))
        ts = np.linspace(-5, 5, 201)
        vals = []
        for t in ts:
            s2 = s.copy()
            s2[k] = t
            vals.append(v @ softmax(s2))
        dv = np.diff(vals)
        assert np.all(dv >= -1e-13) or np.all(dv <= 1e-13)


def test_L4_and_softmax_interval_bounds():
    rng = np.random.default_rng(3)
    for _ in range(500):
        n = int(rng.integers(1, 8))
        lo = rng.standard_normal(n) * 3
        hi = lo + rng.uniform(0, 2, n)
        sb = bounds.softmax_interval_bounds(lo, hi)
        samp = rng.uniform(lo, hi, size=(200, n))
        A = np.array([softmax(s) for s in samp])
        assert np.all(A >= sb["a_lo"] - 1e-14) and np.all(A <= sb["a_hi"] + 1e-14)
        # closed form (12) of Lemma 2
        for j in range(n):
            other = [l for l in range(n) if l != j]
            alo = 1 / (1 + sum(math.exp(hi[l] - lo[j]) for l in other))
            ahi = 1 / (1 + sum(math.exp(lo[l] - hi[j]) for l in other))
            assert abs(alo - sb["a_lo"][j]) < 1e-12 and abs(ahi - sb["a_hi"][j]) < 1e-12
        lse = [math.log(sum(math.exp(x) for x in s)) for s in samp]
        assert min(lse) >= sb["tau_lo"] - 1e-12 and max(lse) <= sb["tau_hi"] + 1e-12
        # margin bound (13) of Lemma 2
        if n >= 2:
            gs = [lo[k] - np.delete(hi, k).max() for k in range(n)]
            js = int(np.argmax(gs))
            gamma = gs[js]
            if gamma > 0:
                e = np.zeros(n)
                e[js] = 1
                dist = np.abs(A - e).sum(axis=1).max()
                assert dist <= bounds.l4_onehot_bound(n, gamma) + 1e-12
                assert bounds.l4_onehot_bound(n, gamma) <= 2 * (n - 1) * math.exp(-gamma)


def test_P7_secant():
    rng = np.random.default_rng(4)
    for _ in range(300):
        n = int(rng.integers(1, 7))
        lo = rng.standard_normal(n)
        Delta = rng.uniform(0, 2)
        hi = lo + Delta * rng.uniform(0, 1, n)
        slope = np.where(hi > lo, (np.exp(hi) - np.exp(lo)) / np.maximum(hi - lo, 1e-300), np.exp(lo))
        s = rng.uniform(lo, hi)
        ell = np.exp(lo) + slope * (s - lo)
        err = ell - np.exp(s)
        assert np.all(err >= -1e-12)
        assert np.all(err <= (hi - lo) ** 2 / 8 * np.exp(hi) + 1e-12)
        D = float(np.max(hi - lo))
        eta = bounds.eta(D)
        assert np.all(ell <= np.exp(s) * (1 + eta) + 1e-12)
        et = np.exp(s) + rng.uniform(0, 1, n) * err
        assert np.abs(et / et.sum() - softmax(s)).sum() <= eta + 1e-12


def _random_set(kind, d, rng):
    c = rng.standard_normal(d)
    if kind == "box":
        return S.box_around(c, rng.uniform(0.1, 1, d))
    if kind == "ball":
        return S.ball(c, rng.uniform(0.1, 1))
    E = rng.standard_normal((d, max(1, d - 1)))
    return S.ellipsoid(c, E, rng.uniform(0.1, 1))


@pytest.mark.parametrize("kind", ["box", "ball", "ellipsoid"])
def test_L8_perspective_exact(kind):
    """max over the perspective set {(a, sum N w_j): w_j in a_j Z_j} of alpha.a + beta.y equals
    max_j (alpha_j + support(Z_j, N^T beta)), the value over the true set; samples never exceed it."""
    import cvxpy as cp

    rng = np.random.default_rng({"box": 5, "ball": 6, "ellipsoid": 7}[kind])
    for _ in range(10):
        d, n = int(rng.integers(1, 5)), int(rng.integers(1, 4))
        Ss = [_random_set(kind, d, rng) for _ in range(n)]
        N = rng.standard_normal((d, d))
        alpha, beta = rng.standard_normal(n), rng.standard_normal(d)
        a = cp.Variable(n)
        cons = [a >= 0, cp.sum(a) == 1]
        y = 0
        for j, Sj in enumerate(Ss):
            rep = S.affine(Sj)
            g = cp.Variable(rep.r)
            if rep.base == "box":
                cons += [g <= a[j] * rep.h, g >= -a[j] * rep.h]
            else:
                cons += [cp.norm(g) <= rep.rho * a[j]]
            y = y + N @ (a[j] * rep.c + rep.B @ g)
        prob = cp.Problem(cp.Maximize(alpha @ a + beta @ y), cons)
        prob.solve(solver=cp.MOSEK)
        true = max(alpha[j] + S.support(Ss[j], N.T @ beta) for j in range(n))
        assert abs(prob.value - true) <= 1e-6 * max(1, abs(true))
        for _ in range(200):
            aa = rng.dirichlet(np.ones(n))
            zs = [S.to_z(S.affine(Sj), S.base_sample(S.affine(Sj), rng, 1)[0]) for Sj in Ss]
            val = alpha @ aa + beta @ sum(aa[j] * N @ zs[j] for j in range(n))
            assert val <= true + 1e-9
