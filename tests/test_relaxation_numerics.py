"""Relaxation numerics: lin_sandwich validity, exp-cone pruning, the dual-certified value, the greedy for the pattern
polytope LP (Proposition 8), the F_cov family, and the norm-based second-order (``bound_F0``) and linearization
(``bound_F``) bounds."""
import numpy as np
import pytest

from aimp import bounds, instances, sets as S
from aimp.lower_bounds import multistart_lb
from aimp.relax_ps import ps_greedy, ps_lp, solve_ps, solve_ps_greedy
from aimp.relax_sdp import solve_sdp
from aimp.theory import bound_F, bound_F0, nominal_decomposition

S1 = ("persp", "simplex_x_z")
S2 = S1 + ("softmax_x_z", "softmax_x_ball")
LIN = S1 + ("lin_sandwich",)
ALL = S2 + ("lin_sandwich",)


def sample_Z(inst, rng):
    return np.stack([S.to_z(S.affine(inst.sets[t]), S.base_sample(S.affine(inst.sets[t]), rng, 1)[0])
                     for t in range(inst.T)])


def margin_instance():
    """A key with a_hi <= 1e-6 (pruned): F_margin with a 25-unit nominal margin."""
    inst = instances.p1_margin(0, 3, 0.01, "ball", False, gamma0=25.0)
    ctrl = bounds.controls(inst)
    assert np.sum(ctrl["a_hi"] <= 1e-6) >= 1
    return inst, ctrl


@pytest.mark.parametrize("st", ["ball", "box"])
@pytest.mark.parametrize("qk", [True, False])
def test_lin_sandwich_true_points_feasible(st, qk):
    rng = np.random.default_rng(3)
    for seed in range(2):
        inst = instances.f_random(seed, d=4, d_k=2, d_v=2, n=3, rho=0.4, set_type=st, query_in_keys=qk)
        ctrl = bounds.controls(inst)
        for _ in range(4):
            r = solve_sdp(inst, ctrl, rlt=ALL, check_point=sample_Z(inst, rng))
            assert r.diagnostics["max_violation"] <= 1e-9, r.diagnostics
            assert abs(r.value - r.diagnostics["true_value"]) <= 1e-10


def test_true_points_feasible_with_pruning_and_fcov():
    rng = np.random.default_rng(4)
    inst, ctrl = margin_instance()
    for _ in range(4):
        assert solve_sdp(inst, ctrl, rlt=ALL, check_point=sample_Z(inst, rng)).diagnostics["max_violation"] <= 1e-9
    for anti in (False, True):
        inst = instances.p1_cov(1, 3, 0.1, anti=anti)
        ctrl = bounds.controls(inst)
        for _ in range(4):
            assert solve_sdp(inst, ctrl, rlt=ALL, check_point=sample_Z(inst, rng)).diagnostics["max_violation"] <= 1e-9


def test_pruning_is_a_relaxation():
    inst, ctrl = margin_instance()
    for fams in (S1, S2):
        rp = solve_sdp(inst, ctrl, rlt=fams, threads=1)
        ru = solve_sdp(inst, ctrl, rlt=fams, threads=1, prune_tol=None)
        assert rp.diagnostics["pruned_keys"] and not ru.diagnostics["pruned_keys"]
        tol = 1e-7 * max(1.0, abs(ru.value))
        assert rp.value >= ru.value - tol  # constraints removed
        assert rp.value <= ru.value + rp.diagnostics["pruning_slack"] + 1e-6


def test_dual_value_geq_primal_value():
    for seed in range(3):
        for st in ("ball", "box"):
            inst = instances.p1_random(seed, 3, 0.1, st, seed % 2 == 0, temp=1.0)
            r = solve_sdp(inst, bounds.controls(inst), rlt=S2, threads=1)
            dg = r.diagnostics
            assert dg["value_dual"] is not None and dg["value_primal"] is not None
            assert dg["value_dual"] >= dg["value_primal"] - 1e-8 * max(1.0, abs(dg["value_primal"]))
            if dg["value_source"] == "dual":
                assert r.value == dg["value_dual"] and dg["dual_resid"] <= 1e-8


def test_greedy_equals_lp_200():
    rng = np.random.default_rng(12)
    worst, checked = 0.0, 0
    for k in range(200):
        n = int(rng.integers(2, 12))
        h = rng.standard_normal(n)
        if k % 5 == 0:
            h = np.round(h, 1)  # ties
        s = rng.standard_normal(n) * rng.uniform(0.1, 3)
        w = rng.uniform(0, 2, n)
        sm = bounds.softmax_interval_bounds(s - w / 2, s + w / 2)
        if k % 3 == 0:
            ahat, eps = np.eye(n)[rng.integers(n)], rng.uniform(0.0, 2.0)
        else:
            e = np.exp(s - s.max())
            ahat, eps = e / e.sum(), rng.uniform(0.0, 1.5)
        if k % 7 == 0:
            ahat = eps = None
        try:
            g, a = ps_greedy(h, sm["a_lo"], sm["a_hi"], ahat, eps)
        except ValueError:
            g = None
        try:
            lp = ps_lp(h, sm["a_lo"], sm["a_hi"], ahat, eps, threads=1)
        except RuntimeError:
            lp = None
        assert (g is None) == (lp is None)
        if g is not None:
            assert abs(a.sum() - 1) <= 1e-12 and np.all(a >= sm["a_lo"] - 1e-12) and np.all(a <= sm["a_hi"] + 1e-12)
            worst = max(worst, abs(g - lp))
            checked += 1
    assert checked >= 150 and worst <= 1e-9


def test_greedy_matches_socp_on_instances():
    for seed in range(3):
        for st in ("ball", "box"):
            inst = instances.p1_random(seed, 4, 0.1, st, False, temp=3.0)
            ctrl = bounds.controls(inst)
            for cut in ("auto", "B", "none"):
                g = solve_ps_greedy(inst, ctrl, cut)
                s = solve_ps(inst, ctrl, cut, threads=1)
                assert abs(g.value - s.value) <= 1e-6 * max(1.0, abs(s.value))
                assert g.recovered_value <= g.value + 1e-9


@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("anti", [False, True])
def test_fcov_construction(n, anti):
    rng = np.random.default_rng(5)
    kappa = 0.05
    inst = instances.p1_cov(2, n, kappa, anti=anti)
    M, N, c = inst.head.M, inst.head.N, inst.c
    q = inst.sets[inst.query].center
    assert not inst.query_in_keys and inst.sets[inst.query].rho == 0.0
    zbar = np.stack([inst.sets[j].center for j in inst.keys])
    sbar = zbar @ (M.T @ q)
    assert np.max(np.abs(sbar)) <= 1e-12  # equal nominal scores (0)
    assert np.allclose(np.linalg.norm(zbar, axis=1), np.sqrt(8)) and np.isclose(np.linalg.norm(q), np.sqrt(8))
    assert np.allclose(bounds.controls(inst)["ahat"], 1.0 / n)  # uniform midpoint pattern
    sign = -1.0 if anti else 1.0
    for _ in range(5):  # x_l = +- kappa y_l: score shift and value shift along c are aligned
        for j in inst.keys:
            u = S.base_sample(S.affine(inst.sets[j]), rng, 1)[0]
            z = zbar[j] + u
            x = q @ M @ z - sbar[j]
            y = c @ N @ z - c @ N @ zbar[j]
            assert abs(x - sign * kappa * y) <= 1e-12
            assert abs(y - sign * np.asarray(inst.meta["w"]) @ u) <= 1e-12


def test_theorem_F0_and_F_hold():
    for seed in range(3):
        for st in ("ball", "box"):
            inst = instances.p1_random(seed, 3, 0.05, st, seed == 1, temp=1.0)
            ctrl = bounds.controls(inst)
            nom = nominal_decomposition(inst)
            lb = multistart_lb(inst, 30, seed=seed).value_lb_verified
            r1 = solve_sdp(inst, ctrl, rlt=S1, threads=1)
            rl = solve_sdp(inst, ctrl, rlt=LIN, threads=1)
            assert r1.value - lb <= bound_F0(inst, ctrl, nom)["bound_F0"] + 1e-6
            assert rl.value - lb <= bound_F(inst, ctrl, nom)["bound_F"] + 1e-6
            assert rl.value <= r1.value + 1e-6  # sdp_lin = sdp_1 + products


def test_F0t_width_and_bound():
    """The loose log-form secant width w_t = max width + (tau_hi - tau_lo) lies in [Delta, 2 Delta]; bound_F0t >=
    bound_F0."""
    from aimp.theory import bound_F0t, t_width

    for seed in range(3):
        inst = instances.p1_ambiguous(seed, 4, 0.01, "ball", False)
        ctrl = bounds.controls(inst)
        w = t_width(ctrl)
        assert ctrl["Delta"] - 1e-12 <= w <= 2 * ctrl["Delta"] + 1e-12
        nom = nominal_decomposition(inst)
        assert bound_F0t(inst, ctrl, nom)["bound_F0t"] >= bound_F0(inst, ctrl, nom)["bound_F0"]
