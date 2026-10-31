"""Log-attention intervals (14), the cone form, the safe dual bound (``value_safe``, Proposition 16), and the norm-based
second-order bounds with the secant width ``w_t`` (``bound_F0t``) and the weighted secant slack ``S``
(``bound_F0s``)."""
import math

import numpy as np
import pytest

from aimp import bounds, instances, sets as S
from aimp.heads import softmax
from aimp.lower_bounds import multistart_lb
from aimp.relax_sdp import solve_sdp
from aimp.safe_bound import safe_dual_bound
from aimp.theory import bound_F0t, nominal_decomposition

S1 = ("persp", "simplex_x_z")
S2 = S1 + ("softmax_x_z", "softmax_x_ball")


def sample_Z(inst, rng):
    return np.stack([S.to_z(S.affine(inst.sets[t]), S.base_sample(S.affine(inst.sets[t]), rng, 1)[0])
                     for t in range(inst.T)])


# ------------------------------------------------------------------ log-attention intervals (14)
def test_tight_t_intervals_contain_log_softmax():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(2, 6))
        s_lo = rng.normal(0, 3, n)
        s_hi = s_lo + rng.exponential(2, n)
        lo_l, hi_l = bounds.t_intervals(s_lo, s_hi, "loose")
        lo_t, hi_t = bounds.t_intervals(s_lo, s_hi, "tight")
        assert np.all(lo_t >= lo_l - 1e-12) and np.all(hi_t <= hi_l + 1e-12)
        assert bounds.t_width(s_lo, s_hi, "tight") <= bounds.t_width(s_lo, s_hi, "loose") + 1e-12
        for _ in range(20):
            s = s_lo + rng.uniform(0, 1, n) * (s_hi - s_lo)
            t = np.log(softmax(s))
            assert np.all(t >= lo_t - 1e-12) and np.all(t <= hi_t + 1e-12)


@pytest.mark.parametrize("st", ["ball", "box"])
@pytest.mark.parametrize("qk", [True, False])
def test_v24_sdp_true_points_feasible(st, qk):
    """Tight intervals + cone form: every true configuration is feasible (rank-1 lift), for sdp_1 and sdp_2."""
    rng = np.random.default_rng(5)
    for seed in range(2):
        inst = instances.f_random(seed, d=4, d_k=2, d_v=2, n=3, temp=3.0, rho=0.4, set_type=st, query_in_keys=qk)
        ctrl = bounds.controls(inst)
        for fams in (S1, S2):
            for _ in range(3):
                r = solve_sdp(inst, ctrl, rlt=fams, check_point=sample_Z(inst, rng))
                assert r.diagnostics["max_violation"] <= 1e-9, r.diagnostics


def test_tight_is_tighter_and_valid():
    for seed in range(3):
        inst = instances.p1_random(seed, 3, 0.1, "ball", True, temp=3.0)
        ctrl = bounds.controls(inst)
        lb = multistart_lb(inst, 20, seed=seed).value
        rl = solve_sdp(inst, ctrl, rlt=S1, threads=1, t_interval="loose")
        rt = solve_sdp(inst, ctrl, rlt=S1, threads=1, t_interval="tight")
        assert rt.value >= lb - 1e-7 and rl.value >= lb - 1e-7
        assert rt.value <= rl.value + 1e-7
        assert rt.diagnostics["w_t_used"] <= rt.diagnostics["w_t_loose"] + 1e-12


def test_cone_form_matches_cvxpy_atoms():
    inst = instances.p1_random(1, 3, 0.05, "ball", False, temp=1.0)
    ctrl = bounds.controls(inst)
    for fams in (S1, S2):  # same feasible set; MOSEK's accuracy differs between the two CVXPY models (~1e-7 here)
        a = solve_sdp(inst, ctrl, rlt=fams, threads=1, cone_form=True, safe="always")
        b = solve_sdp(inst, ctrl, rlt=fams, threads=1, cone_form=False)
        assert abs(a.value - b.value) <= 1e-6
        assert a.diagnostics["value_safe"] >= a.value - 1e-12


# ------------------------------------------------------------------ safe dual bound (Proposition 16)
def test_safe_bound_toy_lp_exact():
    """max x1 + x2 s.t. x1 + 2 x2 <= 4, 3 x1 + x2 <= 6, x >= 0 (opt 2.8): the safe bound is >= 2.8 and within 1e-7."""
    import cvxpy as cp

    X = cp.Variable((3, 3), PSD=True)
    x = X[0, 1:]
    prob = cp.Problem(cp.Maximize(x[0] + x[1]), [X[0, 0] == 1, x[0] + 2 * x[1] <= 4, 3 * x[0] + x[1] <= 6, x >= 0,
                                                 cp.diag(X)[1:] <= 16])
    prob.solve(solver=cp.MOSEK)
    sb = safe_dual_bound(prob, X, np.array([1.0, 16.0, 16.0]))
    assert sb["value_safe"] >= 2.8 - 1e-12
    assert sb["value_safe"] - 2.8 <= 1e-7


def test_safe_bound_perturbed_dual_stays_valid():
    """Perturbing every multiplier keeps value_safe >= the true optimum (the bound is valid for any multipliers)."""
    import cvxpy as cp

    rng = np.random.default_rng(1)
    X = cp.Variable((3, 3), PSD=True)
    x = X[0, 1:]
    cons = [X[0, 0] == 1, x[0] + 2 * x[1] <= 4, 3 * x[0] + x[1] <= 6, x >= 0, cp.diag(X)[1:] <= 16,
            cp.SOC(cp.Constant(3.0), x), cp.constraints.ExpCone(x[0] - 3, cp.Constant(1.0), x[1] + 1)]
    prob = cp.Problem(cp.Maximize(x[0] + x[1]), cons)
    prob.solve(solver=cp.MOSEK)
    opt = prob.value
    for _ in range(20):
        for c in cons:
            dv = c.dual_value
            if isinstance(dv, list):
                c.dual_variables[0].save_value(np.asarray(dv[0]) + rng.normal(0, 0.05, np.shape(dv[0])))
            elif dv is not None:
                c.dual_variables[0].save_value(np.asarray(dv) + rng.normal(0, 0.05, np.shape(dv)))
        sb = safe_dual_bound(prob, X, np.array([1.0, 16.0, 16.0]))
        assert sb["value_safe"] >= opt - 1e-9


@pytest.mark.parametrize("st", ["ball", "box"])
def test_value_safe_geq_value_dual(st):
    """value_safe >= value_dual (computed on every solve with safe='always')."""
    worst = math.inf
    for seed in range(4):
        for temp in (1.0, 10.0):
            inst = instances.p1_random(seed, 3, 0.1, st, seed % 2 == 0, temp=temp)
            ctrl = bounds.controls(inst)
            for fams in (S1, S2):
                r = solve_sdp(inst, ctrl, rlt=fams, threads=1, safe="always")
                dg = r.diagnostics
                if dg["value_dual"] is None or dg["value_safe"] is None:
                    continue
                worst = min(worst, dg["value_safe"] - dg["value_dual"])
                assert dg["value_safe"] >= dg["value_dual"] - 1e-12 * max(1.0, abs(dg["value_dual"])), dg
    assert math.isfinite(worst)


def test_value_safe_close_on_50_clean_instances():
    """On 50 clean instances (dual-certified value), value_safe - value <= 1e-6."""
    n_clean, diffs = 0, []
    fams = ["F_random", "F_ambiguous", "F_margin", "F_rank1"]
    cfg = {"F_random": 1.0, "F_ambiguous": 1.0, "F_margin": 2.0, "F_rank1": 1.0}
    for k in range(80):
        fam = fams[k % 4]
        inst = instances.phase1_instance(fam, k // 4, 2, [0.01, 0.05, 0.1][k % 3], ["ball", "box"][k % 2],
                                         bool(k % 3), cfg[fam])
        ctrl = bounds.controls(inst)
        r = solve_sdp(inst, ctrl, rlt=S1, threads=1, safe="always")
        dg = r.diagnostics
        if dg["value_source"] != "dual" or dg["value_safe"] is None:
            continue
        n_clean += 1
        diffs.append(dg["value_safe"] - r.value)
        if n_clean == 50:
            break
    assert n_clean == 50
    assert max(diffs) <= 1e-6, max(diffs)
    assert min(diffs) >= -1e-12


def test_unreliable_value_uses_safe():
    """A wide-score instance whose dual residual exceeds the relative threshold gets value = value_safe."""
    inst = instances.p1_random(0, 2, 0.2, "box", True, temp=10.0)
    ctrl = bounds.controls(inst)
    r = solve_sdp(inst, ctrl, rlt=S1, threads=1, t_interval="loose")
    dg = r.diagnostics
    if dg["dual_resid"] > 1e-8 * max(1.0, abs(dg["value_dual"])):
        assert dg["value_source"] == "safe" and r.value == dg["value_safe"]
    else:
        assert dg["value_source"] == "dual"
    lb = multistart_lb(inst, 20, seed=0).value
    assert r.value >= lb - 1e-9


# ------------------------------------------------------------------ bound_F0t with the secant width w_t
@pytest.mark.parametrize("fams", [(), S1])
def test_F0_tight_bound_holds(fams):
    """bound_F0t (eta_exp(w_t), w_t the width of the log-attention intervals) holds on sdp_0 (no persp needed) and
    sdp_1."""
    for seed in range(3):
        inst = instances.p1_ambiguous(seed, 3, 0.05, "ball", False)
        ctrl = bounds.controls(inst)
        lb = multistart_lb(inst, 30, seed=seed).value
        r = solve_sdp(inst, ctrl, rlt=fams, threads=1)
        b = bound_F0t(inst, ctrl, nominal_decomposition(inst), mode="tight")
        assert b["F0t_w_t"] <= bounds.t_width(ctrl["s_lo"], ctrl["s_hi"], "loose") + 1e-12
        assert r.value - lb <= b["bound_F0t"] + 1e-6


# ------------------------------------------------------------------ weighted secant slack S (16) and bound_F0s
def test_sandwich_S_formula_and_F0s_holds():
    from aimp.theory import bound_F0s, sandwich_S

    for seed in range(3):
        inst = instances.p1_random(seed, 3, 0.1, "ball", True, temp=1.0)
        ctrl = bounds.controls(inst)
        t_lo, t_hi = bounds.t_intervals(ctrl["s_lo"], ctrl["s_hi"], "tight")
        S = sandwich_S(ctrl)
        assert abs(S - float(np.sum(np.clip(ctrl["a_hi"], 0, 1) * (t_hi - t_lo) ** 2) / 8)) <= 1e-15
        nom = nominal_decomposition(inst)
        fs = bound_F0s(inst, ctrl, nom)
        assert fs["F0s_applies"] == (S < 1)
        lb = multistart_lb(inst, 20, seed=seed).value
        for fams in ((), S1):
            r = solve_sdp(inst, ctrl, rlt=fams, threads=1)
            assert r.value - lb <= fs["bound_F0s"] + 1e-6
