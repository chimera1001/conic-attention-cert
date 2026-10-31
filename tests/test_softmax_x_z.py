"""The ``softmax_x_z`` RLT family (and the other optional families of :mod:`aimp.relax_sdp`): validity, monotonicity,
and the order of the gap on the two-key instance :func:`aimp.instances.prop9`."""
import math

import numpy as np
import pytest

from aimp import bounds, instances, sets as S
from aimp.lower_bounds import multistart_lb
from aimp.relax_sdp import LiftIndex, solve_sdp

S1 = ("persp", "simplex_x_z")
S2 = S1 + ("softmax_x_z",)
ALL = S2 + ("softmax_x_ball", "score_rlt", "cut_x_z", "box_x_box")


def fixed_query(inst):
    """Same instance with the query token fixed at its center (the affine-score case of ``softmax_x_z``)."""
    inst.sets[inst.query] = S.point(inst.sets[inst.query].center)
    return inst


def sample_Z(inst, rng):
    return np.stack([S.to_z(S.affine(inst.sets[t]), S.base_sample(S.affine(inst.sets[t]), rng, 1)[0])
                     for t in range(inst.T)])


def test_lift_index_extras():
    inst = instances.f_random(0, d=3, d_k=2, d_v=2, n=2, rho=0.1, query_in_keys=False)
    L = LiftIndex.for_instance(inst, ("tau", "s0", "s1", "p0", "p1", "q0", "q1"))
    assert L.D == 3 * 3 + 2 + 7
    assert L.x("tau") == L.a(1) + 1 and L.x("q1") == L.D
    lab = L.labels()
    assert lab[L.x("s1")] == "s1" and len(set(lab[1:])) == L.D


@pytest.mark.parametrize("st", ["ball", "box"])
@pytest.mark.parametrize("case", ["fixed_query", "query_in_keys", "query_not_in_keys"])
def test_true_points_feasible(st, case):
    """The rank-1 lift of any true configuration satisfies every constraint of every family."""
    rng = np.random.default_rng(1)
    for seed in range(2):
        inst = instances.f_random(seed, d=4, d_k=2, d_v=2, n=3, rho=0.4, set_type=st,
                                  query_in_keys=(case == "query_in_keys"))
        if case == "fixed_query":
            fixed_query(inst)
        ctrl = bounds.controls(inst)
        for _ in range(4):
            Z = sample_Z(inst, rng)
            r = solve_sdp(inst, ctrl, rlt=ALL, check_point=Z)
            assert r.diagnostics["max_violation"] <= 1e-9, r.diagnostics
            assert abs(r.value - r.diagnostics["true_value"]) <= 1e-10


@pytest.mark.parametrize("st", ["ball", "box"])
@pytest.mark.parametrize("case", ["fixed_query", "query_in_keys", "query_not_in_keys"])
def test_validity_and_monotonicity(st, case):
    """r >= multistart (validity); adding a family never increases the value (monotonicity)."""
    for seed in range(3):
        inst = instances.f_random(seed, d=4, d_k=2, d_v=2, n=3, rho=0.3, temp=1.0 + seed, set_type=st,
                                  query_in_keys=(case == "query_in_keys"))
        if case == "fixed_query":
            fixed_query(inst)
        ctrl = bounds.controls(inst)
        ms = multistart_lb(inst, 20, seed=seed)
        tol = 1e-6 * max(1.0, abs(ms.value))
        r1 = solve_sdp(inst, ctrl, rlt=S1)
        r2 = solve_sdp(inst, ctrl, rlt=S2)
        assert r1.status == r2.status == "optimal"
        assert r2.value >= ms.value - tol and r1.value >= ms.value - tol
        assert r2.value <= r1.value + tol, (r1.value, r2.value)
        for extra in ("softmax_x_ball", "score_rlt", "cut_x_z") + (("box_x_box",) if st == "box" else ()):
            r3 = solve_sdp(inst, ctrl, rlt=S2 + (extra,))
            assert r3.status == "optimal", (extra, r3.status)
            assert ms.value - tol <= r3.value <= r2.value + tol, (extra, r3.value, r2.value)
        if case == "fixed_query":
            assert not any(e.startswith("s") for e in r2.params["lifted_extras"])  # affine scores: no s lifted
        else:
            assert any(e.startswith("s") for e in r2.params["lifted_extras"])


def _order(kappas, gaps):
    return float(np.polyfit(np.log(kappas), np.log(gaps), 1)[0])


def test_prop9_softmax_x_z_order():
    """On the two-key instance ``prop9``, softmax_x_z turns the relaxation with the unit diagonal (order 1 in kappa)
    into order 2; with the interval diagonal (R4) the relaxation is already of order 2."""
    kappas = [0.025, 0.05, 0.1]
    g = {k: [] for k in ("s1_unit", "s2_unit", "s1", "s2")}
    for kappa in kappas:
        inst = instances.prop9(kappa)
        ctrl = bounds.controls(inst)
        p_star = inst.meta["nu_norm"] / (1 + math.exp(kappa))
        for key, fam, da in (("s1_unit", S1, "unit"), ("s2_unit", S2, "unit"), ("s1", S1, "interval"),
                             ("s2", S2, "interval")):
            r = solve_sdp(inst, ctrl, rlt=fam, diag_a=da, t_interval="loose")  # the loose log-attention intervals
            assert r.value >= p_star - 1e-7
            g[key].append(r.value - p_star)
            rt = solve_sdp(inst, ctrl, rlt=fam, diag_a=da, t_interval="tight")  # the intervals (14): valid, never worse
            assert p_star - 1e-7 <= rt.value <= r.value + 1e-7
    assert _order(kappas, g["s1_unit"]) < 1.2
    assert _order(kappas, g["s2_unit"]) > 1.9
    assert _order(kappas, g["s1"]) > 1.9 and _order(kappas, g["s2"]) > 1.9
    assert all(b <= a + 1e-7 for a, b in zip(g["s1"], g["s2"]))
