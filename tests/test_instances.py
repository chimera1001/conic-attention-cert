"""Synthetic instance families (Section 7 of the paper): construction properties, determinism, ids; ibp_obj."""
import math

import numpy as np
import pytest

from aimp import bounds, instances as I, sets as S
from aimp.ibp import solve_ibp, solve_ibp_obj
from aimp.lower_bounds import multistart_lb
from aimp.relax_ps import solve_ps

QK = [True, False]


@pytest.mark.parametrize("qk", QK)
@pytest.mark.parametrize("temp", [0.1, 1.0, 10.0])
def test_f_random_norms(temp, qk):
    for st in ("ball", "box"):
        inst = I.p1_random(3, 3, 0.1, st, qk, temp=temp)
        assert math.isclose(np.linalg.norm(inst.head.M, 2), temp, rel_tol=1e-12)
        assert math.isclose(np.linalg.norm(inst.head.N, 2), 1.0, rel_tol=1e-12)
        assert np.linalg.matrix_rank(inst.head.M) == 4
        assert np.allclose(np.linalg.norm(inst.centers(), axis=1), np.sqrt(8))
        assert math.isclose(np.linalg.norm(inst.c), 1.0)
        for t in inst.tokens:
            assert math.isclose(S.circumradius(inst.sets[t]), 0.1, rel_tol=1e-12)
        assert inst.query_in_keys == qk and inst.meta["family"] == "F_random"


def test_f_rank1():
    inst = I.phase1_instance("F_rank1", 0, 3, 0.1, "ball", True, 1.0)
    assert np.linalg.matrix_rank(inst.head.M) == 1 and inst.meta["d_k"] == 1


@pytest.mark.parametrize("qk", QK)
@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("gamma0", [1.0, 8.0])
def test_f_margin(gamma0, n, qk):
    inst = I.p1_margin(1, n, 0.05, "ball", qk, gamma0=gamma0)
    Z, M, i = inst.centers(), inst.head.M, inst.query
    sbar = np.array([Z[i] @ M @ Z[j] for j in inst.keys])
    k = inst.keys.index(inst.meta["j_star"])
    assert inst.meta["j_star"] != i
    assert abs(sbar[k] - np.max(np.delete(sbar, k)) - gamma0) <= 1e-9


@pytest.mark.parametrize("qk", QK)
@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("sep", [1.0, 0.25])
def test_f_ambiguous(sep, n, qk):
    rho = 0.05
    inst = I.p1_ambiguous(2, n, rho, "ball", qk, sep=sep)
    Z, M, N, i = inst.centers(), inst.head.M, inst.head.N, inst.query
    sbar = np.array([Z[i] @ M @ Z[j] for j in inst.keys])
    assert abs(sbar[0] - sbar[1]) <= 1e-9
    if n > 2:
        assert np.all(sbar[2:] <= sbar[0] - 4 + 1e-9)
    v = N @ (Z[0] - Z[1])
    assert np.allclose(N @ Z[1], (1 - 2 * sep) * (N @ Z[0]))
    assert np.allclose(inst.c, v / np.linalg.norm(v))
    mid = (N @ Z[0] + N @ Z[1]) / 2
    for k in range(2, n):
        assert np.allclose(N @ Z[k], mid)  # other values at the pair's midpoint
    if qk:
        assert i == 0
    ctrl = bounds.controls(inst)
    if sep == 1.0:  # the pair spans the value set: ||N(z1 - z2)|| = 2 rad up to the radius term
        assert np.linalg.norm(v) <= 2 * ctrl["rad"] + 1e-9
        assert np.linalg.norm(v) >= 2 * ctrl["rad"] - 2 * rho * np.linalg.norm(N, 2) - 1e-6


@pytest.mark.parametrize("rho", [0.05, 0.5])
def test_f_prop9_embedded(rho):
    vals = []
    for seed in range(3):
        inst = I.p1_prop9(seed, rho, "ball")
        ctrl = bounds.controls(inst)
        assert abs(ctrl["Delta"] - 2 * rho) <= 1e-7  # Delta = 2 kappa_M rho, kappa_M = 1
        assert abs(ctrl["rad"] - rho) <= 1e-7  # value segment [-rho, rho] nu e1, nu = 1
        ms = multistart_lb(inst, 10, seed=seed)
        p_star = rho / (1 + math.exp(rho))  # nu rho sigma(-kappa_M rho)
        assert abs(ms.value - p_star) <= 1e-8
        vals.append(solve_ps(inst, ctrl, "B").value)
    assert max(vals) - min(vals) <= 1e-7  # rotation invariance (balls)


def test_ibp_obj_valid_and_tighter_than_ibp():
    for seed in range(5):
        for st in ("ball", "box"):
            inst = I.p1_random(seed, 3, 0.2, st, seed % 2 == 0)
            ctrl = bounds.controls(inst)
            ms = multistart_lb(inst, 10, seed=seed)
            o, b = solve_ibp_obj(inst, ctrl), solve_ibp(inst, ctrl)
            assert o.value >= ms.value - 1e-9 and o.diagnostics["value_simplex"] >= ms.value - 1e-9
            assert o.diagnostics["value_simplex"] <= o.value + 1e-12
            assert o.value <= b.value + 1e-9
