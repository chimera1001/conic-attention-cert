"""Regression on the two-key instance :func:`aimp.instances.prop9` (of the type of the sharpness instances of
Proposition 10): exact worst case, score width, value radius, and the first-order gap ``tanh(kappa/2) ||nu||``."""
import math

import pytest

from aimp import bounds, instances
from aimp.exact_gurobi import solve_exact
from aimp.relax_ps import solve_ps


def sig(x):
    return 1 / (1 + math.exp(-x))


@pytest.mark.parametrize("kappa", [0.05, 0.1, 0.2])
def test_prop9(kappa):
    inst = instances.prop9(kappa)
    nu = inst.meta["nu_norm"]
    ctrl = bounds.controls(inst)
    ex = solve_exact(inst, ctrl, time_limit=60)
    p_star = sig(-kappa) * nu
    assert ex.status == "optimal"
    assert abs(ex.value_lb_verified - p_star) <= 1e-6
    assert abs(ex.value - p_star) <= 1e-6
    assert abs(ctrl["Delta"] - 2 * kappa) <= 1e-12
    assert abs(ctrl["rad"] - nu) <= 1e-6
    bound = 2 * kappa * nu
    assert abs(ctrl["bound_B"] - bound) <= 1e-6
    out = {}
    for backend in ("gurobi", "cvxpy"):
        r_full = solve_ps(inst, ctrl, cut="B", backend=backend)
        r_cut = solve_ps(inst, ctrl, cut="B", backend=backend, interval_bounds=False)
        assert abs(r_full.value - sig(kappa) * nu) <= 1e-6
        assert abs(r_cut.value - (0.5 + kappa / 2) * nu) <= 1e-6
        assert abs((r_full.value - p_star) - math.tanh(kappa / 2) * nu) <= 1e-6
        for r in (r_full, r_cut):
            assert r.value - p_star <= bound + 1e-9
        out[backend] = (r_full.value, r_cut.value)
    print(f"\nkappa={kappa}: p*={p_star:.9f} (exact lb {ex.value_lb_verified:.9f}, ub {ex.value:.9f}) "
          f"Delta={ctrl['Delta']:.6f} rad={ctrl['rad']:.6f} r_ps(cut+ib)={out['gurobi'][0]:.9f}/{out['cvxpy'][0]:.9f} "
          f"r_ps(cut)={out['gurobi'][1]:.9f}/{out['cvxpy'][1]:.9f} bound={bound:.6f}")
