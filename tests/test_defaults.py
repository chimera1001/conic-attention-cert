"""Defaults: the width-pair cut, the inherited margin bound (35), and the per-key tolerance of the exact fixed-query
solver (Theorem 7)."""
import math

import numpy as np
import pytest

from aimp import bounds, instances
from aimp.bounds import DEFAULT_CUT, choose_cut, softmax_interval_bounds
from aimp.relax_ps import solve_ps_greedy


def test_default_cut_is_width_pair():
    assert DEFAULT_CUT == "B"
    inst = instances.f_random(3, d=4, d_k=2, d_v=2, n=4, temp=2.0, rho=0.05, set_type="ball")
    ctrl = bounds.controls(inst, threads=1)
    g = solve_ps_greedy(inst, ctrl)
    assert g.diagnostics["cut_used"] == "B"
    assert g.value == pytest.approx(solve_ps_greedy(inst, ctrl, "B").value, abs=0)


def test_margin_cut_implied_by_intervals():
    """Proof of Theorem 9: ``2 (1 - a_lo_{j*}) <= eps_C(n, gamma)`` whenever ``gamma > 0``, so every point of box ∩
    simplex satisfies the margin pair's l1 cut and the width-pair polytope inherits the margin bound (35)
    (``eps_for_bound``)."""
    rng = np.random.default_rng(1829)
    seen = 0
    for _ in range(2000):
        n = int(rng.integers(2, 9))
        c = rng.normal(0, 3, n)
        w = rng.exponential(0.5, n)
        lo, hi = c - w, c + w
        gs = [lo[k] - np.max(np.delete(hi, k)) for k in range(n)]
        js = int(np.argmax(gs))
        gam = float(gs[js])
        if gam <= 0:
            continue
        seen += 1
        a_lo = softmax_interval_bounds(lo, hi)["a_lo"]
        eC = bounds.eps_C(n, gam)
        assert 2 * (1 - a_lo[js]) <= eC * (1 + 1e-12) + 1e-15
        ctrl = {"ahat": np.full(n, 1 / n), "eps_B": float(np.max(hi - lo)) / 2, "gamma": gam, "j_star": js, "eps_C": eC}
        ci = choose_cut(ctrl, n, "B")
        assert ci["cut_used"] == "B" and ci["eps"] == ctrl["eps_B"]
        assert ci["eps_for_bound"] == pytest.approx(min(ctrl["eps_B"], eC), rel=0, abs=0)
    assert seen > 200


def test_gprime_per_key_tolerance():
    """``tol_key = tol_rel x range / n_keys``; the bracket stays certified and lies within the bracket without the
    per-key rule (``per_key_tol=False``) up to the tolerance; at moderate n the per-key rule makes the bracket exact."""
    from aimp.exact_fq import fix_query, fq_problem, solve_max

    inst = fix_query(instances.f_random(17_7_500, d=8, d_k=4, d_v=4, n=400, temp=1.0, rho=0.1, set_type="ball",
                                        query_in_keys=True))
    P = fq_problem(inst)
    new = solve_max(P)
    old = solve_max(P, per_key_tol=False)
    assert new["tol_key"] == pytest.approx(1e-10 * P.range / P.n)
    assert old["tol_key"] is None
    assert new["status"] == "exact"
    tol = 1e-12 * max(1.0, abs(old["p_hi"]))
    assert new["p_lo"] >= old["p_lo"] - 1e-10 * P.range - tol and new["p_hi"] <= old["p_hi"] + 1e-10 * P.range + tol
    assert max(new["p_lo"], old["p_lo"]) <= min(new["p_hi"], old["p_hi"]) + tol
