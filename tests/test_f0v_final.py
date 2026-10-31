"""The second-order gap bound (39) of Theorem 12 — the exact secant gap ``D_j`` (Lemma 3), the fractional knapsack (19),
and the bound against Gurobi on the 50 shared synthetic instances.

(1) ``D_j`` (:func:`theory.secant_gap`) equals the maximum of ``ell_j - exp`` on a dense grid (60-digit mpmath, zoomed
around the best grid point until the grid spacing is negligible) to 1e-12 relative on random intervals with ``w`` from
1e-8 to 60; (2) ``D_j <= min(S_j, B_j - A_j)``; (3) the knapsack equals an LP (scipy ``linprog``) on random data;
(4) bound (39) holds against the exact bracket (never ``violated``; ``satisfied`` when Gurobi proves optimality) and is
never above the per-key-slack form ``bound_F0v`` nor its saturating form (``theory.saturated_slack``)."""
import math

import mpmath as mp
import numpy as np
import pytest
from scipy.optimize import linprog

from aimp.bounds import t_intervals
from aimp.certify import theorem_check
from aimp.lower import neg_instance, strong_bracket
from aimp.relax_ps import solve_ps_greedy
from aimp.theory import (bound_F0v_terms, f0v_final, f0v_final_fields, knapsack, perkey_fields, saturated_slack,
                         secant_gap)
from tests.synthetic_units import SEEDS, synthetic_unit


def _grid_max(t_lo: float, t_hi: float, pts: int = 201, zooms: int = 40) -> mp.mpf:
    """``max_{t in [t_lo, t_hi]} (ell(t) - e^t)`` by a dense grid in 60-digit arithmetic, re-gridded around the best point
    (the function is concave, so the zoomed window always contains the maximizer)."""
    with mp.workdps(60):
        lo, hi = mp.mpf(t_lo), mp.mpf(t_hi)
        A, B = mp.e ** lo, mp.e ** hi
        m = (B - A) / (hi - lo)
        f = lambda t: A + m * (t - lo) - mp.e ** t  # noqa: E731
        a, b = lo, hi
        best = mp.mpf(0)
        for _ in range(zooms):
            h = (b - a) / (pts - 1)
            vals = [f(a + k * h) for k in range(pts)]
            k = max(range(pts), key=lambda q: vals[q])
            best = max(best, vals[k])
            a, b = max(lo, a + (k - 1) * h), min(hi, a + (k + 1) * h)
            if b - a <= (hi - lo) * mp.mpf(10) ** -30:
                break
        return best


def test_secant_gap_matches_dense_grid():
    rng = np.random.default_rng(17_2)
    ws = list(10 ** rng.uniform(-8, math.log10(60), 150)) + [1e-8, 1e-4, 9.99e-3, 1.001e-2, 1.0, 29.99, 30.01, 60.0]
    worst = 0.0
    for w in ws:
        t_lo = float(rng.uniform(-30, 0))
        t_hi = t_lo + float(w)
        D = secant_gap(t_lo, t_hi)
        ref = _grid_max(t_lo, t_hi)
        rel = float(abs(mp.mpf(D) - ref) / ref)
        worst = max(worst, rel)
        assert rel <= 1e-12, (w, t_lo, D, ref)
    assert worst <= 1e-12
    assert secant_gap(-1.0, -1.0) == 0.0 and secant_gap(-1.0, -2.0) == 0.0
    # D ~ A w^2 / 8 as w -> 0
    assert secant_gap(-2.0, -2.0 + 1e-6) == pytest.approx(math.exp(-2.0) * 1e-12 / 8, rel=1e-5)


def test_secant_gap_below_S_and_B_minus_A():
    rng = np.random.default_rng(3)
    for _ in range(3000):
        w = float(10 ** rng.uniform(-8, 2.5))
        t_hi = float(-rng.exponential(2.0))
        t_lo = t_hi - w
        w = t_hi - t_lo  # the width actually represented
        A, B = math.exp(t_lo), math.exp(t_hi)
        D = secant_gap(t_lo, t_hi)
        S = B * w * w / 8  # S_j with a_hi_j = B_j (tight interval: t_hi = log a_hi)
        assert 0.0 <= D <= min(S, B - A) * (1 + 1e-12) + 1e-300, (w, t_lo, D, S, B - A)


def test_knapsack_equals_lp():
    rng = np.random.default_rng(5)
    for _ in range(300):
        n = int(rng.integers(1, 15))
        chi = rng.normal(0, 1, n) * rng.choice([0.0, 1.0], n, p=[0.15, 0.85])
        D = rng.uniform(0, 1, n) * 10 ** rng.uniform(-6, 0, n)
        D_tot = float(rng.uniform(0, 1.3) * D.sum())
        val, d = knapsack(chi, D, D_tot)
        sc = float(D.max())  # scaled LP (HiGHS' feasibility tolerance is absolute, ~1e-7)
        lp = linprog(-chi, A_ub=np.ones((1, n)), b_ub=[D_tot / sc], bounds=list(zip(np.zeros(n), D / sc)),
                     method="highs", options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
        assert lp.status == 0
        assert val == pytest.approx(-lp.fun * sc, rel=1e-8, abs=1e-12 * sc)
        assert np.all(d >= 0) and np.all(d <= D + 1e-15) and d.sum() <= D_tot + 1e-12
        assert float(d @ chi) == pytest.approx(val, rel=1e-12, abs=1e-15)


def test_f0v_final_terms_and_tightening():
    """Hand-checked unit: two keys, budget binding."""
    t_lo = np.log([0.2, 0.3])
    t_hi = np.log([0.6, 0.7])
    out = f0v_final(t_lo, t_hi, [1.0, 2.0], np.exp(t_lo), np.exp(t_hi), [0.1, 0.2], phi_min_cert=0.5,
                    phi_min_strong=0.6, eta=0.0, rad_c=1.0)
    D = [secant_gap(a, b) for a, b in zip(t_lo, t_hi)]
    assert out["D_tot"] == pytest.approx(0.5)
    assert out["F0vf_term_knapsack"] == pytest.approx(D[1] * 1.5 + min(D[0], 0.5 - D[1]) * 0.5, rel=1e-12)
    assert out["Pi_v_star"] == pytest.approx(D[1] * 1.4 + min(D[0], 0.5 - D[1]) * 0.4, rel=1e-12)
    assert out["F0vf_term_cov"] == pytest.approx(0.5 * (0.4 * 0.1 + 0.4 * 0.2))


@pytest.mark.parametrize("seed", SEEDS)
def test_F0v_final_against_exact(seed):
    u = synthetic_unit(seed)
    inst, ctrl, nom, rec = u["inst"], u["ctrl"], u["nom"], u["rec"]
    slack = rec.get("pruning_slack") or 0.0
    phi_min_cert = -solve_ps_greedy(neg_instance(inst), ctrl, "auto").value
    br = strong_bracket(inst, seed, n_random=4, maxiter=300)
    ff = f0v_final_fields(inst, ctrl, phi_min_cert=phi_min_cert, phi_min_strong=br["phi_min_strong"], eta=nom["eta"],
                          pruning_slack=slack)
    pk = perkey_fields(inst, ctrl, phi_min_strong=br["phi_min_strong"], phi_min_cert=phi_min_cert, eta=nom["eta"],
                       pruning_slack=slack)
    # tight interval = [log a_lo, log a_hi]
    t_lo, t_hi = t_intervals(ctrl["s_lo"], ctrl["s_hi"], "tight")
    assert np.allclose(np.exp(t_lo), np.clip(ctrl["a_lo"], 0, 1), rtol=1e-12, atol=1e-300)
    assert np.allclose(np.exp(t_hi), np.clip(ctrl["a_hi"], 0, 1), rtol=1e-12, atol=1e-300)
    # supersedes S_j, min(S_j, a_hi - a_lo) and the saturating form: never larger
    D_sat = saturated_slack(pk["perkey"]["S_j"], ctrl["a_lo"], ctrl["a_hi"])
    assert np.all(ff["D_j"] <= D_sat * (1 + 1e-12) + 1e-300)
    sat = bound_F0v_terms(D_sat, pk["perkey"]["h_max"], ctrl["a_lo"], ctrl["a_hi"], pk["perkey"]["nu_c"], phi_min_cert,
                          nom["eta"], pk["rad_c"], slack)["bound_F0v"]
    assert ff["bound_F0v_final"] <= sat * (1 + 1e-12) + 1e-15
    assert ff["bound_F0v_final"] <= pk["bound_F0v"] * (1 + 1e-12) + 1e-15
    assert ff["Pi_v_star"] <= pk["Pi_v"] * (1 + 1e-12) + 1e-15
    # bound (39) against the exact bracket
    lb, ub = u["lb"], u["ub"]
    ch = theorem_check(rec["value_safe"], ff["bound_F0v_final"], lb, ub)
    assert ch["status"] != "violated", (seed, ch)
    if u["exact_status"] == "optimal":
        assert ch["status"] == "satisfied", (seed, ch)
