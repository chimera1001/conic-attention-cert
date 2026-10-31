"""The value-weighted sandwich identity (27) of Proposition 4 and the second-order gap bound with the per-key slacks
``S_j`` (:func:`aimp.theory.bound_F0v_terms`).

(1) The identity ``sum_j (abar_j - sigma_j) h_j = sum_j d_j (h_j - hbar_sigma)`` holds to 1e-12 on random
``(e, d, h)`` with ``sum (e + d) = 1``, and with ``0 <= d_j <= S_j`` the left side is ``<= sum_j S_j (h_j - hbar_sigma)_+``.
(2) The bound holds against Gurobi exact on the 50 shared synthetic instances (three-valued with the exact bracket;
never ``violated``, ``satisfied`` whenever Gurobi proves optimality), and it is never worse than the directional
total-variation form ``bound_F0s_c`` by more than the ``eta S rad_c`` term."""
import math

import numpy as np
import pytest

from aimp.certify import theorem_check
from aimp.lower import neg_instance, strong_bracket
from aimp.relax_ps import solve_ps_greedy
from aimp.theory import (W_of, bound_F0v_terms, directional_constants, directional_fields, p7v_sides, perkey_S,
                         perkey_fields, sandwich_S, saturated_slack, secant_widths, value_scores)
from tests.synthetic_units import SEEDS, synthetic_unit


def _draw(rng, n):
    e = rng.uniform(0.01, 1.0, n) * np.exp(rng.normal(0, 2, n))
    d = rng.uniform(0, 1, n) * rng.choice([0.0, 1.0], n, p=[0.3, 0.7]) * e.sum() * rng.uniform(0, 0.5)
    s = (e + d).sum()
    return e / s, d / s


def test_p7v_identity_random():
    rng = np.random.default_rng(16_2)
    worst = 0.0
    for _ in range(2000):
        n = int(rng.integers(2, 14))
        e, d = _draw(rng, n)
        assert (e + d).sum() == pytest.approx(1.0, abs=1e-15)
        h = rng.normal(0, 1, n) * 10 ** rng.uniform(-2, 2)
        lhs, rhs = p7v_sides(e, d, h)
        scale = max(1.0, float(np.abs(h).max()))
        worst = max(worst, abs(lhs - rhs) / scale)
        assert abs(lhs - rhs) <= 1e-12 * scale, (lhs, rhs)
        # the bound: 0 <= d_j <= S_j  =>  sum (abar - sigma) h <= sum S_j (h_j - hbar_sigma)_+
        Sj = d * rng.uniform(1.0, 3.0, n)
        hs = float((e / e.sum()) @ h)
        assert lhs <= float(Sj @ np.maximum(h - hs, 0.0)) + 1e-12 * scale
    assert worst <= 1e-12


def test_p7v_identity_edge_cases():
    # d = 0 (exact softmax): both sides 0; a single dominant key; constant h (both sides 0)
    e = np.array([0.5, 0.3, 0.2])
    assert p7v_sides(e, np.zeros(3), np.array([1.0, -2.0, 3.0])) == pytest.approx((0.0, 0.0), abs=1e-15)
    e, d = np.array([0.9, 1e-12, 1e-12]), np.array([0.1 - 4e-12, 1e-12, 1e-12])  # sum (e + d) = 1
    l, r = p7v_sides(e, d, np.array([2.0, -5.0, 7.0]))
    assert l == pytest.approx(r, abs=1e-12)
    e, d = np.array([0.4, 0.2, 0.1]), np.array([0.1, 0.1, 0.1])
    assert p7v_sides(e, d, np.full(3, 4.2)) == pytest.approx((0.0, 0.0), abs=1e-12)


def test_perkey_arrays_consistent():
    u = synthetic_unit(3)
    inst, ctrl = u["inst"], u["ctrl"]
    Sj = perkey_S(ctrl)
    assert Sj.sum() == pytest.approx(sandwich_S(ctrl), rel=1e-12)
    w = secant_widths(ctrl)
    assert w.max() == pytest.approx(ctrl["w_t"], rel=1e-12)
    dc = directional_constants(inst)
    # balls: h_j^max = hbar_j + rho_j ||g_j||
    assert dc["h_sup"] == pytest.approx(dc["hbar"] + dc["rho_g"], rel=1e-12, abs=1e-12)
    P = value_scores(Sj, dc["h_sup"], float(dc["h_sup"].min()) - 1.0)
    assert np.all(P >= 0) and np.all(P >= Sj * 1.0 - 1e-15)


@pytest.mark.parametrize("seed", SEEDS)
def test_F0v_against_exact(seed):
    u = synthetic_unit(seed)
    inst, ctrl, nom, rec = u["inst"], u["ctrl"], u["nom"], u["rec"]
    slack = rec.get("pruning_slack") or 0.0
    phi_min_cert = -solve_ps_greedy(neg_instance(inst), ctrl, "auto").value
    br = strong_bracket(inst, seed, n_random=4, maxiter=300)
    pk = perkey_fields(inst, ctrl, phi_min_strong=br["phi_min_strong"], phi_min_cert=phi_min_cert, eta=nom["eta"],
                       pruning_slack=slack)
    dc = directional_constants(inst)
    # phi_min_cert is a certified lower bound on min phi (<= a feasible value) and >= the bottom of the value range
    assert phi_min_cert <= br["phi_min_strong"] + 1e-9
    assert phi_min_cert >= dc["m_c"] - dc["rad_c"] - 1e-9
    # never worse than bound_F0s_c up to the eta S rad_c term (S < 1); per-key cov term <= nu_c W / 2
    d = directional_fields(inst, ctrl, nom, pruning_slack=slack)
    assert pk["F0v_term_cov"] <= 0.5 * d["nu_c"] * W_of(ctrl) * (1 + 1e-12) + 1e-15
    if d.get("bound_F0s_c") is not None:
        assert pk["bound_F0v"] <= d["bound_F0s_c"] + nom["eta"] * pk["S"] * d["rad_c"] + 1e-12 * max(1.0, d["bound_F0s_c"])
    # Pi_v uses phi_min_strong; the bound's sandwich term uses phi_min_cert (smaller), so it dominates Pi_v
    assert pk["Pi_v"] <= pk["F0v_term_sandwich"] + 1e-12 * max(1.0, pk["F0v_term_sandwich"])
    assert pk["bound_F0v"] == pytest.approx(bound_F0v_terms(
        pk["perkey"]["S_j"], pk["perkey"]["h_max"], ctrl["a_lo"], ctrl["a_hi"], pk["perkey"]["nu_c"], phi_min_cert,
        nom["eta"], dc["rad_c"], slack)["bound_F0v"], rel=1e-14)
    # the bound against the exact bracket
    lb, ub = u["lb"], u["ub"]
    assert lb is not None and math.isfinite(lb)
    ch = theorem_check(rec["value_safe"], pk["bound_F0v"], lb, ub)
    assert ch["status"] != "violated", (seed, ch)
    if u["exact_status"] == "optimal":
        assert ch["status"] == "satisfied", (seed, ch)
    # saturating form (theory.saturated_slack): D_j = min(S_j, a_hi_j - a_lo_j) in place of S_j; <= the bound, and valid
    D = saturated_slack(pk["perkey"]["S_j"], ctrl["a_lo"], ctrl["a_hi"])
    sat = bound_F0v_terms(D, pk["perkey"]["h_max"], ctrl["a_lo"], ctrl["a_hi"], pk["perkey"]["nu_c"], phi_min_cert,
                          nom["eta"], dc["rad_c"], slack)["bound_F0v"]
    assert sat <= pk["bound_F0v"] + 1e-15
    ch = theorem_check(rec["value_safe"], sat, lb, ub)
    assert ch["status"] != "violated", (seed, ch)
    if u["exact_status"] == "optimal":
        assert ch["status"] == "satisfied", (seed, ch)
