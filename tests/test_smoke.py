"""End-to-end smoke tests: certify one synthetic head and one GPT-2 head."""
import math

import numpy as np
import pytest


def test_synthetic_head_end_to_end():
    """A random head with three keys in balls: the certificate is valid against Gurobi's exact maximum, never above the
    first-order certificate, and within the second-order gap bound."""
    from aimp import bounds, instances
    from aimp.certify import certify
    from aimp.exact_gurobi import solve_exact

    inst = instances.phase1_instance("F_random", 0, 3, 0.05, "ball", True, 1.0)
    ctrl = bounds.controls(inst, threads=1)
    rec = certify(inst, ctrl, eps_target=0.0, threads=1, cut="auto")
    assert rec["route"] == "min" and rec["value_safe"] is not None
    ex = solve_exact(inst, ctrl, "log", threads=1, time_limit=120)
    p_lo = ex.value_lb_verified  # a feasible value (numpy re-evaluation of Gurobi's incumbent)
    assert p_lo is not None and p_lo <= ex.value + 1e-6  # incumbent <= Gurobi's bound
    assert rec["cert"] >= p_lo - 1e-6
    assert rec["cert"] <= rec["r_greedy"] + 1e-12
    assert rec["cert"] - p_lo <= rec["bound_F0t"] + 1e-6


@pytest.fixture(scope="module")
def gpt2_ready():
    pytest.importorskip("transformers")
    from aimp import gpt2_units as U

    try:
        g = U.prepare()
    except Exception as e:  # no network and no cache
        pytest.skip(f"GPT-2 weights unavailable: {e!r}")
    assert g["passed"], g["worst"]
    return U


def test_gpt2_head_end_to_end(gpt2_ready):
    """Layer 0, head 1, prompt 0, query position 4, rho_rel = 0.001 with the procedure of Appendix B: the certificate is
    valid against the best feasible value, at most the first-order certificate, inside the second-order bound (39), and
    the relative gap is small (the unit is in the second-order regime)."""
    from aimp.pipeline import evaluate_unit

    rec = evaluate_unit(0, 1, 0, 4, 0.001, threads=min(4, int(__import__("os").environ.get("AIMP_MAX_THREADS", "4"))))
    assert rec["route"] == "min" and rec["sdp_status"] == "optimal"
    assert rec["valid"] and rec["valid_value_safe"] and rec["valid_greedy"]
    assert rec["cert"] <= rec["r_greedy"] + 1e-12
    assert rec["checks"]["F0v_final"] == "satisfied"
    assert rec["w_t"] < 1.0 and 0 <= rec["rel_gap"] < 0.1
    assert rec["rel_gap"] < rec["rel_gap_greedy"]


def test_gpt2_fixed_query_exact(gpt2_ready):
    """The exact worst case with a fixed query (Theorem 7): brackets of width <= 1e-10 x range, the worst-case inputs
    inside them, on the ball boundaries and in the canonical plane; the first-order certificate above the maximum."""
    from aimp.pipeline import evaluate_fq_exact

    r = evaluate_fq_exact(5, 1, 0, 4, 0.01)
    assert r["exact_status"] == "exact"
    assert r["max_input_in_bracket"] and r["min_input_in_bracket"]
    assert r["p_fq_hi"] - r["p_fq_lo"] <= 1.1e-10 * r["range"]
    assert np.allclose(r["norm_max_rel"], 1.0, atol=1e-12) and np.allclose(r["norm_min_rel"], 1.0, atol=1e-12)
    assert r["off_plane_max"] <= 1e-12 * max(1.0, max(r["rho_j"])) and r["off_plane_min"] <= 1e-12 * max(1.0, max(r["rho_j"]))
    assert r["r_greedy"] >= r["p_fq_lo"] - 1e-9
    assert r["min_fq_hi"] <= r["phi_nominal"] <= r["p_fq_lo"] + 1e-12
    assert math.isfinite(r["spread_fq"]) and r["spread_fq"] > 0
