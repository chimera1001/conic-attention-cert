"""GPT-2: extraction gate, tests of the exact reduction to the head subspace (Proposition 15), role-reduced score
intervals, LayerNorm factor."""
import math

import numpy as np
import pytest

from aimp import bounds, gpt2
from aimp import sets as S
from aimp.exact_gurobi import solve_exact
from aimp.heads import forward
from aimp.types import HeadSpec, Instance


def _synthetic(seed=0, d=16, d_k=4, n=2, rho=0.05):
    rng = np.random.default_rng(seed)
    W_Q, W_K, W_V = (rng.standard_normal((d, d_k)) / math.sqrt(d) for _ in range(3))
    W_O = rng.standard_normal((d_k, d)) / math.sqrt(d_k)
    head = HeadSpec.from_weights(W_Q, W_K, W_V, W_O, name="syn")
    Z = rng.standard_normal((n, d))
    c = rng.standard_normal(d)
    c /= np.linalg.norm(c)
    return Instance(head=head, keys=list(range(n)), query=n - 1, sets={t: S.ball(Z[t], rho) for t in range(n)}, c=c)


def test_projection_exact_values_agree():
    """Projection test 1: exact values before/after projection agree (synthetic head d = 16, d_k = 4)."""
    inst = _synthetic()
    proj = gpt2.project_instance(inst)
    assert proj.meta["r"] == 12 and proj.d == 12
    # same head at corresponding points
    pr = gpt2.projection(inst.head.raw["W_Q"], inst.head.raw["W_K"], inst.head.raw["W_V"])
    rng = np.random.default_rng(1)
    for _ in range(5):
        Z = np.stack([S.project(inst.sets[t], inst.sets[t].center + rng.standard_normal(16)) for t in range(inst.T)])
        Zp = Z @ pr.P.T
        assert forward(proj, Zp)["phi"] == pytest.approx(forward(inst, Z)["phi"], abs=1e-12)
    # certified brackets [incumbent, ObjBound]: the d = 16 problem does not close in 30 s, its projection (r = 12) does
    ex0 = solve_exact(inst, bounds.controls(inst, threads=1), threads=1, time_limit=30)
    ex1 = solve_exact(proj, bounds.controls(proj, threads=1), threads=1, time_limit=60)
    assert ex1.status == "optimal", ex1.status
    assert max(ex0.value_lb_verified, ex1.value_lb_verified) <= min(ex0.value, ex1.value) + 1e-7  # brackets overlap
    assert ex0.value_lb_verified - 1e-7 <= ex1.value_lb_verified <= ex0.value + 1e-7  # projected optimum inside
    assert ex1.value_lb_verified == pytest.approx(ex0.value_lb_verified, abs=1e-6)  # same maximizer value
    # the projected maximizer lifts to a point of the original balls with the same value
    Zl = gpt2.lift_point(inst, ex1.recovered_point, pr)
    assert all(S.contains(inst.sets[t], Zl[t], 1e-7) for t in range(inst.T))
    assert forward(inst, Zl)["phi"] == pytest.approx(ex1.value_lb_verified, abs=1e-10)


def test_role_reduced_shor_equals_full_shor():
    inst = gpt2.project_instance(_synthetic(seed=3, n=3, rho=0.4))
    full = bounds.score_intervals(inst, "shor", threads=1)
    red = gpt2.score_intervals_roles(inst, threads=1)
    scale = max(1.0, float(np.max(np.abs(full["s_hi"]))))
    np.testing.assert_allclose(red["s_lo"], full["s_lo"], atol=1e-6 * scale)
    np.testing.assert_allclose(red["s_hi"], full["s_hi"], atol=1e-6 * scale)
    # and they contain sampled scores
    rng = np.random.default_rng(0)
    for _ in range(200):
        Zp = np.stack([inst.sets[t].center + inst.sets[t].E @ S.base_sample(S.affine(inst.sets[t]), rng)[0]
                       for t in range(inst.T)])
        s = forward(inst, Zp)["s"]
        assert np.all(s >= red["s_lo"] - 1e-9) and np.all(s <= red["s_hi"] + 1e-9)


def test_ln_factor_bounds_the_contraction():
    rng = np.random.default_rng(0)
    d = 32
    g = rng.uniform(0.2, 2.0, d)
    b = rng.standard_normal(d)
    x = rng.standard_normal((3, d)) * 3
    out = gpt2.ln_factor(x, g, 1e-5)
    for t in range(3):
        assert out["jac_norm"][t] <= out["f_l6"][t]
        rho_x = 0.3 * out["Px_norm"][t]
        for _ in range(200):
            u = rng.standard_normal(d)
            u *= rho_x * rng.uniform() ** (1 / d) / np.linalg.norm(u)
            dz = gpt2.layernorm(x[t] + u, g, b, 1e-5) - gpt2.layernorm(x[t], g, b, 1e-5)
            assert np.linalg.norm(dz) <= out["f_l6"][t] * rho_x * (1 + 1e-12)


@pytest.fixture(scope="module")
def gpt2_prompt0():
    model, tok = gpt2.load_model()
    W = gpt2.all_weights(model)
    ext = gpt2.extract(model, tok, gpt2.load_prompts()[0])
    return ext, W


def test_extraction_gate(gpt2_prompt0):
    """Reproduction gate: per-head attention output and full block output reproduced to 1e-10."""
    ext, W = gpt2_prompt0
    g = gpt2.gate(ext, W)
    assert g["passed"], g["worst"]


@pytest.mark.parametrize("layer,head,pos", [(0, 0, 4), (5, 1, 8), (11, 7, 12)])
def test_projected_forward_matches_hooked(gpt2_prompt0, layer, head, pos):
    """Projection test 2: the head in projected coordinates reproduces the hooked one to 1e-10."""
    ext, W = gpt2_prompt0
    hw = gpt2.head_weights(W[layer], head)
    z = ext["z"][layer]
    ybar = ext["merged"][layer][pos, head * 64:(head + 1) * 64] @ hw["W_O"]
    inst = gpt2.phase3_instance(z, hw, pos, 0.01, gpt2.spec_vectors(hw["W_O"], ybar, layer, head)["a"])
    out = forward(inst, inst.centers())
    assert np.max(np.abs(out["y"] - ybar)) <= 1e-10
    assert np.max(np.abs(out["a"] - ext["A"][layer][head, pos, :pos + 1])) <= 1e-10
    assert out["phi"] == pytest.approx(np.linalg.norm(ybar), abs=1e-10)
