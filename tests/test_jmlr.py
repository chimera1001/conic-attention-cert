"""The two extensions of ``experiments/jmlr_extensions.py``.

Vertex-Softmax (:func:`aimp.bounds.vertex_softmax`): (a) it equals the brute-force maximum over all ``2^n`` vertices of
the score box for ``n <= 8``, dense sampling inside the box never exceeds it, and it is evaluated in log space; (b) on the
instance of the sharpness proposition (two keys, fixed query, one token on a segment, ``lambda = 1/2``) it equals the
first-order certificate ``r_pp``, with the same gap; (c) ``r_VS <= r_pp`` on random synthetic instances of every family,
and ``r_VS`` lies above sampled head values; ``bounds.controls`` carries it.

Logit-margin specification: ``spec="a"`` builds the instance of the paper; ``spec="lm"`` uses ``c = -c_m``; the numpy
logits reproduce the model's; ``c_m`` is the frozen-scale gradient of the logit difference (these need the model and are
skipped without it)."""
import itertools
import math

import numpy as np
import pytest

from aimp import bounds
from aimp import sets as S
from aimp.heads import forward, logsumexp
from aimp.types import HeadSpec, Instance

FAMILIES = {"F_random": 1.0, "F_margin": 2.0, "F_ambiguous": 1.0, "F_rank1": 1.0, "F_prop9": 1.0}


def _brute(h, lo, hi):
    best = -math.inf
    for bits in itertools.product((0, 1), repeat=h.shape[0]):
        s = np.where(np.array(bits, bool), hi, lo)
        best = max(best, float(np.exp(s - logsumexp(s)) @ h))
    return best


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_vertex_softmax_brute_force_and_sampling(n):
    rng = np.random.default_rng(100 + n)
    for trial in range(20):
        h = rng.standard_normal(n) * rng.choice([0.1, 1.0, 10.0])
        if trial % 5 == 0 and n > 1:
            h[1] = h[0]
        lo = rng.standard_normal(n) * rng.choice([0.1, 1.0, 5.0])
        hi = lo + rng.exponential(1.0, n) * rng.choice([0.01, 1.0, 20.0])
        v = bounds.vertex_softmax(h, lo, hi)
        ref = _brute(h, lo, hi)
        assert abs(v["value"] - ref) <= 1e-12 * max(1.0, abs(ref))
        assert abs(v["value"] - bounds.l3_max_vsigma(h, lo, hi)[0]) <= 1e-12 * max(1.0, abs(ref))
        Sm = lo + rng.uniform(0.0, 1.0, (3000, n)) * (hi - lo)
        E = np.exp(Sm - Sm.max(axis=1, keepdims=True))
        assert float(((E / E.sum(axis=1, keepdims=True)) @ h).max()) <= v["value"] + 1e-12 * max(1.0, abs(ref))


def test_vertex_softmax_log_space():
    h = np.array([1.0, -2.0, 0.5])
    lo, hi = np.array([800.0, 790.0, -900.0]), np.array([805.0, 799.0, 1000.0])
    v = bounds.vertex_softmax(h, lo, hi)
    assert v["k"] == 1 and abs(v["value"] - (1.0 - 3.0 * math.exp(-15.0))) <= 1e-12


def _sharp(lam: float, rho: float) -> Instance:
    """The two-key instance of the sharpness proposition: key 0 = ``e_2 + t e_3``, ``|t| <= rho``; the fixed query
    ``e_1`` is key 1; ``M = lambda e_1 e_3^T``, ``N = -e_1 (e_1 + 2 e_3)^T``, ``c = e_1``; scores ``(lambda t, 0)``,
    values ``(-2t, -1)``."""
    e = np.eye(3)
    head = HeadSpec(M=lam * np.outer(e[0], e[2]), N=-np.outer(e[0], e[0] + 2 * e[2]), d=3, d_k=3, d_v=3, name="sharp")
    sets = {0: S.ellipsoid(e[1], e[2][:, None], float(rho)), 1: S.point(e[0])}
    return Instance(head=head, keys=[0, 1], query=1, sets=sets, c=e[0].copy())


@pytest.mark.parametrize("rho", [0.01, 0.1, 0.5])
def test_sharpness_instance_vs_equals_rpp(rho):
    from aimp.relax_ps import solve_ps_greedy

    lam = 0.5
    inst = _sharp(lam, rho)
    si = {"method": "exact_linear", "s_lo": np.array([-lam * rho, 0.0]), "s_hi": np.array([lam * rho, 0.0])}
    ctrl = bounds.controls(inst, intervals=si, eps_target=0.0, threads=1)
    for cut in ("B", "auto"):
        assert abs(ctrl["r_VS"] - solve_ps_greedy(inst, ctrl, cut).value) <= 1e-12
    p_star = -1.0 + (1 + 2 * rho) / (1 + math.exp(lam * rho))
    assert abs((ctrl["r_VS"] - p_star) - (1 + 2 * rho) * math.tanh(lam * rho / 2)) <= 1e-12


@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("set_type", ["ball", "box"])
def test_vs_below_rpp_every_family(family, set_type):
    from aimp.instances import phase1_instance
    from aimp.relax_ps import solve_ps_greedy

    rng = np.random.default_rng(7)
    for n in ((2,) if family == "F_prop9" else (2, 3, 4)):
        for seed in range(2):
            for qk in ((False,) if family == "F_prop9" else (False, True)):
                inst = phase1_instance(family, seed, n, 0.2, set_type, qk, FAMILIES[family])
                ctrl = bounds.controls(inst, "shor" if set_type == "ball" else "interval", eps_target=1e-3, threads=1)
                v = ctrl["r_VS"]
                for cut in ["auto", "B"] + (["C"] if ctrl["gamma"] > 0 else []):
                    r = solve_ps_greedy(inst, ctrl, cut).value
                    assert v <= r + 1e-12 * max(1.0, abs(r))
                if n == 2:  # softmax(box) = box ∩ simplex with two keys: the bounds coincide
                    assert abs(v - solve_ps_greedy(inst, ctrl, "B").value) <= 1e-12 * max(1.0, abs(v))
                for _ in range(40):
                    Z = inst.centers().copy()
                    for t in inst.tokens:
                        rep = S.affine(inst.sets[t])
                        if not rep.r:
                            continue
                        if rep.base == "box":
                            u = rng.uniform(-1, 1, rep.r) * rep.h
                        else:
                            u = rng.standard_normal(rep.r)
                            u *= rep.rho * rng.uniform() ** (1 / rep.r) / np.linalg.norm(u)
                        Z[t] = S.to_z(rep, u)
                    assert forward(inst, Z)["phi"] <= v + 1e-9 * max(1.0, abs(v))


def test_child_controls_recompute_vs():
    from aimp.instances import phase1_instance

    inst = phase1_instance("F_random", 0, 3, 0.2, "ball", False, 1.0)
    ctrl = bounds.controls(inst, "shor", eps_target=1e-3, threads=1)
    lo, hi = ctrl["s_lo"], ctrl["s_hi"]
    mid = (lo + hi) / 2
    ch = bounds.child_controls(ctrl, lo, np.where(np.arange(3) == 0, mid, hi), 3)
    assert ch["r_VS"] <= ctrl["r_VS"] + 1e-12
    assert abs(ch["r_VS"] - bounds.vertex_softmax(ctrl["h_max"], ch["s_lo"], ch["s_hi"])["value"]) <= 1e-15


# ----------------------------------------------------------------------------------------- logit-margin specification
def _cache_ready() -> bool:
    from aimp.io import cache_dir

    return (cache_dir() / "weights.npz").exists() and all((cache_dir() / f"prompt{p}.npz").exists() for p in range(5))


def test_spec_a_is_the_papers():
    if not _cache_ready():
        pytest.skip("GPT-2 cache not built (aimp.gpt2_units.prepare)")
    from aimp import gpt2_units as U

    inst = U.make_instance(5, 1, 2, 4, 0.01)
    hw, _ = U.head_data(5, 1)
    ybar = U.ctx()["P"][2]["merged"][5][4, 64:128] @ hw["W_O"]
    assert np.array_equal(inst.c, ybar / np.linalg.norm(ybar)) and inst.meta["spec"] == "a"
    assert np.array_equal(U.make_instance(5, 1, 2, 4, 0.01, spec="a").c, inst.c)
    with pytest.raises(ValueError):
        U.make_instance(5, 1, 2, 8, 0.01, spec="lm")


def test_lm_spec_against_the_model():
    pytest.importorskip("transformers")
    from aimp import gpt2

    try:
        model, tok = gpt2.load_model()
    except Exception as e:  # no network and no local copy
        pytest.skip(f"gpt2 not available: {e!r}")
    W, F = gpt2.all_weights(model), gpt2.final_weights(model)
    ext = gpt2.extract(model, tok, gpt2.load_prompts()[0])
    lg = ext["logits"]
    assert np.max(np.abs(gpt2.final_logits(ext["out"][11], F)[4] - lg[4])) <= 1e-10
    assert np.max(np.abs(gpt2.numpy_logits(ext["x"][0], W, F)[4] - lg[4])) <= 1e-10
    order = np.argsort(-lg[4], kind="stable")
    t1, t2 = int(order[0]), int(order[1])
    xbar = ext["out"][11][4]
    lm = gpt2.lm_spec(xbar, F, t1, t2)
    wd = F["wte"][t1] - F["wte"][t2]
    frozen = lambda x: float(wd @ (F["lnf_g"] * (x - x.mean()) / lm["sigma_bar"] + F["lnf_b"]))  # noqa: E731
    rng = np.random.default_rng(1)
    for _ in range(3):
        dlt = rng.standard_normal(768) * 1e-3 * np.linalg.norm(xbar) / math.sqrt(768)
        assert abs((frozen(xbar + dlt) - frozen(xbar)) - lm["c_m"] @ dlt) <= 1e-6 * abs(lm["c_m"] @ dlt)
    assert 0.99 < lm["cos_exact"] <= 1.0 + 1e-12
