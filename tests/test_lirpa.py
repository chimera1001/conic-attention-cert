"""The bound propagation harness of Section 7 (:mod:`aimp.lirpa`): the torch module computes the head exactly (forward
gate against :func:`aimp.heads.forward`), and IBP / CROWN / alpha-CROWN upper bounds are valid (never below a sampled
value of the head) and ordered as expected, on random synthetic heads and on one GPT-2 unit with a varying and with a
fixed query."""
import math

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("auto_LiRPA")

from aimp import lirpa  # noqa: E402
from aimp import sets as S  # noqa: E402
from aimp.exact_fq import fix_query  # noqa: E402
from aimp.heads import forward  # noqa: E402
from aimp.types import HeadSpec, Instance  # noqa: E402


def _synthetic(seed: int, d: int = 6, n: int = 3, rho: float = 0.05, ellipsoid: bool = False) -> Instance:
    """A random head with ``n`` keys, the query the last key, balls (or ellipsoids with a full-rank ``E``)."""
    rng = np.random.default_rng(seed)
    d_k = 3
    W_Q, W_K, W_V = (rng.standard_normal((d, d_k)) / math.sqrt(d) for _ in range(3))
    W_O = rng.standard_normal((d_k, d)) / math.sqrt(d_k)
    head = HeadSpec.from_weights(W_Q, W_K, W_V, W_O, name="syn")
    Z = rng.standard_normal((n, d))
    c = rng.standard_normal(d)
    c /= np.linalg.norm(c)
    if ellipsoid:
        sets = {t: S.ellipsoid(Z[t], np.eye(d) + 0.3 * rng.standard_normal((d, d)), rho) for t in range(n)}
    else:
        sets = {t: S.ball(Z[t], rho) for t in range(n)}
    return Instance(head=head, keys=list(range(n)), query=n - 1, sets=sets, c=c)


def _sampled_max(inst: Instance, rng: np.random.Generator, k: int = 2000) -> float:
    """Largest head value over ``k`` random points of the sets (uniform in each base ball) and the centers."""
    reps = {t: S.affine(inst.sets[t]) for t in inst.tokens}
    best = forward(inst, inst.centers())["phi"]
    for _ in range(k):
        Z = inst.centers().copy()
        for t, rep in reps.items():
            if rep.r:
                Z[t] = S.to_z(rep, S.base_sample(rep, rng)[0])
        best = max(best, forward(inst, Z)["phi"])
    return best


@pytest.mark.parametrize("seed,ellipsoid", [(0, False), (1, True), (2, False)])
def test_forward_gate_synthetic(seed, ellipsoid):
    inst = _synthetic(seed, ellipsoid=ellipsoid, rho=0.5)
    pert = lirpa.perturbed_tokens(inst)
    assert pert == inst.tokens
    mod = lirpa.head_module(inst, pert, threads=1)
    g = lirpa.forward_gate(inst, mod, pert, np.random.default_rng(seed), n=50)
    assert g["gate_ok"], g
    assert g["gate_max_abs"] < 1e-12


def test_fixed_query_is_constant():
    inst = fix_query(_synthetic(3))
    pert = lirpa.perturbed_tokens(inst)
    assert inst.query not in pert and len(pert) == inst.n - 1
    g = lirpa.forward_gate(inst, lirpa.head_module(inst, pert, threads=1), pert, np.random.default_rng(0), n=50)
    assert g["gate_ok"], g


@pytest.mark.parametrize("seed,rho", [(0, 0.05), (4, 0.2)])
def test_bounds_valid_and_ordered(seed, rho):
    inst = _synthetic(seed, rho=rho)
    b = lirpa.bounds(inst, seed, threads=1)
    assert b["gate_ok"]
    ub = {m: r["ub"] for m, r in b["methods"].items()}
    assert all(r["status"] == "ok" for r in b["methods"].values()), b["methods"]
    vmax = _sampled_max(inst, np.random.default_rng(seed + 1))
    for m, v in ub.items():
        assert v >= vmax - 1e-9, (m, v, vmax)
        assert b["methods"][m]["above_samples"]
    assert ub["CROWN"] <= ub["IBP"] + 1e-9
    assert ub["alpha-CROWN"] <= ub["CROWN"] + 1e-9
    assert b["methods"]["alpha-CROWN"]["iterations"] == 20  # the library default


def test_alpha_run_matches_default_and_traces():
    inst = _synthetic(5, rho=0.2)
    b = lirpa.bounds(inst, 0, methods=("alpha-CROWN",), threads=1)
    a = lirpa.alpha_run(inst, 20, 0.5, threads=1)
    assert a["ub"] == pytest.approx(b["methods"]["alpha-CROWN"]["ub"], abs=1e-10)
    assert 1 <= a["iterations_run"] <= 20 and len(a["trace"]) == a["iterations_run"]
    assert a["ub"] == pytest.approx(a["trace_min"], abs=1e-9)  # the library returns the best iterate
    s = lirpa.alpha_settings(inst, iters=(5, 10), lrs=(0.5,), threads=1)
    assert set(s) == {"it5_lr0.5", "it10_lr0.5"}
    assert s["it5_lr0.5"]["trace"] is None and s["it10_lr0.5"]["trace"] is not None


def test_gpt2_unit():
    """One GPT-2 unit (layer 5, head 1, prompt 0, position 4) with a varying and a fixed query: the gate passes, the
    bounds are valid and alpha-CROWN tightens CROWN."""
    from aimp import gpt2_units as U

    if not (U.cache_dir() / "weights.npz").exists():
        pytest.skip("GPT-2 cache missing (run aimp.gpt2_units.prepare())")
    for fixed, rho in ((False, 0.002), (True, 0.005)):
        inst = U.make_instance(5, 1, 0, 4, rho, fixed_query=fixed)
        b = lirpa.bounds(inst, U.unit_seed(5, 1, 0, 4, rho) + 61, threads=1)
        assert b["gate_ok"], b["gate_max_abs"]
        assert b["n_inputs"] == (4 if fixed else 5)
        vmax = _sampled_max(inst, np.random.default_rng(0), k=300)
        ub = {m: r["ub"] for m, r in b["methods"].items()}
        assert min(ub.values()) >= vmax - 1e-9
        assert ub["alpha-CROWN"] <= ub["CROWN"] + 1e-9 <= ub["IBP"] + 2e-9
