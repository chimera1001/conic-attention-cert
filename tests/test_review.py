"""The review computations of ``experiments/review_*.py`` on small instances (synthetic heads and the synthetic gpt-oss
cache)."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from tests.test_gpt_oss import syn_cache  # noqa: F401  (the synthetic gpt-oss cache fixture)

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    import sys

    for p in (ROOT, ROOT / "experiments"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_unit_names_parse():
    C = _load("review_common")
    assert C.parse(Path("L01H03_P0_i04_r0.002_fq.json")) == (1, 3, 0, 4, 0.002)
    assert C.parse(Path("L23H63_P4_i21_r0.0002.json")) == (23, 63, 4, 21, 0.0002)
    with pytest.raises(ValueError):
        C.parse(Path("x.json"))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_taylor_certificate_is_valid_and_second_order(seed):
    """On random fixed-query heads with balls: ``r_T`` is above the exact maximum (Theorem 7), the gradient is
    :func:`aimp.heads.grad_phi`'s, ``|phi''|`` stays below ``2 rad_c Delta_s^2 + 2 Delta_s nu_c`` along random segments
    from ``zbar``, and the gap of ``r_T`` is of second order in the radius."""
    from aimp import instances
    from aimp.exact_fq import exact_fq, fix_query
    from aimp.heads import grad_phi, softmax

    T1 = _load("review_taylor")
    gaps = []
    for rho in (0.01, 0.02):
        inst = fix_query(instances.f_random(seed, d=6, d_k=3, d_v=3, n=4, rho=rho, query_in_keys=False))
        T = T1.taylor(inst)
        ex = exact_fq(inst)
        assert ex["status"] == "exact" and T["r_T"] >= ex["p_fq_hi"]
        _, G = grad_phi(inst, inst.centers())
        for j, (B, b, g, grad, r) in T["_B"].items():
            assert np.allclose(B.T @ G[j], grad, rtol=1e-9, atol=1e-12)
        bound2 = 2 * T["rad_c"] * T["Delta_s"] ** 2 + 2 * T["Delta_s"] * T["nu_c"]
        rng = np.random.default_rng(seed)
        for _ in range(20):
            sig, gam = np.zeros(T["n"]), np.zeros(T["n"])
            for k, j in enumerate(inst.keys):
                if j in T["_B"]:
                    B, b, g, grad, r = T["_B"][j]
                    v = rng.standard_normal(B.shape[1])
                    v *= r / np.linalg.norm(v)
                    sig[k], gam[k] = b @ v, g @ v
            for tt in np.linspace(0, 1, 11):
                a = softmax(T["s_nom"] + tt * sig)
                sb = a @ sig
                d2 = a @ (((sig - sb) ** 2 - a @ (sig - sb) ** 2) * (T["h_nom"] + tt * gam)) + 2 * a @ ((sig - sb) * gam)
                assert abs(d2) <= bound2 * (1 + 1e-12)
        gaps.append(T["r_T"] - 0.5 * (ex["p_fq_lo"] + ex["p_fq_hi"]))
    assert 1.6 < math.log(gaps[1] / gaps[0]) / math.log(2.0) < 2.4


def test_crown_with_score_intervals_on_a_synthetic_head():
    """The plain runs reproduce :func:`aimp.lirpa.bounds`; with certified score intervals (Cauchy-Schwarz here) as
    reference or as the node's bounds every bound stays above the sampled maximum, and the intervals contain the samples."""
    pytest.importorskip("auto_LiRPA")
    from aimp import bounds, instances, lirpa

    T2 = _load("review_crown_intervals")
    inst = instances.f_random(3, d=6, d_k=3, d_v=3, n=3, rho=0.05)
    si = bounds.score_intervals(inst, "cs")
    r = T2.lirpa_runs(inst, si["s_lo"], si["s_hi"], seed=5)
    b = lirpa.bounds(inst, 5, threads=1)
    assert r["gate_ok"] and r["sdp_interval_max_violation_on_samples"] <= 0 and r["sdp_contains_nominal"]
    for m in ("IBP", "CROWN", "alpha-CROWN"):
        assert r["runs"][f"plain:{m}"]["ub"] == b["methods"][m]["ub"]
    for k in ("ref:CROWN", "ref:alpha-CROWN", "set:CROWN", "set:alpha-CROWN"):
        assert r["runs"][k]["status"] == "ok" and r["runs"][k]["ub"] >= b["sampled_max"] - 1e-9 * max(1.0, abs(b["sampled_max"]))


def test_bound_terms_recomputed_from_a_stored_record(syn_cache, tmp_path):  # noqa: F811
    """A fixed-query gpt-oss unit (synthetic weights) stored with the fields of part f, its terms of bound (25) from the
    pipeline's own call (:func:`aimp.pipeline.collapse_fields`): recomputed from the stored intervals they are the same."""
    import json

    from aimp import gpt2_units as U
    from aimp import pipeline
    from aimp.io import jsonable
    from aimp.theory import nominal_decomposition

    T4 = _load("review_bound_terms")
    l, h, p, rho = 0, 9, 1, 2e-2
    xr = pipeline.evaluate_fq_exact(l, h, p, 4, rho, model="gpt-oss-20b")
    fq = U.make_instance(l, h, p, 4, rho, fixed_query=True, model="gpt-oss-20b")
    si, ctrl = pipeline.controls(fq, 1)
    eta = float(nominal_decomposition(fq)["eta"])
    cf = pipeline.collapse_fields(fq, ctrl, phi_min_strong=xr["min_fq"], eta=eta, pruning_slack=0.0)
    rec = {**xr, **{k: v for k, v in cf.items() if k != "perkey"}, "a_lo": ctrl["a_lo"], "a_hi": ctrl["a_hi"],
           "F0_eta": eta, "pruning_slack": 0.0, "phi_min_strong": xr["min_fq"], "cert": xr["p_fq_hi"], "theta": None,
           "range_no_sink": 0.0, "pruned_keys": []}
    f = tmp_path / "gpt_oss" / "f" / "L00" / f"{U.unit_name(l, h, p, 4, rho)}.json"
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps(jsonable(rec)))
    r = T4.evaluate("F", f, tmp_path)
    num = {k: v for k, v in r["deviation_stored_vs_recomputed"].items() if k != "budget_binds"}
    assert max(num.values()) <= 1e-12 * max(1.0, abs(rec["bound_F0v_final"]))
    assert r["deviation_stored_vs_recomputed"]["budget_binds"] is False
    assert r["bound_holds"] and r["F0vf_term_knapsack"] > 0
    assert max(r["checks"][k] for k in ("d_a_lo", "d_a_hi", "d_rad_c", "d_eta_vs_F0_eta")) <= 1e-12


def test_summary_columns_are_those_of_the_figure_data():
    import pandas as pd

    S = _load("review_summary")
    for name, cols in (("P9a_taylor_fixed_query", S.P9A_COLS), ("P9b_crown_sdp_intervals", S.P9B_COLS),
                       ("P9c_gptoss_bound_terms", S.P4A_COLS), ("P9d_gpt2_margin_exceptions", S.P9D_COLS),
                       ("P4a_bound_terms", S.P4A_COLS)):
        assert list(pd.read_csv(ROOT / "figdata" / f"{name}.csv", nrows=0).columns) == cols, name
    assert S.small_slope([0.002, 0.005, 0.01], [4e-6, 2.5e-5, 1e-4]) == pytest.approx(2.0)
    assert S.ls_slope([0.001, 0.002, 0.004], [1e-6, 4e-6, 1.6e-5]) == pytest.approx(2.0)
