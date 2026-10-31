"""Fast tests of the synthetic-head experiment (experiments/synthetic.py): grids, the slope fits behind Table 3, and
every certificate on a tiny instance against the exact optimum."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location("synthetic_experiment", ROOT / "experiments" / "synthetic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


syn = _load()
RHOS = np.array([0.01, 0.02, 0.05, 0.1])


# ----------------------------------------------------------------------------- slope fits
def test_fit_slope_linear_and_quadratic():
    assert syn.fit_slope(RHOS, 3.0 * RHOS) == pytest.approx(1.0, abs=1e-12)
    assert syn.fit_slope(RHOS, 0.5 * RHOS ** 2) == pytest.approx(2.0, abs=1e-12)
    assert syn.fit_slope(RHOS[:2], RHOS[:2]) is None  # fewer than three points
    assert syn.fit_slope(np.array([0.01, 0.012, 0.015]), np.ones(3)) is None  # spans less than a factor four
    y = np.array([1e-9, 1e-8, 2.5e-6, 1e-5])  # two values below the noise floor 1e-7: two points remain
    assert syn.fit_slope(RHOS, y) is None


def test_slope_table_returns_one_and_two():
    import pandas as pd

    rows = []
    for seed in range(4):
        for rho in RHOS:
            rows.append({"family": "F_random", "config": "t1.0", "seed": seed, "n": 2, "set_type": "ball",
                         "query_in_keys": False, "rho": rho, "res": 0.0,
                         "ps_auto": (1 + seed) * rho, "sdp_1": (1 + seed) * rho ** 2,
                         "ibp_obj": 2 * (1 + seed) * rho, "sdp_1_unitdiag": (1 + seed) * rho ** 1.5})
    W = pd.DataFrame(rows)
    W["series"] = W["family"] + "|" + W["config"] + "|s" + W["seed"].astype(str)
    T = syn.slope_table(W, ["ibp_obj", "ps_auto", "sdp_1_unitdiag", "sdp_1"])
    T = T[T["rule"] == "per_method"].set_index("method")
    assert T.loc["ps_auto", "slope_median"] == pytest.approx(1.0, abs=1e-9)
    assert T.loc["ibp_obj", "slope_median"] == pytest.approx(1.0, abs=1e-9)
    assert T.loc["sdp_1", "slope_median"] == pytest.approx(2.0, abs=1e-9)
    assert T.loc["sdp_1_unitdiag", "slope_median"] == pytest.approx(1.5, abs=1e-9)
    assert int(T.loc["sdp_1", "n_series_fitted"]) == 4
    W2 = W.assign(res=1.0)  # nothing resolved: no fit
    T2 = syn.slope_table(W2, ["sdp_1"])
    assert T2["n_series_fitted"].max() == 0


# ----------------------------------------------------------------------------- grids
def test_grids_and_time_limits():
    full = syn.build_tasks("slopes")
    assert len(full) == 4620 + 3160 + 320 == 8100  # the paper's synthetic instances
    assert sum(t["n"] == 2 for t in full) == 4620
    assert not any(not syn.in_paper_grid(t) for t in full)
    assert not syn.in_paper_grid({"family": "F_rank1", "n": 3, "set_type": "ball", "rho": 0.2, "cfg": 0.3, "seed": 5})
    assert syn.in_paper_grid({"family": "F_rank1", "n": 3, "set_type": "ball", "rho": 0.2, "cfg": 0.1, "seed": 5})
    assert not any(t["family"] == "F_prop9" and (t["qk"] or t["n"] != 2) for t in full)
    quick = syn.build_tasks("slopes", quick=True)
    assert len(quick) == 168 and {t["set_type"] for t in quick} == {"ball"}
    assert len(syn.build_tasks("temperature")) == 4 * 2 * 6 * 5
    assert syn.exact_time_limit("slopes", {"n": 2, "rho": 0.5}, None) == 90.0
    assert syn.exact_time_limit("slopes", {"n": 3, "rho": 0.2}, None) == 300.0  # a time-limited global solve
    assert syn.exact_time_limit("slopes", {"n": 4, "rho": 0.1}, 20.0) == 20.0
    assert syn.sdp_settings("tight")["t_interval"] == "tight"
    loose = syn.sdp_settings("loose")
    assert loose["value_mode"] == "primal" and loose["prune_tol"] is None


# ----------------------------------------------------------------------------- one tiny instance, every method
TASK = {"family": "F_random", "cfg": 1.0, "seed": 0, "n": 2, "rho": 0.02, "set_type": "ball", "qk": False}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    cfg = {"sdp": syn.sdp_settings("tight"), "methods": "all", "exact_time_limit": 60.0}
    summary = syn.run_task("slopes", TASK, out, 1, cfg)
    d = next((out / "slopes" / "F_random").iterdir())
    recs = {p.stem: json.loads(p.read_text()) for p in d.glob("*.json") if p.name not in ("instance.json", "done.json")}
    return out, cfg, summary, recs


def test_every_method_is_an_upper_bound(tiny):
    _, _, summary, recs = tiny
    assert summary["errors"] == [] and not summary["flags"]["validity_fail"] and not summary["flags"]["bound_violation"]
    ex = recs["exact"]
    assert ex["status"] in ("optimal", "resolved_stop")
    p_lo = ex["value_lb_verified"]  # a verified feasible value <= p_star <= every relaxation
    assert ex["value"] >= p_lo - 1e-9
    relax = [m for m in recs if m not in ("exact", "multistart")]
    assert {"ibp_obj", "ps_auto", "sdp_1", "sdp_1_unitdiag", "sdp_2", "sdp_3_ball"} <= set(relax)
    for m in relax:
        assert recs[m]["value"] >= p_lo - 1e-6 * max(1.0, abs(p_lo)), m
        assert recs[m]["validity_ok"] is True, m
    assert recs["multistart"]["value"] <= ex["value"] + 1e-9
    g = {m: recs[m]["gap"] for m in relax}
    assert g["ps_auto"] <= g["ibp_obj"] + 1e-7  # the pattern polytope lies in the interval box
    assert g["sdp_1"] <= g["sdp_1_unitdiag"] + 1e-7  # (R4) is stronger than the unit diagonal
    assert g["sdp_1"] < 0.1 * g["ps_auto"]  # second order against first order at small radius
    assert recs["sdp_1"]["params"]["t_interval"] == "tight"


def test_resume_and_settings_guard(tiny):
    out, cfg, summary, _ = tiny
    again = syn.run_task("slopes", TASK, out, 1, cfg)  # every record exists: nothing is re-solved
    assert again["gaps"] == summary["gaps"]
    bad = dict(cfg, sdp=syn.sdp_settings("loose"))
    with pytest.raises(RuntimeError):
        syn.run_task("slopes", TASK, out, 1, bad)


def test_summary_reads_the_records(tiny):
    out = tiny[0]
    df = syn._frame(syn.load_records(out / "slopes"))
    assert df["instance_id"].nunique() == 1
    M2 = syn.min_of_two(df)
    assert len(M2) == 1 and M2["ratio"].iloc[0] >= 1.0 - 1e-12
    chk = syn.checks_table(df)
    assert chk["validity_failures"] == 0 and chk["bound_violations"] == 0


def test_loose_interval_is_not_tighter(tiny):
    from aimp import bounds
    from aimp.relax_sdp import solve_sdp

    inst = syn.task_instance("slopes", TASK)
    ctrl = bounds.controls(inst, threads=1)
    tight = solve_sdp(inst, ctrl, rlt=syn.S1, threads=1, **syn.sdp_settings("tight"))
    loose = solve_sdp(inst, ctrl, rlt=syn.S1, threads=1, **syn.sdp_settings("loose"))
    assert loose.value >= tight.value - 1e-6
    assert loose.diagnostics["w_t_used"] >= tight.diagnostics["w_t_used"] - 1e-12
    assert tight.value == pytest.approx(tiny[3]["sdp_1"]["value"], abs=1e-7)


def test_temperature_for_width():
    from aimp import bounds

    r = syn.temp_for_w(0, 1.0)
    inst = syn.task_instance("width", {"family": "F_random", "seed": 0, "w_target": 1.0, "temp": r["temp"]})
    w_t = bounds.controls(inst, threads=1)["w_t"]
    assert w_t == pytest.approx(1.0, rel=1e-3)
