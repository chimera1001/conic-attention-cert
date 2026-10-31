"""Every solver / library call path takes its thread count from aimp.compute.thread_budget()."""
import os

import numpy as np
import pytest

from aimp import bounds, compute, instances
from aimp.exact_gurobi import solve_exact
from aimp.relax_ps import solve_ps
from aimp.relax_sdp import solve_sdp
from aimp.solvers import gurobi_model


def test_budget_from_env(monkeypatch):
    monkeypatch.setenv("AIMP_MAX_THREADS", "3")
    assert compute.thread_budget() == 3
    assert compute.resolve_threads() == 3
    assert compute.resolve_threads(1) == 1
    with pytest.raises(ValueError):
        compute.resolve_threads(4)


def test_blas_env_set_at_import():
    b = str(compute.thread_budget())
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert os.environ[var] == b


def test_gurobi_model_threads():
    m, used = gurobi_model("t")
    assert m.Params.Threads == compute.thread_budget() == used["Threads"]
    m1, _ = gurobi_model("t1", threads=1)
    assert m1.Params.Threads == 1


def test_mosek_params_threads():
    assert compute.mosek_params()["MSK_IPAR_NUM_THREADS"] == compute.thread_budget()
    assert compute.mosek_params(1)["MSK_IPAR_NUM_THREADS"] == 1


def test_torch_threads():
    n = compute.set_torch_threads()
    import torch

    assert torch.get_num_threads() == n == compute.thread_budget()


@pytest.mark.parametrize("threads", [None, 1])
def test_every_solver_path_records_threads(threads):
    expect = compute.thread_budget() if threads is None else threads
    inst = instances.f_random(0, d=3, d_k=2, d_v=2, n=2, rho=0.1, set_type="box")
    inst_b = instances.f_random(0, d=3, d_k=2, d_v=2, n=2, rho=0.1, set_type="ball")
    calls = [
        lambda: bounds.score_intervals(inst, "exact", threads=threads),
        lambda: bounds.score_intervals(inst_b, "shor", threads=threads),
        lambda: bounds.value_spread(inst_b, threads=threads),
    ]
    ctrl = bounds.controls(inst, threads=threads)
    calls += [
        lambda: solve_ps(inst, ctrl, backend="gurobi", threads=threads),
        lambda: solve_ps(inst, ctrl, backend="cvxpy", threads=threads),
        lambda: solve_sdp(inst, ctrl, threads=threads),
        lambda: solve_exact(inst, ctrl, threads=threads, time_limit=30),
    ]
    from aimp.sdp_mosek import shor_score, solve_sdp1  # MOSEK task API

    ctrl_b = bounds.controls(inst_b, threads=threads)
    calls += [lambda: solve_sdp1(inst_b, ctrl_b, threads=threads), lambda: shor_score(inst_b, 1, threads=threads)]
    for call in calls:
        start = compute.CALL_COUNT
        call()
        new = compute.calls_since(start)
        assert new, "solver path did not record its thread setting"
        assert all(c["threads"] == expect for c in new), new
