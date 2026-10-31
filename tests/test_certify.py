"""The certificate of Definition 4 (``certify()``) and the certified radius (``rho_star()``)."""
import math

import numpy as np
import pytest

from aimp import bounds, instances
from aimp.certify import certify, rho_star, theorem_check
from aimp.exact_gurobi import solve_exact
from aimp.heads import nominal
from aimp.types import HeadSpec, Instance
from aimp import sets as S


def _sdp_instances():
    return [instances.f_random(s, d=4, d_k=2, d_v=2, n=3, temp=1.0, rho=0.2, set_type="ball") for s in range(3)]


@pytest.mark.parametrize("k", range(3))
def test_cert_le_each_component_and_valid(k):
    inst = _sdp_instances()[k]
    ctrl = bounds.controls(inst, threads=1)
    ex = solve_exact(inst, ctrl, threads=1, time_limit=10)  # time-limited: lb = incumbent, ub = ObjBound
    rec = certify(inst, ctrl, eps_target=0.0, threads=1, lb=ex.value_lb_verified, ub=ex.value)
    assert rec["route"] == "min"
    assert rec["value_safe"] is not None
    assert rec["cert"] <= rec["r_greedy"] + 1e-15
    assert rec["cert"] <= rec["value_safe"] + 1e-15
    assert rec["cert"] == min(rec["r_greedy"], rec["value_safe"])
    assert rec["cert"] >= ex.value_lb_verified - 1e-6  # validity
    # norm-based second-order bound checks attached, in both forms, and not refuted
    for key in ("bound_F0t", "F0t_check", "bound_F0s", "F0s_applies", "F0s_check", "S", "w_t", "theta", "gamma",
                "Delta", "rad", "times"):
        assert key in rec
    assert rec["F0t_check"]["status"] in ("satisfied", "inconclusive")
    if rec["F0s_applies"]:
        assert rec["F0s_check"]["status"] in ("satisfied", "inconclusive")
    # value := value_safe always; the safe value dominates the dual value
    assert rec["value_safe"] >= rec["sdp_value_dual"] - 1e-9 * max(1, abs(rec["sdp_value_dual"]))


def test_margin_route_skips_sdp():
    inst = instances.p1_margin(0, 3, 0.01, "ball", True, gamma0=8.0)
    ctrl = bounds.controls(inst, threads=1)
    assert ctrl["bound_C"] is not None
    rec = certify(inst, ctrl, eps_target=10 * ctrl["bound_C"], threads=1)
    assert rec["route"] == "margin" and "value_safe" not in rec
    assert rec["cert"] == rec["r_greedy"]


def test_theorem_check_three_valued():
    assert theorem_check(1.0, 0.5, lb=0.6, ub=0.9)["status"] == "satisfied"
    assert theorem_check(1.0, 0.1, lb=0.6, ub=0.85)["status"] == "violated"
    assert theorem_check(1.0, 0.2, lb=0.6, ub=0.85)["status"] == "inconclusive"


def _uniform_instance(rho: float, seed: int = 0, n: int = 3, d: int = 5) -> Instance:
    """M = 0: the pattern is exactly uniform, so p_star = phi_nominal + rho ||N^T c|| (every key moves along N^T c)."""
    rng = np.random.default_rng(seed)
    N = rng.standard_normal((d, d))
    head = HeadSpec(M=np.zeros((d, d)), N=N, d=d, d_k=d, d_v=d)
    Z = rng.standard_normal((n, d))
    c = rng.standard_normal(d)
    c /= np.linalg.norm(c)
    return Instance(head=head, keys=list(range(n)), query=0, sets={t: S.ball(Z[t], rho) for t in range(n)}, c=c)


def test_rho_star_known_case():
    base = _uniform_instance(1.0)
    g = np.linalg.norm(base.head.N.T @ base.c)

    def gap(rho):
        inst = _uniform_instance(rho)
        rec = certify(inst, bounds.controls(inst, threads=1), eps_target=1e9, threads=1)
        return rec["cert"] - rec["phi_nominal"]

    assert gap(0.3) == pytest.approx(0.3 * g, rel=1e-9)
    lo, hi, steps = 1e-3, 1.0, 6
    for eps in (0.01, 0.1):
        out = rho_star(gap, eps, lo, hi, steps)
        exact = eps / g
        assert out["censored"] is None
        assert out["rho_star"] <= exact <= out["rho_upper"]
        assert out["rho_upper"] / out["rho_star"] == pytest.approx(10 ** (0.5 / 2 ** steps), rel=1e-9)
        assert out["monotone_evals"]


def test_rho_star_first_crossing_non_monotone():
    """A gap that rises and falls again (the cert - lb reading): rho_star is the first crossing, not the far end."""
    gap = lambda r: r if r < 0.3 else 1e-6  # noqa: E731
    out = rho_star(gap, 0.01, 1e-4, 1.0)
    assert out["censored"] is None and out["rho_star"] <= 0.01 <= out["rho_upper"]
    assert out["rho_upper"] / out["rho_star"] < 1.02


def test_rho_star_monotone_in_eps():
    def make(rho):
        return instances.f_random(1, d=4, d_k=2, d_v=2, n=3, temp=1.0, rho=rho, set_type="ball")

    cache = {}

    def gap(rho):
        inst = make(rho)
        rec = certify(inst, bounds.controls(inst, threads=1), threads=1, run_sdp=False)  # greedy component
        return rec["cert"] - rec["phi_nominal"]

    outs = [rho_star(gap, eps, 1e-4, 1.0, 6, cache) for eps in (1e-3, 1e-2, 1e-1)]
    rs = [o["rho_star"] or 0.0 for o in outs]
    assert rs[0] <= rs[1] <= rs[2]
    assert all(o.get("monotone_evals", True) for o in outs)
    # censoring
    assert rho_star(gap, 1e9, 1e-4, 1.0)["censored"] == "above"
    assert rho_star(gap, -1.0, 1e-4, 1.0)["censored"] == "below"
