"""Formulations: trivial sanity-check instances, cross-checks between variants/backends, validity."""
import numpy as np
import pytest

from aimp import bounds, instances, sets as S
from aimp.exact_gurobi import solve_exact
from aimp.heads import nominal
from aimp.ibp import solve_ibp
from aimp.lower_bounds import multistart_lb
from aimp.relax_ps import solve_ps
from aimp.relax_sdp import solve_sdp

T_EXACT = 60


def all_methods(inst, ctrl, exact=True):
    out = {
        "ps_auto_gurobi": solve_ps(inst, ctrl, backend="gurobi"),
        "ps_auto_cvxpy": solve_ps(inst, ctrl, backend="cvxpy"),
        "ps_none": solve_ps(inst, ctrl, cut="none"),
        "sdp_1": solve_sdp(inst, ctrl),
        "ibp": solve_ibp(inst, ctrl),
        "multistart": multistart_lb(inst, 10),
    }
    if exact:
        out["exact_log"] = solve_exact(inst, ctrl, time_limit=T_EXACT)
        out["exact_alg"] = solve_exact(inst, ctrl, form="algebraic", time_limit=T_EXACT)
    return out


def close(x, ref, tol):
    return abs(x - ref) <= tol * max(1.0, abs(ref))


CASES = [(st, qk) for st in ("ball", "box") for qk in (True, False)]


@pytest.mark.parametrize("st,qk", CASES)
def test_trivial_a_rho0(st, qk):
    inst = instances.trivial("rho0", seed=1, set_type=st, query_in_keys=qk)
    ctrl = bounds.controls(inst)
    ref = nominal(inst)["phi"]
    res = all_methods(inst, ctrl)
    for name, r in res.items():
        assert close(r.value, ref, 1e-8), (name, r.value, ref)
        if name.startswith("exact"):
            assert r.status == "optimal" and close(r.value_lb_verified, ref, 1e-8)


@pytest.mark.parametrize("st,qk", CASES)
def test_trivial_b_n1(st, qk):
    inst = instances.trivial("n1", seed=2, set_type=st, query_in_keys=qk)
    ctrl = bounds.controls(inst)
    ref = S.support(inst.sets[inst.keys[0]], inst.head.N.T @ inst.c)  # LP / SOCP in z_1
    res = all_methods(inst, ctrl)
    for name, r in res.items():
        if name == "ibp":
            assert r.value >= ref - 1e-9
            continue
        assert close(r.value, ref, 1e-6), (name, r.value, ref)


@pytest.mark.parametrize("st,qk", CASES)
def test_trivial_c_M0(st, qk):
    inst = instances.trivial("M0", seed=3, set_type=st, query_in_keys=qk)
    ctrl = bounds.controls(inst)
    assert ctrl["Delta"] == 0.0
    ref = np.mean([S.support(inst.sets[j], inst.head.N.T @ inst.c) for j in inst.keys])
    res = all_methods(inst, ctrl)
    for name, r in res.items():
        if name == "ibp":
            assert r.value >= ref - 1e-9
            continue
        assert close(r.value, ref, 1e-6), (name, r.value, ref)


def test_log_vs_algebraic_20():
    """The log and algebraic exact forms agree on box sets (ball sets do not close to 1e-6 in Gurobi within minutes)."""
    for seed in range(20):
        inst = instances.f_random(seed, d=4, d_k=2, d_v=2, n=3, rho=0.3, set_type="box")
        ctrl = bounds.controls(inst)
        ms = multistart_lb(inst, 10, seed=seed)
        A = solve_exact(inst, ctrl, form="log", warm_start=ms.recovered_point, time_limit=T_EXACT)
        B = solve_exact(inst, ctrl, form="algebraic", warm_start=ms.recovered_point, time_limit=T_EXACT)
        assert A.status == B.status == "optimal", (seed, A.status, B.status)
        assert close(A.value_lb_verified, B.value_lb_verified, 1e-6)
        assert close(A.value, B.value, 1e-6)
        assert A.value >= ms.value - 1e-7 and A.value_lb_verified >= ms.value - 1e-6


def test_log_vs_algebraic_balls_consistent():
    """Balls (small rho): both forms' [lb_verified, ub] intervals overlap and incumbents verify."""
    for seed in range(3):
        inst = instances.f_random(seed, d=4, d_k=2, d_v=2, n=3, rho=0.05, set_type="ball")
        ctrl = bounds.controls(inst)
        ms = multistart_lb(inst, 10, seed=seed)
        A = solve_exact(inst, ctrl, form="log", warm_start=ms.recovered_point, time_limit=T_EXACT)
        B = solve_exact(inst, ctrl, form="algebraic", warm_start=ms.recovered_point, time_limit=T_EXACT)
        for R in (A, B):
            assert R.value >= ms.value - 1e-7
            assert abs(R.value_lb - R.value_lb_verified) <= 1e-7 * max(1, abs(R.value))
        assert max(A.value_lb_verified, B.value_lb_verified) <= min(A.value, B.value) + 1e-7


def test_genexp_vs_nl():
    for seed in range(3):
        inst = instances.f_random(seed, d=4, d_k=2, d_v=2, n=3, rho=0.3, set_type="box")
        ctrl = bounds.controls(inst)
        A = solve_exact(inst, ctrl, exp_impl="genexp", time_limit=T_EXACT)
        B = solve_exact(inst, ctrl, exp_impl="nl", time_limit=T_EXACT)
        assert A.status == B.status == "optimal"
        assert close(A.value, B.value, 1e-6)


@pytest.mark.parametrize("st", ["ball", "box"])
def test_ps_backends_agree(st):
    for seed in range(5):
        for qk in (True, False):
            inst = instances.f_random(seed, d=8, d_k=4, d_v=4, n=3, rho=0.1, set_type=st, query_in_keys=qk)
            ctrl = bounds.controls(inst)
            for cut in ("auto", "none", "B"):
                g = solve_ps(inst, ctrl, cut=cut, backend="gurobi")
                c = solve_ps(inst, ctrl, cut=cut, backend="cvxpy")
                assert g.status == "optimal" and c.status == "optimal"
                assert close(g.value, c.value, 1e-6), (seed, qk, cut, g.value, c.value)


def test_validity_and_ordering_10_random():
    """r >= multistart lb for every relaxation (10 instances); r_sdp <= r_ps; r_ps <= ibp-free checks;
    rank-1 => exact; the first-order gap bound (20) of Theorem 9 on box instances where exact is optimal."""
    n_rank1 = 0
    for seed in range(10):
        st = "ball" if seed % 2 else "box"
        inst = instances.f_random(seed, d=4, d_k=2, d_v=2, n=3, rho=0.2, temp=1.0 + seed % 3,
                                  set_type=st, query_in_keys=seed % 3 != 0)
        ctrl = bounds.controls(inst)
        ms = multistart_lb(inst, 20, seed=seed)
        ps = solve_ps(inst, ctrl)
        ps0 = solve_ps(inst, ctrl, cut="none")
        sdp = solve_sdp(inst, ctrl)
        ibp = solve_ibp(inst, ctrl)
        tol = 1e-6 * max(1, abs(ms.value))
        for r in (ps, ps0, sdp, ibp):
            assert r.value >= ms.value - tol, (seed, r.method, r.value, ms.value)
        assert sdp.value <= ps.value + 1e-6
        assert ps.value <= ps0.value + 1e-6
        if sdp.diagnostics["rank1_ratio"] < 1e-6:
            n_rank1 += 1
            assert abs(sdp.value - sdp.recovered_value) < 1e-5
        if st == "box":
            ex = solve_exact(inst, ctrl, warm_start=ms.recovered_point, time_limit=T_EXACT)
            assert ex.value >= ms.value - tol
            if ex.status == "optimal":
                assert ps.value - ex.value <= ctrl["bound_B"] + 1e-6
                assert ps.value >= ex.value_lb_verified - tol and sdp.value >= ex.value_lb_verified - tol


@pytest.mark.parametrize("st", ["ball", "box"])
def test_rank1_implies_exact_nonvacuous(st):
    """n = 1 with the query as the key: the SDP optimum is rank 1 and equals the recovered value."""
    inst = instances.trivial("n1", seed=4, set_type=st, query_in_keys=True)
    ctrl = bounds.controls(inst)
    sdp = solve_sdp(inst, ctrl)
    if st == "ball":
        assert sdp.diagnostics["rank1_ratio"] < 1e-6
    if sdp.diagnostics["rank1_ratio"] < 1e-6:
        assert abs(sdp.value - sdp.recovered_value) < 1e-5


def test_score_intervals_valid_and_self_shor_exact():
    rng = np.random.default_rng(0)
    for seed in range(4):
        for st in ("ball", "box"):
            inst = instances.f_random(seed, d=4, d_k=2, d_v=2, n=3, rho=0.4, set_type=st, query_in_keys=True)
            ivs = {m: bounds.score_intervals(inst, m) for m in ("cs", "interval", "shor", "exact")}
            for _ in range(300):
                Z = np.stack([S.to_z(S.affine(inst.sets[t]), S.base_sample(S.affine(inst.sets[t]), rng, 1)[0])
                              for t in range(inst.T)])
                s = np.array([Z[inst.query] @ inst.head.M @ Z[j] for j in inst.keys])
                for m, iv in ivs.items():
                    assert np.all(s >= iv["s_lo"] - 1e-9) and np.all(s <= iv["s_hi"] + 1e-9), m
            ex = ivs["exact"]
            # attained extremes (incumbents) — every certified interval must contain them
            att_lo = np.array([ex["info"][str(j)]["min"]["obj_val"] if "min" in ex["info"][str(j)] else ex["s_lo"][k]
                               for k, j in enumerate(inst.keys)])
            att_hi = np.array([ex["info"][str(j)]["max"]["obj_val"] if "max" in ex["info"][str(j)] else ex["s_hi"][k]
                               for k, j in enumerate(inst.keys)])
            for m in ("cs", "interval", "shor", "exact"):
                assert np.all(ivs[m]["s_lo"] <= att_lo + 1e-9) and np.all(ivs[m]["s_hi"] >= att_hi - 1e-9), m
            if st == "ball":  # self score over one ball: Shor exact (S-lemma)
                k = inst.keys.index(inst.query)
                assert abs(ivs["shor"]["s_hi"][k] - ex["s_hi"][k]) <= 1e-5 * max(1, abs(ex["s_hi"][k]))
                assert abs(ivs["shor"]["s_lo"][k] - ex["s_lo"][k]) <= 1e-5 * max(1, abs(ex["s_lo"][k]))
