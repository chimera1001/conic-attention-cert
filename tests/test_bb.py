"""``certify.branch_and_bound`` (branch-and-bound over the scores, native task-API path) and the strong lower bound of
:mod:`aimp.lower`."""
import math

import numpy as np
import pytest

from aimp import bounds, instances
from aimp.certify import branch_and_bound, needs_retol, split_box, split_scores
from aimp.exact_gurobi import solve_exact
from aimp.heads import phi
from aimp.lower import boundary_points, strong_bracket


def _inst(seed: int):
    # small balls, sharp heads: a visible sdp_1 gap that branching can close
    return instances.f_random(seed, d=4, d_k=2, d_v=2, n=3, temp=3.0, rho=0.4, set_type="ball")


_EXACT: dict = {}


def _exact(seed):
    if seed not in _EXACT:
        inst = _inst(seed)
        ctrl = bounds.controls(inst, threads=1)
        _EXACT[seed] = (inst, ctrl, solve_exact(inst, ctrl, threads=1, time_limit=20),
                        strong_bracket(inst, seed, n_random=4, maxiter=500))
    return _EXACT[seed]


@pytest.mark.parametrize("rule,guard", [("S", "node"), ("value", "termination"), ("S", "termination")])
@pytest.mark.parametrize("seed", range(3))
def test_bb_ub_ge_exact_and_monotone(seed, rule, guard):
    inst, ctrl, ex, lbr = _exact(seed)
    bb = branch_and_bound(inst, ctrl, lbr["lb_strong"], lbr["phi_min_strong"], K=7, threads=1, rule=rule, guard=guard)
    # UB is a valid certificate: >= the exact value (incumbent <= p_star <= UB)
    assert bb["UB"] >= ex.value_lb_verified - 1e-6
    if ex.status == "optimal":
        assert bb["UB"] >= ex.value - 1e-6
    # LB is a feasible value, never above an exact upper bound; bracket ordered
    assert bb["LB"] <= ex.value + 1e-6
    assert bb["LB"] >= lbr["lb_strong"] - 1e-15
    assert bb["valid"] and bb["UB"] >= bb["LB"] - 1e-9
    # UB non-increasing in solves (trace and the recorded checkpoints)
    ubs = [t["UB"] for t in bb["trace"]]
    assert all(ubs[k + 1] <= ubs[k] + 1e-12 for k in range(len(ubs) - 1))
    assert bb["ub_monotone"]
    at = [bb["UB_at"][k] for k in sorted(bb["UB_at"])]
    assert all(at[k + 1] <= at[k] + 1e-12 for k in range(len(at) - 1))
    assert bb["n_solves"] <= 7
    # every solved child's bound <= its parent's bound
    for nd in bb["nodes"]:
        if nd["parent"] is not None and nd["solved"]:
            assert nd["bound"] <= bb["nodes"][nd["parent"]]["bound"] + 1e-15
    for s in bb["splits"]:
        assert s["predicted_gain"] is None or s["predicted_gain"] >= 1.0
        for k in ("predicted_gain_S", "predicted_gain_v"):
            assert s[k] is None or s[k] >= 1.0
    assert bb["rule"] == rule and bb["guard"] == guard
    assert bb["UB"] <= bb["UB_search"] + 1e-12
    if guard == "termination":  # no re-solve during the search; every re-solve is in the termination log
        assert all("retol" not in nd or any(g["node"] == nd["id"] and g["fired"] for g in bb["guard_log"])
                   for nd in bb["nodes"])
        assert bb["n_retol"] == sum(g["fired"] for g in bb["guard_log"])
        assert bb["guard_log"] and bb["guard_log"][0]["why"] == "root"
        assert bb["UB_at_guarded"][max(bb["UB_at_guarded"])] == bb["UB"]


def test_split_children_cover_parent_box():
    rng = np.random.default_rng(0)
    for _ in range(20):
        lo = rng.normal(size=5)
        hi = lo + rng.uniform(0.1, 2.0, size=5)
        j = int(rng.integers(5))
        (lo1, hi1), (lo2, hi2) = split_box(lo, hi, j)
        # union = parent: identical outside key j; on key j [lo, mid] u [mid, hi]
        assert np.array_equal(np.delete(lo1, j), np.delete(lo, j)) and np.array_equal(np.delete(hi1, j), np.delete(hi, j))
        assert np.array_equal(np.delete(lo2, j), np.delete(lo, j)) and np.array_equal(np.delete(hi2, j), np.delete(hi, j))
        assert lo1[j] == lo[j] and hi2[j] == hi[j] and hi1[j] == lo2[j]
        assert lo[j] < hi1[j] < hi[j]
        # every sample of the parent box lies in a child
        for x in rng.uniform(lo, hi, size=(200, 5)):
            assert np.all((x >= lo1) & (x <= hi1)) or np.all((x >= lo2) & (x <= hi2))


def test_split_scores_rule():
    inst = _inst(0)
    ctrl = bounds.controls(inst, threads=1)
    Sj = split_scores(ctrl)
    from aimp.theory import sandwich_S

    assert Sj.sum() == pytest.approx(sandwich_S(ctrl), rel=1e-12)
    assert np.all(Sj >= 0)


def test_strong_lb_feasible_and_ordered():
    inst = _inst(1)
    b = strong_bracket(inst, 3, n_random=6, maxiter=300)
    assert b["lb_strong"] == pytest.approx(phi(inst, b["lb_Z"]), abs=1e-12)  # attained at a feasible point
    from aimp import sets as S

    for t in inst.tokens:
        assert S.contains(inst.sets[t], b["lb_Z"][t], tol=1e-9)
        assert S.contains(inst.sets[t], b["phi_min_Z"][t], tol=1e-9)
    assert b["phi_min_strong"] <= b["lb_strong"]
    # boundary starts lie on the balls' spheres
    for Z in boundary_points(inst, np.random.default_rng(0), 3):
        for t in inst.tokens:
            assert np.linalg.norm(Z[t] - inst.sets[t].center) == pytest.approx(inst.sets[t].rho, rel=1e-12)


def test_retol_guard():
    assert needs_retol(2e-3, 0.1)
    assert not needs_retol(1e-4, 0.1)
    assert needs_retol(1e-9, 0.0)
    assert not needs_retol(None, 0.1)


def test_value_rule_splits_argmax_P():
    from aimp.theory import support_values, value_scores

    inst, ctrl, ex, lbr = _exact(1)
    pm = lbr["phi_min_strong"]
    bb = branch_and_bound(inst, ctrl, lbr["lb_strong"], pm, K=3, threads=1, rule="value", guard="off")
    s0 = bb["splits"][0]
    Sj = split_scores(ctrl)
    Pj = value_scores(Sj, support_values(inst), pm)
    assert s0["key"] == int(np.argmax(Pj)) and s0["how"] == "value"
    assert s0["alt_key"] == int(np.argmax(Sj))
    assert s0["Pi_v"] == pytest.approx(float(Pj.sum()), rel=1e-12)
    assert s0["predicted_gain"] == pytest.approx(Pj.sum() / (Pj.sum() - 0.75 * Pj.max()), rel=1e-12)
    assert s0["predicted_gain_S"] == pytest.approx(Sj.sum() / (Sj.sum() - 0.75 * Sj[s0["key"]]), rel=1e-12)
    with pytest.raises(ValueError):
        branch_and_bound(inst, ctrl, lbr["lb_strong"], None, K=1, threads=1, rule="value")


def test_termination_guard_fires_and_stays_valid():
    """Loose tolerances (1e-2) make the safe slack large, so the termination re-solve rule fires (seed 2: the root and
    three leaves, the window pass repeating as UB drops); the re-solved bounds stay valid."""
    inst, ctrl, ex, lbr = _exact(2)
    loose = {"MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-2, "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-2,
             "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-2}
    bb = branch_and_bound(inst, ctrl, lbr["lb_strong"], lbr["phi_min_strong"], K=5, threads=1, tol=loose,
                          rule="value", guard="termination")
    assert bb["n_retol"] >= 2 and sum(g["fired"] for g in bb["guard_log"]) == bb["n_retol"]
    assert bb["guard_passes"] >= 1
    assert bb["UB"] >= ex.value_lb_verified - 1e-6
    if ex.status == "optimal":
        assert bb["UB"] >= ex.value - 1e-6
    assert bb["UB"] <= bb["UB_search"] + 1e-12 and bb["root_bound_guarded"] <= bb["root_bound_search"] + 1e-12
    for nd in bb["nodes"]:  # every bound = min over its path (a child is a restriction of its parent)
        if nd["parent"] is not None and not nd.get("discarded"):
            assert nd["bound"] <= bb["nodes"][nd["parent"]]["bound"] + 1e-15
    # the leaf attaining UB has been evaluated by the guard (directly or through the node its bound comes from)
    top = max((nd for nd in bb["nodes"] if nd["leaf"]), key=lambda nd: nd["bound"])
    ev = {g["node"] for g in bb["guard_log"]}
    x = top
    while x is not None and x["id"] not in ev:
        x = None if x["parent"] is None else bb["nodes"][x["parent"]]
    assert x is not None


def test_bb_defaults_q37():
    """Defaults: the re-solve rule is applied at termination (``guard="termination"``) and the split rule is the
    bound-lookahead rule (``rule="lookahead"``)."""
    import inspect

    sig = inspect.signature(branch_and_bound)
    assert sig.parameters["guard"].default == "termination"
    assert sig.parameters["rule"].default == "lookahead"


def _f0v_inputs(inst, ctrl):
    from aimp.lower import neg_instance
    from aimp.relax_ps import solve_ps_greedy
    from aimp.theory import directional_constants, nominal_decomposition

    dc = directional_constants(inst)
    return {"h_max": dc["h_sup"], "nu_c": dc["rho_g"], "rad_c": dc["rad_c"], "eta": nominal_decomposition(inst)["eta"],
            "phi_min_cert": -solve_ps_greedy(neg_instance(inst), ctrl, "auto").value}


@pytest.mark.parametrize("seed", range(3))
def test_lookahead_rule(seed):
    """Bound-lookahead split — valid bracket, the chosen key minimizes max(F^-, F^+) over the keys with D_j > 0,
    predicted gain >= 1 (the children's bounds (39) never exceed the parent's), the S / value keys recorded."""
    from aimp.bounds import child_controls
    from aimp.theory import f0v_final_fields

    inst, ctrl, ex, lbr = _exact(seed)
    fi = _f0v_inputs(inst, ctrl)
    bb = branch_and_bound(inst, ctrl, lbr["lb_strong"], lbr["phi_min_strong"], K=7, threads=1, rule="lookahead",
                          f0v_inputs=fi)
    assert bb["UB"] >= ex.value_lb_verified - 1e-6 and bb["valid"] and bb["ub_monotone"]
    s0 = bb["splits"][0]
    assert s0["how"] == "lookahead" and s0["predicted_gain"] >= 1 - 1e-12
    assert "key_S" in s0 and "key_value" in s0
    best = []
    for j in range(inst.n):
        f = f0v_final_fields(inst, ctrl, phi_min_cert=fi["phi_min_cert"], phi_min_strong=None, eta=fi["eta"])
        if not f["D_j"][j] > 0:
            continue
        (l1, h1), (l2, h2) = split_box(ctrl["s_lo"], ctrl["s_hi"], j)
        Fm = f0v_final_fields(inst, child_controls(ctrl, l1, h1, inst.n), phi_min_cert=fi["phi_min_cert"],
                              phi_min_strong=None, eta=fi["eta"])["bound_F0v_final"]
        Fp = f0v_final_fields(inst, child_controls(ctrl, l2, h2, inst.n), phi_min_cert=fi["phi_min_cert"],
                              phi_min_strong=None, eta=fi["eta"])["bound_F0v_final"]
        best.append((max(Fm, Fp), j))
    assert s0["F_child_max"] == pytest.approx(min(best)[0], rel=1e-9)
    # without f0v_inputs the rule computes them from the instance (the same values)
    from aimp.certify import default_f0v_inputs

    d = default_f0v_inputs(inst, ctrl)
    for k in ("rad_c", "eta", "phi_min_cert"):
        assert d[k] == pytest.approx(fi[k], rel=1e-12)
    bb2 = branch_and_bound(inst, ctrl, lbr["lb_strong"], lbr["phi_min_strong"], K=3, threads=1, rule="lookahead")
    assert bb2["splits"][0]["key"] == s0["key"]
