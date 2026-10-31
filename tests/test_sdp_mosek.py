"""aimp.sdp_mosek: the task-API sdp_1 and Shor score bound equal the CVXPY ones; value_safe is valid."""
import numpy as np
import pytest

from aimp import bounds, gpt2, instances
from aimp import sets as S
from aimp.exact_gurobi import solve_exact
from aimp.relax_sdp import solve_sdp
from aimp.sdp_mosek import shor_score, solve_sdp1


def _cases():
    out = []
    for s in range(3):
        out.append(instances.f_random(s, d=4, d_k=2, d_v=2, n=3, temp=1.0, rho=0.2, set_type="ball"))
    out.append(instances.f_random(5, d=4, d_k=2, d_v=2, n=3, temp=2.0, rho=0.3, set_type="ball", query_in_keys=False))
    out.append(instances.p1_margin(1, 3, 0.05, "ball", True, gamma0=3.0, d=6))
    # an ellipsoid (projected) instance with biases absorbed in the centers
    rng = np.random.default_rng(0)
    d, dk = 10, 2
    W = [rng.standard_normal((d, dk)) / np.sqrt(d) * 2 for _ in range(3)]
    WO = rng.standard_normal((dk, d))
    hw = {"W_Q": W[0], "W_K": W[1], "W_V": W[2], "W_O": WO, "b_Q": rng.standard_normal(dk) * 0.3,
          "b_K": rng.standard_normal(dk) * 0.3, "b_V": rng.standard_normal(dk) * 0.3}
    pr = gpt2.projection(hw["W_Q"], hw["W_K"], hw["W_V"], hw["b_Q"], hw["b_K"], hw["b_V"])
    z = rng.standard_normal((4, d)) * 2
    from aimp.types import Instance

    head = gpt2.projected_head(WO, dk, dk)
    sets = {t: S.ellipsoid(pr.P @ z[t] + pr.b, pr.E, 0.1 * np.linalg.norm(z[t])) for t in range(4)}
    c = rng.standard_normal(d)
    out.append(Instance(head=head, keys=[0, 1, 2, 3], query=3, sets=sets, c=c / np.linalg.norm(c)))
    return out


@pytest.mark.parametrize("cut", ["B", "auto"])  # the default and the setting of the paper's experiments
@pytest.mark.parametrize("k", range(6))
def test_sdp1_task_equals_cvxpy(k, cut):
    inst = _cases()[k]
    ctrl = bounds.controls(inst, threads=1)
    old = solve_sdp(inst, ctrl, rlt=("persp", "simplex_x_z"), threads=1, t_interval="tight", cone_form=True,
                    safe="always", cut=cut)
    new = solve_sdp1(inst, ctrl, threads=1, cut=cut)
    assert new.status == "optimal"
    ref = old.diagnostics["value_dual"]
    scale = max(1.0, abs(ref))
    assert new.diagnostics["value_dual"] == pytest.approx(ref, abs=1e-6 * scale)
    assert new.diagnostics["value_primal"] == pytest.approx(ref, abs=1e-6 * scale)
    vs = new.diagnostics["value_safe"]
    assert vs >= new.diagnostics["value_dual"] - 1e-9 * scale
    assert vs - new.diagnostics["value_dual"] <= 1e-5 * scale  # the safe bound is tight at an accurate solution
    assert new.diagnostics["primal_max_violation"] <= 1e-6
    assert new.recovered_value <= vs + 1e-9


@pytest.mark.parametrize("k", [0, 3])
def test_sdp1_task_valid_against_exact(k):
    inst = _cases()[k]
    ctrl = bounds.controls(inst, threads=1)
    ex = solve_exact(inst, ctrl, threads=1, time_limit=10)
    new = solve_sdp1(inst, ctrl, threads=1)
    assert new.diagnostics["value_safe"] >= ex.value_lb_verified - 1e-7


@pytest.mark.parametrize("k", [0, 3, 5])
def test_shor_task_equals_cvxpy(k):
    inst = _cases()[k]
    for j in inst.keys:
        lo0, hi0, _ = bounds._shor(inst, j, threads=1)
        lo1, hi1, info = shor_score(inst, j, threads=1)
        sc = max(1.0, abs(hi0), abs(lo0))
        assert lo1 == pytest.approx(lo0, abs=1e-6 * sc) and hi1 == pytest.approx(hi0, abs=1e-6 * sc)
        assert lo1 <= lo0 + 1e-9 * sc or lo1 == pytest.approx(lo0, abs=1e-7 * sc)


def _box_cases():
    return [instances.f_random(s, d=4, d_k=2, d_v=2, n=3, temp=1.0, rho=0.3, set_type="box") for s in range(2)] + [
        instances.f_random(7, d=4, d_k=2, d_v=2, n=3, temp=2.0, rho=0.3, set_type="box", query_in_keys=False),
        instances.phase1_instance("F_random", 0, 2, 0.2, "box", True, 1.0)]


@pytest.mark.parametrize("cut", ["B", "auto"])
@pytest.mark.parametrize("k", range(4))
def test_sdp1_task_equals_cvxpy_boxes(k, cut):
    """Box sets: bounds, diagonal and perspective products as in relax_sdp."""
    inst = _box_cases()[k]
    ctrl = bounds.controls(inst, threads=1)
    old = solve_sdp(inst, ctrl, rlt=("persp", "simplex_x_z"), threads=1, t_interval="tight", cone_form=True,
                    safe="always", cut=cut)
    new = solve_sdp1(inst, ctrl, threads=1, cut=cut)
    assert new.status == "optimal"
    ref = old.diagnostics["value_dual"]
    scale = max(1.0, abs(ref))
    assert new.diagnostics["value_dual"] == pytest.approx(ref, abs=1e-6 * scale)
    vs = new.diagnostics["value_safe"]
    assert new.diagnostics["value_dual"] - 1e-9 * scale <= vs <= new.diagnostics["value_dual"] + 1e-5 * scale
    assert new.diagnostics["mosek_dual_resid"] is not None and new.diagnostics["mosek_dual_resid"] < 1e-6
    ex = solve_exact(inst, ctrl, threads=1, time_limit=10)
    assert vs >= ex.value_lb_verified - 1e-7
