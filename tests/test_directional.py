"""Directional constants (``rad_c``, ``nu_c`` of Section 2) and the directional bounds: the first-order bounds (34),
(35) of Theorem 9 with ``rad_c``, and the directional forms of the norm-based second-order bounds.

On 50 random synthetic instances (d = 8, n = 3): every directional bound is <= its undirected counterpart, and the
directional bounds hold against Gurobi exact values (three-valued with the exact bracket; never ``violated``)."""
import math

import numpy as np
import pytest

from aimp.certify import theorem_check
from aimp.theory import bound_F0s, bound_F0t, directional_constants, directional_fields, support_values
from tests.synthetic_units import SEEDS, synthetic_instance, synthetic_unit

TOL = 1e-6


def _inst(seed: int):
    return synthetic_instance(seed)


def test_rad_c_closed_form_balls():
    inst = _inst(3)
    dc = directional_constants(inst)
    g = inst.head.N.T @ inst.c
    hb = np.array([g @ inst.sets[j].center for j in inst.keys])
    rg = np.array([inst.sets[j].rho * np.linalg.norm(g) for j in inst.keys])
    assert dc["rad_c"] == pytest.approx(((hb + rg).max() - (hb - rg).min()) / 2, rel=1e-12)
    assert dc["nu_c"] == pytest.approx(rg.max(), rel=1e-12)
    # rad_c is half the range of h = c^T N z over the union: support_values gives the max side
    assert dc["h_sup"] == pytest.approx(support_values(inst), rel=1e-12)


@pytest.mark.parametrize("seed", SEEDS)
def test_directional_le_undirected_and_valid(seed):
    u = synthetic_unit(seed)
    inst, ctrl, nom, rec = u["inst"], u["ctrl"], u["nom"], u["rec"]
    slack = rec.get("pruning_slack") or 0.0
    d = directional_fields(inst, ctrl, nom, pruning_slack=slack)
    cn = float(np.linalg.norm(inst.c))
    rel = 1e-12
    # (1) directional <= undirected
    assert d["rad_c"] <= cn * ctrl["rad"] * (1 + rel) + 1e-15
    assert d["nu_c"] <= cn * nom["nu"] * (1 + rel) + 1e-15
    assert d["bound_B_c"] <= ctrl["bound_B"] * (1 + rel) + 1e-15
    if ctrl["bound_C"] is not None:
        assert d["bound_C_c"] <= ctrl["bound_C"] * (1 + rel) + 1e-15
    f0t = bound_F0t(inst, ctrl, nom, mode="tight")["bound_F0t"] + slack
    assert d["bound_F0t_c"] <= f0t * (1 + rel) + 1e-15
    assert rec["bound_F0t"] == pytest.approx(f0t, rel=1e-12)
    fs = bound_F0s(inst, ctrl, nom, slack)
    if fs["F0s_applies"]:
        assert d["bound_F0s_c"] <= fs["bound_F0s"] * (1 + rel) + 1e-15
    else:
        assert d["bound_F0s_c"] is None
    assert d["Pi_c"] <= 2 * d["rad_c"] * (1 + rel)
    # (2) the directional bounds hold against the exact value (bracket lb = incumbent, ub = ObjBound)
    lb, ub = u["lb"], u["ub"]  # bracket: incumbent / ObjBound
    assert lb is not None and math.isfinite(lb)
    checks = {"B_c": theorem_check(rec["r_greedy"], d["bound_B_c"], lb, ub),
              "ps_c": theorem_check(rec["r_greedy"], d["bound_ps_c"], lb, ub),
              "F0t_c": theorem_check(rec["value_safe"], d["bound_F0t_c"], lb, ub)}
    if d["bound_F0s_c"] is not None:
        checks["F0s_c"] = theorem_check(rec["value_safe"], d["bound_F0s_c"], lb, ub)
    if d["bound_C_c"] is not None:
        checks["C_c"] = theorem_check(rec["r_greedy"], d["bound_C_c"], lb, ub)
    for k, ch in checks.items():
        assert ch["status"] != "violated", (seed, k, ch)
    if u["exact_status"] == "optimal":  # exact: every check is decided
        for k, ch in checks.items():
            assert ch["status"] == "satisfied", (seed, k, ch, u["exact_status"])
