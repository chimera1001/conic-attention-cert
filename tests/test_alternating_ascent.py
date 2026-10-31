"""The alternating ascent of :func:`aimp.lower.alternating_ascent`: the exact maximization over the keys (Theorem 7 of
the paper) alternated with local ascent over the query; every value is admissible and never exceeds the first-order
certificate."""
import numpy as np
import pytest

from aimp import bounds, instances
from aimp.exact_fq import fix_query


def _cases():
    out = []
    for s in range(3):
        inst = instances.f_random(18_4_00 + s, d=6, d_k=3, d_v=3, n=4, temp=2.0, rho=0.15, set_type="ball",
                                  query_in_keys=True)
        out += [inst, fix_query(inst)]
    return out


def test_alternating_ascent():
    from aimp.lower import alternating_ascent
    from aimp.heads import forward
    from aimp.lower import ascend
    from aimp.lower_bounds import _Param
    from aimp.relax_ps import solve_ps_greedy

    for s in range(3):
        inst = instances.f_random(18_5_00 + s, d=6, d_k=3, d_v=3, n=4, temp=2.0, rho=0.2, set_type="ball",
                                  query_in_keys=True)
        ctrl = bounds.controls(inst, threads=1)
        v0, Z0, _ = ascend(inst, _Param(inst), inst.centers())
        a = alternating_ascent(inst, Z0, spread=1.0)
        assert a["value"] >= v0 - 1e-12
        assert a["admissible_excess"] <= 1e-9
        assert forward(inst, a["Z"])["phi"] == pytest.approx(a["value"], abs=1e-12)
        assert a["value"] <= solve_ps_greedy(inst, ctrl).value + 1e-9
        assert 1 <= a["rounds"] <= 20
