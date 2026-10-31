"""LiftIndex bookkeeping of the moment matrix (index bookkeeping is where bugs hide)."""
import numpy as np
import pytest

from aimp import instances, sets as S
from aimp.relax_sdp import LiftIndex


@pytest.mark.parametrize("qk", [True, False])
@pytest.mark.parametrize("n", [1, 2, 4])
def test_dimension_and_partition(n, qk):
    d = 5
    inst = instances.f_random(0, d=d, d_k=2, d_v=2, n=n, rho=0.1, query_in_keys=qk)
    L = LiftIndex.for_instance(inst)
    assert L.D == (n + (0 if qk else 1)) * d + n  # every token's base coordinates, then the n weights
    rows = []
    for t in inst.tokens:
        rows += list(range(L.u(t).start, L.u(t).stop))
    rows += [L.a(k) for k in range(n)]
    assert sorted(rows) == list(range(1, L.D + 1))  # disjoint, cover 1..D
    assert list(range(L.a_slice.start, L.a_slice.stop)) == [L.a(k) for k in range(n)]
    lab = L.labels()
    assert lab[0] == "1" and lab[L.a(n - 1)] == f"a{n - 1}"


def test_degenerate_tokens_not_lifted():
    inst = instances.prop9(0.1)
    L = LiftIndex.for_instance(inst)
    assert L.D == 1 + 2  # only the segment key (r=1) and a_1, a_2
    assert L.u(2).stop - L.u(2).start == 0  # query fixed
    assert L.u(1).stop - L.u(1).start == 0  # key 2 fixed


def test_blocks_of_rank_one_moment_matrix():
    inst = instances.f_random(3, d=3, d_k=2, d_v=2, n=3, rho=0.2, set_type="box", query_in_keys=False)
    L = LiftIndex.for_instance(inst)
    rng = np.random.default_rng(0)
    xi = np.zeros(L.size)
    xi[0] = 1
    us = {}
    for t in inst.tokens:
        rep = S.affine(inst.sets[t])
        us[t] = S.base_sample(rep, rng, 1)[0]
        xi[L.u(t)] = us[t]
    a = rng.dirichlet(np.ones(inst.n))
    for k in range(inst.n):
        xi[L.a(k)] = a[k]
    X = np.outer(xi, xi)
    for k, j in enumerate(inst.keys):
        assert np.allclose(X[L.a(k), L.u(j)], a[k] * us[j])
        assert np.isclose(X[L.a(k), L.a(k)], a[k] ** 2)
        assert np.allclose(X[L.u(inst.query), L.u(j)], np.outer(us[inst.query], us[j]))
    assert np.allclose(X[0, L.a_slice], a)
