"""Forward pass and analytic gradient (vs central finite differences)."""
import numpy as np
import pytest

from aimp import instances
from aimp.heads import forward, grad_phi, softmax


@pytest.mark.parametrize("qk", [True, False])
@pytest.mark.parametrize("seed", range(5))
def test_gradient_fd(seed, qk):
    inst = instances.f_random(seed, d=5, d_k=3, d_v=3, n=4, temp=2.0, query_in_keys=qk)
    rng = np.random.default_rng(100 + seed)
    Z = inst.centers() + 0.3 * rng.standard_normal((inst.T, inst.d))
    _, G = grad_phi(inst, Z)
    h = 1e-6
    Gfd = np.zeros_like(Z)
    for t in range(inst.T):
        for k in range(inst.d):
            E = np.zeros_like(Z)
            E[t, k] = h
            Gfd[t, k] = (forward(inst, Z + E)["phi"] - forward(inst, Z - E)["phi"]) / (2 * h)
    assert np.allclose(G, Gfd, atol=1e-8, rtol=1e-6)


def test_forward_matches_definition():
    inst = instances.f_random(1, d=4, d_k=2, d_v=2, n=3, query_in_keys=False)
    Z = inst.centers()
    out = forward(inst, Z)
    M, N = inst.head.M, inst.head.N
    q = Z[inst.query]
    s = np.array([q @ M @ Z[j] for j in inst.keys])
    a = np.exp(s) / np.exp(s).sum()
    y = sum(a[k] * N @ Z[j] for k, j in enumerate(inst.keys))
    assert np.allclose(out["s"], s) and np.allclose(out["a"], a) and np.allclose(out["y"], y)
    assert abs(out["phi"] - inst.c @ y) < 1e-14
    assert np.isclose(softmax(np.array([1000.0, 0.0]))[0], 1.0)
