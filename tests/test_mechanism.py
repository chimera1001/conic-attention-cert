"""Tools of the mechanism test (Section 7) on a small random GPT-2-style block (d = 16, 2 heads of 8, MLP 64, 3 tokens, token
1 perturbed): the block's numpy / torch forward passes and Jacobian; the remainder network (``g(0) = 0``, equal to the numpy
remainder) and CROWN's rows on it (they contain every sampled value; second order in the radius); the tape equals the forward
pass and matches auto_LiRPA's nonlinear nodes one to one; the Hessian identity eq. (H0) holds for both softmax readings; the
chains of Lemma "chain"; the softmax Hessians and the small box SDP; the per-operation terms on given half-widths; the
attacks return feasible points whose values lie inside CROWN's bounds; the read layout of the downstream head."""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")

from aimp import block as B  # noqa: E402
from aimp import mechanism as MC  # noqa: E402


def _block(seed: int = 0, d: int = 16, H: int = 2, dh: int = 8):
    rng = np.random.default_rng(19_240 + seed)

    def ln():
        return {"g": 1 + 0.1 * rng.standard_normal(d), "b": 0.1 * rng.standard_normal(d)}

    l1, l2, ln_ = ln(), ln(), ln()
    W = {"ln1_g": l1["g"], "ln1_b": l1["b"], "ln1_eps": 1e-5, "W_attn": rng.standard_normal((d, 3 * d)) / math.sqrt(d),
         "b_attn": 0.1 * rng.standard_normal(3 * d), "W_proj": rng.standard_normal((d, d)) / math.sqrt(d),
         "b_proj": 0.1 * rng.standard_normal(d), "ln2_g": l2["g"], "ln2_b": l2["b"], "ln2_eps": 1e-5,
         "W_fc": rng.standard_normal((d, 4 * d)) / math.sqrt(d), "b_fc": 0.1 * rng.standard_normal(4 * d),
         "W_mp": rng.standard_normal((4 * d, d)) / math.sqrt(4 * d), "b_mp": 0.1 * rng.standard_normal(d)}
    Wn = {"ln1_g": ln_["g"], "ln1_b": ln_["b"], "ln1_eps": 1e-5}
    Bk = B.GPT2Block(W, Wn, d_head=dh, n_heads=H)
    X = rng.standard_normal((3, d)) * 2.0
    return Bk, X, rng


def _setup(rho_rel: float = 0.05, seed: int = 0):
    Bk, X, rng = _block(seed)
    jac = Bk.jacobians(X, check=True)
    rv = B.radii(X, rho_rel, (1,))
    C = rng.standard_normal(X.size)
    return Bk, X, jac, rv, C, rng


def test_forward_and_jacobian():
    Bk, X, jac, *_ = _setup()
    with torch.no_grad():
        zt = Bk.t_forward(torch, torch.as_tensor(X)).numpy()
    assert np.max(np.abs(zt - jac["f"]["zp"])) <= 1e-12
    assert jac["J_check_rel"] <= 1e-12 and jac["J_causal_max"] == 0.0
    assert B.influenced(jac["J"], (1,)) == [1, 2]


def test_remainder_net_gate():
    Bk, X, jac, rv, *_ = _setup(0.01)
    J, zb, d = jac["J"], jac["f"]["zp"], X.shape[1]
    mod = B.remainder_net(Bk, X, [1], J, zb)
    D = B.sample_deltas(np.random.default_rng(1), 40, rv, d)
    assert np.allclose(np.linalg.norm(D[:20, 1], axis=1), rv[1]) and np.all(np.linalg.norm(D[:, 1], axis=1) <= rv[1] * (1 + 1e-12))
    assert not np.any(D[:, [0, 2]])
    ref = B.numpy_remainder(Bk, X, J, zb, D)
    with torch.no_grad():
        assert np.max(np.abs(mod(torch.zeros(1, d, dtype=torch.float64)).numpy())) <= 1e-12
        dev = max(float(np.max(np.abs(mod(torch.as_tensor(D[s, 1][None])).numpy().reshape(3, d) - ref[s])))
                  for s in range(len(D)))
    assert dev <= 1e-12


def test_crown_rows_contain_samples_and_second_order():
    pytest.importorskip("auto_LiRPA")
    vs = []
    for rho_rel in (0.004, 0.002):
        Bk, X, jac, rv, C, rng = _setup(rho_rel)
        J, zb, d = jac["J"], jac["f"]["zp"], X.shape[1]
        Cm = np.vstack([C, rng.standard_normal((2, X.size))])
        bm, bx = B.bounded(B.remainder_net(Bk, X, [1], J, zb), rv, [1], crown_batch=64)
        r = B.crown_rows(bm, bx, Cm, [1], rv)
        assert np.allclose(r["l"], r["lb"], rtol=1e-12, atol=1e-15) and np.allclose(r["v"], r["ub"], rtol=1e-12, atol=1e-15)
        D = B.sample_deltas(np.random.default_rng(2), 400, rv, d)
        Y = B.numpy_remainder(Bk, X, J, zb, D).reshape(400, -1) @ Cm.T
        assert np.all(Y >= r["l"][None] - 1e-12) and np.all(Y <= r["v"][None] + 1e-12)
        # the affine bounds (return_A) hold at every sample too
        lo = np.einsum("fkd,nkd->nf", r["LamL"], D[:, [1]]) + r["betaL"]
        hi = np.einsum("fkd,nkd->nf", r["LamU"], D[:, [1]]) + r["betaU"]
        assert np.all(Y >= lo - 1e-12) and np.all(Y <= hi + 1e-12)
        vs.append(float(np.max(np.maximum(-r["l"], r["v"]))))
    assert 2.5 <= vs[0] / vs[1] <= 6.0, vs  # second order: ~4 when rho halves


def test_tape_equals_forward():
    Bk, X, jac, rv, *_ = _setup()
    rng = np.random.default_rng(1)
    for _ in range(5):
        D = rng.standard_normal(X.shape[1]) * rv[1]
        Df = np.zeros_like(X)
        Df[1] = D
        with torch.no_grad():
            zp, ops = MC.tape(Bk, X, [1], torch.as_tensor(D))
        assert np.max(np.abs(zp.numpy() - Bk.forward(X + Df)["zp"])) <= 1e-12
    assert len(ops) == 3 * 4 + 3 * 3 + 4  # LN: sqr, sqrt, recip, mul; attention per row: qk, softmax, av; GELU: 4


def test_structure_matches_auto_lirpa():
    pytest.importorskip("auto_LiRPA")
    Bk, X, jac, rv, *_ = _setup()
    bm, _ = B.bounded(B.remainder_net(Bk, X, [1], jac["J"], jac["f"]["zp"]), rv, [1])
    st = MC.structure_check(bm, Bk, X, [1], rv)
    assert st["ok"], st
    assert st["node_counts"] == {"BoundSqr": 3, "BoundSqrt": 3, "BoundReciprocal": 3, "BoundMul": 3 + 3 + 3 + 3,
                                 "BoundSoftmax": 3, "BoundTanh": 1}


@pytest.mark.parametrize("softmax", ["lse", "complex"])
def test_identity_and_chains(softmax):
    Bk, X, jac, rv, C, _ = _setup()
    rho = float(rv[1])
    shifts = MC.nominal_shifts(Bk, X, [1]) if softmax == "complex" else None
    jops, _ = MC.operand_jacobians(Bk, X, [1], softmax, shifts, chunk=7)
    lam = MC.adjoints(Bk, X, [1], C[None], softmax, shifts)[0]
    recs = MC.decompose(jops, lam, rho)
    H0, asym, gfun = MC.hessian_t(Bk, X, [1], C)
    Hd = sum(r["Q"] for r in recs)
    assert asym <= 1e-12
    assert np.linalg.norm(Hd - H0) <= 1e-10 * np.linalg.norm(H0)
    fd = MC.fd_check(H0, gfun, float(np.linalg.norm(X[1])), np.random.default_rng(0), 3)
    assert fd["max_best"] <= 1e-6
    sp = MC.spectral(H0, rho)
    Pp, Pm = sum(r["Pp"] for r in recs), sum(r["Pm"] for r in recs)
    tol = 1e-9 * Pp
    assert sp["s_spec_p"] <= sp["half_trace_p"] + tol <= Pp + 2 * tol
    assert sp["s_spec_m"] <= sp["half_trace_m"] + tol <= Pm + 2 * tol
    assert sp["eff_rank_p"] >= 1.0 - 1e-12
    for key in (lambda r: r["group"], lambda r: MC.LAYER_OF[r["group"]], lambda r: r["name"]):
        g = MC.grouped(recs, rho, key)
        assert sp["s_spec_p"] <= g["L_p"] + tol and g["L_p"] <= Pp + tol
        assert sp["s_spec_m"] <= g["L_m"] + tol and g["L_m"] <= Pm + tol


def test_p_t_widths_on_linearized_widths_equal_s_op():
    """With the half-widths ``rho ||g||`` in place of CROWN's, the per-operation terms are the per-operation sum ``s_op``."""
    Bk, X, jac, rv, C, _ = _setup()
    rho = float(rv[1])
    jops, _ = MC.operand_jacobians(Bk, X, [1], "lse")
    lam = MC.adjoints(Bk, X, [1], C[None], "lse")[0]
    recs = MC.decompose(jops, lam, rho)
    hw = {o["name"]: tuple(rho * np.linalg.norm(Jx.reshape(Jx.shape[0], -1).numpy(), axis=0).reshape(tuple(a.shape))
                           for Jx, a in zip(o["jac"], o["args"])) for o in jops}
    w = MC.p_t_widths(jops, lam, hw)
    for k, kk in (("P_p", "Pp"), ("P_m", "Pm"), ("Pc_p", "Pcp"), ("Pc_m", "Pcm")):
        ref = sum(r[kk] for r in recs)
        assert abs(w[k] - ref) <= 1e-10 * ref, (k, w[k], ref)


def test_crown_halfwidths_contain_operands():
    pytest.importorskip("auto_LiRPA")
    Bk, X, jac, rv, C, _ = _setup(0.01)
    J, zb, d = jac["J"], jac["f"]["zp"], X.shape[1]
    bm, bx = B.bounded(B.remainder_net(Bk, X, [1], J, zb), rv, [1])
    st = MC.structure_check(bm, Bk, X, [1], rv, 3)
    B.crown_rows(bm, bx, C[None], [1], rv)
    hw = MC.crown_halfwidths(bm, Bk, X, [1], rv, st["pairs"], 4)
    with torch.no_grad():
        _, ops0 = MC.tape(Bk, X, [1], torch.zeros(d, dtype=torch.float64))
        D = B.sample_deltas(np.random.default_rng(5), 30, rv, d)
        for s in range(len(D)):
            _, ops = MC.tape(Bk, X, [1], torch.as_tensor(D[s, 1]))
            for o, o0 in zip(ops, ops0):
                for a, a0, h in zip(o["args"], o0["args"], hw[o["name"]]):
                    # CROWN's interval of an operand contains its range; it is centred near the nominal value
                    assert np.all(np.abs(a.numpy() - a0.numpy()) <= 2 * h + 1e-12)


def test_softmax_hessians():
    from torch.func import hessian

    rng = np.random.default_rng(3)
    s = rng.standard_normal(4)
    p = np.exp(s - s.max())
    p /= p.sum()
    Hs = MC.softmax_hessians(p)
    for k in range(4):
        Hk = hessian(lambda x: torch.softmax(x, 0)[k])(torch.as_tensor(s)).numpy()
        assert np.max(np.abs(Hs[k] - Hk)) <= 1e-13


def test_sdp_box():
    from scipy.optimize import minimize_scalar

    rng = np.random.default_rng(5)
    for _ in range(40):
        A = rng.standard_normal((2, 2))
        A = A + A.T
        v, flag = MC.sdp_box(A)
        assert flag == "exact"
        # X = [[x, z], [z, y]] >= 0, x, y <= 1: f(x, y) = A00 x + A11 y + 2 |A01| sqrt(x y); concave and positively
        # homogeneous, so the maximum lies on {x = 1} or {y = 1} (or 0)
        f = lambda x, y: A[0, 0] * x + A[1, 1] * y + 2 * abs(A[0, 1]) * np.sqrt(x * y)  # noqa: E731
        e1 = -minimize_scalar(lambda y: -f(1.0, y), bounds=(0, 1), method="bounded", options={"xatol": 1e-12}).fun
        e2 = -minimize_scalar(lambda x: -f(x, 1.0), bounds=(0, 1), method="bounded", options={"xatol": 1e-12}).fun
        ref = max(0.0, e1, e2, f(1.0, 1.0), f(1.0, 0.0), f(0.0, 1.0))
        assert abs(v - ref) <= 1e-8 * max(1.0, abs(v))
        P = rng.uniform(0, 1, (2000, 2))
        assert np.all(f(P[:, 0], P[:, 1]) <= v + 1e-12)
        assert v >= MC.sign_max(A) - 1e-12
        assert v >= max(np.linalg.eigvalsh(A)[-1], 0.0) - 1e-12
    assert MC.sdp_box(np.array([[-2.0]]))[0] == 0.0 and MC.sdp_box(np.array([[3.0]]))[0] == 3.0
    A = rng.standard_normal((4, 4))
    A = A + A.T
    v, flag = MC.sdp_box(A)
    assert flag != "exact" and v >= MC.sign_max(A) - 1e-12
    best = max(np.array(s) @ A @ np.array(s) for s in itertools.product((1, -1), repeat=4))
    assert abs(MC.sign_max(A) - max(best, 0.0)) <= 1e-12


def test_attacks_feasible_and_inside_crown():
    pytest.importorskip("auto_LiRPA")
    Bk, X, jac, rv, C, rng = _setup(0.01)
    J, zb, d = jac["J"], jac["f"]["zp"], X.shape[1]
    Cs = np.vstack([C, rng.standard_normal(X.size)])
    bm, bx = B.bounded(B.remainder_net(Bk, X, [1], J, zb), rv, [1])
    cr = B.crown_rows(bm, bx, Cs, [1], rv)
    ar = B.attacked_remainder(Bk, X, J, zb, rv, 2, 7, n_samples=200, n_best=3, n_random=2, maxiter=50)
    assert ar["value"] >= ar["sampled"] - 1e-15 and np.linalg.norm(ar["delta"][1]) <= rv[1] * (1 + 1e-12)
    assert not np.any(ar["delta"][[0, 2]])
    ms = B.attack_total_rows(Bk, X, J, zb, rv, [1], Cs, cr["LamU"], cr["LamL"], ar["delta"], 11, (0, 1),
                             B.n_unit_rows([1, 2], T=3, pos=2), steps=30, n_samples=200)
    for i in range(2):
        assert ms["max"][i] >= ms["max_samples"][i] and ms["min"][i] <= ms["min_samples"][i]
        assert cr["l"][i] - 1e-12 <= ms["min"][i] <= ms["max"][i] <= cr["v"][i] + 1e-12
    D0 = B.lam_starts(cr["LamU"], rv, [1])
    assert np.allclose(np.linalg.norm(D0, axis=2), rv[1])
    R = B.random_starts(np.random.default_rng(0), 3, rv, [1], d)
    assert R.shape == (3, 1, d) and np.allclose(np.linalg.norm(R, axis=2), rv[1])
    res = B.pgd_rows(Bk, X, J, zb, rv, [1], Cs, np.array([1.0, -1.0]), np.concatenate([D0[:1], R[:1]]), 20)
    Df = np.zeros((2, 3, d))
    Df[:, [1]] = res["delta"]
    vals = np.einsum("nc,nc->n", B.numpy_remainder(Bk, X, J, zb, Df).reshape(2, -1), Cs) * np.array([1.0, -1.0])
    assert np.allclose(vals, res["value"], rtol=1e-8, atol=1e-14)
    assert np.all(np.linalg.norm(res["delta"], axis=2) <= rv[1] * (1 + 1e-12))


def test_read_matrix_and_total_row_gradient():
    """The head's objective depends on token ``j`` only through ``A_j zp_j``: its gradient lies in the row space of
    ``A_j`` (the residual :func:`aimp.block.total_rows` reports)."""
    from aimp.gpt2 import projected_head
    from aimp.heads import grad_phi
    from aimp.types import Instance

    rng = np.random.default_rng(9)
    dk, T, pos = 8, 5, 4
    W_O = rng.standard_normal((dk, 16))
    head = projected_head(W_O, dk, dk)
    c = rng.standard_normal(16)
    Zc = rng.standard_normal((T, 3 * dk))
    _, G0 = grad_phi(Instance(head=head, keys=list(range(T)), query=pos, sets={}, c=c), Zc)
    for j in range(T):
        A = B.read_matrix(j, W_O @ c, pos=pos, dk=dk)
        assert A.shape == ((2 * dk if j == pos else dk) + 1, 3 * dk)
        gam = np.linalg.pinv(A).T @ G0[j]
        assert np.max(np.abs(A.T @ gam - G0[j])) <= 1e-12 * max(1.0, np.max(np.abs(G0[j])))
    assert B.n_unit_rows([1, 2, 3, 4]) == 14
