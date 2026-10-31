"""Second-order terms of a GPT-2 block remainder, operation by operation (Section 6; the mechanism test of Section 7).

The block map (block ``l`` and the ``ln_1`` of block ``l + 1``, :func:`aimp.block.remainder_net`) is re-implemented here as
a *tape* of the elementary operations auto_LiRPA traces in it (matched node by node, by value, against the
``BoundedModule``: :func:`structure_check`):

* unary: ``Sqr`` (the variance's ``c * c`` of a layer normalization; auto_LiRPA turns the product of a tensor with itself
  into ``BoundSqr``), ``Sqrt``, ``Reciprocal`` (layer normalizations), ``Tanh`` (GELU-tanh);
* products (``Mul`` with both operands depending on ``delta``): ``c * rsqrt`` of each layer normalization, the score products
  ``q * k``, the value products ``a * v``, and the three products of GELU-tanh ``(a pre) pre``, ``(a pre^2) pre``,
  ``(pre / 2)(1 + tanh(.))``;
* ``Softmax`` per query row and head: auto_LiRPA 0.8 keeps it as **one** operation (``softmax = "lse"``: an affine bound per
  output ``a_k`` as a function of the row's scores). The alternative tape ``softmax = "complex"`` writes it through
  exponentials, a sum, a reciprocal and products, as Lemma "chain" (eq. (H0)) does.

For ``psi(delta) = C . zp(delta)`` (``f`` of the mechanism test: the remainder's linear part and constant do not change
second derivatives) the Hessian at ``delta = 0`` is eq. (H0) of Lemma "chain",
``H_0 = sum_unary lam kappa g g^T + sum_product lam (g_x g_y^T + g_y g_x^T) [+ sum_softmax sum_k G_s^T (lam_k Hess p_k) G_s]``,
with the adjoints ``lam = d psi / d(operation output)`` (:func:`adjoints`), the operand gradients ``g = d(operand) / d delta``
(:func:`operand_jacobians`) and the second derivatives at the nominal point (:func:`decompose` returns the terms; their sum
equals the Hessian of :func:`hessian_t`, an implementation independent of the tape).

The per-operation sum ``s_op`` of Lemma "chain" (``P^±`` below; ``+`` for ``max psi``, ``-`` for ``min psi``) on a ball of
radius ``rho``: unary ``(1/2) rho^2 (lam kappa)_± ||g||^2``, product ``rho^2 |lam| ||g_x|| ||g_y||``; for a softmax output ``k``
(an operation of several inputs, the softmax counted as one operation) the same construction gives
``(1/2) rho^2 sdp_box(± D K_k D)`` with ``K_k = lam_k Hess p_k``, ``D = diag(||g_i||)`` and ``sdp_box(A) = max tr(A X)`` over
``X >= 0, diag X <= 1``. It equals the unary / product terms for one- and two-input operations and keeps the chain
``s_spec <= (1/2) rho^2 tr(H_0^+) <= s_op`` of Lemma "chain" valid. With CROWN's own operand intervals in place of the
linearized half-widths ``rho ||g||`` (:func:`crown_halfwidths`, :func:`p_t_widths`), the terms are those of the lower bound
(eq. (Tlower)) of Proposition T(iii).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

#: second derivative of each unary operation at its (nominal) input
KAPPA = {
    "sqr": lambda x: np.full_like(x, 2.0),
    "sqrt": lambda x: -0.25 * x ** -1.5,
    "recip": lambda x: 2.0 * x ** -3.0,
    "tanh": lambda x: -2.0 * np.tanh(x) * (1.0 - np.tanh(x) ** 2),
    "exp": lambda x: np.exp(x),
}
#: auto_LiRPA node class of each tape operation kind (structure check)
BOUND_CLASS = {"sqr": "BoundSqr", "sqrt": "BoundSqrt", "recip": "BoundReciprocal", "tanh": "BoundTanh", "exp": "BoundExp",
               "bilinear": "BoundMul", "softmax": "BoundSoftmax"}
#: coarse partition ("layers") of the operation groups
LAYER_OF = {"ln1_vareps": "LN1", "ln1_rsqrt": "LN1", "ln1_out": "LN1", "scores": "attention", "softmax": "attention",
            "att": "attention", "ln2_vareps": "LN2", "ln2_rsqrt": "LN2", "ln2_out": "LN2", "gelu": "MLP",
            "lnf_vareps": "LN1+", "lnf_rsqrt": "LN1+", "lnf_out": "LN1+"}


def tape(Bk, X: np.ndarray, pert: list[int], D, softmax: str = "lse", probes: dict | None = None,
         shifts: dict | None = None):
    """The block map as a tape of elementary operations (module docstring) at ``X + delta`` (``D``: torch, ``(|K_p| d,)``,
    the perturbations of the tokens ``pert``). ``probes``: tensors added to the operation outputs (zeros: the adjoints are
    their gradients). ``shifts``: the constant subtracted inside ``exp`` per query row (``softmax = "complex"``; any constant
    leaves the softmax unchanged). Returns ``zp`` ``(T, d)`` and the operations in forward order, each ``{"name", "group",
    "kind", "fn", "args", "out"}`` (``kind``: ``unary`` / ``bilinear`` / ``softmax``)."""
    import torch

    from aimp.block import GELU_A, GELU_K

    T, d = X.shape
    H, dh = Bk.H, Bk.dh
    p = Bk.torch_params(torch)
    K = len(pert)
    full = torch.zeros(T, d, dtype=torch.float64).index_copy(0, torch.tensor(pert), D.reshape(K, d))
    Xi = (torch.as_tensor(X, dtype=torch.float64) + full)[None]  # (1, T, d) as the remainder network
    ops = []

    def rec(name, group, kind, fn, args, out):
        if probes is not None and name in probes:
            out = out + probes[name]
        ops.append({"name": name, "group": group, "kind": kind, "fn": fn, "args": args, "out": out})
        return out

    def ln(x, g, b, eps, tag):
        c = x - x.mean(-1, keepdim=True)
        sq_c = rec(f"{tag}/sqr", f"{tag}_vareps", "unary", "sqr", (c,), c * c)
        ve = sq_c.mean(-1, keepdim=True) + eps
        sq = rec(f"{tag}/sqrt", f"{tag}_rsqrt", "unary", "sqrt", (ve,), torch.sqrt(ve))
        rs = rec(f"{tag}/recip", f"{tag}_rsqrt", "unary", "recip", (sq,), torch.reciprocal(sq))
        y = rec(f"{tag}/mul", f"{tag}_out", "bilinear", None, (c, rs), c * rs)
        return y * g + b

    z = ln(Xi, p["g1"], p["b1"], Bk.e1, "ln1")
    qkv = z @ p["W_attn"] + p["b_attn"]
    q = qkv[:, :, :d].reshape(1, T, H, dh).transpose(1, 2)
    k_ = qkv[:, :, d:2 * d].reshape(1, T, H, dh).transpose(1, 2)
    v = qkv[:, :, 2 * d:].reshape(1, T, H, dh).transpose(1, 2)
    outs = []
    for j in range(T):
        qj, kj, vj = q[:, :, j:j + 1, :], k_[:, :, :j + 1, :], v[:, :, :j + 1, :]
        s = rec(f"att/qk@{j}", "scores", "bilinear", None, (qj, kj), qj * kj).sum(-1) / math.sqrt(dh)
        if softmax == "lse":
            a = rec(f"att/softmax@{j}", "softmax", "softmax", None, (s,), torch.softmax(s, dim=-1))
        else:
            sh = s - shifts[j]
            e = rec(f"att/exp@{j}", "softmax", "unary", "exp", (sh,), torch.exp(sh))
            S = e.sum(-1, keepdim=True)
            r = rec(f"att/recip@{j}", "softmax", "unary", "recip", (S,), torch.reciprocal(S))
            a = rec(f"att/div@{j}", "softmax", "bilinear", None, (e, r), e * r)
        au = a.unsqueeze(-1)
        o = rec(f"att/av@{j}", "att", "bilinear", None, (au, vj), au * vj).sum(2)
        outs.append(o.reshape(1, 1, H * dh))
    att = torch.cat(outs, 1) @ p["W_proj"] + p["b_proj"]
    x1 = Xi + att
    u = ln(x1, p["g2"], p["b2"], Bk.e2, "ln2")
    pre = u @ p["W_fc"] + p["b_fc"]
    hp = 0.5 * pre
    ap = GELU_A * pre
    b1 = rec("mlp/sq", "gelu", "bilinear", None, (ap, pre), ap * pre)
    b2 = rec("mlp/cube", "gelu", "bilinear", None, (b1, pre), b1 * pre)
    tin = GELU_K * (pre + b2)
    th = rec("mlp/tanh", "gelu", "unary", "tanh", (tin,), torch.tanh(tin))
    w = 1.0 + th
    gl = rec("mlp/mul", "gelu", "bilinear", None, (hp, w), hp * w)
    x2 = x1 + gl @ p["W_mp"] + p["b_mp"]
    zp = ln(x2, p["gn"], p["bn"], Bk.en, "lnf")
    return zp[0], ops


def nominal_shifts(Bk, X: np.ndarray, pert: list[int]) -> dict:
    """The nominal row maxima of the scores (constants for ``softmax = "complex"``)."""
    import torch

    with torch.no_grad():
        _, ops = tape(Bk, X, pert, torch.zeros(len(pert) * X.shape[1], dtype=torch.float64))
    return {int(o["name"].split("@")[1]): o["args"][0].max(-1, keepdim=True).values.clone()
            for o in ops if o["kind"] == "softmax"}


def operand_jacobians(Bk, X: np.ndarray, pert: list[int], softmax: str = "lse", shifts: dict | None = None,
                      chunk: int = 128) -> tuple[list[dict], list]:
    """Forward-mode Jacobians of every operand at ``delta = 0`` (the gradients ``g`` of Lemma "chain"): for each operation its
    nominal operands, output and ``jac`` = tuple of ``(n, *operand shape)`` tensors (``n = |K_p| d``, by ``vmap(jvp)`` over
    chunks of the basis)."""
    import torch
    from torch.func import jvp, vmap

    d = X.shape[1]
    n = len(pert) * d
    D0 = torch.zeros(n, dtype=torch.float64)
    with torch.no_grad():
        _, ops0 = tape(Bk, X, pert, D0, softmax, shifts=shifts)

    def f(Dv):
        _, ops = tape(Bk, X, pert, Dv, softmax, shifts=shifts)
        return tuple(a for o in ops for a in o["args"])

    parts = []
    eye = torch.eye(n, dtype=torch.float64)
    for a in range(0, n, chunk):
        E = eye[a:a + chunk]
        tans = vmap(lambda e: jvp(f, (D0,), (e,))[1])(E)
        parts.append([t.detach() for t in tans])
    flat = [torch.cat([p_[i] for p_ in parts], 0) for i in range(len(parts[0]))]
    out, i = [], 0
    for o in ops0:
        na = len(o["args"])
        out.append({"name": o["name"], "group": o["group"], "kind": o["kind"], "fn": o["fn"],
                    "args": tuple(x.detach() for x in o["args"]), "out": o["out"].detach(), "jac": tuple(flat[i:i + na])})
        i += na
    return out, ops0


def adjoints(Bk, X: np.ndarray, pert: list[int], Cs: np.ndarray, softmax: str = "lse", shifts: dict | None = None) -> list[dict]:
    """The adjoints ``lam = d (C . zp) / d(operation output)`` at ``delta = 0`` for every row ``C`` of ``Cs`` (``(m, T d)``): one
    forward pass with zero probes on every output, one backward pass per row. Returns per row ``{op name: lam (output
    shape)}``."""
    import torch

    d = X.shape[1]
    D0 = torch.zeros(len(pert) * d, dtype=torch.float64)
    with torch.no_grad():
        _, ops0 = tape(Bk, X, pert, D0, softmax, shifts=shifts)
    probes = {o["name"]: torch.zeros_like(o["out"]).requires_grad_(True) for o in ops0}
    zp, ops = tape(Bk, X, pert, D0, softmax, probes=probes, shifts=shifts)
    names = [o["name"] for o in ops]
    res = []
    for r, C in enumerate(np.atleast_2d(Cs)):
        psi = (zp.reshape(-1) * torch.as_tensor(C, dtype=torch.float64)).sum()
        gr = torch.autograd.grad(psi, [probes[nm] for nm in names], retain_graph=r < len(Cs) - 1, allow_unused=True)
        res.append({nm: (g.detach().numpy().copy() if g is not None else np.zeros(tuple(probes[nm].shape)))
                    for nm, g in zip(names, gr)})
    return res


def softmax_hessians(p: np.ndarray) -> np.ndarray:
    """``Hess p_k`` (``(n, n, n)``, first index ``k``) of the softmax at probabilities ``p``:
    ``d^2 p_k / ds_i ds_j = p_k [(delta_ki - p_i)(delta_kj - p_j) - p_i (delta_ij - p_j)]``."""
    n = p.shape[0]
    I = np.eye(n)
    A = I - p[None, :]  # A[k, i] = delta_ki - p_i
    return p[:, None, None] * (A[:, :, None] * A[:, None, :] - (p[:, None] * (I - p[None, :]))[None])


def sdp_box(A: np.ndarray) -> tuple[float, str]:
    """``max tr(A X)`` over ``X >= 0``, ``diag X <= 1`` (``A`` symmetric ``m x m``): exact for ``m <= 2`` (positively
    homogeneous and concave in the diagonal: the maximum lies on the boundary of the box), the diagonal-dominance upper
    bound ``sum_i (A_ii + sum_{j != i} |A_ij|)_+`` for ``m >= 3`` (a valid dual point; flagged)."""
    m = A.shape[0]
    if m == 0:
        return 0.0, "exact"
    if m == 1:
        return max(float(A[0, 0]), 0.0), "exact"
    if m == 2:
        a, c, b = float(A[0, 0]), float(A[1, 1]), abs(float(A[0, 1]))

        def edge(a_, c_):  # x = 1, y in [0, 1]: a_ + c_ y + 2 b sqrt(y)
            if c_ >= 0 or b >= -c_:
                return a_ + c_ + 2 * b
            return a_ + b * b / (-c_)
        return max(0.0, a, c, edge(a, c), edge(c, a)), "exact"
    R = np.sum(np.abs(A), axis=1) - np.abs(np.diag(A))
    return float(np.sum(np.maximum(np.diag(A) + R, 0.0))), "diag-dominance bound"


def sign_max(A: np.ndarray) -> float:
    """``max(0, max over sign vectors s of s^T A s)``: the second-order box maximum, the term of the lower bound (eq.
    (Tlower)) of Proposition T(i) for an operation of several inputs."""
    m = A.shape[0]
    if m == 0:
        return 0.0
    best = -math.inf
    for s in itertools.product((1.0, -1.0), repeat=m - 1):
        v = np.array((1.0,) + s)
        best = max(best, float(v @ A @ v))
    return max(best, 0.0)


def _rows(J) -> np.ndarray:
    """``(N, n)`` gradient rows of an operand Jacobian ``(n, *shape)``."""
    return J.reshape(J.shape[0], -1).T.numpy()


def _bcast(J, shape) -> np.ndarray:
    return J.expand(J.shape[0], *shape).reshape(J.shape[0], -1).T.numpy()


def decompose(jops: list[dict], lam: dict, rho: float, keep_tol: float = 0.0) -> list[dict]:
    """Per-operation terms of one unit (one row's adjoints ``lam``, :func:`adjoints`) on a ball of radius ``rho``: the
    terms ``Q`` of eq. (H0) (``n x n``; they sum to ``H_0``), the per-operation sums ``Pp`` / ``Pm`` (``s_op`` of Lemma
    "chain" for ``max`` / ``min``; the softmax by :func:`sdp_box`), the variant ``Pcp`` / ``Pcm`` with CROWN's relaxation of
    the square (auto_LiRPA's ``BoundSqr``: the secant above, the tangent at an interval endpoint below, so the centre slack
    is ``|lam| h^2`` on both sides), and the softmax sign maxima."""
    res = []
    for o in jops:
        lm = np.asarray(lam[o["name"]], float)
        rec = {"name": o["name"], "group": o["group"], "kind": o["kind"], "fn": o["fn"]}
        if o["kind"] == "unary":
            G = _rows(o["jac"][0])
            x = o["args"][0].numpy().reshape(-1)
            w = lm.reshape(-1) * KAPPA[o["fn"]](x)
            g2 = np.einsum("ij,ij->i", G, G)
            keep = g2 > keep_tol
            Gk = G[keep]
            Q = Gk.T @ (w[keep, None] * Gk)
            pp = 0.5 * rho ** 2 * float(np.sum(np.maximum(w, 0) * g2))
            pm = 0.5 * rho ** 2 * float(np.sum(np.maximum(-w, 0) * g2))
            rec.update(Q=Q, Pp=pp, Pm=pm, n_active=int(keep.sum()))
            if o["fn"] == "sqr":
                pc = 0.5 * rho ** 2 * float(np.sum(np.abs(w) * g2))
                rec.update(Pcp=pc, Pcm=pc)
            else:
                rec.update(Pcp=pp, Pcm=pm)
        elif o["kind"] == "bilinear":
            shp = tuple(o["out"].shape)
            Gx, Gy = _bcast(o["jac"][0], shp), _bcast(o["jac"][1], shp)
            lv = lm.reshape(-1)
            nx, ny = np.linalg.norm(Gx, axis=1), np.linalg.norm(Gy, axis=1)
            keep = (nx * ny) > keep_tol
            M = Gx[keep].T @ (lv[keep, None] * Gy[keep])
            pb = rho ** 2 * float(np.sum(np.abs(lv) * nx * ny))
            rec.update(Q=M + M.T, Pp=pb, Pm=pb, Pcp=pb, Pcm=pb, n_active=int(keep.sum()))
        else:  # softmax, one operation per (head, query row); outputs p_k of the row's scores
            Js = o["jac"][0]  # (n, 1, H, m)
            pbar = o["out"].numpy()[0]  # (H, m)
            H_, m = pbar.shape
            Q = np.zeros((Js.shape[0], Js.shape[0]))
            pp = pm = sp = sm = 0.0
            flags = set()
            na = 0
            for hh in range(H_):
                G = Js[:, 0, hh, :].T.numpy()  # (m, n)
                gn = np.linalg.norm(G, axis=1)
                act = gn > keep_tol
                if not act.any():
                    continue
                Hs = softmax_hessians(pbar[hh])  # (m, m, m)
                lk = lm[0, hh]
                Ksum = np.einsum("k,kij->ij", lk, Hs)
                Ga = G[act]
                Q += Ga.T @ Ksum[np.ix_(act, act)] @ Ga
                Dg = gn[act]
                for k in range(m):
                    A_ = (lk[k] * Hs[k])[np.ix_(act, act)] * Dg[:, None] * Dg[None, :]
                    vp, f1 = sdp_box(A_)
                    vm, f2 = sdp_box(-A_)
                    pp += 0.5 * rho ** 2 * vp
                    pm += 0.5 * rho ** 2 * vm
                    sp += 0.5 * rho ** 2 * sign_max(A_)
                    sm += 0.5 * rho ** 2 * sign_max(-A_)
                    flags |= {f1, f2}
                na = max(na, int(act.sum()))
            rec.update(Q=Q, Pp=pp, Pm=pm, Pcp=pp, Pcm=pm, sign_p=sp, sign_m=sm, max_inputs=na, sdp_flags=sorted(flags),
                       n_active=na)
        res.append(rec)
    return res


def hessian_t(Bk, X: np.ndarray, pert: list[int], C: np.ndarray):
    """``H_0 = grad^2 psi(0)`` with ``psi(delta) = C . zp(X + delta)`` by forward-over-reverse autodiff through
    :meth:`aimp.block.GPT2Block.t_forward` (an implementation independent of the tape; symmetrized, the relative asymmetry
    is returned), and the gradient function ``delta -> grad psi`` for the finite-difference check."""
    import torch
    from torch.func import grad, jacfwd

    T, d = X.shape
    Xt = torch.as_tensor(X, dtype=torch.float64)
    Ct = torch.as_tensor(C, dtype=torch.float64).reshape(T, d)
    idx = torch.tensor(pert)
    K = len(pert)

    def psi(Dv):
        full = torch.zeros(T, d, dtype=torch.float64).index_copy(0, idx, Dv.reshape(K, d))
        return (Bk.t_forward(torch, Xt + full) * Ct).sum()

    g = grad(psi)
    Hm = jacfwd(g)(torch.zeros(K * d, dtype=torch.float64)).numpy()
    asym = float(np.max(np.abs(Hm - Hm.T))) / max(1e-300, float(np.max(np.abs(Hm))))
    return 0.5 * (Hm + Hm.T), asym, (lambda v: g(torch.as_tensor(v, dtype=torch.float64)).numpy())


def fd_check(H: np.ndarray, gfun, xnorm: float, rng, n_vec: int = 5, steps=(1e-4, 1e-5, 1e-6)) -> dict:
    """``H v`` against central differences ``(grad psi(eps v) - grad psi(-eps v)) / (2 eps)`` on ``n_vec`` random unit
    vectors, ``eps = step x ||x_k||``; relative error ``||FD - H v|| / ||H v||``, best step per vector (all recorded)."""
    errs = []
    for _ in range(n_vec):
        v = rng.standard_normal(H.shape[0])
        v /= np.linalg.norm(v)
        Hv = H @ v
        row = {}
        for s in steps:
            eps = s * xnorm
            fd = (gfun(eps * v) - gfun(-eps * v)) / (2 * eps)
            row[s] = float(np.linalg.norm(fd - Hv) / max(1e-300, np.linalg.norm(Hv)))
        errs.append(row)
    best = [min(r.values()) for r in errs]
    return {"errors": [{f"{k:g}": v for k, v in r.items()} for r in errs], "best": best, "max_best": float(max(best))}


def spectral(H: np.ndarray, rho: float) -> dict:
    """``s_spec^± = (1/2) rho^2 lambda_max(±H)_+`` (Lemma "chain"), the traces of the positive / negative parts (and
    ``(1/2) rho^2`` times them), the effective rank ``tr(H^+) / lambda_max(H)`` and the extreme eigenvectors."""
    w, V = np.linalg.eigh(H)
    tp, tm = float(np.sum(np.maximum(w, 0))), float(np.sum(np.maximum(-w, 0)))
    lp, lm = max(float(w[-1]), 0.0), max(float(-w[0]), 0.0)
    return {"eig": w, "v_max": V[:, -1], "v_min": V[:, 0], "lam_max": float(w[-1]), "lam_min": float(w[0]),
            "s_spec_p": 0.5 * rho ** 2 * lp, "s_spec_m": 0.5 * rho ** 2 * lm, "trace_p": tp, "trace_m": tm,
            "half_trace_p": 0.5 * rho ** 2 * tp, "half_trace_m": 0.5 * rho ** 2 * tm,
            "eff_rank_p": tp / lp if lp > 0 else None, "eff_rank_m": tm / lm if lm > 0 else None}


def grouped(recs: list[dict], rho: float, key) -> dict:
    """``L^± = (1/2) rho^2 sum_G lambda_max(±Q_G)_+`` for the partition ``key(op record) -> group`` (an intermediate between
    ``s_spec`` and ``s_op``); per group its ``lambda_max(±Q_G)`` and its share of the per-operation sums."""
    groups: dict = {}
    for r in recs:
        g = key(r)
        e = groups.setdefault(g, {"Q": None, "Pp": 0.0, "Pm": 0.0, "Pcp": 0.0, "Pcm": 0.0})
        e["Q"] = r["Q"] if e["Q"] is None else e["Q"] + r["Q"]
        for k in ("Pp", "Pm", "Pcp", "Pcm"):
            e[k] += r[k]
    out, Lp, Lm = {}, 0.0, 0.0
    for g, e in groups.items():
        w = np.linalg.eigvalsh(e["Q"])
        lp, lm = max(float(w[-1]), 0.0), max(float(-w[0]), 0.0)
        Lp += 0.5 * rho ** 2 * lp
        Lm += 0.5 * rho ** 2 * lm
        out[g] = {"lam_max": float(w[-1]), "lam_min": float(w[0]), "L_p": 0.5 * rho ** 2 * lp, "L_m": 0.5 * rho ** 2 * lm,
                  "P_p": e["Pp"], "P_m": e["Pm"], "Pc_p": e["Pcp"], "Pc_m": e["Pcm"]}
    return {"L_p": Lp, "L_m": Lm, "groups": out}


def structure_check(bm, Bk, X: np.ndarray, pert: list[int], rho: np.ndarray, seed: int = 0) -> dict:
    """The tape (``softmax = "lse"``) against auto_LiRPA's graph of :func:`aimp.block.remainder_net` (``bm``): every
    nonlinear node with perturbed inputs (``Mul`` with both inputs perturbed, ``Sqr``, ``Sqrt``, ``Reciprocal``, ``Tanh``,
    ``Softmax``, ``Exp``) is matched to exactly one tape operation of the same kind with the same forward value at a random
    input (``1e-12`` relative), and every tape operation to a node. ``pairs``: ``{tape op: node name}``."""
    import torch

    d = X.shape[1]
    rng = np.random.default_rng(seed)
    us = []
    for k in pert:
        x = rng.standard_normal(d)
        us.append(rho[k] * 0.7 * x / np.linalg.norm(x))
    with torch.no_grad():
        bm(*[torch.as_tensor(u[None], dtype=torch.float64) for u in us])
        _, ops = tape(Bk, X, pert, torch.as_tensor(np.concatenate(us), dtype=torch.float64))
    nonlin = []
    for nd in bm.nodes():
        cls = type(nd).__name__
        if cls not in set(BOUND_CLASS.values()):
            continue
        ins = [getattr(i, "perturbed", False) for i in nd.inputs]
        if (cls == "BoundMul" and not all(ins)) or (cls != "BoundMul" and not any(ins)):
            continue
        nonlin.append(nd)
    used, pairs, missing = set(), {}, []
    for o in ops:
        cls = BOUND_CLASS[o["fn"] if o["kind"] == "unary" else o["kind"]]
        ref = o["out"].reshape(-1)
        sc = 1e-12 * max(1.0, float(ref.abs().max()))
        hit = [nd for nd in nonlin if type(nd).__name__ == cls and nd.name not in used
               and nd.forward_value.numel() == ref.numel()
               and float((nd.forward_value.reshape(-1) - ref).abs().max()) <= sc]
        if hit:
            used.add(hit[0].name)
            pairs[o["name"]] = hit[0].name
        else:
            missing.append(o["name"])
    extra = [nd.name for nd in nonlin if nd.name not in used]
    counts: dict = {}
    for nd in nonlin:
        counts[type(nd).__name__] = counts.get(type(nd).__name__, 0) + 1
    return {"ok": not missing and not extra, "n_ops": len(ops), "n_nodes": len(nonlin), "node_counts": counts,
            "missing": missing, "extra_nodes": extra, "pairs": pairs}


def crown_halfwidths(bm, Bk, X: np.ndarray, pert: list[int], rho: np.ndarray, pairs: dict, seed: int = 0) -> dict:
    """Half-widths ``(u - l) / 2`` of CROWN's intermediate intervals on every operand of every tape operation: ``bm`` after a
    CROWN pass (its nodes carry the intervals CROWN relaxed each operation on), ``pairs`` = :func:`structure_check`'s
    ``{tape op: node}``. The node's inputs are matched to the tape operands by value at a random input (the order of the
    inputs of a product need not follow the tape). Returns ``{op name: tuple of half-width arrays in the operands'
    shapes}``."""
    import torch

    d = X.shape[1]
    rng = np.random.default_rng(seed)
    us = []
    for k in pert:
        x = rng.standard_normal(d)
        us.append(rho[k] * 0.6 * x / np.linalg.norm(x))
    lb = {nd.name: (nd.lower, nd.upper) for nd in bm.nodes() if getattr(nd, "lower", None) is not None}
    with torch.no_grad():
        bm(*[torch.as_tensor(u[None], dtype=torch.float64) for u in us])
        _, ops = tape(Bk, X, pert, torch.as_tensor(np.concatenate(us), dtype=torch.float64))
    out = {}
    for o in ops:
        nd = bm[pairs[o["name"]]]
        hw = []
        for a in o["args"]:
            ref = a.reshape(-1)
            got = None
            for inp in nd.inputs:
                fv = getattr(inp, "forward_value", None)
                if (isinstance(fv, torch.Tensor) and fv.numel() == ref.numel()
                        and float((fv.reshape(-1) - ref).abs().max()) <= 1e-12 * max(1.0, float(ref.abs().max()))):
                    lo, hi = lb.get(inp.name, (None, None))
                    if lo is None:
                        lo, hi = inp.lower, inp.upper
                    got = (0.5 * (hi - lo)).detach().numpy().reshape(tuple(a.shape))
                    break
            if got is None:
                raise RuntimeError(f"crown_halfwidths: no input of {nd.name} matches an operand of {o['name']}")
            hw.append(got)
        out[o["name"]] = tuple(hw)
    return out


def p_t_widths(jops: list[dict], lam: dict, hw: dict) -> dict:
    """The per-operation terms with given operand half-widths ``h`` (CROWN's intervals, :func:`crown_halfwidths`) in place of
    ``rho ||g||`` — the terms of the lower bound (eq. (Tlower)) of Proposition T(iii) on CROWN's own operations and
    intervals, with the adjoints as multipliers: unary ``(1/2) (lam kappa)_± h^2``, product ``|lam| h_x h_y`` (broadcast as
    the operation), softmax output ``k`` ``(1/2) sdp_box(± D K_k D)`` with ``D = diag(h_s)``; also the variant with CROWN's
    square relaxation (``|lam kappa| / 2`` for ``Sqr``). Per operation and in total."""
    import torch

    per, Pp, Pm, Pcp, Pcm = {}, 0.0, 0.0, 0.0, 0.0
    for o in jops:
        lm = np.asarray(lam[o["name"]], float)
        h = hw[o["name"]]
        if o["kind"] == "unary":
            x = o["args"][0].numpy().reshape(-1)
            w = lm.reshape(-1) * KAPPA[o["fn"]](x)
            h2 = h[0].reshape(-1) ** 2
            pp, pm = 0.5 * float(np.sum(np.maximum(w, 0) * h2)), 0.5 * float(np.sum(np.maximum(-w, 0) * h2))
            pc = 0.5 * float(np.sum(np.abs(w) * h2)) if o["fn"] == "sqr" else None
            pcp, pcm = (pc, pc) if pc is not None else (pp, pm)
        elif o["kind"] == "bilinear":
            shp = tuple(o["out"].shape)
            hx = torch.as_tensor(h[0]).expand(shp).reshape(-1).numpy()
            hy = torch.as_tensor(h[1]).expand(shp).reshape(-1).numpy()
            pp = pm = pcp = pcm = float(np.sum(np.abs(lm.reshape(-1)) * hx * hy))
        else:
            pbar = o["out"].numpy()[0]
            hs = h[0][0]  # (H, m)
            pp = pm = 0.0
            for hh in range(pbar.shape[0]):
                act = hs[hh] > 0
                if not act.any():
                    continue
                Hs = softmax_hessians(pbar[hh])
                Dg = hs[hh][act]
                for k in range(pbar.shape[1]):
                    A_ = (lm[0, hh, k] * Hs[k])[np.ix_(act, act)] * Dg[:, None] * Dg[None, :]
                    pp += 0.5 * sdp_box(A_)[0]
                    pm += 0.5 * sdp_box(-A_)[0]
            pcp, pcm = pp, pm
        per[o["name"]] = {"P_p": pp, "P_m": pm, "Pc_p": pcp, "Pc_m": pcm, "group": o["group"]}
        Pp, Pm, Pcp, Pcm = Pp + pp, Pm + pm, Pcp + pcp, Pcm + pcm
    return {"P_p": Pp, "P_m": Pm, "Pc_p": Pcp, "Pc_m": Pcm, "per_op": per}
