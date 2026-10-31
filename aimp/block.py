"""GPT-2 block maps for the mechanism test of Section 7.

A *configuration* of the mechanism test is a block ``l`` of GPT-2 small (``l`` in :data:`PAIRS`), the residual stream
``xbar`` at the input of block ``l`` on the first :data:`N_TOKENS` tokens of the first prompt, and a relative radius
``rho_rel``: token 1 is perturbed in the ball ``||delta|| <= rho_rel ||xbar_1||`` (``K_p = {1}``). A *unit* adds one of the
12 heads of block ``l + 1``, read at query position :data:`POS`.

* :class:`GPT2Block` — the block map ``F: x -> z+`` (block ``l`` followed by the ``ln_1`` of block ``l + 1``: ``LN1 -> 12
  heads -> + x -> LN2 -> GELU-tanh MLP -> + x' -> LN1^{(l+1)}``) in numpy and torch (float64), and its Jacobian ``J`` at
  ``xbar``. :func:`load_block` checks it against the hooked model (``1e-10``).
* :func:`remainder_net` — the remainder ``g(delta) = F(xbar + delta) - F(xbar) - J delta`` (eq. (blockrem) of Section 6)
  as a float64 ``torch.nn.Module`` for auto_LiRPA; :func:`bounded` wraps it with the l2 ball, :func:`crown_rows` runs CROWN
  on rows ``C g``.
* :func:`total_rows` — the unit's *total row* ``C``: the gradient of the head's nominal output specification with respect
  to the coordinates it reads (its queries, keys and values), mapped back to the normalized tokens. The function of the
  mechanism test is ``f(delta) = C . g(delta)``; it vanishes with its gradient at ``delta = 0``.
* Attacks (lower bounds on ``max f`` / upper bounds on ``min f``): :func:`pgd_rows` (batched projected gradient ascent),
  :func:`attacked_remainder` (L-BFGS-B on ``||g_j||``, whose maximizer is one start of the next attack) and
  :func:`attack_total_rows` (multistart ascent on ``± f`` plus random samples).

HF ``Conv1D`` convention ``y = x W + b`` throughout.
"""
from __future__ import annotations

import math
import os

import numpy as np

from aimp.io import cache_dir

D, H, DH = 768, 12, 64
PROMPT = 0  # the prompt of the mechanism test (first line of data/prompts.txt)
POS = 4  # query position of the downstream head
N_TOKENS = 5  # tokens 0..4 (causal: z+_0..4 depend only on x_0..4)
PAIRS = (0, 3, 6, 9)  # block l and the heads of block l + 1
PERTURBED = (1,)  # K_p = {1}
GELU_K = math.sqrt(2.0 / math.pi)
GELU_A = 0.044715
#: auto_LiRPA batches the CROWN backward passes of large intermediate nodes in chunks of this many neurons (memory only;
#: the bounds do not depend on it)
CROWN_BATCH = 512
GATE_TOL = 1e-10
_BLOCK_KEYS = ("ln1_g", "ln1_b", "ln1_eps", "W_attn", "b_attn", "W_proj", "b_proj", "ln2_g", "ln2_b", "ln2_eps", "W_fc",
               "b_fc", "W_mp", "b_mp")


# ================================================================================================ weights and the block map
def _block_file(l: int):
    return cache_dir() / f"block{l:02d}.npz"


def cache_blocks(layers) -> None:
    """Cache the full float64 weights (``ln_1``, attention, ``ln_2``, MLP; :func:`aimp.gpt2.layer_weights`) of the blocks
    ``layers`` as ``cache_dir()/block{l}.npz`` (downloads ``gpt2`` from the Hugging Face hub once)."""
    missing = [l for l in layers if not _block_file(l).exists()]
    if not missing:
        return
    from aimp import gpt2

    model, _ = gpt2.load_model()
    cache_dir().mkdir(parents=True, exist_ok=True)
    for l in missing:
        W = gpt2.layer_weights(model, l)
        f = _block_file(l)
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez(tmp, **{k: np.asarray(W[k]) for k in _BLOCK_KEYS})
        os.replace(tmp, f)


def block_weights(l: int) -> dict:
    """The full weights of block ``l`` (cached by :func:`cache_blocks`)."""
    cache_blocks([l])
    w = np.load(_block_file(l))
    return {k: (float(w[k]) if k.endswith("eps") else w[k].copy()) for k in _BLOCK_KEYS}


def ln_jacobian(x: np.ndarray, g: np.ndarray, eps: float) -> np.ndarray:
    """``Df(x)`` of the layer normalization ``f(x) = g * sqrt(d) P x / sqrt(||P x||^2 + d eps) + b`` (``d x d``)."""
    d = x.shape[0]
    Px = x - x.mean()
    ne = math.sqrt(float(Px @ Px) + d * eps)
    u = Px / ne
    C = np.eye(d) - np.ones((d, d)) / d
    return (g[:, None] * (math.sqrt(d) / ne) * (np.eye(d) - np.outer(u, u))) @ C


def gelu(x):
    """GELU-tanh ``(x / 2)(1 + tanh(k (x + a x^3)))``."""
    return 0.5 * x * (1.0 + np.tanh(GELU_K * (x + GELU_A * x ** 3)))


def gelu_d1(x):
    u = GELU_K * (x + GELU_A * x ** 3)
    T = np.tanh(u)
    return 0.5 * (1 + T) + 0.5 * x * (1 - T * T) * GELU_K * (1 + 3 * GELU_A * x * x)


class GPT2Block:
    """Block ``l`` of GPT-2 small in float64 followed by the ``ln_1`` of block ``l + 1``: ``x -> z = LN1(x) -> A(z)`` (12
    heads and the ``c_proj`` bias) ``-> x' = x + A -> u = LN2(x') -> m = gelu(u W_fc + b_fc) W_mp + b_mp -> x'' = x' + m ->
    z+ = LN1^{(l+1)}(x'')`` on the first ``T`` tokens (causal). ``W``: :func:`block_weights`; ``W_next``: at least the
    ``ln1_*`` entries of block ``l + 1``."""

    def __init__(self, W: dict, W_next: dict, d_head: int = DH, n_heads: int = H):
        self.W = W
        self.g1, self.b1, self.e1 = W["ln1_g"], W["ln1_b"], float(W["ln1_eps"])
        self.g2, self.b2, self.e2 = W["ln2_g"], W["ln2_b"], float(W["ln2_eps"])
        self.gn, self.bn, self.en = W_next["ln1_g"], W_next["ln1_b"], float(W_next["ln1_eps"])
        self.d = self.g1.shape[0]
        self.dh, self.H = d_head, n_heads
        d = self.d
        Wa, ba, Wp = W["W_attn"], W["b_attn"], W["W_proj"]
        self.heads = []
        for h in range(n_heads):
            sl = slice(h * d_head, (h + 1) * d_head)
            Wv, bv, Wo = Wa[:, 2 * d:][:, sl], ba[2 * d:][sl], Wp[sl, :]
            self.heads.append({"Wq": Wa[:, sl], "bq": ba[sl], "Wk": Wa[:, d:][:, sl], "bk": ba[d:][sl],
                               "Wvo": Wv @ Wo, "bvo": bv @ Wo, "scale": 1.0 / math.sqrt(d_head)})
        self.W_fc, self.b_fc, self.W_mp, self.b_mp = W["W_fc"], W["b_fc"], W["W_mp"], W["b_mp"]

    # -------------------------------------------------------------------------------------------- numpy forward
    @staticmethod
    def _ln(x, g, b, eps):
        mu = x.mean(axis=-1, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
        return g * (x - mu) / np.sqrt(var + eps) + b

    def attn(self, z: np.ndarray) -> np.ndarray:
        T = z.shape[-2]
        out = np.zeros(z.shape[:-1] + (self.d,))
        mask = np.tril(np.ones((T, T), bool))
        for hd in self.heads:
            q = z @ hd["Wq"] + hd["bq"]
            k = z @ hd["Wk"] + hd["bk"]
            s = hd["scale"] * np.einsum("...ia,...ja->...ij", q, k)
            s = np.where(mask, s, -np.inf)
            s = s - s.max(axis=-1, keepdims=True)
            a = np.exp(s)
            a /= a.sum(axis=-1, keepdims=True)
            out = out + a @ (z @ hd["Wvo"] + hd["bvo"])
        return out + self.W["b_proj"]

    def forward(self, X: np.ndarray) -> dict:
        """Stages at ``X`` (``(..., T, d)``): ``z, x1, u, pre, x2, zp`` (``zp = z+``)."""
        z = self._ln(X, self.g1, self.b1, self.e1)
        x1 = X + self.attn(z)
        u = self._ln(x1, self.g2, self.b2, self.e2)
        pre = u @ self.W_fc + self.b_fc
        x2 = x1 + gelu(pre) @ self.W_mp + self.b_mp
        zp = self._ln(x2, self.gn, self.bn, self.en)
        return {"z": z, "x1": x1, "u": u, "pre": pre, "x2": x2, "zp": zp}

    # -------------------------------------------------------------------------------------------- torch forward
    def torch_params(self, torch):
        if getattr(self, "_tp", None) is None:
            T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)  # noqa: E731
            W = self.W
            self._tp = {k: T(v) for k, v in (("g1", self.g1), ("b1", self.b1), ("g2", self.g2), ("b2", self.b2),
                                             ("gn", self.gn), ("bn", self.bn), ("W_attn", W["W_attn"]),
                                             ("b_attn", W["b_attn"]), ("W_proj", W["W_proj"]), ("b_proj", W["b_proj"]),
                                             ("W_fc", self.W_fc), ("b_fc", self.b_fc), ("W_mp", self.W_mp),
                                             ("b_mp", self.b_mp))}
        return self._tp

    def t_attn(self, torch, z):
        p = self.torch_params(torch)
        d, dh, H_ = self.d, self.dh, self.H
        T = z.shape[-2]
        qkv = z @ p["W_attn"] + p["b_attn"]
        q, k, v = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]
        sh = z.shape[:-1] + (H_, dh)
        q, k, v = q.reshape(sh).transpose(-2, -3), k.reshape(sh).transpose(-2, -3), v.reshape(sh).transpose(-2, -3)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        a = torch.softmax(s.masked_fill(~mask, float("-inf")), dim=-1)
        o = (a @ v).transpose(-2, -3).reshape(z.shape[:-1] + (d,))
        return o @ p["W_proj"] + p["b_proj"]

    def t_forward(self, torch, X):
        """``z+`` at ``X`` (torch, ``(..., T, d)``); an implementation independent of the operation tape of
        :func:`aimp.mechanism.tape` (``F.layer_norm``, batched attention)."""
        import torch.nn.functional as F

        p = self.torch_params(torch)
        d = self.d
        z = F.layer_norm(X, (d,), p["g1"], p["b1"], self.e1)
        x1 = X + self.t_attn(torch, z)
        u = F.layer_norm(x1, (d,), p["g2"], p["b2"], self.e2)
        pre = u @ p["W_fc"] + p["b_fc"]
        g = 0.5 * pre * (1.0 + torch.tanh(GELU_K * (pre + GELU_A * pre ** 3)))
        x2 = x1 + g @ p["W_mp"] + p["b_mp"]
        return F.layer_norm(x2, (d,), p["gn"], p["bn"], self.en)

    # -------------------------------------------------------------------------------------------- Jacobian
    def jacobians(self, X: np.ndarray, check: bool = True) -> dict:
        """The Jacobian ``J`` (``(T, T, d, d)``, block ``[j, k] = d z+_j / d x_k``) at ``X`` composed from the stage
        Jacobians (layer normalizations in closed form, attention by autograd, the MLP from ``gelu'``), and the nominal
        stages ``f``. ``check``: ``J`` against autograd of the whole map (relative deviation) and the causal blocks."""
        import torch

        f = self.forward(X)
        T, d = X.shape
        J_ln1 = np.stack([ln_jacobian(X[j], self.g1, self.e1) for j in range(T)])
        zt = torch.as_tensor(f["z"], dtype=torch.float64)
        dA = torch.autograd.functional.jacobian(lambda z: self.t_attn(torch, z), zt, vectorize=True)
        dA = dA.permute(0, 2, 1, 3).numpy().copy()  # (j, k, a, b)
        J_x1 = np.einsum("jkab,kbc->jkac", dA, J_ln1)
        for j in range(T):
            J_x1[j, j] += np.eye(d)
        J_ln2 = np.stack([ln_jacobian(f["x1"][j], self.g2, self.e2) for j in range(T)])
        dpsi = gelu_d1(f["pre"])
        Dm = np.stack([(self.W_mp.T * dpsi[j]) @ self.W_fc.T for j in range(T)])
        J_u = np.einsum("jab,jkbc->jkac", J_ln2, J_x1)
        J_x2 = J_x1 + np.einsum("jab,jkbc->jkac", Dm, J_u)
        J_lnn = np.stack([ln_jacobian(f["x2"][j], self.gn, self.en) for j in range(T)])
        J = np.einsum("jab,jkbc->jkac", J_lnn, J_x2)
        out = {"f": f, "J": J}
        if check:
            Xt = torch.as_tensor(X, dtype=torch.float64)
            Ja = torch.autograd.functional.jacobian(lambda x: self.t_forward(torch, x), Xt, vectorize=True)
            Ja = Ja.permute(0, 2, 1, 3).numpy()
            out["J_check_rel"] = float(np.max(np.abs(Ja - J))) / max(1e-300, float(np.max(np.abs(Ja))))
            out["J_causal_max"] = float(max((np.max(np.abs(Ja[j, k])) for j in range(T) for k in range(j + 1, T)),
                                            default=0.0))
        return out


_BLOCKS: dict = {}


def load_block(l: int) -> dict:
    """Block pair ``(l, l + 1)`` of the mechanism test (cached per process): ``B`` (:class:`GPT2Block`), ``X`` (``xbar``,
    the input of block ``l`` on tokens 0..4 of the first prompt), ``J``, ``zbar = F(xbar)``, and the gate: the numpy block
    against the hooked model (block output and ``ln_1`` of block ``l + 1``, ``1e-10`` relative), ``J`` against autograd
    and the causal blocks ``J_jk = 0`` for ``k > j``."""
    if l in _BLOCKS:
        return _BLOCKS[l]
    from aimp import gpt2_units

    Bk = GPT2Block(block_weights(l), gpt2_units.ctx()["W"][l + 1])
    P = np.load(cache_dir() / f"prompt{PROMPT}.npz")
    X = P["x"][l][:N_TOKENS].copy()
    jac = Bk.jacobians(X, check=True)
    f = jac["f"]

    def rel(a, b):
        return float(np.max(np.abs(a - b))) / max(1.0, float(np.max(np.abs(b))))

    gate = {"block_rel": rel(f["x2"], P["x"][l + 1][:N_TOKENS]), "ln_rel": rel(f["zp"], P["z"][l + 1][:N_TOKENS]),
            "J_check_rel": jac["J_check_rel"], "J_causal_max": jac["J_causal_max"]}
    gate["ok"] = bool(gate["block_rel"] <= GATE_TOL and gate["ln_rel"] <= GATE_TOL and gate["J_check_rel"] <= GATE_TOL
                      and gate["J_causal_max"] <= 1e-12)
    _BLOCKS.clear()
    _BLOCKS[l] = {"B": Bk, "X": X, "J": jac["J"], "zbar": f["zp"], "gate": gate}
    return _BLOCKS[l]


def radii(X: np.ndarray, rho_rel: float, pert=PERTURBED) -> np.ndarray:
    """Per-token radii ``rho_k = rho_rel ||xbar_k||`` for ``k`` in ``pert``, 0 elsewhere."""
    r = np.zeros(X.shape[0])
    for k in pert:
        r[k] = rho_rel * float(np.linalg.norm(X[k]))
    return r


def influenced(J: np.ndarray, pert) -> list[int]:
    """Output tokens whose ``z+`` depends on a perturbed token (nonzero Jacobian block)."""
    return [j for j in range(J.shape[0]) if any(np.any(J[j, k]) for k in pert)]


# ================================================================================================ remainder network
def remainder_net(Bk: GPT2Block, X: np.ndarray, pert: list[int], J: np.ndarray, zbar: np.ndarray):
    """The remainder network ``g(delta) = F(xbar + delta) - F(xbar) - sum_k J_k delta_k`` (``(1, T d)``) as a float64
    ``torch.nn.Module`` with one input ``delta_k`` (``(1, d)``) per perturbed token. The layer normalization is written out
    (mean, centring, the square ``c * c``, ``sqrt``, reciprocal, the product ``c * rsqrt``), the 12 heads with an explicit
    softmax per query row (``q * k``, ``softmax``, ``a * v``), GELU-tanh as ``(x / 2)(1 + tanh(k (x + a x x x)))``: the
    elementary operations auto_LiRPA traces are those of :func:`aimp.mechanism.tape` (checked by
    :func:`aimp.mechanism.structure_check`). Concatenations are along non-batch axes (auto_LiRPA requires it)."""
    import torch

    T, d = X.shape
    H_, dh = Bk.H, Bk.dh
    T64 = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64)  # noqa: E731

    class Remainder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for k, v in Bk.torch_params(torch).items():
                self.register_buffer(k, v.clone())
            self.register_buffer("Xb", T64(X))
            self.register_buffer("zb", T64(zbar).reshape(1, -1))
            for k in pert:
                self.register_buffer(f"J{k}", T64(J[:, k].reshape(T * d, d)).T.contiguous())  # (d, T d)
            self.eps = (Bk.e1, Bk.e2, Bk.en)
            self.pert = list(pert)

        @staticmethod
        def ln(x, g, b, eps):
            mu = x.mean(-1, keepdim=True)
            c = x - mu
            var = (c * c).mean(-1, keepdim=True)
            rs = torch.reciprocal(torch.sqrt(var + eps))
            return c * rs * g + b

        def forward(self, *us):
            rows = []
            for k in range(T):
                xk = self.Xb[k:k + 1].unsqueeze(0)  # (1, 1, d)
                if k in self.pert:
                    xk = xk + us[self.pert.index(k)].unsqueeze(1)
                rows.append(xk)
            Xi = torch.cat(rows, 1)  # (1, T, d)
            z = self.ln(Xi, self.g1, self.b1, self.eps[0])
            qkv = z @ self.W_attn + self.b_attn
            q, k_, v = qkv[:, :, :d], qkv[:, :, d:2 * d], qkv[:, :, 2 * d:]
            q = q.reshape(1, T, H_, dh).transpose(1, 2)
            k_ = k_.reshape(1, T, H_, dh).transpose(1, 2)
            v = v.reshape(1, T, H_, dh).transpose(1, 2)  # (1, H, T, dh)
            outs = []
            for j in range(T):
                s = (q[:, :, j:j + 1, :] * k_[:, :, :j + 1, :]).sum(-1) / math.sqrt(dh)  # (1, H, j + 1)
                a = torch.softmax(s, dim=-1)
                o = (a.unsqueeze(-1) * v[:, :, :j + 1, :]).sum(2)  # (1, H, dh)
                outs.append(o.reshape(1, 1, H_ * dh))
            att = torch.cat(outs, 1) @ self.W_proj + self.b_proj
            x1 = Xi + att
            u = self.ln(x1, self.g2, self.b2, self.eps[1])
            pre = u @ self.W_fc + self.b_fc
            gl = 0.5 * pre * (1.0 + torch.tanh(GELU_K * (pre + GELU_A * pre * pre * pre)))
            x2 = x1 + gl @ self.W_mp + self.b_mp
            zp = self.ln(x2, self.gn, self.bn, self.eps[2])
            out = zp.reshape(1, T * d) - self.zb
            for a_, k in enumerate(self.pert):
                out = out - us[a_] @ getattr(self, f"J{k}")
            return out

    return Remainder().double().eval()


def bounded(mod, rho: np.ndarray, pert: list[int], crown_batch: int = CROWN_BATCH):
    """``(BoundedModule, inputs)`` for ``mod`` with an l2 ball of radius ``rho[k]`` on each perturbed input."""
    import torch
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    d = mod.Xb.shape[1]
    x0 = tuple(torch.zeros(1, d, dtype=torch.float64) for _ in pert)
    opts = {"crown_batch_size": int(crown_batch), "batched_crown_max_vram_ratio": 0.9}
    bm = BoundedModule(mod, x0, device="cpu", verbose=0, bound_opts=opts)
    bx = tuple(BoundedTensor(x0[a], PerturbationLpNorm(norm=2, eps=float(rho[k]))) for a, k in enumerate(pert))
    return bm, bx


def crown_rows(bm, bx, C: np.ndarray, pert: list[int], rho: np.ndarray, ib: dict | None = None,
               method: str = "CROWN") -> dict:
    """CROWN bounds of ``C g`` (``C``: ``(m, T d)``) with the linear coefficients w.r.t. every perturbed input
    (``return_A``): ``Lam^L, Lam^U`` (``(m, |K_p|, d)``), ``beta^L, beta^U`` and the concretizations ``l = beta^L -
    sum_k rho_k ||Lam^L_k||``, ``v = beta^U + sum_k rho_k ||Lam^U_k||`` (the inputs are centred at 0), so that ``l <= C g(delta)
    <= v`` on the balls. ``ib``: intermediate bounds to reuse (``None``: computed in this pass)."""
    import torch

    out_name = bm.output_name[0]
    inputs = [nd.name for nd in bm.nodes() if type(nd).__name__ == "BoundInput"]  # forward-argument order
    if len(inputs) != len(pert):
        raise RuntimeError(f"crown_rows: {len(inputs)} perturbed inputs, expected {len(pert)}")
    Ct = torch.as_tensor(np.asarray(C, float)[None], dtype=torch.float64)
    lb, ub, A = bm.compute_bounds(x=bx, method=method, C=Ct, interm_bounds=ib, return_A=True,
                                  needed_A_dict={out_name: inputs})
    m, d = C.shape[0], int(np.prod(bm[inputs[0]].output_shape[1:]))
    LL = np.zeros((m, len(pert), d))
    LU = np.zeros((m, len(pert), d))
    for a, nm in enumerate(inputs):
        LL[:, a] = A[out_name][nm]["lA"].detach().numpy().reshape(m, d)
        LU[:, a] = A[out_name][nm]["uA"].detach().numpy().reshape(m, d)
    lb = lb.detach().numpy().reshape(m).copy()
    ub = ub.detach().numpy().reshape(m).copy()
    rk = np.array([rho[k] for k in pert])
    cL = -rk[None] * np.linalg.norm(LL, axis=2)  # concretized contribution of each input: (m, |K_p|)
    cU = rk[None] * np.linalg.norm(LU, axis=2)
    bL, bU = lb - cL.sum(axis=1), ub - cU.sum(axis=1)
    return {"LamL": LL, "LamU": LU, "betaL": bL, "betaU": bU, "lb": lb, "ub": ub, "l": bL + cL.sum(axis=1),
            "v": bU + cU.sum(axis=1)}


def numpy_remainder(Bk: GPT2Block, X: np.ndarray, J: np.ndarray, zbar: np.ndarray, D: np.ndarray) -> np.ndarray:
    """``g(D)`` by the numpy forward pass of the block (``D``: ``(..., T, d)``)."""
    return Bk.forward(X + D)["zp"] - zbar - np.einsum("jkab,...kb->...ja", J, D)


def sample_deltas(rng, n: int, rho: np.ndarray, d: int) -> np.ndarray:
    """``n`` perturbations ``(n, T, d)``: half on the spheres of the balls, half uniform inside."""
    G = rng.standard_normal((n, rho.shape[0], d))
    G /= np.linalg.norm(G, axis=2, keepdims=True)
    rad = np.ones((n, rho.shape[0], 1))
    half = n // 2
    rad[half:] = rng.uniform(0, 1, (n - half, rho.shape[0], 1)) ** (1 / d)
    return G * rad * rho[None, :, None]


# ================================================================================================ the unit's total row
def read_matrix(j: int, wv: np.ndarray, pos: int = POS, dk: int = DH) -> np.ndarray:
    """``A_j``: the coordinates of the projected token ``zp_j = (q, k, v)`` (``3 d_k``) the head reads at query ``pos``: q
    and k of the query token, k of the keys, and the value direction ``wv = W_O c`` (``c^T N_p zp = wv . v``)."""
    I = np.eye(3 * dk)
    vrow = np.zeros((1, 3 * dk))
    vrow[0, 2 * dk:] = wv
    return np.vstack([I[:2 * dk] if j == pos else I[dk:2 * dk], vrow])


def total_rows(l: int, zbar: np.ndarray, infl: list[int], heads=range(H)) -> tuple[np.ndarray, float]:
    """The total row ``C`` (``(len(heads), T d)``) of every head ``h`` of block ``l + 1`` at query :data:`POS`: with the
    head's reduced coordinates ``zp_j = P zbar_j + b`` (:func:`aimp.gpt2.projection`) and its specification ``c`` (the
    direction of the nominal head output, :func:`aimp.gpt2_units.make_instance`), ``gam_j = (A_j^+)^T grad_{zp_j} phi``
    (``phi = c^T y``, :func:`aimp.heads.grad_phi`) and ``C`` on token ``j``'s block ``= P^T A_j^T gam_j`` for the influenced
    tokens ``j``. Returns ``C`` and the largest relative residual of ``A_j^T gam_j = grad_{zp_j} phi``."""
    from aimp import gpt2_units
    from aimp.heads import grad_phi
    from aimp.types import Instance

    T, d = zbar.shape
    rows = np.zeros((len(heads), T * d))
    resid = 0.0
    for a, h in enumerate(heads):
        hw, pr = gpt2_units.head_data(l + 1, h)
        inst = gpt2_units.make_instance(l + 1, h, PROMPT, POS, 0.0)
        Zc = zbar @ pr.P.T + pr.b
        _, G0 = grad_phi(Instance(head=inst.head, keys=list(range(T)), query=POS, sets={}, c=inst.c), Zc)
        wv = hw["W_O"] @ inst.c
        for j in range(T):
            A = read_matrix(j, wv)
            gam = np.linalg.pinv(A).T @ G0[j]
            resid = max(resid, float(np.max(np.abs(A.T @ gam - G0[j]))) / max(1.0, float(np.max(np.abs(G0[j])))))
            if j in infl:
                rows[a, j * d:(j + 1) * d] = pr.P.T @ (A.T @ gam)
    return rows, resid


def n_unit_rows(infl: list[int], T: int = N_TOKENS, pos: int = POS) -> int:
    """Number of rows per unit in the random-start stream of :func:`attack_total_rows`: the total row, a key-score row per
    influenced key, a query-score row per key (when the query token is influenced) and a value row per influenced token."""
    return 1 + len(infl) + (T if pos in infl else 0) + len(infl)


# ================================================================================================ attacks
def pgd_rows(Bk: GPT2Block, X: np.ndarray, J: np.ndarray, zbar: np.ndarray, rho: np.ndarray, pert: list[int],
             C: np.ndarray, sign: np.ndarray, D0: np.ndarray, steps: int, eta0: float = 0.3, eta1: float = 0.003,
             chunk: int = 1024) -> dict:
    """Projected gradient ascent of ``sign_b * C_b g(delta)`` over the product of balls for a batch of problems ``b``
    (``C``: ``(B, T d)``, ``sign``: ``(B,)``, starts ``D0``: ``(B, |K_p|, d)``): steps ``eta_t rho_k grad_k / ||grad_k||`` per
    perturbed token (``eta_t`` geometric from ``eta0`` to ``eta1``), radial projection; the best iterate per problem (values
    of the torch float64 forward pass; the caller re-evaluates in numpy). Batched in chunks of ``chunk`` problems."""
    import torch

    T, d = X.shape
    Xt = torch.as_tensor(X, dtype=torch.float64)
    zbt = torch.as_tensor(zbar, dtype=torch.float64)
    Jt = torch.as_tensor(np.ascontiguousarray(J[:, pert]), dtype=torch.float64)  # (T, |K|, d, d)
    rk = torch.as_tensor(np.array([rho[k] for k in pert]), dtype=torch.float64)[None, :, None]
    pidx = torch.tensor(pert)
    Bn = C.shape[0]
    best = np.full(Bn, -np.inf)
    bestD = np.array(D0, float, copy=True)
    etas = eta0 * (eta1 / eta0) ** (np.arange(steps) / max(1, steps - 1))
    for a in range(0, Bn, chunk):
        sl = slice(a, min(Bn, a + chunk))
        Ct = torch.as_tensor(C[sl], dtype=torch.float64)
        sg = torch.as_tensor(sign[sl], dtype=torch.float64)
        D_ = torch.as_tensor(D0[sl], dtype=torch.float64).clone()
        nb = D_.shape[0]
        bv = torch.full((nb,), -math.inf, dtype=torch.float64)
        bD = D_.clone()
        for t in range(steps + 1):
            D_.requires_grad_(True)
            full = torch.zeros(nb, T, d, dtype=torch.float64).index_copy(1, pidx, D_)
            zp = Bk.t_forward(torch, Xt + full)
            g = zp - zbt - torch.einsum("jkab,nkb->nja", Jt, D_)
            f = (g.reshape(nb, -1) * Ct).sum(-1) * sg
            with torch.no_grad():
                imp = f > bv
                bv = torch.where(imp, f, bv)
                bD[imp] = D_.detach()[imp]
            if t == steps:
                break
            (gr,) = torch.autograd.grad(f.sum(), D_)
            with torch.no_grad():
                gn = gr.norm(dim=2, keepdim=True).clamp_min(1e-300)
                Dn = D_.detach() + etas[t] * rk * gr / gn
                nn_ = Dn.norm(dim=2, keepdim=True).clamp_min(1e-300)
                D_ = Dn * torch.clamp(rk / nn_, max=1.0)
        best[sl] = bv.numpy()
        bestD[sl] = bD.numpy()
    return {"value": best, "delta": bestD}


def lam_starts(Lam: np.ndarray, rho: np.ndarray, pert: list[int]) -> np.ndarray:
    """``delta_k = rho_k Lam_k / ||Lam_k||`` (the maximizer of the linear bound ``Lam delta``) per row: ``(m, |K_p|, d)``."""
    rk = np.array([rho[k] for k in pert])[None, :, None]
    n = np.linalg.norm(Lam, axis=2, keepdims=True)
    return np.where(n > 0, rk * Lam / np.maximum(n, 1e-300), 0.0)


def random_starts(rng, m: int, rho: np.ndarray, pert: list[int], d: int) -> np.ndarray:
    """``m`` points uniform on the product of spheres: ``(m, |K_p|, d)``."""
    x = rng.standard_normal((m, len(pert), d))
    rk = np.array([rho[k] for k in pert])[None, :, None]
    return rk * x / np.linalg.norm(x, axis=2, keepdims=True)


def attacked_remainder(Bk: GPT2Block, X: np.ndarray, J: np.ndarray, zbar: np.ndarray, rho: np.ndarray, j: int,
                       seed: int, n_samples: int = 10_000, n_best: int = 20, n_random: int = 10, maxiter: int = 500,
                       chunk: int = 1000) -> dict:
    """A maximizer of ``||g_j(delta)||`` (the remainder of output token ``j``) over the balls: ``n_samples`` random
    perturbations (:func:`sample_deltas`, ``rng(seed)``), then L-BFGS-B on the unconstrained ``p`` with the radial projection
    ``delta_k = p_k min(1, rho_k / ||p_k||)`` from the ``n_best`` best samples and ``n_random`` random sphere points
    (``rng(seed + 1 + j)``), ``maxiter`` iterations each; the value is re-evaluated in numpy at the projected point. Returns
    the best value, its perturbation ``(T, d)`` and the best sampled value."""
    import torch
    from scipy.optimize import minimize

    T, d = X.shape
    free = [k for k in range(T) if rho[k] > 0]
    rng = np.random.default_rng(seed)
    Dl = sample_deltas(rng, n_samples, rho, d)
    r = np.zeros(n_samples)
    for a in range(0, n_samples, chunk):
        Dc = Dl[a:a + chunk]
        Zp = Bk.forward(X[None] + Dc)["zp"]
        r[a:a + chunk] = np.linalg.norm(Zp[:, j] - zbar[j] - np.einsum("kab,skb->sa", J[j], Dc), axis=1)
    order = np.argsort(-r)[:n_best]
    Xt = torch.as_tensor(X, dtype=torch.float64)
    Jt = torch.as_tensor(J[j], dtype=torch.float64)
    zbt = torch.as_tensor(zbar[j], dtype=torch.float64)
    rf = torch.as_tensor(np.array([rho[k] for k in free]), dtype=torch.float64)
    rfn = np.array([rho[k] for k in free])[:, None]
    idx = torch.tensor(free)

    def proj_np(p):
        P = p.reshape(len(free), d)
        Dp = np.zeros((T, d))
        Dp[free] = P * np.minimum(1.0, rfn / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-300))
        return Dp

    def fg(p):
        pt = torch.as_tensor(p.reshape(len(free), d), dtype=torch.float64).requires_grad_(True)
        nrm = torch.linalg.norm(pt, dim=1, keepdim=True)
        rows = pt * torch.clamp(rf[:, None] / torch.clamp(nrm, min=1e-300), max=1.0)
        Dt = torch.zeros(T, d, dtype=torch.float64).index_put((idx,), rows)
        v = torch.linalg.norm(Bk.t_forward(torch, Xt + Dt)[j] - zbt - torch.einsum("kab,kb->a", Jt, Dt))
        v.backward()
        return -float(v.detach()), -pt.grad.numpy().ravel().copy()

    def val_np(Dp):
        return float(np.linalg.norm(Bk.forward(X + Dp)["zp"][j] - zbar[j] - np.einsum("kab,kb->a", J[j], Dp)))

    rng2 = np.random.default_rng(seed + 1 + j)
    pts = [Dl[s][free].ravel() for s in order]
    for _ in range(n_random):
        P = rng2.standard_normal((len(free), d))
        P *= rfn / np.linalg.norm(P, axis=1, keepdims=True)
        pts.append(P.ravel())
    best, best_D, nits = -math.inf, None, []
    for p0 in pts:
        res = minimize(fg, p0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-14})
        Dp = proj_np(res.x)
        val = val_np(Dp)
        nits.append(int(res.nit))
        if val > best:
            best, best_D = val, Dp
    return {"value": float(best), "delta": best_D, "sampled": float(r.max()), "nit": nits}


def attack_total_rows(Bk: GPT2Block, X: np.ndarray, J: np.ndarray, zbar: np.ndarray, rho: np.ndarray, pert: list[int],
                      Cs: np.ndarray, LamU: np.ndarray, LamL: np.ndarray, d_start: np.ndarray, seed: int, heads,
                      n_rows: int, steps: int = 200, n_samples: int = 2000, chunk: int = 250) -> dict:
    """Attacked maximum and minimum of ``f_h = C_h . g`` for the units ``heads`` (``Cs``, ``LamU``, ``LamL``: the total rows
    and their CROWN coefficients ``(len(heads), |K_p|, d)``, :func:`crown_rows`): :func:`pgd_rows` (``steps``) on ``± f_h``
    from four starts per side (the maximizer of CROWN's linear bound, ``d_start`` = :func:`attacked_remainder`'s maximizer for
    the query token, two random sphere points), the best start re-evaluated in numpy; ``n_samples`` random perturbations
    (feasible points too) enter the maximum and minimum. All randomness comes from ``rng(seed)``: the samples first, then two
    random starts per (unit ``0..11``, row ``0..n_rows-1``, side) in that order, the total row being row 0 of each unit (the
    other rows of the stream are not attacked here)."""
    T, d = X.shape
    rng = np.random.default_rng(seed)
    Ds = sample_deltas(rng, n_samples, rho, d)
    Gs = np.concatenate([numpy_remainder(Bk, X, J, zbar, Ds[a:a + chunk]) for a in range(0, n_samples, chunk)])
    sv = Gs.reshape(n_samples, -1) @ Cs.T  # (n_samples, len(heads))
    rnd = {}
    for h in range(H):
        for a in range(n_rows):
            for s_ in (1.0, -1.0):
                R = random_starts(rng, 2, rho, pert, d)
                if a == 0:
                    rnd[(h, s_)] = R
    Cl, sg, D0, key = [], [], [], []
    for i, h in enumerate(heads):
        for s_ in (1.0, -1.0):
            st = [lam_starts((LamU if s_ > 0 else -LamL)[i:i + 1], rho, pert)[0], d_start[pert]] + list(rnd[(h, s_)])
            for D_ in st:
                Cl.append(Cs[i])
                sg.append(s_)
                D0.append(D_)
                key.append((i, s_))
    res = pgd_rows(Bk, X, J, zbar, rho, pert, np.array(Cl), np.array(sg), np.array(D0), steps)
    bb: dict = {}
    for n, k in enumerate(key):
        if k not in bb or res["value"][n] > res["value"][bb[k]]:
            bb[k] = n
    ks = sorted(bb)
    Dk = res["delta"][[bb[k] for k in ks]]
    Df = np.zeros((len(ks), T, d))
    Df[:, pert] = Dk
    vals = np.einsum("nc,nc->n", numpy_remainder(Bk, X, J, zbar, Df).reshape(len(ks), -1), Cs[[k[0] for k in ks]])
    v = {k: float(x) for k, x in zip(ks, vals)}
    out = {"max": [], "min": [], "max_pgd": [], "min_pgd": [], "max_samples": [], "min_samples": []}
    for i in range(len(heads)):
        out["max_pgd"].append(v[(i, 1.0)])
        out["min_pgd"].append(v[(i, -1.0)])
        out["max_samples"].append(float(sv[:, i].max()))
        out["min_samples"].append(float(sv[:, i].min()))
        out["max"].append(max(out["max_pgd"][-1], out["max_samples"][-1]))
        out["min"].append(min(out["min_pgd"][-1], out["min_samples"][-1]))
    return out
