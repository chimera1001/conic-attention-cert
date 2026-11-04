"""GPT-2 small: extraction with the 1e-10 reproduction gate, the exact reduction to the head subspace (Proposition 15
of the paper), a LayerNorm contraction factor, and head instances in the reduced coordinates.

Conventions (HF ``Conv1D``: ``y = x W + b`` with ``W`` of shape (in, out)). For block ``l`` and head ``h``
(``d = 768``, ``d_k = 64``): ``W_Q = c_attn.weight[:, h*64:(h+1)*64]``, ``W_K`` at offset 768, ``W_V`` at 1536 (each
768 x 64), biases likewise; ``W_O = c_proj.weight[h*64:(h+1)*64, :]`` (64 x 768). With post-LN inputs ``z_j`` (rows):
``q = z W_Q + b_Q``, ``s_j = q_i . k_j / 8``, ``a = softmax(s)`` over ``j <= i``, head output ``sum_j a_j (z_j W_V + b_V)``
(64; the input of ``c_proj``), contribution to the block ``sum_j a_j (z_j W_V + b_V) W_O`` (768). ``c_proj.bias`` is shared
by the 12 heads and is not part of any head.

**Projection (Proposition 15).** ``P = [W_Q^T; W_K^T; W_V^T]`` (192 x 768), ``b = [b_Q; b_K; b_V]``, thin SVD
``P = U S V^T``. The ball ``{zbar + u : ||u|| <= rho}`` maps to ``{P zbar + b + U S w : ||w|| <= rho}`` (``w = V^T u``).
In projected coordinates ``zp = (q, k, v)``: ``M_p = [[0, I/8, 0], 0, 0]`` (``s_j = zp_i^T M_p zp_j``) and
``N_p = [0, 0, W_O^T]`` (768 x 192). The biases sit in the ellipsoid centers, so the projected head is exact (no bias
term is dropped).

**Score intervals** (``shor``; Appendix B). The score of key ``j`` sees the query only through ``q_i`` and the key only
through ``k_j``; the Shor bound is computed on the role-reduced two-ellipsoid problem in ``R^64 x R^64`` (``R^128`` for
``j = i``), which has the same value as Shor on the full projected problem (tested).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aimp import sets as S
from aimp.io import ROOT
from aimp.types import HeadSpec, Instance

D, H, DK = 768, 12, 64
PROMPTS = ROOT / "data" / "prompts.txt"
GATE_TOL = 1e-10


# ----------------------------------------------------------------------------------------------- model & prompts
def load_prompts(path: Path = PROMPTS) -> list[str]:
    return [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_model():
    """``gpt2`` in float64, eager attention, eval mode; torch threads from the budget."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    from aimp import compute

    compute.set_torch_threads()
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager", dtype=torch.float64).eval()
    return model, tok


def layer_weights(model, l: int) -> dict:
    """float64 numpy copies of block ``l``'s parameters."""
    blk = model.transformer.h[l]

    def a(t):
        return t.detach().cpu().numpy().astype(np.float64).copy()

    return {
        "ln1_g": a(blk.ln_1.weight), "ln1_b": a(blk.ln_1.bias), "ln1_eps": float(blk.ln_1.eps),
        "W_attn": a(blk.attn.c_attn.weight), "b_attn": a(blk.attn.c_attn.bias),
        "W_proj": a(blk.attn.c_proj.weight), "b_proj": a(blk.attn.c_proj.bias),
        "ln2_g": a(blk.ln_2.weight), "ln2_b": a(blk.ln_2.bias), "ln2_eps": float(blk.ln_2.eps),
        "W_fc": a(blk.mlp.c_fc.weight), "b_fc": a(blk.mlp.c_fc.bias),
        "W_mp": a(blk.mlp.c_proj.weight), "b_mp": a(blk.mlp.c_proj.bias),
    }


def all_weights(model) -> list[dict]:
    return [layer_weights(model, l) for l in range(len(model.transformer.h))]


def head_weights(W: dict, h: int) -> dict:
    """``W_Q, W_K, W_V`` (768 x 64), ``b_Q, b_K, b_V`` (64), ``W_O`` (64 x 768) of head ``h``."""
    sl = slice(h * DK, (h + 1) * DK)
    Wa, ba = W["W_attn"], W["b_attn"]
    return {"W_Q": Wa[:, sl], "W_K": Wa[:, D:][:, sl], "W_V": Wa[:, 2 * D:][:, sl],
            "b_Q": ba[sl], "b_K": ba[D:][sl], "b_V": ba[2 * D:][sl], "W_O": W["W_proj"][sl, :]}


# ------------------------------------------------------------------------------------------- numpy forward
def layernorm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return g * (x - mu) / np.sqrt(var + eps) + b


def gelu_new(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def head_forward(z: np.ndarray, hw: dict) -> dict:
    """All query positions of one head (causal): scores, pattern ``A`` (T x T), head output (T x 64), contribution
    ``(T x 768)``."""
    q = z @ hw["W_Q"] + hw["b_Q"]
    k = z @ hw["W_K"] + hw["b_K"]
    v = z @ hw["W_V"] + hw["b_V"]
    T = z.shape[0]
    s = q @ k.T / math.sqrt(DK)
    s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
    s = s - s.max(axis=1, keepdims=True)
    A = np.exp(s)
    A /= A.sum(axis=1, keepdims=True)
    out = A @ v
    return {"A": A, "out": out, "contrib": out @ hw["W_O"]}


def block_forward(x: np.ndarray, W: dict) -> dict:
    z = layernorm(x, W["ln1_g"], W["ln1_b"], W["ln1_eps"])
    heads = [head_forward(z, head_weights(W, h)) for h in range(H)]
    merged = np.concatenate([hd["out"] for hd in heads], axis=1)
    attn = merged @ W["W_proj"] + W["b_proj"]
    h1 = x + attn
    m = gelu_new(layernorm(h1, W["ln2_g"], W["ln2_b"], W["ln2_eps"]) @ W["W_fc"] + W["b_fc"]) @ W["W_mp"] + W["b_mp"]
    return {"z": z, "A": np.stack([hd["A"] for hd in heads]), "merged": merged, "attn": attn, "out": h1 + m}


# --------------------------------------------------------------------------------------- hooked extraction
def extract(model, tok, prompt: str) -> dict:
    """Run the model once with hooks. Per layer ``l``: ``x[l]`` (block input, T x 768), ``z[l]`` (``ln_1`` output),
    ``merged[l]`` (input of ``attn.c_proj``: the 12 head outputs side by side), ``attn[l]`` (attention output after
    ``c_proj``), ``out[l]`` (block output), ``A[l]`` (12 x T x T attention probabilities)."""
    import torch

    ids = tok(prompt, return_tensors="pt")["input_ids"]
    L = len(model.transformer.h)
    rec = {k: [None] * L for k in ("x", "z", "merged", "attn", "out")}
    hooks = []

    def arr(t):
        return t[0].detach().cpu().numpy().astype(np.float64).copy()

    for l, blk in enumerate(model.transformer.h):
        def pre_block(mod, args, kwargs, l=l):
            hs = args[0] if args else kwargs["hidden_states"]
            rec["x"][l] = arr(hs)

        def post_block(mod, args, kwargs, output, l=l):
            rec["out"][l] = arr(output[0] if isinstance(output, tuple) else output)

        hooks.append(blk.register_forward_pre_hook(pre_block, with_kwargs=True))
        hooks.append(blk.register_forward_hook(post_block, with_kwargs=True))
        hooks.append(blk.ln_1.register_forward_hook(lambda m, i, o, l=l: rec["z"].__setitem__(l, arr(o))))
        hooks.append(blk.attn.c_proj.register_forward_pre_hook(lambda m, i, l=l: rec["merged"].__setitem__(l, arr(i[0]))))
        hooks.append(blk.attn.register_forward_hook(
            lambda m, i, o, l=l: rec["attn"].__setitem__(l, arr(o[0] if isinstance(o, tuple) else o))))
    try:
        with torch.no_grad():
            out = model(ids, output_attentions=True)
    finally:
        for hk in hooks:
            hk.remove()
    A = [a[0].detach().cpu().numpy().astype(np.float64) for a in out.attentions]
    res = {k: np.stack(v) for k, v in rec.items()}
    res["A"] = np.stack(A)
    res["logits"] = out.logits[0].detach().cpu().numpy().astype(np.float64).copy()  # (T x vocab)
    res["ids"] = ids[0].numpy().copy()
    res["tokens"] = [tok.decode([int(t)]) for t in ids[0]]
    return res


def gate(ext: dict, weights: list[dict]) -> dict:
    """The 1e-10 reproduction gate: from the hooked block inputs ``x[l]`` and the extracted weights, recompute
    in numpy ``z = ln_1(x)``, every head's pattern and output (compared with the ``c_proj`` input, i.e. per head),
    the attention output, and the full block output. Max abs and max relative deviations per layer."""
    rows = []
    for l, W in enumerate(weights):
        bf = block_forward(ext["x"][l], W)

        def dev(a, b):
            ab = float(np.max(np.abs(a - b)))
            return ab, ab / max(1e-300, float(np.max(np.abs(b))))

        r = {"layer": l}
        for key, mine, theirs in (("z", bf["z"], ext["z"][l]), ("A", bf["A"], ext["A"][l]),
                                  ("attn", bf["attn"], ext["attn"][l]), ("block_out", bf["out"], ext["out"][l])):
            r[f"{key}_abs"], r[f"{key}_rel"] = dev(mine, theirs)
        per_head = [float(np.max(np.abs(bf["merged"][:, h * DK:(h + 1) * DK] - ext["merged"][l][:, h * DK:(h + 1) * DK])))
                    for h in range(H)]
        r["head_out_abs_max"] = max(per_head)
        r["head_out_abs"] = per_head
        r["block_out_scale"] = float(np.max(np.abs(ext["out"][l])))
        if l + 1 < len(weights):  # the next block's input is this block's output
            r["chain_abs"] = float(np.max(np.abs(ext["out"][l] - ext["x"][l + 1])))
        rows.append(r)
    worst = {k: max(r[k] for r in rows) for k in ("z_abs", "A_abs", "head_out_abs_max", "attn_abs", "block_out_abs")}
    return {"rows": rows, "worst": worst, "tol": GATE_TOL,
            "passed": bool(all(v <= GATE_TOL for v in worst.values()))}


# ------------------------------------------------------------------------------- final LayerNorm and unembedding
def final_weights(model) -> dict:
    """The final LayerNorm ``ln_f`` (``g``, ``b``, ``eps``) and the tied unembedding ``wte`` (vocab x 768), float64."""
    def a(t):
        return t.detach().cpu().numpy().astype(np.float64).copy()

    lnf = model.transformer.ln_f
    return {"lnf_g": a(lnf.weight), "lnf_b": a(lnf.bias), "lnf_eps": float(lnf.eps), "wte": a(model.transformer.wte.weight)}


def final_logits(x: np.ndarray, F: dict) -> np.ndarray:
    """Logits from the residual stream after the last block (rows): ``ln_f(x) wte^T`` (the tied head has no bias)."""
    return layernorm(x, F["lnf_g"], F["lnf_b"], F["lnf_eps"]) @ F["wte"].T


def numpy_logits(x0: np.ndarray, weights: list[dict], F: dict) -> np.ndarray:
    """The whole numpy forward pass from the input of block 0 (the embeddings): every block, then ``ln_f`` and ``wte``."""
    x = np.asarray(x0, float)
    for W in weights:
        x = block_forward(x, W)["out"]
    return final_logits(x, F)


def lnf_sigma(x: np.ndarray, eps: float) -> float:
    """``sigma(x) = (||P x||^2 / d + eps)^(1/2)`` of ``ln_f`` (``P = I - 11^T / d``)."""
    Px = x - x.mean()
    return math.sqrt(float(Px @ Px) / x.shape[0] + eps)


def lm_spec(xbar: np.ndarray, F: dict, t1: int, t2: int) -> dict:
    """The logit-margin specification. With the ``ln_f`` scale frozen at ``sigma_bar = sigma(xbar)``, a change ``y`` of the
    residual stream at the position changes ``logit_t1 - logit_t2`` by ``c_m^T y`` with
    ``c_m = P diag(g) (w_t1 - w_t2) / sigma_bar`` (the bias of ``ln_f`` cancels). Also the exact gradient of the logit
    difference at ``xbar`` (with the derivative of ``sigma``) and its cosine with ``c_m``."""
    d = xbar.shape[0]
    g, wd = F["lnf_g"], F["wte"][t1] - F["wte"][t2]
    sig = lnf_sigma(xbar, F["lnf_eps"])
    u = g * wd
    c_m = (u - u.mean()) / sig
    Px = xbar - xbar.mean()
    grad = c_m - Px * float(Px @ u) / (d * sig ** 3)
    cos = float(c_m @ grad) / (float(np.linalg.norm(c_m)) * float(np.linalg.norm(grad)))
    return {"c_m": c_m, "sigma_bar": sig, "grad_exact": grad, "cos_exact": cos}


# ----------------------------------------------------------------------------------------------- projection
@dataclass
class Projection:
    P: np.ndarray  # (3 d_k, d)
    b: np.ndarray  # (3 d_k,)
    E: np.ndarray  # U S (3 d_k, r)
    Vt: np.ndarray  # (r, d)
    r: int
    d_k: int


def projection(W_Q, W_K, W_V, b_Q=None, b_K=None, b_V=None, rtol: float = 1e-12) -> Projection:
    P = np.vstack([W_Q.T, W_K.T, W_V.T])
    d_k = W_Q.shape[1]
    b = np.concatenate([np.zeros(d_k) if x is None else x for x in (b_Q, b_K, b_V)])
    U, s, Vt = np.linalg.svd(P, full_matrices=False)
    r = int(np.sum(s > rtol * s[0]))
    return Projection(P=P, b=b, E=U[:, :r] * s[:r], Vt=Vt[:r], r=r, d_k=d_k)


def projected_head(W_O: np.ndarray, d_k: int, d_v: int | None = None, name: str = "proj") -> HeadSpec:
    """``M_p``, ``N_p`` acting on ``zp = (q, k, v)`` (the reduced coordinates of Proposition 15). ``W_O``:
    ``d_v x d_out``."""
    d_v = d_v or W_O.shape[0]
    dp = 2 * d_k + d_v
    Mp = np.zeros((dp, dp))
    Mp[:d_k, d_k:2 * d_k] = np.eye(d_k) / math.sqrt(d_k)
    Np = np.zeros((W_O.shape[1], dp))
    Np[:, 2 * d_k:] = W_O.T
    return HeadSpec(M=Mp, N=Np, d=dp, d_k=d_k, d_v=d_v, name=name)


def project_instance(inst: Instance, b_Q=None, b_K=None, b_V=None) -> Instance:
    """The exact reduction of Proposition 15 as a transformation ``(HeadSpec, sets) -> (HeadSpec, sets)``:
    ``inst.head.raw`` must hold ``W_Q, W_K, W_V, W_O`` (rows = inputs) and every set must be a ball (radius 0 allowed).
    Biases (optional) are absorbed into the projected centers."""
    raw = inst.head.raw
    pr = projection(raw["W_Q"], raw["W_K"], raw["W_V"], b_Q, b_K, b_V)
    head = projected_head(raw["W_O"], pr.d_k, raw["W_V"].shape[1], name=f"{inst.head.name}_proj")
    sets = {}
    for t, st in inst.sets.items():
        if st.kind != "ball":
            raise ValueError("projection implemented for ball sets")
        sets[t] = S.ellipsoid(pr.P @ st.center + pr.b, pr.E, float(st.rho))
    return Instance(head=head, keys=list(inst.keys), query=inst.query, sets=sets, c=inst.c.copy(),
                    meta={**inst.meta, "projected": True, "r": pr.r})


def lift_point(inst_orig: Instance, Zp: np.ndarray, pr: Projection) -> np.ndarray:
    """A point of the original balls with the same projection as ``Zp`` (projected ellipsoid points): ``z = zbar + V w``
    with ``w`` the ellipsoid coordinates (least squares on ``E``)."""
    Z = inst_orig.centers().copy()
    for t, st in inst_orig.sets.items():
        w, *_ = np.linalg.lstsq(pr.E, Zp[t] - (pr.P @ st.center + pr.b), rcond=None)
        Z[t] = st.center + pr.Vt.T @ w
    return Z


# --------------------------------------------------------------------------------------- role-reduced Shor
def _reduce(E: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """Thin factor ``F`` (rows x rank) with ``F F^T = E E^T``: same image of the unit ball."""
    U, s, _ = np.linalg.svd(E, full_matrices=False)
    k = int(np.sum(s > rtol * s[0])) if s.size and s[0] > 0 else 0
    return U[:, :k] * s[:k]


def score_intervals_roles(inst: Instance, threads=None, method: str = "shor") -> dict:
    """Certified score intervals for a projected GPT-2 instance (``M_p = [[0, I/sqrt(d_k), 0], 0, 0]``, ellipsoid sets
    sharing one ``E``). Key ``j != i``: Shor on ``q in {c_i[q] + E_q w}``, ``k in {c_j[k] + E_k w'}`` (two 64-dim
    ellipsoids); ``j = i``: Shor on ``(q, k) in {c_i[q,k] + E_qk w}`` (one 128-dim ellipsoid). ``method="cs"``: the
    Cauchy–Schwarz intervals (:func:`aimp.bounds.l5_halfwidth`) on the projected instance (for ``Delta_cs``)."""
    from aimp import bounds
    from aimp.sdp_mosek import shor_score  # bounds._shor in the MOSEK task API

    if inst.meta.get("model", "gpt2") != "gpt2":  # gpt-oss: per-token (rotated) shape matrices and the sink
        from aimp import gptoss

        return gptoss.score_intervals_roles(inst, threads=threads, method=method)
    if method == "cs":
        return bounds.score_intervals(inst, "cs")
    hd = inst.head
    dk = hd.d_k
    i = inst.query
    Si = inst.sets[i]
    s_lo, s_hi, info = [], [], {}
    E = Si.E
    Eq, Ek, Eqk = _reduce(E[:dk]), _reduce(E[dk:2 * dk]), _reduce(E[:2 * dk])
    M1 = hd.M[:dk, dk:2 * dk]
    M2 = np.zeros((2 * dk, 2 * dk))
    M2[:dk, dk:] = M1
    for j in inst.keys:
        Sj = inst.sets[j]
        if Sj.E is not E and not np.array_equal(Sj.E, E):
            raise ValueError("score_intervals_roles expects one shared E")
        if j == i:
            tiny = Instance(head=HeadSpec(M=M2, N=np.zeros((1, 2 * dk)), d=2 * dk, d_k=dk, d_v=1), keys=[0], query=0,
                            sets={0: S.ellipsoid(Si.center[:2 * dk], Eqk, Si.rho)}, c=np.zeros(1))
            lo, hi, inf = shor_score(tiny, 0, threads=threads) if Si.rho > 0 else (*bounds._linear_interval(tiny, 0), {})
        else:
            tiny = Instance(head=HeadSpec(M=M1, N=np.zeros((1, dk)), d=dk, d_k=dk, d_v=1), keys=[1], query=0,
                            sets={0: S.ellipsoid(Si.center[:dk], Eq, Si.rho),
                                  1: S.ellipsoid(Sj.center[dk:2 * dk], Ek, Sj.rho)}, c=np.zeros(1))
            lin = bounds._linear_interval(tiny, 1)
            if lin is not None:
                lo, hi, inf = lin[0], lin[1], {"linear_exact": True}
            else:
                lo, hi, inf = shor_score(tiny, 1, threads=threads)
        s_lo.append(lo)
        s_hi.append(hi)
        info[str(j)] = inf
    return {"method": "shor_roles", "s_lo": np.array(s_lo), "s_hi": np.array(s_hi), "info": info}


# ------------------------------------------------------------------------------------- LayerNorm contraction
def ln_factor(x: np.ndarray, g: np.ndarray, eps: float) -> dict:
    """LayerNorm contraction bound (:func:`aimp.bounds.l6_layernorm_radius`) per token: ``rho_z <= f_j rho_x`` with
    ``f_j = 2 sqrt(d) ||g||_inf / ||P x_j||`` (valid for ``rho_x < ||P x_j||``), and the local Lipschitz constant
    ``||J_LN(x_j)||_2`` for comparison (first order, not a certificate). ``x``: (T x d) block inputs."""
    d = x.shape[1]
    Px = x - x.mean(axis=1, keepdims=True)
    nP = np.linalg.norm(Px, axis=1)
    f_l6 = 2 * math.sqrt(d) * float(np.max(np.abs(g))) / nP
    jac = []
    C = np.eye(d) - np.ones((d, d)) / d
    for t in range(x.shape[0]):
        ne = math.sqrt(nP[t] ** 2 + d * eps)
        u = Px[t] / ne
        J = (g[:, None] * (math.sqrt(d) / ne) * (np.eye(d) - np.outer(u, u))) @ C
        jac.append(float(np.linalg.norm(J, 2)))
    return {"f_l6": f_l6, "Px_norm": nP, "jac_norm": np.array(jac)}


# ------------------------------------------------------------------------------------------ GPT-2 head instances
def spec_vectors(W_O: np.ndarray, ybar: np.ndarray, layer: int, head: int, n_random: int = 3) -> dict:
    """Specifications: ``"a"`` = ``ybar / ||ybar||`` (the direction of the head's nominal output, used in the paper);
    ``"b0".."b2"`` = random unit vectors in the row space of ``W_O`` (``W_O^T g``, ``g ~ N(0, I_64)``, seed
    ``100000 + 1000 layer + 10 head + k``)."""
    out = {"a": ybar / np.linalg.norm(ybar)}
    for k in range(n_random):
        rng = np.random.default_rng(100000 + 1000 * layer + 10 * head + k)
        c = W_O.T @ rng.standard_normal(W_O.shape[0])
        out[f"b{k}"] = c / np.linalg.norm(c)
    return out


def phase3_instance(z: np.ndarray, hw: dict, pos: int, rho_rel: float, c: np.ndarray, pr: Projection | None = None,
                    meta: dict | None = None) -> Instance:
    """Projected instance: query ``pos``, keys ``0..pos``, balls ``rho_j = rho_rel ||zbar_j||`` around the post-LN
    inputs ``z[j]`` mapped by the head's projection."""
    pr = pr or projection(hw["W_Q"], hw["W_K"], hw["W_V"], hw["b_Q"], hw["b_K"], hw["b_V"])
    head = projected_head(hw["W_O"], DK, DK, name=(meta or {}).get("name", "gpt2_head"))
    sets = {}
    for t in range(pos + 1):
        rho = rho_rel * float(np.linalg.norm(z[t]))
        sets[t] = S.ellipsoid(pr.P @ z[t] + pr.b, pr.E, rho)
    return Instance(head=head, keys=list(range(pos + 1)), query=pos, sets=sets, c=np.asarray(c, float),
                    meta={**(meta or {}), "rho_rel": rho_rel, "r": pr.r})


def nominal_value_range(inst: Instance) -> float:
    """``max_j c^T N zbar_j - min_j c^T N zbar_j`` over the keys: the nominal spread of the values along ``c``, which
    normalizes the range-normalized gaps of the paper."""
    g = np.array([inst.c @ inst.head.N @ inst.sets[j].center for j in inst.keys])
    return float(g.max() - g.min())
