"""gpt-oss-20b: extraction with its checks (E1 attention reproduction, E3 logits and the logit-margin vector), the float64
cache, and the head certification units in reduced coordinates (the ``model="gpt-oss-20b"`` option of
:func:`aimp.gpt2_units.make_instance`).

**Model** (``openai/gpt-oss-20b``; Hugging Face ``GptOssForCausalLM``): 24 layers, residual width ``d = 2880``, 64 query heads
of dimension 64 per layer, 8 key-value groups (query head ``h`` reads group ``g = h // 8``), attention biases, RMSNorm,
rotary position embeddings (YaRN; the cos / sin tables carry the YaRN attention factor), one learned sink logit per head,
alternating sliding-window (128) and full attention, mixture-of-experts MLPs (MXFP4 on disk, dequantized on CPU). For a
layer ``l``, head ``h`` and query position ``i`` with normalized tokens ``z_j`` (the output of the layer's input RMSNorm):

    q_i = W_Q z_i + b_Q,  k_j = W_K z_j + b_K,  v_j = W_V z_j + b_V,  s_j = scale (R_i q_i)^T (R_j k_j)   (j <= i),
    a = softmax(s_0, ..., s_i, s_sink),  y_i = W_O sum_{j <= i} a_j v_j

(``W_Q = q_proj.weight[64 h : 64 h + 64]``, ``W_K`` / ``W_V`` the rows of group ``g`` of ``k_proj`` / ``v_proj``, ``W_O =
o_proj.weight[:, 64 h : 64 h + 64]``; ``o_proj.bias`` is shared by the 64 heads and is not part of any head, as GPT-2's
``c_proj.bias``). ``R_j`` is the linear map of the model's own ``apply_rotary_pos_emb`` at position ``j``, built by applying
that function to the identity with the cached cos / sin tables (the convention is not re-derived).

**Reduced coordinates** (Proposition 15, ``d' = 192``): ``P = [W_Q; W_K; W_V]`` (192 x 2880), ``b = [b_Q; b_K; b_V]``, thin SVD
``P = U S V^T``; the ball ``{zbar_j + u : ||u|| <= rho_j}`` maps to the ellipsoid ``{P zbar_j + b + E w : ||w|| <= rho_j}``,
``E = U S``. The rotations act after the projection: token ``t`` gets the invertible map ``D_t = diag(R_i, R_t, I_64)`` on
``(q, k, v)`` (the query block is rotated by the query's position, the key block by the token's own), so its set is the
ellipsoid ``{D_t (P zbar_t + b) + D_t E w}`` and ``s_j = zp_i^T M_p zp_j`` with ``M_p[q, k] = scale I``: the head problem with the
per-key matrix ``M_j = D_i^T M_p D_j`` (qk block ``scale R_i^T R_j``), by a per-token change of variables (exact; the
check E2 and ``tests/test_gpt_oss.py``).
**Sink.** Two constant coordinates ``(e, f)`` are appended: every token has ``e = 1, f = 0``; one extra key (token ``i + 1``,
a point set) has ``zp = (0, 0, 0, 0, 1)`` and ``M_p[e, f] = s_sink``, so its score is ``s_sink`` for every input and its value
``N zp = 0``. The ambient dimension is 194, but the sets vary in the 192-dimensional ellipsoid coordinates only (the lifted
problems use those; the sink adds one key of the pattern and no lifted variable).

:func:`prepare` downloads ``openai/gpt-oss-20b`` from the Hugging Face hub (about 14 GB; the ``original/`` and ``metal/``
folders are not needed), loads it once on CPU (the MXFP4 experts are dequantized; forward passes in float32, about 85 GB),
runs the five prompts with hooks, checks E1 and E3 and caches every array the units read in float64 under
``cache_dir()/gpt-oss-20b`` (5.1 GB of attention weights, the activations, the rotary tables, the final normalization and the
unembedding rows of the top tokens). No later step loads the model.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

from aimp import sets as S
from aimp.io import cache_dir
from aimp.types import HeadSpec, Instance

MODEL_ID = "openai/gpt-oss-20b"
MODEL_NAME = "gpt-oss-20b"
N_LAYERS, N_HEADS, N_KV, DH, D = 24, 64, 8, 64, 2880
GROUP = N_HEADS // N_KV  # 8 query heads per key-value group
SAMPLE_HEADS = (0, 9, 18, 27, 36, 45, 54, 63)  # one query head per key-value group
E1_TOL = 1e-4  # relative
E2_TOL = 1e-10
E3_TOL = 1e-4  # relative, logits at position 4
E3_LIN_TOL = 1e-6  # relative, frozen-scale margin
POS = 4
TOPK = 20
DP = 3 * DH + 2  # ambient reduced dimension (q, k, v, e, f)
IE, IF = 3 * DH, 3 * DH + 1  # the two constant coordinates


# ================================================================================================ extraction (one process)
def model_cache() -> Path:
    """``cache_dir()/gpt-oss-20b`` (``$AIMP_CACHE_DIR``, default ``<repository>/cache``)."""
    return cache_dir() / MODEL_NAME


def load_model(dtype: str = "float32"):
    """gpt-oss-20b on CPU: MXFP4 experts dequantized (bf16), eager attention, then cast to ``dtype``."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from transformers import AutoTokenizer, GptOssForCausalLM, Mxfp4Config

    from aimp import compute

    compute.set_torch_threads()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = GptOssForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, attn_implementation="eager",
                                              quantization_config=Mxfp4Config(dequantize=True), device_map="cpu")
    model.eval()
    if dtype == "float32":
        model = model.float()
    return model, tok


def _np(t) -> np.ndarray:
    return t.detach().cpu().to(dtype=__import__("torch").float64).numpy().copy()


def rotation_matrices(cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """``R[t]`` (T x 64 x 64): the matrix of the model's ``apply_rotary_pos_emb`` at position ``t`` (applied to the identity
    in float64 with the model's own cos / sin tables, which already carry the YaRN attention factor)."""
    import torch
    from transformers.models.gpt_oss.modeling_gpt_oss import apply_rotary_pos_emb

    T = cos.shape[0]
    eye = torch.eye(DH, dtype=torch.float64)[None, None].expand(1, 1, DH, DH)  # (b, heads, rows, dh): rows e_k
    R = np.zeros((T, DH, DH))
    for t in range(T):
        c = torch.as_tensor(cos[t:t + 1][None].repeat(DH, axis=1), dtype=torch.float64)  # (1, DH rows, dh/2)
        s = torch.as_tensor(sin[t:t + 1][None].repeat(DH, axis=1), dtype=torch.float64)
        out, _ = apply_rotary_pos_emb(eye, eye, c, s)  # row k = R e_k
        R[t] = out[0, 0].numpy().T
    return R


def attention_f64(z: np.ndarray, W: dict, R: np.ndarray, window: int | None) -> dict:
    """Float64 re-implementation of one attention sublayer (check E1) from the normalized tokens ``z`` (T x 2880), the
    cached weights, the rotation matrices and the window: per-head pattern with the sink (T x (T + 1)), the head outputs
    (T x 4096, the input of ``o_proj``) and the attention output (``o_proj`` with its bias)."""
    T = z.shape[0]
    q = (z @ W["Wq"].T + W["bq"]).reshape(T, N_HEADS, DH)
    k = (z @ W["Wk"].T + W["bk"]).reshape(T, N_KV, DH)
    v = (z @ W["Wv"].T + W["bv"]).reshape(T, N_KV, DH)
    qr = np.einsum("tab,thb->tha", R, q)
    kr = np.einsum("tab,tgb->tga", R, k)
    scale = float(W["scale"])
    probs = np.zeros((N_HEADS, T, T + 1))
    merged = np.zeros((T, N_HEADS * DH))
    mask = np.tril(np.ones((T, T), bool))
    if window is not None:
        mask &= ~np.tril(np.ones((T, T), bool), -window)
    for h in range(N_HEADS):
        g = h // GROUP
        s = scale * qr[:, h, :] @ kr[:, g, :].T
        s = np.where(mask, s, -np.inf)
        s = np.concatenate([s, np.full((T, 1), float(W["sinks"][h]))], axis=1)
        s = s - s.max(axis=1, keepdims=True)
        a = np.exp(s)
        a /= a.sum(axis=1, keepdims=True)
        probs[h] = a
        merged[:, h * DH:(h + 1) * DH] = a[:, :T] @ v[:, g, :]
    return {"probs": probs, "merged": merged, "attn": merged @ W["Wo"].T + W["bo"], "q_rot": qr, "k_rot": kr}


def rmsnorm(x: np.ndarray, g: np.ndarray, eps: float) -> np.ndarray:
    return g * x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)


def rms_scale(x: np.ndarray, eps: float) -> float:
    """``r(x) = (||x||^2 / d + eps)^(1/2)`` of RMSNorm."""
    return math.sqrt(float(x @ x) / x.shape[0] + eps)


def lm_spec(xbar: np.ndarray, g: np.ndarray, eps: float, u1: np.ndarray, u2: np.ndarray) -> dict:
    """The logit-margin specification: with the final RMSNorm scale frozen at ``r_bar = r(xbar)``, a change ``delta`` of the final residual at the
    position changes ``logit_t1 - logit_t2`` by ``c_m^T delta``, ``c_m = diag(g) (u_t1 - u_t2) / r_bar``; the exact gradient
    (with the derivative of ``r``) and its cosine with ``c_m`` (descriptive)."""
    d = xbar.shape[0]
    r = rms_scale(xbar, eps)
    w = g * (u1 - u2)
    c_m = w / r
    grad = c_m - xbar * float(xbar @ w) / (d * r ** 3)
    cos = float(c_m @ grad) / (float(np.linalg.norm(c_m)) * float(np.linalg.norm(grad)))
    return {"c_m": c_m, "r_bar": r, "grad_exact": grad, "cos_exact": cos}


#: the files of a complete cache (``prepare()`` writes all of them; a directory with any other subset is incomplete)
CACHE_FILES = tuple(f"weights_L{l:02d}.npz" for l in range(N_LAYERS)) + tuple(f"prompt{p}.npz" for p in range(5)) + (
    "final.npz", "extract.json")


def _refuse_symlinks(cache: Path) -> None:
    """``prepare()`` never writes through a symlink: neither ``cache_dir()`` nor ``model_cache()`` nor an entry of it may be
    one (a symlinked cache may belong to another project)."""
    links = [q for q in (cache_dir(), cache) if q.is_symlink()]
    if cache.is_dir() and not cache.is_symlink():
        links += [q for q in cache.iterdir() if q.is_symlink()]
    if links:
        raise SystemExit(f"gpt-oss cache: {', '.join(str(q) for q in links)} is a symlink; prepare() does not write through "
                         "symlinks. Point AIMP_CACHE_DIR at a real directory (or at a complete cache, which is only read).")


def prepare(force: bool = False, log=print, dtype: str = "float32") -> dict:
    """The cache of the gpt-oss units (``model_cache()``: ``weights_L<ll>.npz`` per layer, ``prompt<p>.npz`` per prompt,
    ``final.npz``, ``extract.json``) and the extraction checks E1 / E3 (:func:`_extract_into`). An existing cache is never
    overwritten in place:

    * a complete cache (every file of ``CACHE_FILES``) is only read: its ``extract.json`` is returned (also through a
      symlink, since nothing is written);
    * otherwise nothing is written through a symlink: a cache directory that is, or contains, a symlink is refused
      (:func:`_refuse_symlinks`);
    * an existing incomplete cache (any entry, but not every file) is not re-extracted silently: ``SystemExit`` unless
      ``force=True`` (``python experiments/gpt_oss.py --force``);
    * a new extraction is written to a temporary directory next to the cache and moved into place by one ``os.rename``
      (atomic on one file system) only after its checks passed; with ``force=True`` an existing cache is first renamed to
      ``<name>.replaced-<time>`` and kept. A failed extraction leaves the cache as it was and removes its temporary
      directory."""
    import shutil

    CACHE = model_cache()
    complete = all((CACHE / f).is_file() for f in CACHE_FILES)
    if complete and not force:  # read only (also through a symlink)
        return json.loads((CACHE / "extract.json").read_text())
    _refuse_symlinks(CACHE)  # everything below may write
    entries = sorted(q.name for q in CACHE.iterdir()) if CACHE.is_dir() else []
    if entries and not force:
        missing = [f for f in CACHE_FILES if not (CACHE / f).is_file()]
        raise SystemExit(f"gpt-oss cache {CACHE} is incomplete (missing {', '.join(missing[:4])}"
                         f"{', ...' if len(missing) > 4 else ''}); it is not re-extracted silently. Re-extract with "
                         "`python experiments/gpt_oss.py --force` (the current directory is kept as "
                         f"{CACHE.name}.replaced-<time>) or point AIMP_CACHE_DIR elsewhere.")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.parent / f".{CACHE.name}.tmp-{os.getpid()}-{time.strftime('%Y%m%d-%H%M%S')}"
    tmp.mkdir()
    try:
        out = _extract_into(tmp, log=log, dtype=dtype)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if CACHE.exists():
        old = CACHE.parent / f"{CACHE.name}.replaced-{time.strftime('%Y%m%d-%H%M%S')}"
        os.rename(CACHE, old)
        log(f"extract: the previous cache directory is kept as {old}")
    os.rename(tmp, CACHE)
    _W.clear(), _P.clear(), _F.clear(), _PROJ.clear()
    return out


def _extract_into(CACHE: Path, log=print, dtype: str = "float32") -> dict:
    """Extraction and the checks E1, E3 in one process, written into the (new, empty) directory ``CACHE``: load the model,
    run the five prompts with hooks, cache in float64 (``weights_L<ll>.npz`` per layer, ``prompt<p>.npz`` per prompt,
    ``final.npz``), then free the model. Returns (and writes ``CACHE/extract.json``) the record with the E1 / E3 rows;
    raises ``SystemExit`` if a check fails. E1: for every layer and prompt the float64 re-implementation of the attention
    sublayer from the cached weights and normalized tokens reproduces the model's attention output to 1e-4 relative. E3: the
    cached final residual, RMSNorm and unembedding reproduce the logits at position 4 to 1e-4 relative, and the frozen-scale
    margin of the model's own final gain and unembedding rows changes by ``c_m^T delta`` to 1e-6 relative. Called only by
    :func:`prepare`."""
    import gc

    import torch

    from aimp import gpt2
    from aimp.io import env_info, git_commit, jsonable, write_json_atomic

    rec_path = CACHE / "extract.json"
    t0 = time.perf_counter()
    import psutil

    proc = psutil.Process()
    torch.set_grad_enabled(False)  # inference only (every tensor below is a constant)
    model, tok = load_model(dtype)
    t_load = time.perf_counter() - t0
    rss_load = proc.memory_info().rss / 2 ** 30
    cfg = model.config
    log(f"extract: model loaded in {t_load:.0f} s ({dtype}), RSS {rss_load:.1f} GB, "
        f"params dtype {next(model.parameters()).dtype}")
    layers = model.model.layers
    scale = float(layers[0].self_attn.scaling)
    mscale = float(model.model.rotary_emb.attention_scaling)
    windows = [layers[l].self_attn.sliding_window for l in range(N_LAYERS)]
    # ---- weights per layer (float64; bf16 values on disk, exact in float64)
    for l in range(N_LAYERS):
        p = CACHE / f"weights_L{l:02d}.npz"
        at = layers[l].self_attn
        W = {"Wq": _np(at.q_proj.weight), "bq": _np(at.q_proj.bias), "Wk": _np(at.k_proj.weight), "bk": _np(at.k_proj.bias),
             "Wv": _np(at.v_proj.weight), "bv": _np(at.v_proj.bias), "Wo": _np(at.o_proj.weight), "bo": _np(at.o_proj.bias),
             "sinks": _np(at.sinks), "ln_g": _np(layers[l].input_layernorm.weight),
             "ln_eps": np.float64(layers[l].input_layernorm.variance_epsilon), "scale": np.float64(scale),
             "window": np.int64(-1 if windows[l] is None else windows[l])}
        tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, **W)
        os.replace(tmp, p)
    gf = _np(model.model.norm.weight)
    eps_f = float(model.model.norm.variance_epsilon)
    texts = gpt2.load_prompts()
    rng = np.random.default_rng(8_000_001)
    prompts, e1_rows, fin = [], [], {"lm_rows": {}, "c_m": [], "grad_exact": [], "xbar": [], "t1": [], "t2": [],
                                      "margin": [], "r_bar": []}
    Wlm = model.lm_head.weight  # (vocab x d), float32 (bf16 values)
    for p, text in enumerate(texts):
        enc = tok(text, return_tensors="pt")
        ids = enc["input_ids"]
        T = int(ids.shape[1])
        rec = {k: [None] * N_LAYERS for k in ("x", "z", "attn", "merged")}
        pe = {}
        hooks = []

        def pre_layer(mod, args, kwargs, l=None):
            hs = args[0] if args else kwargs["hidden_states"]
            rec["x"][l] = _np(hs[0])
            cs = kwargs.get("position_embeddings")
            pe[l] = (_np(cs[0][0]), _np(cs[1][0]))

        for l, lay in enumerate(layers):
            hooks.append(lay.register_forward_pre_hook(lambda m, a, k, l=l: pre_layer(m, a, k, l), with_kwargs=True))
            hooks.append(lay.input_layernorm.register_forward_hook(lambda m, i, o, l=l: rec["z"].__setitem__(l, _np(o[0]))))
            hooks.append(lay.self_attn.register_forward_hook(
                lambda m, i, o, l=l: rec["attn"].__setitem__(l, _np((o[0] if isinstance(o, tuple) else o)[0]))))
            hooks.append(lay.self_attn.o_proj.register_forward_pre_hook(
                lambda m, i, l=l: rec["merged"].__setitem__(l, _np(i[0][0]))))
        fz = {}
        hooks.append(model.model.norm.register_forward_pre_hook(lambda m, i: fz.__setitem__("xf", _np(i[0][0]))))
        hooks.append(model.model.norm.register_forward_hook(lambda m, i, o: fz.__setitem__("nf", _np(o[0]))))
        t1 = time.perf_counter()
        try:
            with torch.no_grad():
                out = model(**enc, output_attentions=True, use_cache=False)
        finally:
            for hk in hooks:
                hk.remove()
        t_fwd = time.perf_counter() - t1
        logits = _np(out.logits[0])  # (T x vocab)
        probs_model = np.stack([_np(a[0]) for a in out.attentions])  # (L, H, T, T): the sink column is dropped
        cos0, sin0 = pe[0]
        same_pe = all(np.array_equal(pe[l][0], cos0) and np.array_equal(pe[l][1], sin0) for l in range(N_LAYERS))
        R = rotation_matrices(cos0, sin0)
        # ---- E1: every layer
        for l in range(N_LAYERS):
            Wl = dict(np.load(CACHE / f"weights_L{l:02d}.npz"))
            win = None if int(Wl["window"]) < 0 else int(Wl["window"])
            f = attention_f64(rec["z"][l], Wl, R, win)
            fnw = attention_f64(rec["z"][l], Wl, R, None) if win is not None else f
            dev_attn = float(np.max(np.abs(f["attn"] - rec["attn"][l])))
            sc_attn = float(np.max(np.abs(rec["attn"][l])))
            dev_m = float(np.max(np.abs(f["merged"] - rec["merged"][l])))
            sc_m = float(np.max(np.abs(rec["merged"][l])))
            dev_p = float(np.max(np.abs(f["probs"][:, :, :T] - probs_model[l])))
            z_np = rmsnorm(rec["x"][l], Wl["ln_g"], float(Wl["ln_eps"]))
            dev_z = float(np.max(np.abs(z_np - rec["z"][l]))) / float(np.max(np.abs(rec["z"][l])))
            e1_rows.append({"prompt": p, "layer": l, "attn_abs": dev_attn, "attn_scale": sc_attn,
                            "attn_rel": dev_attn / sc_attn, "merged_rel": dev_m / sc_m, "probs_abs": dev_p,
                            "z_rel_rmsnorm": dev_z, "window": win, "window_binds": bool(win is not None and T > win),
                            "window_effect_abs": float(np.max(np.abs(fnw["attn"] - f["attn"]))),
                            "sink_mass_pos4": float(np.median(f["probs"][:, POS, T])),
                            "ok": bool(dev_attn / sc_attn <= E1_TOL)})
        # ---- E3: logits at position 4 from the cached final residual, RMSNorm and the unembedding (float64)
        xf = fz["xf"]
        nf_np = rmsnorm(xf, gf, eps_f)
        lg_np = np.zeros(logits.shape[1])
        x4 = torch.as_tensor(nf_np[POS], dtype=torch.float64)
        for a0 in range(0, Wlm.shape[0], 16384):
            lg_np[a0:a0 + 16384] = (Wlm[a0:a0 + 16384].to(torch.float64) @ x4).numpy()
        dev_lg = float(np.max(np.abs(lg_np - logits[POS])))
        sc_lg = float(np.max(np.abs(logits[POS])))
        order = np.argsort(-logits[POS], kind="stable")
        order_np = np.argsort(-lg_np, kind="stable")
        t1_, t2_ = int(order[0]), int(order[1])
        u1, u2 = _np(Wlm[t1_]), _np(Wlm[t2_])
        xbar = xf[POS].copy()
        lm = lm_spec(xbar, gf, eps_f, u1, u2)
        # frozen-scale check with the model's own final-norm gain and lm_head rows (float64 copies)
        g_t = model.model.norm.weight.detach().to(torch.float64)
        U_t = Wlm[[t1_, t2_]].detach().to(torch.float64)
        rb = lm["r_bar"]

        def frozen(x):
            lg2 = U_t @ (g_t * x / rb)
            return float(lg2[0] - lg2[1])

        def unfrozen(x):
            lg2 = U_t @ (g_t * x / torch.sqrt((x * x).mean() + eps_f))
            return float(lg2[0] - lg2[1])

        deltas = []
        for _ in range(3):  # head output directions W_O^T u of random heads, and one generic direction
            l_, h_ = int(rng.integers(N_LAYERS)), int(rng.integers(N_HEADS))
            WO = layers[l_].self_attn.o_proj.weight[:, h_ * DH:(h_ + 1) * DH].detach().to(torch.float64).numpy()
            for _ in range(3):
                v_ = WO @ rng.standard_normal(DH)
                deltas.append(v_ * 1e-3 * np.linalg.norm(xbar) / np.linalg.norm(v_))
        v_ = rng.standard_normal(D)
        deltas.append(v_ * 1e-3 * np.linalg.norm(xbar) / np.linalg.norm(v_))
        xb_t = torch.as_tensor(xbar, dtype=torch.float64)
        f0, u0 = frozen(xb_t), unfrozen(xb_t)
        lin = []
        for dl in deltas:
            x_t = xb_t + torch.as_tensor(dl, dtype=torch.float64)
            pred = float(lm["c_m"] @ dl)
            chf, chu = frozen(x_t) - f0, unfrozen(x_t) - u0
            lin.append({"delta_norm": float(np.linalg.norm(dl)), "predicted": pred, "frozen_change": chf,
                        "rel_err": abs(chf - pred) / max(abs(pred), 1e-300), "unfrozen_change": chu,
                        "unfrozen_rel_diff": abs(chu - pred) / max(abs(pred), 1e-300)})
        bos = tok.bos_token_id
        special = set(tok.all_special_ids)
        top = [(int(t), tok.decode([int(t)]), float(logits[POS, t])) for t in order[:TOPK]]
        prow = {"prompt": p, "text": text, "T": T, "ids": ids[0].tolist(), "tokens": [tok.decode([int(t)]) for t in ids[0]],
                "first_token_special": bool(int(ids[0, 0]) in special), "bos_token_id": bos,
                "special_prepended": bool(int(ids[0, 0]) in special),
                "token_at_pos4": tok.decode([int(ids[0, POS])]), "top": top,
                "t1": t1_, "t2": t2_, "t1_str": tok.decode([t1_]), "t2_str": tok.decode([t2_]),
                "logit_t1": float(logits[POS, t1_]), "logit_t2": float(logits[POS, t2_]),
                "margin": float(logits[POS, t1_] - logits[POS, t2_]),
                "margin_to_third": float(logits[POS, t2_] - logits[POS, int(order[2])]),
                "top2_numpy_same": bool(int(order_np[0]) == t1_ and int(order_np[1]) == t2_),
                "E3_logits_abs": dev_lg, "E3_logit_scale": sc_lg, "E3_logits_rel": dev_lg / sc_lg,
                "E3_logits_ok": bool(dev_lg / sc_lg <= E3_TOL),
                "E3_final_norm_rel": float(np.max(np.abs(nf_np - fz["nf"]))) / float(np.max(np.abs(fz["nf"]))),
                "E3_lin": lin, "E3_lin_max_rel": max(r["rel_err"] for r in lin),
                "E3_lin_ok": bool(max(r["rel_err"] for r in lin) <= E3_LIN_TOL),
                "frozen_base": f0, "unfrozen_base": u0, "r_bar": rb, "c_m_norm": float(np.linalg.norm(lm["c_m"])),
                "cos_c_m_exact_gradient": lm["cos_exact"], "rope_tables_identical_across_layers": same_pe,
                "forward_s": t_fwd}
        prompts.append(prow)
        top_ids = np.argsort(-logits, axis=1, kind="stable")[:, :TOPK]
        np.savez(CACHE / f"prompt{p}.tmp.npz", ids=ids[0].numpy(), x=np.stack(rec["x"]), z=np.stack(rec["z"]),
                 attn=np.stack(rec["attn"]), merged=np.stack(rec["merged"]), probs_model=probs_model,
                 xf=xf, nf=fz["nf"], cos=cos0, sin=sin0, R=R, top_ids=top_ids,
                 top_logits=np.take_along_axis(logits, top_ids, axis=1), logits_pos4=logits[POS])
        os.replace(CACHE / f"prompt{p}.tmp.npz", CACHE / f"prompt{p}.npz")
        for t in [int(t) for t in order[:TOPK]]:
            fin["lm_rows"][t] = _np(Wlm[t])
        for k_, v2 in (("c_m", lm["c_m"]), ("grad_exact", lm["grad_exact"]), ("xbar", xbar), ("t1", t1_), ("t2", t2_),
                       ("margin", prow["margin"]), ("r_bar", rb)):
            fin[k_].append(v2)
        log(f"extract: prompt {p}: T={T}, forward {t_fwd:.1f} s, E1 worst {max(r['attn_rel'] for r in e1_rows if r['prompt'] == p):.2e}, "
            f"E3 logits {dev_lg / sc_lg:.2e}, t1 {prow['t1_str']!r} t2 {prow['t2_str']!r} margin {prow['margin']:.3f}, "
            f"E3 lin {prow['E3_lin_max_rel']:.2e}, cos {lm['cos_exact']:.4f}")
    rows_ids = np.array(sorted(fin["lm_rows"]))
    np.savez(CACHE / "final.tmp.npz", g_f=gf, eps_f=np.float64(eps_f), lm_ids=rows_ids,
             lm_rows=np.stack([fin["lm_rows"][t] for t in rows_ids]), c_m=np.stack(fin["c_m"]),
             grad_exact=np.stack(fin["grad_exact"]), xbar=np.stack(fin["xbar"]), t1=np.array(fin["t1"]),
             t2=np.array(fin["t2"]), margin=np.array(fin["margin"]), r_bar=np.array(fin["r_bar"]), pos=np.int64(POS),
             scale=np.float64(scale), mscale=np.float64(mscale))
    os.replace(CACHE / "final.tmp.npz", CACHE / "final.npz")
    peak = proc.memory_info().rss / 2 ** 30
    import resource

    out = {"model": MODEL_ID, "revision": getattr(cfg, "_commit_hash", None), "dtype_forward": dtype,
           "dtype_cache": "float64", "attn_implementation": "eager", "mxfp4": "dequantized on CPU (bf16), then cast",
           "config": {k: getattr(cfg, k, None) for k in ("num_hidden_layers", "hidden_size", "num_attention_heads",
                                                         "num_key_value_heads", "head_dim", "sliding_window", "rms_norm_eps",
                                                         "vocab_size", "rope_theta")},
           "rope_scaling": getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None),
           "layer_types": list(cfg.layer_types), "score_scale": scale, "yarn_attention_factor": mscale,
           "windows": windows, "prompts": prompts, "E1": e1_rows,
           "E1_ok": all(r["ok"] for r in e1_rows), "E1_worst_rel": max(r["attn_rel"] for r in e1_rows),
           "E1_worst_probs_abs": max(r["probs_abs"] for r in e1_rows),
           "window_binds_any": any(r["window_binds"] for r in e1_rows),
           "E3_logits_ok": all(r["E3_logits_ok"] for r in prompts), "E3_lin_ok": all(r["E3_lin_ok"] for r in prompts),
           "E3_top2_same": all(r["top2_numpy_same"] for r in prompts),
           "E3_worst_logits_rel": max(r["E3_logits_rel"] for r in prompts),
           "E3_worst_lin_rel": max(r["E3_lin_max_rel"] for r in prompts),
           "load_s": t_load, "rss_after_load_gb": rss_load, "rss_end_gb": peak,
           "ru_maxrss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
           "wall_s": time.perf_counter() - t0, "env": env_info(), "git": git_commit(),
           "torch": torch.__version__, "transformers": __import__("transformers").__version__}
    del model, Wlm
    gc.collect()
    write_json_atomic(rec_path, jsonable(out))
    _W.clear(), _P.clear(), _F.clear(), _PROJ.clear()
    if not (out["E1_ok"] and out["E3_logits_ok"] and out["E3_lin_ok"] and out["E3_top2_same"]):
        raise SystemExit(f"gpt-oss extraction checks failed: E1 {out['E1_ok']} ({out['E1_worst_rel']:.2e}), E3 logits "
                         f"{out['E3_logits_ok']}, E3 linear {out['E3_lin_ok']}, top-2 {out['E3_top2_same']}")
    return out


# ================================================================================================ cache (workers)
_W: dict = {}
_P: dict = {}
_F: dict = {}


def _need_cache() -> Path:
    """The cache for the units; never extracts (an extraction is started only by an explicit :func:`prepare`, e.g. by
    ``experiments/gpt_oss.py`` before its parts)."""
    c = model_cache()
    if not (c / "final.npz").exists():
        raise SystemExit(f"gpt-oss cache {c} not found or incomplete: run `python experiments/gpt_oss.py --part calib` (it "
                         "extracts once) or set AIMP_CACHE_DIR to a complete cache")
    return c


def layer_weights(l: int) -> dict:
    """The cached float64 attention weights of layer ``l`` (the two most recent layers are kept per process)."""
    if l not in _W:
        if len(_W) >= 2:
            _W.pop(next(iter(_W)))
        _W[l] = dict(np.load(_need_cache() / f"weights_L{l:02d}.npz"))
    return _W[l]


def prompt_data(p: int) -> dict:
    if p not in _P:
        a = np.load(_need_cache() / f"prompt{p}.npz")
        _P[p] = {k: a[k] for k in ("ids", "z", "R", "merged", "attn", "x", "xf")}
    return _P[p]


def final_data() -> dict:
    if not _F:
        a = np.load(_need_cache() / "final.npz")
        _F.update({k: a[k] for k in a.files})
    return _F


def n_tokens(p: int) -> int:
    return int(prompt_data(p)["ids"].shape[0])


def head_weights(l: int, h: int) -> dict:
    """``W_Q, W_K, W_V`` (2880 x 64, rows = inputs, GPT-2's convention), ``b_Q, b_K, b_V`` (64), ``W_O`` (64 x 2880) of query
    head ``h`` and its key-value group ``g = h // 8``; the sink logit and the score scale."""
    W = layer_weights(l)
    g = h // GROUP
    qs, ks = slice(h * DH, (h + 1) * DH), slice(g * DH, (g + 1) * DH)
    return {"W_Q": W["Wq"][qs].T, "W_K": W["Wk"][ks].T, "W_V": W["Wv"][ks].T, "b_Q": W["bq"][qs], "b_K": W["bk"][ks],
            "b_V": W["bv"][ks], "W_O": W["Wo"][:, qs].T, "sink": float(W["sinks"][h]), "scale": float(W["scale"]),
            "kv": g, "window": int(W["window"])}


def head_forward_f64(l: int, h: int, p: int, pos: int) -> dict:
    """The head at query position ``pos`` in float64 from the cached normalized tokens and weights: the pattern
    over the keys ``0..pos`` and the sink (``pos + 2``), the output ``y = W_O^T sum_j a_j v_j`` (2880) and its 64-dimensional
    head output (the corresponding slice of the ``o_proj`` input)."""
    hw = head_weights(l, h)
    P = prompt_data(p)
    z = P["z"][l][:pos + 1]
    R = P["R"]
    q = R[pos] @ (hw["W_Q"].T @ z[pos] + hw["b_Q"])
    k = np.einsum("tab,tb->ta", R[:pos + 1], z @ hw["W_K"] + hw["b_K"])
    v = z @ hw["W_V"] + hw["b_V"]
    s = np.r_[hw["scale"] * (k @ q), hw["sink"]]
    if hw["window"] > 0 and pos + 1 > hw["window"]:
        raise ValueError("the sliding window binds; not implemented (it does not bind for the five prompts)")
    a = np.exp(s - s.max())
    a /= a.sum()
    o = a[:pos + 1] @ v
    return {"a": a, "s": s, "head_out": o, "y": o @ hw["W_O"]}


# ================================================================================================ units
_PROJ: dict = {}


def head_projection(l: int, h: int):
    from aimp import gpt2

    if (l, h) not in _PROJ:
        hw = head_weights(l, h)
        if len(_PROJ) > 16:
            _PROJ.pop(next(iter(_PROJ)))
        _PROJ[(l, h)] = (hw, gpt2.projection(hw["W_Q"], hw["W_K"], hw["W_V"], hw["b_Q"], hw["b_K"], hw["b_V"]))
    return _PROJ[(l, h)]


def reduced_head(W_O: np.ndarray, scale: float, sink: float, name: str) -> HeadSpec:
    """``M_p`` (194 x 194): ``M_p[q, k] = scale I``, ``M_p[e, f] = s_sink``; ``N_p`` (2880 x 194) = ``[0, 0, W_O^T, 0, 0]``."""
    M = np.zeros((DP, DP))
    M[:DH, DH:2 * DH] = scale * np.eye(DH)
    M[IE, IF] = sink
    N = np.zeros((W_O.shape[1], DP))
    N[:, 2 * DH:3 * DH] = W_O.T
    return HeadSpec(M=M, N=N, d=DP, d_k=DH, d_v=DH, name=name)


def spec_vector(l: int, h: int, p: int, pos: int, spec: str) -> np.ndarray:
    """``"a"``: the direction of the head's nominal output ``ybar / ||ybar||`` (``ybar`` from the float64 re-implementation);
    ``"lm"``: ``c = -c_m`` of the prompt (position 4 only; unnormalized, in logits)."""
    if spec == "a":
        y = head_forward_f64(l, h, p, pos)["y"]
        return y / np.linalg.norm(y)
    if spec == "lm":
        F = final_data()
        if pos != int(F["pos"]):
            raise ValueError(f"spec lm is defined at position {int(F['pos'])}, not {pos}")
        return -np.asarray(F["c_m"][p], float)
    raise ValueError(spec)


def make_instance(l: int, h: int, p: int, pos: int, rho_rel: float, spec: str = "a", fixed_query: bool = False) -> Instance:
    """The gpt-oss unit (module docstring): tokens ``0..pos`` with ellipsoids ``{D_t (P zbar_t + b) + D_t E w : ||w|| <=
    rho_rel ||zbar_t||}`` (plus ``e = 1, f = 0``), the sink as token ``pos + 1`` (a point), keys ``0..pos+1``, query ``pos``."""
    hw, pr = head_projection(l, h)
    P = prompt_data(p)
    z = P["z"][l]
    R = P["R"]
    head = reduced_head(hw["W_O"], hw["scale"], hw["sink"], name=f"L{l}H{h}")
    sets = {}
    for t in range(pos + 1):
        c0 = pr.P @ z[t] + pr.b
        cen = np.zeros(DP)
        Et = np.zeros((DP, pr.E.shape[1]))
        cen[:DH], cen[DH:2 * DH], cen[2 * DH:3 * DH] = R[pos] @ c0[:DH], R[t] @ c0[DH:2 * DH], c0[2 * DH:]
        cen[IE] = 1.0
        Et[:DH], Et[DH:2 * DH], Et[2 * DH:3 * DH] = R[pos] @ pr.E[:DH], R[t] @ pr.E[DH:2 * DH], pr.E[2 * DH:]
        sets[t] = S.ellipsoid(cen, Et, rho_rel * float(np.linalg.norm(z[t])))
    sk = np.zeros(DP)
    sk[IF] = 1.0
    sets[pos + 1] = S.point(sk)
    c = spec_vector(l, h, p, pos, spec)
    meta = {"model": MODEL_NAME, "layer": l, "head": h, "kv": hw["kv"], "prompt": p, "pos": pos, "spec": spec,
            "name": f"L{l}H{h}", "rho_rel": rho_rel, "r": pr.r, "sink_token": pos + 1, "sink": hw["sink"],
            "scale": hw["scale"]}
    inst = Instance(head=head, keys=list(range(pos + 2)), query=pos, sets=sets, c=np.asarray(c, float), meta=meta)
    if fixed_query:
        from aimp.exact_fq import fix_query

        inst = fix_query(inst)
    return inst


def fq_instance(l: int, h: int, p: int, pos: int, rho_rel: float, spec: str = "a") -> Instance:
    return make_instance(l, h, p, pos, rho_rel, spec, fixed_query=True)


def e2_check(inst: Instance) -> dict:
    """Check E2: the unit's instance (with ``M_j``, the sink and the reduced coordinates) reproduces the head's nominal
    pattern (keys and sink) and output (2880; and ``c^T y``) of the float64 re-implementation, to 1e-10."""
    from aimp.heads import nominal

    m = inst.meta
    ref = head_forward_f64(m["layer"], m["head"], m["prompt"], m["pos"])
    nm = nominal(inst)
    Zc = inst.centers()
    y = inst.head.N @ (nm["a"] @ Zc)
    s = np.array([Zc[inst.query] @ inst.head.M @ Zc[j] for j in inst.keys])
    out = {"E2_pattern_abs": float(np.max(np.abs(nm["a"] - ref["a"]))),
           "E2_scores_abs": float(np.max(np.abs(s - ref["s"]))) / max(1.0, float(np.max(np.abs(ref["s"])))),
           "E2_output_rel": float(np.max(np.abs(y - ref["y"]))) / max(1e-300, float(np.max(np.abs(ref["y"])))),
           "E2_phi_rel": abs(float(nm["phi"]) - float(inst.c @ ref["y"])) / max(1.0, abs(float(inst.c @ ref["y"]))),
           "sink_weight": float(ref["a"][-1])}
    out["E2_ok"] = bool(max(out["E2_pattern_abs"], out["E2_scores_abs"], out["E2_output_rel"], out["E2_phi_rel"]) <= E2_TOL)
    return out


def _reduce(E: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    U, s, _ = np.linalg.svd(E, full_matrices=False)
    k = int(np.sum(s > rtol * s[0])) if s.size and s[0] > 0 else 0
    return U[:, :k] * s[:k]


def score_intervals_roles(inst: Instance, threads=None, method: str = "shor") -> dict:
    """Certified score intervals of a gpt-oss unit: the role-reduced Shor bound of ``gpt2.score_intervals_roles`` with each
    token's own (rotated) shape matrix — key ``j != i``: ``q`` in ``{c_i[q] + E_i[q] w}``, ``k`` in ``{c_j[k] + E_j[k] w'}``;
    ``j = i``: ``(q, k)`` in ``{c_i[q, k] + E_i[q, k] w}`` (Shor is invariant under the invertible per-token maps, so these are the
    Shor bounds of the problem with ``M_j``); the sink: ``[s_sink, s_sink]`` (exact). ``method="cs"``: 01 L5 on the instance."""
    from aimp import bounds
    from aimp.sdp_mosek import shor_score

    if method == "cs":
        return bounds.score_intervals(inst, "cs")
    hd = inst.head
    dk = hd.d_k
    i = inst.query
    Si = inst.sets[i]
    sink = inst.meta["sink_token"]
    M1 = hd.M[:dk, dk:2 * dk]
    M2 = np.zeros((2 * dk, 2 * dk))
    M2[:dk, dk:] = M1
    Eq = _reduce(Si.E[:dk]) if Si.rho > 0 else None
    s_lo, s_hi, info = [], [], {}
    for j in inst.keys:
        Sj = inst.sets[j]
        if j == sink:
            v = float(Si.center @ hd.M @ Sj.center)
            lo, hi, inf = v, v, {"sink": True}
        elif j == i:
            tiny = Instance(head=HeadSpec(M=M2, N=np.zeros((1, 2 * dk)), d=2 * dk, d_k=dk, d_v=1), keys=[0], query=0,
                            sets={0: S.ellipsoid(Si.center[:2 * dk], _reduce(Si.E[:2 * dk]) if Si.rho > 0 else Si.E[:2 * dk],
                                                 Si.rho)}, c=np.zeros(1))
            lo, hi, inf = shor_score(tiny, 0, threads=threads) if Si.rho > 0 else (*bounds._linear_interval(tiny, 0), {})
        else:
            Ek = _reduce(Sj.E[dk:2 * dk]) if Sj.rho > 0 else Sj.E[dk:2 * dk]
            tiny = Instance(head=HeadSpec(M=M1, N=np.zeros((1, dk)), d=dk, d_k=dk, d_v=1), keys=[1], query=0,
                            sets={0: S.ellipsoid(Si.center[:dk], Eq if Eq is not None else Si.E[:dk], Si.rho),
                                  1: S.ellipsoid(Sj.center[dk:2 * dk], Ek, Sj.rho)}, c=np.zeros(1))
            lin = bounds._linear_interval(tiny, 1)
            if lin is not None:
                lo, hi, inf = lin[0], lin[1], {"linear_exact": True}
            else:
                lo, hi, inf = shor_score(tiny, 1, threads=threads)
        s_lo.append(lo)
        s_hi.append(hi)
        info[str(j)] = inf
    return {"method": "shor_roles_gptoss", "s_lo": np.array(s_lo), "s_hi": np.array(s_hi), "info": info}


def nominal_value_range(inst: Instance) -> dict:
    """The nominal value range over the keys, with the sink (a key with value 0; ``gpt2.nominal_value_range``) and without it."""
    g = np.array([inst.c @ inst.head.N @ inst.sets[j].center for j in inst.keys])
    real = g[:-1]
    return {"range": float(g.max() - g.min()), "range_no_sink": float(real.max() - real.min())}


def canonical_angles(fq: Instance, var_keys: list, Zp: np.ndarray) -> dict:
    """The canonical angles of a worst-case input of a fixed-query unit (``benchmark/SCHEMA.md`` with RoPE): key ``j`` moves in
    the plane of ``alpha_j = scale W_K R_j^T R_i q_i`` (its score changes by ``alpha_j . delta_j``; the plane depends on ``j``
    through ``R_j``) and ``beta = W_V W_O c`` (its value along ``c``); ``e1_j = alpha_j / ||alpha_j||``, ``e2_j`` = the
    normalized component of ``beta`` orthogonal to ``e1_j``. ``Zp`` holds the worst case in reduced coordinates; the
    2880-dimensional perturbation is ``delta_j = V w_j`` (``w_j`` the ellipsoid coordinates)."""
    m = fq.meta
    l, h, p, pos = m["layer"], m["head"], m["prompt"], m["pos"]
    hw, pr = head_projection(l, h)
    P = prompt_data(p)
    z, R = P["z"][l], P["R"]
    qr = R[pos] @ (hw["W_Q"].T @ z[pos] + hw["b_Q"])
    beta = hw["W_V"] @ (hw["W_O"] @ fq.c)
    th, nrm, off = [], [], []
    for j in var_keys:
        alpha = hw["scale"] * (hw["W_K"] @ (R[j].T @ qr))
        e1 = alpha / np.linalg.norm(alpha)
        b2 = beta - (beta @ e1) * e1
        e2 = b2 / np.linalg.norm(b2)
        st = fq.sets[j]
        w, *_ = np.linalg.lstsq(st.E, Zp[j] - st.center, rcond=None)
        dlt = pr.Vt.T @ w
        a_, b_ = float(dlt @ e1), float(dlt @ e2)
        th.append(math.atan2(b_, a_))
        nrm.append(float(np.linalg.norm(dlt)))
        off.append(float(np.linalg.norm(dlt - a_ * e1 - b_ * e2)))
    return {"theta": th, "norm": nrm, "off_plane": off}
