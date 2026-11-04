"""GPT-2 small units: cached weights and prompt activations, and the certification instance of one unit.

A *unit* is (layer ``l``, head ``h``, prompt ``p``, query position ``pos``, relative radius ``rho_rel``). Its token sets
are balls of radius ``rho_j = rho_rel ||zbar_j||`` around the post-normalization tokens ``zbar_j`` (the output of
``ln_1`` of block ``l``) for ``j = 0..pos``, mapped to the head's reduced coordinates ``(q, k, v)`` (Section 5 of the
paper; :func:`aimp.gpt2.phase3_instance`). The specification ``c`` is the direction of the head's nominal output
(its contribution ``sum_j a_j v_j W_O`` to the residual stream at ``pos``; ``spec="a"``, the paper's), or, with
``spec="lm"`` (position 4), the logit-margin specification ``c = -c_m``: ``c_m^T y`` is the change of the margin between
the model's two top next-token logits at the position caused by a change ``y`` of the head's output, through the direct
path with the scale of the final normalization frozen (:func:`prepare_lm`). With ``fixed_query=True`` the query token
(and key ``pos``) is held at its nominal value (:func:`aimp.exact_fq.fix_query`).

:func:`prepare` downloads ``gpt2`` from the Hugging Face hub (once), extracts every prompt's activations with hooks,
checks the numpy re-implementation against the model to ``1e-10`` (:func:`aimp.gpt2.gate`) and caches the arrays in
``cache_dir()`` (``$AIMP_CACHE_DIR``, default ``<repository>/cache``).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from aimp.io import cache_dir, env_info, results_dir, write_json_atomic  # noqa: F401

LAYERS = tuple(range(12))
HEADS = tuple(range(12))
N_PROMPTS = 5
_KEEP = ("ln1_g", "ln1_b", "ln1_eps", "W_attn", "b_attn", "W_proj", "b_proj")


def positions(T: int) -> list[int]:
    """The query positions of the paper: 4, 8, 12 and the last token (counted from zero)."""
    return [4, 8, 12, T - 1]


def unit_name(l: int, h: int, p: int, pos: int, rho_rel: float) -> str:
    return f"L{l:02d}H{h:02d}_P{p}_i{pos:02d}_r{rho_rel:g}"


def unit_seed(l: int, h: int, p: int, pos: int, rho_rel: float) -> int:
    """The per-unit seed of the multistart ascents."""
    return int(1_000_000 * l + 10_000 * h + 1000 * p + 10 * pos + round(1000 * rho_rel))


def prepare(force: bool = False) -> dict:
    """Cache the weights (attention part of every block) and the activations of the five prompts; run the
    extraction gate. Returns the gate record (also written to ``cache_dir()/gate.json``)."""
    from aimp import gpt2

    cd = cache_dir()
    gate_path = cd / "gate.json"
    if gate_path.exists() and not force and (cd / "weights.npz").exists() and all(
            (cd / f"prompt{p}.npz").exists() for p in range(N_PROMPTS)):
        return json.loads(gate_path.read_text())
    t0 = time.perf_counter()
    model, tok = gpt2.load_model()
    W = gpt2.all_weights(model)
    cd.mkdir(parents=True, exist_ok=True)
    np.savez(cd / "weights.npz", **{f"L{l}_{k}": np.asarray(W[l][k]) for l in range(12) for k in _KEEP})
    rows = []
    for p, text in enumerate(gpt2.load_prompts()):
        ext = gpt2.extract(model, tok, text)
        g = gpt2.gate(ext, W)
        rows.append({"prompt": p, "text": text, "T": int(len(ext["ids"])), "tokens": ext["tokens"],
                     "passed": g["passed"], "worst": g["worst"]})
        np.savez(cd / f"prompt{p}.npz", x=ext["x"], z=ext["z"], merged=ext["merged"], A=ext["A"], ids=ext["ids"])
    out = {"passed": all(r["passed"] for r in rows), "tol": gpt2.GATE_TOL, "prompts": rows,
           "worst": {k: max(r["worst"][k] for r in rows) for k in rows[0]["worst"]},
           "model": "gpt2 (Hugging Face), float64, eager attention", "env": env_info(),
           "wall_s": time.perf_counter() - t0}
    write_json_atomic(gate_path, out)
    if not out["passed"]:
        raise SystemExit(f"extraction gate failed: {out['worst']}")
    return out


LM_POS = 4  # the logit-margin specification is built at query position 4
LM_L0_TOL = 1e-10  # numpy logits at the position against the model's (absolute)
LM_L1_TOL = 1e-6  # frozen-scale logit difference against c_m^T delta (relative)
LM_L2_TOL = 1e-10  # c^T ybar against the head's direct contribution in the forward pass (relative to max(1, |.|))


def _lm_l1(model, xbar, sig, t1, t2, c_m, deltas) -> dict:
    """``(L(xbar + delta) - L(xbar))`` with the model's own ``ln_f`` (scale frozen at ``sigma_bar``) and ``lm_head``
    against ``c_m^T delta``; the change with the unfrozen ``ln_f`` is recorded too."""
    import torch

    lnf, head = model.transformer.ln_f, model.lm_head

    def frozen(x):
        lg = head((x - x.mean()) / sig * lnf.weight + lnf.bias)
        return lg[t1] - lg[t2]

    def full(x):
        lg = head(lnf(x[None, :]))[0]
        return lg[t1] - lg[t2]

    rows = []
    with torch.no_grad():
        xb = torch.as_tensor(xbar, dtype=torch.float64)
        f0, u0 = float(frozen(xb)), float(full(xb))
        for dlt in deltas:
            x = xb + torch.as_tensor(dlt, dtype=torch.float64)
            pred = float(c_m @ dlt)
            ch_f, ch_u = float(frozen(x)) - f0, float(full(x)) - u0
            rows.append({"predicted": pred, "frozen_change": ch_f, "rel_err": abs(ch_f - pred) / max(abs(pred), 1e-300),
                         "unfrozen_change": ch_u})
    return {"rows": rows, "max_rel_err": max(r["rel_err"] for r in rows)}


def prepare_lm(force: bool = False) -> dict:
    """The logit-margin specification of every prompt at position 4, cached in ``cache_dir()/lm_spec.npz`` (record
    ``cache_dir()/lm_gate.json``). Checks: L0 the numpy logits at the position (``ln_f`` and ``wte`` applied to the hooked
    final residual stream, and the whole numpy forward pass from the embeddings) equal the model's to ``1e-10``; L1 for
    random ``delta`` of norm ``1e-3 ||xbar||`` the frozen-scale logit difference computed with the model's own ``ln_f``
    and ``lm_head`` changes by ``c_m^T delta`` to ``1e-6`` relative; L2 ``c^T ybar`` of every unit's instance equals the
    head's direct contribution in the forward pass (all 144 heads x 5 prompts). Raises ``SystemExit`` if a check fails."""
    from aimp import gpt2
    from aimp.heads import nominal

    cd = cache_dir()
    gpath, npath = cd / "lm_gate.json", cd / "lm_spec.npz"
    if gpath.exists() and npath.exists() and not force:
        return json.loads(gpath.read_text())
    ctx()  # the cached activations of prepare()
    t0 = time.perf_counter()
    model, tok = gpt2.load_model()
    W = gpt2.all_weights(model)
    F = gpt2.final_weights(model)
    rng = np.random.default_rng(7_000_001)
    rows, arr, merged = [], {k: [] for k in ("c_m", "t1", "t2", "logit1", "logit2", "margin", "sigma_bar", "pos")}, []
    for p, text in enumerate(gpt2.load_prompts()):
        ext = gpt2.extract(model, tok, text)
        lg = ext["logits"]
        dev_h = float(np.max(np.abs(gpt2.final_logits(ext["out"][11], F)[LM_POS] - lg[LM_POS])))
        dev_c = float(np.max(np.abs(gpt2.numpy_logits(ext["x"][0], W, F)[LM_POS] - lg[LM_POS])))
        order = np.argsort(-lg[LM_POS], kind="stable")
        t1, t2 = int(order[0]), int(order[1])
        xbar = ext["out"][11][LM_POS].copy()
        lm = gpt2.lm_spec(xbar, F, t1, t2)
        deltas = []
        for _ in range(3):  # nine deltas in the output space of three random heads (W_O^T u), one generic direction
            l_, h_ = int(rng.integers(12)), int(rng.integers(12))
            WO = W[l_]["W_proj"][h_ * 64:(h_ + 1) * 64, :]
            for _ in range(3):
                v = WO.T @ rng.standard_normal(64)
                deltas.append(v * 1e-3 * np.linalg.norm(xbar) / np.linalg.norm(v))
        v = rng.standard_normal(xbar.shape[0])
        deltas.append(v * 1e-3 * np.linalg.norm(xbar) / np.linalg.norm(v))
        l1 = _lm_l1(model, xbar, lm["sigma_bar"], t1, t2, lm["c_m"], deltas)
        rows.append({"prompt": p, "t1": t1, "t2": t2, "t1_str": tok.decode([t1]), "t2_str": tok.decode([t2]),
                     "logit_t1": float(lg[LM_POS, t1]), "logit_t2": float(lg[LM_POS, t2]),
                     "margin": float(lg[LM_POS, t1] - lg[LM_POS, t2]), "sigma_bar": lm["sigma_bar"],
                     "c_m_norm": float(np.linalg.norm(lm["c_m"])), "cos_c_m_exact": lm["cos_exact"],
                     "L0_hooked_abs": dev_h, "L0_chain_abs": dev_c, "L1_max_rel_err": l1["max_rel_err"], "L1": l1["rows"]})
        for k, v_ in (("c_m", lm["c_m"]), ("t1", t1), ("t2", t2), ("logit1", lg[LM_POS, t1]), ("logit2", lg[LM_POS, t2]),
                      ("margin", lg[LM_POS, t1] - lg[LM_POS, t2]), ("sigma_bar", lm["sigma_bar"]), ("pos", LM_POS)):
            arr[k].append(v_)
        merged.append(ext["merged"])
    cd.mkdir(parents=True, exist_ok=True)
    tmp = cd / "lm_spec.tmp.npz"
    np.savez(tmp, **{k: np.asarray(v) for k, v in arr.items()})
    os.replace(tmp, npath)
    _LM.clear()
    l2 = 0.0
    for l in LAYERS:
        for h in HEADS:
            hw, _ = head_data(l, h)
            for p in range(N_PROMPTS):
                inst = make_instance(l, h, p, LM_POS, 0.01, spec="lm")
                direct = float(lm_vector(p) @ (merged[p][l][LM_POS, h * 64:(h + 1) * 64] @ hw["W_O"]))
                l2 = max(l2, abs(nominal(inst)["phi"] - direct) / max(1.0, abs(direct)))
    out = {"prompts": rows, "L0_ok": all(r["L0_hooked_abs"] <= LM_L0_TOL and r["L0_chain_abs"] <= LM_L0_TOL for r in rows),
           "L1_ok": all(r["L1_max_rel_err"] <= LM_L1_TOL for r in rows), "L2_max_rel": l2, "L2_ok": bool(l2 <= LM_L2_TOL),
           "tolerances": {"L0": LM_L0_TOL, "L1": LM_L1_TOL, "L2": LM_L2_TOL}, "env": env_info(),
           "wall_s": time.perf_counter() - t0}
    write_json_atomic(gpath, out)
    if not (out["L0_ok"] and out["L1_ok"] and out["L2_ok"]):
        raise SystemExit(f"logit-margin checks failed: L0 {out['L0_ok']} L1 {out['L1_ok']} L2 {out['L2_ok']}")
    return out


_LM: dict = {}


def lm_vector(p: int) -> np.ndarray:
    """``c = -c_m`` of prompt ``p`` (unnormalized, in logits), from ``cache_dir()/lm_spec.npz`` (built by
    :func:`prepare_lm` when missing)."""
    if not _LM:
        f = cache_dir() / "lm_spec.npz"
        if not f.exists():
            prepare_lm()
        a = np.load(f)
        _LM.update({"c_m": a["c_m"], "pos": a["pos"], "margin": a["margin"], "t1": a["t1"], "t2": a["t2"]})
    return -np.asarray(_LM["c_m"][p], float)


_CTX: dict = {}


def ctx() -> dict:
    """Per-process cache of weights, prompt arrays and head projections (calls :func:`prepare` when needed)."""
    if not _CTX:
        cd = cache_dir()
        if not (cd / "weights.npz").exists():
            prepare()
        w = np.load(cd / "weights.npz")
        _CTX["W"] = [{k: w[f"L{l}_{k}"] for k in _KEEP} for l in range(12)]
        _CTX["P"] = []
        for p in range(N_PROMPTS):
            a = np.load(cd / f"prompt{p}.npz")
            _CTX["P"].append({k: a[k] for k in ("z", "merged", "ids")})
        _CTX["proj"] = {}
    return _CTX


def prompt_length(p: int) -> int:
    return int(len(ctx()["P"][p]["ids"]))


def head_data(l: int, h: int) -> tuple[dict, object]:
    """Head weights and the head-subspace projection (``P = [W_Q^T; W_K^T; W_V^T]`` and its thin SVD)."""
    from aimp import gpt2

    C = ctx()
    if (l, h) not in C["proj"]:
        hw = gpt2.head_weights(C["W"][l], h)
        C["proj"][(l, h)] = (hw, gpt2.projection(hw["W_Q"], hw["W_K"], hw["W_V"], hw["b_Q"], hw["b_K"], hw["b_V"]))
        if len(C["proj"]) > 32:
            C["proj"].pop(next(iter(C["proj"])))
    return C["proj"][(l, h)]


def make_instance(l: int, h: int, p: int, pos: int, rho_rel: float, fixed_query: bool = False, spec: str = "a",
                  model: str = "gpt2"):
    """The unit's instance in reduced coordinates, with the specification ``c`` = direction of the nominal head
    output at ``pos`` (``spec="a"``) or the logit-margin specification ``c = -c_m`` of the prompt (``spec="lm"``,
    position 4 only). ``model="gpt-oss-20b"``: the gpt-oss unit of :func:`aimp.gptoss.make_instance` (query head ``h`` of
    layer ``l``, rotary maps folded into the token sets, the attention sink as a key with a fixed score and value zero);
    the default ``"gpt2"`` is unchanged."""
    if model != "gpt2":
        from aimp import gptoss

        if model != gptoss.MODEL_NAME:
            raise ValueError(f"unknown model {model!r}")
        return gptoss.make_instance(l, h, p, pos, rho_rel, spec, fixed_query=fixed_query)
    from aimp import gpt2

    hw, pr = head_data(l, h)
    P = ctx()["P"][p]
    z = P["z"][l]
    if spec == "a":
        ybar = P["merged"][l][pos, h * 64:(h + 1) * 64] @ hw["W_O"]
        c = ybar / np.linalg.norm(ybar)
    elif spec == "lm":
        if pos != LM_POS:
            raise ValueError(f"the logit-margin specification is defined at position {LM_POS}, not {pos}")
        c = lm_vector(p)
    else:
        raise ValueError(f"unknown specification {spec!r}")
    meta = {"layer": l, "head": h, "prompt": p, "pos": pos, "spec": spec, "name": f"L{l}H{h}"}
    inst = gpt2.phase3_instance(z, hw, pos, rho_rel, c, pr, meta)
    if fixed_query:
        from aimp.exact_fq import fix_query

        inst = fix_query(inst)
    return inst
