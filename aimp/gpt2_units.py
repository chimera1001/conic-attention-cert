"""GPT-2 small units: cached weights and prompt activations, and the certification instance of one unit.

A *unit* is (layer ``l``, head ``h``, prompt ``p``, query position ``pos``, relative radius ``rho_rel``). Its token sets
are balls of radius ``rho_j = rho_rel ||zbar_j||`` around the post-normalization tokens ``zbar_j`` (the output of
``ln_1`` of block ``l``) for ``j = 0..pos``, mapped to the head's reduced coordinates ``(q, k, v)`` (Section 5 of the
paper; :func:`aimp.gpt2.phase3_instance`). The specification ``c`` is the direction of the head's nominal output
(its contribution ``sum_j a_j v_j W_O`` to the residual stream at ``pos``). With ``fixed_query=True`` the query token
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


def make_instance(l: int, h: int, p: int, pos: int, rho_rel: float, fixed_query: bool = False):
    """The unit's instance in reduced coordinates, with the specification ``c`` = direction of the nominal head
    output at ``pos``."""
    from aimp import gpt2

    hw, pr = head_data(l, h)
    P = ctx()["P"][p]
    z = P["z"][l]
    ybar = P["merged"][l][pos, h * 64:(h + 1) * 64] @ hw["W_O"]
    c = ybar / np.linalg.norm(ybar)
    meta = {"layer": l, "head": h, "prompt": p, "pos": pos, "spec": "a", "name": f"L{l}H{h}"}
    inst = gpt2.phase3_instance(z, hw, pos, rho_rel, c, pr, meta)
    if fixed_query:
        from aimp.exact_fq import fix_query

        inst = fix_query(inst)
    return inst
