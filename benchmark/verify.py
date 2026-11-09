"""Verify the fixed-query worst cases stored in this directory (see SCHEMA.md): GPT-2 small (``units.csv``,
``angles.csv.gz``, ``prompts.txt`` here) and gpt-oss-20b (the same files in ``gpt-oss-20b/``).

For every selected unit the script rebuilds the instance from the Hugging Face weights and the prompt alone: the
post-normalization tokens ``zbar_j`` read by the head (the output of ``ln_1`` of the block for GPT-2, of the layer's
``input_layernorm`` for gpt-oss-20b), the radii ``rho_j = rho_rel ||zbar_j||``, the fixed query, the specification ``c``
(direction of the head's nominal output) and the canonical frame of every key (one plane for GPT-2; for gpt-oss-20b the
plane of key ``j`` depends on ``j`` through its rotary map). It then rebuilds the maximizing and the minimizing input from
the stored angles and checks that

1. every perturbation lies in its ball (``||delta_j|| <= rho_j``) and the stored radii follow the radius rule;
2. the nominal output equals the stored one;
3. the head's output along ``c`` at each worst-case input, computed twice -- by a direct numpy evaluation of the head
   in float64 and by the model's own code (GPT-2: the Hugging Face model with the perturbed tokens injected at the output
   of ``ln_1``; gpt-oss-20b: the layer's Hugging Face attention module, copied to float64, on the perturbed tokens) --
   lies in the stored certified bracket (of the maximum, resp. of the minimum).

The tolerance of check 3 is ``1e-9 (1 + |bound|)``. gpt-oss-20b runs in float32 (as for the benchmark), so its normalized
tokens are reproduced bit for bit only with the same float32 arithmetic: the thread count of the benchmark (10, the
default of ``--threads`` for this model) on a CPU of the same type. The script measures the reproduction on every unit
(the radius rule and the nominal output); where they are not reproduced to ``1e-10``, check 3 uses ``TOL_F32 (1 + |bound|)``
instead, and the summary reports the units checked at that level separately. For GPT-2 (float64) both must hold to
``1e-10``.

Only numpy, torch (CPU) and transformers are needed; the ``aimp`` package is not used.

Usage::

    python benchmark/verify.py --sample 200 --seed 0                       # GPT-2 small, 200 random units
    python benchmark/verify.py --rho 0.01                                  # all 2,880 units of one radius
    python benchmark/verify.py --unit L05H01_P0_i04_r0.01
    python benchmark/verify.py --all
    python benchmark/verify.py --model gpt-oss-20b --sample 200 --seed 0   # gpt-oss-20b (about 85 GB of memory)
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import gzip
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOL_REL = 1e-9  # a value inside [lo, hi] up to 1e-9 (1 + |bound|): double-precision rounding of two implementations
TOL_F32 = 1e-5  # gpt-oss-20b, when the float32 forward pass does not reproduce the stored tokens bit for bit
REPRO_TOL = 1e-10  # relative errors of the radius rule and of the nominal output up to which the tokens count as reproduced
RADIUS_TOL = 1e-12


class _Stop(Exception):
    pass


def load_units(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["unit_id"]: r for r in csv.DictReader(fh)}


def load_angles(path: Path, wanted: set) -> dict:
    out: dict = defaultdict(list)
    with gzip.open(path, "rt", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["unit_id"] in wanted:
                out[r["unit_id"]].append((int(r["key"]), float(r["rho_j"]), float(r["theta_max"]), float(r["theta_min"])))
    return out


def softmax(s: np.ndarray) -> np.ndarray:
    a = np.exp(s - s.max())
    return a / a.sum()


def rope_matrix(cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """The linear map of gpt-oss's rotary embedding at one position: the first and the second half of a 64-vector are
    rotated pairwise, ``(x1, x2) -> (x1 cos - x2 sin, x2 cos + x1 sin)`` (``cos``, ``sin``: the model's 32-entry tables at
    that position, which carry the YaRN attention factor)."""
    C, S = np.diag(cos), np.diag(sin)
    return np.block([[C, -S], [S, C]])


def head_output(z: np.ndarray, hw: dict, pos: int, R: np.ndarray | None = None) -> np.ndarray:
    """Head output at ``pos`` (its contribution to the residual stream, before the output bias that the heads share),
    causal mask. GPT-2: ``s_j = q . k_j / 8``. gpt-oss-20b (``R`` the rotary maps): ``s_j = (R_pos q) . (R_j k_j) / 8``,
    then the head's sink logit as one more entry of the softmax with value zero."""
    q = z[pos] @ hw["W_Q"] + hw["b_Q"]
    k = z[:pos + 1] @ hw["W_K"] + hw["b_K"]
    v = z[:pos + 1] @ hw["W_V"] + hw["b_V"]
    if R is not None:
        q = R[pos] @ q
        k = np.einsum("tab,tb->ta", R[:pos + 1], k)
    s = (k @ q) * hw["scale"]
    if "sink" in hw:
        s = np.r_[s, hw["sink"]]
    a = softmax(s)
    return (a[:pos + 1] @ v) @ hw["W_O"]


class GPT2:
    """``gpt2`` in float64 with hooks: ``ln_1`` outputs of every block, and the value of one head along ``c`` with the
    output of ``ln_1`` of one block replaced at chosen positions."""

    name, D, DK = "gpt2", 768, 64

    def __init__(self, threads: int):
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = GPT2TokenizerFast.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager", dtype=torch.float64).eval()

    def ids(self, text: str):
        return self.tok(text, return_tensors="pt")["input_ids"]

    def prompt(self, text: str) -> dict:
        ids = self.ids(text)
        z, hooks = [None] * len(self.model.transformer.h), []
        for l, blk in enumerate(self.model.transformer.h):
            hooks.append(blk.ln_1.register_forward_hook(
                lambda m, i, o, l=l: z.__setitem__(l, o[0].detach().numpy().astype(np.float64).copy())))
        try:
            with self.torch.no_grad():
                self.model(ids)
        finally:
            for hk in hooks:
                hk.remove()
        return {"ids": ids, "z": z, "R": None}

    def head_value(self, P: dict, l: int, h: int, pos: int, delta: dict, hw: dict, c: np.ndarray) -> float:
        """``c . (o_pos W_O)`` with ``o_pos`` head ``h``'s output at ``pos`` (the input of ``c_proj``), the ``ln_1``
        output of block ``l`` shifted by ``delta[j]`` at the positions ``j``."""
        blk = self.model.transformer.h[l]
        got = {}

        def ln_hook(m, i, o):
            o = o.clone()
            for j, dj in delta.items():
                o[0, j] += self.torch.as_tensor(dj, dtype=o.dtype)
            return o

        def proj_hook(m, i):
            got["merged"] = i[0][0, pos, h * self.DK:(h + 1) * self.DK].detach().numpy().astype(np.float64).copy()
            raise _Stop

        h1 = blk.ln_1.register_forward_hook(ln_hook)
        h2 = blk.attn.c_proj.register_forward_pre_hook(proj_hook)
        try:
            with self.torch.no_grad():
                self.model(P["ids"])
        except _Stop:
            pass
        finally:
            h1.remove()
            h2.remove()
        return float(c @ (got["merged"] @ hw["W_O"]))

    def head_weights(self, l: int, h: int) -> dict:
        D, DK = self.D, self.DK
        blk = self.model.transformer.h[l]
        Wa = blk.attn.c_attn.weight.detach().numpy().astype(np.float64)
        ba = blk.attn.c_attn.bias.detach().numpy().astype(np.float64)
        sl = slice(h * DK, (h + 1) * DK)
        return {"W_Q": Wa[:, :D][:, sl], "W_K": Wa[:, D:2 * D][:, sl], "W_V": Wa[:, 2 * D:][:, sl],
                "b_Q": ba[:D][sl], "b_K": ba[D:2 * D][sl], "b_V": ba[2 * D:][sl],
                "W_O": blk.attn.c_proj.weight.detach().numpy().astype(np.float64)[sl, :], "scale": 1.0 / math.sqrt(DK)}

    def release(self, layers: set) -> None:
        """GPT-2 stays loaded (its Hugging Face check reruns the model)."""


class GptOss:
    """``openai/gpt-oss-20b`` as for the benchmark: the MXFP4 experts dequantized on CPU, eager attention, forward passes in
    float32 (``input_layernorm`` outputs of every layer and the rotary tables, read by hooks). After the prompts, the
    attention modules of the layers needed are copied to float64 and the model is freed; the Hugging Face check calls the
    float64 copy of the layer's attention module on the perturbed normalized tokens."""

    name, D, DK, GROUP = "gpt-oss-20b", 2880, 64, 8
    MODEL_ID = "openai/gpt-oss-20b"
    REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"  # the revision of the benchmark

    def __init__(self, threads: int):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch
        from transformers import AutoTokenizer, GptOssForCausalLM, Mxfp4Config

        torch.set_num_threads(threads)
        torch.set_grad_enabled(False)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.MODEL_ID, revision=self.REVISION)
        model = GptOssForCausalLM.from_pretrained(self.MODEL_ID, revision=self.REVISION, dtype=torch.bfloat16,
                                                  attn_implementation="eager", device_map="cpu",
                                                  quantization_config=Mxfp4Config(dequantize=True))
        self.model = model.eval().float()
        self.attn: dict = {}
        self.weights: dict = {}

    def prompt(self, text: str) -> dict:
        torch = self.torch
        enc = self.tok(text, return_tensors="pt")
        layers = self.model.model.layers
        z, pe, hooks = [None] * len(layers), {}, []
        hooks.append(layers[0].register_forward_pre_hook(
            lambda m, a, k: pe.__setitem__("cs", k["position_embeddings"]), with_kwargs=True))
        for l, lay in enumerate(layers):
            hooks.append(lay.input_layernorm.register_forward_hook(
                lambda m, i, o, l=l: z.__setitem__(l, o[0].detach().to(torch.float64).numpy().copy())))
        try:
            with torch.no_grad():
                self.model(**enc, output_attentions=True, use_cache=False)
        finally:
            for hk in hooks:
                hk.remove()
        cos, sin = (t[0].detach().to(torch.float64).numpy().copy() for t in pe["cs"])
        return {"ids": enc["input_ids"], "z": z, "cos": cos, "sin": sin,
                "R": np.stack([rope_matrix(cos[t], sin[t]) for t in range(cos.shape[0])])}

    def release(self, layers: set) -> None:
        """Copy the attention modules (float64) and the head weights of ``layers``, then free the model."""
        for l in sorted(layers):
            at = self.model.model.layers[l].self_attn
            self.weights[l] = {k: getattr(at, k).weight.detach().to(self.torch.float64).numpy().copy()
                               for k in ("q_proj", "k_proj", "v_proj", "o_proj")}
            self.weights[l].update({f"b_{k}": getattr(at, k).bias.detach().to(self.torch.float64).numpy().copy()
                                    for k in ("q_proj", "k_proj", "v_proj")})
            self.weights[l]["sinks"] = at.sinks.detach().to(self.torch.float64).numpy().copy()
            self.weights[l]["scale"] = float(at.scaling)
            self.attn[l] = copy.deepcopy(at).double().eval()
        del self.model
        gc.collect()

    def head_weights(self, l: int, h: int) -> dict:
        """Rows ``64h..64h+63`` of ``q_proj``, rows ``64g..64g+63`` of ``k_proj`` / ``v_proj`` (``g = h // 8``), columns
        ``64h..64h+63`` of ``o_proj`` (GPT-2's orientation: ``W_Q`` etc. are 2880 x 64, ``W_O`` is 64 x 2880), the sink."""
        W, DK = self.weights[l], self.DK
        g = h // self.GROUP
        qs, ks = slice(h * DK, (h + 1) * DK), slice(g * DK, (g + 1) * DK)
        return {"W_Q": W["q_proj"][qs].T, "W_K": W["k_proj"][ks].T, "W_V": W["v_proj"][ks].T, "b_Q": W["b_q_proj"][qs],
                "b_K": W["b_k_proj"][ks], "b_V": W["b_v_proj"][ks], "W_O": W["o_proj"][:, qs].T,
                "sink": float(W["sinks"][h]), "scale": W["scale"]}

    def head_value(self, P: dict, l: int, h: int, pos: int, delta: dict, hw: dict, c: np.ndarray) -> float:
        """``c . (o_pos W_O)`` with ``o_pos`` head ``h``'s output at ``pos`` (the input of ``o_proj``), computed by the
        layer's own attention module (float64 copy: its projections, ``apply_rotary_pos_emb``, the sinks and the eager
        attention) on the normalized tokens ``0..pos`` shifted by ``delta[j]``, with a causal mask (the sliding window,
        128, does not bind)."""
        torch = self.torch
        z = P["z"][l][:pos + 1].copy()
        for j, dj in delta.items():
            z[j] = z[j] + dj
        x = torch.as_tensor(z[None], dtype=torch.float64)
        cs = tuple(torch.as_tensor(P[k][None, :pos + 1], dtype=torch.float64) for k in ("cos", "sin"))
        mask = torch.full((pos + 1, pos + 1), float("-inf"), dtype=torch.float64).triu(1)[None, None]
        got = {}
        at = self.attn[l]
        hk = at.o_proj.register_forward_pre_hook(lambda m, i: got.__setitem__("merged", i[0][0, pos].detach().numpy().copy()))
        try:
            with torch.no_grad():
                at(hidden_states=x, position_embeddings=cs, attention_mask=mask)
        finally:
            hk.remove()
        return float(c @ (got["merged"][h * self.DK:(h + 1) * self.DK] @ hw["W_O"]))


def frames(z: np.ndarray, hw: dict, pos: int, c: np.ndarray, keys: list, R: np.ndarray | None) -> dict:
    """The canonical plane of every varying key: its score changes by ``alpha_j . delta_j`` and its value along ``c`` by
    ``beta . delta_j``, ``beta = W_V W_O c``; ``alpha_j = W_K q / 8`` (GPT-2, the same for every key) or
    ``alpha_j = W_K R_j^T R_pos q / 8`` (gpt-oss-20b); ``e1_j = alpha_j / ||alpha_j||``, ``e2_j`` = the normalized
    component of ``beta`` orthogonal to ``e1_j``."""
    q = z[pos] @ hw["W_Q"] + hw["b_Q"]
    beta = hw["W_V"] @ (hw["W_O"] @ c)
    out = {}
    for j in keys:
        alpha = hw["scale"] * (hw["W_K"] @ (q if R is None else R[j].T @ (R[pos] @ q)))
        e1 = alpha / np.linalg.norm(alpha)
        b2 = beta - (beta @ e1) * e1
        out[j] = (e1, b2 / np.linalg.norm(b2))
    return out


def check_unit(M, u: dict, ang: list, cache: dict) -> dict:
    l, h, p, pos = int(u["layer"]), int(u["head"]), int(u["prompt"]), int(u["query_pos"])
    rho_rel = float(u["rho_rel"])
    P = cache[p]
    z, R = P["z"][l], P["R"]
    hw = M.head_weights(l, h)
    o = head_output(z, hw, pos, R)
    c = o / np.linalg.norm(o)
    res = {"unit_id": u["unit_id"], "ok": True, "notes": []}
    keys = sorted(ang)
    if [k[0] for k in keys] != list(range(pos)):
        res["ok"] = False
        res["notes"].append(f"varying keys {[k[0] for k in keys]} != 0..{pos - 1}")
    n_keys = pos + 1 + (R is not None)  # gpt-oss-20b: the sink is a key
    if int(u["n_keys"]) != n_keys or int(u["n_varying"]) != pos:
        res["ok"] = False
        res["notes"].append(f"n_keys / n_varying {u['n_keys']} / {u['n_varying']} != {n_keys} / {pos}")
    fr = frames(z, hw, pos, c, [k[0] for k in keys], R)
    rad_err = max((abs(rj - rho_rel * float(np.linalg.norm(z[j]))) / (rho_rel * float(np.linalg.norm(z[j])))
                   for j, rj, _, _ in keys), default=0.0)
    res["radius_rule_err"] = rad_err
    nom = float(c @ o)
    res["nominal_err"] = abs(nom - float(u["nominal_output"])) / max(1.0, abs(nom))
    res["reproduced"] = bool(max(rad_err, res["nominal_err"]) <= REPRO_TOL)
    tol_rel = TOL_REL
    if not res["reproduced"]:  # GPT-2 (float64) must reproduce them; gpt-oss-20b (float32) to TOL_F32 at least
        if M.name == "gpt2" or max(rad_err, res["nominal_err"]) > TOL_F32:
            res["ok"] = False
            res["notes"].append(f"stored radii or nominal output not reproduced (radius rule rel. {rad_err:.1e}, nominal "
                                f"rel. {res['nominal_err']:.1e})")
        tol_rel = TOL_F32
    worst_excess = 0.0
    for side, col, lo, hi in (("max", 2, float(u["max_lo"]), float(u["max_hi"])),
                              ("min", 3, float(u["min_lo"]), float(u["min_hi"]))):
        delta = {}
        for rec in keys:
            j, rj, th = rec[0], rec[1], rec[col]
            e1, e2 = fr[j]
            dj = rj * (math.cos(th) * e1 + math.sin(th) * e2)
            if np.linalg.norm(dj) > rj * (1 + RADIUS_TOL):
                res["ok"] = False
                res["notes"].append(f"{side}: key {j} outside its ball")
            delta[j] = dj
        zp = z.copy()
        for j, dj in delta.items():
            zp[j] = z[j] + dj
        v_np = float(c @ head_output(zp, hw, pos, R))
        v_hf = M.head_value(P, l, h, pos, delta, hw, c)
        tol = tol_rel * (1 + max(abs(lo), abs(hi)))
        for name, v in (("numpy", v_np), ("hf", v_hf)):
            ex = max(lo - v, v - hi, 0.0)
            worst_excess = max(worst_excess, ex / (1 + max(abs(lo), abs(hi))))
            if ex > tol:
                res["ok"] = False
                res["notes"].append(f"{side}: {name} value {v!r} outside [{lo!r}, {hi!r}] (tolerance {tol:.1e})")
        res[f"value_{side}_numpy"], res[f"value_{side}_hf"] = v_np, v_hf
        res[f"impl_diff_{side}"] = abs(v_np - v_hf)
    res["worst_excess_rel"] = worst_excess
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", type=int, help="number of random units")
    g.add_argument("--rho", type=float, help="every unit of this relative radius")
    g.add_argument("--unit", nargs="+", help="unit ids")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--model", choices=("gpt2", "gpt-oss-20b"), default="gpt2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None, help="torch threads (default 1 for gpt2, 10 for gpt-oss-20b, the "
                    "thread count of its benchmark)")
    ap.add_argument("--dir", default=None, help="directory holding units.csv, angles.csv.gz, prompts.txt (default: this "
                    "directory for gpt2, gpt-oss-20b/ for gpt-oss-20b)")
    args = ap.parse_args()
    base = Path(args.dir) if args.dir else (HERE if args.model == "gpt2" else HERE / "gpt-oss-20b")
    threads = args.threads or (1 if args.model == "gpt2" else 10)
    units = load_units(base / "units.csv")
    ids = sorted(units)
    if args.sample:
        rng = np.random.default_rng(args.seed)
        ids = sorted(rng.choice(ids, size=min(args.sample, len(ids)), replace=False).tolist())
    elif args.rho is not None:
        ids = [k for k in ids if math.isclose(float(units[k]["rho_rel"]), args.rho, rel_tol=1e-12)]
    elif args.unit:
        ids = args.unit
    if not ids:
        raise SystemExit("no unit selected")
    ang = load_angles(base / "angles.csv.gz", set(ids))
    prompts = [s.strip() for s in (base / "prompts.txt").read_text(encoding="utf-8").splitlines() if s.strip()]
    t0 = time.perf_counter()
    M = (GPT2 if args.model == "gpt2" else GptOss)(threads)
    cache = {p: M.prompt(prompts[p]) for p in sorted({int(units[k]["prompt"]) for k in ids})}
    M.release({int(units[k]["layer"]) for k in ids})
    print(f"{args.model}: model and {len(cache)} prompt(s) ready in {time.perf_counter() - t0:.0f} s ({threads} threads)",
          flush=True)
    bad, worst, impl, rad, nomerr, n_f32 = [], 0.0, 0.0, 0.0, 0.0, 0
    for k, uid in enumerate(ids):
        r = check_unit(M, units[uid], ang[uid], cache)
        worst = max(worst, r["worst_excess_rel"])
        impl = max(impl, r["impl_diff_max"], r["impl_diff_min"])
        rad = max(rad, r["radius_rule_err"])
        nomerr = max(nomerr, r["nominal_err"])
        n_f32 += not r["reproduced"]
        if not r["ok"]:
            bad.append(r)
            print(f"FAIL {uid}: {'; '.join(r['notes'])}", flush=True)
        if (k + 1) % 200 == 0:
            print(f"{k + 1}/{len(ids)} checked ({time.perf_counter() - t0:.0f} s)", flush=True)
    print(f"checked {len(ids)} units in {time.perf_counter() - t0:.1f} s: {len(ids) - len(bad)} passed, {len(bad)} failed")
    print(f"largest excess over a bracket (relative to 1 + |bound|): {worst:.2e} (tolerance {TOL_REL:g}); "
          f"largest numpy / Hugging Face difference: {impl:.2e}; largest radius-rule error: {rad:.2e}; "
          f"largest nominal-output error: {nomerr:.2e}")
    if n_f32:
        print(f"{n_f32} unit(s) with normalized tokens not reproduced to {REPRO_TOL:g} (float32 forward pass on another "
              f"machine or thread count): their brackets were checked to {TOL_F32:g} (1 + |bound|)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
