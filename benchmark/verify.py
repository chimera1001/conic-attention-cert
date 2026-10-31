"""Verify the fixed-query worst cases of GPT-2 small stored in this directory (see SCHEMA.md).

For every selected unit the script rebuilds the instance from the Hugging Face ``gpt2`` weights and the prompt alone:
the post-normalization tokens ``zbar_j`` read by the head (the output of ``ln_1`` of the block), the radii
``rho_j = rho_rel ||zbar_j||``, the fixed query, the specification ``c`` (direction of the head's nominal output) and
the canonical frame ``(e1, e2)``. It then rebuilds the maximizing and the minimizing input from the stored angles and
checks that

1. every perturbation lies in its ball (``||delta_j|| <= rho_j``) and the stored radii follow the radius rule;
2. the nominal output equals the stored one;
3. the head's output along ``c`` at each worst-case input, computed twice — by a direct numpy evaluation of the head and
   by the Hugging Face model itself with the perturbed tokens injected at the output of ``ln_1`` — lies in the stored
   certified bracket (of the maximum, resp. of the minimum).

Only numpy, torch (CPU) and transformers are needed; the ``aimp`` package is not used.

Usage::

    python benchmark/verify.py --sample 200 --seed 0     # 200 random units
    python benchmark/verify.py --rho 0.01                # all 2,880 units of one radius
    python benchmark/verify.py --unit L05H01_P0_i04_r0.01
    python benchmark/verify.py --all
"""
from __future__ import annotations

import argparse
import csv
import gzip
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D, DK = 768, 64
TOL_REL = 1e-9  # a value inside [lo, hi] up to 1e-9 (1 + |bound|): double-precision rounding of two implementations
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


class Model:
    """``gpt2`` in float64 with hooks: ``ln_1`` outputs of every block, and the value of one head along ``c`` with the
    output of ``ln_1`` of one block replaced at chosen positions."""

    def __init__(self, threads: int):
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = GPT2TokenizerFast.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager", dtype=torch.float64).eval()

    def ids(self, text: str):
        return self.tok(text, return_tensors="pt")["input_ids"]

    def ln1_outputs(self, ids) -> list[np.ndarray]:
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
        return z

    def head_value(self, ids, l: int, h: int, pos: int, delta: dict, W_O: np.ndarray, c: np.ndarray) -> float:
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
            got["merged"] = i[0][0, pos, h * DK:(h + 1) * DK].detach().numpy().astype(np.float64).copy()
            raise _Stop

        h1 = blk.ln_1.register_forward_hook(ln_hook)
        h2 = blk.attn.c_proj.register_forward_pre_hook(proj_hook)
        try:
            with self.torch.no_grad():
                self.model(ids)
        except _Stop:
            pass
        finally:
            h1.remove()
            h2.remove()
        return float(c @ (got["merged"] @ W_O))

    def head_weights(self, l: int, h: int) -> dict:
        blk = self.model.transformer.h[l]
        Wa = blk.attn.c_attn.weight.detach().numpy().astype(np.float64)
        ba = blk.attn.c_attn.bias.detach().numpy().astype(np.float64)
        sl = slice(h * DK, (h + 1) * DK)
        return {"W_Q": Wa[:, :D][:, sl], "W_K": Wa[:, D:2 * D][:, sl], "W_V": Wa[:, 2 * D:][:, sl],
                "b_Q": ba[:D][sl], "b_K": ba[D:2 * D][sl], "b_V": ba[2 * D:][sl],
                "W_O": blk.attn.c_proj.weight.detach().numpy().astype(np.float64)[sl, :]}


def head_output(z: np.ndarray, hw: dict, pos: int) -> np.ndarray:
    """Head output at ``pos`` (contribution to the residual stream before the shared ``c_proj`` bias), causal mask."""
    q = z[pos] @ hw["W_Q"] + hw["b_Q"]
    k = z[:pos + 1] @ hw["W_K"] + hw["b_K"]
    v = z[:pos + 1] @ hw["W_V"] + hw["b_V"]
    s = k @ q / math.sqrt(DK)
    a = np.exp(s - s.max())
    a /= a.sum()
    return (a @ v) @ hw["W_O"]


def check_unit(M: Model, u: dict, ang: list, cache: dict) -> dict:
    l, h, p, pos = int(u["layer"]), int(u["head"]), int(u["prompt"]), int(u["query_pos"])
    rho_rel = float(u["rho_rel"])
    if p not in cache:
        ids = M.ids(cache["prompts"][p])
        cache[p] = (ids, M.ln1_outputs(ids))
    ids, Z = cache[p]
    z = Z[l]
    hw = M.head_weights(l, h)
    o = head_output(z, hw, pos)
    c = o / np.linalg.norm(o)
    q = z[pos] @ hw["W_Q"] + hw["b_Q"]
    alpha = hw["W_K"] @ q / math.sqrt(DK)
    beta = hw["W_V"] @ (hw["W_O"] @ c)
    e1 = alpha / np.linalg.norm(alpha)
    b2 = beta - (beta @ e1) * e1
    e2 = b2 / np.linalg.norm(b2)
    res = {"unit_id": u["unit_id"], "ok": True, "notes": []}
    keys = sorted(ang)
    if [k[0] for k in keys] != list(range(pos)):
        res["ok"] = False
        res["notes"].append(f"varying keys {[k[0] for k in keys]} != 0..{pos - 1}")
    rad_err = max((abs(rj - rho_rel * float(np.linalg.norm(z[j]))) / (rho_rel * float(np.linalg.norm(z[j])))
                   for j, rj, _, _ in keys), default=0.0)
    res["radius_rule_err"] = rad_err
    if rad_err > 1e-10:
        res["ok"] = False
        res["notes"].append(f"stored radii differ from rho_rel ||zbar_j|| (rel. {rad_err:.1e})")
    nom = float(c @ o)
    res["nominal_err"] = abs(nom - float(u["nominal_output"])) / max(1.0, abs(nom))
    if res["nominal_err"] > 1e-10:
        res["ok"] = False
        res["notes"].append(f"nominal output differs (rel. {res['nominal_err']:.1e})")
    worst_excess = 0.0
    for side, col, lo, hi in (("max", 2, float(u["max_lo"]), float(u["max_hi"])),
                              ("min", 3, float(u["min_lo"]), float(u["min_hi"]))):
        delta = {}
        for rec in keys:
            j, rj, th = rec[0], rec[1], rec[col]
            dj = rj * (math.cos(th) * e1 + math.sin(th) * e2)
            if np.linalg.norm(dj) > rj * (1 + RADIUS_TOL):
                res["ok"] = False
                res["notes"].append(f"{side}: key {j} outside its ball")
            delta[j] = dj
        zp = z.copy()
        for j, dj in delta.items():
            zp[j] = z[j] + dj
        v_np = float(c @ head_output(zp, hw, pos))
        v_hf = M.head_value(ids, l, h, pos, delta, hw["W_O"], c)
        tol = TOL_REL * (1 + max(abs(lo), abs(hi)))
        for name, v in (("numpy", v_np), ("hf", v_hf)):
            ex = max(lo - v, v - hi, 0.0)
            worst_excess = max(worst_excess, ex / (1 + max(abs(lo), abs(hi))))
            if ex > tol:
                res["ok"] = False
                res["notes"].append(f"{side}: {name} value {v!r} outside [{lo!r}, {hi!r}]")
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--dir", default=str(HERE), help="directory holding units.csv, angles.csv.gz, prompts.txt")
    args = ap.parse_args()
    base = Path(args.dir)
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
    M = Model(args.threads)
    cache: dict = {"prompts": prompts}
    bad, worst, impl, rad = [], 0.0, 0.0, 0.0
    for k, uid in enumerate(ids):
        r = check_unit(M, units[uid], ang[uid], cache)
        worst = max(worst, r["worst_excess_rel"])
        impl = max(impl, r["impl_diff_max"], r["impl_diff_min"])
        rad = max(rad, r["radius_rule_err"])
        if not r["ok"]:
            bad.append(r)
            print(f"FAIL {uid}: {'; '.join(r['notes'])}", flush=True)
        if (k + 1) % 200 == 0:
            print(f"{k + 1}/{len(ids)} checked ({time.perf_counter() - t0:.0f} s)", flush=True)
    print(f"checked {len(ids)} units in {time.perf_counter() - t0:.1f} s: {len(ids) - len(bad)} passed, {len(bad)} failed")
    print(f"largest excess over a bracket (relative to 1 + |bound|): {worst:.2e} (tolerance {TOL_REL:g}); "
          f"largest numpy / Hugging Face difference: {impl:.2e}; largest radius-rule error: {rad:.2e}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
