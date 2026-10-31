"""The certificate on every head of GPT-2 small: the accuracy table, the radius ladder and branching (Section 7).

Parts:

* ``units`` — the 6,480 level-one units: query position 4 at relative radii 0.001, 0.002, 0.005, 0.01, 0.02 (the
  ladder of Figure 2) and positions 8 and 12 at 0.01 and 0.02; every head, five prompts, query varying, every token
  perturbed. Each unit runs the procedure of Appendix B (:func:`aimp.pipeline.evaluate_unit`).
* ``bracket`` — branch-and-bound on the score boxes (Appendix B): the largest-``S_j`` split rule on the 144 heads at
  prompt 0, position 4, ``rho_rel = 0.01``, and on the 24 hardest of them at 0.002; the bound-lookahead rule on 72 of
  the heads at 0.01 (the 24 with the largest final bracket, plus four per layer drawn evenly by root gap).
* ``radius`` — tightness radii at the tolerances 0.01 and 0.1 of the nominal value spread, on three heads per layer
  (smallest, median and largest ``S`` at prompt 0, position 4, 0.01), for the certificate and the first-order one.
* ``ascent`` — the alternating ascent (exact maximization over the keys for the current query, local ascent over the
  query) from the five best multistart points of every ladder unit.
* ``summary`` — Table 4, ``ladder.csv`` (the data of Figure 2, columns as ``figdata/P3b_ladder.csv``), the slopes and
  the other numbers of the paragraphs on GPT-2.

Usage (from the repository root)::

    python experiments/gpt2_ladder.py                      # units + summary, full settings (~77 h on 3 x 4 threads)
    python experiments/gpt2_ladder.py --quick              # two heads, prompt 0, position 4, three radii
    python experiments/gpt2_ladder.py --part bracket
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

LADDER_RHOS = (0.001, 0.002, 0.005, 0.01, 0.02)
SMALL = (0.001, 0.002, 0.005)
BAND_RHOS = (0.01, 0.02)
BAND_POS = (8, 12)
FLOOR_REL = 1e-7  # gaps <= 1e-7 x range are at the numerical floor
BRACKET_UNIT = (0, 4, 0.01)  # (prompt, position, rho_rel)
BRACKET_HARD = (0, 4, 0.002)
N_HARD = 24
N_TOP, N_PER_LAYER, SEL_SEED = 24, 4, 164
RADIUS_EPS = (0.01, 0.1)  # x the nominal value spread
RADIUS_BRACKET = (1e-4, 1.0)
ALT_STARTS = 5
QUICK = {"heads": ((1, 3), (7, 6)), "prompts": (0,), "rhos": (0.001, 0.002, 0.005)}  # two typical heads


def log(base: Path, msg: str) -> None:
    base.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(base / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _init(threads: int) -> None:
    os.environ["AIMP_MAX_THREADS"] = str(threads)
    from aimp import compute

    compute.apply_thread_budget(threads)


def unit_path(base: Path, l, h, p, pos, rho) -> Path:
    from aimp.gpt2_units import unit_name

    return base / "units" / f"L{l:02d}" / f"{unit_name(l, h, p, pos, rho)}.json"


def load_unit(base: Path, l, h, p, pos, rho) -> dict | None:
    f = unit_path(base, l, h, p, pos, rho)
    return json.loads(f.read_text()) if f.exists() else None


def pool_run(base: Path, fn, tasks: list, workers: int, tps: int, label: str) -> None:
    from aimp import compute

    if workers * tps > compute.thread_budget():
        raise SystemExit(f"workers x threads = {workers * tps} exceeds AIMP_MAX_THREADS = {compute.thread_budget()}")
    log(base, f"=== {label}: {len(tasks)} tasks, {workers} workers x {tps} threads")
    t0, done = time.perf_counter(), 0
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(tps,), maxtasksperchild=20) as pool:
        for out in pool.imap_unordered(fn, tasks):
            done += 1
            if "fatal" in out:
                log(base, f"!!! FATAL {out['name']}: {out['fatal']}\n{out['tb']}")
            elif out.get("invalid"):
                log(base, f"!!! {out['name']}: {out['invalid']}")
            if not out.get("skipped"):
                log(base, f"{label}: {done}/{len(tasks)} {out['name']} t={out.get('t', 0):.1f}s "
                          f"({(time.perf_counter() - t0) / 60:.1f} min)")
    log(base, f"=== {label} finished in {(time.perf_counter() - t0) / 60:.1f} min")


# ------------------------------------------------------------------------------------------------ units
def unit_tasks(base: Path, quick: bool) -> list[tuple]:
    from aimp.gpt2_units import HEADS, LAYERS, N_PROMPTS

    if quick:
        return [(str(base), l, h, p, 4, r) for r in QUICK["rhos"] for (l, h) in QUICK["heads"] for p in QUICK["prompts"]]
    hp = [(l, h, p) for l in LAYERS for h in HEADS for p in range(N_PROMPTS)]
    out = [(str(base), l, h, p, 4, r) for r in sorted(LADDER_RHOS, reverse=True) for (l, h, p) in hp]
    out += [(str(base), l, h, p, pos, r) for pos in BAND_POS for r in BAND_RHOS for (l, h, p) in hp]
    return out


def run_unit(task: tuple) -> dict:
    base, l, h, p, pos, rho = task
    base = Path(base)
    path = unit_path(base, l, h, p, pos, rho)
    if path.exists():
        return {"name": path.stem, "skipped": True}
    try:
        from aimp.io import jsonable, write_json_atomic
        from aimp.pipeline import evaluate_unit

        rec = evaluate_unit(l, h, p, pos, rho, int(os.environ.get("AIMP_MAX_THREADS", "4")))
        write_json_atomic(path, jsonable(rec))
        bad = [k for k in ("valid", "valid_value_safe", "valid_greedy") if rec[k] is False]
        bad += [f"check {k} violated" for k, v in rec["checks"].items() if v == "violated"]
        return {"name": path.stem, "t": rec["times"]["unit_total_s"], "invalid": bad}
    except Exception as e:
        return {"name": path.stem, "fatal": repr(e), "tb": traceback.format_exc()[-3000:]}


# ------------------------------------------------------------------------------------------------ bracket
def bracket_path(base: Path, rule: str, l, h, p, pos, rho) -> Path:
    from aimp.gpt2_units import unit_name

    return base / f"bracket_{rule}" / f"{unit_name(l, h, p, pos, rho)}.json"


def run_bracket(task: tuple) -> dict:
    """Branch-and-bound with 24 semidefinite solves from the unit's controls, best feasible value and smallest output."""
    base, rule, l, h, p, pos, rho = task
    base = Path(base)
    path = bracket_path(base, rule, l, h, p, pos, rho)
    if path.exists():
        return {"name": path.stem, "skipped": True}
    try:
        from aimp import gpt2_units as U
        from aimp.certify import branch_and_bound, default_f0v_inputs
        from aimp.gpt2 import nominal_value_range
        from aimp.io import env_info, git_commit, jsonable, write_json_atomic
        from aimp.lower import strong_bracket
        from aimp.pipeline import SDP_TOL, controls
        from aimp.relax_ps import solve_ps_greedy

        threads = int(os.environ.get("AIMP_MAX_THREADS", "4"))
        t0 = time.perf_counter()
        inst = U.make_instance(l, h, p, pos, rho)
        _, ctrl = controls(inst, threads)
        rg = solve_ps_greedy(inst, ctrl, "auto").value
        src = load_unit(base, l, h, p, pos, rho)
        if src is not None:
            lb0, pm0 = src["lb_strong"], src["phi_min_strong"]
        else:
            b = strong_bracket(inst, U.unit_seed(l, h, p, pos, rho) + 31)
            lb0, pm0 = b["lb_strong"], b["phi_min_strong"]
        if rule == "S":
            bb = branch_and_bound(inst, ctrl, lb0, pm0, threads=threads, tol=SDP_TOL, r_greedy=rg, rule="S",
                                  guard="node", cut="auto")
        else:
            fi = default_f0v_inputs(inst, ctrl, "auto")
            bb = branch_and_bound(inst, ctrl, lb0, pm0, threads=threads, tol=SDP_TOL, r_greedy=rg, rule="lookahead",
                                  guard="termination", h_max=np.asarray(fi["h_max"], float), f0v_inputs=fi, cut="auto")
        bb.pop("LB_Z", None)
        spread = bb["LB"] - pm0
        root = bb["nodes"][0]
        out = {"rule": rule, "layer": l, "head": h, "prompt": p, "pos": pos, "rho_rel": rho, "n": inst.n,
               "range": nominal_value_range(inst), "lb_input": lb0, "phi_min_strong": pm0, "r_greedy": rg,
               "root_bound": root["bound"], "spread": spread,
               "rel_gap_cert_root": (min(rg, root["bound"]) - bb["LB"]) / spread if spread > 0 else None,
               "rel_gap_final": (bb["cert_bb"] - bb["LB"]) / spread if spread > 0 else None,
               **bb, "wall_total_s": time.perf_counter() - t0, "threads": threads, "env": env_info(), "git": git_commit()}
        write_json_atomic(path, jsonable(out))
        return {"name": path.stem, "t": out["wall_total_s"], "invalid": [] if bb["valid"] else ["UB < LB"]}
    except Exception as e:
        return {"name": path.stem, "fatal": repr(e), "tb": traceback.format_exc()[-3000:]}


def hard_heads(base: Path) -> list[tuple[int, int]]:
    """The 24 heads with the largest relative gap of the certificate at prompt 0, position 4, 0.01."""
    rows = []
    for l in range(12):
        for h in range(12):
            r = load_unit(base, l, h, *BRACKET_UNIT)
            if r is None:
                raise SystemExit("bracket: run --part units first (the hardest heads are chosen from the units at 0.01)")
            rows.append((r.get("rel_gap") or 0.0, l, h))
    rows.sort(reverse=True)
    return [(l, h) for _, l, h in rows[:N_HARD]]


def lookahead_heads(base: Path) -> list[tuple[int, int]]:
    """The 24 heads with the largest final bracket of the largest-S_j rule at (prompt 0, position 4, 0.01), plus four
    per layer among the rest, one drawn per equal-count stratum of the root gap."""
    B = {}
    for l in range(12):
        for h in range(12):
            f = bracket_path(base, "S", l, h, *BRACKET_UNIT)
            if not f.exists():
                raise SystemExit("bracket: the lookahead heads are chosen from the 144 brackets of the S rule")
            r = json.loads(f.read_text())
            B[(l, h)] = (r["rel_gap_final"] or 0.0, r["rel_gap_cert_root"] or 0.0)
    top = sorted(B, key=lambda k: -B[k][0])[:N_TOP]
    out = list(top)
    for l in range(12):
        rest = sorted([k for k in B if k[0] == l and k not in top], key=lambda k: B[k][1])
        rng = np.random.default_rng(SEL_SEED + l)
        for stratum in np.array_split(np.arange(len(rest)), N_PER_LAYER):
            if len(stratum):
                out.append(rest[int(rng.choice(stratum))])
    return out


# ------------------------------------------------------------------------------------------------ radius
def cert_at(l, h, p, pos, rho, use_sdp: bool, threads=None) -> dict:
    """The certificate (or the first-order one) at ``rho`` and its gap to a quick multistart value."""
    from aimp import gpt2_units as U
    from aimp.certify import certify
    from aimp.lower_bounds import multistart_lb
    from aimp.pipeline import SDP_TOL, controls

    inst = U.make_instance(l, h, p, pos, rho)
    _, ctrl = controls(inst, threads)
    rec = certify(inst, ctrl, eps_target=0.0, threads=threads, run_sdp=use_sdp, sdp_tol=SDP_TOL, cut="auto")
    ms = multistart_lb(inst, 1, seed=U.unit_seed(l, h, p, pos, rho), maxiter=300)
    lb = max(ms.value, rec["lb_best"])
    return {"rho_rel": rho, "cert": rec["cert"], "r_greedy": rec["r_greedy"], "lb": lb, "gap_lb": rec["cert"] - lb,
            "greedy_gap_lb": rec["r_greedy"] - lb}


def radius_path(base: Path, l, h, p, pos) -> Path:
    return base / "radius" / f"L{l:02d}H{h:02d}_P{p}_i{pos:02d}.json"


def run_radius(task: tuple) -> dict:
    base, l, h, p, pos = task
    base = Path(base)
    path = radius_path(base, l, h, p, pos)
    if path.exists():
        return {"name": path.stem, "skipped": True}
    try:
        from aimp import gpt2_units as U
        from aimp.certify import rho_star
        from aimp.gpt2 import nominal_value_range
        from aimp.io import jsonable, write_json_atomic

        threads = int(os.environ.get("AIMP_MAX_THREADS", "4"))
        t0 = time.perf_counter()
        cache: dict = {}

        def ev(r):
            if r not in cache:
                cache[r] = cert_at(l, h, p, pos, r, True, threads)
            return cache[r]

        rng_ = nominal_value_range(U.make_instance(l, h, p, pos, RADIUS_BRACKET[0]))
        out = {"layer": l, "head": h, "prompt": p, "pos": pos, "range": rng_, "results": {}}
        for comp, key in (("cert", "gap_lb"), ("greedy", "greedy_gap_lb")):
            for e in RADIUS_EPS:
                rs = rho_star(lambda r: ev(r)[key], e * rng_, *RADIUS_BRACKET, 6)
                rs.pop("evals", None)
                out["results"][f"{comp}_eps{e:g}"] = rs
        out["evals"] = [cache[r] for r in sorted(cache)]
        out["wall_s"] = time.perf_counter() - t0
        write_json_atomic(path, jsonable(out))
        return {"name": path.stem, "t": out["wall_s"]}
    except Exception as e:
        return {"name": path.stem, "fatal": repr(e), "tb": traceback.format_exc()[-3000:]}


def radius_heads(base: Path) -> list[tuple[int, int]]:
    """Per layer, the heads with the smallest, median and largest ``S`` at (prompt 0, position 4, 0.01)."""
    out = []
    for l in range(12):
        rows = []
        for h in range(12):
            r = load_unit(base, l, h, 0, 4, 0.01)
            if r is None:
                raise SystemExit("radius: run --part units first (heads are chosen by S at prompt 0, position 4, 0.01)")
            rows.append((r["S"], h))
        rows.sort()
        out += [(l, rows[k][1]) for k in (0, len(rows) // 2, len(rows) - 1)]
    return out


# ------------------------------------------------------------------------------------------------ ascent
def run_ascent(task: tuple) -> dict:
    base, l, h, p, pos, rho = task
    base = Path(base)
    from aimp.gpt2_units import unit_name

    path = base / "ascent" / f"{unit_name(l, h, p, pos, rho)}.json"
    if path.exists():
        return {"name": path.stem, "skipped": True}
    try:
        from aimp import gpt2_units as U
        from aimp.io import jsonable, write_json_atomic
        from aimp.lower import alternating_ascent, ascend, boundary_points, greedy_point
        from aimp.lower_bounds import _Param

        t0 = time.perf_counter()
        rec = load_unit(base, l, h, p, pos, rho)
        if rec is None:
            return {"name": path.stem, "skipped": True, "reason": "no unit record"}
        inst = U.make_instance(l, h, p, pos, rho)
        lbs, pms = float(rec["lb_strong"]), float(rec["phi_min_strong"])
        spread = lbs - pms
        prm = rec["lb_info"]["params"]
        P = _Param(inst)
        named = [("center", inst.centers()), ("greedy", greedy_point(inst))]
        named += [("random", Z) for Z in boundary_points(inst, np.random.default_rng(int(prm["seed"])), int(prm["n_random"]))]
        pts = sorted(((float(v), k, np.asarray(Z, float)) for k, Z0 in named
                      for v, Z, _ in [ascend(inst, P, Z0, int(prm["maxiter"]))]), key=lambda x: -x[0])
        chosen = []
        for v, k, Z in pts:  # distinct query points
            if all(np.linalg.norm(Z[inst.query] - Zc[inst.query]) > 1e-9 * max(1.0, np.linalg.norm(Zc[inst.query]))
                   for _, _, Zc in chosen):
                chosen.append((v, k, Z))
            if len(chosen) == ALT_STARTS:
                break
        runs = [alternating_ascent(inst, Z, spread) | {"start_type": k, "start_value": v0} for v0, k, Z in chosen]
        lb_alt = max(r["value"] for r in runs)
        adm = max(r["admissible_excess"] for r in runs)
        out = {"layer": l, "head": h, "prompt": p, "pos": pos, "rho_rel": rho, "lb_strong": lbs, "phi_min_strong": pms,
               "spread": spread, "cert": rec["cert"], "lb_alt": lb_alt,
               "lb_alt_gain_rel": (lb_alt - lbs) / spread if spread > 0 else None,
               "improves": bool(spread > 0 and lb_alt > lbs + 1e-9 * spread), "admissible": bool(adm <= 1e-9),
               "rounds": [r["rounds"] for r in runs], "wall_s": time.perf_counter() - t0}
        write_json_atomic(path, jsonable(out))
        return {"name": path.stem, "t": out["wall_s"], "invalid": [] if out["admissible"] else ["inadmissible input"]}
    except Exception as e:
        return {"name": path.stem, "fatal": repr(e), "tb": traceback.format_exc()[-3000:]}


# ------------------------------------------------------------------------------------------------ summary
def _fit(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    if len(x) < 2:
        return None
    A = np.vstack([x, np.ones_like(x)]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][0])


def _f(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def load_units(base: Path):
    import pandas as pd

    keys = ("layer", "head", "prompt", "pos", "rho_rel", "n", "range", "cert", "r_greedy", "value_safe", "lb_strong",
            "phi_min_strong", "w_t", "S", "theta", "Pi_v", "bound_F0v_final", "F0vf_term_knapsack", "gamma", "valid",
            "margin_regime")
    rows = []
    for f in sorted((base / "units").glob("L*/*.json")):
        r = json.loads(f.read_text())
        rows.append({k: r.get(k) for k in keys} | {"t_s": r["times"]["unit_total_s"]})
    U = pd.DataFrame(rows)
    if U.empty:
        return U
    U["spread"] = U.lb_strong - U.phi_min_strong
    U["gap"] = U.cert - U.lb_strong
    U["greedy_gap"] = U.r_greedy - U.lb_strong
    U["rel_gap"] = U.gap / U.spread.where(U.spread > 0)
    U["greedy_rel_gap"] = U.greedy_gap / U.spread.where(U.spread > 0)
    U["gap_over_range"] = U.gap / U.range.where(U.range > 0)
    U["floor"] = U.gap <= FLOOR_REL * U.range
    return U


def ladder_table(U):
    """Per (layer, head, prompt) at position 4: gaps and per-unit slopes (the columns of figdata/P3b_ladder.csv)."""
    import pandas as pd

    L = U[(U.pos == 4) & U.rho_rel.isin(LADDER_RHOS)]
    rows = []
    for key, g in L.groupby(["layer", "head", "prompt"]):
        g = g.set_index("rho_rel")
        row = {"layer": key[0], "head": key[1], "prompt": key[2], "n_radii": len(g)}
        for q, col in (("cert", "gap"), ("greedy", "greedy_gap"), ("spread", "spread")):
            ok = {}
            for r in LADDER_RHOS:
                if r in g.index:
                    v = g.at[r, col]
                    if _f(v) and v > FLOOR_REL * g.at[r, "range"]:
                        ok[r] = v
                row[f"{q}_{r:g}"] = ok.get(r)
            for win, rs in (("small", SMALL), ("all", LADDER_RHOS)):
                pts = [(r, ok[r]) for r in rs if r in ok]
                row[f"slope_{q}_{win}"] = _fit(*zip(*pts)) if len(pts) >= 2 else None
                row[f"npts_{q}_{win}"] = len(pts)
            for a, b in zip(LADDER_RHOS[:-1], LADDER_RHOS[1:]):
                row[f"slope_{q}_{a:g}-{b:g}"] = _fit([a, b], [ok[a], ok[b]]) if (a in ok and b in ok) else None
        rows.append(row)
    return pd.DataFrame(rows)


def summary(base: Path) -> None:
    import pandas as pd

    U = load_units(base)
    if U.empty:
        print("no unit records")
        return
    print("Table 4 (relative gap = (cert - best feasible value) / range of motion; range-normalized = / nominal spread)")
    print("rho_rel  units  median w_t  cert    first order  range-norm  <=1%  <=10%")
    for rho, g in U.groupby("rho_rel"):
        rg = g.rel_gap.astype(float)
        print(f"{rho:<7g}  {len(g):5d}  {g.w_t.median():9.3g}  {rg.median():.3g}  {g.greedy_rel_gap.median():11.3g}  "
              f"{g.gap_over_range.median():10.2g}  {(rg <= 0.01).mean():4.0%}  {(rg <= 0.1).mean():5.0%}")
    print(f"w_t / rho_rel (median): {(U.w_t / U.rho_rel).median():.0f}")
    for rho in (0.01,):
        g = U[U.rho_rel == rho]
        if len(g):
            print(f"share with 1 <= w_t <= 8 at rho_rel {rho:g}: {((g.w_t >= 1) & (g.w_t <= 8)).mean():.1%}")
    print(f"units with a certified dominant key (margin bound <= 1% of the range of motion): "
          f"{int(U.margin_regime.astype(bool).sum())} / {len(U)}")
    band = U[U.rho_rel.isin(BAND_RHOS)]
    if len(band):
        print(f"band units (rho_rel 0.01, 0.02): certificate strictly tighter than first order on "
              f"{(band.cert < band.r_greedy).mean():.1%}; median gap ratio first order / certificate "
              f"{(band.greedy_gap / band.gap).median():.2f}")
    print(f"valid: {int(U.valid.astype(bool).sum())} / {len(U)}; median wall per unit {U.t_s.median():.1f} s")
    P = ladder_table(U)
    if len(P):
        P.to_csv(base / "ladder.csv", index=False)
        for q, lab in (("cert", "certificate"), ("greedy", "first order"), ("spread", "range of motion")):
            s = P[f"slope_{q}_small"].dropna()
            if len(s):
                print(f"ladder slope on {SMALL}, {lab}: {s.median():.2f} [{s.quantile(.25):.2f}, {s.quantile(.75):.2f}] "
                      f"(n={len(s)})")
        cons = [f"{a:g}-{b:g}: {P[f'slope_cert_{a:g}-{b:g}'].dropna().median():.2f}"
                for a, b in zip(LADDER_RHOS[:-1], LADDER_RHOS[1:]) if P[f"slope_cert_{a:g}-{b:g}"].notna().any()]
        print("consecutive-radius slopes of the certificate: " + ", ".join(cons))
        print(f"wrote {base / 'ladder.csv'}")
    for rule in ("S", "lookahead"):
        d = base / f"bracket_{rule}"
        if d.exists():
            B = pd.DataFrame([json.loads(f.read_text()) for f in sorted(d.glob("*.json"))])
            for rho, g in B.groupby("rho_rel"):
                print(f"branching ({rule} rule), rho_rel {rho:g}, {len(g)} heads: median relative bracket "
                      f"{g.rel_gap_cert_root.median():.3g} -> {g.rel_gap_final.median():.3g}")
    rd = base / "radius"
    if rd.exists():
        ex = {"cert": [], "greedy": []}
        for f in sorted(rd.glob("*.json")):
            r = json.loads(f.read_text())["results"]
            for comp in ex:
                a, b = r[f"{comp}_eps0.01"], r[f"{comp}_eps0.1"]
                ra, rb = a.get("rho_star"), b.get("rho_star")
                if ra is None or a.get("censored") == "above":
                    continue  # no information at either tolerance
                rb = RADIUS_BRACKET[1] if (rb is None or b.get("censored") == "above") else rb
                ex[comp].append(math.log10(rb / ra))
        for comp, v in ex.items():
            if v:
                print(f"tightness-radius exponent ({comp}): median {np.median(v):.2f} over {len(v)} heads "
                      "(radii beyond the largest tested enter at that radius)")
    ad = base / "ascent"
    if ad.exists():
        A = pd.DataFrame([json.loads(f.read_text()) for f in sorted(ad.glob("*.json"))])
        if len(A):
            print(f"alternating ascent: improves the best feasible value by > 1e-9 x spread on {int(A.improves.sum())} / "
                  f"{len(A)} units; max gain {A.lb_alt_gain_rel.max():.2e} x spread")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", choices=("all", "units", "bracket", "radius", "ascent", "summary"), default="all")
    ap.add_argument("--quick", action="store_true", help="two heads, prompt 0, position 4, three radii")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--threads-per-solve", type=int, default=None)
    ap.add_argument("--out", default=None, help="results directory (default: $AIMP_RESULTS_DIR/gpt2_ladder)")
    args = ap.parse_args()
    from aimp import compute, gpt2_units
    from aimp.io import results_dir

    base = Path(args.out) if args.out else results_dir() / ("gpt2_ladder_quick" if args.quick else "gpt2_ladder")
    budget = compute.thread_budget()
    tps = args.threads_per_solve or min(4, budget)
    w = args.workers or max(1, budget // tps)
    gpt2_units.prepare()
    parts = ("units", "summary") if args.part == "all" else (args.part,)
    if "units" in parts:
        pool_run(base, run_unit, [t for t in unit_tasks(base, args.quick) if not unit_path(base, *t[1:]).exists()],
                 w, tps, "units")
    if "bracket" in parts:
        hp = list(QUICK["heads"]) if args.quick else [(l, h) for l in range(12) for h in range(12)]
        tasks = [(str(base), "S", l, h, *BRACKET_UNIT) for l, h in hp]
        pool_run(base, run_bracket, tasks, w, tps, "bracket S, 0.01")
        if not args.quick:
            pool_run(base, run_bracket, [(str(base), "S", l, h, *BRACKET_HARD) for l, h in hard_heads(base)], w, tps,
                     "bracket S, 0.002")
            pool_run(base, run_bracket, [(str(base), "lookahead", l, h, *BRACKET_UNIT) for l, h in lookahead_heads(base)],
                     w, tps, "bracket lookahead, 0.01")
    if "radius" in parts:
        hp = list(QUICK["heads"][:1]) if args.quick else radius_heads(base)
        pool_run(base, run_radius, [(str(base), l, h, 0, 4) for l, h in hp], w, tps, "radius")
    if "ascent" in parts:
        tasks = [(str(base), *t[1:]) for t in unit_tasks(base, args.quick) if t[4] == 4]
        pool_run(base, run_ascent, tasks, budget, 1, "ascent")
    if "summary" in parts or args.part == "all":
        summary(base)


if __name__ == "__main__":
    main()
