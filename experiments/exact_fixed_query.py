"""Exact worst cases with a fixed query on GPT-2 small (Table 5 of the paper and the benchmark of ``benchmark/``).

Parts:

* ``exact`` — every head, five prompts, query positions 4, 8, 12 and the last one, relative radii 0.001, 0.002, 0.005,
  0.01, 0.02, 0.05 (17,280 units): the certified brackets of the maximum and of the minimum of the head's output along
  its nominal direction (Theorem 7), the first-order certificate, and the canonical angles of the worst-case inputs.
* ``sdp`` — the certificate on the 2,880 units at position 4 and radii 0.002–0.02, against the exact worst case.
* ``benchmark`` — writes ``benchmark/units.csv`` and ``benchmark/angles.csv.gz`` from the ``exact`` records.
* ``summary`` — Table 5 (relative gaps, slopes between 0.002 and 0.005, share within 1%), the gap as a fraction of the
  second-order bound (25), multistart against the exact worst case, and the ratio of the secant widths with a varying
  and a fixed query (needs the records of ``experiments/gpt2_ladder.py``).

Usage (from the repository root)::

    python experiments/exact_fixed_query.py                 # all parts, full settings
    python experiments/exact_fixed_query.py --quick         # 2 heads x 1 prompt, a few minutes
    python experiments/exact_fixed_query.py --part exact --workers 12
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

EXACT_RHOS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
SDP_RHOS = (0.002, 0.005, 0.01, 0.02)
SDP_POS = 4
QUICK = {"heads": ((1, 3), (7, 6)), "prompts": (0,), "sdp_rhos": (0.002, 0.005)}  # two typical heads


def out_dir(args) -> Path:
    from aimp.io import results_dir

    return Path(args.out) if args.out else results_dir() / "exact_fixed_query"


def exact_path(base: Path, l, h, p, pos, rho) -> Path:
    from aimp.gpt2_units import unit_name

    return base / "exact" / f"L{l:02d}" / f"{unit_name(l, h, p, pos, rho)}_fq.json"


def sdp_path(base: Path, l, h, p, pos, rho) -> Path:
    from aimp.gpt2_units import unit_name

    return base / "sdp" / f"L{l:02d}" / f"{unit_name(l, h, p, pos, rho)}_fq.json"


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


# ------------------------------------------------------------------------------------------------ tasks
def heads_prompts(quick: bool) -> list[tuple[int, int, int]]:
    from aimp.gpt2_units import HEADS, LAYERS, N_PROMPTS

    if quick:
        return [(l, h, p) for (l, h) in QUICK["heads"] for p in QUICK["prompts"]]
    return [(l, h, p) for l in LAYERS for h in HEADS for p in range(N_PROMPTS)]


def exact_tasks(base: Path, quick: bool) -> list[tuple]:
    from aimp.gpt2_units import positions, prompt_length

    out = []
    for rank in range(4):  # position 4 first (the units of the certificate), then 8, 12 and the last one
        for l, h, p in heads_prompts(quick):
            pos = positions(prompt_length(p))[rank]
            out.append((str(base), l, h, p, pos))
    return out


def run_exact(task: tuple) -> dict:
    """One (layer, head, prompt, position): every radius (the head's projection is shared)."""
    base, l, h, p, pos = task
    base = Path(base)
    from aimp.io import jsonable, write_json_atomic
    from aimp.pipeline import evaluate_fq_exact

    done, bad = 0, []
    try:
        for rho in EXACT_RHOS:
            path = exact_path(base, l, h, p, pos, rho)
            if path.exists():
                continue
            out = evaluate_fq_exact(l, h, p, pos, rho, int(os.environ.get("AIMP_MAX_THREADS", "1")))
            write_json_atomic(path, jsonable(out))
            done += 1
            for k in ("valid_greedy", "max_input_in_bracket", "min_input_in_bracket"):
                if out[k] is False:
                    bad.append(f"{path.stem}:{k}")
            if out["exact_status"] != "exact":
                bad.append(f"{path.stem}:unresolved")
        return {"name": f"L{l}H{h}P{p}i{pos}", "units": done, "invalid": bad}
    except Exception as e:
        return {"name": f"L{l}H{h}P{p}i{pos}", "fatal": repr(e), "tb": traceback.format_exc()[-3000:]}


def sdp_tasks(base: Path, quick: bool) -> list[tuple]:
    rhos = QUICK["sdp_rhos"] if quick else SDP_RHOS
    return [(str(base), l, h, p, SDP_POS, r) for r in rhos for (l, h, p) in heads_prompts(quick)]


def run_sdp(task: tuple) -> dict:
    base, l, h, p, pos, rho = task
    base = Path(base)
    from aimp.io import jsonable, write_json_atomic
    from aimp.pipeline import evaluate_fq_exact, evaluate_fq_sdp

    path = sdp_path(base, l, h, p, pos, rho)
    if path.exists():
        return {"name": path.stem, "skipped": True}
    try:
        threads = int(os.environ.get("AIMP_MAX_THREADS", "4"))
        xp = exact_path(base, l, h, p, pos, rho)
        if xp.exists():
            xr = json.loads(xp.read_text())
        else:
            xr = evaluate_fq_exact(l, h, p, pos, rho)
            write_json_atomic(xp, jsonable(xr))
        rec = evaluate_fq_sdp(l, h, p, pos, rho, xr, threads)
        write_json_atomic(path, jsonable(rec))
        bad = [k for k in ("valid", "valid_value_safe", "valid_greedy") if rec[k] is False]
        bad += [f"check {k} violated" for k, v in rec["checks"].items() if v == "violated"]
        return {"name": path.stem, "t": rec["times"]["unit_total_s"], "invalid": bad}
    except Exception as e:
        return {"name": path.stem, "fatal": repr(e), "tb": traceback.format_exc()[-3000:]}


def pool_run(base: Path, fn, tasks: list, workers: int, tps: int, label: str) -> None:
    from aimp import compute

    if workers * tps > compute.thread_budget():
        raise SystemExit(f"workers x threads = {workers * tps} exceeds AIMP_MAX_THREADS = {compute.thread_budget()}")
    log(base, f"=== {label}: {len(tasks)} tasks, {workers} workers x {tps} threads")
    t0, done = time.perf_counter(), 0
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(tps,), maxtasksperchild=50) as pool:
        for out in pool.imap_unordered(fn, tasks):
            done += 1
            if "fatal" in out:
                log(base, f"!!! FATAL {out['name']}: {out['fatal']}\n{out['tb']}")
            elif out.get("invalid"):
                log(base, f"!!! {out['name']}: {out['invalid']}")
            if done % 50 == 0 or done == len(tasks):
                log(base, f"{label}: {done}/{len(tasks)} ({(time.perf_counter() - t0) / 60:.1f} min)")
    log(base, f"=== {label} finished in {(time.perf_counter() - t0) / 60:.1f} min")


# ------------------------------------------------------------------------------------------------ benchmark
UNIT_COLUMNS = ("unit_id", "layer", "head", "prompt", "query_pos", "rho_rel", "radius_rule", "n_keys", "n_varying",
                "max_lo", "max_hi", "min_lo", "min_hi", "nominal_output", "range_of_motion", "status",
                "value_at_max_input", "value_at_min_input", "first_order_certificate")


def write_benchmark(base: Path, dest: Path, allow_partial: bool = False) -> dict:
    """``units.csv`` (one row per unit) and ``angles.csv.gz`` (one row per varying key of every unit)."""
    from aimp.gpt2_units import HEADS, LAYERS, N_PROMPTS, positions, prompt_length, unit_name

    rows, arows, missing = [], [], []
    for l in LAYERS:
        for h in HEADS:
            for p in range(N_PROMPTS):
                for pos in positions(prompt_length(p)):
                    for rho in EXACT_RHOS:
                        f = exact_path(base, l, h, p, pos, rho)
                        if not f.exists():
                            missing.append(f.stem)
                            continue
                        r = json.loads(f.read_text())
                        uid = unit_name(l, h, p, pos, rho)
                        ok = r["exact_status"] == "exact"
                        rows.append({"unit_id": uid, "layer": l, "head": h, "prompt": p, "query_pos": pos,
                                     "rho_rel": f"{rho:g}", "radius_rule": "rho_j = rho_rel * ||zbar_j||",
                                     "n_keys": r["n"], "n_varying": r["n_var"],
                                     "max_lo": repr(r["p_fq_lo"]), "max_hi": repr(r["p_fq_hi"]),
                                     "min_lo": repr(r["min_fq_lo"]), "min_hi": repr(r["min_fq_hi"]),
                                     "nominal_output": repr(r["phi_nominal"]),
                                     "range_of_motion": repr(r["spread_fq"]) if ok else "",
                                     "status": r["exact_status"], "value_at_max_input": repr(r["value_max_input"]),
                                     "value_at_min_input": repr(r["value_min_input"]),
                                     "first_order_certificate": repr(r["r_greedy"])})
                        for j, rj, tmax, tmin in zip(r["varying_keys"], r["rho_j"], r["theta_max"], r["theta_min"]):
                            arows.append((uid, j, repr(float(rj)), repr(float(tmax)), repr(float(tmin))))
    if missing and not allow_partial:
        raise SystemExit(f"benchmark: {len(missing)} units missing (first: {missing[:3]}); run --part exact first")
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "units.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=UNIT_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with gzip.open(dest / "angles.csv.gz", "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(("unit_id", "key", "rho_j", "theta_max", "theta_min"))
        w.writerows(arows)
    return {"units": len(rows), "angle_rows": len(arows), "missing": len(missing)}


# ------------------------------------------------------------------------------------------------ summary
def _slope(x, y) -> float | None:
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return None
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def _q(v) -> str:
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return "n/a"
    return f"{np.median(v):.4g} [{np.quantile(v, .25):.4g}, {np.quantile(v, .75):.4g}] (n={v.size})"


def summary(base: Path, refs: Path) -> dict:
    """Table 5 and the numbers of the exact fixed-query paragraph of Section 7."""
    import pandas as pd

    from aimp.gpt2_units import unit_name

    X = []
    for f in sorted((base / "exact").glob("L*/*.json")):
        r = json.loads(f.read_text())
        X.append({k: r.get(k) for k in ("layer", "head", "prompt", "pos", "rho_rel", "exact_status", "width_rel_max",
                                        "width_rel_min", "spread_fq", "w_t", "valid_greedy", "max_input_in_bracket",
                                        "min_input_in_bracket", "p_fq_lo", "p_fq_hi", "min_fq_lo", "min_fq_hi")}
                 | {"t_s": r["times"]["exact_s"]})
    X = pd.DataFrame(X)
    out: dict = {}
    if len(X):
        out["exact_units"] = int(len(X))
        out["exact_all_resolved"] = int((X.exact_status == "exact").sum())
        out["max_bracket_rel"] = float(np.nanmax(X[["width_rel_max", "width_rel_min"]].to_numpy(float)))
        out["inputs_in_brackets"] = int((X.max_input_in_bracket & X.min_input_in_bracket).sum())
        out["exact_time_median_s"] = float(X.t_s.median())
        wr = pd.concat([(X.p_fq_hi - X.p_fq_lo) / X.spread_fq, (X.min_fq_hi - X.min_fq_lo) / X.spread_fq]).astype(float)
        print(f"exact units: {len(X)}; exact (brackets narrower than 1e-10 of the nominal value spread): "
              f"{out['exact_all_resolved']}; widest bracket {out['max_bracket_rel']:.2e} of the nominal spread; brackets "
              f"relative to the range of motion: median {wr.median():.1e}, max {wr.max():.1e}; worst-case inputs inside "
              f"their brackets: {out['inputs_in_brackets']}; median exact solve {out['exact_time_median_s']:.3f} s")
    Q = []
    for f in sorted((base / "sdp").glob("L*/*.json")):
        r = json.loads(f.read_text())
        Q.append({k: r.get(k) for k in ("layer", "head", "prompt", "pos", "rho_rel", "rel_gap_exact",
                                        "rel_gap_greedy_exact", "gap_over_bound", "lb_strong_gap_exact", "spread_fq",
                                        "valid", "w_t", "range")})
    Q = pd.DataFrame(Q)
    if len(Q):
        print("\nTable 5 (fixed query, position 4; relative gap = (cert - exact max) / exact range of motion)")
        print("rho_rel  units  certificate median [IQR]   max    first order  within 1%")
        for rho, g in Q.groupby("rho_rel"):
            rg = g.rel_gap_exact.astype(float)
            print(f"{rho:<7g}  {len(g):5d}  {np.median(rg):.4g} [{rg.quantile(.25):.4g}, {rg.quantile(.75):.4g}]  "
                  f"{rg.max():.3g}  {g.rel_gap_greedy_exact.astype(float).median():.3g}  {(rg <= 0.01).mean():.0%}")
        sl = {"cert": [], "greedy": [], "spread": []}
        for _, g in Q[Q.rho_rel.isin([0.002, 0.005])].groupby(["layer", "head", "prompt"]):
            if len(g) == 2:
                g = g.sort_values("rho_rel")
                sl["cert"].append(_slope(g.rho_rel, g.rel_gap_exact.astype(float) * g.spread_fq.astype(float)))
                sl["greedy"].append(_slope(g.rho_rel, g.rel_gap_greedy_exact.astype(float) * g.spread_fq.astype(float)))
                sl["spread"].append(_slope(g.rho_rel, g.spread_fq.astype(float)))
        print(f"per-unit slopes on 0.002-0.005: certificate {_q(sl['cert'])}, first order {_q(sl['greedy'])}, "
              f"range of motion {_q(sl['spread'])}")
        gb = Q.gap_over_bound.astype(float)
        print(f"gap / second-order bound (25): max {gb.max():.3g}, median {gb.median():.3g}")
        lg = (Q.lb_strong_gap_exact.astype(float) / Q["range"].astype(float)).abs()
        print(f"multistart vs exact worst case: max |p* - lb| / nominal value spread = {lg.max():.2e} "
              f"(within 1e-9 of the nominal spread on {(lg <= 1e-9).mean():.1%} of units)")
        print(f"valid on {int(Q.valid.astype(bool).sum())} / {len(Q)} units")
        out["table3_units"] = int(len(Q))
    # secant width with a varying query / with a fixed query, same units (position 4)
    lad = refs / "gpt2_ladder" / "units"
    if len(X) and lad.exists():
        rat = []
        for _, r in X[X.pos == 4].iterrows():
            f = lad / f"L{int(r.layer):02d}" / f"{unit_name(int(r.layer), int(r.head), int(r.prompt), 4, r.rho_rel)}.json"
            if f.exists():
                rat.append((r.rho_rel, json.loads(f.read_text())["w_t"] / r.w_t))
        if rat:
            R = pd.DataFrame(rat, columns=["rho_rel", "ratio"])
            print("secant width varying / fixed query (median per radius): " +
                  ", ".join(f"{k:g}: {v:.3g}" for k, v in R.groupby("rho_rel").ratio.median().items()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", choices=("all", "exact", "sdp", "benchmark", "summary"), default="all")
    ap.add_argument("--quick", action="store_true", help="2 heads x 1 prompt (a few minutes)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--threads-per-solve", type=int, default=None)
    ap.add_argument("--out", default=None, help="results directory (default: $AIMP_RESULTS_DIR/exact_fixed_query)")
    ap.add_argument("--refs", default=None, help="directory holding gpt2_ladder/ (default: $AIMP_RESULTS_DIR)")
    ap.add_argument("--benchmark-dir", default=str(ROOT / "benchmark"))
    ap.add_argument("--allow-partial", action="store_true", help="write the benchmark from the units computed so far")
    args = ap.parse_args()
    from aimp import compute, gpt2_units
    from aimp.io import results_dir

    base = out_dir(args)
    if args.quick and args.out is None:
        base = results_dir() / "exact_fixed_query_quick"
    budget = compute.thread_budget()
    gpt2_units.prepare()
    parts = ("exact", "sdp", "summary") if args.part == "all" else (args.part,)
    if args.part == "all" and not args.quick:
        parts = ("exact", "benchmark", "sdp", "summary")
    if "exact" in parts:
        w = args.workers or budget
        pool_run(base, run_exact, [t for t in exact_tasks(base, args.quick)
                                   if not all(exact_path(base, *t[1:], r).exists() for r in EXACT_RHOS)],
                 w, args.threads_per_solve or 1, "exact")
    if "benchmark" in parts:
        print(write_benchmark(base, Path(args.benchmark_dir), args.allow_partial or args.quick))
    if "sdp" in parts:
        tps = args.threads_per_solve or min(4, budget)
        w = args.workers or max(1, budget // tps)
        pool_run(base, run_sdp, [t for t in sdp_tasks(base, args.quick) if not sdp_path(base, *t[1:]).exists()],
                 w, tps, "sdp")
    if "summary" in parts:
        summary(base, Path(args.refs) if args.refs else results_dir())


if __name__ == "__main__":
    main()
