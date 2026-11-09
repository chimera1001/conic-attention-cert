"""The certificates on a second trained model, gpt-oss-20b (the computational results of Section 7 repeated on
``openai/gpt-oss-20b``: 24 layers, 64 query heads per layer, 8 key-value groups, rotary position embeddings with YaRN, one
attention sink per head, alternating full and sliding-window attention).

A head reads its tokens through a 192-dimensional subspace, as in GPT-2 small. The rotary maps act after the projection, so
the score of key ``j`` is ``(W_Q z_i + b_Q)^T M_j (W_K z_j + b_K)`` with ``M_j = R_i^T R_j / 8``. They are folded into each
token's set (an invertible linear map per token). The sink is an extra key with a fixed score and value zero.
:mod:`aimp.gptoss` builds the units (``model="gpt-oss-20b"`` of :func:`aimp.gpt2_units.make_instance`); the first run
downloads the model (about 14 GB), runs the five prompts once on CPU (float32, about 85 GB of memory), checks the
extraction and caches what the units need (5.1 GB under ``$AIMP_CACHE_DIR/gpt-oss-20b``). No later step loads the model.

Parts (``--part``; default ``x f v l``, then the summary):

* ``calib`` -- the radius calibration: score intervals of the 192 sampled heads (8 per layer, one per key-value group,
  ``h in {0, 9, ..., 63}``; prompt 0, position 4, varying query) at seven radii; ``r0`` = the largest ``rho_rel`` with
  median secant width ``w_t <= 0.3``. The radii below are those of this calibration (``r0 = 0.0002``).
* ``x`` -- every one of the 1,536 query heads, five prompts, query positions 4 and the last token, ``R4``: the exact maximum and
  minimum with the query fixed (Theorem 7), the first-order certificate and the Vertex-Softmax bound; no semidefinite
  solve. ``--benchmark`` writes the benchmark files from these records (``benchmark/gpt-oss-20b``: ``units.csv``,
  ``angles.csv.gz``, ``prompts.txt``; the format of ``benchmark/SCHEMA.md``).
* ``f`` -- the sampled heads, five prompts, position 4, ``R4``, query fixed: the certificate, the first-order and vertex
  bounds, the exact references, IBP, CROWN and alpha-CROWN.
* ``v`` -- the same heads, ``R3``, query varying: the certificate with the re-solve rule of Appendix B, the best feasible
  values of the maximization and the minimization, the bounds above.
* ``l`` -- the units of ``f`` with the logit-margin specification and ``R3``.
* ``--summarize`` (also run after the parts) -- ``table4.csv``, ``table5.csv``, ``crown_vs_summary.csv``, ``lm_summary.csv``
  (the columns of ``figdata/P4b_crown_summary.csv`` plus ``VS_*``; the files ``figdata/P8_crown_vs_summary.csv`` and
  ``figdata/P8_lm_summary.csv``), ``x_summary.csv`` (``figdata/P8_x_summary.csv``), ``gpt2_compare.csv`` (Figure 6,
  ``figdata/P8_gpt2_compare.csv``; needs the GPT-2 records of ``experiments/jmlr_extensions.py --part vs-gpt2``),
  ``units.csv`` and a copy of the benchmark files, in ``<out>/summary``; the tables are printed.

Relative gaps as in Table 6: fixed query ``(ub - p*) / (p* - p_min)``, varying query ``(ub - lb) / (lb - phi_min)``. Every unit
checks the validity of every bound against its reference, the theorem checks and E2 (the instance reproduces the head's
nominal pattern and output computed from the raw weights to 1e-10); a failure stops the part.

Usage::

    python experiments/gpt_oss.py --part calib --workers 12
    python experiments/gpt_oss.py --part x --workers 12
    python experiments/gpt_oss.py --part f v l --workers 3 --threads-per-solve 4
    python experiments/gpt_oss.py --summarize
    python experiments/gpt_oss.py --benchmark
    python experiments/gpt_oss.py --quick --workers 2 --threads-per-solve 2
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

from aimp import compute  # noqa: E402
from aimp import gptoss as G  # noqa: E402
from aimp import gpt2_units as U  # noqa: E402
from aimp.io import jsonable, results_dir, write_json_atomic  # noqa: E402
from jmlr_extensions import (ALL_M, METHODS, _csv, _gaps, _iqr, _log, _print_table, _slope, _slopes,  # noqa: E402
                             _summary_rows, run_part, v_checks)

MODEL = G.MODEL_NAME
PARTS = ("calib", "x", "f", "v", "l")
POS = 4
CAL_RHOS = (1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2)
R4 = (0.0002, 0.0004, 0.001, 0.002)  # r0 x {1, 2, 5, 10}, r0 = 0.0002 (part calib)
R3 = R4[:3]
QUICK = {"heads": ((5, 9), (11, 36)), "prompts": (0,), "R4": R4[:2], "R3": R3[:2]}
LABEL = {"fq": "fixed", "free": "varying"}
WT_BINS = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, np.inf)  # bins of the secant width w_t (comparison with GPT-2 small)


def sample_heads() -> list[tuple[int, int]]:
    return [(l, h) for l in range(G.N_LAYERS) for h in G.SAMPLE_HEADS]


def all_heads() -> list[tuple[int, int]]:
    return [(l, h) for l in range(G.N_LAYERS) for h in range(G.N_HEADS)]


# ============================================================================================ parts
def cal_path(out: Path, l, h) -> Path:
    return out / "calib" / f"L{l:02d}H{h:02d}.json"


def run_calib(task) -> dict:
    """Score intervals of one head (prompt 0, position 4, varying query) at the seven radii; E2 at the nominal point."""
    l, h, out = task
    path = cal_path(out, l, h)
    if path.exists():
        return {"name": path.stem, "skipped": True}
    try:
        from aimp import bounds

        t0 = time.perf_counter()
        rows = []
        for rho in CAL_RHOS:
            inst = U.make_instance(l, h, 0, POS, rho, model=MODEL)
            si = G.score_intervals_roles(inst, threads=1)
            rows.append({"rho_rel": rho, "w_t": bounds.t_width(si["s_lo"], si["s_hi"], "tight"),
                         "Delta": float(np.max(si["s_hi"] - si["s_lo"]))})
        e2 = G.e2_check(U.make_instance(l, h, 0, POS, 1e-3, model=MODEL))
        write_json_atomic(path, jsonable({"layer": l, "head": h, "rows": rows, "E2": e2}))
        return {"name": path.stem, "t": time.perf_counter() - t0, "bad": [] if e2["E2_ok"] else ["E2"]}
    except Exception as e:
        return {"name": path.stem, "fatal": repr(e), "tb": traceback.format_exc()[-3000:]}


def x_path(out: Path, l, h, p, pos, rho) -> Path:
    return out / "x" / f"L{l:02d}" / f"{U.unit_name(l, h, p, pos, rho)}.json"


def run_x(task) -> dict:
    """One exact fixed-query unit (Theorem 7), the first-order certificate, the vertex bound and E2."""
    l, h, p, pos, rho, out = task
    path = x_path(out, l, h, p, pos, rho)
    if path.exists():
        return {"name": path.stem, "skipped": True}
    try:
        from aimp import pipeline

        t0 = time.perf_counter()
        r = pipeline.evaluate_fq_exact(l, h, p, pos, rho, threads=1, model=MODEL)
        fq = U.make_instance(l, h, p, pos, rho, fixed_query=True, model=MODEL)
        r.update({"kv": h // G.GROUP, **G.e2_check(fq), **{f"{k}": v for k, v in G.nominal_value_range(fq).items()},
                  "v_checks": v_checks(r["r_VS"], r["p_fq_hi"], r["r_greedy"], r["range"]), "wall_s": time.perf_counter() - t0})
        write_json_atomic(path, jsonable(r))
        bad = [k for k in ("valid_greedy", "valid_VS", "max_input_in_bracket", "min_input_in_bracket", "E2_ok") if r.get(k) is False]
        bad += [k for k in ("V1", "V2") if r["v_checks"][k] is False]
        if r["exact_status"] != "exact":
            bad.append("unresolved bracket")
        return {"name": path.stem, "t": r["wall_s"], "bad": bad}
    except Exception as e:
        return {"name": path.stem, "fatal": repr(e), "tb": traceback.format_exc()[-3000:]}


def unit_path(out: Path, part: str, l, h, p, rho) -> Path:
    return out / part / f"L{l:02d}" / f"{U.unit_name(l, h, p, POS, rho)}.json"


def run_unit(task) -> dict:
    """One unit of parts ``f`` (fixed query, spec a), ``v`` (varying query, spec a) or ``l`` (fixed query, logit margin):
    the certificate's procedure, the references, ``r_VS``, E2, IBP / CROWN / alpha-CROWN."""
    part, l, h, p, rho, out, threads = task
    path = unit_path(out, part, l, h, p, rho)
    if path.exists():
        return {"name": path.stem, "skipped": True}
    try:
        from aimp import lirpa, pipeline

        t0 = time.perf_counter()
        spec = "lm" if part == "l" else "a"
        if part in ("f", "l"):
            xr = pipeline.evaluate_fq_exact(l, h, p, POS, rho, threads=threads, spec=spec, model=MODEL)
            rec = pipeline.evaluate_fq_sdp(l, h, p, POS, rho, xr, threads=threads, spec=spec, model=MODEL)
            rec.update({k: xr[k] for k in ("min_fq_lo", "min_fq_hi", "phi_min_cert", "max_input_in_bracket",
                                           "min_input_in_bracket", "rel_gap_greedy", "w_t", "Delta")})
            ref_hi = rec["p_fq_hi"]
            inst = U.make_instance(l, h, p, POS, rho, fixed_query=True, spec=spec, model=MODEL)
        else:
            rec = pipeline.evaluate_unit(l, h, p, POS, rho, threads=threads, spec=spec, model=MODEL)
            ref_hi = rec["lb_strong"]
            inst = U.make_instance(l, h, p, POS, rho, spec=spec, model=MODEL)
        rec.pop("perkey", None)
        t1 = time.perf_counter()
        b = lirpa.bounds(inst, U.unit_seed(l, h, p, POS, rho) + 61, METHODS, threads=threads)
        rec.update({"part": part, "kind": "free" if part == "v" else "fq", "kv": h // G.GROUP, **G.e2_check(inst),
                    "range_no_sink": G.nominal_value_range(inst)["range_no_sink"], "lirpa": b,
                    "lirpa_s": time.perf_counter() - t1, "v_checks": v_checks(rec["r_VS"], ref_hi, rec["r_greedy"], rec["range"]),
                    "wall_s": time.perf_counter() - t0})
        write_json_atomic(path, jsonable(rec))
        c = rec["v_checks"]
        bad = [k for k in ("V1", "V2") if c[k] is False]
        bad += [k for k in ("valid", "valid_value_safe", "valid_greedy", "valid_VS", "E2_ok") if rec.get(k) is False]
        bad += [f"theorem {k}" for k, v in rec["checks"].items() if v == "violated"]
        if not b["gate_ok"]:
            bad.append("lirpa forward gate")
        return {"name": path.stem, "t": rec["wall_s"], "bad": bad}
    except Exception as e:
        return {"name": path.stem, "fatal": repr(e), "tb": traceback.format_exc()[-3000:]}


# ============================================================================================ summaries
def summarize_calib(out: Path, heads) -> None:
    import pandas as pd

    rows = [r | {"layer": rec["layer"], "head": rec["head"]} for l, h in heads if cal_path(out, l, h).exists()
            for rec in [json.loads(cal_path(out, l, h).read_text())] for r in rec["rows"]]
    if not rows:
        return
    D = pd.DataFrame(rows)
    T = D.groupby("rho_rel").agg(heads=("w_t", "size"), median_w_t=("w_t", "median"),
                                 q25_w_t=("w_t", lambda x: x.quantile(.25)), q75_w_t=("w_t", lambda x: x.quantile(.75)),
                                 median_Delta=("Delta", "median")).reset_index()
    ok = T[T.median_w_t <= 0.3].rho_rel
    r0 = float(ok.max()) if len(ok) else None
    _csv(T, out / "summary" / "calibration.csv")
    print("\nradius calibration (median w_t over the heads):")
    for _, r in T.iterrows():
        print(f"  rho_rel {r.rho_rel:g}: median w_t {r.median_w_t:.4g} [{r.q25_w_t:.3g}, {r.q75_w_t:.3g}]")
    print(f"  r0 = {r0} (largest rho_rel with median w_t <= 0.3); the radii of this script assume r0 = {R4[0]}")


def x_records(out: Path) -> list[dict]:
    """The records of part x (``<out>/x/L<ll>/<unit>.json``)."""
    return [json.loads(f.read_text()) for f in sorted((out / "x").glob("*/*.json"))]


def summarize_x(out: Path, recs: list[dict] | None = None) -> None:
    """Part x by query position and radius in the columns of ``figdata/P8_x_summary.csv`` (``x_summary.csv``; the share of
    units resolved to ``1e-10`` of the nominal value spread, the relative gaps of the first-order and vertex bounds, the
    share of the first-order gap the vertex bound removes, the range of motion over the nominal value spread, the bracket
    width over the range of motion, the time per unit), the per-unit slopes (``x_slopes.csv``), one row per unit
    (``x_units.csv``) and the benchmark files in ``<out>/summary/benchmark`` (:func:`write_benchmark`, partial allowed)."""
    import pandas as pd

    recs = x_records(out) if recs is None else recs
    if not recs:
        return
    D = pd.DataFrame([{k: r.get(k) for k in ("layer", "head", "kv", "prompt", "pos", "rho_rel", "n", "w_t", "p_fq", "min_fq",
                                             "p_fq_lo", "p_fq_hi", "spread_fq", "r_greedy", "r_VS", "exact_status", "range",
                                             "range_no_sink")} | {"time_s": r["times"]["total_s"]} for r in recs])
    D["position"] = np.where(D.pos == POS, "4", "last")
    D["rel_gap_greedy"] = (D.r_greedy - D.p_fq) / D.spread_fq
    D["rel_gap_vs"] = (D.r_VS - D.p_fq) / D.spread_fq
    D["VS_removed"] = (D.r_greedy - D.r_VS) / (D.r_greedy - D.p_fq)
    D["width_over_spread"] = (D.p_fq_hi - D.p_fq_lo) / D.spread_fq

    def mx(x):
        x = pd.to_numeric(x, errors="coerce").dropna()
        return float(x.max()) if len(x) else None

    S = []
    for (pk, rho), g in D.groupby(["position", "rho_rel"]):
        gr, gv = _iqr(g.rel_gap_greedy), _iqr(g.rel_gap_vs)
        S.append({"position": pk, "rho_rel": rho, "units": len(g), "resolved": int((g.exact_status == "exact").sum()),
                  "share_resolved": float((g.exact_status == "exact").mean()), "median_w_t": g.w_t.median(),
                  "greedy_median": gr["median"], "greedy_q25": gr["q25"], "greedy_q75": gr["q75"], "greedy_max": mx(g.rel_gap_greedy),
                  "VS_median": gv["median"], "VS_q25": gv["q25"], "VS_q75": gv["q75"], "VS_max": mx(g.rel_gap_vs),
                  "VS_removed_median": _iqr(g.VS_removed)["median"],
                  "spread_over_range_median": _iqr(g.spread_fq / g["range"])["median"],
                  "bracket_width_over_spread_median": _iqr(g.width_over_spread)["median"],
                  "bracket_width_over_spread_max": mx(g.width_over_spread),
                  "time_median_s": g.time_s.median(), "n_keys_median": g.n.median()})
    S = pd.DataFrame(S)
    sl = []
    for (pk, l, h, p), g in D.groupby(["position", "layer", "head", "prompt"]):
        g = g.sort_values("rho_rel")
        sl.append({"position": pk, "layer": l, "head": h, "prompt": p, "slope_greedy": _slope(g.rho_rel, g.r_greedy - g.p_fq),
                   "slope_VS": _slope(g.rho_rel, g.r_VS - g.p_fq), "slope_spread": _slope(g.rho_rel, g.spread_fq)})
    SL = pd.DataFrame(sl).groupby("position")[["slope_greedy", "slope_VS", "slope_spread"]].median().reset_index()
    _csv(D, out / "summary" / "x_units.csv")
    _csv(S, out / "summary" / "x_summary.csv")
    _csv(SL, out / "summary" / "x_slopes.csv")
    print("\nPart x (exact, fixed query): relative gaps of first order and Vertex-Softmax")
    print(S[["position", "rho_rel", "units", "share_resolved", "median_w_t", "greedy_median", "VS_median",
             "VS_removed_median"]].to_string(index=False))
    print("median per-unit slopes over the radii:\n" + SL.to_string(index=False))
    print(write_benchmark(recs, out / "summary" / "benchmark", allow_partial=True))


#: the columns of ``benchmark/gpt-oss-20b/units.csv``: those of the GPT-2 benchmark plus the key-value group and the
#: Vertex-Softmax bound
BENCH_COLUMNS = ["unit_id", "layer", "head", "kv_group", "prompt", "query_pos", "rho_rel", "radius_rule", "n_keys", "n_varying",
                 "max_lo", "max_hi", "min_lo", "min_hi", "nominal_output", "range_of_motion", "status", "value_at_max_input",
                 "value_at_min_input", "first_order_certificate", "vertex_softmax_bound"]


def write_benchmark(recs: list[dict], dest: Path, allow_partial: bool = False) -> dict:
    """The benchmark of gpt-oss-20b from the records of part x, in the format of ``benchmark/SCHEMA.md``: ``units.csv``
    (one row per unit, ordered by layer, head, prompt, query position and radius; numbers with 17 significant digits),
    ``angles.csv.gz`` (one row per varying key: its radius and the canonical angles of the maximizing and the minimizing
    input; written with a fixed gzip time stamp, so that the bytes depend only on the records) and ``prompts.txt``. Without
    ``allow_partial`` every one of the 1,536 heads x five prompts x {4, last position} x ``R4`` must be present."""
    import io
    import shutil

    by = {(r["layer"], r["head"], r["prompt"], r["pos"], r["rho_rel"]): r for r in recs}
    last = {}
    for (_, _, p, pos, _) in by:
        last[p] = max(last.get(p, POS), pos)
    rows, arows = [], []
    for k in sorted(by):
        r = by[k]
        uid = U.unit_name(r["layer"], r["head"], r["prompt"], r["pos"], r["rho_rel"])
        ok = r["exact_status"] == "exact"
        rows.append({"unit_id": uid, "layer": r["layer"], "head": r["head"], "kv_group": r["kv"], "prompt": r["prompt"],
                     "query_pos": r["pos"], "rho_rel": f"{r['rho_rel']:g}", "radius_rule": "rho_j = rho_rel * ||zbar_j||",
                     "n_keys": r["n"], "n_varying": r["n_var"], "max_lo": repr(r["p_fq_lo"]), "max_hi": repr(r["p_fq_hi"]),
                     "min_lo": repr(r["min_fq_lo"]), "min_hi": repr(r["min_fq_hi"]), "nominal_output": repr(r["phi_nominal"]),
                     "range_of_motion": repr(r["spread_fq"]) if ok else "", "status": r["exact_status"],
                     "value_at_max_input": repr(r["value_max_input"]), "value_at_min_input": repr(r["value_min_input"]),
                     "first_order_certificate": repr(r["r_greedy"]), "vertex_softmax_bound": repr(r["r_VS"])})
        for j, rj, tmax, tmin in zip(r["varying_keys"], r["rho_j"], r["theta_max"], r["theta_min"]):
            arows.append((uid, j, repr(float(rj)), repr(float(tmax)), repr(float(tmin))))
    expected = {(l, h, p, pos, rho) for l, h in all_heads() for p in range(5) for pos in (POS, last.get(p)) for rho in R4}
    missing = sorted(expected - set(by), key=str)
    if not allow_partial and (missing or len(last) < 5 or any(v == POS for v in last.values())):
        raise SystemExit(f"benchmark: {len(missing)} units of part x missing (first: {missing[:3]}); run --part x first")
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "units.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=BENCH_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with open(dest / "angles.csv.gz", "wb") as raw, gzip.GzipFile(str(dest / "angles.csv.gz"), "wb", fileobj=raw, mtime=0) as gz, \
            io.TextIOWrapper(gz, encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(("unit_id", "key", "rho_j", "theta_max", "theta_min"))
        w.writerows(arows)
    shutil.copyfile(ROOT / "data" / "prompts.txt", dest / "prompts.txt")
    return {"benchmark": str(dest), "units": len(rows), "angle_rows": len(arows), "missing": len(missing),
            "not_exact": sum(r["status"] != "exact" for r in rows)}


def summarize_units(out: Path, R4_, R3_, gpt2_units: Path | None = None) -> None:
    """Parts f, v, l: one row per unit (``units.csv``), the analogues of Tables 4-7 (``table4.csv``, ``table5.csv``,
    ``crown_vs_summary.csv``, ``lm_summary.csv``: the columns of ``figdata/P8_crown_vs_summary.csv`` and
    ``figdata/P8_lm_summary.csv``), the slopes, and the comparison with GPT-2 small at matched secant width
    (:func:`gpt2_compare`) when ``gpt2_units`` exists."""
    import pandas as pd

    rows = []
    for part in ("f", "v", "l"):
        for f in sorted((out / part).glob("*/*.json")):
            r = json.loads(f.read_text())
            fq = part != "v"
            row = {"part": part, "kind": r["kind"], "spec": r["spec"], "layer": r["layer"], "head": r["head"], "kv": r["kv"],
                   "prompt": r["prompt"], "rho_rel": r["rho_rel"], "n": r["n"], "w_t": r.get("w_t"), "Delta": r.get("Delta"),
                   "rad_c": r.get("rad_c"), "S": r.get("S"), "knapsack_term": r.get("F0vf_term_knapsack"),
                   "bound_F0v_final": r.get("bound_F0v_final"), "range": r["range"], "range_no_sink": r["range_no_sink"],
                   "reference": r["p_fq"] if fq else r["lb_strong"], "reference_hi": r["p_fq_hi"] if fq else r["lb_strong"],
                   "spread": r["spread_fq"] if fq else r["lb_strong"] - r["phi_min_strong"], "cert": r["cert"],
                   "VS_ub": r["r_VS"], "greedy_ub": r["r_greedy"], "cert_ub": r["cert"], "retol": r["retol"],
                   "time_cert_s": r["times"].get("total_s"), "time_unit_s": r["times"].get("unit_total_s"),
                   "VS_time_s": r["times"].get("controls_s")}
            for m in METHODS:
                x = r["lirpa"]["methods"].get(m) or {}
                row.update({f"{m}_ub": x.get("ub"), f"{m}_time_s": x.get("time_s")})
            rows.append(row)
    if not rows:
        return
    D = pd.DataFrame(rows)
    for m in METHODS:
        D[f"{m}_ub"] = pd.to_numeric(D[f"{m}_ub"], errors="coerce")
    _gaps(D)
    D["cert_gap"] = D.cert - D.reference
    _csv(D, out / "summary" / "units.csv")
    tc = {m: f"{m}_time_s" for m in METHODS} | {"VS": "VS_time_s"}
    A, L_ = D[D.spec == "a"], D[D.spec == "lm"]
    if len(A):
        S6 = _summary_rows(A, tc)
        _csv(S6, out / "summary" / "crown_vs_summary.csv")
        _print_table(S6, "gpt-oss-20b, spec a (Table 6 with the vertex columns)")
        T4 = []
        for (kind, rho), g in A.groupby(["kind", "rho_rel"]):
            T4.append({"kind": kind, "rho_rel": rho, "units": len(g), "median_w_t": g.w_t.median(),
                       "median_rel_gap_cert": g.cert_rel_gap.median(), "median_rel_gap_first_order": g.greedy_rel_gap.median(),
                       "median_gap_over_range": (g.cert_gap / g["range"]).median(),
                       "share_within_1pct": float((g.cert_rel_gap <= 0.01).mean()),
                       "share_within_10pct": float((g.cert_rel_gap <= 0.10).mean())})
        _csv(pd.DataFrame(T4), out / "summary" / "table4.csv")
        F = A[A.part == "f"]
        T5 = []
        for rho, g in F.groupby("rho_rel"):
            s = _iqr(g.cert_rel_gap)
            T5.append({"rho_rel": rho, "units": len(g), "cert_median": s["median"], "cert_q25": s["q25"], "cert_q75": s["q75"],
                       "cert_max": float(g.cert_rel_gap.max()), "first_order_median": g.greedy_rel_gap.median(),
                       "share_within_1pct": float((g.cert_rel_gap <= 0.01).mean())})
        sl = []
        for (l, h, p), g in F.groupby(["layer", "head", "prompt"]):
            g = g.sort_values("rho_rel")
            g2 = g[g.rho_rel.isin(R4_[:2])]
            sl.append({"two": _slope(g2.rho_rel, g2.cert_gap), "all": _slope(g.rho_rel, g.cert_gap) if len(g) == len(R4_) else None,
                       "two_fo": _slope(g2.rho_rel, g2.greedy_ub - g2.reference)})
        sl = pd.DataFrame(sl)
        if len(sl):
            T5.append({"rho_rel": "slope (two smallest radii)", "units": len(sl), "cert_median": _iqr(sl["two"])["median"],
                       "first_order_median": _iqr(sl["two_fo"])["median"]})
            T5.append({"rho_rel": "slope (all four radii)", "units": int(sl["all"].notna().sum()),
                       "cert_median": _iqr(sl["all"])["median"]})
        _csv(pd.DataFrame(T5), out / "summary" / "table5.csv")
        SL = _slopes(A, {"all_radii": None, "two_smallest": R3_[:2]})
        _csv(SL, out / "summary" / "slopes.csv")
        if gpt2_units is not None:
            gpt2_compare(A, gpt2_units, out)
    if len(L_):
        S7 = _summary_rows(L_, tc)
        _csv(S7, out / "summary" / "lm_summary.csv")
        _print_table(S7, "gpt-oss-20b, logit margin (Table 7 analogue)")


def gpt2_compare(A, gpt2_units: Path, out: Path) -> None:
    """The data of Figure 6 (``gpt2_compare.csv``, the columns of ``figdata/P8_gpt2_compare.csv``): by query type and bin of
    the secant width ``w_t`` (:data:`WT_BINS`, closed on the left), the units, the median ``w_t`` and the median and
    quartiles of the relative gaps of the certificate, first order, CROWN, alpha-CROWN and the vertex bound, for
    gpt-oss-20b (parts f and v, ``A``) and GPT-2 small (the 5,760 units of Table 6 with the vertex bound,
    ``vs_gpt2_units.csv`` of ``experiments/jmlr_extensions.py --part vs-gpt2``). Figure 6 draws the bins that hold at
    least 30 units of each model."""
    import pandas as pd

    if not gpt2_units.exists():
        print(f"(comparison with GPT-2 small skipped: {gpt2_units} not found; run experiments/jmlr_extensions.py --part vs-gpt2)")
        return
    cols = ["kind", "rho_rel", "w_t"] + [f"{m}_rel_gap" for m in ("cert", "greedy", "CROWN", "alpha-CROWN", "VS")]
    X = pd.concat([pd.read_csv(gpt2_units)[cols].assign(model="gpt2"), A[cols].assign(model="gpt-oss-20b")],
                  ignore_index=True)
    X["w_bin"] = pd.cut(X.w_t, WT_BINS, right=False)
    rows = []
    for (kind, b, model), g in X.groupby(["kind", "w_bin", "model"], observed=True):
        row = {"kind": kind, "w_t_bin": str(b), "w_lo": b.left, "w_hi": b.right, "model": model, "units": len(g),
               "median_w_t": g.w_t.median()}
        for m in ("cert", "greedy", "CROWN", "alpha-CROWN", "VS"):
            s_ = _iqr(g[f"{m}_rel_gap"])
            row.update({f"{m}_median": s_["median"], f"{m}_q25": s_["q25"], f"{m}_q75": s_["q75"], f"{m}_n": s_["n"]})
        rows.append(row)
    _csv(pd.DataFrame(rows), out / "summary" / "gpt2_compare.csv")


# ============================================================================================ main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", nargs="+", choices=PARTS, default=["x", "f", "v", "l"])
    ap.add_argument("--quick", action="store_true", help=f"heads {list(QUICK['heads'])}, prompt 0, two radii (minutes)")
    ap.add_argument("--heads", default=None, help="comma-separated layer:head list (overrides the head sample)")
    ap.add_argument("--prompts", default=None, help="comma-separated prompt indices")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads-per-solve", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help="default results_dir()/gpt_oss (with --quick .../gpt_oss_quick)")
    ap.add_argument("--summarize", action="store_true", help="only summarize the stored records")
    ap.add_argument("--refs", type=Path, default=None, help="root of the GPT-2 records for the comparison at matched secant "
                    "width (default results_dir(); reads jmlr/summary/vs_gpt2_units.csv, with --quick jmlr_quick/...)")
    ap.add_argument("--benchmark", action="store_true", help="only write the benchmark files from the records of part x "
                    "(units.csv, angles.csv.gz, prompts.txt) into --benchmark-dir")
    ap.add_argument("--benchmark-dir", type=Path, default=ROOT / "benchmark" / "gpt-oss-20b")
    ap.add_argument("--allow-partial", action="store_true", help="--benchmark: write the units computed so far")
    ap.add_argument("--force", action="store_true", help="re-extract the cache even if one exists (the current directory "
                    "is kept as <name>.replaced-<time>); without it an incomplete cache is an error, never re-extracted")
    a = ap.parse_args(argv)
    out = a.out or results_dir() / ("gpt_oss_quick" if a.quick else "gpt_oss")
    sample = list(QUICK["heads"]) if a.quick else sample_heads()
    xheads = list(QUICK["heads"]) if a.quick else all_heads()
    if a.heads:
        sample = xheads = [tuple(int(v) for v in x.split(":")) for x in a.heads.split(",")]
    prompts = list(QUICK["prompts"]) if a.quick else list(range(5))
    if a.prompts:
        prompts = [int(x) for x in a.prompts.split(",")]
    R4_, R3_ = (QUICK["R4"], QUICK["R3"]) if a.quick else (R4, R3)
    if a.benchmark:
        print(write_benchmark(x_records(out), a.benchmark_dir, allow_partial=a.allow_partial or a.quick))
        return
    if not a.summarize:
        if a.workers * a.threads_per_solve > compute.thread_budget():
            raise SystemExit(f"workers x threads = {a.workers * a.threads_per_solve} exceeds AIMP_MAX_THREADS = "
                             f"{compute.thread_budget()}")
        ex = G.prepare(force=a.force, log=lambda m: _log(out, m))
        _log(out, f"gpt-oss extraction checks: E1 {ex['E1_ok']} (worst {ex['E1_worst_rel']:.1e}), E3 logits {ex['E3_logits_ok']} "
                  f"(worst {ex['E3_worst_logits_rel']:.1e}), E3 linear {ex['E3_lin_ok']} (worst {ex['E3_worst_lin_rel']:.1e})")
        if "calib" in a.part:
            run_part(run_calib, [(l, h, out) for l, h in sample if not cal_path(out, l, h).exists()], a.workers, 1, out, "calib")
        if "x" in a.part:
            jobs = [(l, h, p, pos, rho, out) for rank in (0, 1) for l, h in xheads for p in prompts
                    for pos in [(POS, G.n_tokens(p) - 1)[rank]] for rho in R4_]
            run_part(run_x, [j for j in jobs if not x_path(out, *j[:5]).exists()], a.workers, 1, out, "x")
        for part, rhos in (("f", R4_), ("v", R3_), ("l", R3_)):
            if part in a.part:
                jobs = [(part, l, h, p, rho, out, a.threads_per_solve) for rho in rhos for l, h in sample for p in prompts]
                run_part(run_unit, [j for j in jobs if not unit_path(out, *j[:5]).exists()], a.workers, a.threads_per_solve,
                         out, part)
    if "calib" in a.part or a.summarize:
        summarize_calib(out, sample)
    summarize_x(out)
    refs = a.refs or results_dir()
    summarize_units(out, R4_, R3_, refs / ("jmlr_quick" if a.quick else "jmlr") / "summary" / "vs_gpt2_units.csv")


if __name__ == "__main__":
    main()
