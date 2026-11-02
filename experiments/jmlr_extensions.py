"""Two extensions of the computational results: the Vertex-Softmax baseline and the logit-margin specification.

Parts (``--part``; default: all three, then the summary):

* ``vs-gpt2`` -- the Vertex-Softmax bound ``r_VS`` (:func:`aimp.bounds.vertex_softmax`: the exact maximum of
  ``sum_j softmax(s)_j h_j^max`` over the box of certified scores, at the threshold vertices of the remark on scores in a
  box) on the 5,760 GPT-2 units of the bound-propagation comparison (Table 6): every head, five prompts, query position 4,
  rho_rel in {0.002, 0.005, 0.01, 0.02} with the query fixed and {0.001, 0.002, 0.005, 0.01} with it varying. The score
  intervals and support values are those of the certificate of the same unit (no semidefinite relaxation is solved);
  ``r_greedy`` and ``r_VS_cs`` (the same maximum with Cauchy-Schwarz score intervals) are recorded too.
* ``vs-synthetic`` -- ``r_VS`` on the stored synthetic instances of ``experiments/synthetic.py`` (``slopes`` and
  ``width`` parts), from their stored score intervals; ``r_greedy`` is recomputed and compared with the stored ``ps_auto``.
* ``lm`` -- the logit-margin specification (``spec="lm"`` of :func:`aimp.gpt2_units.make_instance`: ``c = -c_m``, the
  worst decrease of the head's direct contribution to the margin between the model's two top next-token logits at
  position 4, with the scale of the final normalization frozen) on every head, five prompts, position 4, rho_rel in
  {0.002, 0.005, 0.01}: with the query fixed, the exact maximum and minimum (Theorem 7), the certificate, ``r_greedy`` and
  ``r_VS``; with it varying, the certificate with the re-solve rule of Appendix B, ``r_greedy``, ``r_VS`` and the best
  feasible values of the maximization and the minimization; on both, IBP, CROWN and alpha-CROWN (auto_LiRPA defaults).
  :func:`aimp.gpt2_units.prepare_lm` builds the specification and checks it first (the numpy logits reproduce the model's
  to 1e-10; the frozen-scale logit difference of the model's own final normalization and head changes by ``c_m^T delta``
  to 1e-6; ``c^T ybar`` equals the head's direct contribution in the forward pass).
* ``--summarize`` (also run after the parts) -- joins the records with the references and writes to ``<out>/summary``:
  ``crown_vs_summary.csv`` (the columns of ``figdata/P4b_crown_summary.csv`` plus ``VS_*``), ``vs_gpt2_slopes.csv``,
  ``vs_gpt2_shares.csv``, ``vs_table3.csv``, ``vs_ratios.csv``, ``vs_width.csv``, ``vs_gap_vs_rho.csv`` (the format of
  ``figdata/P1_gap_vs_rho.csv``), ``lm_summary.csv``, ``lm_units.csv``, ``lm_slopes.csv``, ``lm_shares.csv``,
  ``lm_decrease.csv``; and prints the tables.

Relative gaps as in Table 6: fixed query ``(ub - p*) / (p* - p_min)``, varying query ``(ub - lb) / (lb - phi_min)``;
ratios are per-unit ``(ub - ref) / (cert - ref)``. Checks (every unit; a failure stops the part): ``r_VS >= p*`` (fixed
query, upper end of the exact bracket) or ``r_VS >= lb`` (varying query; the verified incumbent on synthetic heads) to
``1e-9 max(1, |ref|)``, and ``r_VS <= r_greedy + 1e-10 range + 1e-15 max(1, |r_greedy|)``. With two keys the softmax image
of a score box is ``box ∩ simplex``, so ``r_VS = r_pp`` exactly.

References (``--refs``, default ``results_dir()``): ``exact_fixed_query/{exact,sdp}``, ``gpt2_ladder/units``,
``bound_propagation/bounds`` (for ``vs-gpt2``) and ``synthetic/{slopes,width}`` (for ``vs-synthetic``); ``--quick`` reads
the ``*_quick`` directories of those experiments.

Usage::

    python experiments/jmlr_extensions.py --part vs-gpt2 vs-synthetic --workers 12
    python experiments/jmlr_extensions.py --part lm --workers 3 --threads-per-solve 4
    python experiments/jmlr_extensions.py --summarize
    python experiments/jmlr_extensions.py --quick --workers 2 --threads-per-solve 2
    python experiments/jmlr_extensions.py --part lm --heads 1:3,7:6 --prompts 0
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

from aimp import compute  # noqa: E402
from aimp import gpt2_units as U  # noqa: E402
from aimp.io import env_info, git_commit, jsonable, peak_rss_gb, results_dir, write_json_atomic  # noqa: E402

PARTS = ("vs-gpt2", "vs-synthetic", "lm")
POS = 4
VG_RHOS = {"fq": (0.002, 0.005, 0.01, 0.02), "free": (0.001, 0.002, 0.005, 0.01)}
LM_RHOS = (0.002, 0.005, 0.01)
QUICK = {"heads": ((1, 3), (7, 6)), "prompts": (0,), "lm_rhos": (0.002, 0.005)}
LABEL = {"fq": "fixed", "free": "varying"}
V1_TOL, V2_TOL, V2_ULP, VALID_TOL = 1e-9, 1e-10, 1e-15, 1e-7
METHODS = ("IBP", "CROWN", "alpha-CROWN")
ALL_M = ("IBP", "CROWN", "alpha-CROWN", "VS", "greedy", "cert")


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def v_checks(r_vs: float, ref, r_greedy: float, rng: float) -> dict:
    ref = _num(ref)
    v1 = None if ref is None else bool(r_vs >= ref - V1_TOL * max(1.0, abs(ref)))
    return {"V1": v1, "V1_margin": None if ref is None else r_vs - ref,
            "V2": bool(r_vs <= r_greedy + V2_TOL * max(rng, 0.0) + V2_ULP * max(1.0, abs(r_greedy))),
            "V2_excess": r_vs - r_greedy}


def _cs(inst, h) -> float:
    from aimp import bounds

    si = bounds.score_intervals(inst, "cs")
    return bounds.vertex_softmax(h, si["s_lo"], si["s_hi"])["value"]


# ============================================================================================ vs-gpt2
def vg_path(out: Path, kind, l, h, p, rho) -> Path:
    return out / "vs_gpt2" / kind / f"L{l:02d}" / f"{U.unit_name(l, h, p, POS, rho)}.json"


def run_vg(task) -> dict:
    """The Vertex-Softmax bound of one unit of the Table 6 comparison (no semidefinite relaxation)."""
    kind, l, h, p, rho, out, threads = task
    path = vg_path(out, kind, l, h, p, rho)
    if path.exists():
        return {"name": path.stem, "skipped": True}
    try:
        from aimp import pipeline
        from aimp.gpt2 import nominal_value_range
        from aimp.relax_ps import solve_ps_greedy

        t0 = time.perf_counter()
        inst = U.make_instance(l, h, p, POS, rho, fixed_query=(kind == "fq"))
        si, ctrl = pipeline.controls(inst, threads)
        t1 = time.perf_counter()
        g = solve_ps_greedy(inst, ctrl, "auto")
        rng_ = nominal_value_range(inst)
        rec = {"kind": kind, "layer": l, "head": h, "prompt": p, "pos": POS, "rho_rel": rho, "n": inst.n, "range": rng_,
               "s_lo": ctrl["s_lo"], "s_hi": ctrl["s_hi"], "h_max": ctrl["h_max"], "r_VS": ctrl["r_VS"],
               "VS_k": ctrl["VS_k"], "r_VS_cs": _cs(inst, ctrl["h_max"]), "r_greedy": g.value, "w_t": ctrl["w_t"],
               "Delta": ctrl["Delta"], "VS_time_s": t1 - t0, "checks": v_checks(ctrl["r_VS"], None, g.value, rng_),
               "threads": threads, "env": env_info(), "git": git_commit(), "peak_rss_gb": peak_rss_gb()}
        write_json_atomic(path, jsonable(rec))
        return {"name": path.stem, "t": time.perf_counter() - t0, "bad": [] if rec["checks"]["V2"] else ["V2"]}
    except Exception as e:
        return {"name": path.stem, "fatal": repr(e), "tb": traceback.format_exc()[-3000:]}


def vg_tasks(heads, prompts) -> list[tuple]:
    return [(k, l, h, p, r) for k in ("fq", "free") for r in VG_RHOS[k] for (l, h) in heads for p in prompts]


# ============================================================================================ vs-synthetic
def vsyn_path(out: Path, part: str, d: Path) -> Path:
    return out / "vs_synthetic" / part / d.parent.name / f"{d.name}.json"


def _stored_instance(d: Path, meta: dict):
    """The instance of a stored synthetic record: its structure from the generator, its arrays from ``instance.npz``."""
    import dataclasses

    from aimp.instances import p2_delta, phase1_instance

    if meta.get("part") == "2A" or str(meta.get("config", "")).startswith("w"):
        t = meta.get("task") or meta
        inst = p2_delta(int(t["seed"]), float(t["w_target"]), float(t["temp"]))
    else:
        cfg = meta["config"]
        val = next(float(cfg[len(k):]) for k in ("sep", "t", "g", "k") if cfg.startswith(k))
        inst = phase1_instance(meta["family"], meta["seed"], meta["n"], meta["rho"], meta["set_type"],
                               meta["query_in_keys"], val)
    z = np.load(d / "instance.npz")
    sets = {}
    for t_, st in inst.sets.items():
        kw = {"center": np.asarray(z["zbar"][t_], float)}
        for k in ("l", "u", "E", "points"):
            if getattr(st, k) is not None:
                kw[k] = np.asarray(z[f"{k}_{t_}"], float)
        sets[t_] = dataclasses.replace(st, **kw)
    head = dataclasses.replace(inst.head, M=np.asarray(z["M"], float), N=np.asarray(z["N"], float))
    return dataclasses.replace(inst, head=head, sets=sets, c=np.asarray(z["c"], float))


def run_vsyn(task) -> dict:
    part, ds, out = task
    d = Path(ds)
    path = vsyn_path(out, part, d)
    if path.exists():
        return {"name": d.name, "skipped": True}
    try:
        from aimp import bounds
        from aimp.io import derive_fields
        from aimp.relax_ps import solve_ps_greedy

        t0 = time.perf_counter()
        meta = json.loads((d / "instance.json").read_text())
        inst = _stored_instance(d, meta)
        s0 = meta["controls"]["score_intervals"]
        si = {"method": s0["method"], "s_lo": np.asarray(s0["s_lo"], float), "s_hi": np.asarray(s0["s_hi"], float)}
        ctrl = bounds.controls(inst, intervals=si, eps_target=meta["controls"].get("eps_target", 1e-3), threads=1)
        g = solve_ps_greedy(inst, ctrl, "auto")
        recs = {}
        for m in ("ps_auto", "sdp_1", "ibp_obj"):
            f = d / f"{m}.json"
            recs[m] = derive_fields(json.loads(f.read_text())) if f.exists() else {}
        p_ref = _num((recs["ps_auto"] or recs["sdp_1"]).get("p_ref"))
        hn = np.array([inst.c @ inst.head.N @ inst.sets[j].center for j in inst.keys])
        rng_ = float(hn.max() - hn.min())
        st = _num(recs["ps_auto"].get("value"))
        rec = {"instance_id": d.name, "part": part, "family": meta["family"], "n": meta["n"], "set_type": meta["set_type"],
               "rho": meta["rho"], "w_target": meta.get("w_target"), "r_VS": ctrl["r_VS"], "VS_k": ctrl["VS_k"],
               "r_VS_cs": _cs(inst, ctrl["h_max"]), "r_greedy": g.value, "ps_auto_stored": st, "p_ref": p_ref,
               "range_nominal": rng_, "checks": {**v_checks(ctrl["r_VS"], p_ref, g.value, rng_),
                                                 "V3": None if st is None else bool(abs(g.value - st) <= 1e-7 * max(1.0, abs(st)))},
               "wall_s": time.perf_counter() - t0}
        write_json_atomic(path, jsonable(rec))
        c = rec["checks"]
        return {"name": d.name, "t": rec["wall_s"], "bad": [k for k in ("V1", "V2") if c[k] is False]}
    except Exception as e:
        return {"name": d.name, "fatal": repr(e), "tb": traceback.format_exc()[-3000:]}


def vsyn_tasks(refs: Path, quick: bool) -> list[tuple]:
    base = refs / ("synthetic_quick" if quick else "synthetic")
    out = []
    for part in ("slopes", "width"):
        for done in sorted((base / part).glob("*/*/done.json")):
            if not done.parent.parent.name.startswith("_"):
                out.append((part, str(done.parent)))
    return out


# ============================================================================================ lm
def lm_path(out: Path, kind, l, h, p, rho) -> Path:
    return out / "lm" / kind / f"L{l:02d}" / f"{U.unit_name(l, h, p, POS, rho)}.json"


def run_lm(task) -> dict:
    """One unit of the logit-margin comparison: the certificate's procedure, the references, ``r_VS`` and LiRPA."""
    kind, l, h, p, rho, out, threads = task
    path = lm_path(out, kind, l, h, p, rho)
    if path.exists():
        return {"name": path.stem, "skipped": True}
    try:
        from aimp import lirpa, pipeline

        t0 = time.perf_counter()
        if kind == "fq":
            xr = pipeline.evaluate_fq_exact(l, h, p, POS, rho, threads=threads, spec="lm")
            rec = pipeline.evaluate_fq_sdp(l, h, p, POS, rho, xr, threads=threads, spec="lm")
            rec.update({k: xr[k] for k in ("min_fq_lo", "min_fq_hi", "phi_min_cert", "max_input_in_bracket",
                                           "min_input_in_bracket")})
            ref, ref_hi = rec["p_fq"], rec["p_fq_hi"]
            inst = U.make_instance(l, h, p, POS, rho, fixed_query=True, spec="lm")
        else:
            rec = pipeline.evaluate_unit(l, h, p, POS, rho, threads=threads, spec="lm")
            ref = ref_hi = rec["lb_strong"]
            inst = U.make_instance(l, h, p, POS, rho, spec="lm")
        rec.pop("perkey", None)
        t1 = time.perf_counter()
        b = lirpa.bounds(inst, U.unit_seed(l, h, p, POS, rho) + 61, METHODS, threads=threads)
        rec.update({"kind": kind, "lirpa": b, "lirpa_s": time.perf_counter() - t1,
                    "v_checks": v_checks(rec["r_VS"], ref_hi, rec["r_greedy"], rec["range"]),
                    "wall_s": time.perf_counter() - t0})
        write_json_atomic(path, jsonable(rec))
        c = rec["v_checks"]
        bad = [k for k in ("V1", "V2") if c[k] is False]
        bad += [k for k in ("valid", "valid_value_safe", "valid_greedy", "valid_VS") if rec.get(k) is False]
        bad += [f"theorem {k}" for k, v in rec["checks"].items() if v == "violated"]
        if not b["gate_ok"]:
            bad.append("lirpa forward gate")
        return {"name": path.stem, "t": rec["wall_s"], "bad": bad, "ref": ref}
    except Exception as e:
        return {"name": path.stem, "fatal": repr(e), "tb": traceback.format_exc()[-3000:]}


def lm_tasks(heads, prompts, rhos) -> list[tuple]:
    return [(k, l, h, p, r) for k in ("fq", "free") for r in rhos for (l, h) in heads for p in prompts]


# ============================================================================================ runner
def _init(threads: int) -> None:
    compute.apply_thread_budget(threads)


def _log(out: Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_part(fn, jobs: list, workers: int, threads: int, out: Path, label: str) -> None:
    _log(out, f"=== {label}: {len(jobs)} jobs, {workers} worker(s) x {threads} thread(s)")
    if not jobs:
        return
    t0, failed = time.perf_counter(), []

    def handle(i, r) -> bool:
        if "fatal" in r:
            _log(out, f"!!! {r['name']}: {r['fatal']}\n{r['tb']}")
            failed.append(r["name"])
            return True
        if r.get("skipped"):
            return True
        if r.get("bad"):
            _log(out, f"!!! failed check on {r['name']}: {r['bad']}; stopping")
            return False
        if i % 50 == 0 or i == len(jobs):
            _log(out, f"{label}: {i}/{len(jobs)} {r['name']} t={r['t']:.2f}s elapsed={(time.perf_counter() - t0) / 60:.1f} min")
        return True

    ok = True
    if workers <= 1:
        _init(threads)
        for i, j in enumerate(jobs, 1):
            if not handle(i, fn(j)):
                ok = False
                break
    else:
        with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(threads,), maxtasksperchild=50) as pool:
            for i, r in enumerate(pool.imap_unordered(fn, jobs, chunksize=1), 1):
                if not handle(i, r):
                    ok = False
                    pool.terminate()
                    break
    _log(out, f"=== {label} {'finished' if ok else 'STOPPED'} in {(time.perf_counter() - t0) / 60:.1f} min"
         + (f"; {len(failed)} failed (rerun to retry): {failed[:10]}" if failed else ""))
    if not ok:
        raise SystemExit(1)


# ============================================================================================ summaries
def _iqr(x) -> dict:
    x = np.asarray([v for v in x if v is not None and v == v and np.isfinite(v)], float)
    if not x.size:
        return {"n": 0, "median": None, "q25": None, "q75": None}
    return {"n": int(x.size), "median": float(np.median(x)), "q25": float(np.quantile(x, .25)),
            "q75": float(np.quantile(x, .75))}


def _slope(rhos, ys):
    pts = [(math.log(r), math.log(y)) for r, y in zip(rhos, ys) if y is not None and y == y and y > 0]
    if len(pts) < 2:
        return None
    x, y = np.array(pts).T
    return float(np.polyfit(x, y, 1)[0])


def _csv(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)


def _gaps(D, ref="reference", sp="spread", cert="cert", ref_hi="reference_hi") -> None:
    for m in ALL_M:
        D[f"{m}_rel_gap"] = (D[f"{m}_ub"] - D[ref]) / D[sp]
        if m != "cert":
            D[f"{m}_ratio_to_cert"] = np.where(D[cert] - D[ref] > 0, (D[f"{m}_ub"] - D[ref]) / (D[cert] - D[ref]), np.nan)
        D[f"{m}_valid"] = [None if (u is None or u != u) else bool(u >= r - VALID_TOL * max(1.0, abs(r)))
                           for u, r in zip(D[f"{m}_ub"], D[ref_hi])]


def _summary_rows(D, time_cols: dict) -> "object":
    import pandas as pd

    S = []
    for (kind, rho), g in D.groupby(["kind", "rho_rel"]):
        row = {"kind": kind, "rho_rel": rho, "units": len(g)}
        for m in ("IBP", "CROWN", "alpha-CROWN", "greedy", "cert"):
            s = _iqr(g[f"{m}_rel_gap"])
            row.update({f"{m}_median": s["median"], f"{m}_q25": s["q25"], f"{m}_q75": s["q75"], f"{m}_n": s["n"]})
        for m in METHODS:
            row[f"{m}_ratio_to_cert_median"] = _iqr(g[f"{m}_ratio_to_cert"])["median"]
            row[f"{m}_time_median_s"] = _iqr(g[time_cols[m]])["median"] if time_cols.get(m) in g else None
            row[f"{m}_invalid"] = int((g[f"{m}_valid"] == False).sum())  # noqa: E712
        s = _iqr(g["VS_rel_gap"])
        row.update({"VS_median": s["median"], "VS_q25": s["q25"], "VS_q75": s["q75"], "VS_n": s["n"],
                    "VS_ratio_to_cert_median": _iqr(g["VS_ratio_to_cert"])["median"],
                    "VS_time_median_s": _iqr(g[time_cols["VS"]])["median"],
                    "VS_invalid": int((g["VS_valid"] == False).sum())})  # noqa: E712
        S.append(row)
    return pd.DataFrame(S)


def _slopes(D, fits: dict):
    import pandas as pd

    U_ = []
    for (kind, l, h, p), g in D.groupby(["kind", "layer", "head", "prompt"]):
        g = g.sort_values("rho_rel")
        row = {"kind": kind, "layer": l, "head": h, "prompt": p}
        for fit, rh in fits.items():
            gg = g if rh is None else g[g.rho_rel.isin(rh)]
            for m in ALL_M:
                row[f"{fit}_{m}"] = _slope(gg.rho_rel, gg[f"{m}_ub"] - gg.reference)
        U_.append(row)
    U_ = pd.DataFrame(U_)
    SL = []
    for kind, g in U_.groupby("kind") if len(U_) else []:
        for fit in fits:
            row = {"kind": kind, "fit": fit, "pairs": len(g)}
            for m in ALL_M:
                s = _iqr(g[f"{fit}_{m}"])
                row.update({f"{m}_median": s["median"], f"{m}_q25": s["q25"], f"{m}_q75": s["q75"]})
            SL.append(row)
    return pd.DataFrame(SL)


def _print_table(S, title: str) -> None:
    def g_(v, nd=3):
        return "-" if v is None or v != v else f"{v:.{nd}g}"

    print(f"\n{title}")
    hdr = (f"{'query':8s} {'rho_rel':>7s} {'units':>5s} {'IBP':>6s} {'CROWN':>7s} {'alpha':>7s} {'VS':>7s} {'first':>7s} "
           f"{'cert':>7s} {'r CROWN':>7s} {'r alpha':>7s} {'r VS':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in S.iterrows():
        print(f"{LABEL[r['kind']]:8s} {r['rho_rel']:>7g} {int(r['units']):>5d} {g_(r['IBP_median'], 2):>6s} "
              f"{g_(r['CROWN_median']):>7s} {g_(r['alpha-CROWN_median']):>7s} {g_(r['VS_median']):>7s} "
              f"{g_(r['greedy_median']):>7s} {g_(r['cert_median'], 2):>7s} {g_(r['CROWN_ratio_to_cert_median'], 2):>7s} "
              f"{g_(r['alpha-CROWN_ratio_to_cert_median'], 2):>7s} {g_(r['VS_ratio_to_cert_median'], 2):>6s}")


def summarize_vg(out: Path, refs: Path, universe: list[tuple], quick: bool) -> None:
    import pandas as pd

    import bound_propagation as BP

    BP.REF_SUFFIX = "_quick" if quick else ""
    bp_out = refs / ("bound_propagation_quick" if quick else "bound_propagation")
    rows, missing = [], 0
    for kind, l, h, p, rho in universe:
        f = vg_path(out, kind, l, h, p, rho)
        ref, _ = BP.load_reference(refs, kind, l, h, p, rho)
        if not f.exists() or ref is None:
            missing += 1
            continue
        r = json.loads(f.read_text())
        row = {"kind": kind, "layer": l, "head": h, "prompt": p, "rho_rel": rho, "range": r["range"], **ref,
               "VS_ub": r["r_VS"], "VS_cs_ub": r["r_VS_cs"], "r_greedy_recomputed": r["r_greedy"], "VS_k": r["VS_k"],
               "VS_time_s": r["VS_time_s"], "w_t": r["w_t"]}
        row.update(v_checks(r["r_VS"], ref["reference_hi"], r["r_greedy"], r["range"]))
        row["V3"] = bool(abs(r["r_greedy"] - ref["r_greedy"]) <= 1e-7 * max(1.0, abs(ref["r_greedy"])))
        bf = BP.unit_path(bp_out, "D", kind, l, h, p, rho)
        b = json.loads(bf.read_text()) if bf.exists() else {"methods": {}}
        for m in METHODS:
            x = b["methods"].get(m) or {}
            row.update({f"{m}_ub": _num(x.get("ub")), f"{m}_time_s": x.get("time_s")})
        rows.append(row)
    print(f"\nvs-gpt2: {len(rows)} units joined, {missing} without a record or a reference")
    if not rows:
        return
    D = pd.DataFrame(rows)
    n_mis = int((~D.V3).sum())
    D["greedy_ub"] = D.r_greedy_recomputed if n_mis else D.r_greedy
    D["cert_ub"] = D.cert
    _gaps(D)
    D["VS_removed"] = (D.greedy_ub - D.VS_ub) / (D.greedy_ub - D.reference)
    _csv(D, out / "summary" / "vs_gpt2_units.csv")
    S = _summary_rows(D, {"IBP": "IBP_time_s", "CROWN": "CROWN_time_s", "alpha-CROWN": "alpha-CROWN_time_s", "VS": "VS_time_s"})
    _csv(S, out / "summary" / "crown_vs_summary.csv")
    SL = _slopes(D, {"all radii": None, "0.002-0.005": (0.002, 0.005)})
    _csv(SL, out / "summary" / "vs_gpt2_slopes.csv")
    SH = pd.DataFrame([{"kind": k, "rho_rel": r, "units": len(g), "share_cert_lt_VS": float((g.cert < g.VS_ub).mean()),
                        "VS_removed_median": _iqr(g.VS_removed)["median"]} for (k, r), g in D.groupby(["kind", "rho_rel"])])
    _csv(SH, out / "summary" / "vs_gpt2_shares.csv")
    _print_table(S, "Table 6 units with the Vertex-Softmax bound (relative gaps, medians; ratios to the certificate's gap)")
    print(f"checks: V1 {int(D.V1.sum())}/{len(D)}, V2 {int(D.V2.sum())}/{len(D)}, recomputed r_greedy = stored "
          f"{int(D.V3.sum())}/{len(D)} ({'recomputed' if n_mis else 'stored'} first-order values used)")
    for _, r in SH.iterrows():
        print(f"  {LABEL[r['kind']]:8s} {r['rho_rel']:>6g}: cert < r_VS on {r['share_cert_lt_VS']:.1%}; VS removes a median "
              f"{r['VS_removed_median']:.3f} of the first-order gap")


def summarize_vsyn(out: Path, refs: Path, quick: bool) -> None:
    import pandas as pd

    import synthetic as SY

    recs = [json.loads(f.read_text()) for f in sorted((out / "vs_synthetic").glob("*/*/*.json"))]
    print(f"\nvs-synthetic: {len(recs)} instances")
    if not recs:
        return
    V = pd.DataFrame([{**{k: r[k] for k in ("instance_id", "part", "family", "n", "set_type", "rho", "w_target", "r_VS",
                                            "r_greedy", "ps_auto_stored", "p_ref", "VS_k")}, **r["checks"]} for r in recs])
    print(f"checks: V1 {int((V.V1 != False).sum())}/{len(V)}, V2 {int(V.V2.sum())}/{len(V)}, recomputed r_greedy = stored "  # noqa: E712
          f"ps_auto {int((V.V3 != False).sum())}/{len(V)}")  # noqa: E712
    base = refs / ("synthetic_quick" if quick else "synthetic")
    df = SY._frame(SY.load_records(base / "slopes"))
    if not df.empty:
        W = SY.wide(df)
        v = V[V.part == "slopes"].set_index("instance_id")
        W = W.join(v[["r_VS", "r_greedy"]], how="left")
        W["vs"] = W.r_VS - W.p_ref
        W = W.reset_index()
        W2 = W[W.n == 2]
        T = SY.slope_table(W2, ["ibp_obj", "ps_auto", "sdp_1_unitdiag", "sdp_1", "vs"])
        _csv(T, out / "summary" / "vs_table3.csv")
        print("\nTable 3 with the Vertex-Softmax row (balls, n = 2, rho <= 0.1; per-series slopes, median [IQR]):")
        for m in ("ps_auto", "sdp_1", "vs"):
            cells = []
            for fam, lab in SY.TABLE_FAMILIES:
                r = T[(T.method == m) & (T.family == fam) & (T.set_type == "ball") & (T.rule == "per_method")]
                cells.append(f"{lab} {SY._fmt(r.iloc[0].to_dict()) if len(r) else '-'}")
            print(f"  {m:8s}: " + "; ".join(cells))
        R = []
        for (fam, n), G in W.groupby(["family", "n"]):
            row = {"family": fam, "n": int(n), "instances": int(len(G))}
            for den in ("sdp_1", "ps_auto"):
                if den not in G:
                    continue
                ok = G[den].notna() & (G[den] > SY.GAP_FLOOR) & (G.res <= SY.RES_FRAC * G[den])
                s = _iqr((G.vs / G[den])[ok])
                row.update({f"VS_over_{den}_median": s["median"], f"VS_over_{den}_q25": s["q25"],
                            f"VS_over_{den}_q75": s["q75"], f"VS_over_{den}_n": s["n"]})
            R.append(row)
        R = pd.DataFrame(R)
        _csv(R, out / "summary" / "vs_ratios.csv")
        print("\nper family and n: median gap(VS) / gap(level one) and gap(VS) / gap(first order):")
        for _, r in R.iterrows():
            print(f"  {r['family']:12s} n={r['n']}: {r.get('VS_over_sdp_1_median')!s:>8.6s} / {r.get('VS_over_ps_auto_median')!s:>8.6s}")
        data = []
        for st in ("ball", "box"):
            for fam in SY.FAMILY_ORDER:
                G = W2[(W2.family == fam) & (W2.set_type == st)]
                H = G[(G.res <= SY.RES_FRAC * G.vs) & (G.vs > SY.GAP_FLOOR)]
                if H.empty:
                    continue
                q = H.groupby("rho")["vs"].quantile([0.25, 0.5, 0.75]).unstack()
                cnt = H.groupby("rho")["vs"].count()
                for rho, row in q.iterrows():
                    data.append({"family": fam, "set_type": st, "method": "vs", "rho": rho, "q25": row[0.25],
                                 "median": row[0.5], "q75": row[0.75], "n_resolved": int(cnt[rho])})
        _csv(pd.DataFrame(data, columns=["family", "set_type", "method", "rho", "q25", "median", "q75", "n_resolved"]),
             out / "summary" / "vs_gap_vs_rho.csv")
    wf = SY._frame(SY.load_records(base / "width"))
    if not wf.empty:
        A = wf[wf.method.isin(["ps_auto", "sdp_1"])].copy()
        A["rel_gap"] = A.gap / (A.p_ref - A.phi_min_ms)
        v = V[V.part == "width"].set_index("instance_id")
        rows = []
        for wt, G in A.groupby("w_target"):
            ids = G.instance_id.unique()
            g1 = G[G.method == "ps_auto"].set_index("instance_id")
            g2 = G[G.method == "sdp_1"].set_index("instance_id")
            vv = v.reindex(ids)
            rel_vs = ((vv.r_VS - vv.p_ref) / (g1.p_ref - g1.phi_min_ms).reindex(ids)).median()
            rows.append({"w_target": wt, "n": len(ids), "ps_auto_median": float(g1.rel_gap.median()),
                         "sdp_1_median": float(g2.rel_gap.median()), "vs_median": float(rel_vs),
                         "ratio_ps_auto_over_sdp_1": float(g1.rel_gap.median() / g2.rel_gap.median()),
                         "ratio_vs_over_sdp_1": float(rel_vs / g2.rel_gap.median())})
        WT = pd.DataFrame(rows)
        _csv(WT, out / "summary" / "vs_width.csv")
        print("\nwidth experiment: first order / level one and VS / level one (ratios of the medians):")
        for _, r in WT.iterrows():
            print(f"  w_t = {r['w_target']:g}: {r['ratio_ps_auto_over_sdp_1']:.3g} / {r['ratio_vs_over_sdp_1']:.3g}")


def summarize_lm(out: Path, universe: list[tuple]) -> None:
    import pandas as pd

    gate = json.loads((U.cache_dir() / "lm_gate.json").read_text()) if (U.cache_dir() / "lm_gate.json").exists() else None
    margin = {r["prompt"]: r["margin"] for r in (gate or {}).get("prompts", [])}
    rows = []
    for kind, l, h, p, rho in universe:
        f = lm_path(out, kind, l, h, p, rho)
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        if kind == "fq":
            ref, ref_hi, sp = r["p_fq"], r["p_fq_hi"], r["spread_fq"]
        else:
            ref, ref_hi, sp = r["lb_strong"], r["lb_strong"], r["lb_strong"] - r["phi_min_strong"]
        row = {"kind": kind, "layer": l, "head": h, "prompt": p, "rho_rel": rho, "reference": ref, "reference_hi": ref_hi,
               "spread": sp, "cert": r["cert"], "cert_ub": r["cert"], "greedy_ub": r["r_greedy"], "VS_ub": r["r_VS"],
               "phi_nominal": r["phi_nominal"], "m": margin.get(p), "w_t": r["w_t"], "Delta": r["Delta"],
               "rad_c": r.get("rad_c"), "S": r.get("S"), "time_cert_s": r["times"].get("total_s"),
               "VS_time_s": r["times"].get("controls_s"), **{f"V_{k}": v for k, v in r["v_checks"].items()},
               "violated": sum(v == "violated" for v in r["checks"].values())}
        for m in METHODS:
            x = r["lirpa"]["methods"].get(m) or {}
            row.update({f"{m}_ub": _num(x.get("ub")), f"{m}_time_s": x.get("time_s")})
        rows.append(row)
    print(f"\nlm: {len(rows)} of {len(universe)} units")
    if not rows:
        return
    D = pd.DataFrame(rows)
    _gaps(D)
    D["decrease_cert"] = D.cert - D.phi_nominal
    D["decrease_cert_over_m"] = D.decrease_cert / D.m
    _csv(D, out / "summary" / "lm_units.csv")
    S = _summary_rows(D, {"IBP": "IBP_time_s", "CROWN": "CROWN_time_s", "alpha-CROWN": "alpha-CROWN_time_s", "VS": "VS_time_s"})
    _csv(S, out / "summary" / "lm_summary.csv")
    _csv(_slopes(D, {"all radii": None, "0.002-0.005": (0.002, 0.005)}), out / "summary" / "lm_slopes.csv")
    SH = []
    for (k, rho), g in D.groupby(["kind", "rho_rel"]):
        best = g[[f"{m}_ub" for m in METHODS]].astype(float).min(axis=1)
        SH.append({"kind": k, "rho_rel": rho, "units": len(g), "share_cert_tighter_than_best_lirpa": float((g.cert < best).mean()),
                   "share_cert_lt_VS": float((g.cert < g.VS_ub).mean()),
                   **{f"share_within_1pct_{m}": float((g[f"{m}_rel_gap"] <= 0.01).mean()) for m in ALL_M}})
    _csv(pd.DataFrame(SH), out / "summary" / "lm_shares.csv")
    DEC = pd.DataFrame([{"kind": k, "rho_rel": rho, "units": len(g), "decrease_cert_median": float(g.decrease_cert.median()),
                         "decrease_cert_over_m_median": float(g.decrease_cert_over_m.median()),
                         "share_decrease_cert_below_m": float((g.decrease_cert < g.m).mean())}
                        for (k, rho), g in D.groupby(["kind", "rho_rel"])])
    _csv(DEC, out / "summary" / "lm_decrease.csv")
    _print_table(S, "Logit-margin specification (relative gaps, medians; ratios to the certificate's gap)")
    print(f"checks: V1 {int((D.V_V1 != False).sum())}/{len(D)}, V2 {int(D.V_V2.sum())}/{len(D)}, theorem checks violated "  # noqa: E712
          f"{int(D.violated.sum())}; invalid bounds {[int((D[f'{m}_valid'] == False).sum()) for m in ALL_M]}")  # noqa: E712
    print("The certified decrease concerns one head's direct path to the logits with the final normalization scale "
          "frozen, not the model's prediction.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", nargs="+", choices=PARTS, default=list(PARTS))
    ap.add_argument("--quick", action="store_true", help=f"heads {list(QUICK['heads'])}, prompt 0 (a few minutes)")
    ap.add_argument("--heads", default=None, help="comma-separated layer:head list (overrides --quick's heads)")
    ap.add_argument("--prompts", default=None, help="comma-separated prompt indices (overrides --quick's prompt)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads-per-solve", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help="default results_dir()/jmlr (with --quick .../jmlr_quick)")
    ap.add_argument("--refs", type=Path, default=None, help="root of the reference records (default results_dir())")
    ap.add_argument("--summarize", action="store_true", help="only join the stored records")
    a = ap.parse_args(argv)
    out = a.out or results_dir() / ("jmlr_quick" if a.quick else "jmlr")
    refs = a.refs or results_dir()
    heads = list(QUICK["heads"]) if a.quick else [(l, h) for l in U.LAYERS for h in U.HEADS]
    if a.heads:
        heads = [tuple(int(v) for v in x.split(":")) for x in a.heads.split(",")]
    prompts = QUICK["prompts"] if a.quick else range(U.N_PROMPTS)
    if a.prompts:
        prompts = [int(x) for x in a.prompts.split(",")]
    rhos = QUICK["lm_rhos"] if a.quick else LM_RHOS
    vg_u = vg_tasks(heads, prompts)
    lm_u = lm_tasks(heads, prompts, rhos)
    if not a.summarize:
        if a.workers * a.threads_per_solve > compute.thread_budget():
            raise SystemExit(f"workers x threads = {a.workers * a.threads_per_solve} exceeds AIMP_MAX_THREADS = "
                             f"{compute.thread_budget()}")
        if "vs-gpt2" in a.part or "lm" in a.part:
            U.prepare()
        if "vs-gpt2" in a.part:
            jobs = [(*t, out, a.threads_per_solve) for t in vg_u if not vg_path(out, *t).exists()]
            run_part(run_vg, jobs, a.workers, a.threads_per_solve, out, "vs-gpt2")
        if "vs-synthetic" in a.part:
            jobs = [(pt, d, out) for pt, d in vsyn_tasks(refs, a.quick) if not vsyn_path(out, pt, Path(d)).exists()]
            run_part(run_vsyn, jobs, a.workers, 1, out, "vs-synthetic")
        if "lm" in a.part:
            gate = U.prepare_lm()
            _log(out, f"lm: specification checks L0 {gate['L0_ok']} L1 {gate['L1_ok']} L2 {gate['L2_ok']} "
                      f"(L2 worst {gate['L2_max_rel']:.1e})")
            jobs = [(*t, out, a.threads_per_solve) for t in lm_u if not lm_path(out, *t).exists()]
            run_part(run_lm, jobs, a.workers, a.threads_per_solve, out, "lm")
    if "vs-gpt2" in a.part:
        summarize_vg(out, refs, vg_u, a.quick)
    if "vs-synthetic" in a.part:
        summarize_vsyn(out, refs, a.quick)
    if "lm" in a.part:
        summarize_lm(out, lm_u)


if __name__ == "__main__":
    main()
