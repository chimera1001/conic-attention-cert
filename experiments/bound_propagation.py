"""Linear bound propagation against the certificate on GPT-2 small (Section 7: Table 6 and Figure 4).

Part ``D`` (default): IBP, CROWN and alpha-CROWN upper bounds (auto_LiRPA with its default options, float64, the
softmax bounded as one operation; :mod:`aimp.lirpa`) on the 5,760 units of the comparison. These are all 144 heads x
5 prompts at query position 4: 2,880 units with the query fixed, rho_rel in {0.002, 0.005, 0.01, 0.02}, measured
against the exact worst case, and 2,880 units with a varying query, rho_rel in {0.001, 0.002, 0.005, 0.01}, measured
against the best feasible value.

Part ``alpha``: alpha-CROWN at 20 / 100 / 500 iterations and learning rates 0.5 / 0.1 on the 1,152 units of prompt 0
(144 heads x 4 fixed-query + 4 varying-query radii).

Every unit writes one JSON atomically under ``--out``; a rerun skips finished units. ``--summarize`` (also run at the
end of every part) joins the bounds with the reference and certificate records of the exact fixed-query and GPT-2
ladder experiments under ``--refs`` and writes

* part D: ``crown_summary.csv`` (per query type and radius: medians and quartiles of the relative gaps, median
  ratios to the certificate's gap, median times; the format of ``figdata/P4b_crown_summary.csv`` read by
  ``figscripts/make_figs.py``), ``crown_units.csv`` (one row per unit), ``crown_slopes.csv`` and
  ``crown_missing.csv``, and prints the table with the slopes, the share of units where the certificate is tighter
  and the timings;
* part alpha: ``alpha_settings.csv`` (one row per unit and setting) and ``alpha_settings_summary.csv``.

Relative gaps. Fixed query: ``(ub - p*) / (p* - p_min)``, with ``p*`` and ``p_min`` the exact maximum and minimum of
the fixed-query unit. Varying query: ``(ub - lb) / (lb - phi_min)``, with ``lb`` and ``phi_min`` the best feasible
values of the maximization and the minimization. The certificate and the first-order certificate are measured the
same way, and the ratio columns are per-unit ratios ``(ub - ref) / (cert - ref)``.

Reference records (``--refs``, default ``results_dir()``; ``<unit>`` = :func:`aimp.gpt2_units.unit_name`)::

    exact_fixed_query/exact/L<ll>/<unit>_fq.json   p_fq, p_fq_hi, min_fq, spread_fq, exact_status
    exact_fixed_query/sdp/L<ll>/<unit>_fq.json     cert, r_greedy
    gpt2_ladder/units/L<ll>/<unit>.json            cert, r_greedy, lb_strong, phi_min_strong

Units without bounds or without references are listed in ``crown_missing.csv`` and counted in the printout.

Usage::

    python experiments/bound_propagation.py [--part D|alpha] [--quick] [--workers 3] [--threads-per-solve 1]
    python experiments/bound_propagation.py --summarize [--part D|alpha] [--refs DIR]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from aimp import compute  # noqa: E402
from aimp import gpt2_units as U  # noqa: E402
from aimp.io import git_commit, peak_rss_gb, results_dir, write_json_atomic  # noqa: E402

POS = 4
FIXED_RHOS = (0.002, 0.005, 0.01, 0.02)
VARYING_RHOS = (0.001, 0.002, 0.005, 0.01)
#: query types: ``fq`` = fixed query, ``free`` = varying query (the codes of the figure data)
KINDS = ("fq", "free")
RHOS = {"fq": FIXED_RHOS, "free": VARYING_RHOS}
LABEL = {"fq": "fixed", "free": "varying"}
METHODS = ("IBP", "CROWN", "alpha-CROWN")
QUICK_HEADS = ((1, 3), (7, 6))  # the quick heads of gpt2_ladder.py and exact_fixed_query.py
REF_SUFFIX = ""  # "_quick": read the references of the --quick runs of the other experiments
GATE_SEED_OFFSET = 61  # forward-gate seed = unit_seed + 61
VALID_TOL = 1e-7  # a bound is valid if ub >= reference_hi - 1e-7 max(1, |reference_hi|)
BASE_SETTING = "it20_lr0.5"  # auto_LiRPA's defaults (the setting of part D)


# ============================================================================================ units and paths
def instance(kind: str, l: int, h: int, p: int, rho: float):
    return U.make_instance(l, h, p, POS, rho, fixed_query=(kind == "fq"))


def tasks(heads, prompts) -> list[tuple]:
    """(kind, layer, head, prompt, rho_rel) for every query type and radius."""
    return [(kind, l, h, p, rho) for kind in KINDS for rho in RHOS[kind] for (l, h) in heads for p in prompts]


def unit_path(out: Path, part: str, kind: str, l: int, h: int, p: int, rho: float) -> Path:
    sub = "bounds" if part == "D" else "alpha_settings"
    return out / sub / kind / f"L{l:02d}" / f"{U.unit_name(l, h, p, POS, rho)}.json"


@functools.lru_cache(maxsize=1)
def _provenance() -> dict:
    """Machine and versions (once per process); ``aimp.io.env_info`` also reports the solver versions when the
    solvers are installed."""
    from aimp import lirpa

    try:
        from aimp.io import env_info

        env = env_info()
    except Exception:  # the solvers are not needed here
        env = {"node": platform.node(), "os": platform.platform(), "python": platform.python_version(),
               "numpy": np.__version__, "thread_budget": compute.thread_budget()}
    return {"env": {**env, **lirpa.versions()}, "git": git_commit()}


def _base(kind, l, h, p, rho, inst) -> dict:
    from aimp import gpt2

    return {"kind": kind, "query": LABEL[kind], "layer": l, "head": h, "prompt": p, "pos": POS, "rho_rel": rho,
            "n": inst.n, "range": gpt2.nominal_value_range(inst), "seed": U.unit_seed(l, h, p, POS, rho) + GATE_SEED_OFFSET}


# ============================================================================================ workers
def run_bounds(task: tuple) -> dict:
    """Part D on one unit: forward gate, then IBP / CROWN / alpha-CROWN."""
    kind, l, h, p, rho, out, threads = task
    path = unit_path(out, "D", kind, l, h, p, rho)
    name = f"{kind}:{path.stem}"
    if path.exists():
        return {"name": name, "skipped": True}
    try:
        from aimp import lirpa

        t0 = time.perf_counter()
        inst = instance(kind, l, h, p, rho)
        base = _base(kind, l, h, p, rho, inst)
        b = lirpa.bounds(inst, base["seed"], METHODS, threads=threads)
        rec = {**base, **b, "threads": threads, "wall_s": time.perf_counter() - t0, **_provenance(),
               "peak_rss_gb": peak_rss_gb()}
        write_json_atomic(path, rec)
        return {"name": name, "t": rec["wall_s"], "gate_ok": b["gate_ok"], "gate": b["gate_max_abs"],
                "below_samples": [m for m, r in b["methods"].items() if r["above_samples"] is False],
                "errors": [m for m, r in b["methods"].items() if r["status"] != "ok"]}
    except Exception as e:
        return {"name": name, "fatal": repr(e), "tb": traceback.format_exc()[-3000:]}


def run_alpha(task: tuple) -> dict:
    """Part alpha on one unit: forward gate (for the sampled maximum), then every alpha-CROWN setting."""
    kind, l, h, p, rho, out, threads = task
    path = unit_path(out, "alpha", kind, l, h, p, rho)
    name = f"{kind}:{path.stem}"
    if path.exists():
        return {"name": name, "skipped": True}
    try:
        from aimp import lirpa

        t0 = time.perf_counter()
        inst = instance(kind, l, h, p, rho)
        base = _base(kind, l, h, p, rho, inst)
        pert = lirpa.perturbed_tokens(inst)
        gate = lirpa.forward_gate(inst, lirpa.head_module(inst, pert, threads), pert, np.random.default_rng(base["seed"]))
        settings = lirpa.alpha_settings(inst, lirpa.ALPHA_ITERS, lirpa.ALPHA_LRS, threads=threads)
        tol = lirpa.SAMPLE_TOL * max(1.0, abs(gate["sampled_max"]))
        for s in settings.values():
            s["above_samples"] = bool(s["ub"] >= gate["sampled_max"] - tol)
        rec = {**base, **gate, "perturbed": pert, "n_inputs": len(pert), "settings": settings, "threads": threads,
               "wall_s": time.perf_counter() - t0, **_provenance(), "peak_rss_gb": peak_rss_gb()}
        write_json_atomic(path, rec)
        return {"name": name, "t": rec["wall_s"], "gate_ok": gate["gate_ok"], "gate": gate["gate_max_abs"],
                "below_samples": [k for k, s in settings.items() if not s["above_samples"]], "errors": []}
    except Exception as e:
        return {"name": name, "fatal": repr(e), "tb": traceback.format_exc()[-3000:]}


def _init(threads: int) -> None:
    """Pool worker: the per-worker thread budget (BLAS / OpenMP / torch)."""
    compute.apply_thread_budget(threads)


class Log:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def run_part(part: str, todo: list[tuple], out: Path, workers: int, threads: int, log: Log) -> None:
    fn = run_bounds if part == "D" else run_alpha
    jobs = [(*t, out, threads) for t in todo]
    log(f"=== part {part}: {len(jobs)} units to run, {workers} worker(s) x {threads} thread(s), out {out}")
    if not jobs:
        return
    t0 = time.perf_counter()
    ok, failed = True, []

    def handle(i: int, r: dict) -> bool:
        if "fatal" in r:
            log(f"!!! {r['name']}: {r['fatal']}\n{r['tb']}")
            failed.append(r["name"])
            return True
        if r.get("skipped"):
            return True
        if not r["gate_ok"]:
            log(f"!!! forward gate failed on {r['name']}: max deviation {r['gate']:.3e}; stopping")
            return False
        if r["below_samples"]:
            log(f"!!! bound below a sampled value of the head on {r['name']}: {r['below_samples']}")
        if r["errors"]:
            log(f"!!! library error on {r['name']}: {r['errors']} (recorded in the unit's JSON)")
        if part == "alpha" or i % 50 == 0 or i == len(jobs):
            log(f"part {part}: {i}/{len(jobs)} {r['name']} t={r['t']:.2f}s elapsed={(time.perf_counter() - t0) / 60:.1f} min")
        return True

    if workers <= 1:
        _init(threads)
        for i, job in enumerate(jobs, 1):
            if not handle(i, fn(job)):
                ok = False
                break
    else:
        with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(threads,), maxtasksperchild=200) as pool:
            for i, r in enumerate(pool.imap_unordered(fn, jobs, chunksize=1), 1):
                if not handle(i, r):
                    ok = False
                    pool.terminate()
                    break
    log(f"=== part {part} {'finished' if ok else 'STOPPED'} in {(time.perf_counter() - t0) / 60:.1f} min"
        + (f"; {len(failed)} unit(s) failed (no JSON written, rerun to retry): {failed[:10]}" if failed else ""))
    if not ok:
        raise SystemExit(1)


# ============================================================================================ references
def _num(x):
    """A finite float or None (records write non-finite values as strings)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _read(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_reference(refs: Path, kind: str, l: int, h: int, p: int, rho: float) -> tuple[dict | None, str]:
    """The unit's reference value, its upper end (for validity), the spread, the certificate and the first-order
    certificate; ``(None, reason)`` when a record or a value is missing."""
    name, ll = U.unit_name(l, h, p, POS, rho), f"L{l:02d}"
    if kind == "fq":
        fe = refs / f"exact_fixed_query{REF_SUFFIX}" / "exact" / ll / f"{name}_fq.json"
        fs = refs / f"exact_fixed_query{REF_SUFFIX}" / "sdp" / ll / f"{name}_fq.json"
        miss = [str(f) for f in (fe, fs) if not f.exists()]
        if miss:
            return None, "missing " + " ".join(miss)
        x, q = _read(fe), _read(fs)
        ref = _num(x.get("p_fq"))
        out = {"reference": ref, "reference_hi": _num(x.get("p_fq_hi", ref)), "spread": _num(x.get("spread_fq")),
               "cert": _num(q.get("cert")), "r_greedy": _num(q.get("r_greedy", x.get("r_greedy"))),
               "exact_status": x.get("exact_status")}
    else:
        f = refs / f"gpt2_ladder{REF_SUFFIX}" / "units" / ll / f"{name}.json"
        if not f.exists():
            return None, f"missing {f}"
        r = _read(f)
        lb, lo = _num(r.get("lb_strong")), _num(r.get("phi_min_strong"))
        out = {"reference": lb, "reference_hi": lb, "spread": None if (lb is None or lo is None) else lb - lo,
               "cert": _num(r.get("cert")), "r_greedy": _num(r.get("r_greedy")), "exact_status": None}
    bad = [k for k in ("reference", "reference_hi", "spread", "cert", "r_greedy") if out[k] is None]
    return (None, f"non-finite or absent {bad}") if bad else (out, "")


# ============================================================================================ summaries
def _iqr(x) -> dict:
    x = np.asarray([v for v in x if v is not None and v == v and np.isfinite(v)], float)
    if not x.size:
        return {"n": 0, "median": None, "q25": None, "q75": None}
    return {"n": int(x.size), "median": float(np.median(x)), "q25": float(np.quantile(x, .25)),
            "q75": float(np.quantile(x, .75))}


def _slope(rhos, ys):
    """Least-squares slope of log y on log rho over the points with y > 0."""
    pts = [(math.log(r), math.log(y)) for r, y in zip(rhos, ys) if y is not None and y == y and y > 0]
    if len(pts) < 2:
        return None
    x, y = np.array(pts).T
    return float(np.polyfit(x, y, 1)[0])


def _csv(df, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)


def _rel(v, ref, den):
    return None if (v is None or not den) else (v - ref) / den


def _g(v, nd=3) -> str:
    return "-" if v is None or v != v else f"{v:.{nd}g}"


def _f(v, nd=1) -> str:
    return "-" if v is None or v != v else f"{v:.{nd}f}"


def summarize_D(universe: list[tuple], out: Path, refs: Path) -> None:
    import pandas as pd

    rows, missing = [], []
    for kind, l, h, p, rho in universe:
        f = unit_path(out, "D", kind, l, h, p, rho)
        key = {"kind": kind, "layer": l, "head": h, "prompt": p, "rho_rel": rho}
        if not f.exists():
            missing.append({**key, "missing": "bounds"})
            continue
        ref, why = load_reference(refs, kind, l, h, p, rho)
        if ref is None:
            missing.append({**key, "missing": f"reference: {why}"})
            continue
        r = _read(f)
        ref0, sp, cert = ref["reference"], ref["spread"], ref["cert"]
        row = {**key, "pos": POS, "n": r["n"], "range": r["range"], "reference": ref0, "spread": sp, "cert": cert,
               "r_greedy": ref["r_greedy"], "cert_rel_gap": _rel(cert, ref0, sp),
               "greedy_rel_gap": _rel(ref["r_greedy"], ref0, sp), "exact_status": ref["exact_status"],
               "gate_max_abs": r["gate_max_abs"], "gate_ok": r["gate_ok"], "n_inputs": r["n_inputs"], "build_s": r["build_s"]}
        tol = VALID_TOL * max(1.0, abs(ref["reference_hi"]))
        for m in METHODS:
            x = r["methods"].get(m) or {}
            ub = _num(x.get("ub"))
            row.update({f"{m}_ub": ub, f"{m}_rel_gap": _rel(ub, ref0, sp),
                        f"{m}_valid": None if ub is None else bool(ub >= ref["reference_hi"] - tol),
                        f"{m}_above_samples": x.get("above_samples"),
                        f"{m}_ratio_to_cert": None if (ub is None or cert - ref0 <= 0) else (ub - ref0) / (cert - ref0),
                        f"{m}_time_s": x.get("time_s"), f"{m}_status": x.get("status")})
        rows.append(row)
    out.mkdir(parents=True, exist_ok=True)
    _csv(pd.DataFrame(missing, columns=["kind", "layer", "head", "prompt", "rho_rel", "missing"]), out / "crown_missing.csv")
    n_nob = sum(m["missing"] == "bounds" for m in missing)
    print(f"\n{len(universe)} units: {len(rows)} with bounds and references; {n_nob} without bounds, "
          f"{len(missing) - n_nob} without references (listed in {out / 'crown_missing.csv'})")
    if not rows:
        return
    D = pd.DataFrame(rows)
    _csv(D, out / "crown_units.csv")

    # per query type and radius: the figure data
    S = []
    for (kind, rho), g in D.groupby(["kind", "rho_rel"]):
        row = {"kind": kind, "rho_rel": rho, "units": len(g)}
        for m in ("IBP", "CROWN", "alpha-CROWN", "greedy", "cert"):
            s = _iqr(g[f"{m}_rel_gap"])
            row.update({f"{m}_median": s["median"], f"{m}_q25": s["q25"], f"{m}_q75": s["q75"], f"{m}_n": s["n"]})
        for m in METHODS:
            row[f"{m}_ratio_to_cert_median"] = _iqr(g[f"{m}_ratio_to_cert"])["median"]
            row[f"{m}_time_median_s"] = _iqr(g[f"{m}_time_s"])["median"]
            row[f"{m}_invalid"] = int((g[f"{m}_valid"] == False).sum())  # noqa: E712
        S.append(row)
    S = pd.DataFrame(S)
    _csv(S, out / "crown_summary.csv")

    # per-unit slopes in rho_rel over the four radii of the query type
    U_ = []
    for (kind, l, h, p), g in D.groupby(["kind", "layer", "head", "prompt"]):
        if len(g) < len(RHOS[kind]):
            continue
        g = g.sort_values("rho_rel")
        row = {"kind": kind, "layer": l, "head": h, "prompt": p}
        for m in METHODS:
            row[f"slope_{m}"] = _slope(g.rho_rel, g[f"{m}_ub"] - g.reference)
        row["slope_cert"] = _slope(g.rho_rel, g.cert - g.reference)
        row["slope_greedy"] = _slope(g.rho_rel, g.r_greedy - g.reference)
        U_.append(row)
    SL = []
    for kind, g in pd.DataFrame(U_).groupby("kind") if U_ else []:
        row = {"kind": kind, "heads_prompts": len(g)}
        for m in ("IBP", "CROWN", "alpha-CROWN", "cert", "greedy"):
            s = _iqr(g[f"slope_{m}"])
            row.update({f"slope_{m}_median": s["median"], f"slope_{m}_q25": s["q25"], f"slope_{m}_q75": s["q75"]})
        SL.append(row)
    SL = pd.DataFrame(SL)
    _csv(SL, out / "crown_slopes.csv")

    # the table
    print("\nRelative gaps of linear bound propagation and of the certificates (Section 7, Table 6); medians.")
    print("fixed query: against the exact worst case; varying query: against the best feasible value; "
          "ratio = median per-unit (ub - ref) / (cert - ref)")
    hdr = f"{'query':8s} {'rho_rel':>7s} {'units':>5s} {'IBP':>7s} {'CROWN':>7s} {'alpha':>7s} {'first':>7s} {'cert':>7s}" \
          f" {'ratio CROWN':>11s} {'ratio alpha':>11s}"
    print(hdr)
    print("-" * len(hdr))
    for r in S.to_dict("records"):
        print(f"{LABEL[r['kind']]:8s} {r['rho_rel']:>7g} {r['units']:>5d} {_f(r['IBP_median']):>7s} "
              f"{_g(r['CROWN_median']):>7s} {_g(r['alpha-CROWN_median']):>7s} {_g(r['greedy_median']):>7s} "
              f"{_g(r['cert_median'], 2):>7s} {_f(r['CROWN_ratio_to_cert_median']):>11s} "
              f"{_f(r['alpha-CROWN_ratio_to_cert_median']):>11s}")
    print("\nPer-unit slope of log(bound - reference) in log rho_rel over the four radii; median [IQR] over head-prompt pairs:")
    names = {"IBP": "IBP", "CROWN": "CROWN", "alpha-CROWN": "alpha-CROWN", "greedy": "first order", "cert": "certificate"}
    for r in SL.to_dict("records"):
        parts = [f"{names[m]} {r[f'slope_{m}_median']:.2f} [{r[f'slope_{m}_q25']:.2f}, {r[f'slope_{m}_q75']:.2f}]"
                 for m in ("IBP", "CROWN", "alpha-CROWN", "greedy", "cert") if r[f"slope_{m}_median"] is not None]
        print(f"  {LABEL[r['kind']]:8s} ({r['heads_prompts']} pairs): " + "; ".join(parts))

    # certificate against the best bound propagation bound, unit by unit
    ubs = D[[f"{m}_ub" for m in METHODS]].astype(float)
    best = ubs.min(axis=1, skipna=True)
    have = best.notna()
    tighter = (D.cert < best) & have
    f2 = ((best - D.reference) >= 2 * (D.cert - D.reference)) & have
    print(f"\nCertificate tighter than the best of IBP / CROWN / alpha-CROWN on {int(tighter.sum())} / {int(have.sum())} "
          f"units; by at least a factor two (gap) on {f2[have].mean():.1%} "
          f"({', '.join(f'{LABEL[k]} {f2[have & (D.kind == k)].mean():.1%}' for k in KINDS if (have & (D.kind == k)).any())})."
          f" Smallest margin best - cert: {float((best - D.cert)[have].min()):.3g}.")
    print("Median time per unit: " + "; ".join(
        f"{m} " + " / ".join(f"{_g(D[D.kind == k][f'{m}_time_s'].median())} s ({LABEL[k]})" for k in KINDS
                             if (D.kind == k).any()) for m in METHODS))
    inval = {m: int((D[f"{m}_valid"] == False).sum()) for m in METHODS}  # noqa: E712
    below = {m: int((D[f"{m}_above_samples"] == False).sum()) for m in METHODS}  # noqa: E712
    errs = {m: int((D[f"{m}_status"] != "ok").sum()) for m in METHODS}
    print(f"Validity: bounds below the reference {inval}; below a sampled value {below}; library errors {errs}; "
          f"forward gate failures {int((~D.gate_ok.astype(bool)).sum())} (max deviation {D.gate_max_abs.max():.2e}).")
    ne = int((D.kind.eq("fq") & D.exact_status.ne("exact")).sum())
    if ne:
        print(f"Fixed-query units whose reference is not certified exact: {ne}.")
    print(f"\nWritten: {out / 'crown_summary.csv'}, {out / 'crown_units.csv'}, {out / 'crown_slopes.csv'}")


def summarize_alpha(universe: list[tuple], out: Path, refs: Path) -> None:
    import pandas as pd

    rows, missing = [], []
    for kind, l, h, p, rho in universe:
        f = unit_path(out, "alpha", kind, l, h, p, rho)
        key = {"kind": kind, "layer": l, "head": h, "prompt": p, "rho_rel": rho}
        if not f.exists():
            missing.append({**key, "missing": "bounds"})
            continue
        ref, why = load_reference(refs, kind, l, h, p, rho)
        if ref is None:
            missing.append({**key, "missing": f"reference: {why}"})
            continue
        r = _read(f)
        ref0, sp, cert = ref["reference"], ref["spread"], ref["cert"]
        tol = VALID_TOL * max(1.0, abs(ref["reference_hi"]))
        for k, s in r["settings"].items():
            rows.append({**key, "reference": ref0, "spread": sp, "cert": cert, "setting": k, "iterations": s["iterations"],
                         "lr_alpha": s["lr_alpha"], "ub": s["ub"], "valid": bool(s["ub"] >= ref["reference_hi"] - tol),
                         "above_samples": s.get("above_samples"), "time_s": s["time_s"],
                         "iterations_run": s["iterations_run"], "iter_best": s["iter_best"],
                         "rel_gap": _rel(s["ub"], ref0, sp),
                         "ratio_to_cert_gap": (s["ub"] - ref0) / (cert - ref0) if cert - ref0 > 0 else None})
    out.mkdir(parents=True, exist_ok=True)
    _csv(pd.DataFrame(missing, columns=["kind", "layer", "head", "prompt", "rho_rel", "missing"]), out / "alpha_missing.csv")
    n_nob = sum(m["missing"] == "bounds" for m in missing)
    print(f"\n{len(universe)} units: {len(universe) - len(missing)} with settings and references; {n_nob} without settings, "
          f"{len(missing) - n_nob} without references (listed in {out / 'alpha_missing.csv'})")
    if not rows:
        return
    A = pd.DataFrame(rows)
    _csv(A, out / "alpha_settings.csv")
    S = []
    for (kind, st), g in list(A.groupby(["kind", "setting"])) + [(("all", st), g) for st, g in A.groupby("setting")]:
        row = {"kind": kind, "setting": st, "iterations": int(g.iterations.iloc[0]), "lr_alpha": float(g.lr_alpha.iloc[0]),
               "units": len(g), "invalid": int((g.valid == False).sum()),  # noqa: E712
               "below_samples": int((g.above_samples == False).sum())}  # noqa: E712
        for k in ("ratio_to_cert_gap", "rel_gap", "time_s", "iterations_run"):
            s = _iqr(g[k])
            row.update({f"{k}_median": s["median"], f"{k}_q25": s["q25"], f"{k}_q75": s["q75"]})
        row["share_beats_cert"] = float((g.ub < g.cert).mean())
        S.append(row)
    S = pd.DataFrame(S).sort_values(["kind", "lr_alpha", "iterations"], ascending=[True, False, True])
    base = S[S.setting == BASE_SETTING].set_index("kind")["ratio_to_cert_gap_median"]
    S["ratio_change_vs_default"] = [(r.ratio_to_cert_gap_median - base.get(r.kind, np.nan)) / base.get(r.kind, np.nan)
                                    for r in S.itertuples()]
    _csv(S, out / "alpha_settings_summary.csv")
    print(f"\nalpha-CROWN settings: median over units of (ub - ref) / (cert - ref); change against the library default "
          f"({BASE_SETTING})")
    print(f"{'units':8s} {'setting':12s} {'n':>5s} {'median ratio':>12s} {'change':>9s} {'median time':>11s} "
          f"{'iterations run':>14s} {'invalid':>7s} {'beats cert':>10s}")
    for r in S.to_dict("records"):
        print(f"{LABEL.get(r['kind'], r['kind']):8s} {r['setting']:12s} {r['units']:>5d} {r['ratio_to_cert_gap_median']:>12.5f} "
              f"{r['ratio_change_vs_default']:>9.4%} {r['time_s_median']:>10.3f}s {r['iterations_run_median']:>14g} "
              f"{r['invalid']:>7d} {r['share_beats_cert']:>10.1%}")
    a = S[S.kind == "all"].ratio_to_cert_gap_median
    print(f"\nAll {int(S[S.kind == 'all'].units.max())} units: the median ratio ranges over [{a.min():.5f}, {a.max():.5f}] "
          f"across the {len(a)} settings, a spread of {(a.max() - a.min()) / a.min():.4%}; largest change against the "
          f"default {S[S.kind == 'all'].ratio_change_vs_default.abs().max():.4%}.")
    print(f"Written: {out / 'alpha_settings_summary.csv'}, {out / 'alpha_settings.csv'}")


# ============================================================================================ main
def _heads(spec: str | None, quick: bool) -> list[tuple[int, int]]:
    if spec:
        return [tuple(int(v) for v in s.split(":")) for s in spec.split(",")]
    return list(QUICK_HEADS) if quick else [(l, h) for l in U.LAYERS for h in U.HEADS]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=("D", "alpha"), default="D",
                    help="D: IBP / CROWN / alpha-CROWN on the 5,760 units; alpha: the alpha-CROWN settings on prompt 0")
    ap.add_argument("--quick", action="store_true", help=f"heads {list(QUICK_HEADS)}, prompt 0 only (a few minutes)")
    ap.add_argument("--heads", default=None, help="comma-separated layer:head list (overrides --quick's heads)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads-per-solve", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="run at most this many pending units")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default results_dir()/bound_propagation, with --quick .../bound_propagation_quick)")
    ap.add_argument("--refs", type=Path, default=None, help="root of the reference records (default results_dir())")
    ap.add_argument("--summarize", action="store_true", help="only join the stored bounds with the references")
    a = ap.parse_args(argv)

    out = a.out or results_dir() / ("bound_propagation_quick" if a.quick else "bound_propagation")
    refs = a.refs or results_dir()
    if a.quick and a.refs is None:
        global REF_SUFFIX
        REF_SUFFIX = "_quick"
    heads = _heads(a.heads, a.quick)
    universe = tasks(heads, range(U.N_PROMPTS) if (a.part == "D" and not a.quick) else (0,))
    if not a.summarize:
        if a.workers * a.threads_per_solve > compute.thread_budget():
            raise SystemExit(f"workers x threads = {a.workers * a.threads_per_solve} exceeds the budget "
                             f"AIMP_MAX_THREADS = {compute.thread_budget()}")
        U.prepare()  # builds the GPT-2 cache once, before the workers start
        todo = [t for t in universe if not unit_path(out, a.part, *t).exists()]
        run_part(a.part, todo[:a.limit] if a.limit else todo, out, a.workers, a.threads_per_solve, Log(out / "run.log"))
    (summarize_D if a.part == "D" else summarize_alpha)(universe, out, refs)


if __name__ == "__main__":
    main()
