"""CROWN and alpha-CROWN on the GPT-2 varying-query units along the head's output, with the certificate's semidefinite score
intervals supplied as intermediate bounds on the score node (Section 7.2, the comparison with linear bound propagation;
``figdata/P9b_crown_sdp_intervals.csv``).

Units: the varying-query units of Table 6 (all 144 heads, five prompts, query position 4, rho_rel 0.001 / 0.002 / 0.005 /
0.01; 2,880 units): ``bound_propagation/bounds/free/L<ll>/<unit>.json`` of ``experiments/bound_propagation.py --part D``
(the original IBP / CROWN / alpha-CROWN bounds). The reference (best feasible value), the range of motion and the
certificate are those of the comparison table (``bound_propagation.load_reference``: ``gpt2_ladder/units``).

Score intervals: the ones the certificate's semidefinite program used, read from the certificate's record
(``gpt2_ladder/units/L<ll>/<unit>.json``, ``s_lo`` / ``s_hi``; no new solve). They are role-reduced Shor bounds, the
largest of the primal, dual and safe objectives plus a ``1e-7`` relative pad (:func:`aimp.sdp_mosek.shor_score`).

Bound propagation: the module and settings of Table 6 (:mod:`aimp.lirpa`: auto_LiRPA 0.8.0, library defaults, float64,
each perturbed token a separate input in its own l2 ball), one thread. Per unit, three runs on fresh ``BoundedModule``
objects::

  plain  IBP, CROWN, alpha-CROWN in the order of the original runs (reproduces the stored bounds; CROWN's own score-node
         intervals are recorded after CROWN and after alpha-CROWN);
  ref    CROWN, alpha-CROWN with reference_bounds = {score node: (s_lo, s_hi)} (the library intersects its own
         intermediate bounds with them, so the tighter of the two is used per key);
  set    CROWN, alpha-CROWN with interm_bounds = {score node: (s_lo, s_hi)} (the intervals replace the node's bounds).

The score node is the input of the softmax node (the concatenation of the per-key scores). A forward gate (the module
against the float64 head at 100 random admissible inputs) and a check that the intervals contain the scores of those
inputs run first. Units without a reference record (with ``--quick``, the radius 0.01, which ``gpt2_ladder.py --quick`` does
not run) are skipped and counted.

Usage::

    python experiments/review_crown_intervals.py --workers 12     # one JSON per unit under <out>/crown_intervals/
    python experiments/review_crown_intervals.py --quick
"""
from __future__ import annotations

import argparse
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

import review_common as C

GATE_N = 100
GATE_SEED_OFFSET = 61  # the seed of the forward gate of experiments/bound_propagation.py: unit_seed + 61


def unit_files(refs: Path, sfx: str) -> list[Path]:
    return sorted((refs / f"bound_propagation{sfx}" / "bounds" / "free").glob("L*/*.json"))


def sdp_intervals(refs: Path, sfx: str, l, h, p, pos, rho) -> dict:
    from aimp import gpt2_units as U

    f = refs / f"gpt2_ladder{sfx}" / "units" / f"L{l:02d}" / f"{U.unit_name(l, h, p, pos, rho)}.json"
    r = C.read_json(f)
    return {"s_lo": np.asarray(r["s_lo"], float), "s_hi": np.asarray(r["s_hi"], float), "source": C.relpath(f, refs)}


def _score_node(bm):
    sm = [m for _, m in bm._modules.items() if type(m).__name__ == "BoundSoftmax"]
    if len(sm) != 1:
        raise RuntimeError(f"expected one softmax node, found {len(sm)}")
    return sm[0], sm[0].inputs[0]


def _bounds(node):
    if node.lower is None or node.upper is None:
        return None, None
    return node.lower.detach().cpu().numpy().ravel().astype(float), node.upper.detach().cpu().numpy().ravel().astype(float)


def lirpa_runs(inst, s_lo, s_hi, seed: int) -> dict:
    import torch

    from aimp import lirpa
    from aimp import sets as S
    from aimp.heads import forward

    reps = {t: S.affine(inst.sets[t]) for t in inst.tokens}
    pert = lirpa.perturbed_tokens(inst)
    mod = lirpa.head_module(inst, pert, threads=1)
    rng = np.random.default_rng(seed)  # the forward gate of bound_propagation.py, and the intervals at the same points
    gate, s_out = 0.0, 0.0
    for _ in range(GATE_N):
        us, Z = [], inst.centers().copy()
        for t in pert:
            x = rng.standard_normal(reps[t].r)
            x *= reps[t].rho * rng.uniform(0, 1) / np.linalg.norm(x)
            us.append(torch.as_tensor(x[None, :], dtype=torch.float64))
            Z[t] = reps[t].c + reps[t].B @ x
        with torch.no_grad():
            v = float(mod(*us))
        fw = forward(inst, Z)
        gate = max(gate, abs(v - fw["phi"]))
        s_out = max(s_out, float(np.max(np.r_[s_lo - fw["s"], fw["s"] - s_hi])))
    nom = forward(inst, inst.centers())
    out = {"gate_max_abs": gate, "gate_ok": bool(gate <= lirpa.GATE_TOL * max(1.0, abs(nom["phi"]))), "perturbed": pert,
           "sdp_interval_max_violation_on_samples": s_out,
           "sdp_contains_nominal": bool(np.all(s_lo <= nom["s"]) and np.all(nom["s"] <= s_hi))}
    x0, bx = lirpa._inputs(inst, pert)
    lo_t = torch.as_tensor(np.asarray(s_lo, float), dtype=torch.float64)[None, :]
    hi_t = torch.as_tensor(np.asarray(s_hi, float), dtype=torch.float64)[None, :]
    BoundedModule = lirpa._lirpa().BoundedModule
    runs = {}
    for mode, methods in (("plain", ("IBP", "CROWN", "alpha-CROWN")), ("ref", ("CROWN", "alpha-CROWN")),
                          ("set", ("CROWN", "alpha-CROWN"))):
        with lirpa._quiet():
            bm = BoundedModule(mod, x0, device="cpu", verbose=0)
        sm, sc = _score_node(bm)
        runs.setdefault("score_node", sc.name)
        runs.setdefault("softmax_option", getattr(sm, "option", None))
        for meth in methods:
            kw = {}
            if mode == "ref":
                kw["reference_bounds"] = {sc.name: (lo_t, hi_t)}
            elif mode == "set":
                kw["interm_bounds"] = {sc.name: (lo_t, hi_t)}
            t1 = time.perf_counter()
            try:
                with lirpa._quiet():
                    _, ub = bm.compute_bounds(x=bx, method=lirpa.METHODS[meth], bound_lower=False, bound_upper=True, **kw)
                rec = {"ub": float(ub.detach().reshape(-1)[0]), "status": "ok"}
                if meth == "alpha-CROWN":
                    rec["iterations"] = int(bm.bound_opts.get("optimize_bound_args", {}).get("iteration", -1))
            except Exception as e:  # recorded, not fatal
                rec = {"ub": None, "status": f"error: {e!r}"[:300]}
            rec["time_s"] = time.perf_counter() - t1
            rec["score_lo"], rec["score_hi"] = _bounds(sc)
            runs[f"{mode}:{meth}"] = rec
    out["runs"] = runs
    return out


def evaluate(f: Path, refs: Path, sfx: str) -> dict:
    import bound_propagation as BP

    from aimp import gpt2_units as U

    t0 = time.perf_counter()
    l, h, p, pos, rho = C.parse(f)
    cr = C.read_json(f)
    BP.REF_SUFFIX = sfx
    ref_rec, why = BP.load_reference(refs, "free", l, h, p, rho)
    if ref_rec is None:
        raise RuntimeError(f"no reference: {why}")
    inst = U.make_instance(l, h, p, pos, rho)
    si = sdp_intervals(refs, sfx, l, h, p, pos, rho)
    lr = lirpa_runs(inst, si["s_lo"], si["s_hi"], U.unit_seed(l, h, p, pos, rho) + GATE_SEED_OFFSET)
    ref, ref_hi, spread, cert = ref_rec["reference"], ref_rec["reference_hi"], ref_rec["spread"], ref_rec["cert"]
    gc = cert - ref
    rows = {}
    for key, r in lr["runs"].items():
        if not isinstance(r, dict):
            continue
        ub = r["ub"]
        rows[key] = {**r, "gap": None if ub is None else ub - ref,
                     "rel_gap": None if (ub is None or not spread) else (ub - ref) / spread,
                     "ratio_to_cert_gap": None if (ub is None or gc <= 0) else (ub - ref) / gc,
                     "valid": None if ub is None else bool(ub >= ref_hi - C.VALID_TOL * max(1.0, abs(ref_hi)))}
    orig = {}
    for m in ("IBP", "CROWN", "alpha-CROWN"):
        ub = (cr["methods"].get(m) or {}).get("ub")
        orig[m] = {"ub": ub, "rel_gap": None if (ub is None or not spread) else (ub - ref) / spread,
                   "ratio_to_cert_gap": None if (ub is None or gc <= 0) else (ub - ref) / gc}
    w_sdp = si["s_hi"] - si["s_lo"]
    cw = rows["plain:CROWN"]
    w_crown = (cw["score_hi"] - cw["score_lo"]) if cw["score_lo"] is not None else None
    aw = rows["plain:alpha-CROWN"]
    w_acrown = (aw["score_hi"] - aw["score_lo"]) if aw["score_lo"] is not None else None
    return {"layer": l, "head": h, "prompt": p, "pos": pos, "rho_rel": rho, "unit": f.stem, "n": inst.n,
            "record": C.relpath(f, refs), "intervals": si["source"], "reference": ref, "reference_hi": ref_hi,
            "spread": spread, "cert": cert, "cert_gap": gc, "cert_rel_gap": gc / spread if spread else None,
            "original": orig, "runs": rows, "score_node": lr["runs"]["score_node"],
            "softmax_option": lr["runs"]["softmax_option"], "s_lo_sdp": si["s_lo"], "s_hi_sdp": si["s_hi"],
            "width_sdp_max": float(w_sdp.max()),
            "width_crown_max": None if w_crown is None else float(w_crown.max()),
            "width_acrown_max": None if w_acrown is None else float(w_acrown.max()),
            "width_ratio_sdp_to_crown_max_key": None if w_crown is None else float(np.max(w_sdp / np.maximum(w_crown, 1e-300))),
            "width_ratio_sdp_to_crown_median_key": None if w_crown is None else float(np.median(w_sdp / np.maximum(w_crown, 1e-300))),
            "width_ratio_sdp_to_crown_of_max": None if w_crown is None else float(w_sdp.max() / w_crown.max()),
            "sdp_inside_crown": None if w_crown is None else bool(np.all(si["s_lo"] >= cw["score_lo"] - 1e-12)
                                                                  and np.all(si["s_hi"] <= cw["score_hi"] + 1e-12)),
            "repro_abs": {m: (None if (rows.get(f"plain:{m}", {}).get("ub") is None or orig[m]["ub"] is None)
                              else abs(rows[f"plain:{m}"]["ub"] - orig[m]["ub"])) for m in ("IBP", "CROWN", "alpha-CROWN")},
            **{k: lr[k] for k in ("gate_max_abs", "gate_ok", "perturbed", "sdp_interval_max_violation_on_samples",
                                  "sdp_contains_nominal")},
            "auto_lirpa": __import__("auto_LiRPA").__version__, "wall_s": time.perf_counter() - t0}


def unit_path(out: Path, f: Path) -> Path:
    return out / "crown_intervals" / f.parent.name / f.name


def run_task(task: tuple) -> dict:
    fs, out, refs, sfx = task
    warnings.filterwarnings("ignore")
    f = Path(fs)
    path = unit_path(out, f)
    if path.exists():
        return {"name": f.stem, "skipped": True}
    try:
        from aimp.io import peak_rss_gb

        r = evaluate(f, refs, sfx)
        r["peak_rss_gb"] = peak_rss_gb()
        C.write_unit(path, r)
        bad = [f"{k} invalid" for k, x in r["runs"].items() if isinstance(x, dict) and x.get("valid") is False]
        if not r["gate_ok"]:
            bad.append("forward gate")
        if r["sdp_interval_max_violation_on_samples"] > 0:
            bad.append("a sampled score outside the semidefinite intervals")
        return {"name": f.stem, "t": r["wall_s"], "bad": bad}
    except Exception as e:
        return {"name": f.stem, "fatal": repr(e), "tb": traceback.format_exc()[-3000:]}


def main(argv=None) -> None:
    from jmlr_extensions import run_part

    from aimp import compute

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="read the records of the other scripts' --quick runs")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help="default results_dir()/review (with --quick .../review_quick)")
    ap.add_argument("--refs", type=Path, default=None, help="root of the stored records (default results_dir())")
    a = ap.parse_args(argv)
    out, refs, sfx = C.dirs(a.out, a.refs, a.quick)
    if a.workers > compute.thread_budget():
        raise SystemExit(f"--workers {a.workers} exceeds AIMP_MAX_THREADS = {compute.thread_budget()}")
    import bound_propagation as BP

    BP.REF_SUFFIX = sfx
    files, missing = [], []
    for f in unit_files(refs, sfx):
        l, h, p, pos, rho = C.parse(f)
        ok = BP.load_reference(refs, "free", l, h, p, rho)[0] is not None
        (files if ok else missing).append(f)
    print(f"CROWN with the semidefinite score intervals: {len(files)} units"
          + (f"; {len(missing)} without a reference record skipped (first: {missing[0].stem})" if missing else ""))
    run_part(run_task, [(str(f), out, refs, sfx) for f in files if not unit_path(out, f).exists()], a.workers, 1, out,
             "crown_intervals")


if __name__ == "__main__":
    main()
