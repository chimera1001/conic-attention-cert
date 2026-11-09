"""The three terms of the second-order bound (25) (Theorem 12, :func:`aimp.theory.f0v_final`) on the gpt-oss-20b
semidefinite units along the head's output, fixed query (part f, 3,840 units) and varying query (part v, 2,880 units), as
in Section 7.3 (``figdata/P9c_gptoss_bound_terms.csv``, the columns of ``figdata/P4a_bound_terms.csv``)::

  bound (25) = KN(chi, D, D_tot) + eta rad_c + (1/2) sum_j (a_hi_j - a_lo_j) nu_cj   (+ the pruning slack).

The records of ``experiments/gpt_oss.py --part f`` store the three terms; those of part v do not, and they are computed here
by the same call ``theory.f0v_final_fields(inst, ctrl, phi_min_cert, phi_min_strong, eta, pruning_slack, dc)`` from stored
quantities: the unit's certified score intervals (``s_lo``, ``s_hi`` of the record; no new semidefinite solve), the
controls ``bounds.controls(inst, intervals=..., eps_target=0)``, the certified lower bound ``phi_min_cert = -r_pp(-c)``
(the first-order linear program, as in part f), the score covariance ``eta`` of :func:`aimp.theory.nominal_decomposition`
and the directional constants of :func:`aimp.theory.directional_constants`. The same recomputation runs on every part f
unit and is compared with its stored terms (a check of the pipeline).

Gap: part f ``cert - p*`` (the exact fixed-query worst case, midpoint of its bracket); part v ``cert - lb_strong`` (the best
feasible value), as in the paper's gpt-oss-20b table. ``floor``: gap <= 1e-7 x range (as ``figdata/P4a_bound_terms.csv``).
``Pi_v = sum_j S_j (h_max_j - phi_min_strong)_+`` with ``S_j = a_hi_j w_j^2 / 8`` (the value-weighted slack of Figure 3).

Usage::

    python experiments/review_bound_terms.py --workers 12     # one JSON per unit under <out>/bound_terms/<part>/
    python experiments/review_bound_terms.py --quick
"""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np

import review_common as C

PARTS = {"F": "f", "V": "v"}
TERMS = ("F0vf_term_knapsack", "F0vf_term_eta", "F0vf_term_cov", "bound_F0v_final", "D_tot", "budget_binds",
         "F0vf_pruning_slack")


def record_files(refs: Path, sfx: str, part: str) -> list[Path]:
    return sorted((refs / f"gpt_oss{sfx}" / PARTS[part]).glob("L*/*.json"))


def recompute(part: str, rec: dict, l: int, h: int, p: int, pos: int, rho: float) -> dict:
    """Bound (25) from the stored intervals (module docstring)."""
    from aimp import bounds
    from aimp import gpt2_units as U
    from aimp.lower import neg_instance
    from aimp.relax_ps import solve_ps_greedy
    from aimp.theory import directional_constants, f0v_final_fields, nominal_decomposition

    inst = U.make_instance(l, h, p, pos, rho, fixed_query=(part == "F"), model="gpt-oss-20b")
    si = {"method": "stored", "s_lo": np.asarray(rec["s_lo"], float), "s_hi": np.asarray(rec["s_hi"], float)}
    ctrl = bounds.controls(inst, intervals=si, eps_target=0.0, threads=1)
    phi_min_cert = -float(solve_ps_greedy(neg_instance(inst), ctrl, "auto").value)
    nom = nominal_decomposition(inst)
    dc = directional_constants(inst)
    pms = rec["phi_min_strong"] if part == "V" else (rec["min_fq"] if rec.get("exact_status") == "exact" else rec["phi_min_strong"])
    ff = f0v_final_fields(inst, ctrl, phi_min_cert=phi_min_cert, phi_min_strong=pms, eta=nom["eta"],
                          pruning_slack=rec.get("pruning_slack") or 0.0, dc=dc)
    return {"ff": ff, "phi_min_cert": phi_min_cert, "eta": float(nom["eta"]), "rad_c": float(dc["rad_c"]),
            "nu_cj": np.asarray(dc["rho_g"], float), "w_t_recomputed": float(ctrl["w_t"]),
            "d_a_lo": float(np.max(np.abs(np.asarray(ctrl["a_lo"]) - np.asarray(rec["a_lo"])))),
            "d_a_hi": float(np.max(np.abs(np.asarray(ctrl["a_hi"]) - np.asarray(rec["a_hi"])))),
            "d_rad_c": abs(float(dc["rad_c"]) - float(rec["rad_c"])),
            "d_eta_vs_F0_eta": abs(float(nom["eta"]) - float(rec.get("F0_eta") or 0.0)),
            "d_h_max": float(np.max(np.abs(np.asarray(dc["h_sup"]) - np.asarray(rec["h_max"])))) if rec.get("h_max") else None}


def evaluate(part: str, f: Path, refs: Path) -> dict:
    t0 = time.perf_counter()
    l, h, p, pos, rho = C.parse(f)
    rec = C.read_json(f)
    R = recompute(part, rec, l, h, p, pos, rho)
    ff = R["ff"]
    if part == "F":
        ref = rec["p_fq"] if rec.get("p_fq") is not None else 0.5 * (rec["p_fq_lo"] + rec["p_fq_hi"])
        terms = {k: rec[k] for k in TERMS}  # the stored terms (the paper's); the recomputation is the check
        dev = {k: (abs(float(ff[k]) - float(rec[k])) if not isinstance(rec[k], bool) else (bool(ff[k]) != bool(rec[k])))
               for k in TERMS}
    else:
        ref = rec["lb_strong"]
        terms = {k: ff[k] for k in TERMS}
        dev = None
    gap = rec["cert"] - ref
    a_hi = np.clip(np.asarray(rec["a_hi"], float), 0.0, 1.0)
    w = np.asarray(ff["w_j"], float)
    Sj = a_hi * w ** 2 / 8.0
    pms = rec["phi_min_strong"]
    Pi_v = float(np.sum(Sj * np.maximum(np.asarray(rec["h_max"], float) - pms, 0.0)))
    b = float(terms["bound_F0v_final"])
    return {"part": part, "source": f"P8{part}", "unit": f.stem.replace("_fq", ""), "layer": l, "head": h, "prompt": p,
            "pos": pos, "rho_rel": rho, "n": rec["n"], "range": rec["range"], "range_no_sink": rec.get("range_no_sink"),
            "S": rec["S"], "w_t": rec["w_t"], "theta": rec["theta"], "gap": gap,
            "floor": bool(gap <= 1e-7 * rec["range"]), **terms, "n_pruned": len(rec.get("pruned_keys") or []),
            "Pi_v": Pi_v, "reference": ref, "reference_kind": "p* (exact)" if part == "F" else "lb_strong",
            "cert": rec["cert"], "phi_min_cert": R["phi_min_cert"], "phi_min_strong": pms, "eta": R["eta"],
            "rad_c": rec["rad_c"], "gap_over_bound": gap / b if b > 0 else None, "bound_holds": bool(gap <= b),
            "S_recomputed": float(Sj.sum()), "w_t_recomputed": R["w_t_recomputed"],
            "checks": {k: R[k] for k in ("d_a_lo", "d_a_hi", "d_rad_c", "d_eta_vs_F0_eta", "d_h_max")},
            "recomputed": {k: ff[k] for k in TERMS}, "deviation_stored_vs_recomputed": dev, "chi_j": ff["chi_j"],
            "D_j": ff["D_j"], "d_knapsack": ff["d_knapsack"], "a_lo": rec["a_lo"], "a_hi": rec["a_hi"], "nu_cj": R["nu_cj"],
            "h_max": rec["h_max"], "record": C.relpath(f, refs), "wall_s": time.perf_counter() - t0}


def unit_path(out: Path, part: str, f: Path) -> Path:
    return out / "bound_terms" / part / f.parent.name / f.name


def run_task(task: tuple) -> dict:
    part, files, out, refs = task
    name = f"{part}:{Path(files[0]).stem[:6]}"
    try:
        t0, bad = time.perf_counter(), []
        for fs in files:
            f = Path(fs)
            path = unit_path(out, part, f)
            if path.exists():
                continue
            r = evaluate(part, f, refs)
            C.write_unit(path, r)
            if not r["bound_holds"]:
                bad.append(f"{f.stem}: gap above bound (25)")
        return {"name": name, "t": time.perf_counter() - t0, "bad": bad}
    except Exception as e:
        return {"name": name, "fatal": repr(e), "tb": traceback.format_exc()[-3000:]}


def tasks(out: Path, refs: Path, sfx: str) -> list[tuple]:
    res = []
    for part in PARTS:
        groups: dict = {}
        for f in record_files(refs, sfx, part):
            groups.setdefault(C.parse(f)[:2], []).append(str(f))
        res += [(part, fs, out, refs) for _, fs in sorted(groups.items())]
    return res


def main(argv=None) -> None:
    from jmlr_extensions import run_part

    from aimp import compute

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="read the records of experiments/gpt_oss.py --quick")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help="default results_dir()/review (with --quick .../review_quick)")
    ap.add_argument("--refs", type=Path, default=None, help="root of the stored records (default results_dir())")
    a = ap.parse_args(argv)
    out, refs, sfx = C.dirs(a.out, a.refs, a.quick)
    if a.workers > compute.thread_budget():
        raise SystemExit(f"--workers {a.workers} exceeds AIMP_MAX_THREADS = {compute.thread_budget()}")
    T = tasks(out, refs, sfx)
    print(f"terms of bound (25), gpt-oss-20b: {sum(len(t[1]) for t in T)} units in {len(T)} groups")
    run_part(run_task, [t for t in T if not all(unit_path(out, t[0], Path(f)).exists() for f in t[1])], a.workers, 1,
             out, "bound_terms")


if __name__ == "__main__":
    main()
