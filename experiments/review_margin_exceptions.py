"""The GPT-2 logit-margin exceptions (fixed query, position 4): the units on which the best of IBP / CROWN / alpha-CROWN is
not looser than the certificate, read through the second-order bound (39) (Section 7.2, the logit-margin comparison;
``figdata/P9d_gpt2_margin_exceptions.csv``).

Selection: over the fixed-query logit-margin units of ``experiments/jmlr_extensions.py --part lm``
(``jmlr/lm/fq/L<ll>/<unit>.json``, 2,160 units; the bound propagation results are in the same records),
``best = min(IBP, CROWN, alpha-CROWN)`` upper bound, an exception if ``cert >= best``; a *tie at the numerical floor* if
``cert - p* <= 1e-6 max(1, |p*|)`` (``p*`` the exact fixed-query worst case, midpoint of its bracket). On the paper's units
this gives 13 exceptions, 7 of them ties.

Per exception: the terms of bound (39) as stored (``F0vf_term_knapsack``, ``F0vf_term_eta`` (0 with a fixed query),
``F0vf_term_cov``, ``bound_F0v_final``), the certificate's gap, the best bound-propagation gap, gap / bound and the distance
``value_safe - primal objective`` of the solve behind the final ``value_safe``. When the final ``value_safe`` comes from the
re-solve of Appendix B (1e-9), the record holds that solve's dual objective but not its primal objective: the re-solve is
repeated (the call of :func:`aimp.certify.sdp_retol`, ``solve_sdp1(..., tol=None, cut="auto")``, on the controls of
:func:`aimp.pipeline.controls`) in a fresh process at the record's thread count, and its primal objective is used only if it
reproduces the stored ``value_safe`` exactly. The per-key quantities of the bound (``chi_j``) are recomputed from the stored
intervals (:func:`aimp.theory.f0v_final_fields`, as ``experiments/review_bound_terms.py``) and the recomputed bound is
compared with the stored one.

Usage::

    python experiments/review_margin_exceptions.py          # <out>/margin_exceptions/exceptions.json (+ resolve/*.json)
    python experiments/review_margin_exceptions.py --quick
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import review_common as C

METHODS = ("IBP", "CROWN", "alpha-CROWN")


def all_units(refs: Path, sfx: str) -> list[dict]:
    rows = []
    for f in sorted((refs / f"jmlr{sfx}" / "lm" / "fq").glob("L*/*.json")):
        l, h, p, pos, rho = C.parse(f)
        r = C.read_json(f)
        ref = r["p_fq"] if r.get("p_fq") is not None else 0.5 * (r["p_fq_lo"] + r["p_fq_hi"])
        ubs = {m: r["lirpa"]["methods"][m]["ub"] for m in METHODS}
        best_m = min(ubs, key=lambda m: ubs[m])
        rows.append({"file": str(f), "layer": l, "head": h, "prompt": p, "pos": pos, "rho_rel": rho, "rec": r,
                     "reference": ref, "spread": r["spread_fq"], "cert": r["cert"], "ubs": ubs, "best_method": best_m,
                     "best_ub": ubs[best_m]})
    return rows


def final_solve(r: dict) -> dict:
    """Which solve produced the final value_safe, and its stored primal / dual objectives."""
    ri = r.get("retol_info") or {}
    if r.get("retol") and ri.get("value_safe") is not None and ri["value_safe"] == r["value_safe"]:
        return {"final_solve": "re-solve (1e-9)", "primal": ri.get("value_primal"), "dual": ri.get("value_dual"),
                "safe_slack": ri.get("safe_slack")}
    return {"final_solve": "first solve (Appendix B tolerances)", "primal": r.get("sdp_value_primal"),
            "dual": r.get("sdp_value_dual"), "safe_slack": r.get("safe_slack")}


def chi_and_bound(r: dict, l: int, h: int, p: int, rho: float) -> dict:
    """``chi_j`` and bound (39) recomputed from the stored score intervals (no semidefinite solve)."""
    from aimp import bounds
    from aimp import gpt2_units as U
    from aimp.lower import neg_instance
    from aimp.relax_ps import solve_ps_greedy
    from aimp.theory import directional_constants, f0v_final_fields

    inst = U.make_instance(l, h, p, C.POS, rho, fixed_query=True, spec="lm")
    ctrl = bounds.controls(inst, intervals={"method": "stored", "s_lo": np.asarray(r["s_lo"], float),
                                            "s_hi": np.asarray(r["s_hi"], float)}, eps_target=0.0, threads=1)
    pmc = -float(solve_ps_greedy(neg_instance(inst), ctrl, "auto").value)
    pms = r["min_fq"] if r.get("exact_status") == "exact" else r["phi_min_strong"]
    ff = f0v_final_fields(inst, ctrl, phi_min_cert=pmc, phi_min_strong=pms, eta=r.get("F0_eta") or 0.0,
                          pruning_slack=r.get("pruning_slack") or 0.0, dc=directional_constants(inst))
    return {"chi_j": np.asarray(ff["chi_j"], float), "bound": float(ff["bound_F0v_final"])}


# --------------------------------------------------------------------------------------------- repeated re-solve (child)
def resolve_child(l: int, h: int, p: int, rho: float, threads: int) -> dict:
    from aimp import gpt2_units as U
    from aimp.pipeline import controls
    from aimp.sdp_mosek import solve_sdp1

    fq = U.make_instance(l, h, p, C.POS, rho, fixed_query=True, spec="lm")
    si, ctrl = controls(fq, threads)
    t0 = time.perf_counter()
    r = solve_sdp1(fq, ctrl, threads=threads, time_limit=None, tol=None, cut="auto")
    dg = r.diagnostics
    return {"value_safe": dg.get("value_safe"), "value_primal": dg.get("value_primal"), "value_dual": dg.get("value_dual"),
            "status": r.status, "solsta": dg.get("solsta"), "primal_max_violation": dg.get("primal_max_violation"),
            "num_iters": dg.get("num_iters"), "s_lo": ctrl["s_lo"], "s_hi": ctrl["s_hi"], "wall_s": time.perf_counter() - t0}


def repeat_resolve(u: dict, out: Path) -> dict:
    r = u["rec"]
    threads = int(r.get("threads") or 4)
    name = Path(u["file"]).stem
    path = out / "margin_exceptions" / "resolve" / f"{name}.json"
    if path.exists():
        return C.read_json(path)
    env = dict(os.environ)
    env.update({"AIMP_MAX_THREADS": str(threads), "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
                "OPENBLAS_NUM_THREADS": str(threads), "CUDA_VISIBLE_DEVICES": ""})
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", str(u["layer"]), str(u["head"]), str(u["prompt"]),
           repr(u["rho_rel"]), str(threads)]
    pr = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    res = json.loads(pr.stdout.strip().splitlines()[-1])
    ri = r.get("retol_info") or {}
    res.update({"unit": name, "threads": threads, "stored_value_safe": r["value_safe"],
                "stored_retol_value_safe": ri.get("value_safe"), "stored_retol_value_dual": ri.get("value_dual"),
                "reproduced_value_safe": bool(res["value_safe"] == ri.get("value_safe")),
                "reproduced_value_dual": bool(res["value_dual"] == ri.get("value_dual")),
                "d_value_safe": abs(res["value_safe"] - ri["value_safe"]) if ri.get("value_safe") is not None else None,
                "s_intervals_equal_stored": bool(np.array_equal(np.asarray(res["s_lo"]), np.asarray(r["s_lo"])) and
                                                 np.array_equal(np.asarray(res["s_hi"]), np.asarray(r["s_hi"])))})
    C.write_unit(path, res)
    res["threads"] = threads
    return res


def exceptions(out: Path, refs: Path, sfx: str) -> dict:
    U_ = all_units(refs, sfx)
    exc = []
    for u in U_:
        if not (u["cert"] >= u["best_ub"]):
            continue
        r = u["rec"]
        ref = u["reference"]
        gc, gb = u["cert"] - ref, u["best_ub"] - ref
        fs = final_solve(r)
        rs = None
        if fs["primal"] is None:
            rs = repeat_resolve(u, out)
            if rs["reproduced_value_safe"]:
                fs.update({"primal": rs["value_primal"], "primal_source": "repeated re-solve (reproduces the stored value_safe)"})
            else:
                fs.update({"primal_source": "repeated re-solve did NOT reproduce the stored value_safe; primal not used"})
        else:
            fs["primal_source"] = "record"
        terms = {"knapsack": r["F0vf_term_knapsack"], "eta": r["F0vf_term_eta"], "cov": r["F0vf_term_cov"]}
        b = r["bound_F0v_final"]
        dom3 = max(terms, key=lambda k: terms[k])
        comps = {**terms, "pruning slack": r.get("F0vf_pruning_slack") or 0.0}
        dom = max(comps, key=lambda k: comps[k])
        cb = chi_and_bound(r, u["layer"], u["head"], u["prompt"], u["rho_rel"])
        exc.append({"unit": Path(u["file"]).stem.replace("_fq", ""), "layer": u["layer"], "head": u["head"],
                    "prompt": u["prompt"], "rho_rel": u["rho_rel"], "n": r["n"], "w_t": r["w_t"], "S": r["S"],
                    "Delta": r["Delta"], "theta": r["theta"], "rad_c": r["rad_c"], "D_tot": r["D_tot"],
                    "budget_binds": r["budget_binds"], "keys_with_chi_pos": int((cb["chi_j"] > 0).sum()),
                    "d_bound_recomputed": abs(cb["bound"] - b),
                    "term_knapsack": terms["knapsack"], "term_eta": terms["eta"], "term_cov": terms["cov"],
                    "pruning_slack": r.get("F0vf_pruning_slack") or 0.0, "n_pruned": len(r.get("pruned_keys") or []),
                    "bound_39": b, "dominant_term": dom, "dominant_of_three": dom3,
                    "share_knapsack": terms["knapsack"] / b, "share_eta": terms["eta"] / b, "share_cov": terms["cov"] / b,
                    "share_pruning_slack": comps["pruning slack"] / b,
                    "gap_over_knapsack": gc / terms["knapsack"] if terms["knapsack"] > 0 else None,
                    "p_star": ref, "p_star_lo": r["p_fq_lo"], "p_star_hi": r["p_fq_hi"], "exact_status": r["exact_status"],
                    "spread": u["spread"], "cert": u["cert"], "route": r.get("route"), "r_greedy": r["r_greedy"],
                    "value_safe": r["value_safe"], "cert_from": "SDP value_safe" if r["value_safe"] <= r["r_greedy"] else "greedy",
                    "cert_gap": gc, "cert_rel_gap": gc / u["spread"], "best_method": u["best_method"],
                    "best_bp_ub": u["best_ub"], "best_bp_gap": gb, "best_bp_rel_gap": gb / u["spread"],
                    "IBP_ub": u["ubs"]["IBP"], "CROWN_ub": u["ubs"]["CROWN"], "alpha_CROWN_ub": u["ubs"]["alpha-CROWN"],
                    "ratio_cert_to_best": gc / gb if gb > 0 else None, "cert_minus_best": u["cert"] - u["best_ub"],
                    "gap_over_bound": gc / b if b > 0 else None,
                    "floor_tie": bool(gc <= 1e-6 * max(1.0, abs(ref))), "retol": bool(r.get("retol")),
                    "final_solve": fs["final_solve"], "final_primal": fs["primal"], "final_dual": fs["dual"],
                    "primal_source": fs["primal_source"],
                    "safe_minus_primal": (r["value_safe"] - fs["primal"]) if fs["primal"] is not None else None,
                    "safe_minus_dual": (r["value_safe"] - fs["dual"]) if fs["dual"] is not None else None,
                    "first_solve_safe_minus_primal": r["value_safe_first"] - r["sdp_value_primal"],
                    "first_solve_value_safe": r["value_safe_first"], "first_solve_primal": r["sdp_value_primal"],
                    "primal_max_violation_first": r.get("primal_max_violation"),
                    "gap_minus_safe_minus_primal": (gc - (r["value_safe"] - fs["primal"])) if fs["primal"] is not None else None,
                    "resolve_check": None if rs is None else {k: rs[k] for k in ("reproduced_value_safe", "reproduced_value_dual",
                                                                                 "d_value_safe", "s_intervals_equal_stored",
                                                                                 "value_primal", "primal_max_violation", "threads")},
                    "record": C.relpath(Path(u["file"]), refs), "lirpa_record": C.relpath(Path(u["file"]), refs)})
    ctx = []  # the same quantities over all units, for the percentiles and the comparison
    for u in U_:
        r = u["rec"]
        b = r["bound_F0v_final"]
        terms = {"knapsack": r["F0vf_term_knapsack"], "eta": r["F0vf_term_eta"], "cov": r["F0vf_term_cov"]}
        comps = {**terms, "pruning slack": r.get("F0vf_pruning_slack") or 0.0}
        gc = u["cert"] - u["reference"]
        ctx.append({"layer": u["layer"], "head": u["head"], "prompt": u["prompt"], "rho_rel": u["rho_rel"],
                    "dominant": max(comps, key=lambda k: comps[k]), "dominant_of_three": max(terms, key=lambda k: terms[k]),
                    "share_cov": terms["cov"] / b if b > 0 else None,
                    "share_knapsack": terms["knapsack"] / b if b > 0 else None, "gap_over_bound": gc / b if b > 0 else None,
                    "gap_over_knapsack": gc / terms["knapsack"] if terms["knapsack"] > 0 else None,
                    "pruning_slack": comps["pruning slack"], "w_t": r["w_t"],
                    "cert_rel_gap": gc / u["spread"], "best_bp_rel_gap": (u["best_ub"] - u["reference"]) / u["spread"],
                    "cert_gap": gc, "best_bp_gap": u["best_ub"] - u["reference"],
                    "floor_tie": bool(gc <= 1e-6 * max(1.0, abs(u["reference"]))), "exception": bool(u["cert"] >= u["best_ub"])})
    return {"n_units": len(U_), "n_exceptions": len(exc), "exceptions": exc, "context": ctx}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="read the records of experiments/jmlr_extensions.py --quick")
    ap.add_argument("--out", type=Path, default=None, help="default results_dir()/review (with --quick .../review_quick)")
    ap.add_argument("--refs", type=Path, default=None, help="root of the stored records (default results_dir())")
    ap.add_argument("--child", nargs=5, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child:
        from aimp.io import jsonable

        l, h, p, rho, threads = int(a.child[0]), int(a.child[1]), int(a.child[2]), float(a.child[3]), int(a.child[4])
        print(json.dumps(jsonable(resolve_child(l, h, p, rho, threads))))
        return
    from aimp import compute

    compute.apply_thread_budget(1)
    out, refs, sfx = C.dirs(a.out, a.refs, a.quick)
    E = exceptions(out, refs, sfx)
    C.write_unit(out / "margin_exceptions" / "exceptions.json", E)
    nt = sum(e["floor_tie"] for e in E["exceptions"])
    print(f"logit-margin exceptions: {E['n_exceptions']} of {E['n_units']} fixed-query units ({nt} ties at the floor); "
          f"{out / 'margin_exceptions' / 'exceptions.json'}")


if __name__ == "__main__":
    main()
