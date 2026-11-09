"""The tables of the review computations (``experiments/review_*.py``) from their per-unit records, the figure-data files
``P9a``-``P9d`` (the files of ``figdata/``) and the statistics quoted in Sections 7.2 and 7.3.

Writes into ``<out>/summary`` (default ``results_dir()/review/summary``, with ``--quick`` ``results_dir()/review_quick/summary``):

* ``taylor_fixed_query.csv`` (one row per unit of ``review_taylor.py``: the curvature certificate and its parts, the
  references, the certificate, gaps, ratios, per-series slopes, reconstruction checks) and ``P9a_taylor_fixed_query.csv``;
* ``crown_sdp_intervals.csv`` (``review_crown_intervals.py``: the original and the new CROWN / alpha-CROWN bounds, with
  the semidefinite score intervals as reference bounds (``ref``) and as the node's bounds (``set``), their ratios to the
  certificate's gap, the interval widths and the checks) and ``P9b_crown_sdp_intervals.csv``;
* ``gptoss_bound_terms.csv`` (``review_bound_terms.py``, the columns of ``figdata/P4a_bound_terms.csv``) and
  ``P9c_gptoss_bound_terms.csv``;
* ``gpt2_margin_exceptions.csv`` (``review_margin_exceptions.py``, one row per exception, with the records it was read
  from) and ``P9d_gpt2_margin_exceptions.csv`` (the same without the two record columns; ``bound_F0v`` is bound (25)).

The ``P9*`` files hold the columns of the figure data with 10 significant digits (``P9d`` in full). Each is derived from
the full table as written (read back with pandas), as the paper's files were, so that on the stored records of the paper
a rerun reproduces the four files of ``figdata/`` byte for byte; ``python figscripts/stats.py`` prints the paper's
numbers from them.

Usage::

    python experiments/review_summary.py [--quick] [--out DIR]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import review_common as C

P4A_COLS = ["source", "unit", "layer", "head", "prompt", "pos", "rho_rel", "n", "range", "S", "w_t", "theta", "gap", "floor",
            "bound_F0v_final", "F0vf_term_knapsack", "F0vf_term_eta", "F0vf_term_cov", "D_tot", "budget_binds", "n_pruned",
            "Pi_v"]
T1_COLS = ["set", "model", "spec", "layer", "head", "prompt", "pos", "rho_rel", "n", "n_var", "phi_nominal", "lin_term",
           "Delta_s", "nu_c", "rad_c", "curv_rad", "curv_cov", "curv_term", "r_lin", "r_T", "p_star", "p_star_lo",
           "p_star_hi", "exact_status", "spread", "cert", "r_greedy", "w_t", "Delta", "S", "gap_T", "gap_lin", "gap_cert",
           "rel_gap_T", "rel_gap_cert", "rel_gap_greedy", "ratio_T_to_cert", "valid_T_strict", "valid_T",
           "margin_T_over_hi", "T_vacuous_range", "T_above_greedy", "T_tighter_than_cert", "curv_share_of_gap_T",
           "slope_T_ls", "slope_T_small", "slope_cert_ls", "slope_cert_small", "slope_range_ls", "d_phi_nominal",
           "d_rad_c", "d_s_interval", "d_h_max", "record", "git"]
P9A_COLS = ["set", "model", "spec", "layer", "head", "prompt", "pos", "rho_rel", "Delta_s", "nu_c", "rad_c", "r_T", "p_star",
            "p_star_hi", "cert", "r_greedy", "spread", "slope_T_small", "slope_T_ls", "slope_cert_small"]
RUNS = ("plain:IBP", "plain:CROWN", "plain:alpha-CROWN", "ref:CROWN", "ref:alpha-CROWN", "set:CROWN", "set:alpha-CROWN")
P9B_COLS = ["layer", "head", "prompt", "pos", "rho_rel", "cert_gap", "spread", "orig_CROWN_ratio", "orig_alpha-CROWN_ratio",
            "ref_CROWN_ratio", "ref_alpha-CROWN_ratio", "width_ratio_sdp_to_crown_max_key"]
T5_COLS = ["unit", "layer", "head", "prompt", "rho_rel", "n", "w_t", "S", "term_knapsack", "term_eta", "term_cov",
           "pruning_slack", "bound_F0v", "cert_gap", "best_bp_gap", "best_method", "gap_over_bound", "safe_minus_primal",
           "dominant_term", "dominant_of_three", "share_knapsack", "share_cov", "share_pruning_slack", "gap_over_knapsack",
           "pct_cert_rel_gap", "pct_best_bp_rel_gap", "pct_gap_over_bound", "primal_minus_best", "n_pruned", "floor_tie",
           "ratio_cert_to_best", "cert_minus_best", "cert_rel_gap", "best_bp_rel_gap", "p_star", "p_star_lo", "p_star_hi",
           "spread", "cert", "value_safe", "r_greedy", "cert_from", "IBP_ub", "CROWN_ub", "alpha_CROWN_ub", "retol",
           "final_solve", "final_primal", "final_dual", "primal_source", "safe_minus_dual", "first_solve_value_safe",
           "first_solve_primal", "first_solve_safe_minus_primal", "gap_minus_safe_minus_primal", "Delta", "theta", "rad_c",
           "D_tot", "budget_binds", "keys_with_chi_pos", "record", "lirpa_record"]
P9D_COLS = [c for c in T5_COLS if c not in ("record", "lirpa_record")]
SET_LABEL = {"gpt2_a": "GPT-2, head's output", "gpt2_lm": "GPT-2, logit margin", "oss_a": "gpt-oss-20b, head's output",
             "oss_lm": "gpt-oss-20b, logit margin"}


def _csv(df, path: Path, float_format: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, lineterminator="\n", float_format=float_format)
    tmp.replace(path)


def _figdata(full: Path, path: Path, cols: list, float_format: str | None) -> None:
    """A figure-data file: the columns ``cols`` of the full table ``full`` as written, read back with pandas' default
    parser (as the paper's files were derived from the full tables)."""
    import pandas as pd

    _csv(pd.read_csv(full)[cols], path, float_format)


def spear(a, b):
    import pandas as pd
    from scipy.stats import spearmanr

    a, b = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    m = a.notna() & b.notna()
    return float(spearmanr(a[m], b[m]).statistic) if m.sum() > 2 else None


def ls_slope(rhos, ys):
    pts = [(math.log(r), math.log(y)) for r, y in zip(rhos, ys) if y is not None and np.isfinite(y) and y > 0]
    if len(pts) < 2:
        return None
    x, y = np.array(pts).T
    return float(np.polyfit(x, y, 1)[0])


def small_slope(rhos, ys):
    """Slope between the two smallest radii (as the paper's exact table)."""
    o = np.argsort(rhos)[:2]
    r, y = np.asarray(rhos, float)[o], np.asarray(ys, float)[o]
    if len(r) < 2 or not (y > 0).all():
        return None
    return float((math.log(y[1]) - math.log(y[0])) / (math.log(r[1]) - math.log(r[0])))


# ------------------------------------------------------------------------------------------------ curvature certificate
def taylor(out: Path):
    import pandas as pd

    rows = C.load_units(out, "taylor")
    if not rows:
        return None
    U = pd.DataFrame(rows)
    keys = ["set", "layer", "head", "prompt"]
    sl = [{**dict(zip(keys, k)), "slope_T_ls": ls_slope(g.rho_rel, g.gap_T), "slope_T_small": small_slope(g.rho_rel, g.gap_T),
           "slope_cert_ls": ls_slope(g.rho_rel, g.gap_cert), "slope_cert_small": small_slope(g.rho_rel, g.gap_cert),
           "slope_range_ls": ls_slope(g.rho_rel, g.spread)} for k, g in U.groupby(keys)]
    U = U.merge(pd.DataFrame(sl), on=keys, how="left")
    U["T_vacuous_range"] = U.rel_gap_T > 1.0
    U["T_above_greedy"] = U.r_T > U.r_greedy
    U["T_tighter_than_cert"] = U.r_T < U.cert
    U["curv_share_of_gap_T"] = U.curv_term / U.gap_T
    U = U.sort_values(["set", "layer", "head", "prompt", "rho_rel"])[T1_COLS]
    _csv(U, out / "summary" / "taylor_fixed_query.csv")
    _figdata(out / "summary" / "taylor_fixed_query.csv", out / "summary" / "P9a_taylor_fixed_query.csv", P9A_COLS, "%.10g")
    print("\nCurvature (Taylor) certificate, fixed query: relative gaps (medians), median per-unit ratio of its gap to the "
          "certificate's")
    S = U.groupby(["set", "rho_rel"], sort=False).agg(
        units=("r_T", "size"), violations=("valid_T", lambda x: int((~x.astype(bool)).sum())),
        rel_gap_T=("rel_gap_T", "median"), rel_gap_cert=("rel_gap_cert", "median"),
        ratio_median=("ratio_T_to_cert", "median"), ratio_min=("ratio_T_to_cert", "min"),
        above_first_order=("T_above_greedy", "mean"), gap_above_range=("T_vacuous_range", "mean")).reset_index()
    print(S.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    ser = U.drop_duplicates(["set", "layer", "head", "prompt"])
    print("median per-series slopes (two smallest radii / all radii):")
    for s, g in ser.groupby("set", sort=False):
        print(f"  {SET_LABEL.get(s, s)}: curvature certificate {g.slope_T_small.median():.2f} / {g.slope_T_ls.median():.2f}, "
              f"certificate {g.slope_cert_small.median():.2f} / {g.slope_cert_ls.median():.2f} ({len(g)} series)")
    d = np.nanmax(U[["d_phi_nominal", "d_rad_c", "d_s_interval", "d_h_max"]].to_numpy(dtype=float))
    print(f"valid on {int(U.valid_T.sum())} / {len(U)} units (strictly above the bracket's upper end on "
          f"{int(U.valid_T_strict.sum())}); instance reconstruction against the records to {d:.1e}")
    st = out / "taylor_selftest.json"
    if st.exists():
        r = C.read_json(st)
        print(f"self-test: {'passed' if r['ok'] else 'FAILED'} on {len(r['rows'])} units (largest |phi''| / bound "
              f"{max(x['phi2_over_bound_max'] for x in r['rows']):.3f}, gradient error "
              f"{max(x['grad_rel_err_max'] for x in r['rows']):.1e})")
    return U


# ------------------------------------------------------------------------------------------------ CROWN with the intervals
def crown_intervals(out: Path):
    import pandas as pd

    rows = []
    for d in C.load_units(out, "crown_intervals"):
        r = {k: d[k] for k in ("layer", "head", "prompt", "pos", "rho_rel", "n", "reference", "reference_hi", "spread",
                               "cert", "cert_gap", "cert_rel_gap", "width_sdp_max", "width_crown_max", "width_acrown_max",
                               "width_ratio_sdp_to_crown_max_key", "width_ratio_sdp_to_crown_median_key",
                               "width_ratio_sdp_to_crown_of_max", "sdp_inside_crown", "gate_ok", "gate_max_abs",
                               "sdp_interval_max_violation_on_samples", "sdp_contains_nominal", "score_node", "softmax_option",
                               "intervals", "record", "auto_lirpa", "git")}
        for m in ("IBP", "CROWN", "alpha-CROWN"):
            o = d["original"].get(m, {})
            r.update({f"orig_{m}_ub": o.get("ub"), f"orig_{m}_rel_gap": o.get("rel_gap"),
                      f"orig_{m}_ratio": o.get("ratio_to_cert_gap"), f"repro_{m}_abs": d["repro_abs"].get(m)})
        for k in RUNS:
            x, p = d["runs"][k], k.replace(":", "_")
            r.update({f"{p}_ub": x["ub"], f"{p}_rel_gap": x["rel_gap"], f"{p}_ratio": x["ratio_to_cert_gap"],
                      f"{p}_valid": x["valid"], f"{p}_time_s": x["time_s"], f"{p}_status": x["status"]})
        rows.append(r)
    if not rows:
        return None
    D = pd.DataFrame(rows).sort_values(["rho_rel", "layer", "head", "prompt"])
    D["set_minus_ref_CROWN"] = (D["set_CROWN_ub"] - D["ref_CROWN_ub"]).abs()
    D["set_minus_ref_alpha"] = (D["set_alpha-CROWN_ub"] - D["ref_alpha-CROWN_ub"]).abs()
    _csv(D, out / "summary" / "crown_sdp_intervals.csv")
    _figdata(out / "summary" / "crown_sdp_intervals.csv", out / "summary" / "P9b_crown_sdp_intervals.csv", P9B_COLS, "%.10g")
    print("\nCROWN / alpha-CROWN with the certificate's semidefinite score intervals (varying query): median per-unit ratios "
          "to the certificate's gap, without / with the intervals")
    S = D.groupby("rho_rel").agg(units=("cert", "size"), CROWN=("orig_CROWN_ratio", "median"),
                                 CROWN_sdp=("ref_CROWN_ratio", "median"), alpha=("orig_alpha-CROWN_ratio", "median"),
                                 alpha_sdp=("ref_alpha-CROWN_ratio", "median"),
                                 smallest_alpha_sdp=("ref_alpha-CROWN_ratio", "min"),
                                 width_sdp_over_crown=("width_ratio_sdp_to_crown_of_max", "median")).reset_index()
    print(S.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    rep = np.nanmax(D[[f"repro_{m}_abs" for m in ("IBP", "CROWN", "alpha-CROWN")]].to_numpy(dtype=float))
    inval = int(sum((D[f"{k.replace(':', '_')}_valid"] == False).sum() for k in RUNS))  # noqa: E712
    print(f"certificate tighter than alpha-CROWN with the intervals on {(D['ref_alpha-CROWN_ratio'] > 1).mean():.1%}; "
          f"stored bounds reproduced to {rep:.1e}; invalid bounds {inval}; forward gate {int(D.gate_ok.sum())} / {len(D)}; "
          f"largest sampled score outside the intervals {D.sdp_interval_max_violation_on_samples.max():.1e}; "
          f"ref vs set {max(D.set_minus_ref_CROWN.max(), D.set_minus_ref_alpha.max()):.1e}")
    return D


# ------------------------------------------------------------------------------------------------ terms of bound (25)
def bound_terms(out: Path):
    import pandas as pd

    rows = C.load_units(out, "bound_terms")
    if not rows:
        return None
    D = pd.DataFrame(rows)
    T = D[P4A_COLS].sort_values(["source", "rho_rel", "layer", "head", "prompt"])
    _csv(T, out / "summary" / "gptoss_bound_terms.csv")
    _figdata(out / "summary" / "gptoss_bound_terms.csv", out / "summary" / "P9c_gptoss_bound_terms.csv", P4A_COLS, "%.10g")
    print("\nTerms of bound (25) on gpt-oss-20b (KN = knapsack term, cov = covariance term; shares of the bound)")
    for part, P in D.groupby("part"):
        tm = P[["F0vf_term_knapsack", "F0vf_term_eta", "F0vf_term_cov"]].to_numpy(dtype=float)
        a = P[~P.floor]
        print(f"  {'fixed' if part == 'F' else 'varying'} query ({len(P)} units): median shares KN / eta rad_c / cov "
              f"{(P.F0vf_term_knapsack / P.bound_F0v_final).median():.2f} / {(P.F0vf_term_eta / P.bound_F0v_final).median():.2f}"
              f" / {(P.F0vf_term_cov / P.bound_F0v_final).median():.2f}; KN largest on {(np.argmax(tm, axis=1) == 0).mean():.1%}; "
              f"gap < KN + eta rad_c on {(P.gap < P.F0vf_term_knapsack + P.F0vf_term_eta).mean():.1%}; Spearman(gap, KN) "
              f"{spear(a.gap, a.F0vf_term_knapsack):.2f}, (gap, S) {spear(a.gap, a.S):.2f}; gap / bound median "
              f"{(P.gap / P.bound_F0v_final).median():.2f}, max {(P.gap / P.bound_F0v_final).max():.2f}; bound holds on "
              f"{int(P.bound_holds.sum())}")
    dev = pd.DataFrame([x for x in D.deviation_stored_vs_recomputed if isinstance(x, dict)])
    if len(dev):
        num = [c for c in dev.columns if c != "budget_binds"]
        print(f"  part f: the recomputation reproduces the stored terms to {float(dev[num].abs().max().max()):.1e}")
    return D


# ------------------------------------------------------------------------------------------------ logit-margin exceptions
def margin_exceptions(out: Path):
    import pandas as pd

    f = out / "margin_exceptions" / "exceptions.json"
    if not f.exists():
        return None
    E = C.read_json(f)
    D, X = pd.DataFrame(E["exceptions"]), pd.DataFrame(E["context"])
    D = D.rename(columns={"bound_39": "bound_F0v"})  # the name in records written before the rename
    print(f"\nGPT-2 logit-margin exceptions: {len(D)} of {E['n_units']} fixed-query units")
    if not len(D):
        return D
    for c, n in (("cert_rel_gap", "pct_cert_rel_gap"), ("best_bp_rel_gap", "pct_best_bp_rel_gap"),
                 ("gap_over_bound", "pct_gap_over_bound")):
        X[n] = X.groupby("rho_rel")[c].rank(pct=True)  # percentile among the units of the same radius
    D = D.merge(X[["layer", "head", "prompt", "rho_rel", "pct_cert_rel_gap", "pct_best_bp_rel_gap", "pct_gap_over_bound"]],
                on=["layer", "head", "prompt", "rho_rel"], how="left")
    D["primal_minus_best"] = D.final_primal - D.best_bp_ub  # the relaxation's primal objective above the best bound?
    D = D.sort_values(["floor_tie", "ratio_cert_to_best"])
    _csv(D[T5_COLS], out / "summary" / "gpt2_margin_exceptions.csv")
    _figdata(out / "summary" / "gpt2_margin_exceptions.csv", out / "summary" / "P9d_gpt2_margin_exceptions.csv", P9D_COLS, None)
    print(f"  {int(D.floor_tie.sum())} ties at the 1e-6 floor; the others: "
          + ", ".join(f"{u} ({m}, cert. gap / best gap {r:.3f})" for u, m, r in
                      D[~D.floor_tie][["unit", "best_method", "ratio_cert_to_best"]].itertuples(index=False)))
    nf = D[~D.floor_tie]
    if len(nf):
        print(f"  non-ties: covariance share of bound (25) {nf.share_cov.min():.2f}-{nf.share_cov.max():.2f}, gap / bound "
              f"{nf.gap_over_bound.min():.3f}-{nf.gap_over_bound.max():.3f} (all units: median {X.gap_over_bound.median():.3f})")
    rs = D[D.resolve_check.notna()]
    if len(rs):
        print(f"  repeated re-solves: {sum(x['reproduced_value_safe'] for x in rs.resolve_check)} / {len(rs)} reproduce "
              "the stored value_safe")
    if "d_bound_recomputed" in D:
        print(f"  bound (25) recomputed from the stored intervals: largest difference {D.d_bound_recomputed.max():.1e}")
    return D


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="summarize the --quick records")
    ap.add_argument("--out", type=Path, default=None, help="default results_dir()/review (with --quick .../review_quick)")
    a = ap.parse_args(argv)
    out, _, _ = C.dirs(a.out, None, a.quick)
    done = [n for n, fn in (("curvature certificate", taylor), ("CROWN with the intervals", crown_intervals),
                            ("terms of bound (25)", bound_terms), ("logit-margin exceptions", margin_exceptions))
            if fn(out) is not None]
    print(f"\nsummaries ({', '.join(done) or 'none: no records'}) in {out / 'summary'}")


if __name__ == "__main__":
    main()
