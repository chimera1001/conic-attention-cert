"""The gap of the certificate against the value-weighted secant slack (Figure 3 and the paragraph after it).

Reads the unit records of ``experiments/gpt2_ladder.py`` (every quantity is computed there, in
:func:`aimp.pipeline.collapse_fields`) and writes

* ``collapse_points.csv`` — per unit: the weighted secant slack ``S``, the value-weighted slack ``Pi_v``, the dispersion
  ``theta`` of the nominal pattern, the nominal spread and the gap (the data of Figure 3; columns as
  ``figdata/P3c_collapse_points.csv``);
* ``bound_terms.csv`` — per unit: the second-order bound (25) and its knapsack, eta and covariance terms (columns as
  ``figdata/P4a_bound_terms.csv``),

then prints the rank correlations and ratios quoted in Section 7 (the same code as ``figscripts/stats.py``).

Usage::

    python experiments/collapse.py                 # after experiments/gpt2_ladder.py
    python experiments/collapse.py --quick         # on the records of gpt2_ladder.py --quick
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FLOOR_REL = 1e-7


def build(units_dir: Path):
    import pandas as pd

    from aimp.gpt2_units import unit_name

    rows = []
    for f in sorted(units_dir.glob("L*/*.json")):
        r = json.loads(f.read_text())
        gap = r["cert"] - r["lb_strong"]
        rows.append({"source": "ladder" if r["pos"] == 4 and r["rho_rel"] < 0.01 else "band",
                     "unit": unit_name(r["layer"], r["head"], r["prompt"], r["pos"], r["rho_rel"]),
                     "layer": r["layer"], "head": r["head"], "prompt": r["prompt"], "pos": r["pos"],
                     "rho_rel": r["rho_rel"], "n": r["n"], "range": r["range"], "S": r["S"], "w_t": r["w_t"],
                     "theta": r["theta"], "gap": gap, "floor": bool(gap <= FLOOR_REL * r["range"]),
                     "Pi_v": r["Pi_v"], "bound_F0v": r["bound_F0v"], "bound_F0v_final": r["bound_F0v_final"],
                     "F0vf_term_knapsack": r["F0vf_term_knapsack"], "F0vf_term_eta": r["F0vf_term_eta"],
                     "F0vf_term_cov": r["F0vf_term_cov"], "D_tot": r["D_tot"], "budget_binds": r["budget_binds"],
                     "n_pruned": r["n_pruned"]})
    D = pd.DataFrame(rows)
    if D.empty:
        return D, D
    D["gap_over_Pi_v"] = D.gap / D.Pi_v
    D["gap_over_F0v"] = D.gap / D.bound_F0v
    pts = D[["source", "unit", "rho_rel", "pos", "S", "w_t", "theta", "range", "gap", "Pi_v", "bound_F0v",
             "gap_over_Pi_v", "gap_over_F0v"]]
    terms = D[["source", "unit", "layer", "head", "prompt", "pos", "rho_rel", "n", "range", "S", "w_t", "theta", "gap",
               "floor", "bound_F0v_final", "F0vf_term_knapsack", "F0vf_term_eta", "F0vf_term_cov", "D_tot",
               "budget_binds", "n_pruned", "Pi_v"]]
    return pts, terms


def report(terms) -> None:
    """The collapse statistics of Section 7 (as in ``figscripts/stats.py``)."""
    from scipy.stats import spearmanr

    B = terms[~terms.floor]
    lo = B.S < 1

    def sp(col):
        return " / ".join(f"{spearmanr(B.gap[m], B[col][m]).statistic:.2f}" if m.sum() > 2 else "n/a"
                          for m in (B.S >= -1, lo, ~lo))

    print(f"units above the numerical floor: {len(B)} (S<1: {int(lo.sum())}, S>=1: {int((~lo).sum())})")
    print("Spearman(gap, S) all:", f"{spearmanr(B.gap, B.S).statistic:.2f}")
    print("Spearman(gap, value-weighted slack of Figure 3) all / S<1 / S>=1:", sp("Pi_v"))
    print("Spearman(gap, knapsack term of bound (25)) all / S<1 / S>=1:", sp("F0vf_term_knapsack"))
    r = B.gap / B.bound_F0v_final
    print(f"gap / bound (25): median {r.median():.3f}, max {r.max():.3f}")
    print(f"share of units below the diagonal of Figure 3: {(B.gap < B.Pi_v).mean():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="use the records of gpt2_ladder.py --quick")
    ap.add_argument("--ladder", default=None, help="results directory of gpt2_ladder.py")
    ap.add_argument("--out", default=None, help="output directory (default: $AIMP_RESULTS_DIR/collapse)")
    args = ap.parse_args()
    from aimp.io import results_dir

    lad = Path(args.ladder) if args.ladder else results_dir() / ("gpt2_ladder_quick" if args.quick else "gpt2_ladder")
    out = Path(args.out) if args.out else results_dir() / ("collapse_quick" if args.quick else "collapse")
    pts, terms = build(lad / "units")
    if pts.empty:
        raise SystemExit(f"no unit records under {lad / 'units'}; run experiments/gpt2_ladder.py first")
    out.mkdir(parents=True, exist_ok=True)
    pts.to_csv(out / "collapse_points.csv", index=False)
    terms.to_csv(out / "bound_terms.csv", index=False)
    print(f"wrote {out / 'collapse_points.csv'} and {out / 'bound_terms.csv'} ({len(pts)} units)")
    report(terms)


if __name__ == "__main__":
    main()
