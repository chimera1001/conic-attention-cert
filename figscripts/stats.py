"""Print the statistics quoted in Section 7 that are computed from the shipped figure data.
Run:  python stats.py   (reads ../figdata relative to this file)."""
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figdata") + os.sep
def iqr(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return f"{np.median(x):.3f} [{np.quantile(x,.25):.3f}, {np.quantile(x,.75):.3f}] (n={len(x)})"

# Ladder (Figure 2): per-unit slopes on the three smallest radii, position four
L = pd.read_csv(D + "P3b_ladder.csv")
print("ladder slope, certificate:", iqr(L.slope_cert_small))
print("ladder slope, first order:", iqr(L.slope_greedy_small))
print("ladder slope, range of motion:", iqr(L.slope_spread_small))

# Collapse (Figure 3 and the paragraph after it): units above the numerical floor
B = pd.read_csv(D + "P4a_bound_terms.csv"); B = B[~B.floor]
lo = B.S < 1
def sp(col):
    return " / ".join(f"{spearmanr(B.gap[m], B[col][m]).statistic:.2f}" for m in (B.S >= -1, lo, ~lo))
print(f"units: {len(B)} (S<1: {lo.sum()}, S>=1: {(~lo).sum()})")
print("Spearman(gap, S) all:", f"{spearmanr(B.gap, B.S).statistic:.2f}")
print("Spearman(gap, value-weighted slack of Figure 3) all / S<1 / S>=1:", sp("Pi_v"))
print("Spearman(gap, knapsack term of bound (39)) all / S<1 / S>=1:", sp("F0vf_term_knapsack"))
r = B.gap / B.bound_F0v_final
print(f"gap / bound (39): median {r.median():.3f}, max {r.max():.3f}")
print(f"share of units below the diagonal of Figure 3: {(B.gap < B.Pi_v).mean():.3f}")

# Mechanism test (Figure 5)
M = pd.read_csv(D + "P5c_mechanism.csv"); S = M[M.rho_rel <= 0.001]
y = M.v_tot / M.att_max; x = M.P_T_p_complex / M.s_spec_p; x1 = M.P_T_p / M.s_spec_p
print(f"Spearman(looseness, predicted) all: {spearmanr(x, y).statistic:.2f}, rho<=0.001: "
      f"{spearmanr(x[M.rho_rel<=0.001], y[M.rho_rel<=0.001]).statistic:.2f} "
      f"(softmax as one operation: {spearmanr(x1[M.rho_rel<=0.001], y[M.rho_rel<=0.001]).statistic:.2f})")
print(f"attained max / s_spec, rho<=0.001: {S.ratio_T1.min():.4f} to {S.ratio_T1.max():.4f}")
q = S.v_tot / S.P_T_p_complex
print(f"CROWN / s_op, rho<=0.001: {q.min():.2f} to {q.max():.2f}, median {q.median():.2f}")
print(f"CROWN / per-operation terms on CROWN's intervals, rho<=0.001: {S.ratio_T2_crownh.min():.2f} to {S.ratio_T2_crownh.max():.2f}")
print(f"one-operation / decomposed s_op: {(M.P_T_p/M.P_T_p_complex).min():.2f} to {(M.P_T_p/M.P_T_p_complex).max():.2f}")
print(f"predicted looseness: median {x.median():.0f} ({x.min():.0f} to {x.max():.0f})")
print(f"effective rank: median {M.eff_rank_p.median():.1f} ({M.eff_rank_p.min():.1f} to {M.eff_rank_p.max():.1f})")
print(f"per-neuron sum / half trace: median {(M.P_T_p_complex/M.half_trace_p).median():.1f}")
