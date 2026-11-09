"""Print the statistics quoted in Section 7 that are computed from the shipped figure data.
Run from figscripts/:  python3 stats.py   (reads ../figdata)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
D = "../figdata/"
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

print("Spearman(gap, knapsack term of bound (39)) all / S<1 / S>=1:", sp("F0vf_term_knapsack"))
r = B.gap / B.bound_F0v_final
print(f"gap / bound (39): median {r.median():.3f}, max {r.max():.3f}")
print(f"share of units below the diagonal of Figure 3 (gap < knapsack term): {(B.gap < B.F0vf_term_knapsack).mean():.3f}")
T = B[["F0vf_term_knapsack", "F0vf_term_eta", "F0vf_term_cov"]]
for c in T.columns:
    print(f"median share of {c} in bound (39): {(B[c] / B.bound_F0v_final).median():.3f}")
print(f"knapsack term is the largest of the three: {(T.idxmax(axis=1) == 'F0vf_term_knapsack').mean():.3f}")
print(f"gap below knapsack + eta terms: {(B.gap < B.F0vf_term_knapsack + B.F0vf_term_eta).mean():.3f}")

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

# Phase 9 (review): curvature certificate, score-interval ablation, terms of (39) on gpt-oss-20b
T = pd.read_csv(D + "P9a_taylor_fixed_query.csv")
T["ratio"] = (T.r_T - T.p_star) / (T.cert - T.p_star)
print(f"curvature certificate valid on every unit: {(T.r_T >= T.p_star_hi).all()}")
for s, G in T.groupby("set"):
    r = G.groupby("rho_rel").ratio.median()
    print(f"{s}: median ratio to certificate {r.min():.0f} to {r.max():.0f}, smallest {G.ratio.min():.1f}, "
          f"slope (two smallest radii) {G.drop_duplicates(['layer','head','prompt']).slope_T_small.median():.2f}")
G = T[T.set == "gpt2_a"]
for rho in (0.002, 0.02):
    g = G[G.rho_rel == rho]
    print(f"GPT-2 output, rho {rho}: above first-order certificate {(g.r_T > g.r_greedy).mean():.3f}, "
          f"gap above range of motion {((g.r_T - g.p_star) > g.spread).mean():.3f}")
C = pd.read_csv(D + "P9b_crown_sdp_intervals.csv")
r = C.groupby("rho_rel")[["orig_CROWN_ratio", "ref_CROWN_ratio", "orig_alpha-CROWN_ratio", "ref_alpha-CROWN_ratio"]].median()
print("median ratios to the certificate, without / with the semidefinite score intervals:\n", r.round(2))
print(f"smallest ratio with the intervals (alpha-CROWN): {C['ref_alpha-CROWN_ratio'].min():.2f}")
O = pd.read_csv(D + "P9c_gptoss_bound_terms.csv")
for src, G in O.groupby("source"):
    T3 = G[["F0vf_term_knapsack", "F0vf_term_eta", "F0vf_term_cov"]]
    print(f"gpt-oss {src}: knapsack largest {(T3.idxmax(axis=1) == 'F0vf_term_knapsack').mean():.3f}, "
          f"gap < knapsack + eta {(G.gap < G.F0vf_term_knapsack + G.F0vf_term_eta).mean():.3f}, "
          f"Spearman(gap, KN) {spearmanr(G.gap, G.F0vf_term_knapsack).statistic:.2f}, "
          f"Spearman(gap, S) {spearmanr(G.gap, G.S).statistic:.2f}, "
          f"gap/bound median {(G.gap / G.bound_F0v_final).median():.2f} max {(G.gap / G.bound_F0v_final).max():.2f}")
