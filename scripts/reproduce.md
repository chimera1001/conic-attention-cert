# Reproducing Section 7 and Appendix B

All commands run from the repository root with the interpreter of your environment (`python` below). Results go to
`$AIMP_RESULTS_DIR` (default `./results`). Every script is resumable (finished units are skipped) and takes
`--workers` and `--threads-per-solve` (`workers x threads <= AIMP_MAX_THREADS`). Run times are those measured for the
paper on the workstation described in `README.md`; `--quick` versions run in minutes.

The figure data of the paper are in `figdata/`; `python figscripts/stats.py` prints the numbers computed from them and
`python figscripts/make_figs.py` redraws the five figures into `figs/`. The experiments write CSV files with the same
columns, so a rerun can replace the files in `figdata/`.

## GPT-2 setup (all GPT-2 experiments)

`aimp.gpt2_units.prepare()` (called by every GPT-2 script) downloads `gpt2`, extracts the activations of the five prompts
of `data/prompts.txt` (19, 21, 19, 21 and 22 tokens) and checks a numpy re-implementation of every block against the
model to `1e-10` (worst deviations 3.6e-15 for the normalized tokens and attention patterns, 9.1e-13 for block outputs).

## Synthetic heads (`experiments/synthetic.py`)

| paper item | command | output |
|---|---|---|
| Figure 1 and Table 3 (slopes of the gap in rho on rho <= 0.1: interval pattern bounds, first order, level one with the unit diagonal, level one), "boxes give the same picture" | `python experiments/synthetic.py --part slopes --t-interval loose` then `--part summary` | `results/synthetic/slopes/<family>/<instance>/*.json`; `results/synthetic/summary/table3_slopes.csv`, `results/synthetic/summary/P1_gap_vs_rho.csv` (the format of `figdata/P1_gap_vs_rho.csv`) |
| interval pattern bounds 9.5 to 28 times looser than the level-one certificate, depending on the family and the number of keys | `python experiments/synthetic.py --part summary` | `results/synthetic/summary/interval_pattern_ratio.csv` |
| the minimum of the two certificates (Definition 4) costs a median 0.2 s per synthetic instance | `python experiments/synthetic.py --part summary` | `results/synthetic/summary/min_of_two.csv` (column `cost_s`) |
| level one 125 / 6 / 1.6 times tighter than first order at w_t = 0.05 / 1 / 8; half of the seeds have a dominant key at every width, and its certified weight exceeds 0.9 on all five from w_t = 12 | `python experiments/synthetic.py --part width` then `--part summary` | `results/synthetic/width/...`, `results/synthetic/summary/width.csv` (columns `margin_share`, `dominant_key_share`) |
| order 1.97 to 2.03 of the level-one gap in the temperature on random heads | `python experiments/synthetic.py --part temperature` then `--part summary` | `results/synthetic/temperature/...`, `results/synthetic/summary/temperature_orders.csv` |

The default grid of `--part slopes` is exactly the 8,100 instances of the paper. It covers five families: random heads
at temperatures 0.1–10, a dominant key of margin 1–8, ambiguous heads, rank one, and the two-key family of the type of
the sharpness instances of Proposition 10. Each has ball and box sets, n = 2 (4,620 instances), n = 3 (3,160) and n = 4
(320). The regular grid at n = 3 would have 40 more instances (F_rank1, balls, rho = 0.2, temperatures 0.3, 1, 3 and 10,
seeds 5–9, both query modes). They are not among the paper's instances, so `synthetic.py` leaves them out
(`NOT_IN_PAPER`). The reference of every instance is a time-limited global solve with Gurobi (300 s for rho <= 0.2, 90 s
above). For 40 of the 8,100 instances (the same F_rank1 configurations, seeds 0–4) the stored records of the paper have
no exact solve, and their reference there was the best feasible value; the release runs the exact solve on all of them.

As the paper states, Figure 1 and Table 3 used the wider log-attention intervals `[s_lo - tau_hi, s_hi - tau_lo]` and the
solver's primal value, which is within 4e-8 of the safe value. `--t-interval loose` reproduces them: stored instances
rerun with it agree to 3e-8, and the summary reproduces Table 3 and `figdata/P1_gap_vs_rho.csv` exactly from those
records. `--t-interval tight` (the default) is the relaxation of Definition 3 with the intervals of equation (14). It
reports the certified dual value or the safe dual bound and gives the same slopes (quick run: 2.03 / 2.01 for the level
one, about 1 for the others). With it the level-one gaps are smaller (a median 0.35 times those with the wider
intervals on the stored instances), so the ratio of the interval pattern bounds to the level-one certificate is larger
than the paper's 9.5 to 28.

Run time: about 13 hours on 3 workers with `--t-interval loose` (39 CPU hours). The default tight intervals make the
early-stopped exact references slower (estimated 60–100 CPU hours). `--quick` (n = 2, balls, random and ambiguous
heads, 3 seeds, four radii, short exact time limit, small width and temperature grids) takes about 10 minutes on 3
workers. The summary also prints, as a diagnostic that is not in the paper, the share of instances on which the minimum
of the two certificates is within 1.2 of the best of the methods run (which includes strengthened relaxations).

## GPT-2: accuracy table, radius ladder, branching (`experiments/gpt2_ladder.py`)

| paper item | command | output |
|---|---|---|
| Table 4; w_t ~ 270 rho_rel; 94% of units with 1 <= w_t <= 8 at 0.01; 2 units with a certified dominant key; certificate strictly tighter than first order on 99.5%, median factor 2 | `python experiments/gpt2_ladder.py --part units` then `--part summary` | `results/gpt2_ladder/units/L*/*.json`; table printed |
| Figure 2 (left and right); ladder slopes 2.04 / 1.06 / 1.00 over 716 / 720 / 720 units; consecutive-radius slopes (1.83 at 0.01–0.02) | `python experiments/gpt2_ladder.py --part summary` | `results/gpt2_ladder/ladder.csv` (columns of `figdata/P3b_ladder.csv`) |
| tightness-radius exponents 0.52 / 0.87 | `python experiments/gpt2_ladder.py --part radius` (needs `units`) | `results/gpt2_ladder/radius/*.json`; printed by `--part summary` |
| stronger ascents: <= 1.4e-9 of the spread | `python experiments/gpt2_ladder.py --part ascent` (needs `units`) | `results/gpt2_ladder/ascent/*.json`; printed by `--part summary` |
| branching: 0.087 -> 0.020 (144 heads, 0.01); 0.037 -> 0.0049 (24 hardest, 0.002); lookahead 0.013 vs 0.029 (72 heads) | `python experiments/gpt2_ladder.py --part bracket` (needs `units`) | `results/gpt2_ladder/bracket_{S,lookahead}/*.json`; printed by `--part summary` |

Units: 6,480 semidefinite certificates (position 4 at five radii, positions 8 and 12 at 0.01 and 0.02). Run time for the
paper: about 77 hours at 3 workers x 4 threads (median 2 minutes per unit; up to about 5 minutes at position 12);
branching about 14 hours (S rule, 168 brackets of 24 solves) and 3.6 hours (lookahead, 72 brackets) on the same lanes;
tightness radii about 5 hours; alternating ascent about 30 minutes on 2 threads. `--quick`: 6 units (two heads, prompt 0, position 4, radii 0.001–0.005), about 5 minutes.

## GPT-2: collapse onto the value-weighted slack (`experiments/collapse.py`)

| paper item | command | output |
|---|---|---|
| Figure 3; rank correlations 0.76 (S), 0.91 (value-weighted slack), 0.93 / 0.99 / 0.74 (knapsack term of (39)); gap 21% (median) and 56% (max) of bound (39); 97% of units below the diagonal | `python experiments/collapse.py` (after `gpt2_ladder.py --part units`) | `results/collapse/collapse_points.csv` (columns of `figdata/P3c_collapse_points.csv`), `results/collapse/bound_terms.csv` (columns of `figdata/P4a_bound_terms.csv`) |

Seconds. `--quick` reads the records of `gpt2_ladder.py --quick`.

## GPT-2: exact worst cases with a fixed query (`experiments/exact_fixed_query.py`)

| paper item | command | output |
|---|---|---|
| 17,280 units solved exactly, with certified brackets narrower than 1e-10 of the nominal spread of the values (a median 6e-10 of the range of motion) | `python experiments/exact_fixed_query.py --part exact` | `results/exact_fixed_query/exact/L*/*_fq.json` |
| the benchmark | `python experiments/exact_fixed_query.py --part benchmark` | `benchmark/units.csv`, `benchmark/angles.csv.gz` |
| Table 5 (2,880 units at position 4); slopes 2.04 / 1.04 / 1.00 between 0.002 and 0.005; gap <= 0.78 of bound (39); multistart finds the exact worst case to 1e-9 of the nominal spread | `python experiments/exact_fixed_query.py --part sdp` then `--part summary` | `results/exact_fixed_query/sdp/L*/*_fq.json`; table printed |
| secant width with a varying query: a median 2.36 to 2.42 times that with a fixed query, depending on the radius | `python experiments/exact_fixed_query.py --part summary` (needs `gpt2_ladder.py --part units`) | printed |

Run time: `exact` about 25 minutes at 4 workers x 1 thread (median 0.13 s per unit for the exact solve); `sdp` about 3.5
hours at 3 x 4 threads. `--quick`: 2 heads x 1 prompt (48 exact units, 4 certificates), about 3 minutes.

## GPT-2: linear bound propagation (`experiments/bound_propagation.py`)

| paper item | command | output |
|---|---|---|
| Table 6 and Figure 4 (IBP, CROWN, alpha-CROWN, first order and certificate on the same 5,760 units; ratio columns) | `python experiments/bound_propagation.py --part D` (references: `exact_fixed_query.py --part exact sdp` and `gpt2_ladder.py --part units`) | `results/bound_propagation/bounds/{fq,free}/L*/*.json`; `crown_summary.csv` (columns of `figdata/P4b_crown_summary.csv`), `crown_units.csv`, `crown_slopes.csv`, `crown_missing.csv` |
| slopes 2.11 (fixed) / 2.39 (varying) vs 2.05 for the certificate; certificate tighter on 5,760 / 5,760 units, by >= 2x on 96%; 0.01–0.02 s (CROWN), 0.2–0.7 s (alpha-CROWN) | `python experiments/bound_propagation.py --summarize` | printed; `crown_slopes.csv` |
| 20–500 alpha-CROWN iterations at two learning rates move the median ratio by < 0.1% (1,152 units of prompt 0) | `python experiments/bound_propagation.py --part alpha` | `results/bound_propagation/alpha_settings/...`, `alpha_settings.csv`, `alpha_settings_summary.csv` |

Run time: part D about 20 minutes at 3 workers x 1 thread; part alpha about 2.3 hours. `--quick` (the two quick heads of
the other GPT-2 scripts, references from their `--quick` runs): part D about 30 seconds, part alpha about 4 minutes.

## GPT-2: the mechanism test (`experiments/mechanism.py`)

| paper item | command | output |
|---|---|---|
| Figure 5; rank correlation 0.90 (132 units) and 0.98 (96 units with rho_rel <= 0.001, also with the softmax as one operation); attained maximum within 0.04–1.5% of s_spec; CROWN / s_op 1.26–2.34 (median 1.63); 1.11–1.54 against the per-operation terms on CROWN's own intervals; softmax as one operation multiplies s_op by 0.75–1.01; predicted looseness median 140 (33–722); effective rank median 11 (2.4–50); per-neuron sum / trace about 14 | `python experiments/mechanism.py` | `results/mechanism/L*_r*.json` (+ `.npz`), `results/mechanism/mechanism.csv` (columns of `figdata/P5c_mechanism.csv`); the numbers are printed |
| shares of the per-operation sum carried by the activation's operations and the products of attention weights and values, 79–92% | as above | `units.<h>.lse.groups` in the per-configuration JSON (groups `gelu` and `att`) |

The test runs on 11 configurations (blocks 0–1, 3–4, 6–7, 9–10; rho_rel 0.0005, 0.001, 0.002, without blocks 0–1 at 0.002)
x 12 heads of the next block, one perturbed token. For each configuration: the block map and its Jacobian (gated
against the Hugging Face model to `1e-10`), the per-head total rows, CROWN on the remainder network, the attacked
maximum, the Hessian `H_0` with its decomposition over the block's operations (finite-difference checked), `s_spec`,
`s_op` in both readings of the softmax, the chains of Lemma 19, and the per-operation terms on CROWN's intervals.
Run time: 51 minutes at 1 worker x 4 threads (about 5 minutes and 5 GB per configuration). `--quick`: blocks 9–10 at
rho_rel 0.0005, heads 0–2, about 4 minutes.

## Extensions: Vertex-Softmax baseline and logit-margin specification (`experiments/jmlr_extensions.py`)

These two comparisons extend Section 7; they are not in the arXiv version of the paper.

| item | command | output |
|---|---|---|
| Vertex-Softmax bound `r_VS` (the exact maximum of `sum_j softmax(s)_j h_j^max` over the box of certified scores, at the threshold vertices of the remark on scores in a box) on the 5,760 units of Table 6, with the Table 6 columns: median relative gap, median ratio `(r_VS - ref) / (cert - ref)`, per-unit slopes, the share of units with `cert < r_VS`, the part of the first-order gap that it removes | `python experiments/jmlr_extensions.py --part vs-gpt2` (references: `exact_fixed_query.py --part exact sdp`, `gpt2_ladder.py --part units`, `bound_propagation.py --part D`) | `results/jmlr/vs_gpt2/{fq,free}/L*/*.json`; `results/jmlr/summary/crown_vs_summary.csv` (the columns of `figdata/P4b_crown_summary.csv` plus `VS_median, VS_q25, VS_q75, VS_n, VS_ratio_to_cert_median, VS_time_median_s, VS_invalid`), `vs_gpt2_units.csv`, `vs_gpt2_slopes.csv`, `vs_gpt2_shares.csv` |
| `r_VS` on the synthetic instances: a Table 3 row (with two keys `r_VS = r_pp` exactly, since the softmax image of a score box is the whole segment), per family and number of keys the ratio of its gap to the level-one and first-order gaps, its ratio to the level-one gap across the secant widths | `python experiments/jmlr_extensions.py --part vs-synthetic` (after `synthetic.py --part slopes width`) | `results/jmlr/vs_synthetic/...`; `results/jmlr/summary/vs_table3.csv`, `vs_ratios.csv`, `vs_width.csv`, `vs_gap_vs_rho.csv` (the format of `figdata/P1_gap_vs_rho.csv`, `method = vs`) |
| the logit-margin specification: `c = -c_m`, `c_m = P diag(g) (w_t1 - w_t2) / sigma_bar` (the head's direct contribution to the margin between the model's two top next-token logits at position 4, the scale of the final normalization frozen); 144 heads x 5 prompts x rho_rel {0.002, 0.005, 0.01}, fixed query (exact maximum and minimum, certificate) and varying query (certificate, best feasible values), with IBP, CROWN, alpha-CROWN, `r_VS` and the first-order certificate | `python experiments/jmlr_extensions.py --part lm` | `cache/lm_spec.npz`, `cache/lm_gate.json` (the checks of the specification); `results/jmlr/lm/{fq,free}/L*/*.json`; `results/jmlr/summary/lm_summary.csv` (format of `crown_vs_summary.csv`), `lm_units.csv`, `lm_slopes.csv`, `lm_shares.csv`, `lm_decrease.csv` |

`spec="lm"` of `aimp.gpt2_units.make_instance` builds the logit-margin units; `aimp.gpt2_units.prepare_lm` computes `c_m`
per prompt and checks it (the numpy logits at the position reproduce the model's to `1e-10`; the frozen-scale logit
difference computed with the model's own final normalization and head changes by `c_m^T delta` to `1e-6` relative;
`c^T ybar` of every unit equals the head's direct contribution in the forward pass). The certified decrease
`cert - c^T ybar` concerns one head's direct path to the logits with the normalization scale frozen, not the model's
prediction. `aimp.bounds.controls` carries `h_max` and `r_VS`, and the per-unit procedures of `aimp/pipeline.py` take
`spec` and record `r_VS`. Every unit checks `r_VS >= p*` (fixed query) or `r_VS >= lb` (varying query) to
`1e-9 max(1, |ref|)` and `r_VS <= r_greedy + 1e-10 range + 1e-15 max(1, |r_greedy|)`.

Run time (measured on the workstation of `README.md`): `vs-gpt2` about 4 minutes at 10 workers x 1 thread (the Shor
score intervals cost about 0.5 s per varying-query unit, the fixed-query intervals and the vertex enumeration
milliseconds); `vs-synthetic` about 2.5 minutes at 2 workers on the stored records; `lm` about 3 hours at 3 x 4 threads
for the 2,160 fixed-query units (median 11 s per unit; the re-solve rule fires on 14% of them, mostly at rho_rel = 0.002)
and about 5 hours for the 2,160 varying-query units (median 25 s; re-solves on 4%); IBP, CROWN and alpha-CROWN add about
1 s per unit (alpha-CROWN 0.1–0.8 s).
`--quick` (two heads, prompt 0; references from the `--quick` runs of the other scripts) takes a few minutes; `--heads`
and `--prompts` select other units.

## Benchmark check

```bash
python benchmark/verify.py --sample 200 --seed 0
python benchmark/verify.py --rho 0.01
```

## Appendix B

Appendix B describes the implementation used by all of the above; the code locations are:

| item | code |
|---|---|
| level-one relaxation (R1)–(R7) for balls, ellipsoids and boxes | `aimp/sdp_mosek.py::solve_sdp1` (MOSEK task interface; GPT-2), `aimp/relax_sdp.py::solve_sdp` (CVXPY; synthetic heads, incl. the unit-diagonal and box-product variants) |
| pattern cut with the width pair, or the margin pair when it gives the smaller radius | `aimp/bounds.py::choose_cut(cut="auto")` (the experiments call every certificate with `cut="auto"`) |
| pruning of keys with `a_hi <= 1e-6` and its slack | `prune_tol=1e-6` in `solve_sdp1` / `solve_sdp`; `pruning_slack` in `aimp/certify.py::certify` |
| safe dual bound (projection onto the dual cones, both sign conventions, eigenvalue or entrywise term, `64 u` rounding allowance) | `aimp/sdp_mosek.py::ConicModel.safe_value`, `aimp/safe_bound.py::safe_dual_bound` |
| Shor score intervals (side 129 on GPT-2; S-lemma for the query's own score), widened by `1e-7 max(1, |v|)` | `aimp/gpt2.py::score_intervals_roles`, `aimp/sdp_mosek.py::shor_score`, `aimp/bounds.py::SHOR_PAD_REL` |
| attention, normalizer and log-attention intervals | `aimp/bounds.py::softmax_interval_bounds`, `log_softmax_bounds`, `t_intervals` |
| exact secant gaps with `expm1`, `log1p` and a short series | `aimp/theory.py::secant_gap` |
| branch-and-bound (best first, 24 solves, stop at `1e-3` of the range of motion) | `aimp/certify.py::branch_and_bound` |
| multistart starts, L-BFGS-B with projection, <= 2000 iterations | `aimp/lower.py::strong_bracket`, `sdp_starts`, `boundary_points` |
| GPT-2 tolerances `1e-6 / 1e-6 / 1e-8` and the re-solve at `1e-9` | `aimp/sdp_mosek.py::PHASE3_TOL`, `aimp/certify.py::needs_retol`, `sdp_retol` |
| exact references on synthetic heads (log form, Gurobi spatial branch-and-bound, relative gap `1e-6` or a time limit) | `aimp/exact_gurobi.py::solve_exact`, `aimp/solvers.py::GUROBI_DEFAULTS` |
