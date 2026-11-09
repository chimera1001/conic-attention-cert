# conic-attention-cert

Code and benchmark for the paper **Conic Certification of Transformer Attention** (Masoud Ataei, University of Toronto,
2026; [arXiv:2611.00585](https://arxiv.org/abs/2611.00585)).

Repository: https://github.com/chimera1001/conic-attention-cert

> **Branches.** This branch, `jmlr`, follows the journal version of the paper submitted to the Journal of Machine
> Learning Research: sections, tables and figures below are numbered as there. It adds a second trained model
> (gpt-oss-20b), the Vertex-Softmax baseline, the logit-margin specification, the computations added for the review and
> the benchmark of gpt-oss-20b. The branch `main` matches the arXiv version (arXiv:2611.00585v1).

The paper certifies the output of one attention head when the tokens it reads move in given sets (balls, ellipsoids or
boxes). This repository contains

* the certificates of the paper: interval controls (score, attention and log-attention intervals, exact secant gaps,
  knapsack), the first-order pattern-polytope certificate (a greedy linear program), the level-one exponential-cone
  relaxation of Definition 3 in MOSEK's task interface with the safe dual bound, branching on the score intervals, and
  the exact worst case with a fixed query of Theorem 7;
* the GPT-2 small pipeline: weight extraction with a reproduction check, reduction of each head to the subspace it reads,
  Shor score intervals and multistart ascent;
* gpt-oss-20b (Section 7.3): the instance builder from the released weights (rotary position embeddings with YaRN
  scaling, grouped keys and values, attention biases, RMSNorm, one sink per head). The rotary maps make the score matrix
  key-dependent, `M_j = R_i^T R_j / 8`; they are folded into each token's set by an invertible linear map per token, and
  the sink is a fixed key with a constant score and value zero, so every method of the paper applies unchanged
  (`aimp/gptoss.py`, the `model="gpt-oss-20b"` option of `aimp.gpt2_units.make_instance`);
* the baselines and diagnostics of Section 7: an auto_LiRPA harness (IBP, CROWN, alpha-CROWN) on the same heads, the
  Vertex-Softmax bound, the curvature (Taylor) certificate, and the Hessian and per-operation tools of the mechanism test;
* one script per experiment of Section 7, the figure data and scripts of the paper, and a benchmark of the exact
  fixed-query worst cases of every head of GPT-2 small (17,280 units) and of gpt-oss-20b (61,440 units) (`benchmark/`).

## Layout

```
aimp/                 the package
  sets.py, heads.py, types.py, instances.py     token sets, the attention head, synthetic families
  bounds.py, theory.py                          controls, secant gaps, knapsack, vertex bound, the gap bounds of the paper
  relax_ps.py, ibp.py                           first-order certificate, interval pattern bounds
  sdp_mosek.py, relax_sdp.py, safe_bound.py     level-one relaxation (MOSEK task API; CVXPY for synthetic heads), safe bound
  certify.py, option_b.py                       the certificate, the re-solve rules of Appendix B, branch-and-bound
  exact_fq.py, exact_gurobi.py                  exact fixed-query worst case; exact references (Gurobi) for synthetic heads
  lower_bounds.py, lower.py                     multistart ascent, alternating ascent
  gpt2.py, gpt2_units.py, pipeline.py           GPT-2 extraction, units, the per-unit procedures of Section 7
  gptoss.py                                     gpt-oss-20b: extraction with its checks, the units (rotary maps, sink)
  lirpa.py                                      auto_LiRPA baselines
  block.py, mechanism.py                        GPT-2 block, remainder network, Hessian and per-operation tools
experiments/          one script per experiment (table below; details in scripts/reproduce.md)
benchmark/            exact fixed-query worst cases of GPT-2 small and gpt-oss-20b (SCHEMA.md, verify.py)
figdata/, figscripts/ figure data of the paper (P1-P9); make_figs.py draws the six figures, stats.py prints the statistics
tests/                unit tests and end-to-end smoke tests
data/prompts.txt      the five prompts
scripts/reproduce.md  table, figure and number -> command and output file
```

## What each experiment reproduces

Every script runs the paper's full settings by default, is resumable (finished units are skipped), takes `--workers`
and `--threads-per-solve`, and has a `--quick` option that runs in minutes. `scripts/reproduce.md` lists, for every
table, figure and number, the command, the output file and the run time measured for the paper.

| script | paper (journal version) |
|---|---|
| `experiments/synthetic.py` | Section 7.1: Figure 1, Table 4 (slopes of the gaps in the radius), the width and temperature experiments |
| `experiments/gpt2_ladder.py` | Section 7.2: Table 5 (accuracy on GPT-2 small), Figure 2 (radius ladder, per-unit slopes), tightness radii, branching |
| `experiments/collapse.py` | Section 7.2: Figure 3 (gap against the weighted secant slack and the knapsack term), the shares of the terms of bound (25) |
| `experiments/exact_fixed_query.py` | Section 7.2: Table 6 (exact worst cases with a fixed query) and the GPT-2 benchmark |
| `experiments/bound_propagation.py` | Section 7.2: Table 7 and Figure 4, top (IBP, CROWN, alpha-CROWN on the same units), the alpha-CROWN settings |
| `experiments/jmlr_extensions.py` | Section 7.2: the vertex-bound columns of Table 7 and Figure 4; the logit-margin specification, Table 8 and Figure 4, bottom; the vertex bound on synthetic heads (Section 7.1) |
| `experiments/mechanism.py` | Section 7.2: Figure 5 (looseness of CROWN on block remainders against the per-operation prediction) |
| `experiments/gpt_oss.py` | Section 7.3: Table 9 (accuracy on gpt-oss-20b), Table 10 (bound propagation and the vertex bound, both specifications), Figure 6 (the two models at matched secant width), the radius calibration, the gpt-oss-20b benchmark |
| `experiments/review_taylor.py` | Sections 7.2 and 7.3: the curvature certificate (Table 3, "the constant, not the order"; `figdata/P9a`) |
| `experiments/review_crown_intervals.py` | Section 7.2: CROWN and alpha-CROWN with the certificate's semidefinite score intervals (`figdata/P9b`) |
| `experiments/review_bound_terms.py` | Section 7.3: the three terms of bound (25) on gpt-oss-20b (`figdata/P9c`) |
| `experiments/review_margin_exceptions.py` | Section 7.2: the 13 logit-margin units where bound propagation is not looser (`figdata/P9d`) |
| `experiments/review_summary.py` | the tables of the four `review_*` scripts and the files `P9a`-`P9d` |
| `benchmark/verify.py` | checks the benchmark of either model against the Hugging Face weights (Code and Data Availability) |
| `figscripts/make_figs.py`, `figscripts/stats.py` | Figures 1-6 and the statistics computed from `figdata/` |

Table 3 of the paper summarizes the tests; Appendix B describes the implementation (code locations in
`scripts/reproduce.md`).

## Installation

Python 3.11 or later (the experiments ran on Python 3.14.7). CPU only; no GPU is used (`CUDA_VISIBLE_DEVICES=""`).

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps --ignore-requires-python "auto_LiRPA @ https://github.com/Verified-Intelligence/auto_LiRPA/archive/e3930353a3589f7d10879f1e71223d1eefb57196.zip"
```

auto_LiRPA 0.8.0 (the commit above) is needed only for the bound propagation baselines (`bound_propagation.py`,
`jmlr_extensions.py`, `mechanism.py`, `gpt_oss.py`, `review_crown_intervals.py`). It declares older Python and torch
versions than those used here but works with them; `--no-deps` keeps it from changing installed packages.

**Versions used for the paper.** Python 3.14.7, MOSEK 11.2.4, Gurobi 13.0.3, auto_LiRPA 0.8.0, CVXPY 1.9.3, PyTorch
2.14.0 (CPU), transformers 5.17.0, numpy 2.4.6, scipy 1.18.1 (all pinned in `requirements.txt`).

**Solver licences.** The semidefinite relaxations need MOSEK (version 11) with a licence; MOSEK licences are free for
academic use (https://www.mosek.com/products/academic-licenses/). Gurobi (version 13) is used only for the exact
reference values on synthetic heads in `experiments/synthetic.py` (free academic licences exist as well); nothing else
needs it. No licence file is part of this repository.

**Model weights.** No weights are distributed here. GPT-2 small (`gpt2`, about 500 MB) is downloaded from the Hugging
Face hub on first use and cached by `transformers`. For gpt-oss-20b see the next section.

## gpt-oss-20b

The weights are those released by OpenAI on the Hugging Face hub, `openai/gpt-oss-20b` (the paper used revision
`6cee5e81ee83917806bbde320786a8fb61efebee`, Apache 2.0 licence). Only the Hugging Face files are needed, about 13 GB; the
`original/` and `metal/` folders are not:

```bash
hf download openai/gpt-oss-20b --exclude "original/*" --exclude "metal/*"   # or let the first run download it
```

The first run of `experiments/gpt_oss.py` loads the model once on CPU (the MXFP4 experts dequantized, forward passes in
float32; about 85 GB of memory and about one minute after the download on the workstation below), runs the five prompts,
checks the extraction (a float64 re-implementation of every attention sublayer reproduces the model to `1e-4` relative;
the logits and the logit-margin vector at position four) and caches every array the units read, in float64, under
`$AIMP_CACHE_DIR/gpt-oss-20b` (5.1 GB). No later step loads the model; a unit then needs 1-2 GB of memory. An existing
cache is never overwritten (see `scripts/reproduce.md`).

```bash
python experiments/gpt_oss.py --quick --workers 2 --threads-per-solve 2     # extraction, then two heads (minutes)
python experiments/gpt_oss.py --part calib --workers 12                     # radius calibration
python experiments/gpt_oss.py --part x --workers 12                         # 61,440 exact fixed-query units
python experiments/gpt_oss.py --part f v l --workers 6 --threads-per-solve 4
python experiments/gpt_oss.py --summarize                                   # Tables 9 and 10, Figure 6 data
python experiments/gpt_oss.py --benchmark                                   # benchmark/gpt-oss-20b from part x
```

Run times on the workstation below: part `x` about 19 worker hours (10 hours at 2 workers), parts `f`, `v`, `l` about
24, 42 and 41 worker hours at 4 threads per solve (with 6 workers x 4 threads about 5, 6 and 6 hours).

## The benchmark

`benchmark/` holds the exact worst cases with a fixed query (Theorem 7) of every head of both models: the certified
brackets of the maximum and of the minimum of the head's output along its nominal direction when every key token moves in
an l2 ball of radius `rho_rel * ||z_j||` around the normalized token the head reads, together with the maximizing and
minimizing angle of every key, so that the worst-case inputs can be rebuilt. `benchmark/SCHEMA.md` states the threat
model and the file formats.

| model | units | factors | files |
|---|---|---|---|
| GPT-2 small | 17,280 | 144 heads x 5 prompts x query positions 4, 8, 12, last x `rho_rel` 0.001-0.05 (6) | `units.csv`, `angles.csv.gz`, `prompts.txt` |
| gpt-oss-20b | 61,440 | 1,536 query heads x 5 prompts x query positions 4, last x `rho_rel` 0.0002-0.002 (4) | `gpt-oss-20b/units.csv`, `gpt-oss-20b/angles.csv.gz`, `gpt-oss-20b/prompts.txt` |

All units are exact (brackets narrower than `1e-10` of the nominal value spread). `benchmark/verify.py` rebuilds units
from the Hugging Face weights and checks them with two independent implementations (a numpy head and the model's own
code); on the machine of the paper every unit of both models passes:

```bash
python benchmark/verify.py --sample 200 --seed 0                       # GPT-2 small (about a minute)
python benchmark/verify.py --model gpt-oss-20b --sample 200 --seed 0   # gpt-oss-20b (about 85 GB of memory while loading)
python benchmark/verify.py --model gpt-oss-20b --all                   # all 61,440 units (about 25 minutes)
```

## Configuration

Everything is set through environment variables with defaults; no path or machine setting is hard-coded.

| variable | default | meaning |
|---|---|---|
| `AIMP_MAX_THREADS` | 4 | total CPU threads of every solver, BLAS and torch call; worker pools use `workers x threads <= AIMP_MAX_THREADS` |
| `AIMP_RESULTS_DIR` | `./results` | where the experiments write |
| `AIMP_CACHE_DIR` | `./cache` | cached GPT-2 weights and prompt activations, and `gpt-oss-20b/` |
| `AIMP_GUROBI_MEMLIMIT_GB` | 20 | Gurobi memory limit of the exact references |
| `HF_HOME` | the `transformers` default | Hugging Face cache |

## Quick start

```bash
python -m pytest -q tests/
python experiments/exact_fixed_query.py --quick
python experiments/gpt2_ladder.py --quick
python experiments/collapse.py --quick
python benchmark/verify.py --sample 200 --seed 0
python figscripts/stats.py
python figscripts/make_figs.py
```

The review computations read the records of the other scripts; after their `--quick` runs (the gpt-oss-20b sets are
skipped when `experiments/gpt_oss.py --quick` has not run):

```bash
python experiments/bound_propagation.py --quick
python experiments/jmlr_extensions.py --quick --part vs-gpt2 lm
python experiments/review_taylor.py --quick
python experiments/review_crown_intervals.py --quick
python experiments/review_summary.py --quick
```

## Hardware used for the paper

One workstation: Intel Xeon w5-2465X processor (16 cores, 3.1 GHz), 256 GB of memory, Debian GNU/Linux 13, no
graphics processor. The GPT-2 experiments used at most 14 threads, the gpt-oss-20b experiments at most 28. The largest
semidefinite programs (GPT-2 heads at query position 12) take about five minutes and 4 GB each at 4 threads.

## Citation

```bibtex
@misc{ataei2026conic,
  title         = {Conic Certification of Transformer Attention},
  author        = {Ataei, Masoud},
  year          = {2026},
  eprint        = {2611.00585},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2611.00585},
  url           = {https://arxiv.org/abs/2611.00585}
}
```

The journal version extends the arXiv version (this branch follows it); please cite the arXiv entry until the journal
reference is available.

## Licence

MIT (see `LICENSE`). MOSEK and Gurobi are commercial solvers with their own licences and are not distributed here. The
model weights are not distributed here either; GPT-2 small and gpt-oss-20b are subject to their own licences.
