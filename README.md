# conic-attention-cert

Code and benchmark for the paper **Conic Certification of Transformer Attention** (Masoud Ataei, University of Toronto,
2026; [arXiv:2611.00585](https://arxiv.org/abs/2611.00585)).

Repository: https://github.com/chimera1001/conic-attention-cert

The paper certifies the output of one attention head when the tokens it reads move in given sets (balls, ellipsoids or
boxes). This repository contains

* the certificates of the paper: interval controls (score, attention and log-attention intervals, exact secant gaps,
  knapsack), the first-order pattern-polytope certificate (a greedy linear program), the level-one exponential-cone
  relaxation of Definition 3 in MOSEK's task interface with the safe dual bound of Proposition 16, branching on the score
  intervals, and the exact worst case with a fixed query of Theorem 7;
* the GPT-2 small pipeline: weight extraction with a reproduction check, reduction of each head to the subspace it reads,
  Shor score intervals and multistart ascent;
* the baselines and diagnostics of Section 7: an auto_LiRPA harness (IBP, CROWN, alpha-CROWN) on the same heads, and
  the Hessian and per-operation tools of the mechanism test;
* one script per experiment of Section 7, the figure data and scripts of the paper, and a benchmark of 17,280 exact
  worst cases of GPT-2 small heads (`benchmark/`);
* two extensions of Section 7 that are not in the arXiv version: the Vertex-Softmax baseline (the exact maximum of the
  softmax-weighted support values over the box of certified scores) and a logit-margin specification, in
  `experiments/jmlr_extensions.py` (see `scripts/reproduce.md`);
* a second trained model, gpt-oss-20b (rotary position embeddings, grouped keys and values, attention sinks), with
  `model="gpt-oss-20b"` of the unit construction and `experiments/gpt_oss.py` (see `scripts/reproduce.md`).

## Layout

```
aimp/                 the package
  sets.py, heads.py, types.py, instances.py     token sets, the attention head, synthetic families
  bounds.py, theory.py                          controls, secant gaps, knapsack, the gap bounds of the paper
  relax_ps.py, ibp.py                           first-order certificate, interval pattern bounds
  sdp_mosek.py, relax_sdp.py, safe_bound.py     level-one relaxation (MOSEK task API; CVXPY for synthetic heads), safe bound
  certify.py                                    the certificate, the re-solve rule, branch-and-bound, tightness radius
  exact_fq.py, exact_gurobi.py                  exact fixed-query worst case; exact references (Gurobi) for synthetic heads
  lower_bounds.py, lower.py                     multistart ascent, alternating ascent
  gpt2.py, gpt2_units.py, pipeline.py           GPT-2 extraction, units, the per-unit procedures of Section 7
  lirpa.py                                      auto_LiRPA baselines
  block.py, mechanism.py                        GPT-2 block, remainder network, Hessian and per-operation tools
experiments/          one script per experiment (see scripts/reproduce.md)
benchmark/            exact fixed-query worst cases of GPT-2 small (SCHEMA.md, verify.py)
figdata/, figscripts/ figure data of the paper; make_figs.py, stats.py
tests/                unit tests and end-to-end smoke tests
data/prompts.txt      the five prompts
scripts/reproduce.md  table, figure and number -> command and output file
```

## Installation

Python 3.11 or later (the experiments ran on Python 3.14.7). CPU only; no GPU is used.

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps --ignore-requires-python "auto_LiRPA @ https://github.com/Verified-Intelligence/auto_LiRPA/archive/e3930353a3589f7d10879f1e71223d1eefb57196.zip"
```

auto_LiRPA 0.8.0 (the commit above) is needed only for `experiments/bound_propagation.py` and `experiments/mechanism.py`.
It declares older Python and torch versions than those used here but works with them; `--no-deps` keeps it from changing
installed packages.

**Solver licences.** The semidefinite relaxations need MOSEK (version 11) with a licence; MOSEK licences are free for
academic use (https://www.mosek.com/products/academic-licenses/). Gurobi (version 13) is used only for the exact
reference values on synthetic heads in `experiments/synthetic.py` (free academic licences exist as well); nothing else
needs it. The GPT-2 weights are downloaded from the Hugging Face hub on first use (about 500 MB, cached by
`transformers`).

## Configuration

Everything is set through environment variables with defaults; no path or machine setting is hard-coded.

| variable | default | meaning |
|---|---|---|
| `AIMP_MAX_THREADS` | 4 | total CPU threads of every solver, BLAS and torch call; worker pools use `workers x threads <= AIMP_MAX_THREADS` |
| `AIMP_RESULTS_DIR` | `./results` | where the experiments write |
| `AIMP_CACHE_DIR` | `./cache` | cached GPT-2 weights and prompt activations |
| `AIMP_GUROBI_MEMLIMIT_GB` | 20 | Gurobi memory limit of the exact references |
| `HF_HOME` | the `transformers` default | Hugging Face cache |

## Quick start

```bash
python -m pytest -q tests/
python experiments/exact_fixed_query.py --quick
python experiments/gpt2_ladder.py --quick
python experiments/collapse.py --quick
python benchmark/verify.py --sample 200 --seed 0
cd figscripts && python stats.py && python make_figs.py
```

Every experiment script runs the paper's full settings by default and has a `--quick` option that runs in minutes.
`scripts/reproduce.md` lists, for every table, figure and number of Section 7, the command that produces it and its
output file, with the run times measured for the paper.

## Hardware used for the paper

One workstation: Intel Xeon w5-2465X processor (16 cores, 3.1 GHz), 256 GB of memory, Debian GNU/Linux 13, no
graphics processor. The experiments used at most 14 threads. The largest semidefinite programs (GPT-2 heads at query
position 12) take about five minutes and 4 GB each at 4 threads.

## The benchmark

`benchmark/` contains, for every head of GPT-2 small, five prompts, query positions 4, 8, 12 and the last one, and six
relative radii (17,280 units), the certified brackets of the maximum and of the minimum of the head's output along its
nominal direction when every key token moves in an l2 ball of radius `rho_rel * ||z_j||` around the normalized token the
head reads and the query is fixed, together with the maximizing and minimizing angle of every key, so that the
worst-case inputs can be rebuilt. `benchmark/SCHEMA.md` states the threat model; `benchmark/verify.py` rebuilds units from
the Hugging Face weights and checks them.

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

## Licence

MIT (see `LICENSE`). MOSEK and Gurobi are commercial solvers with their own licences and are not distributed here.
