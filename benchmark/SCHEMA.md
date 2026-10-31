# Exact fixed-query worst cases of GPT-2 small

This directory holds the exact worst cases of every attention head of GPT-2 small with a fixed query, computed with
Theorem 7 of the paper *Conic Certification of Transformer Attention*. They are 17,280 units:

| factor | values | count |
|---|---|---|
| layer `l` | 0–11 | 12 |
| head `h` | 0–11 | 12 |
| prompt `p` | the five sentences of `prompts.txt` (ids 0–4, 19, 21, 19, 21 and 22 tokens) | 5 |
| query position `i` | 4, 8, 12 and the last token (counted from zero) | 4 |
| relative radius `rho_rel` | 0.001, 0.002, 0.005, 0.01, 0.02, 0.05 | 6 |

Each unit has the certified brackets of the maximum and of the minimum of the head's output along its nominal direction,
the nominal output, the range of motion, and the maximizing and the minimizing input of every varying key, so that the
worst-case inputs can be rebuilt and checked (`verify.py`).

## Threat model

* **Model.** GPT-2 small (`gpt2` on the Hugging Face hub), weights in float64, eager attention. Prompts are tokenized
  with the GPT-2 tokenizer, without special tokens.
* **What the head reads.** Head `h` of block `l` reads the post-normalization tokens `zbar_j = ln_1(x_j) in R^768`
  (`x_j` the residual stream entering block `l` on the unperturbed prompt), for `j = 0, ..., i` (causal mask).
* **Perturbations.** Every key token `j < i` may move independently in the Euclidean ball
  `Z_j = { z : ||z - zbar_j||_2 <= rho_j }`, with radius `rho_j = rho_rel * ||zbar_j||_2` proportional to its own norm.
  The query token `i` is fixed at `zbar_i`, so the query `q_i`, and the key and value of token `i`, are fixed. Tokens
  after `i` do not affect the output at `i`. The perturbation acts on the normalized tokens of block `l` only: the
  residual stream, the normalization and earlier blocks are not perturbed.
* **Output.** With `q_i = zbar_i W_Q + b_Q`, `k_j = z_j W_K + b_K`, `v_j = z_j W_V + b_V` (the head's 64-dimensional
  slices of `c_attn`), the head's output at position `i` is its contribution to the residual stream,

      o_i(z) = sum_{j <= i} a_j v_j W_O,    a = softmax_j ( q_i . k_j / 8 ),

  where `W_O` is the head's 64 x 768 slice of `c_proj.weight` (the bias of `c_proj` is shared by the heads and is not
  part of any head).
* **Specification.** `c = o_i(zbar) / ||o_i(zbar)||`, the direction of the head's nominal output. The certified quantity
  is `phi(z) = c . o_i(z)` over the product of the balls `Z_0 x ... x Z_{i-1}`; the benchmark gives its maximum and its
  minimum.

## Files

* `units.csv` — one row per unit:

  | column | meaning |
  |---|---|
  | `unit_id` | `L{l:02d}H{h:02d}_P{p}_i{i:02d}_r{rho_rel}` |
  | `layer`, `head`, `prompt`, `query_pos`, `rho_rel` | the instance |
  | `radius_rule` | `rho_j = rho_rel * ||zbar_j||` (the same for every unit) |
  | `n_keys`, `n_varying` | `i + 1` keys, of which `i` vary |
  | `max_lo`, `max_hi` | certified bracket of `max phi`: `max_lo <= max phi <= max_hi` |
  | `min_lo`, `min_hi` | certified bracket of `min phi` |
  | `nominal_output` | `phi(zbar) = ||o_i(zbar)||` |
  | `range_of_motion` | `(max_lo + max_hi)/2 - (min_lo + min_hi)/2` |
  | `status` | `exact` when both brackets are narrower than `1e-10` times the nominal value spread `max_j c.(v_j W_O) - min_j c.(v_j W_O)` |
  | `value_at_max_input`, `value_at_min_input` | `phi` at the stored worst-case inputs (numpy forward pass) |
  | `first_order_certificate` | the first-order certificate `r_pp` of the paper (an upper bound on `max phi`) |

* `angles.csv.gz` — one row per varying key of every unit: `unit_id`, `key` (`j`), `rho_j`, `theta_max`, `theta_min`.
* `prompts.txt` — the five prompts, one per line, in the order of their ids.
* `verify.py` — rebuilds units from the Hugging Face weights and checks them (below).

Numbers are written with 17 significant digits, so that the inputs can be rebuilt bit for bit.

## Rebuilding the worst-case inputs

Every varying key moves in the same plane of `R^768`. Its score changes by `alpha . delta_j` with
`alpha = W_K q_i / 8`, and its value along `c` by `beta . delta_j` with `beta = W_V (W_O c)` (`W_K`, `W_V` are
768 x 64, `W_O` is 64 x 768). Let

    e1 = alpha / ||alpha||,    e2 = (beta - (beta . e1) e1) / ||beta - (beta . e1) e1||.

The worst case lies on the boundary of every ball and in this plane (Theorem 7). The maximizing input is

    z_j = zbar_j + rho_j ( cos(theta_max_j) e1 + sin(theta_max_j) e2 ),   j = 0, ..., i - 1,
    z_i = zbar_i,

and the minimizing input is the same with `theta_min_j`. The angles are canonical: they depend only on the weights, the
prompt and the instance, not on any factorization used by the solver.

## How the brackets were computed

For a fixed query, `max phi` is the unique root of the strictly decreasing function
`H(lambda) = sum_j max_{z_j in Z_j} e^{s_j} (g_j - lambda)`, which separates over the keys (Theorem 7(i)–(ii)). Each
per-key maximum is taken over the boundary of the planar ellipse traced by `(s_j, g_j)`. It is bounded above by a
certified subdivision of the boundary angle, with first- and second-order bounds on every cell, rather than by the closed
form of Theorem 7(iii). The root is bracketed by Dinkelbach steps and safeguarded bisection until the bracket is below
`1e-10` of the nominal value spread (a median 6e-10, at most 6e-6, of the range of motion). Every evaluated quantity
carries a rounding allowance. The lower end of a bracket is certified either by `H > 0` or by a feasible input; the
upper end by `H < 0`. All 17,280 units are `exact`.

## Verifying

    python benchmark/verify.py --sample 200 --seed 0   # 200 random units
    python benchmark/verify.py --rho 0.01              # all 2,880 units of one radius
    python benchmark/verify.py --unit L05H01_P0_i04_r0.01

`verify.py` needs only numpy, torch (CPU) and transformers. For each unit it recomputes `zbar` from the model, checks the
radius rule and the nominal output, rebuilds both worst-case inputs from the angles and checks that each perturbation
lies in its ball. It then evaluates `phi` twice, once with a direct numpy implementation of the head and once through the
Hugging Face model with the perturbed tokens injected at the output of `ln_1`. Both values must lie in the stored bracket
up to `1e-9 (1 + |bound|)`.
