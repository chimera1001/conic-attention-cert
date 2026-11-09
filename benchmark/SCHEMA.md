# Exact fixed-query worst cases of GPT-2 small and gpt-oss-20b

This directory holds exact worst cases of attention heads with a fixed query, computed with Theorem 7 of the paper
*Conic Certification of Transformer Attention*, for every head of two trained models:

| model | files | units |
|---|---|---|
| GPT-2 small (`gpt2`) | `units.csv`, `angles.csv.gz`, `prompts.txt` (this directory) | 17,280 |
| gpt-oss-20b (`openai/gpt-oss-20b`) | `gpt-oss-20b/units.csv`, `gpt-oss-20b/angles.csv.gz`, `gpt-oss-20b/prompts.txt` | 61,440 |

Each unit has the certified brackets of the maximum and of the minimum of the head's output along its nominal direction,
the nominal output, the range of motion, and the maximizing and the minimizing input of every varying key, so that the
worst-case inputs can be rebuilt and checked (`verify.py`). The two models share the threat model below; the section on
gpt-oss-20b lists what differs.

## GPT-2 small

| factor | values | count |
|---|---|---|
| layer `l` | 0–11 | 12 |
| head `h` | 0–11 | 12 |
| prompt `p` | the five sentences of `prompts.txt` (ids 0–4, 19, 21, 19, 21 and 22 tokens) | 5 |
| query position `i` | 4, 8, 12 and the last token (counted from zero) | 4 |
| relative radius `rho_rel` | 0.001, 0.002, 0.005, 0.01, 0.02, 0.05 | 6 |

### Threat model

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

### Files

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

### Rebuilding the worst-case inputs

Every varying key moves in the same plane of `R^768`. Its score changes by `alpha . delta_j` with
`alpha = W_K q_i / 8`, and its value along `c` by `beta . delta_j` with `beta = W_V (W_O c)` (`W_K`, `W_V` are
768 x 64, `W_O` is 64 x 768). Let

    e1 = alpha / ||alpha||,    e2 = (beta - (beta . e1) e1) / ||beta - (beta . e1) e1||.

The worst case lies on the boundary of every ball and in this plane (Theorem 7). The maximizing input is

    z_j = zbar_j + rho_j ( cos(theta_max_j) e1 + sin(theta_max_j) e2 ),   j = 0, ..., i - 1,
    z_i = zbar_i,

and the minimizing input is the same with `theta_min_j`. The angles are canonical: they depend only on the weights, the
prompt and the instance, not on any factorization used by the solver.

### How the brackets were computed

For a fixed query, `max phi` is the unique root of the strictly decreasing function
`H(lambda) = sum_j max_{z_j in Z_j} e^{s_j} (g_j - lambda)`, which separates over the keys (Theorem 7(i)–(ii)). Each
per-key maximum is taken over the boundary of the planar ellipse traced by `(s_j, g_j)`. It is bounded above by a
certified subdivision of the boundary angle, with first- and second-order bounds on every cell, rather than by the closed
form of Theorem 7(iii). The root is bracketed by Dinkelbach steps and safeguarded bisection until the bracket is below
`1e-10` of the nominal value spread (a median 6e-10, at most 6e-6, of the range of motion). Every evaluated quantity
carries a rounding allowance. The lower end of a bracket is certified either by `H > 0` or by a feasible input; the
upper end by `H < 0`. All 17,280 units are `exact`.

## gpt-oss-20b (`gpt-oss-20b/`)

| factor | values | count |
|---|---|---|
| layer `l` | 0–23 | 24 |
| query head `h` | 0–63, reading the key-value group `g = h // 8` | 64 |
| prompt `p` | the five sentences of `gpt-oss-20b/prompts.txt` (the same as for GPT-2; ids 0–4, 19, 21, 19, 21 and 22 tokens with the gpt-oss tokenizer) | 5 |
| query position `i` | 4 and the last token | 2 |
| relative radius `rho_rel` | 0.0002, 0.0004, 0.001, 0.002 (`r0 x {1, 2, 5, 10}`, `r0` from the radius calibration of the paper) | 4 |

### Threat model (what differs from GPT-2)

* **Model.** `openai/gpt-oss-20b` on the Hugging Face hub, revision `6cee5e81ee83917806bbde320786a8fb61efebee`, eager
  attention; the MXFP4 expert weights are dequantized on CPU and the model runs in float32 (all weights are bf16 values,
  exact in float32 and float64). Prompts are tokenized with the model's tokenizer, which adds no special token.
* **What the head reads.** Query head `h` of layer `l` reads `zbar_j = input_layernorm(x_j) in R^2880`, the output of the
  RMSNorm before the attention sublayer (`x_j` the residual stream entering layer `l` on the unperturbed prompt), for
  `j = 0, ..., i`. The perturbations, the radius rule and the fixed query are as for GPT-2.
* **Output.** With `q_i = W_Q zbar_i + b_Q`, `k_j = W_K z_j + b_K`, `v_j = W_V z_j + b_V` (rows `64h..64h+63` of
  `q_proj`, rows `64g..64g+63` of `k_proj` and `v_proj`, with their biases), the scores are
  `s_j = (R_i q_i) . (R_j k_j) / 8`, with `R_j` the model's rotary map at position `j` (`apply_rotary_pos_emb` with the
  model's cos and sin tables, which include the YaRN attention factor), and the head's output at position `i` is

      o_i(z) = W_O sum_{j <= i} a_j v_j,    a = softmax(s_0, ..., s_i, s_sink),

  where `s_sink` is the head's learned sink logit (the sink is a key with a fixed score and value zero) and `W_O` is
  columns `64h..64h+63` of `o_proj.weight` (the bias of `o_proj` is shared by the heads). The layers alternate full and
  sliding-window attention (window 128); the window does not bind on these prompts.
* **Specification.** `c = o_i(zbar) / ||o_i(zbar)||` and `phi(z) = c . o_i(z)`, as for GPT-2.

### Files

* `gpt-oss-20b/units.csv` — the columns of `units.csv`, plus `kv_group` (`h // 8`) and `vertex_softmax_bound` (the
  Vertex-Softmax bound `r_VS`, the exact maximum of the softmax-weighted support values over the box of certified scores;
  an upper bound on `max phi`, between it and the first-order certificate). `n_keys = i + 2` counts the sink;
  `n_varying = i`. `status` is `exact` when both brackets are narrower than `1e-10` times the nominal value spread over
  the keys, the sink's value 0 included.
* `gpt-oss-20b/angles.csv.gz` — as `angles.csv.gz`.
* `gpt-oss-20b/prompts.txt` — the five prompts.

### Rebuilding the worst-case inputs

The plane of key `j` depends on `j` through its rotary map: its score changes by `alpha_j . delta_j` with
`alpha_j = W_K^T R_j^T R_i q_i / 8`, and its value along `c` by `beta . delta_j` with `beta = W_V^T W_O^T c` (here
`W_K`, `W_V` are 64 x 2880 and `W_O` is 2880 x 64, the orientation of the model's linear layers). With
`e1_j = alpha_j / ||alpha_j||` and `e2_j` the normalized component of `beta` orthogonal to `e1_j`, the maximizing input is

    z_j = zbar_j + rho_j ( cos(theta_max_j) e1_j + sin(theta_max_j) e2_j ),   j = 0, ..., i - 1,
    z_i = zbar_i,

and the minimizing input is the same with `theta_min_j`.

### How the brackets were computed

As for GPT-2. The rotary maps act after the projection, so the score of key `j` is `q_i^T M_j k_j` with
`M_j = R_i^T R_j / 8`; they are folded into each token's set by an invertible linear map per token, and the sink is a key
with a constant score and value zero, so Theorem 7 applies unchanged. All 61,440 units are `exact`: both brackets are
below `1e-10` of the nominal value spread (at most 5e-11 of it). Relative to the range of motion they are a median 4e-9
and at most 5e-6, except on the 297 units whose range of motion is below `1e-5` of the value spread (heads that barely
move), where they reach 0.39 of it.

## Verifying

    python benchmark/verify.py --sample 200 --seed 0                       # GPT-2 small, 200 random units
    python benchmark/verify.py --rho 0.01                                  # all 2,880 units of one radius
    python benchmark/verify.py --unit L05H01_P0_i04_r0.01
    python benchmark/verify.py --all                                       # all 17,280 units
    python benchmark/verify.py --model gpt-oss-20b --sample 200 --seed 0   # gpt-oss-20b
    python benchmark/verify.py --model gpt-oss-20b --all                   # all 61,440 units

`verify.py` needs only numpy, torch (CPU) and transformers (with `accelerate` for gpt-oss-20b). For each unit it
recomputes `zbar` from the model, checks the radius rule and the nominal output, rebuilds both worst-case inputs from the
angles and checks that each perturbation lies in its ball. It then evaluates `phi` twice, once with a direct numpy
implementation of the head in float64 and once with the model's own code: for GPT-2 the Hugging Face model with the
perturbed tokens injected at the output of `ln_1`; for gpt-oss-20b the layer's Hugging Face attention module (its
projections, rotary embedding, sinks and eager attention), copied to float64, on the perturbed normalized tokens. Both
values must lie in the stored bracket up to `1e-9 (1 + |bound|)`.

gpt-oss-20b is loaded once: about 13 GB of weights from the Hugging Face hub (the `original/` and `metal/` folders are
not needed), about 85 GB of memory while the five prompts run, about 1.5 minutes on the workstation of the paper; the
model is then freed and the checks take about 25 ms per unit (all 61,440 units in 25 minutes). The stored tokens come from a float32 forward pass at 10
threads (the default `--threads` for this model), which the same computation reproduces bit for bit on a CPU of the
same type. A different thread count or CPU changes the float32 arithmetic: the normalized tokens then differ at about
`1e-7` relative, and `verify.py` reports it and checks those units' brackets to `1e-5 (1 + |bound|)` instead (with 4
threads on the workstation of the paper the largest excess over a bracket was `9.5e-7 (1 + |bound|)`).
