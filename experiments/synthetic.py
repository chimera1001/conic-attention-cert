"""Synthetic-head experiments of the paper: Table 3, Figure 1 and the numbers of the synthetic part of the computational
results.

Parts (``--part``; default: all four, at full settings):

* ``slopes`` -- every certificate on every instance of the synthetic families (``d = 8``, ``d_k = d_v = 4``):
  F_random (random heads, temperature ``||M||`` in {0.1, 0.3, 1, 3, 10}), F_margin (a dominant key of nominal margin 1
  to 8), F_ambiguous (two keys tie in score but differ in value along the specification), F_rank1 (rank-one query-key
  matrix) and F_prop9 (the two-key sharpness-type family, ``n = 2``, whose values are proportional to the displacement).
  Ball and box token sets of radius ``rho`` around the nominal tokens; query in / not in the key set. Grid: ``n = 2``
  (10 seeds, ``rho`` in RHOS), ``n = 3`` (10 seeds, ``rho <= 0.2``, without the 40 F_rank1 ball instances of
  :data:`NOT_IN_PAPER`), ``n = 4`` (F_random with ``||M||`` in {1, 3} and F_ambiguous, 5 seeds,
  ``rho <= 0.1``). Methods:

  - ``ibp_obj``: the interval pattern bounds (interval bounds on the pattern, exact support of each value);
  - ``ps_auto``: the first-order certificate over the pattern polytope with the certified pair chosen automatically;
    ``ps_C``: the same with the margin pair (only when the certified margin is positive);
  - ``sdp_1``: the level-one relaxation of Definition 3; ``sdp_1_unitdiag``: the same with the unit diagonal
    ``Y[a_j, a_j] <= a_j`` in place of the interval diagonal (R4);
  - ``sdp_2``, ``sdp_3_score``, ``sdp_3_cut``, ``sdp_3_bxb`` (boxes), ``sdp_3_ball`` (balls): the level-one relaxation
    strengthened by products of the softmax sandwich with the sets, of the score intervals with the weights, of the
    l1 cut with the sets, and between box blocks (reference methods for "the best of all methods");
  - ``multistart``: best feasible value of 50 local ascents; ``exact``: the nonconvex program in log form (Gurobi,
    relative gap 1e-6 or a time limit, in which case only its bounds are used).

* ``width`` -- the secant-width grid: F_random, ``n = 3``, balls, ``rho = 0.1``, query = key 0, 10 seeds; the
  temperature of each instance is chosen by bisection so that the certified tight secant width ``w_t`` hits each
  value of W_GRID. Methods: ``ibp_obj``, ``ps_auto`` / ``ps_C`` (greedy), ``sdp_1``, ``multistart``, ``exact``;
  ``phi_min.json`` holds a feasible estimate of the minimum of the output (relative gaps).

* ``temperature`` -- the gap against the temperature ``kappa = ||M||`` at fixed radius: F_random, F_ambiguous,
  F_ambiguous_ts (F_ambiguous built at temperature 1, then ``M -> kappa M``) and F_degen (a key varying only in the
  null space of the specification, a degenerate control); ``n = 3``, balls, query = key 0, 5 seeds,
  ``rho`` in {0.05, 0.2}, ``kappa`` in KAPPAS. Methods: ``ps_auto`` / ``ps_C`` (greedy), ``sdp_0`` (no products with
  the sets), ``sdp_1``, ``multistart``, ``exact``.

* ``summary`` -- computes from the stored per-instance records: Table 3 (median and interquartile range of per-series
  least-squares slopes of ``log(gap)`` against ``log(rho)`` on ``rho <= 0.1``, ``n = 2``), the data of Figure 1
  (``P1_gap_vs_rho.csv``), the ratio of the interval pattern bounds to the level-one certificate, the minimum of the
  first-order and level-one certificates against the best of all methods, the width ratios and the temperature orders;
  prints them next to the values of the paper and writes CSV files to ``<out>/summary``.

Gaps are ``value - p_ref`` with ``p_ref`` the best verified feasible value of the instance; an instance enters a fit
only if the exact bracket ``res = p_star_ub - p_ref`` is at most a tenth of the gap of the method (resolved), and gaps
below ``1e-7`` are solver noise. The exact solve stops once its bracket is below 5% of the smallest relaxation gap.

``--t-interval tight`` (default) is the formulation of Definition 3: the log-attention intervals are
``[log a_lo_j, log a_hi_j]`` intersected with ``[s_lo_j - tau_hi, s_hi_j - tau_lo]``; the reported value is MOSEK's
dual objective when its dual residual is at most ``1e-8 max(1, |value|)``, else the safe dual bound
(``aimp.safe_bound``), which is computed on every solve. ``--t-interval loose`` uses ``[s_lo_j - tau_hi,
s_hi_j - tau_lo]`` with the solver settings of the stored loose records (CVXPY's primal objective, no exponential-cone
pruning, no explicit cone form); it reproduces those records.

Records: ``<out>/<part>/<family>/<instance_id>/{instance.json, instance.npz, <method>.json, done.json}`` (written
atomically; a method whose record exists is not recomputed and an instance with ``done.json`` is skipped, so a sweep
can be interrupted and resumed); log ``<out>/<part>/sweep.log``. The summary reads any directory with this layout,
including records of earlier runs of the same code.

Usage::

    AIMP_MAX_THREADS=3 python experiments/synthetic.py --quick --workers 3
    AIMP_MAX_THREADS=14 python experiments/synthetic.py --part slopes width temperature --workers 14
    python experiments/synthetic.py --part summary --out <records directory>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from aimp import bounds, compute  # noqa: E402
from aimp.io import (derive_fields, env_info, jsonable, result_record, results_dir, save_instance,  # noqa: E402
                     write_json_atomic)

PARTS = ("slopes", "width", "temperature")

# ----------------------------------------------------------------------------- grids
FAMILY_CONFIGS = {  # family -> configurations (temperature, margin, separation or kappa_M)
    "F_prop9": [1.0],  # kappa_M: Delta = 2 kappa_M rho on balls
    "F_random": [0.1, 0.3, 1.0, 3.0, 10.0],  # temperature ||M||
    "F_ambiguous": [1.0, 0.25],  # value separation of the tied pair (1 = antipodal values; 0.25 = control)
    "F_margin": [1.0, 2.0, 4.0, 8.0],  # nominal margin of the dominant key
    "F_rank1": [0.1, 0.3, 1.0, 3.0, 10.0],  # temperature ||M||, d_k = 1
}
RHOS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
SET_TYPES = ("box", "ball")
MULTI_KEY = {k: v for k, v in FAMILY_CONFIGS.items() if k != "F_prop9"}  # families with n >= 3
#: (family, n, set type, rho, configurations, seeds) of grid points that are not among the paper's 8,100 instances:
#: F_rank1 with three keys in balls at rho = 0.2, temperatures 0.3, 1, 3, 10, seeds 5-9 (40 instances).
NOT_IN_PAPER = [("F_rank1", 3, "ball", 0.2, (0.3, 1.0, 3.0, 10.0), range(5, 10))]


def in_paper_grid(task: dict) -> bool:
    """False for the grid points of :data:`NOT_IN_PAPER` (the slopes grid is then exactly the 8,100 instances)."""
    for fam, n, st, rho, cfgs, seeds in NOT_IN_PAPER:
        if (task["family"] == fam and task["n"] == n and task["set_type"] == st and abs(task["rho"] - rho) < 1e-12
                and any(abs(task["cfg"] - c) < 1e-12 for c in cfgs) and task["seed"] in seeds):
            return False
    return True
SLOPES_GRID = [  # blocks of the slopes sweep (the synthetic instances of the paper)
    {"n": 2, "families": FAMILY_CONFIGS, "seeds": range(10), "rhos": RHOS, "set_types": SET_TYPES},
    {"n": 3, "families": MULTI_KEY, "seeds": range(10), "rhos": [r for r in RHOS if r <= 0.2], "set_types": SET_TYPES},
    {"n": 4, "families": {"F_random": [1.0, 3.0], "F_ambiguous": [1.0, 0.25]}, "seeds": range(5),
     "rhos": [r for r in RHOS if r <= 0.1], "set_types": SET_TYPES},
]
QUICK_SLOPES_GRID = [
    {"n": 2, "families": {"F_random": FAMILY_CONFIGS["F_random"], "F_ambiguous": FAMILY_CONFIGS["F_ambiguous"]},
     "seeds": range(3), "rhos": [0.01, 0.02, 0.05, 0.1], "set_types": ("ball",)},
]
W_GRID = [0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 30.0, 50.0, 80.0]
QUICK_W_GRID = [0.05, 1.0, 8.0]
KAPPAS = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4]
TEMPERATURE_FAMILIES = ("F_random", "F_ambiguous", "F_ambiguous_ts", "F_degen")
TEMPERATURE_GRID = {"families": TEMPERATURE_FAMILIES, "rhos": [0.05, 0.2], "kappas": KAPPAS, "seeds": range(5)}
QUICK_TEMPERATURE_GRID = {"families": ("F_random",), "rhos": [0.05], "kappas": KAPPAS[:4], "seeds": range(3)}
WIDTH_SEEDS = QUICK_WIDTH_SEEDS = range(10)  # the ratios are medians over the seeds: keep all ten
QUICK_EXACT_TIME_LIMIT = 20.0

T_SDP, T_PS = 300.0, 60.0  # time limits of one relaxation solve (s)
STOP_FRACTION = 0.05  # exact early stop: bracket <= 5% of the smallest relaxation gap (twice the resolution rule)
MIPGAP_ABS_FLOOR = 1e-8  # below this, gaps are within the solver tolerances
ATTRIB_TOL = 1e-9  # a deficit is solver inaccuracy if the reference point satisfies every constraint to this accuracy
S1 = ("persp", "simplex_x_z")  # (R5): products of the simplex and of the sets with each other
S2 = S1 + ("softmax_x_z",)  # + products of the softmax sandwich (secant, tangents) with the linear set bounds


def sdp_settings(t_interval: str) -> dict:
    """Keyword arguments of :func:`aimp.relax_sdp.solve_sdp` for ``--t-interval`` (see the module docstring)."""
    from aimp.relax_sdp import PRUNE_TOL

    if t_interval == "tight":
        return {"t_interval": "tight", "value_mode": "certified", "prune_tol": PRUNE_TOL, "cone_form": True,
                "safe": "always"}
    if t_interval == "loose":
        return {"t_interval": "loose", "value_mode": "primal", "prune_tol": None, "cone_form": False, "safe": "never"}
    raise ValueError(t_interval)


def exact_time_limit(part: str, task: dict, override: float | None) -> float | None:
    """Time limit of the exact solve (the reference of every synthetic instance is a time-limited global solve)."""
    if part == "slopes":
        n, rho = task["n"], task["rho"]
        default = (300.0 if rho <= 0.2 else 90.0) if n == 2 else (300.0 if n == 3 else 120.0)
    else:
        default = 300.0 if part == "width" else 120.0
    return default if override is None else float(override)


# ----------------------------------------------------------------------------- tasks and instances
def build_tasks(part: str, quick: bool = False, out: Path | None = None, threads: int = 1) -> list[dict]:
    """The tasks of a part, in the order they are run (small ``n`` and small ``rho`` first; boxes before balls)."""
    tasks = []
    if part == "slopes":
        for blk in (QUICK_SLOPES_GRID if quick else SLOPES_GRID):
            for rho in blk["rhos"]:
                for st in blk["set_types"]:
                    for fam, cfgs in blk["families"].items():
                        for cfg in cfgs:
                            for seed in blk["seeds"]:
                                for qk in (True, False):
                                    if fam == "F_prop9" and qk:  # fixed query, not a key
                                        continue
                                    t = {"family": fam, "cfg": cfg, "seed": int(seed), "n": blk["n"], "rho": rho,
                                         "set_type": st, "qk": qk}
                                    if quick or in_paper_grid(t):
                                        tasks.append(t)
    elif part == "width":
        seeds, grid = (QUICK_WIDTH_SEEDS, QUICK_W_GRID) if quick else (WIDTH_SEEDS, W_GRID)
        temps = width_temperatures(out, seeds, grid, threads) if out is not None else {}
        for wt in grid:  # small w_t first (cheap exact solves)
            for seed in seeds:
                tasks.append({"family": "F_random", "seed": int(seed), "w_target": wt,
                              "temp": temps.get(f"{seed}_{wt}", {}).get("temp")})
    elif part == "temperature":
        g = QUICK_TEMPERATURE_GRID if quick else TEMPERATURE_GRID
        for rho in g["rhos"]:
            for kappa in g["kappas"]:
                for fam in g["families"]:
                    for seed in g["seeds"]:
                        tasks.append({"family": fam, "seed": int(seed), "rho": rho, "kappa": kappa})
    else:
        raise ValueError(part)
    return tasks


def task_instance(part: str, task: dict):
    from aimp.instances import p2_delta, p2_f1, phase1_instance

    if part == "slopes":
        return phase1_instance(task["family"], task["seed"], task["n"], task["rho"], task["set_type"], task["qk"],
                               task["cfg"])
    if part == "width":
        return p2_delta(task["seed"], task["w_target"], task["temp"])
    return p2_f1(task["family"], task["seed"], task["rho"], task["kappa"], True)


def task_dir(out: Path, part: str, inst) -> Path:
    return Path(out) / part / inst.meta["family"] / inst.meta["instance_id"]


def temp_for_w(seed: int, w_target: float, threads: int = 1) -> dict:
    """Temperature giving the tight secant width ``w_t = w_target`` for the width instance of ``seed``. Scores are linear
    in ``M = temp M_1``, so the certified score intervals at ``temp`` are ``temp`` times those at ``temp = 1`` (up to the
    solver pad); ``w_t(temp)`` is evaluated on the scaled intervals and solved for by bisection in ``log temp``. The
    instance is then rebuilt at that temperature, and its own certified intervals give the achieved ``w_t``."""
    from aimp.instances import p1_random

    inst1 = p1_random(seed, 3, 0.1, "ball", True, temp=1.0)
    si = bounds.score_intervals(inst1, "shor", threads=threads)
    lo1, hi1 = np.asarray(si["s_lo"]), np.asarray(si["s_hi"])

    def w(T):
        return bounds.t_width(T * lo1, T * hi1, "tight")

    a, b = math.log(1e-4), math.log(1e3)
    for _ in range(200):
        m = (a + b) / 2
        if w(math.exp(m)) < w_target:
            a = m
        else:
            b = m
    T = math.exp((a + b) / 2)
    return {"seed": seed, "w_target": w_target, "temp": T, "w_t_scaled": w(T)}


def width_temperatures(out: Path, seeds, grid, threads: int = 1) -> dict:
    """``<out>/width/temps.json``: the temperature of every (seed, w_t target), computed once and reused."""
    path = Path(out) / "width" / "temps.json"
    temps = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    missing = [(s, wt) for s in seeds for wt in grid if f"{s}_{wt}" not in temps]
    for s, wt in missing:
        temps[f"{s}_{wt}"] = temp_for_w(int(s), wt, threads)
    if missing:
        write_json_atomic(path, temps)
    return temps


# ----------------------------------------------------------------------------- methods
def _solver_of(method: str, part: str) -> dict:
    from aimp.solvers import cvxpy_version, gurobi_version, mosek_version

    if method == "exact" or (method.startswith("ps") and part == "slopes"):
        return {"name": "Gurobi", "version": gurobi_version()}
    if method.startswith("sdp"):
        return {"name": "MOSEK", "version": mosek_version(), "via": f"cvxpy {cvxpy_version()}"}
    return {"name": "numpy/scipy"}


def method_table(part: str, inst, ctrl: dict, threads: int, cfg: dict, t_exact: float | None) -> dict:
    """name -> (callable(ctx) -> SolveResult, bound spec for the first-order gap bound). Bound spec: None | "auto" | "C"
    (``ps_*`` and every SDP with the products (R5), whose value never exceeds the first-order certificate's)."""
    from aimp.ibp import solve_ibp_obj
    from aimp.lower_bounds import multistart_lb
    from aimp.relax_ps import solve_ps, solve_ps_greedy
    from aimp.relax_sdp import solve_sdp

    seed = int(inst.meta.get("seed") or 0)
    kinds = {inst.sets[t].kind for t in inst.tokens if inst.sets[t].rho != 0.0}
    full = cfg["methods"] == "all"

    def sdp(fams, **kw):
        return lambda ctx: solve_sdp(inst, ctrl, rlt=fams, threads=threads, time_limit=T_SDP, **cfg["sdp"], **kw)

    m = {"multistart": (lambda ctx: multistart_lb(inst, 50, seed=seed), None)}
    if part == "slopes":
        m["ps_auto"] = (lambda ctx: solve_ps(inst, ctrl, "auto", threads=threads, time_limit=T_PS), "auto")
        if full and ctrl["gamma"] > 0:
            m["ps_C"] = (lambda ctx: solve_ps(inst, ctrl, "C", threads=threads, time_limit=T_PS), "C")
        m["sdp_1"] = (sdp(S1), "auto")
        m["sdp_1_unitdiag"] = (sdp(S1, diag_a="unit"), "auto")
        if full:
            m["sdp_2"] = (sdp(S2), "auto")
            m["sdp_3_score"] = (sdp(S2 + ("score_rlt",)), "auto")
            m["sdp_3_cut"] = (sdp(S2 + ("cut_x_z",)), "auto")
            if "box" in kinds:
                m["sdp_3_bxb"] = (sdp(S2 + ("box_x_box",)), "auto")
            if kinds & {"ball", "ellipsoid"}:
                m["sdp_3_ball"] = (sdp(S2 + ("softmax_x_ball",)), "auto")
        m["ibp_obj"] = (lambda ctx: solve_ibp_obj(inst, ctrl), None)
    else:
        m["ps_auto"] = (lambda ctx: solve_ps_greedy(inst, ctrl, "auto"), "auto")
        if ctrl["gamma"] > 0:
            m["ps_C"] = (lambda ctx: solve_ps_greedy(inst, ctrl, "C"), "C")
        if part == "width":
            m["ibp_obj"] = (lambda ctx: solve_ibp_obj(inst, ctrl), None)
        else:
            m["sdp_0"] = (sdp(()), None)  # no products (R5): not dominated by the first-order certificate
        m["sdp_1"] = (sdp(S1), "auto")
    if t_exact is not None:
        m["exact"] = (lambda ctx: _exact(inst, ctrl, threads, ctx, t_exact), None)
    return m


def _exact(inst, ctrl: dict, threads: int, ctx: dict, time_limit: float):
    """The exact nonconvex program, warm-started at the best verified feasible point found so far. It stops once the
    bracket ``ub - p_ref`` is below ``STOP_FRACTION`` times the smallest relaxation gap (then the instance is resolved
    with a factor-two margin), with the absolute and relative gap tolerances tightened (never loosened) accordingly."""
    from aimp.exact_gurobi import solve_exact
    from aimp.heads import forward

    best, Z0 = -math.inf, None
    for res in ctx.values():
        if res.recovered_point is not None and res.recovered_value is not None and res.recovered_value > best:
            best, Z0 = res.recovered_value, res.recovered_point
    if Z0 is not None:
        assert abs(forward(inst, Z0)["phi"] - best) <= 1e-9 * max(1, abs(best))
    gaps = [r.value - best for k, r in ctx.items() if (k.startswith("ps") or k.startswith("sdp"))
            and r.status in ("optimal", "suboptimal") and r.value is not None and math.isfinite(r.value)]
    thr, tol = None, {}
    if gaps and min(gaps) > 0 and math.isfinite(best):
        thr = best + STOP_FRACTION * min(gaps)
        gap_abs = max(MIPGAP_ABS_FLOOR, min(1e-7, STOP_FRACTION * min(gaps)))
        tol = {"MIPGapAbs": gap_abs, "MIPGap": min(1e-6, gap_abs / max(abs(best), 1e-12))}
    res = solve_exact(inst, ctrl, "log", warm_start=Z0, time_limit=time_limit, threads=threads,
                      stop_if_ub_below=thr, **tol)
    if res.status == "interrupted":
        res.status = "resolved_stop"
        res.approximations = ["early_stop_resolved"]
    return res


def _bound_for(spec, ctrl: dict, n: int):
    """The first-order gap bound ``bound_general(eps, rad, ||c||)`` of the certified pair the method uses."""
    if spec is None:
        return None, None
    rad, cn = ctrl["rad"], ctrl["c_norm"]
    if spec == "C":
        return "C", bounds.bound_general(ctrl["eps_C"], rad, cn)
    cut = bounds.choose_cut(ctrl, n, "auto")
    return f"auto:{cut['cut_used']}", bounds.bound_general(cut["eps"], rad, cn)


def phi_min(inst, ctrl: dict, seed: int) -> dict:
    """``min c^T y`` over the admissible set: certified lower bound ``-r_ps(-c)`` (greedy) and feasible upper bound
    (20-start multistart on ``-c``). The relative gaps of the width part use ``p_ref - phi_min_ms``."""
    from aimp.lower_bounds import multistart_lb
    from aimp.relax_ps import solve_ps_greedy
    from aimp.types import Instance

    neg = Instance(head=inst.head, keys=inst.keys, query=inst.query, sets=inst.sets, c=-inst.c, meta=dict(inst.meta))
    ms = multistart_lb(neg, 20, seed=seed + 1000)
    ps = solve_ps_greedy(neg, ctrl, "auto")
    return {"phi_min_ms": -ms.value, "phi_min_lb": -ps.value, "ms_wall_s": ms.wall_time_s}


def _attribute(inst, ctrl: dict, rec: dict, Z_ref: np.ndarray, cfg: dict) -> dict:
    """A relaxation value below ``p_ref``: evaluate every constraint of the same formulation at the rank-one lift of the
    reference point. If all hold to ``ATTRIB_TOL``, then ``r >= p_ref`` holds mathematically and the deficit is solver
    inaccuracy; otherwise the formulation is violated at a feasible point (a bug)."""
    from aimp.relax_sdp import solve_sdp

    p = rec.get("params", {})
    fams = tuple(f for f in (p.get("rlt") or []) if f != "diag")
    chk = solve_sdp(inst, ctrl, rlt=fams, diag_a=p.get("diag_a", "interval"), check_point=Z_ref,
                    t_interval=cfg["sdp"]["t_interval"], prune_tol=cfg["sdp"]["prune_tol"],
                    cone_form=cfg["sdp"]["cone_form"])
    mv = chk.diagnostics["max_violation"]
    return {"kind": "solver_tolerance" if mv <= ATTRIB_TOL else "formulation_violated_at_reference_point",
            "max_violation_at_p_ref_point": mv, "objective_at_p_ref_point": chk.value, "deficit": rec.get("gap"),
            "worst_constraint": chk.diagnostics["worst_constraint_str"][:200]}


def _bound_attribution(rec: dict) -> dict:
    """A failed first-order bound check on an SDP whose value is not the solver's certified dual objective (``safe``: the
    safe dual bound, an over-estimate; ``max_unreliable``: the larger of the primal and dual objectives) is solver
    inaccuracy when the primal value satisfies the bound: the optimum of the relaxation lies between the two, and only the
    optimum is dominated by the first-order certificate (its safe value need not be)."""
    dg = rec.get("diagnostics") or {}
    vp, p_ref, bound = dg.get("value_primal"), rec.get("p_ref"), rec.get("bound_value")
    ok = (dg.get("value_source") in ("safe", "max_unreliable") and _finite(vp) and _finite(p_ref) and _finite(bound)
          and vp - p_ref <= bound + 1e-6)
    return {"kind": "solver_tolerance_duality_gap" if ok else "unattributed", "value_source": dg.get("value_source"),
            "value_primal": vp, "value_dual": dg.get("value_dual"), "value_safe": dg.get("value_safe"),
            "dual_resid": dg.get("dual_resid"), "bound": bound}


def _ctx_from_rec(rec: dict):
    from aimp.types import SolveResult

    Z = rec.get("_recovered_point")
    return SolveResult(method=rec["method"], status=rec["status"], value=rec.get("value"),
                       recovered_point=None if Z is None else np.asarray(Z, float),
                       recovered_value=rec.get("recovered_value"))


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def run_settings(part: str, task: dict, cfg: dict) -> dict:
    return jsonable({"sdp": cfg["sdp"], "exact_time_limit": exact_time_limit(part, task, cfg["exact_time_limit"])})


def check_settings(d: Path, settings: dict) -> None:
    """Refuse to mix records made with different SDP settings or exact time limits in one directory."""
    p = Path(d) / "instance.json"
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8")).get("run_settings")
        if old is not None and old != settings:
            raise RuntimeError(f"{d} holds records made with {old}, not {settings}; use another --out (or --force)")


# ----------------------------------------------------------------------------- one instance
def run_task(part: str, task: dict, out: Path, threads: int, cfg: dict, force: bool = False) -> dict:
    """Run every method on one instance and write its records; returns a compact summary (also ``done.json``)."""
    from aimp.types import SolveResult

    t_start = time.perf_counter()
    inst = task_instance(part, task)
    iid = inst.meta["instance_id"]
    d = task_dir(out, part, inst)
    settings = run_settings(part, task, cfg)
    if not force:
        check_settings(d, settings)
    t_ex = settings["exact_time_limit"]
    inst.meta["T_exact"] = t_ex
    ctrl = bounds.controls(inst, threads=threads)
    if force or not (d / "instance.json").exists():
        save_instance(inst, ctrl, d)
        meta = json.loads((d / "instance.json").read_text(encoding="utf-8"))
        meta["controls"].update({"w_t": ctrl["w_t"], "w_t_loose": ctrl["w_t_loose"]})
        meta["task"] = {"part": part, **task}
        meta["run_settings"] = settings
        write_json_atomic(d / "instance.json", jsonable(meta))
    if part == "width" and (force or not (d / "phi_min.json").exists()):
        write_json_atomic(d / "phi_min.json", phi_min(inst, ctrl, int(inst.meta["seed"])))
    recs, ctx, errors = {}, {}, []
    for name, (fn, bspec) in method_table(part, inst, ctrl, threads, cfg, t_ex).items():
        path = d / f"{name}.json"
        if path.exists() and not force:
            recs[name] = json.loads(path.read_text(encoding="utf-8"))
            if name == "multistart" or recs[name].get("_recovered_point") is not None:
                ctx[name] = _ctx_from_rec(recs[name])
            continue
        c0 = time.process_time()
        try:
            res = fn(ctx)
        except Exception as e:  # recorded, not fatal
            res = SolveResult(method=name, status="error", value=float("nan"),
                              diagnostics={"message": repr(e), "traceback": traceback.format_exc()[-2000:]})
            errors.append(f"{name}: {e!r}")
        if res.cpu_time_s is None:
            res.cpu_time_s = time.process_time() - c0
        res.method = name
        ctx[name] = res
        bused, bval = _bound_for(bspec, ctrl, inst.n)
        rec = result_record(res, inst, ctrl, None, bused, bval, _solver_of(name, part))
        if res.recovered_point is not None:
            rec["_recovered_point"] = res.recovered_point
        rec["params"]["threads_per_solve"] = threads
        write_json_atomic(path, rec)
        recs[name] = json.loads(path.read_text(encoding="utf-8"))
    # bracket of p_star: certified upper bound from the exact solve, verified lower bound from every feasible point
    ex = recs.get("exact", {})
    ub = ex.get("value") if (ex.get("status") in ("optimal", "time_limit", "suboptimal", "resolved_stop")
                             and _finite(ex.get("value"))) else None
    lbs = [r.get(k) for r in recs.values() for k in ("value_lb_verified", "recovered_value") if _finite(r.get(k))]
    lb = max(lbs) if lbs else None
    Z_ref = None
    for r in recs.values():  # the verified feasible point whose value is p_ref
        if Z_ref is None and r.get("_recovered_point") is not None and lb is not None \
                and lb in (r.get("value_lb_verified"), r.get("recovered_value")):
            Z_ref = np.asarray(r["_recovered_point"], float)
    flags = {"validity_fail": [], "validity_fail_solver_tolerance": [], "bound_violation": [],
             "bound_violation_solver_tolerance": []}
    for name, rec in recs.items():
        rec["p_star_ub"], rec["p_star_lb_verified"] = ub, lb
        derive_fields(rec)
        if rec.get("validity_ok") is False and name.startswith("sdp") and Z_ref is not None:
            if "validity_attribution" not in rec:
                rec["validity_attribution"] = _attribute(inst, ctrl, rec, Z_ref, cfg)
            kind = rec["validity_attribution"]["kind"]
            flags["validity_fail_solver_tolerance" if kind == "solver_tolerance" else "validity_fail"].append(name)
        elif rec.get("validity_ok") is False:
            flags["validity_fail"].append(name)
        if rec.get("bound_satisfied") is False:
            att = _bound_attribution(rec)
            rec["bound_attribution"] = {"bound_satisfied": att}
            flags["bound_violation" if att["kind"] == "unattributed" else "bound_violation_solver_tolerance"].append(name)
        write_json_atomic(d / f"{name}.json", rec)
    summary = {"instance_id": iid, "part": part, "task": task, "wall_s": time.perf_counter() - t_start,
               "errors": errors, "Delta": ctrl["Delta"], "w_t": ctrl["w_t"], "rad": ctrl["rad"], "gamma": ctrl["gamma"],
               "p_ref": lb, "p_star_ub": ub, "exact_status": ex.get("status"), "flags": flags,
               "run_settings": settings, "methods": cfg["methods"], "gaps": {k: r.get("gap") for k, r in recs.items()}}
    write_json_atomic(d / "done.json", summary)
    return summary


# ----------------------------------------------------------------------------- pool
def _worker(args):
    part, task, out, threads, cfg, force = args
    try:
        return run_task(part, task, Path(out), threads, cfg, force)
    except Exception as e:
        return {"part": part, "task": task, "fatal": repr(e), "traceback": traceback.format_exc()[-3000:]}


def _log(out: Path, part: str, msg: str) -> None:
    path = Path(out) / part / "sweep.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_part(part: str, out: Path, cfg: dict, workers: int, threads: int, quick: bool = False, force: bool = False,
             instances: list[str] | None = None, limit: int | None = None) -> dict:
    """Run one part (``slopes``, ``width``, ``temperature``) on a spawn pool of ``workers`` processes."""
    budget = compute.thread_budget()
    if workers * threads > budget:
        raise SystemExit(f"workers x threads-per-solve = {workers * threads} > AIMP_MAX_THREADS = {budget}")
    tasks = build_tasks(part, quick, out, threads)
    todo = []
    for t in tasks:
        inst = task_instance(part, t)
        if instances and inst.meta["instance_id"] not in instances:
            continue
        d = task_dir(out, part, inst)
        if not force:
            check_settings(d, run_settings(part, t, cfg))
            done = d / "done.json"
            if done.exists() and (cfg["methods"] == "core"
                                  or json.loads(done.read_text(encoding="utf-8")).get("methods", "all") == "all"):
                continue
        todo.append(t)
    if limit is not None:
        todo = todo[:limit]
    _log(out, part, f"=== {part}: {len(tasks)} tasks, {len(todo)} to run, workers={workers}, threads_per_solve={threads}, "
         f"sdp={cfg['sdp']}, methods={cfg['methods']}, exact_time_limit={cfg['exact_time_limit']}; env {env_info()}")
    t0, n_done, n_fatal = time.time(), 0, 0
    if todo:
        with mp.get_context("spawn").Pool(workers, initializer=compute.apply_thread_budget, initargs=(threads,),
                                          maxtasksperchild=25) as pool:
            args = [(part, t, str(out), threads, cfg, force) for t in todo]
            for res in pool.imap_unordered(_worker, args, chunksize=1):
                n_done += 1
                if "fatal" in res:
                    n_fatal += 1
                    _log(out, part, f"!!! FATAL {res['task']}: {res['fatal']}\n{res['traceback']}")
                    continue
                g = res["gaps"]

                def f(k):
                    v = g.get(k)
                    return "-" if v is None else f"{v:.3g}"

                fl = res["flags"]
                _log(out, part, f"[{n_done}/{len(todo)}] {res['instance_id']} {res['wall_s']:.1f}s "
                     f"exact={res['exact_status']} w_t={res['w_t']:.3g} gamma={res['gamma']:.3g} "
                     f"gaps ibp_obj={f('ibp_obj')} ps={f('ps_auto')} sdp_1={f('sdp_1')} ex={f('exact')}"
                     + (f" ERR {res['errors']}" if res["errors"] else "")
                     + (f" !!! {fl}" if fl["validity_fail"] or fl["bound_violation"] else ""))
    _log(out, part, f"=== {part} finished: ran {n_done} ({n_fatal} fatal) in {time.time() - t0:.0f}s")
    return {"ran": n_done, "fatal": n_fatal}


# ============================================================================= summary
GAP_FLOOR = 1e-7  # gaps below this are solver noise (not used in fits)
RES_FRAC = 0.1  # resolved: res <= 0.1 * gap
SMALL_RHO = (0.01, 0.1)  # the range of the slope fits (Table 3)
WITHIN = 1.2  # "within a factor 1.2 of the best of all methods"
DOMINANT = 0.9  # a key is dominant when its certified attention weight is at least this
NOT_RELAXATIONS = {"exact", "exact_grid", "multistart"}
SKIP_FILES = {"instance.json", "done.json", "score_intervals.json", "phi_min.json", "router.json"}
TABLE_METHODS = [("ibp_obj", "interval pattern bounds"), ("ps_auto", "first order, R_pp"),
                 ("sdp_1_unitdiag", "level one, unit diagonal"), ("sdp_1", "level one, R_lift")]
TABLE_FAMILIES = [("F_random", "random"), ("F_ambiguous", "ambiguous"), ("F_margin", "margin"), ("F_rank1", "rank one")]
FIG_METHODS = ["ibp_obj", "ps_auto", "sdp_1_unitdiag", "sdp_1", "sdp_2", "sdp_3_cut"]
FAMILY_ORDER = ["F_prop9", "F_random", "F_ambiguous", "F_margin", "F_rank1"]
PAPER = {  # the values printed in the paper (balls, n = 2, rho <= 0.1)
    "table1": {("ibp_obj", "F_random"): "1.02 [1.00,1.06]", ("ibp_obj", "F_ambiguous"): "1.01 [0.98,1.02]",
               ("ibp_obj", "F_margin"): "1.04 [0.96,1.09]", ("ibp_obj", "F_rank1"): "1.01 [0.98,1.06]",
               ("ps_auto", "F_random"): "1.00 [0.94,1.01]", ("ps_auto", "F_ambiguous"): "0.99 [0.98,0.99]",
               ("ps_auto", "F_margin"): "1.02 [0.86,1.12]", ("ps_auto", "F_rank1"): "1.00 [0.96,1.03]",
               ("sdp_1_unitdiag", "F_random"): "1.10 [1.03,1.33]", ("sdp_1_unitdiag", "F_ambiguous"): "1.35 [1.24,1.54]",
               ("sdp_1_unitdiag", "F_margin"): "1.28 [1.18,1.44]", ("sdp_1_unitdiag", "F_rank1"): "1.14 [1.03,1.41]",
               ("sdp_1", "F_random"): "2.01 [1.93,2.01]", ("sdp_1", "F_ambiguous"): "2.00 [1.99,2.01]",
               ("sdp_1", "F_margin"): "2.02 [1.80,2.05]", ("sdp_1", "F_rank1"): "2.00 [2.00,2.02]"},
    "width_ratio": {0.05: 125.0, 1.0: 6.0, 8.0: 1.6},
    "temperature_order": "1.97 to 2.03 (random heads, sdp_1)",
    "interval_pattern_ratio": "9.5 to 28, depending on the family and the number of keys",
    "min_of_two": "median cost 0.2 s per synthetic instance",
    "dominant_keys": "half of the seeds have a dominant key at every width; its certified weight exceeds 0.9 on all "
                     "five from w_t = 12",
}


def load_records(root: Path) -> list[dict]:
    """One row per (instance, method) under ``root/<family>/<instance_id>/`` (finished instances only). The derived
    fields (``p_ref``, ``gap``, ``res``, ``validity_ok``, ...) are recomputed from the stored values."""
    rows = []
    for done in sorted(Path(root).glob("*/*/done.json")):
        d = done.parent
        if d.parent.name.startswith("_"):
            continue
        try:
            inst = json.loads((d / "instance.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        ctrl = inst.get("controls") or {}
        pm = {}
        if (d / "phi_min.json").exists():
            pm = json.loads((d / "phi_min.json").read_text(encoding="utf-8"))
        for p in sorted(d.glob("*.json")):
            if p.name in SKIP_FILES:
                continue
            try:
                r = derive_fields(json.loads(p.read_text(encoding="utf-8")))
            except ValueError:
                continue
            rows.append({
                "instance_id": d.name, "part": inst.get("part"), "family": inst.get("family"),
                "config": inst.get("config"), "seed": inst.get("seed"), "n": inst.get("n"),
                "set_type": inst.get("set_type"), "query_in_keys": inst.get("query_in_keys"), "rho": inst.get("rho"),
                "temp": inst.get("temp"), "kappa": inst.get("kappa"), "w_target": inst.get("w_target"),
                "gamma": ctrl.get("gamma"), "w_t": ctrl.get("w_t"), "phi_min_ms": pm.get("phi_min_ms"),
                "a_lo_max": max(ctrl["a_lo"]) if ctrl.get("a_lo") else None,
                "method": p.stem, "status": r.get("status"), "value": r.get("value"), "p_ref": r.get("p_ref"),
                "p_star_ub": r.get("p_star_ub"), "res": r.get("res"), "gap": r.get("gap"),
                "validity_ok": r.get("validity_ok"), "bound_satisfied": r.get("bound_satisfied"),
                "attribution": (r.get("validity_attribution") or {}).get("kind"),
                "bound_attribution": ((r.get("bound_attribution") or {}).get("bound_satisfied") or {}).get("kind"),
                "wall_time_s": r.get("wall_time_s"),
            })
    return rows


def _frame(rows: list[dict]):
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("value", "p_ref", "p_star_ub", "res", "gap", "rho", "temp", "kappa", "w_target", "gamma", "w_t",
              "phi_min_ms", "a_lo_max", "wall_time_s"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["series"] = (df["family"].astype(str) + "|" + df["config"].astype(str) + "|s" + df["seed"].astype(str) + "|n"
                    + df["n"].astype(str) + "|" + df["set_type"].astype(str) + "|q" + df["query_in_keys"].astype(str))
    return df


def wide(df, col: str = "gap"):
    """One row per instance, one column per relaxation (``col``), plus the instance attributes and the bracket ``res``
    of the exact solve."""
    attrs = ["family", "config", "seed", "n", "set_type", "query_in_keys", "rho", "gamma", "series"]
    W = df.drop_duplicates("instance_id").set_index("instance_id")[attrs]
    ex = df[df["method"] == "exact"].set_index("instance_id")
    W["p_ref"] = df.groupby("instance_id")["p_ref"].first()
    W["res"] = ex["res"]
    piv = df[~df["method"].isin(NOT_RELAXATIONS)].pivot_table(index="instance_id", columns="method", values=col,
                                                              aggfunc="first")
    return W.join(piv)


def fit_slope(x, y) -> float | None:
    """Least-squares slope of ``log y`` against ``log x`` over the points with ``y > GAP_FLOOR``; ``None`` with fewer
    than three such points or when they span less than a factor four in ``x``."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(y) & (y > GAP_FLOOR)
    if ok.sum() < 3 or np.ptp(np.log(x[ok])) < np.log(4):
        return None
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


def slope_table(W, methods, rho_range=SMALL_RHO):
    """Per family x set type x method: median and interquartile range of the per-series slopes of ``log gap`` against
    ``log rho`` on ``rho_range`` (a series = family, configuration, seed, n, set type, query mode), on the instances
    resolved for that method (``rule = per_method``, Table 3) or for all methods of the table together (``common``)."""
    import pandas as pd

    rows = []
    Wd = W[(W["rho"] >= rho_range[0] - 1e-12) & (W["rho"] <= rho_range[1] + 1e-12)]
    methods = [m for m in methods if m in W]
    common = (Wd["res"] <= RES_FRAC * Wd[methods].min(axis=1)) & Wd["res"].notna()
    for (fam, st), G in Wd.groupby(["family", "set_type"]):
        for m in methods:
            for rule in ("per_method", "common"):
                ok = (G["res"] <= RES_FRAC * G[m]) if rule == "per_method" else common.loc[G.index]
                H = G[ok & G[m].notna()]
                sl = [s for _, S_ in H.groupby("series") if (s := fit_slope(S_["rho"], S_[m])) is not None]
                med = H.groupby("rho")[m].median()
                rows.append({"family": fam, "set_type": st, "method": m, "rule": rule,
                             "n_instances_resolved": int(len(H)), "n_instances_total": int(G[m].notna().sum()),
                             "n_series_fitted": len(sl),
                             "slope_median": float(np.median(sl)) if sl else None,
                             "slope_q25": float(np.quantile(sl, 0.25)) if sl else None,
                             "slope_q75": float(np.quantile(sl, 0.75)) if sl else None,
                             "slope_of_medians": fit_slope(med.index.values, med.values) if len(med) >= 3 else None})
    return pd.DataFrame(rows)


def gap_vs_rho(W):
    """The data of Figure 1: per family x set type x method x rho, quartiles of the gap over the resolved instances
    (gap above the noise floor) and their number (columns of ``figdata/P1_gap_vs_rho.csv``)."""
    import pandas as pd

    data = []
    fams = [f for f in FAMILY_ORDER if f in set(W["family"])]
    for st in [s for s in ("ball", "box") if s in set(W["set_type"])]:
        for fam in fams:
            G = W[(W["family"] == fam) & (W["set_type"] == st)]
            for m in [m for m in FIG_METHODS if m in W]:
                H = G[(G["res"] <= RES_FRAC * G[m]) & (G[m] > GAP_FLOOR)]
                if H.empty:
                    continue
                q = H.groupby("rho")[m].quantile([0.25, 0.5, 0.75]).unstack()
                cnt = H.groupby("rho")[m].count()
                for rho, row in q.iterrows():
                    data.append({"family": fam, "set_type": st, "method": m, "rho": rho, "q25": row[0.25],
                                 "median": row[0.5], "q75": row[0.75], "n_resolved": int(cnt[rho])})
    return pd.DataFrame(data, columns=["family", "set_type", "method", "rho", "q25", "median", "q75", "n_resolved"])


def interval_pattern_ratio(W):
    """Per (n, family): median of ``gap(ibp_obj) / gap(sdp_1)`` over the instances resolved for ``sdp_1`` (its gap
    above the noise floor)."""
    import pandas as pd

    rows = []
    if not {"ibp_obj", "sdp_1"} <= set(W.columns):
        return pd.DataFrame(rows)
    D = W[~W["family"].isin(["F_prop9", "F_cov", "F_cov_anti"])]
    for (n, fam), G in D.groupby(["n", "family"]):
        ok = G["ibp_obj"].notna() & G["sdp_1"].notna() & (G["sdp_1"] > GAP_FLOOR) & (G["res"] <= RES_FRAC * G["sdp_1"])
        r = (G["ibp_obj"] / G["sdp_1"])[ok]
        if len(r):
            rows.append({"n": int(n), "family": fam, "count": int(len(r)), "median": float(r.median()),
                         "q25": float(r.quantile(0.25)), "q75": float(r.quantile(0.75))})
    return pd.DataFrame(rows)


def min_of_two(df):
    """Per instance: the certificate ``min(first order, level one)`` (``ps_auto``, ``sdp_1``) against the smallest valid
    gap of all relaxations on the instance (value not below ``p_ref`` beyond ``1e-6 max(1, |p_ref|)``); ratio of the two
    gaps with both floored at ``GAP_FLOOR``; cost = solve time of the two certificates."""
    import pandas as pd

    R = df[~df["method"].isin(NOT_RELAXATIONS) & df["gap"].notna()]
    rows = []
    for iid, G in R.groupby("instance_id"):
        g = G.set_index("method")
        p_ref = float(g["p_ref"].iloc[0])
        tol = 1e-6 * max(1.0, abs(p_ref))
        valid = g[(g["validity_ok"] != False) & (g["gap"] >= -tol)]  # noqa: E712
        two = [float(g.loc[m, "gap"]) for m in ("ps_auto", "sdp_1") if m in g.index]
        if valid.empty or not two:
            continue
        best, m2 = float(valid["gap"].min()), min(two)
        cost = sum(float(g.loc[m, "wall_time_s"]) for m in ("ps_auto", "sdp_1")
                   if m in g.index and np.isfinite(g.loc[m, "wall_time_s"]))
        rows.append({"instance_id": iid, "family": g["family"].iloc[0], "n": int(g["n"].iloc[0]),
                     "best_method": valid["gap"].idxmin(), "best_gap": best, "min_of_two_gap": m2,
                     "ratio": max(m2, GAP_FLOOR) / max(best, GAP_FLOOR), "cost_s": cost})
    return pd.DataFrame(rows)


def width_table(df):
    """Per ``w_t`` target: medians over the seeds of the relative gap ``gap / (p_ref - phi_min)`` of each method, the
    ratios of the medians to the level-one certificate's, and the share of seeds with a dominant key: a key whose
    certified attention weight is at least ``DOMINANT`` (``max_j a_lo_j >= 0.9``). The share with a positive certified
    margin ``gamma`` is reported too; it does not depend on the temperature, since the scores and their certified
    intervals are all linear in ``M``."""
    import pandas as pd

    A = df[df["part"] == "2A"].copy()
    if A.empty:
        return pd.DataFrame()
    A["rel_gap"] = A["gap"] / (A["p_ref"] - A["phi_min_ms"])
    A["rel_res"] = A["res"] / (A["p_ref"] - A["phi_min_ms"])
    rows = []
    for wt, G in A.groupby("w_target"):
        inst = G.drop_duplicates("instance_id")
        row = {"w_target": wt, "n": int(len(inst)), "w_t_median": float(inst["w_t"].median()),
               "rel_res_median": float(inst["rel_res"].median()),
               "dominant_key_share": float((inst["a_lo_max"] >= DOMINANT).mean()),
               "margin_share": float((inst["gamma"] > 0).mean())}
        for m in ("ibp_obj", "ps_auto", "sdp_1"):
            v = G[G["method"] == m]["rel_gap"].dropna()
            row[f"{m}_median"] = float(v.median()) if len(v) else None
        s1 = row.get("sdp_1_median")
        row["ratio_ps_auto_over_sdp_1"] = row["ps_auto_median"] / s1 if s1 else None
        row["ratio_ibp_obj_over_sdp_1"] = row["ibp_obj_median"] / s1 if s1 else None
        rows.append(row)
    return pd.DataFrame(rows)


def temperature_table(df):
    """Per family x rho x method: per-seed least-squares slopes of ``log gap`` against ``log kappa`` (resolved points,
    gap above the noise floor, at least three points spanning a factor four), median and interquartile range, and the
    slope of the per-kappa medians."""
    import pandas as pd

    B = df[(df["part"] == "2B") & df["method"].isin(["ps_auto", "sdp_0", "sdp_1"])].copy()
    if B.empty:
        return pd.DataFrame()
    B["resolved"] = B["res"] <= RES_FRAC * B["gap"]
    rows = []
    for (fam, rho, m), G in B.groupby(["family", "rho", "method"]):
        slopes = []
        for _, S_ in G.groupby("seed"):
            S2 = S_[S_["resolved"].fillna(False) & (S_["gap"] > GAP_FLOOR)]
            if len(S2) >= 3 and np.ptp(np.log(S2["kappa"])) >= np.log(4):
                slopes.append(float(np.polyfit(np.log(S2["kappa"]), np.log(S2["gap"]), 1)[0]))
        med = G.groupby("kappa")["gap"].median()
        okm = med > GAP_FLOOR
        rows.append({"family": fam, "rho": rho, "method": m, "n_series_fitted": len(slopes),
                     "order_median": float(np.median(slopes)) if slopes else None,
                     "order_q25": float(np.quantile(slopes, 0.25)) if slopes else None,
                     "order_q75": float(np.quantile(slopes, 0.75)) if slopes else None,
                     "order_of_medians": float(np.polyfit(np.log(med.index[okm]), np.log(med[okm]), 1)[0])
                     if okm.sum() >= 3 else None, "n": int(len(G))})
    return pd.DataFrame(rows)


def checks_table(df) -> dict:
    """Validity (``value >= p_ref - 1e-6 max(1, |p_ref|)`` for every relaxation) and the first-order gap bound
    (``value - p_ref <= bound + 1e-6`` for the first-order certificates and every SDP with the products (R5))."""
    R = df[~df["method"].isin(NOT_RELAXATIONS)]
    v = R[R["validity_ok"].notna()]
    fail = v[v["validity_ok"] == False]  # noqa: E712
    b = R[R["bound_satisfied"].notna()]
    ex = df[df["method"] == "exact"]
    return {"instances": int(df["instance_id"].nunique()), "validity_checks": int(len(v)),
            "validity_failures": int(len(fail)),
            "validity_failures_solver_tolerance": int((fail["attribution"] == "solver_tolerance").sum()),
            "bound_checks": int(len(b)), "bound_violations": int((b["bound_satisfied"] == False).sum()),  # noqa: E712
            "bound_violations_solver_tolerance": int(((b["bound_satisfied"] == False)  # noqa: E712
                                                      & (b["bound_attribution"] == "solver_tolerance_duality_gap")).sum()),
            "exact_status": {str(k): int(c) for k, c in ex["status"].value_counts().items()},
            "solver_errors": int((R["status"] == "error").sum())}


def _fmt(row) -> str:
    if row is None or row.get("slope_median") is None or not np.isfinite(row["slope_median"]):
        return "-"
    return f"{row['slope_median']:.2f} [{row['slope_q25']:.2f},{row['slope_q75']:.2f}]"


def summarize(out: Path, write: bool = True) -> dict:
    """Compute and print every synthetic number of the paper from the records under ``out``."""
    import pandas as pd

    out = Path(out)
    sdir = out / "summary"
    result = {}
    print(f"\n=== summary of {out}")
    slopes = _frame(load_records(out / "slopes"))
    if not slopes.empty:
        result["slopes_checks"] = checks_table(slopes)
        n2 = slopes[pd.to_numeric(slopes["n"]) == 2]
        per_n = slopes.drop_duplicates("instance_id")["n"].value_counts().sort_index()
        print(f"slopes: {slopes['instance_id'].nunique()} instances "
              f"({', '.join(f'n = {n}: {c}' for n, c in per_n.items())}); checks {result['slopes_checks']}")
        if not n2.empty:
            W = wide(n2)
            T = slope_table(W, [m for m, _ in TABLE_METHODS] + ["sdp_2", "sdp_3_cut", "sdp_3_ball", "sdp_3_bxb"])
            F = gap_vs_rho(W)
            result["table1"], result["figure1"] = T, F
            for st in ("ball", "box"):
                Tp = T[(T["rule"] == "per_method") & (T["set_type"] == st)]
                if Tp.empty:
                    continue
                print(f"\nTable 3{'' if st == 'ball' else ' (boxes)'}: slope of the gap in rho, "
                      f"{'balls' if st == 'ball' else 'boxes'}, rho <= 0.1, n = 2 "
                      "(median [IQR] of per-series fits; series fitted)"
                      + ("; paper below each row" if st == "ball" else ""))
                print(f"{'':28s}" + "".join(f"{lab:>24s}" for _, lab in TABLE_FAMILIES))
                for m, lab in TABLE_METHODS:
                    cells, paper = [], []
                    for fam, _ in TABLE_FAMILIES:
                        r = Tp[(Tp["method"] == m) & (Tp["family"] == fam)]
                        r = r.iloc[0].to_dict() if len(r) else None
                        cells.append(f"{_fmt(r)} ({r['n_series_fitted'] if r else 0})")
                        paper.append(PAPER["table1"].get((m, fam), "-"))
                    print(f"{lab:28s}" + "".join(f"{c:>24s}" for c in cells))
                    if st == "ball":
                        print(f"{'  paper':28s}" + "".join(f"{c:>24s}" for c in paper))
            Tb = T[(T["rule"] == "per_method") & T["family"].isin([f for f, _ in TABLE_FAMILIES])]
            for m, lab in TABLE_METHODS:
                v = Tb[Tb["method"] == m]["slope_median"].dropna()
                if len(v):
                    print(f"  range of family medians, {lab}: {v.min():.2f} to {v.max():.2f} (balls and boxes)")
            print(f"Figure 1 data: {len(F)} rows (family x set type x method x rho)")
        RT = interval_pattern_ratio(wide(slopes))
        result["interval_pattern_ratio"] = RT
        if not RT.empty:
            print("\ninterval pattern bounds / level one (median gap ratio, resolved instances; paper: "
                  f"{PAPER['interval_pattern_ratio']}):")
            for n, G in RT.groupby("n"):
                print(f"  n = {n}: " + ", ".join(f"{r.family} {r.median:.1f}" for r in G.itertuples())
                      + f"  -> {G['median'].min():.1f} to {G['median'].max():.1f}")
        M2 = min_of_two(slopes)
        result["min_of_two"] = M2
        if not M2.empty:
            frac = float((M2["ratio"] <= WITHIN).mean())
            result["min_of_two_fraction"] = frac
            print(f"\nmin(first order, level one): median cost {M2['cost_s'].median():.2f} s per instance over "
                  f"{len(M2)} instances (paper: {PAPER['min_of_two']}); diagnostic, not in the paper: within {WITHIN} "
                  f"of the best of the methods run here on {100 * frac:.1f}%")
            print("  best method counts: " + ", ".join(f"{k} {v}" for k, v in M2["best_method"].value_counts().items()))
    width = _frame(load_records(out / "width"))
    WT = width_table(width) if not width.empty else pd.DataFrame()
    result["width"] = WT
    if not WT.empty:
        print("\nsecant-width grid (F_random, n = 3, balls, rho = 0.1; medians of relative gaps over the seeds):")
        print(WT[["w_target", "n", "w_t_median", "ibp_obj_median", "ps_auto_median", "sdp_1_median",
                  "ratio_ps_auto_over_sdp_1", "ratio_ibp_obj_over_sdp_1", "dominant_key_share",
                  "margin_share"]].to_string(
            index=False, float_format=lambda x: f"{x:.3g}"))
        for wt, ref in PAPER["width_ratio"].items():
            r = WT[np.isclose(WT["w_target"], wt)]
            if len(r):
                print(f"  w_t = {wt:g}: first order / level one = {r['ratio_ps_auto_over_sdp_1'].iloc[0]:.3g} "
                      f"(paper: {ref:g})")
        dom = WT[WT["dominant_key_share"] >= WT["margin_share"] - 1e-12]
        print(f"  seeds with a dominant key (positive certified margin): {WT['margin_share'].min():.0%} to "
              f"{WT['margin_share'].max():.0%} of the seeds over the widths")
        if len(dom) and WT["margin_share"].max() > 0:
            print(f"  its certified weight is >= {DOMINANT:g} on all of these seeds from w_t = {dom['w_target'].min():g} "
                  f"(paper: {PAPER['dominant_keys']})")
        big = WT[WT["w_target"] >= 20]
        if len(big):
            print("  first order at w_t >= 20 (median relative gap): " + ", ".join(
                f"{r.w_target:g}: {r.ps_auto_median:.2g}" for r in big.itertuples()))
        result["width_checks"] = checks_table(width)
        print(f"  checks {result['width_checks']}")
    temp = _frame(load_records(out / "temperature"))
    TT = temperature_table(temp) if not temp.empty else pd.DataFrame()
    result["temperature"] = TT
    if not TT.empty:
        print("\ntemperature order of the gap at fixed rho (per-seed fits, median [IQR]):")
        print(TT.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
        r = TT[(TT["family"] == "F_random") & (TT["method"] == "sdp_1")]["order_median"].dropna()
        if len(r):
            print(f"  random heads, level one: order {r.min():.2f} to {r.max():.2f} (paper: {PAPER['temperature_order']})")
        result["temperature_checks"] = checks_table(temp)
        print(f"  checks {result['temperature_checks']}")
    if write:
        sdir.mkdir(parents=True, exist_ok=True)
        for name, key in (("table3_slopes.csv", "table1"), ("P1_gap_vs_rho.csv", "figure1"),
                          ("interval_pattern_ratio.csv", "interval_pattern_ratio"), ("min_of_two.csv", "min_of_two"),
                          ("width.csv", "width"), ("temperature_orders.csv", "temperature")):
            D = result.get(key)
            if D is not None and not D.empty:
                tmp = sdir / (name + ".tmp")
                D.to_csv(tmp, index=False, lineterminator="\n")
                tmp.replace(sdir / name)
        write_json_atomic(sdir / "checks.json", {k: v for k, v in result.items() if k.endswith("checks")
                                                 or k == "min_of_two_fraction"})
        print(f"\nwrote {sdir}")
    return result


# ============================================================================= entry point
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--part", nargs="+", default=["all"], choices=list(PARTS) + ["summary", "all"],
                    help="parts to run (default: all = slopes width temperature summary)")
    ap.add_argument("--quick", action="store_true",
                    help="reduced grids (slopes: n = 2, balls, F_random and F_ambiguous, 3 seeds, rho <= 0.1; width: "
                         "w_t in {0.05, 1, 8}; temperature: F_random, rho = 0.05, 3 seeds) with a 20 s exact time limit")
    ap.add_argument("--workers", type=int, default=None, help="worker processes (default AIMP_MAX_THREADS // "
                                                               "threads-per-solve)")
    ap.add_argument("--threads-per-solve", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None,
                    help="records directory (default <results>/synthetic, or <results>/synthetic_quick with --quick)")
    ap.add_argument("--t-interval", choices=("tight", "loose"), default="tight",
                    help="log-attention intervals of the SDPs: tight (Definition 3, default) or loose (with the solver "
                         "settings of the stored loose records)")
    ap.add_argument("--exact-time-limit", type=float, default=None,
                    help="time limit of every exact solve in seconds (default: 300 s for rho <= 0.2 and 90 s above at "
                         "n = 2, 300 s at n = 3, 120 s at n = 4 and in the temperature part, 300 s in the width part; "
                         f"{QUICK_EXACT_TIME_LIMIT:g} s with --quick)")
    ap.add_argument("--methods", choices=("all", "core"), default="all",
                    help="slopes part: all methods, or only those of Table 3 (+ multistart, exact)")
    ap.add_argument("--instances", nargs="+", default=None, help="run only these instance ids")
    ap.add_argument("--limit", type=int, default=None, help="run at most this many instances per part")
    ap.add_argument("--force", action="store_true", help="recompute existing records")
    a = ap.parse_args(argv)
    parts = list(PARTS) + ["summary"] if "all" in a.part else a.part
    out = a.out if a.out is not None else results_dir() / ("synthetic_quick" if a.quick else "synthetic")
    workers = a.workers if a.workers is not None else max(1, compute.thread_budget() // a.threads_per_solve)
    t_ex = a.exact_time_limit if a.exact_time_limit is not None else (QUICK_EXACT_TIME_LIMIT if a.quick else None)
    cfg = {"sdp": sdp_settings(a.t_interval), "methods": a.methods, "exact_time_limit": t_ex}
    t0 = time.time()
    for part in parts:
        if part in PARTS:
            run_part(part, out, cfg, workers, a.threads_per_solve, quick=a.quick, force=a.force,
                     instances=a.instances, limit=a.limit)
    if "summary" in parts:
        summarize(out)
    print(f"total wall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
