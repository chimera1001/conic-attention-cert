"""Result I/O: JSON (atomic), npz instance arrays, provenance (git, versions, machine)."""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import platform
import subprocess
from pathlib import Path

import numpy as np

from aimp.types import Instance, SolveResult

ROOT = Path(__file__).resolve().parents[1]


def results_dir() -> Path:
    """Where the experiments write: ``$AIMP_RESULTS_DIR``, default ``<repository>/results``."""
    return Path(os.environ.get("AIMP_RESULTS_DIR", str(ROOT / "results"))).expanduser()


def cache_dir() -> Path:
    """Cached GPT-2 weights and activations: ``$AIMP_CACHE_DIR``, default ``<repository>/cache``."""
    return Path(os.environ.get("AIMP_CACHE_DIR", str(ROOT / "cache"))).expanduser()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        x = float(x)
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=1), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def git_commit() -> str:
    """Short commit hash of this repository (``-dirty`` if ``aimp/`` has uncommitted changes); ``"unknown"`` outside a
    git checkout of this repository (for example when it sits inside another repository's tree)."""
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.strip()
        if Path(top).resolve() != ROOT.resolve():
            return "unknown"
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "aimp"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def processor_name() -> str:
    """CPU model: ``platform.processor()`` (Windows), else ``model name`` from /proc/cpuinfo (Linux, where it is '')."""
    name = platform.processor()
    if name and name not in ("x86_64", "i386", "unknown"):
        return name
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return name


def env_info() -> dict:
    from aimp import compute
    from aimp.solvers import cvxpy_version, gurobi_version, mosek_version

    return {
        "node": platform.node(), "os": platform.platform(), "processor": processor_name(),
        "python": platform.python_version(), "numpy": np.__version__,
        "gurobi": gurobi_version(), "mosek": mosek_version(), "cvxpy": cvxpy_version(),
        "thread_budget": compute.thread_budget(),
    }


def peak_rss_gb() -> float | None:
    """Peak resident set size of this process (GB): ``peak_wset`` on Windows, ``ru_maxrss`` on POSIX."""
    try:
        import psutil

        peak = getattr(psutil.Process().memory_info(), "peak_wset", None)
        if peak:
            return peak / 2**30
    except Exception:
        pass
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 2**30  # Linux: KiB
    except Exception:
        return None


def save_instance(inst: Instance, ctrl: dict, directory: Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {"M": inst.head.M, "N": inst.head.N, "c": inst.c, "zbar": inst.centers()}
    for t, s in inst.sets.items():
        for k in ("l", "u", "E", "points"):
            v = getattr(s, k)
            if v is not None:
                arrays[f"{k}_{t}"] = v
    np.savez(directory / "instance.npz", **arrays)
    from aimp.heads import nominal

    nom = nominal(inst)
    si = ctrl["score_intervals"]
    obj = {
        **inst.meta, "keys": inst.keys, "query": inst.query,
        "set_kinds": {t: s.kind for t, s in inst.sets.items()},
        "radii": {t: s.rho for t, s in inst.sets.items()},
        "norms": {"M_op": float(np.linalg.norm(inst.head.M, 2)), "N_op": float(np.linalg.norm(inst.head.N, 2)),
                  "c_2": float(np.linalg.norm(inst.c))},
        "arrays_file": "instance.npz",
        "nominal_forward": nom,
        "controls": {
            "score_intervals": {"method": si["method"], "s_lo": si["s_lo"], "s_hi": si["s_hi"]},
            **{k: ctrl[k] for k in ("Delta", "rad", "rad_is_upper_bound", "R", "gamma", "top2_overlap",
                                    "ahat", "eps_B", "eps_C", "regime", "eps_target", "bound_B", "bound_C",
                                    "a_lo", "a_hi", "tau_lo", "tau_hi")},
        },
        "git_commit": git_commit(), "created": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    write_json_atomic(directory / "instance.json", obj)


def result_record(res: SolveResult, inst: Instance, ctrl: dict, pstar: dict | None = None,
                  bound_used: str | None = None, bound_value: float | None = None, solver: dict | None = None) -> dict:
    """``<method>.json`` result record of one method on one instance. ``pstar`` = {"ub":..., "lb_verified":...}."""
    rec = {
        "instance_id": inst.meta.get("instance_id"), "method": res.method,
        "rlt_families": res.params.get("rlt"), "solver": solver or {},
        "status": res.status, "value": res.value, "value_lb": res.value_lb,
        "value_lb_verified": res.value_lb_verified, "recovered_value": res.recovered_value,
        "approximations": res.approximations, "params": res.params, "wall_time_s": res.wall_time_s,
        "git_commit": git_commit(), "peak_rss_gb": peak_rss_gb(),
        "diagnostics": {k: v for k, v in res.diagnostics.items() if k != "X"},
    }
    rec["rank1_ratio"] = res.diagnostics.get("rank1_ratio")
    rec["top_eigs"] = res.diagnostics.get("top_eigs")
    rec["dual_value"] = res.diagnostics.get("dual_obj")
    if res.recovered_value is not None and res.value is not None:
        rec["gap_constructive"] = res.value - res.recovered_value
    rec["cpu_time_s"] = res.cpu_time_s
    rec["threads"] = threads_of(res.params)
    if bound_value is not None:
        rec["bound_used"] = bound_used
        rec["bound_value"] = bound_value
    if pstar is not None:
        rec["p_star_ub"], rec["p_star_lb_verified"] = pstar.get("ub"), pstar.get("lb_verified")
        derive_fields(rec)
    return rec


RESOLVED_FRACTION = 0.1  # an instance is resolved for a method if res <= 0.1 * gap
BOUND_TOL = 1e-6
LOWER_BOUND_METHODS = {"multistart"}  # values are feasible-point values (<= p_star), not relaxations


def threads_of(params: dict | None) -> int | None:
    """Thread count a solve was configured with (Gurobi ``Threads`` / MOSEK ``MSK_IPAR_NUM_THREADS``)."""
    params = params or {}
    for k in ("Threads", "MSK_IPAR_NUM_THREADS"):
        if isinstance(params.get(k), (int, float)):
            return int(params[k])
    return None


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def derive_fields(rec: dict) -> dict:
    """(Re)compute the derived fields of a method record from ``value``, ``p_star_ub``,
    ``p_star_lb_verified`` and ``bound_value``:

    * ``p_ref = p_star_lb_verified`` (numpy-verified incumbent), ``res = p_star_ub - p_ref``;
    * ``gap = value - p_ref`` (primary), ``gap_vs_pstar_ub = value - p_star_ub`` (secondary);
    * ``resolved = res <= 0.1 * gap`` (per method; analyses re-apply the rule with the smallest gap);
    * ``validity_ok = value >= p_ref - 1e-6 max(1, |p_ref|)`` (None if no finite value);
    * ``bound_satisfied`` — the STRONG check ``value - p_ref <= bound_value + 1e-6``;
      ``ratio_gap_over_bound = gap / bound_value``.
    Modifies and returns ``rec``."""
    v, ub, lb = rec.get("value"), rec.get("p_star_ub"), rec.get("p_star_lb_verified")
    v = v if _finite(v) else None
    ub = ub if _finite(ub) else None
    lb = lb if _finite(lb) else None
    rec["p_ref"] = lb
    rec["res"] = None if (ub is None or lb is None) else ub - lb
    rec["gap"] = None if (v is None or lb is None) else v - lb
    rec["gap_vs_pstar_ub"] = None if (v is None or ub is None) else v - ub
    rec["gap_vs_pstar_lb"] = rec["gap"]
    rec["resolved"] = (None if (rec["res"] is None or rec["gap"] is None)
                       else bool(rec["res"] <= RESOLVED_FRACTION * rec["gap"]))
    if lb is not None and rec.get("method") in LOWER_BOUND_METHODS:  # a lower bound: no validity check
        rec["validity_ok"] = None
        rec["lb_shortfall"] = None if v is None else lb - v
    elif lb is not None:
        rec["validity_ok"] = None if v is None else bool(v >= lb - 1e-6 * max(1.0, abs(lb)))
    bval = rec.get("bound_value")
    if _finite(bval) and rec["gap"] is not None:
        rec["ratio_gap_over_bound"] = rec["gap"] / bval if bval > 0 else None
        rec["bound_satisfied"] = bool(rec["gap"] <= bval + BOUND_TOL)
        rec["bound_check"] = "strong: value - p_ref <= bound + 1e-6"
    elif _finite(bval):
        rec["bound_satisfied"] = None
    return rec
