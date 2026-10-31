"""CPU thread budget for the whole package.

The budget is ``AIMP_MAX_THREADS`` (environment; default 4). Every solver and numerical
library takes its thread count from :func:`thread_budget` (or a smaller per-worker value
validated by :func:`resolve_threads`). This module is imported first by ``aimp/__init__.py``
so that the BLAS/OpenMP environment variables are set as early as possible.

Every solver call path calls :func:`record` with the thread count it actually configured;
``tests/test_threads.py`` inspects :data:`CALL_LOG` to assert the budget is respected.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")

#: (solver, threads) for every solve configured in this process, most recent last.
CALL_LOG: list[dict] = []
#: total number of recorded calls (monotone; CALL_LOG keeps only the most recent 1000).
CALL_COUNT = 0


def thread_budget() -> int:
    """Total CPU budget for all solver and library calls (``AIMP_MAX_THREADS``, default 4)."""
    return max(1, int(os.environ.get("AIMP_MAX_THREADS", "4")))


def resolve_threads(threads: Optional[int] = None) -> int:
    """Thread count for one solve: ``threads`` if given (must not exceed the budget), else the budget."""
    budget = thread_budget()
    n = budget if threads is None else int(threads)
    if n < 1 or n > budget:
        raise ValueError(f"threads={n} outside the budget 1..{budget} (AIMP_MAX_THREADS)")
    return n


def record(solver: str, threads: int) -> None:
    """Log that a solver call was configured with ``threads`` threads."""
    global CALL_COUNT
    CALL_LOG.append({"solver": solver, "threads": int(threads)})
    CALL_COUNT += 1
    del CALL_LOG[:-1000]


def calls_since(count: int) -> list[dict]:
    """Calls recorded after the moment ``CALL_COUNT`` was ``count`` (at most the last 1000)."""
    k = CALL_COUNT - count
    return CALL_LOG[-k:] if k > 0 else []


def apply_thread_budget(threads: Optional[int] = None) -> int:
    """Set OMP/MKL/OpenBLAS (env + threadpoolctl) and torch (if imported) to ``threads``."""
    n = resolve_threads(threads)
    for var in _ENV_VARS:
        os.environ[var] = str(n)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(n)
    except Exception:  # threadpoolctl missing or no pool loaded yet
        pass
    if "torch" in sys.modules:
        set_torch_threads(n)
    return n


def set_torch_threads(threads: Optional[int] = None) -> int:
    """Import torch and set its intra-op thread count (call before any torch work)."""
    import torch

    n = resolve_threads(threads)
    torch.set_num_threads(n)
    record("torch", n)
    return n


def configure_gurobi(model, threads: Optional[int] = None) -> int:
    """Set ``Threads`` on a gurobipy model from the budget; returns the value used."""
    n = resolve_threads(threads)
    model.Params.Threads = n
    record("gurobi", n)
    return n


def mosek_params(threads: Optional[int] = None, **extra) -> dict:
    """MOSEK parameter dict (for CVXPY ``mosek_params``) with the thread count and project tolerances."""
    n = resolve_threads(threads)
    params = {
        "MSK_IPAR_NUM_THREADS": n,
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-9,
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-9,
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-9,
    }
    params.update(extra)
    record("mosek", n)
    return params


apply_thread_budget()
