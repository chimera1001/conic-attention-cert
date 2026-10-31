"""The 50 seeded synthetic instances of the directional and second-order bound tests (d = 8, n = 3), with their
certificate and a 5 s Gurobi bracket, cached per test session so that tests/test_directional.py, tests/test_f0v.py and
tests/test_f0v_final.py share one exact solve per instance."""
from __future__ import annotations

import functools
import math

import numpy as np

from aimp import bounds, instances
from aimp.certify import certify
from aimp.exact_gurobi import solve_exact
from aimp.theory import nominal_decomposition

SEEDS = range(50)
EXACT_T = 5.0


def synthetic_instance(seed: int):
    rng = np.random.default_rng(10_000 + seed)
    rho = float(rng.choice([0.05, 0.1, 0.2, 0.4]))
    temp = float(rng.choice([0.5, 1.0, 2.0]))
    return instances.f_random(seed, d=8, d_k=4, d_v=4, n=3, temp=temp, rho=rho, set_type="ball",
                              query_in_keys=bool(seed % 2))


@functools.lru_cache(maxsize=None)
def synthetic_unit(seed: int) -> dict:
    """``inst``, ``ctrl``, ``nom``, the certificate record ``rec`` (``eps_target = 0``: the SDP always runs) and the
    Gurobi bracket ``lb`` (verified incumbent) / ``ub`` (ObjBound, ``None`` if not finite), ``exact_status``."""
    inst = synthetic_instance(seed)
    ctrl = bounds.controls(inst, threads=1)
    nom = nominal_decomposition(inst)
    rec = certify(inst, ctrl, eps_target=0.0, threads=1, nom=nom)
    ex = solve_exact(inst, ctrl, threads=1, time_limit=EXACT_T)
    lb, ub = ex.value_lb_verified, ex.value
    ub = ub if (ub is not None and math.isfinite(ub)) else None
    return {"inst": inst, "ctrl": ctrl, "nom": nom, "rec": rec, "lb": lb, "ub": ub, "exact_status": ex.status}
