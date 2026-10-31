"""Linear bound propagation baselines on one attention head: IBP, CROWN and alpha-CROWN through auto_LiRPA.

The comparison of Table 6 and Figure 4 (Section 7) gives the head to the bound propagation library on exactly the
footing of the certificate. The head is a torch module in the same reduced coordinates as the instance
(:func:`aimp.gpt2_units.make_instance`): every perturbed token ``t`` is a separate input ``u_t`` in its own l2 ball
``||u_t|| <= rho_t``, and the module computes ``z_t = c_t + E_t u_t``, the scores ``s_k = (z_i^T M) z_k``, the pattern
``a = softmax(s)`` and the specification ``phi = sum_k a_k c^T N z_k``. Tokens with a point set (the query of a
fixed-query unit) are constants. Everything is float64, and the softmax is bounded by the library as one operation.

Before any bound, a *forward gate* checks the module against :func:`aimp.heads.forward` on random points of the
sets (:func:`forward_gate`). The bounds use the library's defaults (:func:`bounds`). For alpha-CROWN these are 20
iterations, ``lr_alpha = 0.5`` and early stop after 10 iterations without improvement. :func:`alpha_run` reruns
alpha-CROWN with a given number of iterations and learning rate; the settings of the paper's robustness check are
:data:`ALPHA_ITERS` x :data:`ALPHA_LRS`.

Torch's intra-op thread count is set from the project budget (:func:`aimp.compute.set_torch_threads`) before every
bound computation.
"""
from __future__ import annotations

import contextlib
import io
import time
import warnings

import numpy as np

from aimp import compute
from aimp import sets as S
from aimp.heads import forward
from aimp.types import Instance

#: methods in the order they are run on one ``BoundedModule``; the values are auto_LiRPA's method names
METHODS = {"IBP": "IBP", "CROWN": "CROWN", "alpha-CROWN": "CROWN-Optimized"}
#: random points of the forward gate, and its tolerance (relative to ``max(1, |phi(centers)|)``)
GATE_N = 100
GATE_TOL = 1e-12
#: tolerance of the check ``ub >= max(sampled values)`` (relative to ``max(1, |max sampled|)``)
SAMPLE_TOL = 1e-9
#: alpha-CROWN settings of the robustness check: iterations x learning rates (0.5 is auto_LiRPA 0.8.0's default
#: ``lr_alpha``; 0.1 is the default of the alpha-beta-CROWN verifier)
ALPHA_ITERS = (20, 100, 500)
ALPHA_LRS = (0.5, 0.1)


def _torch(threads: int | None = None):
    """Import torch with its intra-op threads set from the budget."""
    import torch

    compute.set_torch_threads(threads)
    return torch


def _lirpa():
    """Import auto_LiRPA. Its import resets the warning filters and triggers a torch.jit deprecation warning on
    recent Pythons; the warning is silenced here and the filters are restored."""
    import sys

    if "auto_LiRPA" not in sys.modules:
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore", FutureWarning)
            import auto_LiRPA  # noqa: F401
            from auto_LiRPA.perturbations import PerturbationLpNorm  # noqa: F401
    return sys.modules["auto_LiRPA"]


@contextlib.contextmanager
def _quiet():
    """Silence two expected library messages: the constant tokens carry a batch dimension of one (shape ``(1, d)``),
    which auto_LiRPA warns about, and alpha-CROWN prints its early stops."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Constant operand of node")
        with contextlib.redirect_stdout(io.StringIO()):
            yield


def versions() -> dict:
    """Versions of auto_LiRPA and torch (for the result records)."""
    import torch

    return {"auto_lirpa": _lirpa().__version__, "torch": torch.__version__}


def perturbed_tokens(inst: Instance) -> list[int]:
    """Tokens with a nondegenerate set (in :attr:`Instance.tokens` order: the query first); every one must be an l2
    ball in its base coordinates (balls and ellipsoids)."""
    out = []
    for t in inst.tokens:
        rep = S.affine(inst.sets[t])
        if rep.r and rep.rho:
            if rep.base != "ball":
                raise ValueError(f"token {t}: base set {rep.base!r}, only l2 balls are supported")
            out.append(t)
        elif rep.base not in ("point", "ball"):
            raise ValueError(f"token {t}: base set {rep.base!r}, only l2 balls are supported")
    return out


def head_module(inst: Instance, perturbed: list[int] | None = None, threads: int | None = None):
    """The head as a float64 torch module ``(u_t for t in perturbed) -> phi`` of shape ``(batch, 1)``.

    Token ``t`` is ``c_t + u_t E_t^T`` (``u_t`` of shape ``(batch, r_t)``) when ``t`` is perturbed and the constant
    ``c_t`` otherwise; ``s_k = (z_i M) . z_k`` over the keys, ``a = softmax(s)``, ``phi = sum_k a_k (z_k N^T c)``.
    """
    torch = _torch(threads)
    perturbed = perturbed_tokens(inst) if perturbed is None else list(perturbed)
    reps = {t: S.affine(inst.sets[t]) for t in inst.tokens}

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            T64 = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64)  # noqa: E731
            self.toks = list(inst.tokens)
            self.keys = list(inst.keys)
            self.q = inst.query
            self.pert = list(perturbed)
            for t in self.toks:
                self.register_buffer(f"c{t}", T64(reps[t].c)[None, :])
                if t in self.pert:
                    self.register_buffer(f"B{t}", T64(reps[t].B.T))  # (r, d): z = c + u B^T
            self.register_buffer("M", T64(inst.head.M))
            self.register_buffer("gN", T64(inst.head.N.T @ np.asarray(inst.c, float))[:, None])

        def forward(self, *us):
            Z = {}
            for t in self.toks:
                c = getattr(self, f"c{t}")
                Z[t] = c + us[self.pert.index(t)] @ getattr(self, f"B{t}") if t in self.pert else c
            qv = Z[self.q] @ self.M
            s = torch.cat([(qv * Z[k]).sum(-1, keepdim=True) for k in self.keys], dim=-1)
            a = torch.softmax(s, dim=-1)
            g = torch.cat([Z[k] @ self.gN for k in self.keys], dim=-1)
            return (a * g).sum(-1, keepdim=True)

    return Head().double().eval()


def forward_gate(inst: Instance, mod, perturbed: list[int], rng: np.random.Generator, n: int = GATE_N) -> dict:
    """The module against :func:`aimp.heads.forward` at ``n`` random points of the sets (each perturbed token at a
    uniformly random radius in a uniformly random direction). Returns the largest absolute deviation, the pass flag
    (``<= GATE_TOL max(1, |phi(centers)|)``) and the largest value of ``phi`` seen (centers included), which every
    valid upper bound must reach."""
    import torch

    reps = {t: S.affine(inst.sets[t]) for t in inst.tokens}
    phi0 = forward(inst, inst.centers())["phi"]
    gate, vmax = 0.0, phi0
    for _ in range(n):
        us, Z = [], inst.centers().copy()
        for t in perturbed:
            x = rng.standard_normal(reps[t].r)
            x *= reps[t].rho * rng.uniform(0, 1) / np.linalg.norm(x)
            us.append(torch.as_tensor(x[None, :], dtype=torch.float64))
            Z[t] = reps[t].c + reps[t].B @ x
        with torch.no_grad():
            v = float(mod(*us))
        ref = forward(inst, Z)["phi"]
        gate = max(gate, abs(v - ref))
        vmax = max(vmax, ref)
    return {"gate_max_abs": gate, "gate_ok": bool(gate <= GATE_TOL * max(1.0, abs(phi0))), "sampled_max": vmax,
            "phi_nominal": phi0}


def _inputs(inst: Instance, perturbed: list[int]):
    """Zero base points and the ``BoundedTensor`` inputs, one l2 ball of radius ``rho_t`` per perturbed token."""
    import torch

    _lirpa()
    from auto_LiRPA import BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    reps = {t: S.affine(inst.sets[t]) for t in perturbed}
    x0 = tuple(torch.zeros(1, reps[t].r, dtype=torch.float64) for t in perturbed)
    bx = tuple(BoundedTensor(x0[a], PerturbationLpNorm(norm=2, eps=float(reps[t].rho))) for a, t in enumerate(perturbed))
    return x0, bx


def bounds(inst: Instance, seed: int, methods=tuple(METHODS), threads: int | None = None) -> dict:
    """Upper bounds of ``phi = c^T y`` by IBP, CROWN and alpha-CROWN (library defaults, run in this order on one
    ``BoundedModule``), each with its time; the forward gate first (``seed`` drives its random points).

    Returns ``{"gate_max_abs", "gate_ok", "sampled_max", "phi_nominal", "perturbed", "n_inputs", "build_s", "dtype",
    "methods": {method: {"ub", "status", "time_s", "above_samples"[, "iterations"]}}}``. A library error is recorded
    in ``status`` (``ub = None``), not raised.
    """
    _torch(threads)
    BoundedModule = _lirpa().BoundedModule
    pert = perturbed_tokens(inst)
    mod = head_module(inst, pert, threads)
    out = {**forward_gate(inst, mod, pert, np.random.default_rng(seed)), "perturbed": pert, "n_inputs": len(pert),
           "methods": {}}
    x0, bx = _inputs(inst, pert)
    t0 = time.perf_counter()
    with _quiet():
        bm = BoundedModule(mod, x0, device="cpu", verbose=0)
    out["build_s"] = time.perf_counter() - t0
    tol = SAMPLE_TOL * max(1.0, abs(out["sampled_max"]))
    for meth in methods:
        t1 = time.perf_counter()
        try:
            with _quiet():
                _, ub = bm.compute_bounds(x=bx, method=METHODS[meth], bound_lower=False, bound_upper=True)
            rec = {"ub": float(ub.detach().reshape(-1)[0]), "status": "ok"}
            if meth == "alpha-CROWN":
                rec["iterations"] = int(bm.bound_opts.get("optimize_bound_args", {}).get("iteration", -1))
        except Exception as e:  # recorded, not fatal: a library issue
            rec = {"ub": None, "status": f"error: {e!r}"[:300]}
        rec["time_s"] = time.perf_counter() - t1
        rec["above_samples"] = None if rec["ub"] is None else bool(rec["ub"] >= out["sampled_max"] - tol)
        out["methods"][meth] = rec
    out["dtype"] = str(next(mod.buffers()).dtype)
    return out


def alpha_run(inst: Instance, iters: int, lr: float, threads: int | None = None) -> dict:
    """alpha-CROWN upper bound with ``iteration = iters`` and ``lr_alpha = lr`` (the library's other defaults, incl.
    the early stop). The bound after every iteration is recorded with its time through ``stop_criterion_func``,
    which always returns False, so the library's behaviour is unchanged.

    Returns ``{"ub", "time_s", "iterations_run", "iter_best", "trace_min", "trace": [(t_s, bound), ...]}``.
    """
    _torch(threads)
    BoundedModule = _lirpa().BoundedModule
    pert = perturbed_tokens(inst)
    mod = head_module(inst, pert, threads)
    x0, bx = _inputs(inst, pert)
    trace = []
    t0 = time.perf_counter()

    def rec(neg_u):
        trace.append((time.perf_counter() - t0, -float(neg_u.detach().reshape(-1)[0])))
        return False

    with _quiet():
        bm = BoundedModule(mod, x0, device="cpu", verbose=0,
                           bound_opts={"optimize_bound_args": {"iteration": int(iters), "lr_alpha": float(lr),
                                                               "stop_criterion_func": rec}})
        _, ub = bm.compute_bounds(x=bx, method=METHODS["alpha-CROWN"], bound_lower=False, bound_upper=True)
    wall = time.perf_counter() - t0
    vals = [v for _, v in trace]
    best_i = int(np.argmin(vals)) + 1 if vals else None
    return {"ub": float(ub.detach().reshape(-1)[0]), "time_s": wall, "iterations_run": len(trace),
            "iter_best": best_i, "trace_min": min(vals) if vals else None, "trace": trace}


def setting_name(iters: int, lr: float) -> str:
    return f"it{int(iters)}_lr{float(lr):g}"


def alpha_settings(inst: Instance, iters=ALPHA_ITERS, lrs=ALPHA_LRS, threads: int | None = None,
                   keep_trace: bool = True) -> dict:
    """:func:`alpha_run` for every (learning rate, iterations) pair; ``{setting_name: record}``. The per-iteration
    trace is kept for the largest iteration count only (``keep_trace``)."""
    out = {}
    for lr in lrs:
        for it in iters:
            a = alpha_run(inst, it, lr, threads)
            if not (keep_trace and it == max(iters)):
                a["trace"] = None
            out[setting_name(it, lr)] = {"iterations": int(it), "lr_alpha": float(lr), **a}
    return out
