"""The level-one relaxation in MOSEK's task API — the same relaxation as :mod:`aimp.relax_sdp`, a different solver
interface.

**Why.** CVXPY always *dualizes* a continuous conic problem before handing it to MOSEK (``MOSEK.apply`` ->
``Dualize``): every entry of the moment matrix ``X`` (a CVXPY variable with ``PSD=True``) becomes an equality row of the
MOSEK task, i.e. ``m = (D+1)(D+2)/2`` rows (467 072 at ``D = 965``). MOSEK's Schur complement over those rows needed
> 180 GB at the GPT-2 sizes of the paper (Section 5), and the 129 x 129 Shor score SDPs took ~7 s each. Here ``X`` is a
MOSEK bar (PSD) variable and every constraint is an affine conic constraint (ACC) on ``X`` and a few scalars, so MOSEK
sees only the ``~n r`` rows of the relaxation itself.

**Model layer.** :class:`ConicModel` holds one PSD variable ``X`` (side ``m``) and scalar variables ``x``; each constraint
block is ``F x + <Fbar_r, X> + g in K`` with ``K`` in {zero, nonneg (``>= 0``), soc (``t >= ||rest||``), pexp
(``x1 >= x2 exp(x3/x2)``)}. ``X`` coefficients are given in *entry* semantics: ``coef * X[k, l]`` (``k != l`` allowed in
either order; stored as a symmetric matrix with ``coef/2`` on both sides, as MOSEK's ``<F, X>`` expects).

**Safe value** (the safe dual bound of Proposition 16, as in :mod:`aimp.safe_bound`, on explicit data). With multipliers
``lambda_k`` (MOSEK's ACC duals, sign fixed by the stationarity residual and projected onto ``K_k^*``):
``f(x, X) <= l0 + <r, x> + <R, X>`` for every feasible point, ``l0 = c0 + sum <lambda_k, g_k>``, ``r = c + F^T lambda``,
``R = C + Fbar^T lambda``; the sup over the a-priori set (``x`` in its box, ``X >= 0`` with ``X_kk <= D_k`` and the ball
groups) is bounded exactly as in ``safe_bound.safe_dual_bound``, plus the same rounding allowance.

:func:`solve_sdp1` builds ``sdp_1`` = the level-one relaxation of Definition 3 (lifted sets, scores, the secant on the
log-attention intervals (14), the interval diagonal, ``persp`` + ``simplex_x_z``, the l1 cut) with the exp-cone pruning
of Appendix B — constraint for constraint the model of
``relax_sdp.solve_sdp(rlt=("persp", "simplex_x_z"), t_interval="tight", cone_form=True)`` for ball / ellipsoid / point sets
(tests/test_sdp_mosek.py compares the two). :func:`shor_score` is the two-set Shor score bound of ``bounds._shor``.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import scipy.sparse as sp

from aimp import sets as S
from aimp.types import Instance, SolveResult

U_ROUND = np.finfo(float).eps / 2
KINDS = ("zero", "nonneg", "soc", "pexp")


class ConicModel:
    def __init__(self, m: int):
        self.m = m
        self.nv = 0
        self.nrows = 0
        self._sc = ([], [], [])  # rows, cols, vals
        self._bar = ([], [], [], [])  # rows, k, l, vals (entry semantics)
        self._g = []
        self.blocks: list[dict] = []
        self.obj_c: dict[int, float] = {}
        self.obj_bar = ([], [], [])
        self.obj_const = 0.0
        self.var_box: list[tuple[float, float]] = []

    # ---- variables / objective
    def var(self, n: int, lo: float, hi: float) -> np.ndarray:
        """``n`` free scalars; ``[lo, hi]`` is an a-priori box implied by the constraints (used only by the safe bound)."""
        idx = np.arange(self.nv, self.nv + n)
        self.nv += n
        self.var_box += [(lo, hi)] * n
        return idx

    def objective(self, sc=None, bar=None, const: float = 0.0):
        if sc is not None:
            for c, v in zip(*sc):
                self.obj_c[int(c)] = self.obj_c.get(int(c), 0.0) + float(v)
        if bar is not None:
            k, l, v = (np.atleast_1d(np.asarray(a)) for a in bar)
            self.obj_bar[0].append(k.astype(np.int64))
            self.obj_bar[1].append(l.astype(np.int64))
            self.obj_bar[2].append(v.astype(float))
        self.obj_const += const

    # ---- constraints
    def add(self, kind: str, n: int, g, sc=None, bar=None, name: str = "") -> int:
        """Block of ``n`` rows (local row indices ``0..n-1`` in ``sc = (r, col, v)`` and ``bar = (r, k, l, v)``)."""
        if kind not in KINDS:
            raise ValueError(kind)
        r0 = self.nrows
        g = np.broadcast_to(np.asarray(g, float), (n,)).copy()
        if sc is not None:
            r, c, v = (np.atleast_1d(np.asarray(a)) for a in sc)
            if r.size:
                self._sc[0].append(r.astype(np.int64) + r0)
                self._sc[1].append(c.astype(np.int64))
                self._sc[2].append(v.astype(float))
        if bar is not None:
            r, k, l, v = (np.atleast_1d(np.asarray(a)) for a in bar)
            if r.size:
                self._bar[0].append(r.astype(np.int64) + r0)
                self._bar[1].append(k.astype(np.int64))
                self._bar[2].append(l.astype(np.int64))
                self._bar[3].append(v.astype(float))
        self._g.append(g)
        self.blocks.append({"kind": kind, "start": r0, "n": n, "name": name})
        self.nrows += n
        return len(self.blocks) - 1

    # ---- assembled data
    def assemble(self):
        nr = self.nrows
        F = sp.coo_matrix((np.concatenate(self._sc[2]) if self._sc[2] else np.zeros(0),
                           (np.concatenate(self._sc[0]) if self._sc[0] else np.zeros(0, int),
                            np.concatenate(self._sc[1]) if self._sc[1] else np.zeros(0, int))),
                          shape=(nr, max(self.nv, 1))).tocsr()
        F.sum_duplicates()
        # bar part as a sparse (nr x m*m) matrix on vec(X) with symmetric (lower+upper) halves
        if self._bar[0]:
            r = np.concatenate(self._bar[0])
            k = np.concatenate(self._bar[1])
            l = np.concatenate(self._bar[2])
            v = np.concatenate(self._bar[3])
        else:
            r = k = l = np.zeros(0, np.int64)
            v = np.zeros(0)
        lo, hi = np.maximum(k, l), np.minimum(k, l)
        vv = np.where(k == l, v, v / 2)  # MOSEK: lower triangle of a symmetric F; off-diagonal counted twice
        Fb = sp.coo_matrix((vv, (r, lo * self.m + hi)), shape=(nr, self.m * self.m)).tocsr()
        Fb.sum_duplicates()
        g = np.concatenate(self._g) if self._g else np.zeros(0)
        if self.obj_bar[0]:
            ok, ol, ov = (np.concatenate(a) for a in self.obj_bar)
        else:
            ok = ol = np.zeros(0, np.int64)
            ov = np.zeros(0)
        olo, ohi = np.maximum(ok, ol), np.minimum(ok, ol)
        Cb = sp.coo_matrix((np.where(ok == ol, ov, ov / 2), (olo, ohi)), shape=(self.m, self.m)).tocsr()
        Cb.sum_duplicates()
        c = np.zeros(max(self.nv, 1))
        for j, v_ in self.obj_c.items():
            c[j] += v_
        return F, Fb, g, c, Cb

    # ---- solve
    def solve(self, threads: int | None = None, time_limit: float | None = None, tol: dict | None = None) -> dict:
        import mosek

        from aimp import compute

        t0 = time.perf_counter()
        F, Fb, g, c, Cb = self.assemble()
        params = compute.mosek_params(threads, **(tol or {}))
        if time_limit is not None:
            params["MSK_DPAR_OPTIMIZER_MAX_TIME"] = float(time_limit)
        with mosek.Env() as env, env.Task() as task:
            if os.environ.get("AIMP_MOSEK_LOG"):
                task.set_Stream(mosek.streamtype.log, lambda msg: print(msg, end="", flush=True))
            for name, val in params.items():
                if name.startswith("MSK_IPAR"):
                    task.putnaintparam(name, int(val))
                else:
                    task.putnadouparam(name, float(val))
            nv = max(self.nv, 1)
            task.appendvars(nv)
            task.putvarboundsliceconst(0, nv, mosek.boundkey.fr, -math.inf, math.inf)
            task.appendbarvars([self.m])
            task.appendafes(self.nrows)
            Fc = F.tocoo()
            if Fc.nnz:
                task.putafefentrylist(Fc.row.astype(np.int64), Fc.col.astype(np.int32), Fc.data)
            Bc = Fb.tocoo()
            if Bc.nnz:
                task.putafebarfblocktriplet(Bc.row.astype(np.int64), np.zeros(Bc.nnz, np.int32),
                                            (Bc.col // self.m).astype(np.int32), (Bc.col % self.m).astype(np.int32),
                                            Bc.data)
            task.putafegslice(0, self.nrows, g)
            doms = {}
            for b in self.blocks:
                key = (b["kind"], b["n"])
                if key not in doms:
                    doms[key] = {"zero": task.appendrzerodomain, "nonneg": task.appendrplusdomain,
                                 "soc": task.appendquadraticconedomain}.get(b["kind"], None)
                    doms[key] = (doms[key](b["n"]) if doms[key] is not None else task.appendprimalexpconedomain())
                task.appendacc(doms[key], list(range(b["start"], b["start"] + b["n"])), None)
            task.putclist(list(range(nv)), c.tolist())
            Cc = Cb.tocoo()
            if Cc.nnz:
                task.putbarcblocktriplet(np.zeros(Cc.nnz, np.int32), Cc.row.astype(np.int32), Cc.col.astype(np.int32),
                                         Cc.data)
            task.putcfix(self.obj_const)
            task.putobjsense(mosek.objsense.maximize)
            t_build = time.perf_counter() - t0
            t1 = time.perf_counter()
            trm = task.optimize()
            t_opt = time.perf_counter() - t1
            soltype = mosek.soltype.itr
            prosta, solsta = task.getprosta(soltype), task.getsolsta(soltype)
            out = {"prosta": str(prosta), "solsta": str(solsta), "termcode": str(trm), "t_build_s": t_build,
                   "t_optimize_s": t_opt, "num_iters": task.getintinf(mosek.iinfitem.intpnt_iter),
                   "threads": params["MSK_IPAR_NUM_THREADS"], "mosek_params": params, "nrows": self.nrows,
                   "m": self.m, "nv": self.nv}
            try:  # residuals of our primal / our dual (here = the task's primal / dual): solvers.py's measure, roles swapped
                (_po, pvc, pvv, pvb, pvk, _pvi, _do, dvc, dvv, dvb, dvk) = task.getsolutioninfo(soltype)
                out["mosek_primal_resid"] = float(max(pvc, pvv, pvb, pvk))
                out["mosek_dual_resid"] = float(max(dvc, dvv, dvb, dvk))
            except Exception as e:  # never fatal
                out["mosek_resid_error"] = repr(e)
            if solsta in (mosek.solsta.optimal, mosek.solsta.unknown) or prosta == mosek.prosta.prim_and_dual_feas:
                out["pobj"] = task.getprimalobj(soltype)
                out["dobj"] = task.getdualobj(soltype)
                out["x"] = np.asarray(task.getxx(soltype))[: self.nv]
                low = np.asarray(task.getbarxj(soltype, 0))
                X = np.zeros((self.m, self.m))
                # MOSEK packs the lower triangle column by column
                cols = np.repeat(np.arange(self.m), np.arange(self.m, 0, -1))
                rows = np.concatenate([np.arange(j, self.m) for j in range(self.m)])
                X[rows, cols] = low
                X[cols, rows] = low
                out["X"] = X
                out["doty"] = np.concatenate([np.asarray(task.getaccdoty(soltype, k)) for k in range(len(self.blocks))])
            status = ("optimal" if solsta == mosek.solsta.optimal else
                      "infeasible" if solsta in (mosek.solsta.prim_infeas_cer, mosek.solsta.dual_infeas_cer) else
                      "suboptimal" if "X" in out else "error")
            out["status"] = status
        out.update({"F": F, "Fb": Fb, "g": g, "c": c, "Cb": Cb})
        return out

    # ---- rigorous bound from the duals
    def safe_value(self, sol: dict, D: np.ndarray, groups: list[tuple[list[int], float]]) -> dict:
        """The safe dual bound (Proposition 16) on explicit data. ``D[k] >= X_kk`` on the feasible set; ``groups``:
        ``sum_{k in idx} X_kk/D_k <= budget``. Any multipliers in the dual cones give a valid bound, so both signs of
        MOSEK's ``doty`` (its sign convention for a maximization) are projected and evaluated, and the smaller bound is
        kept."""
        t0 = time.perf_counter()
        if "doty" not in sol:
            return {"value_safe": None, "reason": "no dual solution"}
        outs = [self._safe_for(sol, sgn, D, groups) for sgn in (+1.0, -1.0)]
        best = min(outs, key=lambda o: o["value_safe"])
        best["value_safe_other_sign"] = max(o["value_safe"] for o in outs)
        best["time_s"] = time.perf_counter() - t0
        return best

    def _safe_for(self, sol: dict, sgn: float, D, groups) -> dict:
        F, Fb, g, c, Cb = sol["F"], sol["Fb"], sol["g"], sol["c"], sol["Cb"]
        lam, nfix = self._project_dual(sgn * sol["doty"])
        r = c + F.T @ lam
        res = float(np.max(np.abs(r[: self.nv]))) if self.nv else 0.0
        m = self.m
        # lower triangle (row >= col) of R = C + sum_r lambda_r F_r, off-diagonal entries holding the half coefficient
        Rl = np.asarray(Fb.T @ lam).reshape(m, m) + Cb.toarray()
        R = Rl + np.tril(Rl, -1).T  # full symmetric: <R, X> = sum_kl R_kl X_kl = MOSEK's <F, X>
        Al = np.asarray(abs(Fb).T @ np.abs(lam)).reshape(m, m) + abs(Cb).toarray()
        Rabs = Al + np.tril(Al, -1).T
        l0 = self.obj_const + float(lam @ g)
        l0_abs = abs(self.obj_const) + float(np.abs(lam) @ np.abs(g))
        box, box_abs = 0.0, 0.0
        cabs = np.abs(c) + np.asarray(abs(F).T @ np.abs(lam)).ravel()
        for j in range(self.nv):
            lo, hi = self.var_box[j]
            box += max(r[j] * lo, r[j] * hi)
            box_abs += float(cabs[j]) * max(abs(lo), abs(hi))
        D = np.maximum(np.asarray(D, float), 0.0)
        sD = np.sqrt(D)
        budget, in_group = 0.0, np.zeros(m, bool)
        for idx, b in groups:
            idx = [k for k in idx if D[k] > 0]
            if idx:
                budget += float(b)
                in_group[idx] = True
        budget += float(np.sum((D > 0) & ~in_group))
        SRS = R * np.outer(sD, sD)
        lam_max = float(np.linalg.eigvalsh(SRS)[-1])
        eig_err = 8 * m * U_ROUND * float(np.linalg.norm(SRS, "fro"))
        psd_eig = (max(lam_max, 0.0) + eig_err) * budget
        ev, V = np.linalg.eigh(R)
        Rp = (V * np.maximum(ev, 0.0)) @ V.T
        psd_ent = float(np.sum(np.abs(Rp) * np.outer(sD, sD))) + \
            8 * m * U_ROUND * float(np.linalg.norm(R, "fro")) * float(sD.sum()) ** 2
        psd_term = min(psd_eig, psd_ent)
        rounding = 64 * U_ROUND * (l0_abs + box_abs + float(np.sum(Rabs * np.outer(sD, sD))))
        val = l0 + box + psd_term + rounding
        return {"value_safe": float(val), "l0": l0, "box_term": box, "psd_term": psd_term, "psd_term_eig": psd_eig,
                "psd_term_entrywise": psd_ent, "lam_max_SRS": lam_max, "psd_budget": budget, "rounding": rounding,
                "dual_sign": sgn, "resid_free_max": res, "fixes": nfix, "R_min_eig": float(ev[0]),
                "R_max_eig": float(ev[-1])}

    def _project_dual(self, y: np.ndarray) -> tuple[np.ndarray, dict]:
        """Map ``y`` into the dual cones (zero: free; nonneg: clip; soc: raise ``t``; pexp: raise to K_exp^*)."""
        lam = y.copy()
        nfix = {"nonneg": 0, "soc": 0, "exp": 0}
        for b in self.blocks:
            s, n = b["start"], b["n"]
            v = lam[s:s + n]
            if b["kind"] == "nonneg":
                nfix["nonneg"] += int(np.sum(v < 0))
                lam[s:s + n] = np.maximum(v, 0.0)
            elif b["kind"] == "soc":
                nr = float(np.linalg.norm(v[1:]))
                if v[0] < nr:
                    lam[s] = nr * (1 + 4 * U_ROUND)
                    nfix["soc"] += 1
            elif b["kind"] == "pexp":
                # MOSEK primal exp cone x1 >= x2 exp(x3/x2); dual: s1 >= -s3 exp(s2/s3 - 1), s3 < 0 (or s3 = 0, s1, s2 >= 0)
                s1, s2, s3 = v
                if s3 < 0:
                    need = -s3 * math.exp(min(s2 / s3 - 1.0, 700.0))
                    if s1 < need:
                        lam[s] = need * (1 + 4 * U_ROUND)
                        nfix["exp"] += 1
                else:
                    if s3 != 0 or s1 < 0 or s2 < 0:
                        nfix["exp"] += 1
                    lam[s:s + 3] = (max(s1, 0.0), max(s2, 0.0), 0.0)
        return lam, nfix

    def max_violation(self, sol: dict) -> float:
        if "X" not in sol:
            return float("nan")
        F, Fb, g = sol["F"], sol["Fb"], sol["g"]
        X = sol["X"]
        xv = (2 * np.tril(X) - np.diag(np.diag(X))).reshape(-1)  # Fb holds halves for off-diagonal: <F, X> = sum 2*half*X_kl
        v = F @ np.concatenate([sol["x"], np.zeros(F.shape[1] - sol["x"].size)]) + Fb @ xv + g
        worst = 0.0
        for b in self.blocks:
            w = v[b["start"]:b["start"] + b["n"]]
            if b["kind"] == "zero":
                worst = max(worst, float(np.max(np.abs(w))))
            elif b["kind"] == "nonneg":
                worst = max(worst, float(max(0.0, -np.min(w))))
            elif b["kind"] == "soc":
                worst = max(worst, float(max(0.0, np.linalg.norm(w[1:]) - w[0])))
            elif b["kind"] == "pexp":
                x1, x2, x3 = w
                worst = max(worst, float(max(0.0, x2 * math.exp(min(x3 / x2, 700)) - x1))) if x2 > 0 else worst
        return worst


# ------------------------------------------------------------------------------------------------ helpers
def _rows(n):
    return np.arange(n)


def _ball_rep(rep) -> bool:
    return rep.base in ("ball", "box", "point")


#: Interior-point tolerances of the GPT-2 runs (Appendix B): value := value_safe is rigorous from any dual point, and
#: MOSEK stalls at dual feasibility ~1e-6 on these SDPs, so the package default 1e-9 only adds stalled iterations.
PHASE3_TOL = {"MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-6, "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-6,
              "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-8}


def solve_sdp1(inst: Instance, ctrl: dict, threads: int | None = None, time_limit: float | None = None,
               prune_tol: float | None = 1e-6, cut: str = "B", tol: dict | None = None,
               keep_X: bool = False) -> SolveResult:
    """``sdp_1`` (Definition 3: persp + simplex_x_z + interval diagonal; secant on the log-attention intervals (14);
    width-pair cut by default, ``cut="auto"`` the pair with the smaller radius; exp-cone pruning) with
    ``value := value_safe``. Ball / ellipsoid / box / point sets only. ``keep_X``: attach the moment matrix and the
    ``u``-block positions as ``res.X`` / ``res.upos`` (starts of the strong lower bound, :mod:`aimp.lower`), not
    serialized."""
    from aimp.bounds import _quad_objective, choose_cut, t_intervals
    from aimp.heads import forward
    from aimp.relax_sdp import LiftIndex

    t0 = time.perf_counter()
    c0 = time.process_time()
    M, N, cvec = inst.head.M, inst.head.N, inst.c
    n, i = inst.n, inst.query
    reps = {t: S.affine(inst.sets[t]) for t in inst.tokens}
    if not all(_ball_rep(r) for r in reps.values()):
        raise NotImplementedError("sdp_mosek: ball / ellipsoid / box / point sets only (hulls are not supported)")
    cutinfo = choose_cut(ctrl, n, cut)
    L = LiftIndex.for_instance(inst, ())
    m = L.size
    mod = ConicModel(m)
    s_lo, s_hi = np.asarray(ctrl["s_lo"], float), np.asarray(ctrl["s_hi"], float)
    tau_lo, tau_hi = float(ctrl["tau_lo"]), float(ctrl["tau_hi"])
    a_lo = np.maximum(np.asarray(ctrl["a_lo"], float), 0.0)
    a_hi = np.minimum(np.asarray(ctrl["a_hi"], float), 1.0)
    t_lo, t_hi = t_intervals(s_lo, s_hi, "tight")
    pruned = [k for k in range(n) if prune_tol is not None and float(ctrl["a_hi"][k]) <= prune_tol]
    pos_a = np.array([L.a(k) for k in range(n)])
    upos = {t: np.arange(L.u(t).start, L.u(t).stop) for t in inst.tokens}
    Z0 = np.zeros(0, np.int64)

    # (1) X00 = 1
    mod.add("zero", 1, -1.0, bar=([0], [0], [0], [1.0]), name="X00")
    # (2) lifted set constraints: ball ||u_t|| <= rho, tr U_tt <= rho^2; box |u_t| <= h, diag U_tt <= h^2 (relax_sdp)
    for t in inst.tokens:
        rep, up = reps[t], upos[t]
        if not rep.r:
            continue
        r_ = up.size
        if rep.base == "box":
            z_ = np.zeros(r_, int)
            mod.add("nonneg", 2 * r_, np.r_[rep.h, rep.h], bar=(np.arange(2 * r_), np.r_[z_, z_], np.r_[up, up],
                                                                np.r_[-np.ones(r_), np.ones(r_)]), name=f"box{t}")
            mod.add("nonneg", r_, rep.h ** 2, bar=(np.arange(r_), up, up, -np.ones(r_)), name=f"boxdiag{t}")
            continue
        mod.add("soc", r_ + 1, np.r_[rep.rho, np.zeros(r_)], bar=(np.arange(1, r_ + 1), np.zeros(r_, int), up, np.ones(r_)),
                name=f"ball{t}")
        mod.add("nonneg", 1, rep.rho ** 2, bar=(np.zeros(r_, int), up, up, -np.ones(r_)), name=f"trace{t}")
    # (3) scores s_j = const + lin_i.u_i + lin_j.u_j + <Q, U_ij> as bar forms (row-local)
    score = []
    for idx, j in enumerate(inst.keys):
        same = j == i
        const, lin_i, lin_j, Q = _quad_objective(M, reps[i], reps[j], same)
        ks, ls, vs = [], [], []
        if reps[i].r:
            ks.append(np.zeros(upos[i].size, int)); ls.append(upos[i]); vs.append(lin_i)
        if not same and reps[j].r:
            ks.append(np.zeros(upos[j].size, int)); ls.append(upos[j]); vs.append(lin_j)
        if reps[i].r and (same or reps[j].r):
            uj = upos[i] if same else upos[j]
            KK, LL = np.meshgrid(upos[i], uj, indexing="ij")
            ks.append(KK.ravel()); ls.append(LL.ravel()); vs.append(np.asarray(Q, float).ravel())
        k_ = np.concatenate(ks) if ks else Z0
        l_ = np.concatenate(ls) if ls else Z0
        v_ = np.concatenate(vs) if vs else np.zeros(0)
        nz = v_ != 0
        score.append((float(const), k_[nz], l_[nz], v_[nz]))
    for idx, (const, k_, l_, v_) in enumerate(score):
        if not v_.size:
            continue
        nn = v_.size
        mod.add("nonneg", 2, [const - s_lo[idx], s_hi[idx] - const],
                bar=(np.r_[np.zeros(nn, int), np.ones(nn, int)], np.r_[k_, k_], np.r_[l_, l_], np.r_[v_, -v_]),
                name=f"s{idx}")
    # (4) tau (scalar)
    tau = mod.var(1, tau_lo, tau_hi)[0]
    mod.add("nonneg", 2, [-tau_lo, tau_hi], sc=([0, 1], [tau, tau], [1.0, -1.0]), name="tau")
    # (5) t_j = s_j - tau in [t_lo, t_hi]; exp cone a_j >= exp(t_j); secant
    for idx in range(n):
        const, k_, l_, v_ = score[idx]
        nn = v_.size
        mod.add("nonneg", 2, [const - t_lo[idx], t_hi[idx] - const], sc=([0, 1], [tau, tau], [-1.0, 1.0]),
                bar=(np.r_[np.zeros(nn, int), np.ones(nn, int)], np.r_[k_, k_], np.r_[l_, l_], np.r_[v_, -v_]),
                name=f"t{idx}")
        if idx in pruned:
            continue
        pa = pos_a[idx]
        # pexp: (a_j, 1, t_j): rows x1 = a_j = X[0, a_j]; x2 = 1; x3 = s_j - tau
        mod.add("pexp", 3, [0.0, 1.0, const], sc=([2], [tau], [-1.0]),
                bar=(np.r_[0, np.full(nn, 2)], np.r_[0, k_], np.r_[pa, l_], np.r_[1.0, v_]), name=f"exp{idx}")
        w = t_hi[idx] - t_lo[idx]
        if w > 1e-12:
            slope = (math.exp(t_hi[idx]) - math.exp(t_lo[idx])) / w
            # e^{t_lo} + slope (s - tau - t_lo) - a >= 0
            mod.add("nonneg", 1, math.exp(t_lo[idx]) + slope * (const - t_lo[idx]), sc=([0], [tau], [-slope]),
                    bar=(np.r_[np.zeros(nn, int), 0], np.r_[k_, 0], np.r_[l_, pa], np.r_[slope * v_, -1.0]),
                    name=f"secant{idx}")
        else:
            mod.add("nonneg", 1, math.exp(t_hi[idx]), bar=([0], [0], [pa], [-1.0]), name=f"secant{idx}")
    # (6) a in [0, 1] ∩ [a_lo, a_hi], sum a = 1
    za = np.zeros(n, int)
    mod.add("nonneg", n, 0.0, bar=(_rows(n), za, pos_a, np.ones(n)), name="a>=0")
    mod.add("nonneg", n, 1.0, bar=(_rows(n), za, pos_a, -np.ones(n)), name="a<=1")
    mod.add("nonneg", n, -np.asarray(ctrl["a_lo"], float), bar=(_rows(n), za, pos_a, np.ones(n)), name="a>=a_lo")
    mod.add("nonneg", n, np.asarray(ctrl["a_hi"], float), bar=(_rows(n), za, pos_a, -np.ones(n)), name="a<=a_hi")
    mod.add("zero", 1, -1.0, bar=(za, za, pos_a, np.ones(n)), name="sum_a")
    # (7) interval diagonal (R4): (a_lo + a_hi) a - a_lo a_hi - X[a, a] >= 0
    mod.add("nonneg", n, -a_lo * a_hi, bar=(np.r_[_rows(n), _rows(n)], np.r_[za, pos_a], np.r_[pos_a, pos_a],
                                            np.r_[a_lo + a_hi, -np.ones(n)]), name="diag_a")
    # (8) l1 cut (p, q scalars, not lifted)
    if cutinfo["eps"] is not None:
        eps = float(cutinfo["eps"])
        p = mod.var(n, 0.0, eps)
        q = mod.var(n, 0.0, eps)
        mod.add("nonneg", 2 * n, 0.0, sc=(_rows(2 * n), np.r_[p, q], np.ones(2 * n)), name="pq>=0")
        mod.add("zero", n, -np.asarray(cutinfo["ahat"], float), sc=(np.r_[_rows(n), _rows(n)], np.r_[p, q],
                                                                    np.r_[-np.ones(n), np.ones(n)]),
                bar=(_rows(n), za, pos_a, np.ones(n)), name="cut_eq")
        mod.add("nonneg", 1, eps, sc=(np.zeros(2 * n, int), np.r_[p, q], -np.ones(2 * n)), name="cut_l1")
    # (9) objective c^T y, y = sum_j a_j N c_j + (N B_j) X[a_j, u_j]
    ok, ol, ov = [], [], []
    for idx, j in enumerate(inst.keys):
        rep = reps[j]
        ok.append([0]); ol.append([pos_a[idx]]); ov.append([float(cvec @ (N @ rep.c))])
        if rep.r:
            w_ = (cvec @ N) @ rep.B
            ok.append(np.full(rep.r, pos_a[idx])); ol.append(upos[j]); ov.append(w_)
    mod.objective(bar=(np.concatenate(ok), np.concatenate(ol), np.concatenate(ov)))
    # (10) persp: ||G_j|| <= rho a_j, ||u_j - G_j|| <= rho (1 - a_j), G_j = X[a_j, u_j]
    for idx, j in enumerate(inst.keys):
        rep = reps[j]
        if not rep.r:
            continue
        up, pa, r_ = upos[j], pos_a[idx], upos[j].size
        if rep.base == "box":  # relax_sdp: |G| <= a h, |u - G| <= (1 - a) h (G = X[a_j, u_j])
            rr = np.arange(r_)
            za, fa = np.zeros(r_, int), np.full(r_, pa)
            for sg in (1.0, -1.0):
                # a h - sg G >= 0
                mod.add("nonneg", r_, 0.0, bar=(np.r_[rr, rr], np.r_[za, fa], np.r_[fa, up], np.r_[rep.h, -sg * np.ones(r_)]),
                        name=f"persp_box_a{idx}")
                # (1 - a) h - sg (u - G) >= 0
                mod.add("nonneg", r_, rep.h, bar=(np.r_[rr, rr, rr], np.r_[za, za, fa], np.r_[fa, up, up],
                                                  np.r_[-rep.h, -sg * np.ones(r_), sg * np.ones(r_)]), name=f"persp_box_1ma{idx}")
            continue
        mod.add("soc", r_ + 1, 0.0, bar=(np.r_[0, np.arange(1, r_ + 1)], np.r_[0, np.full(r_, pa)], np.r_[pa, up],
                                         np.r_[rep.rho, np.ones(r_)]), name=f"persp_a{idx}")
        mod.add("soc", r_ + 1, np.r_[rep.rho, np.zeros(r_)],
                bar=(np.r_[0, np.arange(1, r_ + 1), np.arange(1, r_ + 1)], np.r_[0, np.zeros(r_, int), np.full(r_, pa)],
                     np.r_[pa, up, up], np.r_[-rep.rho, np.ones(r_), -np.ones(r_)]), name=f"persp_1ma{idx}")
    # (11) simplex_x_z: sum_l X[a_l, u_t] = u_t (every lifted token), sum_l X[a_l, a_k] = a_k, X[a, a] >= 0
    for t in inst.tokens:
        up = upos[t]
        if not up.size:
            continue
        r_ = up.size
        rr = np.repeat(_rows(r_), n + 1)
        kk = np.tile(np.r_[pos_a, 0], r_)
        ll = np.repeat(up, n + 1)
        vv = np.tile(np.r_[np.ones(n), -1.0], r_)
        mod.add("zero", r_, 0.0, bar=(rr, kk, ll, vv), name=f"sxz_u{t}")
    rr = np.repeat(_rows(n), n + 1)
    kk = np.tile(np.r_[pos_a, 0], n)
    ll = np.repeat(pos_a, n + 1)
    vv = np.tile(np.r_[np.ones(n), -1.0], n)
    mod.add("zero", n, 0.0, bar=(rr, kk, ll, vv), name="sxz_a")
    iu = np.triu_indices(n)
    mod.add("nonneg", iu[0].size, 0.0, bar=(_rows(iu[0].size), pos_a[iu[0]], pos_a[iu[1]], np.ones(iu[0].size)),
            name="Xaa>=0")
    t_model = time.perf_counter() - t0

    sol = mod.solve(threads, time_limit, tol)
    # a-priori diagonal bounds (as relax_sdp._safe): balls rho^2 (group budget 1), a_j <= a_hi_j^2
    D = np.zeros(m)
    D[0] = 1.0
    groups = []
    for t in inst.tokens:
        if reps[t].r and reps[t].base == "box":
            D[upos[t]] = reps[t].h ** 2
        elif reps[t].r:
            D[upos[t]] = reps[t].rho ** 2
            groups.append((upos[t].tolist(), 1.0))
    D[pos_a] = a_hi ** 2
    sb = mod.safe_value(sol, D, groups)
    status = sol["status"]
    vs = sb.get("value_safe")
    diag = {"value_safe": vs, "safe_info": {k: v for k, v in sb.items() if k != "value_safe"},
            "value_primal": sol.get("pobj"), "value_dual": sol.get("dobj"), "value_source": "safe",
            "dual_resid": sb.get("resid_free_max"), "primal_resid": None, "unreliable": False,
            "mosek_dual_resid": sol.get("mosek_dual_resid"), "mosek_primal_resid": sol.get("mosek_primal_resid"),
            "t_optimize_s": sol["t_optimize_s"],
            "time_build_s": t_model + sol["t_build_s"], "solve_call_s": sol["t_optimize_s"],
            "solve_time": sol["t_optimize_s"], "compilation_time": sol["t_build_s"], "time_safe_s": sb.get("time_s"),
            "num_iters": sol.get("num_iters"), "prosta": sol["prosta"], "solsta": sol["solsta"],
            "nrows": sol["nrows"], "pruned_keys": pruned, "cut_used": cutinfo["cut_used"], "eps": cutinfo["eps"],
            "w_t_used": float(np.max(t_hi - t_lo)), "t_interval": "tight", "solver": "mosek_task"}
    pruning_slack = 0.0
    if pruned:
        from aimp.theory import support_values

        hv = support_values(inst)
        pruning_slack = float(sum(float(ctrl["a_hi"][k]) * (hv[k] - hv.min()) for k in pruned))
    diag["pruning_slack"] = pruning_slack
    res = SolveResult(method="sdp_1", status=status, value=vs if vs is not None else float("nan"),
                      params={**sol["mosek_params"], "rlt": ["diag", "persp", "simplex_x_z"], "D": L.D, "cut": cut,
                              "prune_tol": prune_tol, "t_interval": "tight", "solver": "mosek_task"},
                      diagnostics=diag,
                      approximations=([] if status == "optimal" else [status]) + ([f"exp_pruned:{len(pruned)}"] if pruned else []))
    if "X" in sol:
        X = sol["X"]
        ev = np.linalg.eigvalsh(X)[::-1]
        diag.update({"X_min_eig": float(ev[-1]), "top_eigs": ev[:5].tolist(),
                     "rank1_ratio": float(ev[1] / ev[0]) if ev.size > 1 and ev[0] > 0 else 0.0,
                     "a": X[0, pos_a].copy(), "primal_max_violation": mod.max_violation(sol)})
        Z = inst.centers().copy()
        for t in inst.tokens:
            if reps[t].r:
                Z[t] = S.to_z(reps[t], S.base_project(reps[t], X[0, upos[t]]))
        val = forward(inst, Z)["phi"]
        res.recovered_point, res.recovered_value = Z, float(val)
        if keep_X:
            res.X, res.upos = X, upos
            res.pos_a = pos_a  # the a-block positions and the scalar tau of the primal (for attribution analyses)
            res.tau = float(sol["x"][tau]) if "x" in sol else None
        diag["sdp_recovered_value"] = float(val)
        if vs is not None:
            diag["sdp_gap_constructive"] = float(vs - val)
    res.wall_time_s = time.perf_counter() - t0
    res.cpu_time_s = time.process_time() - c0
    return res


def shor_score(inst: Instance, j: int, threads: int | None = None) -> tuple[float, float, dict]:
    """``bounds._shor`` (level-0 Shor bound on the score of key ``j``; ball / ellipsoid sets) in the task API. Interval
    ends: ``max(primal, dual, safe)`` objective plus the ``SHOR_PAD_REL`` pad (min side: the mirror)."""
    from aimp.bounds import SHOR_PAD_REL, _quad_objective, _score_pieces

    rep_i, rep_j, same = _score_pieces(inst, j)
    const, lin_i, lin_j, Q = _quad_objective(inst.head.M, rep_i, rep_j, same)
    ri = rep_i.r
    rj = 0 if same else rep_j.r
    m = 1 + ri + rj
    ui = np.arange(1, 1 + ri)
    uj = np.arange(1 + ri, 1 + ri + rj)
    out, info = [], {}
    for sense in ("min", "max"):
        sg = 1.0 if sense == "max" else -1.0
        mod = ConicModel(m)
        mod.add("zero", 1, -1.0, bar=([0], [0], [0], [1.0]))
        for up, rep in ((ui, rep_i), (uj, rep_j)):
            if not up.size:
                continue
            if rep.base != "ball":
                raise NotImplementedError(rep.base)
            r_ = up.size
            mod.add("soc", r_ + 1, np.r_[rep.rho, np.zeros(r_)], bar=(np.arange(1, r_ + 1), np.zeros(r_, int), up, np.ones(r_)))
            mod.add("nonneg", 1, rep.rho ** 2, bar=(np.zeros(r_, int), up, up, -np.ones(r_)))
        ks, ls, vs = [np.zeros(ri, int)], [ui], [lin_i]
        if same:
            KK, LL = np.meshgrid(ui, ui, indexing="ij")
        else:
            ks.append(np.zeros(rj, int)); ls.append(uj); vs.append(lin_j)
            KK, LL = np.meshgrid(ui, uj, indexing="ij")
        ks.append(KK.ravel()); ls.append(LL.ravel()); vs.append(np.asarray(Q, float).ravel())
        mod.objective(bar=(np.concatenate(ks), np.concatenate(ls), sg * np.concatenate(vs)), const=sg * const)
        sol = mod.solve(threads)
        if sol["status"] not in ("optimal", "suboptimal"):
            raise RuntimeError(f"shor score interval {sense} for key {j}: {sol['status']} {sol['solsta']}")
        D = np.zeros(m)
        D[0] = 1.0
        groups = []
        for up, rep in ((ui, rep_i), (uj, rep_j)):
            if up.size:
                D[up] = rep.rho ** 2
                groups.append((up.tolist(), 1.0))
        sb = mod.safe_value(sol, D, groups)
        cand = [v for v in (sol.get("pobj"), sol.get("dobj"), sb.get("value_safe")) if v is not None]
        v = max(cand)  # an upper bound on max(sg * s)
        pad = SHOR_PAD_REL * max(1.0, abs(v))
        out.append(sg * (v + pad))
        X = sol.get("X")
        ev = np.linalg.eigvalsh(X)[::-1] if X is not None else None
        info[sense] = {"status": sol["status"], "pobj": sol.get("pobj"), "dobj": sol.get("dobj"),
                       "value_safe": sb.get("value_safe"), "rank1_ratio": float(ev[1] / ev[0]) if ev is not None and ev[0] > 0 else None,
                       "t_s": sol["t_optimize_s"]}
    return out[0], out[1], info
