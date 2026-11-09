"""The curvature (Taylor) certificate with a fixed query, checked against the exact fixed-query worst cases and compared with
the certificate's gap (Section 7.2 and its gpt-oss-20b counterpart in Section 7.3; Table 3, row "curvature certificate";
``figdata/P9a_taylor_fixed_query.csv``).

Unit sets (query position 4, fixed query; the stored records give ``p*``, its bracket, the range of motion and the
certificate; ``<unit>`` as :func:`aimp.gpt2_units.unit_name`)::

  gpt2_a   GPT-2 small, head's output,  rho_rel 0.002 / 0.005 / 0.01 / 0.02  exact_fixed_query/sdp/L<ll>/<unit>_fq.json  (2,880)
  gpt2_lm  GPT-2 small, logit margin,   rho_rel 0.002 / 0.005 / 0.01         jmlr/lm/fq/L<ll>/<unit>.json              (2,160)
  oss_a    gpt-oss-20b, head's output,  rho_rel 2e-4 / 4e-4 / 1e-3 / 2e-3    gpt_oss/f/L<ll>/<unit>.json               (3,840)
  oss_lm   gpt-oss-20b, logit margin,   rho_rel 2e-4 / 4e-4 / 1e-3           gpt_oss/l/L<ll>/<unit>.json               (2,880)

(the records of ``experiments/exact_fixed_query.py --part sdp``, ``experiments/jmlr_extensions.py --part lm`` and
``experiments/gpt_oss.py --part f l``).

Certificate. In the coordinates the certificate uses (``u_j`` in a ball of radius ``rho_j``; the query token, and for
gpt-oss-20b the sink, are points), with ``b_j`` the gradient of the score ``s_j`` in ``u_j`` (affine for a fixed query;
for gpt-oss-20b the key-dependent ``M_j`` is carried by the per-token rotated coordinates of :mod:`aimp.gptoss`) and ``g_j``
the gradient of the value ``h_j = c^T N z_j``::

  Delta_s = max_j rho_j ||b_j||,  nu_c = max_j rho_j ||g_j||,  rad_c = (max_j h_max_j - min_j h_min_j) / 2 (all keys,
  the fixed query and the sink, value 0, included),
  r_T = phi(zbar) + sum_j rho_j ||grad_{u_j} phi(zbar)|| + rad_c Delta_s^2 + Delta_s nu_c,
  grad_{u_j} phi(zbar) = a_j g_j + a_j (h_j - phi) b_j  (nominal a, h, phi).

It is valid because along every segment from ``zbar`` to an admissible point ``|phi''(t)| <= 2 rad_c Delta_s^2 + 2 Delta_s
nu_c``. ``--selftest`` checks the gradient against :func:`aimp.heads.grad_phi`, the analytic ``phi''`` against finite
differences of the exact forward pass, the curvature bound on random segments and ``r_T >= phi`` at their ends.

Usage::

    python experiments/review_taylor.py --selftest
    python experiments/review_taylor.py --workers 12          # one JSON per unit under <out>/taylor/<set>/
    python experiments/review_taylor.py --quick               # the units of the other scripts' --quick runs
"""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np

import review_common as C

SETS = {
    "gpt2_a": {"model": "gpt2", "spec": "a", "dir": ("exact_fixed_query{}", "sdp"), "rhos": (0.002, 0.005, 0.01, 0.02)},
    "gpt2_lm": {"model": "gpt2", "spec": "lm", "dir": ("jmlr{}", "lm", "fq"), "rhos": (0.002, 0.005, 0.01)},
    "oss_a": {"model": "gpt-oss-20b", "spec": "a", "dir": ("gpt_oss{}", "f"), "rhos": (0.0002, 0.0004, 0.001, 0.002)},
    "oss_lm": {"model": "gpt-oss-20b", "spec": "lm", "dir": ("gpt_oss{}", "l"), "rhos": (0.0002, 0.0004, 0.001)},
}


def set_dir(refs: Path, sfx: str, set_name: str) -> Path:
    d = SETS[set_name]["dir"]
    return refs.joinpath(d[0].format(sfx), *d[1:])


def record_files(refs: Path, sfx: str, set_name: str) -> list[Path]:
    s = SETS[set_name]
    return [f for f in sorted(set_dir(refs, sfx, set_name).glob("L*/*.json"))
            if C.parse(f)[3] == C.POS and C.parse(f)[4] in s["rhos"]]


def instance(set_name: str, l: int, h: int, p: int, rho: float):
    from aimp import gpt2_units as U

    s = SETS[set_name]
    return U.make_instance(l, h, p, C.POS, rho, fixed_query=True, spec=s["spec"], model=s["model"])


# ------------------------------------------------------------------------------------------------------------ the bound
def taylor(inst) -> dict:
    """The quantities of the certificate on a fixed-query instance (module docstring); per key in ``inst.keys`` order."""
    from aimp import sets as S
    from aimp.heads import softmax

    M, N, c = inst.head.M, inst.head.N, np.asarray(inst.c, float)
    Zc = inst.centers()
    q = Zc[inst.query]
    Mtq, Ntc = M.T @ q, N.T @ c
    keys = list(inst.keys)
    sbar = np.array([float(Mtq @ Zc[j]) for j in keys])
    hbar = np.array([float(Ntc @ Zc[j]) for j in keys])
    a = softmax(sbar)
    phi = float(a @ hbar)
    rho = np.zeros(len(keys))
    bn = np.zeros(len(keys))
    gn = np.zeros(len(keys))
    gradn = np.zeros(len(keys))
    B_of = {}
    for k, j in enumerate(keys):
        rep = S.affine(inst.sets[j])
        if rep.r == 0 or rep.base == "point":
            continue
        if rep.base != "ball":
            raise ValueError(f"key {j}: base {rep.base} (the certificate is written for balls in base coordinates)")
        b = rep.B.T @ Mtq
        g = rep.B.T @ Ntc
        grad = a[k] * g + a[k] * (hbar[k] - phi) * b
        rho[k], bn[k], gn[k], gradn[k] = rep.rho, np.linalg.norm(b), np.linalg.norm(g), np.linalg.norm(grad)
        B_of[j] = (rep.B, b, g, grad, rep.rho)
    ds, nu = rho * bn, rho * gn
    h_max, h_min = hbar + nu, hbar - nu
    Delta_s, nu_c = float(ds.max()), float(nu.max())
    rad_c = 0.5 * float(h_max.max() - h_min.min())
    lin = float(rho @ gradn)
    curv_rad, curv_cov = rad_c * Delta_s ** 2, Delta_s * nu_c
    return {"phi_nominal": phi, "lin_term": lin, "Delta_s": Delta_s, "nu_c": nu_c, "rad_c": rad_c,
            "curv_rad": curv_rad, "curv_cov": curv_cov, "curv_term": curv_rad + curv_cov,
            "r_lin": phi + lin, "r_T": phi + lin + curv_rad + curv_cov,
            "n": len(keys), "n_var": int((rho > 0).sum()), "s_nom": sbar, "a_nom": a, "h_nom": hbar,
            "s_lo_rec": sbar - ds, "s_hi_rec": sbar + ds, "h_max": h_max, "h_min": h_min, "rho_j": rho,
            "_B": B_of}


def reference(rec: dict) -> dict:
    """``p*`` (midpoint of the exact bracket; a bracket that is not certified to ``1e-10`` of the value spread keeps its
    midpoint), its ends, and the exact range of motion."""
    lo, hi = rec["p_fq_lo"], rec["p_fq_hi"]
    p = rec.get("p_fq")
    spread = rec.get("spread_fq")
    if p is None:
        p = 0.5 * (lo + hi)
        if rec.get("min_fq_lo") is not None and rec.get("min_fq_hi") is not None:
            spread = p - 0.5 * (rec["min_fq_lo"] + rec["min_fq_hi"])
    return {"p_star": p, "p_star_lo": lo, "p_star_hi": hi, "spread": spread, "exact_status": rec.get("exact_status")}


def evaluate(set_name: str, f: Path, refs: Path) -> dict:
    t0 = time.perf_counter()
    l, h, p, _, rho = C.parse(f)
    rec = C.read_json(f)
    T = taylor(instance(set_name, l, h, p, rho))
    ref = reference(rec)
    ps, hi = ref["p_star"], ref["p_star_hi"]
    cert = rec["cert"]
    gap_T, gap_c = T["r_T"] - ps, cert - ps
    sp = ref["spread"]
    return {"set": set_name, "model": SETS[set_name]["model"], "spec": SETS[set_name]["spec"], "layer": l, "head": h,
            "prompt": p, "pos": C.POS, "rho_rel": rho, "unit": f.stem, "record": C.relpath(f, refs),
            **{k: T[k] for k in ("n", "n_var", "phi_nominal", "lin_term", "Delta_s", "nu_c", "rad_c", "curv_rad",
                                 "curv_cov", "curv_term", "r_lin", "r_T")},
            **ref, "cert": cert, "r_greedy": rec.get("r_greedy"), "r_VS": rec.get("r_VS"), "w_t": rec.get("w_t"),
            "Delta": rec.get("Delta"), "S": rec.get("S"), "route": rec.get("route"),
            "gap_T": gap_T, "gap_lin": T["r_lin"] - ps, "gap_cert": gap_c,
            "rel_gap_T": gap_T / sp if sp else None, "rel_gap_cert": gap_c / sp if sp else None,
            "rel_gap_greedy": (rec["r_greedy"] - ps) / sp if sp and rec.get("r_greedy") is not None else None,
            "ratio_T_to_cert": gap_T / gap_c if gap_c > 0 else None,
            "valid_T_strict": bool(T["r_T"] >= hi),
            "valid_T": bool(T["r_T"] >= hi - C.VALID_TOL * max(1.0, abs(hi))),
            "margin_T_over_hi": T["r_T"] - hi,
            # the instance is the certificate's: reconstruction checks against the stored record
            "d_phi_nominal": abs(T["phi_nominal"] - rec["phi_nominal"]) if rec.get("phi_nominal") is not None else None,
            "d_rad_c": abs(T["rad_c"] - rec["rad_c"]) if rec.get("rad_c") is not None else None,
            "d_s_interval": (float(np.max(np.abs(np.r_[T["s_lo_rec"] - np.asarray(rec["s_lo"]),
                                                       T["s_hi_rec"] - np.asarray(rec["s_hi"])])))
                             if rec.get("s_lo") is not None else None),
            "d_h_max": (float(np.max(np.abs(T["h_max"] - np.asarray(rec["h_max"])))) if rec.get("h_max") is not None else None),
            "wall_s": time.perf_counter() - t0}


def unit_path(out: Path, set_name: str, f: Path) -> Path:
    return out / "taylor" / set_name / f.parent.name / f.name


def run_task(task: tuple) -> dict:
    """One (set, layer, head): every prompt and radius (the projection and weights are cached per process)."""
    set_name, files, out, refs = task
    name = f"{set_name}:{Path(files[0]).stem[:6]}"
    try:
        t0, bad = time.perf_counter(), []
        for fs in files:
            f = Path(fs)
            path = unit_path(out, set_name, f)
            if path.exists():
                continue
            r = evaluate(set_name, f, refs)
            C.write_unit(path, r)
            if not r["valid_T"]:
                bad.append(f"{f.stem}: r_T below p*")
        return {"name": name, "t": time.perf_counter() - t0, "bad": bad}
    except Exception as e:
        return {"name": name, "fatal": repr(e), "tb": traceback.format_exc()[-3000:]}


def tasks(out: Path, refs: Path, sfx: str) -> list[tuple]:
    res = []
    for s in SETS:
        groups: dict = {}
        for f in record_files(refs, sfx, s):
            groups.setdefault(C.parse(f)[:2], []).append(str(f))
        res += [(s, fs, out, refs) for _, fs in sorted(groups.items())]
    return res


# ------------------------------------------------------------------------------------------------------------ self-test
def phi_along(inst, B_of: dict, deltas: dict, t: np.ndarray) -> np.ndarray:
    from aimp.heads import forward

    Zc = inst.centers()
    vals = []
    for tt in t:
        Z = Zc.copy()
        for j, (B, *_rest) in B_of.items():
            Z[j] = Zc[j] + B @ (tt * deltas[j])
        vals.append(forward(inst, Z)["phi"])
    return np.array(vals)


def selftest(refs: Path, sfx: str, n_per_set: int = 3, n_dirs: int = 40, seed: int = 20261105) -> dict:
    """(1) ``grad_{u_j} phi(zbar)`` against :func:`aimp.heads.grad_phi` mapped through ``B_j``; (2) the analytic ``phi''``
    along random segments from ``zbar`` (directions on and inside the balls, and the maximizer of the linear term) against
    central finite differences of the exact forward pass; (3) ``max |phi''| <= 2 rad_c Delta_s^2 + 2 Delta_s nu_c``;
    (4) ``r_T >= phi`` at the segment ends. Sets without records are skipped."""
    from aimp.heads import grad_phi, softmax

    rng = np.random.default_rng(seed)
    rows = []
    for s in SETS:
        files = record_files(refs, sfx, s)
        if not files:
            continue
        pick = [files[i] for i in rng.choice(len(files), size=min(n_per_set, len(files)), replace=False)]
        for f in pick:
            l, h, p, _, rho = C.parse(f)
            inst = instance(s, l, h, p, rho)
            T = taylor(inst)
            _, Gr = grad_phi(inst, inst.centers())
            gerr = max(float(np.max(np.abs(B.T @ Gr[j] - grad))) / max(1e-300, float(np.max(np.abs(grad))) + 1e-300)
                       for j, (B, b, g, grad, r) in T["_B"].items())
            bound2 = 2 * T["rad_c"] * T["Delta_s"] ** 2 + 2 * T["Delta_s"] * T["nu_c"]
            worst, fd_err, end_ok = 0.0, 0.0, True
            for d in range(n_dirs):
                deltas = {}
                for j, (B, b, g, grad, r) in T["_B"].items():
                    if d == 0:  # the maximizer of the linear term
                        v = grad / max(np.linalg.norm(grad), 1e-300)
                    else:
                        v = rng.standard_normal(B.shape[1])
                        v /= np.linalg.norm(v)
                        if d % 2:
                            v *= rng.uniform() ** (1.0 / B.shape[1])
                    deltas[j] = r * v
                sig = np.zeros(T["n"])
                gam = np.zeros(T["n"])
                for k, j in enumerate(inst.keys):
                    if j in T["_B"]:
                        B, b, g, grad, r = T["_B"][j]
                        sig[k], gam[k] = float(b @ deltas[j]), float(g @ deltas[j])
                d2 = []
                for tt in np.linspace(0.0, 1.0, 41):
                    a = softmax(T["s_nom"] + tt * sig)
                    hh = T["h_nom"] + tt * gam
                    sb = float(a @ sig)
                    V = float(a @ (sig - sb) ** 2)
                    d2.append(float(a @ (((sig - sb) ** 2 - V) * hh)) + 2 * float(a @ ((sig - sb) * gam)))
                d2 = np.array(d2)
                worst = max(worst, float(np.max(np.abs(d2))) / bound2 if bound2 > 0 else 0.0)
                hstep = 1e-3
                for tt, an in ((0.5, d2[20]), (0.25, d2[10])):
                    v = phi_along(inst, T["_B"], deltas, np.array([tt - hstep, tt, tt + hstep]))
                    fd = (v[0] - 2 * v[1] + v[2]) / hstep ** 2
                    fd_err = max(fd_err, abs(fd - an) / max(abs(an), 1e-12 * max(1.0, abs(T["phi_nominal"]))))
                end_ok &= bool(T["r_T"] >= phi_along(inst, T["_B"], deltas, np.array([1.0]))[0])
            rows.append({"set": s, "unit": f.stem, "grad_rel_err_max": gerr, "phi2_over_bound_max": worst,
                         "phi2_fd_rel_err_max": fd_err, "r_T_ge_phi_at_ends": end_ok, "n_dirs": n_dirs,
                         "phi_nominal_err": abs(T["phi_nominal"] - C.read_json(f)["phi_nominal"])})
            print(f"selftest {s} {f.stem}: grad rel err {gerr:.1e}, max|phi''|/bound {worst:.3f}, "
                  f"phi'' vs FD rel err {fd_err:.1e}, r_T >= phi at ends {end_ok}", flush=True)
    ok = bool(rows) and all(r["grad_rel_err_max"] < 1e-8 and r["phi2_over_bound_max"] <= 1.0 and r["r_T_ge_phi_at_ends"]
                            for r in rows)
    return {"ok": ok, "rows": rows}


def main(argv=None) -> None:
    from jmlr_extensions import run_part

    from aimp import compute

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="the checks of the module docstring on sample units")
    ap.add_argument("--quick", action="store_true", help="read the records of the other scripts' --quick runs")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help="default results_dir()/review (with --quick .../review_quick)")
    ap.add_argument("--refs", type=Path, default=None, help="root of the stored records (default results_dir())")
    a = ap.parse_args(argv)
    out, refs, sfx = C.dirs(a.out, a.refs, a.quick)
    if a.workers > compute.thread_budget():
        raise SystemExit(f"--workers {a.workers} exceeds AIMP_MAX_THREADS = {compute.thread_budget()}")
    if a.selftest:
        compute.apply_thread_budget(1)
        r = selftest(refs, sfx)
        C.write_unit(out / "taylor_selftest.json", r)
        print(f"selftest ok = {r['ok']} ({len(r['rows'])} units; {out / 'taylor_selftest.json'})")
        raise SystemExit(0 if r["ok"] else 1)
    T = tasks(out, refs, sfx)
    n_units = sum(len(t[1]) for t in T)
    print(f"curvature certificate: {n_units} units in {len(T)} (set, layer, head) groups")
    run_part(run_task, [t for t in T if not all(unit_path(out, t[0], Path(f)).exists() for f in t[1])], a.workers, 1,
             out, "taylor")


if __name__ == "__main__":
    main()
