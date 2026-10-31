"""The mechanism test of Section 7: is the looseness of bound propagation on a GPT-2 block remainder the per-operation excess
that Proposition T and Lemma "chain" predict?

For a block ``l`` of GPT-2 small, one perturbed token (token 1 of the first prompt, ``||delta|| <= rho_rel ||xbar_1||``) and a
head ``h`` of block ``l + 1`` (query position 4), ``f(delta) = C_h . g(delta)`` is the remainder ``g`` of the block map read
along the head's total row ``C_h`` (:func:`aimp.block.total_rows`). Per unit ``(l, rho_rel, h)``:

* CROWN's bounds ``l_tot <= f <= v_tot`` on the remainder network (:func:`aimp.block.remainder_net`, auto_LiRPA);
* the attained maximum ``att_max`` (and minimum ``att_min``): multistart projected gradient ascent with random samples
  (:func:`aimp.block.attack_total_rows`), refined by 200 steps from ``± rho v_max(H_0)`` (``± rho v_min`` for the minimum);
* ``H_0``, the Hessian of ``f`` at 0 (:func:`aimp.mechanism.hessian_t`), with a finite-difference check; its decomposition
  eq. (H0) over the operations of the block for both readings of the softmax (one operation, as CROWN bounds it: ``P_T_p``;
  exponentials, a sum, a reciprocal and products: ``P_T_p_complex``) with the identity ``sum_o Q_o = H_0``; ``s_spec``, the
  traces, the effective rank, the per-operation sum ``s_op``, the partition bounds ``L`` and the chains of Lemma "chain";
* the per-operation terms on CROWN's own operand intervals (Proposition T(iii)): ``P_TC_p``, ``ratio_T2_crownh = v_tot /
  P_TC_p``.

Full settings: blocks 0-1, 3-4, 6-7, 9-10 x ``rho_rel`` in {0.0005, 0.001, 0.002} without blocks 0-1 at 0.002 (CROWN's box of
the remainder is vacuous there) = 11 configurations x 12 heads = 132 units. One JSON (and npz) per configuration, written
atomically; reruns skip finished configurations. At the end ``mechanism.csv`` (one row per unit, the columns of
``figdata/P5c_mechanism.csv``) and the statistics quoted in Section 7.

Usage (from the repository root)::

    python experiments/mechanism.py                    # full settings, 132 units (about 5 min and 5 GB per configuration
                                                       # with 4 threads: about 50 min in total)
    python experiments/mechanism.py --quick            # blocks 9-10, rho_rel 0.0005, heads 0-2 (about 4 min)
    python experiments/mechanism.py --workers 2 --threads-per-solve 2 --layers 9 --rhos 0.0005 0.001
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from aimp import compute  # noqa: E402
from aimp.io import env_info, git_commit, jsonable, peak_rss_gb, results_dir, write_json_atomic  # noqa: E402

RHOS = (0.0005, 0.001, 0.002)
CONFIGS = tuple((l, r) for r in RHOS for l in (0, 3, 6, 9) if not (l == 0 and r == 0.002))
QUICK = ((9, 0.0005), (0, 1, 2))
ALL_HEADS = tuple(range(12))
FD_STEPS = (1e-4, 1e-5, 1e-6)  # x ||xbar_1||: central-difference steps of the H_0 check (best per vector recorded)
FD_TOL = 1e-6  # the finite-difference check of H_0 (relative)
ID_TOL = 1e-8  # the identity sum_o Q_o = H_0 (relative Frobenius error)
CHAIN_TOL = 1e-9  # the chains of Lemma "chain" (relative, plus the decomposition's own rounding)
CONT_REL, CONT_ABS = 1e-9, 1e-12  # attained values inside CROWN's [l_tot, v_tot] up to this
STEPS = 200  # projected-gradient steps of the refinement
MECH_SEED = 21_300  # + 10 l + round(1e4 rho_rel): tape gate, structure check (+1), finite differences (+2)
WIDTH_SEED = 21_350  # + 10 l + round(1e4 rho_rel) + 1: the random input that matches operands to auto_LiRPA's nodes
START_SEED = 18_700  # + 10 l: attacked remainder of the query token (a start of the multistart ascent)
ATTACK_SEED = 20_301  # + 10 l: the multistart ascent's samples and random starts
#: the columns of ``figdata/P5c_mechanism.csv`` (``a2b_max`` / ``a2b_min``: the multistart ascent's attained values; ``P_T_*``:
#: ``s_op`` with the softmax as one operation, ``*_complex``: decomposed; ``P_TC_*``: the terms on CROWN's own intervals)
COLUMNS = ("pair", "layer", "rho_rel", "head", "rho", "lam_max", "lam_min", "s_spec_p", "s_spec_m", "half_trace_p",
           "half_trace_m", "eff_rank_p", "eff_rank_m", "v_tot", "l_tot", "att_max", "att_min", "a2b_max", "a2b_min",
           "refine_max", "refine_min", "ratio_T1", "ratio_T1_lower", "ratio_T2", "ratio_T2_lower", "ratio_T2c",
           "ratio_T2c_lower", "T3_x", "T3_y", "H0_asym_rel", "att_max_source", "att_min_source", "psi_at_vmax_plus",
           "psi_at_vmax_minus", "P_T_p", "P_T_m", "P_Tc_p", "P_Tc_m", "L_S_p", "L_S_m", "L_layer_p", "L_layer_m",
           "L_node_p", "L_node_m", "P_T_p_complex", "P_T_m_complex", "L_S_p_complex", "softmax_sdp_p", "softmax_sign_p",
           "softmax_sdp_m", "softmax_sign_m", "identity_rel", "identity_rel_complex", "fd_max_rel", "chains_ok",
           "chains_ok_complex", "chain_min_margin", "contained", "contain_excess", "ratio_T2_complex",
           "ratio_T2_complex_lower", "P_T_over_trace", "L_S_over_spec", "P_T_over_L_S", "v_tot_over_spec",
           "third_order_asym", "P_TC_p", "P_TC_m", "P_TCc_p", "ratio_T2_crownh", "ratio_T2_crownh_lower",
           "ratio_T2_crownh_sqr", "interval_inflation_P")


def config_name(l: int, rho_rel: float, heads=ALL_HEADS) -> str:
    tail = "" if tuple(heads) == ALL_HEADS else "_h" + "-".join(str(h) for h in heads)
    return f"L{l:02d}_r{rho_rel:g}{tail}"


def _le(a: float, b: float, slack: float) -> dict:
    """``a <= b`` up to ``CHAIN_TOL`` relative plus ``slack`` (the decomposition's own rounding)."""
    tol = CHAIN_TOL * max(abs(a), abs(b)) + slack
    return {"a": a, "b": b, "ok": bool(a <= b + tol), "margin_rel": (b - a) / max(abs(b), 1e-300)}


def _contained(v: float, lo: float, hi: float) -> tuple[bool, float]:
    ex = max(lo - (CONT_REL * abs(lo) + CONT_ABS) - v, v - hi - (CONT_REL * abs(hi) + CONT_ABS))
    return bool(ex <= 0), float(ex)


def _ratio(a, b):
    return a / b if b > 0 else None


# ================================================================================================ one configuration
def evaluate(l: int, rho_rel: float, heads=ALL_HEADS) -> tuple[dict, dict]:
    """The mechanism test on one configuration (block ``l``, ``rho_rel``) for the downstream heads ``heads``. Returns the
    record (JSON) and the arrays (eigenvalues, extreme eigenvectors and attained points of every unit)."""
    import torch

    from aimp import block as B
    from aimp import mechanism as MC

    t0 = time.perf_counter()
    T_ = {}
    heads = tuple(heads)
    blk = B.load_block(l)
    Bk, X, J, zb = blk["B"], blk["X"], blk["J"], blk["zbar"]
    T, d = X.shape
    pert = list(B.PERTURBED)
    rv = B.radii(X, rho_rel, pert)
    k0 = pert[0]
    rk, xnorm = float(rv[k0]), float(np.linalg.norm(X[k0]))
    infl = B.influenced(J, pert)
    Cs, gam_resid = B.total_rows(l, zb, infl, heads)
    seed = MECH_SEED + 10 * l + int(round(rho_rel * 1e4))
    rec = {"layer": l, "pair": f"{l}-{l + 1}", "rho_rel": rho_rel, "rho": rv, "pert": pert, "influenced": infl,
           "heads": list(heads), "block_gate": blk["gate"], "gam_resid": gam_resid, "invalid": [], "violation": [],
           "threads": torch.get_num_threads(), "auto_lirpa": __import__("auto_LiRPA").__version__}
    T_["block_s"] = time.perf_counter() - t0
    if not blk["gate"]["ok"]:
        rec["invalid"].append(f"block gate failed: {blk['gate']}")
        return {**rec, "times": {**T_, "total_s": time.perf_counter() - t0}}, {}
    # the tape against the numpy block at 0 and at 20 random inputs
    Ds = B.sample_deltas(np.random.default_rng(seed), 20, rv, d)
    with torch.no_grad():
        dev = float(np.max(np.abs(MC.tape(Bk, X, pert, torch.zeros(d, dtype=torch.float64))[0].numpy() - zb)))
        for s in range(Ds.shape[0]):
            ref = Bk.forward(X + Ds[s])["zp"]
            dev = max(dev, float(np.max(np.abs(MC.tape(Bk, X, pert, torch.as_tensor(Ds[s, k0]))[0].numpy() - ref))))
    rec["tape_gate"] = {"dev_max_abs": dev, "ok": bool(dev <= 1e-12 * max(1.0, float(np.max(np.abs(zb)))))}
    if not rec["tape_gate"]["ok"]:
        rec["invalid"].append(f"tape gate failed (deviation {dev:.2e})")
        return {**rec, "times": {**T_, "total_s": time.perf_counter() - t0}}, {}
    # CROWN on the remainder network: the tape's operations = auto_LiRPA's nonlinear nodes; the total rows (with their
    # linear coefficients); the half-widths of the intervals CROWN relaxed every operation on
    t1 = time.perf_counter()
    mod = B.remainder_net(Bk, X, pert, J, zb)
    bm, bx = B.bounded(mod, rv, pert)
    st = MC.structure_check(bm, Bk, X, pert, rv, seed + 1)
    rec["structure"] = {k: v for k, v in st.items() if k != "pairs"}
    if not st["ok"]:
        rec["invalid"].append(f"tape differs from auto_LiRPA's graph: missing {st['missing']}, extra {st['extra_nodes']}")
        return {**rec, "times": {**T_, "total_s": time.perf_counter() - t0}}, {}
    cr = B.crown_rows(bm, bx, Cs, pert, rv)
    T_["crown_s"] = time.perf_counter() - t1
    hw = MC.crown_halfwidths(bm, Bk, X, pert, rv, st["pairs"], WIDTH_SEED + 10 * l + int(round(rho_rel * 1e4)) + 1)
    del bm, bx, mod
    # operand Jacobians (shared by the units) and adjoints (per unit), both softmax readings
    t2 = time.perf_counter()
    shifts = MC.nominal_shifts(Bk, X, pert)
    jl, _ = MC.operand_jacobians(Bk, X, pert, "lse")
    jc, _ = MC.operand_jacobians(Bk, X, pert, "complex", shifts)
    lam_l = MC.adjoints(Bk, X, pert, Cs, "lse")
    lam_c = MC.adjoints(Bk, X, pert, Cs, "complex", shifts)
    T_["jacobians_s"] = time.perf_counter() - t2
    # interval inflation: CROWN's half-width over the linearized half-width rho ||g|| per operand group
    infl_g: dict = {}
    for o in jl:
        for Jx, h_ in zip(o["jac"], hw[o["name"]]):
            gn = rk * np.linalg.norm(Jx.reshape(Jx.shape[0], -1).numpy(), axis=0)
            hv = np.asarray(h_).reshape(-1)
            ok = gn > 1e-14 * max(1.0, float(gn.max()) if gn.size else 1.0)
            if ok.any():
                infl_g.setdefault(o["group"], []).extend((hv[ok] / gn[ok]).tolist())
    rec["inflation"] = {g: {"median": float(np.median(v)), "q25": float(np.quantile(v, 0.25)),
                            "q75": float(np.quantile(v, 0.75)), "max": float(np.max(v)), "n": len(v)}
                        for g, v in infl_g.items()}
    # per unit: H_0, spectral terms, the decomposition eq. (H0), the chains of Lemma "chain", the terms on CROWN's intervals
    t3 = time.perf_counter()
    units, arr, fd_rng = {}, {}, np.random.default_rng(seed + 2)
    for i, h in enumerate(heads):
        H0, asym, gfun = MC.hessian_t(Bk, X, pert, Cs[i])
        fd = MC.fd_check(H0, gfun, xnorm, fd_rng, 5, FD_STEPS)
        sp = MC.spectral(H0, rk)
        u = {"head": h, "H0_asym_rel": asym, "fd": fd, "fd_ok": bool(fd["max_best"] <= FD_TOL),
             **{k: v for k, v in sp.items() if k not in ("eig", "v_max", "v_min")}}
        arr[f"h{h}|eig"], arr[f"h{h}|v_max"], arr[f"h{h}|v_min"] = sp["eig"], sp["v_max"], sp["v_min"]
        nH = float(np.linalg.norm(H0))
        for tag, jops, lam in (("lse", jl, lam_l[i]), ("complex", jc, lam_c[i])):
            recs = MC.decompose(jops, lam, rk)
            Hd = sum(r["Q"] for r in recs)
            err = float(np.linalg.norm(Hd - H0)) / max(nH, 1e-300)
            slack = 0.5 * rk ** 2 * float(np.linalg.norm(Hd - H0, 2))
            Pp, Pm = sum(r["Pp"] for r in recs), sum(r["Pm"] for r in recs)
            parts = {"group": MC.grouped(recs, rk, lambda r: r["group"]),
                     "layer": MC.grouped(recs, rk, lambda r: MC.LAYER_OF[r["group"]]),
                     "node": MC.grouped(recs, rk, lambda r: r["name"])}
            chains = {"spec<=tr+": _le(sp["s_spec_p"], sp["half_trace_p"], slack),
                      "tr+<=PT+": _le(sp["half_trace_p"], Pp, slack),
                      "spec<=tr-": _le(sp["s_spec_m"], sp["half_trace_m"], slack),
                      "tr-<=PT-": _le(sp["half_trace_m"], Pm, slack)}
            for pn, ps in parts.items():
                chains[f"spec<=L+[{pn}]"] = _le(sp["s_spec_p"], ps["L_p"], slack)
                chains[f"L+<=PT+[{pn}]"] = _le(ps["L_p"], Pp, slack)
                chains[f"spec<=L-[{pn}]"] = _le(sp["s_spec_m"], ps["L_m"], slack)
                chains[f"L-<=PT-[{pn}]"] = _le(ps["L_m"], Pm, slack)
            sm = [r for r in recs if r["kind"] == "softmax"]
            u[tag] = {"identity_rel": err, "identity_ok": bool(err <= ID_TOL), "P_T_p": Pp, "P_T_m": Pm,
                      "P_Tc_p": sum(r["Pcp"] for r in recs), "P_Tc_m": sum(r["Pcm"] for r in recs),
                      "chains": chains, "chains_ok": all(c["ok"] for c in chains.values()),
                      "L_S_p": parts["group"]["L_p"], "L_S_m": parts["group"]["L_m"],
                      "L_layer_p": parts["layer"]["L_p"], "L_layer_m": parts["layer"]["L_m"],
                      "L_node_p": parts["node"]["L_p"], "L_node_m": parts["node"]["L_m"],
                      "groups": parts["group"]["groups"], "layers": parts["layer"]["groups"],
                      "softmax_sign_p": sum(r["sign_p"] for r in sm), "softmax_sign_m": sum(r["sign_m"] for r in sm),
                      "softmax_sdp_p": sum(r["Pp"] for r in sm), "softmax_sdp_m": sum(r["Pm"] for r in sm),
                      "softmax_flags": sorted({f for r in sm for f in r["sdp_flags"]})}
        w = MC.p_t_widths(jl, lam_l[i], hw)
        groups: dict = {}
        for e in w["per_op"].values():
            g = groups.setdefault(e["group"], {"P_p": 0.0, "P_m": 0.0})
            g["P_p"] += e["P_p"]
            g["P_m"] += e["P_m"]
        u["crown_widths"] = {"P_TC_p": w["P_p"], "P_TC_m": w["P_m"], "P_TCc_p": w["Pc_p"], "P_TCc_m": w["Pc_m"],
                             "groups": groups}
        units[h] = u
    T_["units_s"] = time.perf_counter() - t3
    del jl, jc, lam_l, lam_c, hw
    # attacks: the multistart ascent on ± f (one start: the attacked remainder of the query token) ...
    t4 = time.perf_counter()
    ar = B.attacked_remainder(Bk, X, J, zb, rv, B.POS, START_SEED + 10 * l)
    rec["start_attack"] = {"value": ar["value"], "sampled": ar["sampled"], "nit": ar["nit"]}
    ms = B.attack_total_rows(Bk, X, J, zb, rv, pert, Cs, cr["LamU"], cr["LamL"], ar["delta"], ATTACK_SEED + 10 * l, heads,
                             B.n_unit_rows(infl), STEPS)
    # ... and the refinement from ± rho v_max (upper side) and ± rho v_min (lower side)
    Cl, sg, D0, key = [], [], [], []
    for i, h in enumerate(heads):
        for side, vec in ((1.0, arr[f"h{h}|v_max"]), (-1.0, arr[f"h{h}|v_min"])):
            for s_ in (1.0, -1.0):
                Cl.append(Cs[i])
                sg.append(side)
                D0.append((s_ * rk * vec)[None])
                key.append((h, side, s_))
    D0 = np.array(D0)
    Cm = np.array(Cl)
    res = B.pgd_rows(Bk, X, J, zb, rv, pert, Cm, np.array(sg), D0, STEPS)

    def psi_np(Dk):  # f at the points Dk (numpy forward pass; one point per problem)
        Df = np.zeros((len(Dk), T, d))
        Df[:, pert] = Dk
        return np.einsum("nc,nc->n", B.numpy_remainder(Bk, X, J, zb, Df).reshape(len(Dk), -1), Cm)

    vb, vs = psi_np(res["delta"]), psi_np(D0)
    T_["attack_s"] = time.perf_counter() - t4
    for i, h in enumerate(heads):
        u = units[h]
        idx = {kk: n for n, kk in enumerate(key) if kk[0] == h}
        up = [vb[idx[(h, 1.0, s_)]] for s_ in (1.0, -1.0)]
        lo = [vb[idx[(h, -1.0, s_)]] for s_ in (1.0, -1.0)]
        u["psi_at_vmax"] = [float(vs[idx[(h, 1.0, s_)]]) for s_ in (1.0, -1.0)]
        u["psi_at_vmin"] = [float(vs[idx[(h, -1.0, s_)]]) for s_ in (1.0, -1.0)]
        u["refine_max"], u["refine_min"] = float(max(up)), float(min(lo))
        u["multistart"] = {k: ms[k][i] for k in ms}
        u["multistart_max"], u["multistart_min"] = ms["max"][i], ms["min"][i]
        u["att_max"] = max(u["refine_max"], u["multistart_max"])
        u["att_min"] = min(u["refine_min"], u["multistart_min"])
        u["att_max_source"] = "refine" if u["refine_max"] > u["multistart_max"] else "multistart"
        u["att_min_source"] = "refine" if u["refine_min"] < u["multistart_min"] else "multistart"
        u["v_tot"], u["l_tot"] = float(cr["v"][i]), float(cr["l"][i])
        ok1, ex1 = _contained(u["att_max"], u["l_tot"], u["v_tot"])
        ok2, ex2 = _contained(u["att_min"], u["l_tot"], u["v_tot"])
        u["contained"], u["contain_excess"] = bool(ok1 and ok2), max(ex1, ex2)
        if not u["contained"]:
            rec["violation"].append(f"head {h}: attained value outside CROWN's [l_tot, v_tot] (excess "
                                    f"{u['contain_excess']:.2e})")
        best_up = idx[(h, 1.0, 1.0)] if up[0] >= up[1] else idx[(h, 1.0, -1.0)]
        best_lo = idx[(h, -1.0, 1.0)] if lo[0] <= lo[1] else idx[(h, -1.0, -1.0)]
        arr[f"h{h}|att_max_delta"] = res["delta"][best_up][0]
        arr[f"h{h}|att_min_delta"] = res["delta"][best_lo][0]
        L, sp_p, sp_m = u["lse"], u["s_spec_p"], u["s_spec_m"]
        u["ratio_T1"] = _ratio(u["att_max"], sp_p)  # attained maximum / s_spec
        u["ratio_T1_lower"] = _ratio(-u["att_min"], sp_m)
        u["ratio_T2"] = _ratio(u["v_tot"], L["P_T_p"])  # CROWN / s_op (softmax as one operation)
        u["ratio_T2_lower"] = _ratio(-u["l_tot"], L["P_T_m"])
        u["ratio_T2c"] = _ratio(u["v_tot"], L["P_Tc_p"])
        u["ratio_T2c_lower"] = _ratio(-u["l_tot"], L["P_Tc_m"])
        u["T3_x"] = _ratio(u["v_tot"], u["att_max"])  # CROWN's looseness
        u["T3_y"] = _ratio(L["P_T_p"], sp_p)  # predicted looseness
        cw = u["crown_widths"]
        cw["ratio_v_PTC"] = _ratio(u["v_tot"], cw["P_TC_p"])  # CROWN / per-operation terms on CROWN's intervals
        cw["ratio_l_PTC"] = _ratio(-u["l_tot"], cw["P_TC_m"])
        cw["ratio_v_PTCc"] = _ratio(u["v_tot"], cw["P_TCc_p"])
        cw["inflation_P"] = _ratio(cw["P_TC_p"], L["P_T_p"])
        for tag in ("lse", "complex"):
            if not u[tag]["identity_ok"]:
                rec["invalid"].append(f"head {h}: identity ({tag}) error {u[tag]['identity_rel']:.2e} > {ID_TOL:g}")
            if not u[tag]["chains_ok"]:
                bad = [k for k, c in u[tag]["chains"].items() if not c["ok"]]
                rec["violation"].append(f"head {h}: chain ({tag}) violated: {bad}")
        if not u["fd_ok"]:
            rec["invalid"].append(f"head {h}: H_0 finite-difference check {u['fd']['max_best']:.2e} > {FD_TOL:g}")
    T_["total_s"] = time.perf_counter() - t0
    return {**rec, "units": units, "steps": STEPS, "fd_steps": FD_STEPS, "xnorm": xnorm, "times": T_,
            "git": git_commit(), "env": env_info(), "peak_rss_gb": peak_rss_gb()}, arr


# ================================================================================================ CSV and statistics
def unit_rows(rec: dict) -> list[dict]:
    """One row per unit of a configuration record, in the columns of ``figdata/P5c_mechanism.csv``."""
    def f(x):
        return math.nan if x is None or isinstance(x, str) else float(x)

    rows = []
    for hs, u in rec["units"].items():
        L, Cx, W = u["lse"], u["complex"], u["crown_widths"]
        r = {"pair": rec["pair"], "layer": rec["layer"], "rho_rel": rec["rho_rel"], "head": int(hs),
             "rho": f(rec["rho"][rec["pert"][0]])}
        for k in ("lam_max", "lam_min", "s_spec_p", "s_spec_m", "half_trace_p", "half_trace_m", "eff_rank_p", "eff_rank_m",
                  "v_tot", "l_tot", "att_max", "att_min", "refine_max", "refine_min", "ratio_T1", "ratio_T1_lower",
                  "ratio_T2", "ratio_T2_lower", "ratio_T2c", "ratio_T2c_lower", "T3_x", "T3_y", "H0_asym_rel"):
            r[k] = f(u[k])
        r["a2b_max"], r["a2b_min"] = f(u["multistart_max"]), f(u["multistart_min"])  # column names of the figure data
        r.update({"att_max_source": u["att_max_source"], "att_min_source": u["att_min_source"],
                  "psi_at_vmax_plus": f(u["psi_at_vmax"][0]), "psi_at_vmax_minus": f(u["psi_at_vmax"][1])})
        for k in ("P_T_p", "P_T_m", "P_Tc_p", "P_Tc_m", "L_S_p", "L_S_m", "L_layer_p", "L_layer_m", "L_node_p",
                  "L_node_m"):
            r[k] = L[k]
        r.update({"P_T_p_complex": Cx["P_T_p"], "P_T_m_complex": Cx["P_T_m"], "L_S_p_complex": Cx["L_S_p"],
                  "softmax_sdp_p": L["softmax_sdp_p"], "softmax_sign_p": L["softmax_sign_p"],
                  "softmax_sdp_m": L["softmax_sdp_m"], "softmax_sign_m": L["softmax_sign_m"],
                  "identity_rel": L["identity_rel"], "identity_rel_complex": Cx["identity_rel"],
                  "fd_max_rel": u["fd"]["max_best"], "chains_ok": L["chains_ok"], "chains_ok_complex": Cx["chains_ok"],
                  "chain_min_margin": min(c["margin_rel"] for c in L["chains"].values()),
                  "contained": u["contained"], "contain_excess": u["contain_excess"]})
        r["ratio_T2_complex"] = r["v_tot"] / r["P_T_p_complex"] if r["P_T_p_complex"] > 0 else math.nan
        r["ratio_T2_complex_lower"] = -r["l_tot"] / r["P_T_m_complex"] if r["P_T_m_complex"] > 0 else math.nan
        r["P_T_over_trace"] = r["P_T_p"] / r["half_trace_p"] if r["half_trace_p"] > 0 else math.nan
        r["L_S_over_spec"] = r["L_S_p"] / r["s_spec_p"] if r["s_spec_p"] > 0 else math.nan
        r["P_T_over_L_S"] = r["P_T_p"] / r["L_S_p"] if r["L_S_p"] > 0 else math.nan
        r["v_tot_over_spec"] = r["v_tot"] / r["s_spec_p"] if r["s_spec_p"] > 0 else math.nan
        s = r["psi_at_vmax_plus"] + r["psi_at_vmax_minus"]
        r["third_order_asym"] = (r["psi_at_vmax_plus"] - r["psi_at_vmax_minus"]) / s if s != 0 else math.nan
        r.update({"P_TC_p": f(W["P_TC_p"]), "P_TC_m": f(W["P_TC_m"]), "P_TCc_p": f(W["P_TCc_p"]),
                  "ratio_T2_crownh": f(W["ratio_v_PTC"]), "ratio_T2_crownh_lower": f(W["ratio_l_PTC"]),
                  "ratio_T2_crownh_sqr": f(W["ratio_v_PTCc"]), "interval_inflation_P": f(W["inflation_P"])})
        rows.append({k: r[k] for k in COLUMNS})
    return rows


def print_stats(M) -> None:
    """The statistics of the mechanism test quoted in Section 7 (Figure 5), computed from the unit table ``M``."""
    from scipy.stats import spearmanr

    S = M[M.rho_rel <= 0.001]
    y = M.v_tot / M.att_max; x = M.P_T_p_complex / M.s_spec_p; x1 = M.P_T_p / M.s_spec_p  # noqa: E702
    print(f"Spearman(looseness, predicted) all: {spearmanr(x, y).statistic:.2f}, rho<=0.001: "
          f"{spearmanr(x[M.rho_rel<=0.001], y[M.rho_rel<=0.001]).statistic:.2f} "
          f"(softmax as one operation: {spearmanr(x1[M.rho_rel<=0.001], y[M.rho_rel<=0.001]).statistic:.2f})")
    print(f"attained max / s_spec, rho<=0.001: {S.ratio_T1.min():.4f} to {S.ratio_T1.max():.4f}")
    q = S.v_tot / S.P_T_p_complex
    print(f"CROWN / s_op, rho<=0.001: {q.min():.2f} to {q.max():.2f}, median {q.median():.2f}")
    print(f"CROWN / per-operation terms on CROWN's intervals, rho<=0.001: {S.ratio_T2_crownh.min():.2f} to "
          f"{S.ratio_T2_crownh.max():.2f}")
    print(f"one-operation / decomposed s_op: {(M.P_T_p/M.P_T_p_complex).min():.2f} to "
          f"{(M.P_T_p/M.P_T_p_complex).max():.2f}")
    print(f"predicted looseness: median {x.median():.0f} ({x.min():.0f} to {x.max():.0f})")
    print(f"effective rank: median {M.eff_rank_p.median():.1f} ({M.eff_rank_p.min():.1f} to {M.eff_rank_p.max():.1f})")
    print(f"per-neuron sum / half trace: median {(M.P_T_p_complex/M.half_trace_p).median():.1f}")


# ================================================================================================ driver
def _init_worker(threads: int) -> None:
    import torch  # noqa: F401  (imported first so that its intra-op pool takes the budget below)

    compute.apply_thread_budget(threads)


def run_task(task: tuple) -> dict:
    l, rho_rel, heads, out = task
    name = config_name(l, rho_rel, heads)
    path = Path(out) / f"{name}.json"
    if path.exists():
        return {"name": name, "skipped": True}
    try:
        rec, arr = evaluate(l, rho_rel, heads)
        if arr:
            tmp = path.with_name(name + ".tmp.npz")
            np.savez_compressed(tmp, **arr)
            os.replace(tmp, path.with_suffix(".npz"))
        write_json_atomic(path, jsonable(rec))
        return {"name": name, "t": rec["times"]["total_s"], "invalid": rec["invalid"], "violation": rec["violation"],
                "peak_rss_gb": rec.get("peak_rss_gb")}
    except Exception as e:
        return {"name": name, "fatal": repr(e), "tb": traceback.format_exc()[-3000:]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="one cheap configuration (blocks 9-10, rho_rel 0.0005, heads 0-2)")
    ap.add_argument("--workers", type=int, default=1, help="parallel configurations (default 1)")
    ap.add_argument("--threads-per-solve", type=int, default=None,
                    help="threads per configuration (default AIMP_MAX_THREADS // workers)")
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="restrict to these blocks l (of 0 3 6 9)")
    ap.add_argument("--rhos", type=float, nargs="*", default=None, help="restrict to these rho_rel")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default <results>/mechanism)")
    a = ap.parse_args(argv)
    budget = compute.thread_budget()
    tps = a.threads_per_solve or max(1, budget // max(1, a.workers))
    if a.workers < 1 or a.workers * tps > budget:
        ap.error(f"workers x threads-per-solve = {a.workers} x {tps} exceeds AIMP_MAX_THREADS = {budget}")
    out = (a.out or results_dir() / "mechanism").resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.quick:
        selected, heads = [QUICK[0]], QUICK[1]
    else:
        selected, heads = list(CONFIGS), ALL_HEADS
    if a.layers is not None:
        selected = [c for c in selected if c[0] in a.layers]
    if a.rhos is not None:
        selected = [c for c in selected if any(abs(c[1] - r) < 1e-12 for r in a.rhos)]
    if not selected:
        ap.error("no configuration selected")

    from aimp import block as B
    from aimp import gpt2_units

    gpt2_units.prepare()  # cached weights and activations (downloads gpt2 once)
    B.cache_blocks(sorted({l for l, _ in selected}))
    compute.apply_thread_budget(tps)
    tasks = [(l, r, heads, str(out)) for l, r in selected if not (out / f"{config_name(l, r, heads)}.json").exists()]
    print(f"mechanism test: {len(selected)} configuration(s), {len(tasks)} to run, {a.workers} worker(s) x {tps} "
          f"thread(s); output {out}", flush=True)
    t0 = time.perf_counter()
    bad = []
    if tasks:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(a.workers, len(tasks)), initializer=_init_worker, initargs=(tps,), maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(run_task, tasks):
                if r.get("fatal"):
                    print(f"  {r['name']}: FAILED {r['fatal']}\n{r['tb']}", flush=True)
                    bad.append(r["name"])
                    continue
                if r.get("skipped"):
                    continue
                print(f"  {r['name']}: {r['t']:.0f} s, peak RSS {r['peak_rss_gb'] or math.nan:.1f} GB"
                      + (f", INVALID {r['invalid']}" if r["invalid"] else "")
                      + (f", VIOLATION {r['violation']}" if r["violation"] else ""), flush=True)
                if r["invalid"] or r["violation"]:
                    bad.append(r["name"])
        print(f"wall time {time.perf_counter() - t0:.0f} s", flush=True)

    import pandas as pd

    names = {config_name(l, r, heads) for l, r in (selected if a.quick else CONFIGS)}
    recs = [json.loads(p.read_text()) for p in sorted(out.glob("*.json")) if p.stem in names]
    rows = [row for rec in recs if rec.get("units") for row in unit_rows(rec)]
    if not rows:
        print("no finished configuration")
        return 1
    M = pd.DataFrame(rows, columns=list(COLUMNS)).sort_values(["rho_rel", "layer", "head"])
    csv = out / ("mechanism_quick.csv" if a.quick else "mechanism.csv")
    tmp = csv.with_suffix(".csv.tmp")
    M.to_csv(tmp, index=False)
    os.replace(tmp, csv)
    print(f"{len(M)} units -> {csv}" + ("" if a.quick or len(M) == 132 else " (the full test has 132)"))
    print_stats(M)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
