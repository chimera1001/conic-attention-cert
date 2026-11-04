"""The gpt-oss-20b units of :mod:`aimp.gptoss` (``model="gpt-oss-20b"`` of :func:`aimp.gpt2_units.make_instance`).

Most tests build a small cache in the format of :func:`aimp.gptoss.prepare` from random weights (residual width 256, all
64 query heads and 8 key-value groups, rotary maps of the model's convention with a YaRN-like factor, one sink per head),
so they run without the 13 GB model: the rotation matrices are the model's own ``apply_rotary_pos_emb``; the reduced
instance (rotary maps folded into each token's set, the sink as a fixed key with a constant score and value zero) gives the
head's scores, pattern and output at random admissible inputs, computed from the raw weights and the full-width tokens;
E2; the score intervals contain sampled scores (the sink's is a point, the fixed-query ones are exact); the exact
fixed-query bracket contains every sampled value and lies below the first-order and vertex bounds; the worst-case inputs
rebuilt from the canonical angles lie in their planes; the auto_LiRPA module with the sink passes its forward gate; the
logit-margin vector is the frozen-scale linearization; the GPT-2 default of the unit construction is unchanged. The last
test runs E2 on real heads when the real cache exists (skipped otherwise)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from aimp import gptoss as G
from aimp import gpt2_units as U

D_SYN, T_SYN = 256, 9


def _clear():
    for c in (G._W, G._P, G._F, G._PROJ):
        c.clear()


def _rope(T: int, mscale: float = 1.35) -> tuple[np.ndarray, np.ndarray]:
    inv = 1.0 / (150000.0 ** (np.arange(0, G.DH, 2) / G.DH))
    ang = np.arange(T)[:, None] * inv[None, :]
    return mscale * np.cos(ang), mscale * np.sin(ang)


@pytest.fixture()
def syn_cache(tmp_path, monkeypatch):
    """A cache in the format of ``prepare()`` from random weights (layer 0 only, five prompts of ``T_SYN`` tokens)."""
    monkeypatch.setenv("AIMP_CACHE_DIR", str(tmp_path))
    _clear()
    c = G.model_cache()
    c.mkdir(parents=True)
    rng = np.random.default_rng(0)
    d = D_SYN
    W = {"Wq": rng.standard_normal((G.N_HEADS * G.DH, d)) / math.sqrt(d) * 3, "bq": rng.standard_normal(G.N_HEADS * G.DH) * 0.3,
         "Wk": rng.standard_normal((G.N_KV * G.DH, d)) / math.sqrt(d) * 3, "bk": rng.standard_normal(G.N_KV * G.DH) * 0.3,
         "Wv": rng.standard_normal((G.N_KV * G.DH, d)) / math.sqrt(d), "bv": rng.standard_normal(G.N_KV * G.DH) * 0.1,
         "Wo": rng.standard_normal((d, G.N_HEADS * G.DH)) / math.sqrt(G.DH), "bo": rng.standard_normal(d) * 0.1,
         "sinks": rng.standard_normal(G.N_HEADS), "ln_g": np.ones(d), "ln_eps": np.float64(1e-5), "scale": np.float64(0.125),
         "window": np.int64(128)}
    np.savez(c / "weights_L00.npz", **W)
    cos, sin = _rope(T_SYN)
    R = G.rotation_matrices(cos, sin)
    for p in range(5):
        x = rng.standard_normal((1, T_SYN, d)) * 3
        z = G.rmsnorm(x, W["ln_g"], 1e-5)
        f = G.attention_f64(z[0], W, R, 128)
        np.savez(c / f"prompt{p}.npz", ids=np.arange(T_SYN) + p, z=z, R=R, merged=f["merged"][None], attn=f["attn"][None],
                 x=x, xf=x[0])
    g_f = 1.0 + 0.1 * rng.standard_normal(d)
    rows = rng.standard_normal((4, d))
    xbar = rng.standard_normal((5, d)) * 3
    c_m = np.stack([G.lm_spec(xbar[p], g_f, 1e-5, rows[0], rows[1])["c_m"] for p in range(5)])
    np.savez(c / "final.npz", g_f=g_f, eps_f=np.float64(1e-5), lm_ids=np.arange(4), lm_rows=rows, c_m=c_m, grad_exact=c_m,
             xbar=xbar, t1=np.zeros(5, int), t2=np.ones(5, int), margin=np.ones(5), r_bar=np.ones(5), pos=np.int64(4),
             scale=np.float64(0.125), mscale=np.float64(1.35))
    yield c
    _clear()


UNITS = [(0, 0, 0, 4), (0, 9, 1, 4), (0, 36, 2, 4), (0, 63, 3, 8)]


def _direct(l, h, p, pos, Z, c):
    hw = G.head_weights(l, h)
    R = G.prompt_data(p)["R"]
    q = R[pos] @ (hw["W_Q"].T @ Z[pos] + hw["b_Q"])
    s = np.array([hw["scale"] * (R[j] @ (hw["W_K"].T @ Z[j] + hw["b_K"])) @ q for j in range(pos + 1)] + [hw["sink"]])
    a = np.exp(s - s.max())
    a /= a.sum()
    y = (a[:pos + 1] @ (Z @ hw["W_V"] + hw["b_V"])) @ hw["W_O"]
    return {"s": s, "a": a, "y": y, "phi": float(c @ y)}


def test_rotation_matrices_are_the_models_function():
    import torch
    from transformers.models.gpt_oss.modeling_gpt_oss import apply_rotary_pos_emb

    rng = np.random.default_rng(0)
    T = 7
    cos, sin = rng.standard_normal((T, G.DH // 2)), rng.standard_normal((T, G.DH // 2))
    R = G.rotation_matrices(cos, sin)
    x = rng.standard_normal((1, 3, T, G.DH))
    q, _ = apply_rotary_pos_emb(torch.as_tensor(x), torch.as_tensor(x), torch.as_tensor(cos[None]), torch.as_tensor(sin[None]))
    assert np.max(np.abs(np.einsum("tab,htb->hta", R, x[0]) - q[0].numpy())) <= 1e-14


@pytest.mark.parametrize("unit", UNITS)
def test_reduced_instance_reproduces_the_head_at_random_inputs(syn_cache, unit):
    from aimp.heads import forward

    l, h, p, pos = unit
    inst = U.make_instance(l, h, p, pos, 2e-2, model="gpt-oss-20b")
    hw, pr = G.head_projection(l, h)
    z = G.prompt_data(p)["z"][l]
    R = G.prompt_data(p)["R"]
    rng = np.random.default_rng(1 + h)
    for _ in range(5):
        Z, Zp = z[:pos + 1].copy(), inst.centers().copy()
        for t in range(pos + 1):
            u = rng.standard_normal(z.shape[1])
            u *= inst.sets[t].rho * rng.uniform(0.2, 1.0) / np.linalg.norm(u)
            Z[t] = z[t] + u
            Zp[t] = inst.sets[t].center + inst.sets[t].E @ (pr.Vt @ u)
        ref = _direct(l, h, p, pos, Z, inst.c)
        out = forward(inst, Zp)
        assert np.max(np.abs(out["s"] - ref["s"])) <= 1e-10 * max(1.0, np.max(np.abs(ref["s"])))
        assert np.max(np.abs(out["a"] - ref["a"])) <= 1e-12
        assert np.max(np.abs(inst.head.N @ (out["a"] @ Zp) - ref["y"])) <= 1e-10 * max(1.0, np.max(np.abs(ref["y"])))
        q0 = hw["W_Q"].T @ Z[pos] + hw["b_Q"]
        for j in range(pos + 1):  # the per-key matrix M_j = scale R_i^T R_j on the unrotated projections
            k0 = hw["W_K"].T @ Z[j] + hw["b_K"]
            assert abs(q0 @ (hw["scale"] * R[pos].T @ R[j]) @ k0 - ref["s"][j]) <= 1e-10 * max(1.0, abs(ref["s"][j]))


@pytest.mark.parametrize("unit", UNITS)
def test_e2_sink_and_specifications(syn_cache, unit):
    l, h, p, pos = unit
    for spec in (("a", "lm") if pos == 4 else ("a",)):
        for fixed in (False, True):
            inst = U.make_instance(l, h, p, pos, 1e-2, fixed_query=fixed, spec=spec, model="gpt-oss-20b")
            e = G.e2_check(inst)
            assert e["E2_ok"], e
        sk = inst.meta["sink_token"]
        assert inst.keys[-1] == sk and inst.sets[sk].rho == 0.0 and np.all(inst.head.N @ inst.sets[sk].center == 0.0)
        if spec == "lm":
            assert np.array_equal(inst.c, -G.final_data()["c_m"][p])
    inst = U.make_instance(l, h, p, pos, 1e-2, model="gpt-oss-20b")
    Si, sk = inst.sets[inst.query], inst.meta["sink_token"]
    rng = np.random.default_rng(3)
    for _ in range(5):
        w = rng.standard_normal(Si.E.shape[1])
        zq = Si.center + Si.E @ (w * Si.rho / np.linalg.norm(w))
        assert abs(zq @ inst.head.M @ inst.sets[sk].center - inst.meta["sink"]) <= 1e-13 * max(1.0, abs(inst.meta["sink"]))


def test_score_intervals_contain_samples(syn_cache):
    from aimp import bounds, gpt2
    from aimp.heads import forward

    inst = U.make_instance(0, 9, 1, 4, 2e-2, model="gpt-oss-20b")
    si = gpt2.score_intervals_roles(inst, threads=1)
    assert si["method"] == "shor_roles_gptoss" and si["s_lo"][-1] == si["s_hi"][-1]
    rng = np.random.default_rng(5)
    for _ in range(200):
        Z = inst.centers().copy()
        for t in range(5):
            st = inst.sets[t]
            w = rng.standard_normal(st.E.shape[1])
            Z[t] = st.center + st.E @ (w * st.rho * rng.uniform() ** 0.1 / np.linalg.norm(w))
        s = forward(inst, Z)["s"]
        assert np.all(s >= si["s_lo"] - 1e-9) and np.all(s <= si["s_hi"] + 1e-9)
    fq = U.make_instance(0, 9, 1, 4, 2e-2, fixed_query=True, model="gpt-oss-20b")
    sf = gpt2.score_intervals_roles(fq, threads=1)
    for idx, j in enumerate(fq.keys):
        lo, hi = bounds._linear_interval(fq, j)
        assert abs(sf["s_lo"][idx] - lo) <= 1e-12 * max(1.0, abs(lo)) and abs(sf["s_hi"][idx] - hi) <= 1e-12 * max(1.0, abs(hi))


def test_exact_fixed_query_unit_and_angles(syn_cache):
    from aimp import pipeline
    from aimp.heads import forward

    r = pipeline.evaluate_fq_exact(0, 36, 2, 4, 2e-2, model="gpt-oss-20b")
    assert r["exact_status"] == "exact" and r["model"] == "gpt-oss-20b"
    assert r["max_input_in_bracket"] and r["min_input_in_bracket"] and r["valid_greedy"] and r["valid_VS"]
    assert r["r_VS"] <= r["r_greedy"] + 1e-12 * max(1.0, abs(r["r_greedy"]))
    assert max(r["off_plane_max"], r["off_plane_min"]) <= 1e-9 * max(r["rho_j"])
    assert np.all(np.abs(np.asarray(r["norm_max_rel"]) - 1.0) <= 1e-9)
    fq = U.make_instance(0, 36, 2, 4, 2e-2, fixed_query=True, model="gpt-oss-20b")
    rng = np.random.default_rng(7)
    best = -np.inf
    for _ in range(300):
        Z = fq.centers().copy()
        for t in range(4):
            st = fq.sets[t]
            w = rng.standard_normal(st.E.shape[1])
            Z[t] = st.center + st.E @ (w * st.rho / np.linalg.norm(w))
        best = max(best, forward(fq, Z)["phi"])
    assert best <= r["p_fq_hi"] + 1e-9 * max(1.0, abs(r["p_fq_hi"]))


def test_lirpa_module_with_sink(syn_cache):
    from aimp import lirpa

    fq = U.make_instance(0, 0, 0, 4, 2e-2, fixed_query=True, model="gpt-oss-20b")
    b = lirpa.bounds(fq, 1, ("IBP", "CROWN"), threads=1)
    assert b["gate_ok"] and fq.meta["sink_token"] not in b["perturbed"]
    assert all(b["methods"][m]["above_samples"] for m in ("IBP", "CROWN"))


def test_lm_spec_is_the_frozen_scale_linearization():
    rng = np.random.default_rng(11)
    d = 64
    g, u1, u2, x = 1 + 0.1 * rng.standard_normal(d), rng.standard_normal(d), rng.standard_normal(d), rng.standard_normal(d)
    lm = G.lm_spec(x, g, 1e-5, u1, u2)
    rb = G.rms_scale(x, 1e-5)
    for _ in range(3):
        dl = rng.standard_normal(d) * 1e-3
        ch = (u1 - u2) @ (g * (x + dl) / rb) - (u1 - u2) @ (g * x / rb)
        assert abs(ch - lm["c_m"] @ dl) <= 1e-10 * abs(lm["c_m"] @ dl)
    eps = 1e-7  # the exact gradient (with the derivative of the RMS scale) by central differences
    f = lambda y: (u1 - u2) @ (g * y / G.rms_scale(y, 1e-5))  # noqa: E731
    num = np.array([(f(x + eps * e) - f(x - eps * e)) / (2 * eps) for e in np.eye(d)])
    assert np.max(np.abs(num - lm["grad_exact"])) <= 1e-6 * np.max(np.abs(num))


def test_gpt2_default_unchanged():
    """``model="gpt2"`` is the default path (no ``model`` in the meta; the GPT-2 score intervals); needs the GPT-2 cache."""
    from aimp.io import cache_dir

    if not (cache_dir() / "weights.npz").exists():
        pytest.skip("GPT-2 cache missing")
    a = U.make_instance(5, 1, 0, 4, 0.01)
    b = U.make_instance(5, 1, 0, 4, 0.01, model="gpt2")
    assert "model" not in a.meta and np.array_equal(a.c, b.c)
    assert all(np.array_equal(a.sets[t].center, b.sets[t].center) for t in a.sets)


def test_real_cache_e2():
    """E2 on real heads of gpt-oss-20b when ``cache_dir()/gpt-oss-20b`` holds the extraction (skipped otherwise)."""
    _clear()
    c = G.model_cache()
    if not ((c / "final.npz").exists() and (c / "extract.json").exists()):
        pytest.skip("gpt-oss-20b cache missing (python experiments/gpt_oss.py --part calib builds it)")
    rec = json.loads((c / "extract.json").read_text())
    assert rec["E1_ok"] and rec["E3_logits_ok"] and rec["E3_lin_ok"]
    for l, h, p, pos in ((0, 0, 0, 4), (11, 36, 2, 4), (23, 63, 4, 4)):
        assert G.e2_check(U.make_instance(l, h, p, pos, 1e-3, model="gpt-oss-20b"))["E2_ok"]
    _clear()


def test_classification_rule():
    from aimp.option_b import classify

    lb, ub = 5.0, 5.0 + 1e-12
    assert classify(lb + 5e-7, 1e-7, lb, ub, 1e-9, 0.0) == "satisfied"
    assert classify(lb + 3e-6, 1e-7, lb, ub, 3e-6, 0.0) == "unresolved"
    assert classify(lb + 3e-6, 1e-7, lb, ub, 1e-9, 5e-6) == "unresolved"
    assert classify(lb + 3e-6, 1e-7, lb, ub, 1e-9, 1e-7) == "violated"


def test_rescaled_formulation_is_exact(syn_cache):
    """``solve_sdp1(scale_a=True)``: the rank-one lift of the exact maximizer is feasible in the transformed model with the same
    objective, and the rescaled safe value is a valid upper bound on p*."""
    import math

    from aimp import bounds
    from aimp import sdp_mosek as SM
    from aimp import sets as S
    from aimp.exact_fq import exact_fq
    from aimp.heads import forward
    from aimp.relax_sdp import LiftIndex

    fq = U.make_instance(0, 9, 1, 4, 2e-2, fixed_query=True, model="gpt-oss-20b")
    si = G.score_intervals_roles(fq, threads=1)
    ctrl = bounds.controls(fq, intervals=si, eps_target=0.0, threads=1)
    ex = exact_fq(fq)
    cap = {}
    orig = SM.ConicModel.solve

    def fake(self, *a, **k):
        cap["mod"] = self
        raise RuntimeError("captured")

    SM.ConicModel.solve = fake
    try:
        try:
            SM.solve_sdp1(fq, ctrl, threads=1, tol=None, cut="auto", scale_a=True)
        except RuntimeError:
            pass
    finally:
        SM.ConicModel.solve = orig
    mod = cap["mod"]
    F, Fb, g, c, Cb = mod.assemble()
    L = LiftIndex.for_instance(fq, ())
    fw = forward(fq, ex["Z_max"])
    v = np.zeros(mod.m)
    v[0] = 1.0
    a_hi = np.minimum(np.asarray(ctrl["a_hi"], float), 1.0)
    for idx in range(fq.n):
        v[L.a(idx)] = fw["a"][idx] / a_hi[idx]
    for t in fq.tokens:
        rep = S.affine(fq.sets[t])
        if rep.r:
            v[L.u(t)] = np.linalg.lstsq(rep.B, ex["Z_max"][t] - rep.c, rcond=None)[0]
    X = np.outer(v, v)
    x = np.zeros(max(mod.nv, 1))
    s_ = fw["s"]
    x[0] = float(np.log(np.exp(s_ - s_.max()).sum()) + s_.max())
    ci = bounds.choose_cut(ctrl, fq.n, "auto")
    if ci["eps"] is not None:
        d = fw["a"] - np.asarray(ci["ahat"], float)
        x[1:1 + fq.n], x[1 + fq.n:1 + 2 * fq.n] = np.maximum(d, 0), np.maximum(-d, 0)
    xv = (2 * np.tril(X) - np.diag(np.diag(X))).reshape(-1)
    val = F @ np.concatenate([x[:mod.nv], np.zeros(F.shape[1] - mod.nv)]) + Fb @ xv + g
    for b in mod.blocks:
        w = val[b["start"]:b["start"] + b["n"]]
        if b["kind"] == "zero":
            assert np.max(np.abs(w)) <= 1e-10, b["name"]
        elif b["kind"] == "nonneg":
            assert np.min(w) >= -1e-12, b["name"]
        elif b["kind"] == "soc":
            assert np.linalg.norm(w[1:]) <= w[0] + 1e-12, b["name"]
        else:
            assert w[1] * math.exp(w[2] / w[1]) <= w[0] * (1 + 1e-12), b["name"]
    obj = float(np.sum((Cb.toarray() + np.tril(Cb.toarray(), -1).T) * X)) + mod.obj_const
    assert abs(obj - fw["phi"]) <= 1e-12 * max(1.0, abs(fw["phi"]))
    r = SM.solve_sdp1(fq, ctrl, threads=1, tol=None, cut="auto", scale_a=True)
    assert r.diagnostics["value_safe"] >= ex["p_fq_hi"] - 1e-9 * max(1.0, abs(ex["p_fq_hi"]))
