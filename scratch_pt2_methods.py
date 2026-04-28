"""Three PT2 methods on top of sparse_det SCI — validated against FCI.

Addresses the core trade-off in the HI+NQS+SQD pipeline:
  - Full SCI-style expansion (singles/doubles from every V det) → basis explodes
  - Pure NQS-sampled basis → misses rare-but-important external dets

Methods implemented
-------------------
M1 Epstein-Nesbet PT2 correction:
    Keep V fixed; compute E_PT2 = Σ_{a∉V} t_a² / (E_var - H_aa)
    where t_a = ⟨a|H|ψ_V⟩ = Σ_{i∈V} c_i H_ai.
    Single pass over V's connected space, external dets do NOT grow V.

M2 Two-threshold SHCI (Sharma-Holmes-Umrigar 2017):
    |t_a| > ε₁  → add to V, re-diagonalize
    ε₂ < |t_a| ≤ ε₁ → E_PT2 bucket (not in V)
    |t_a| ≤ ε₂ → discarded
    Returns E_var(V') + E_PT2'.

M3 Semistochastic PT2 with |H_ai|-weighted proposal:
    Sample N external dets via the proposal q(a) ∝ |t_a| (heat-bath style);
    unbiased IS estimator of E_PT2. Demonstrates how to avoid enumerating
    the full external space when it's too large (e.g. N2 40Q at 150k V).

Validated against pyscf.direct_spin1 FCI on H2O/NH3/N2 small systems — all
three methods should monotonically lower energy toward FCI.
"""
from __future__ import annotations

import math
import time
from typing import Tuple, List

import numpy as np
from pyscf.fci import direct_spin1

from src.molecules import get_molecule
from src.methods.sparse_det_solver import (
    _sign_single,
    _sign_double_same_spin,
    _bit_positions,
    solve_sparse_det_ci,
)
from pyscf import ao2mo


# --------------------------------------------------------------------------
# Core primitive: enumerate external space and its couplings to V
# --------------------------------------------------------------------------
def build_pt2_candidates(
    V_alpha: np.ndarray,
    V_beta: np.ndarray,
    c_V: np.ndarray,
    h1e: np.ndarray,
    eri: np.ndarray,
    norb: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (ext_alpha, ext_beta, t, H_diag) for every external det connected to V.

    t_a = ⟨a|H|ψ_V⟩ = Σ_{i∈V} c_i H_ai

    Only singles/doubles are enumerated — Slater-Condon guarantees all other
    H_ai = 0 for |a⟩ that differ from all |i⟩ by ≥3 orbitals AT ONCE, which
    can actually happen here: a det a might be a double from i1 but quadruple
    from i2 (contribution 0). So the total external space is the UNION of
    singles/doubles of each i ∈ V.
    """
    g = ao2mo.restore(1, np.asarray(eri, dtype=np.float64), norb)

    V_set = set(zip(V_alpha.tolist(), V_beta.tolist()))
    N = len(V_alpha)

    # Accumulator: (alpha, beta) → t_value
    t_accum: dict = {}

    for i in range(N):
        a_i = int(V_alpha[i])
        b_i = int(V_beta[i])
        c_i = float(c_V[i])
        if abs(c_i) < 1e-12:
            continue

        occ_a = _bit_positions(a_i, norb)
        occ_b = _bit_positions(b_i, norb)
        vir_a = [o for o in range(norb) if o not in set(occ_a)]
        vir_b = [o for o in range(norb) if o not in set(occ_b)]

        # --- alpha single (p ∈ occ_a, q ∈ vir_a) ---
        for p in occ_a:
            for q in vir_a:
                a_new = a_i ^ (1 << p) ^ (1 << q)
                key = (a_new, b_i)
                if key in V_set:
                    continue
                sign = _sign_single(a_i, p, q)
                elem = h1e[q, p]
                for k in occ_a:
                    if k != p:
                        elem += g[q, p, k, k] - g[q, k, k, p]
                for k in occ_b:
                    elem += g[q, p, k, k]
                h_ai = sign * elem
                if abs(h_ai) > 1e-12:
                    t_accum[key] = t_accum.get(key, 0.0) + c_i * h_ai

        # --- beta single ---
        for p in occ_b:
            for q in vir_b:
                b_new = b_i ^ (1 << p) ^ (1 << q)
                key = (a_i, b_new)
                if key in V_set:
                    continue
                sign = _sign_single(b_i, p, q)
                elem = h1e[q, p]
                for k in occ_b:
                    if k != p:
                        elem += g[q, p, k, k] - g[q, k, k, p]
                for k in occ_a:
                    elem += g[q, p, k, k]
                h_ai = sign * elem
                if abs(h_ai) > 1e-12:
                    t_accum[key] = t_accum.get(key, 0.0) + c_i * h_ai

        # --- aa double ---
        n_oa = len(occ_a)
        n_va = len(vir_a)
        for pi in range(n_oa):
            p1 = occ_a[pi]
            for p2i in range(pi + 1, n_oa):
                p2 = occ_a[p2i]
                for qi in range(n_va):
                    q1 = vir_a[qi]
                    for q2i in range(qi + 1, n_va):
                        q2 = vir_a[q2i]
                        a_new = a_i ^ (1 << p1) ^ (1 << p2) ^ (1 << q1) ^ (1 << q2)
                        key = (a_new, b_i)
                        if key in V_set:
                            continue
                        sign = _sign_double_same_spin(a_i, p1, p2, q1, q2)
                        val = sign * (g[q1, p1, q2, p2] - g[q1, p2, q2, p1])
                        if abs(val) > 1e-12:
                            t_accum[key] = t_accum.get(key, 0.0) + c_i * val

        # --- bb double ---
        n_ob = len(occ_b)
        n_vb = len(vir_b)
        for pi in range(n_ob):
            p1 = occ_b[pi]
            for p2i in range(pi + 1, n_ob):
                p2 = occ_b[p2i]
                for qi in range(n_vb):
                    q1 = vir_b[qi]
                    for q2i in range(qi + 1, n_vb):
                        q2 = vir_b[q2i]
                        b_new = b_i ^ (1 << p1) ^ (1 << p2) ^ (1 << q1) ^ (1 << q2)
                        key = (a_i, b_new)
                        if key in V_set:
                            continue
                        sign = _sign_double_same_spin(b_i, p1, p2, q1, q2)
                        val = sign * (g[q1, p1, q2, p2] - g[q1, p2, q2, p1])
                        if abs(val) > 1e-12:
                            t_accum[key] = t_accum.get(key, 0.0) + c_i * val

        # --- ab mixed double ---
        for pa in occ_a:
            for qa in vir_a:
                a_new = a_i ^ (1 << pa) ^ (1 << qa)
                sign_a = _sign_single(a_i, pa, qa)
                for pb in occ_b:
                    for qb in vir_b:
                        b_new = b_i ^ (1 << pb) ^ (1 << qb)
                        key = (a_new, b_new)
                        if key in V_set:
                            continue
                        sign_b = _sign_single(b_i, pb, qb)
                        val = sign_a * sign_b * g[qa, pa, qb, pb]
                        if abs(val) > 1e-12:
                            t_accum[key] = t_accum.get(key, 0.0) + c_i * val

    # Pack into arrays + compute diagonals
    keys = list(t_accum.keys())
    M = len(keys)
    if M == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                np.zeros(0), np.zeros(0))

    ext_alpha = np.array([k[0] for k in keys], dtype=np.int64)
    ext_beta = np.array([k[1] for k in keys], dtype=np.int64)
    t = np.array([t_accum[k] for k in keys], dtype=np.float64)

    # Diagonals: H_aa using the same formula as sparse_det_solver
    h_diag = np.diag(h1e)
    J = np.einsum("iijj->ij", g)
    K = np.einsum("ijji->ij", g)
    H_diag = np.zeros(M, dtype=np.float64)
    for m in range(M):
        oa = np.array(_bit_positions(int(ext_alpha[m]), norb), dtype=np.int64)
        ob = np.array(_bit_positions(int(ext_beta[m]), norb), dtype=np.int64)
        e1 = h_diag[oa].sum() + h_diag[ob].sum()
        e2 = (J[np.ix_(oa, oa)].sum() + J[np.ix_(oa, ob)].sum()
              + J[np.ix_(ob, oa)].sum() + J[np.ix_(ob, ob)].sum()
              - K[np.ix_(oa, oa)].sum() - K[np.ix_(ob, ob)].sum())
        H_diag[m] = e1 + 0.5 * e2

    return ext_alpha, ext_beta, t, H_diag


# --------------------------------------------------------------------------
# M1: Post-hoc Epstein-Nesbet PT2 correction (no change to V)
# --------------------------------------------------------------------------
def method1_posthoc_pt2(
    V_alpha, V_beta, c_V, E_var, h1e, eri, norb, ecore,
    eps2: float = 1e-10,
):
    ext_alpha, ext_beta, t, H_diag = build_pt2_candidates(
        V_alpha, V_beta, c_V, h1e, eri, norb
    )
    mask = np.abs(t) > eps2
    if mask.sum() == 0:
        return 0.0, 0
    denom = (E_var - ecore) - H_diag[mask]   # E_var was returned with ecore added;
                                             # H_diag was computed without ecore
    safe_denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    e_pt2 = (t[mask] ** 2 / safe_denom).sum()
    return float(e_pt2), int(mask.sum())


# --------------------------------------------------------------------------
# M2: Two-threshold SHCI — split external dets into (V-upgrade, PT2-bucket)
# --------------------------------------------------------------------------
def method2_two_threshold(
    V_alpha, V_beta, c_V, E_var, h1e, eri, norb, ecore,
    eps1: float = 1e-3, eps2: float = 1e-10,
):
    ext_alpha, ext_beta, t, H_diag = build_pt2_candidates(
        V_alpha, V_beta, c_V, h1e, eri, norb
    )

    abs_t = np.abs(t)
    idx_var = abs_t > eps1
    idx_pt2 = (abs_t > eps2) & ~idx_var

    n_var = int(idx_var.sum())
    n_pt2 = int(idx_pt2.sum())

    if n_var == 0:
        # Nothing to upgrade — behaves like M1
        denom = (E_var - ecore) - H_diag[idx_pt2]
        safe_denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        e_pt2 = (t[idx_pt2] ** 2 / safe_denom).sum()
        return float(E_var + e_pt2), n_var, n_pt2, float(E_var)

    # Upgrade: V' = V ∪ ext[idx_var], re-diagonalize
    new_alpha = np.concatenate([V_alpha, ext_alpha[idx_var]])
    new_beta = np.concatenate([V_beta, ext_beta[idx_var]])
    # Seed c_V' from c_V
    ci0 = np.concatenate([c_V, np.zeros(n_var, dtype=np.float64)])
    result = solve_sparse_det_ci(
        new_alpha, new_beta, h1e, eri, norb, ecore=ecore, ci0=ci0,
    )
    E_var_new = result.energy
    c_new = result.amplitudes

    # PT2 on the remaining bucket — need to recompute t w.r.t. NEW ψ_V'
    ext_alpha2, ext_beta2, t2, H_diag2 = build_pt2_candidates(
        new_alpha, new_beta, c_new, h1e, eri, norb
    )
    mask2 = np.abs(t2) > eps2
    if mask2.sum() == 0:
        return float(E_var_new), n_var, 0, float(E_var_new)
    denom2 = (E_var_new - ecore) - H_diag2[mask2]
    safe2 = np.where(np.abs(denom2) < 1e-12, 1e-12, denom2)
    e_pt2_new = (t2[mask2] ** 2 / safe2).sum()

    return float(E_var_new + e_pt2_new), n_var, int(mask2.sum()), float(E_var_new)


# --------------------------------------------------------------------------
# M3: Semistochastic PT2 via |t|-weighted importance sampling
# --------------------------------------------------------------------------
def method3_stochastic_pt2(
    V_alpha, V_beta, c_V, E_var, h1e, eri, norb, ecore,
    N_samples: int = 5000, proposal: str = "abs_t", seed: int = 0,
):
    """Unbiased MC estimate of E_PT2 using q(a) ∝ w_a, where
    w_a = |t_a| by default. Samples N_samples external dets with replacement.

    E_PT2 = Σ_a t_a² / (E - H_aa)
          = Σ_a (t_a² / (E - H_aa)) · q_a / q_a
          ≈ (1/N) Σ_{a ~ q} (t_a² / (E - H_aa)) / q_a
    with q_a = w_a / Σ_b w_b.

    This demonstrates that even when the external space is too large to
    enumerate, a heat-bath-style importance proposal gives a low-variance
    estimate. If proposal='uniform', q_a = 1/M which is the naive baseline
    (high variance).
    """
    rng = np.random.default_rng(seed)
    ext_alpha, ext_beta, t, H_diag = build_pt2_candidates(
        V_alpha, V_beta, c_V, h1e, eri, norb
    )
    M = len(t)
    if M == 0:
        return 0.0, 0.0

    denom = (E_var - ecore) - H_diag
    safe_denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    f = t ** 2 / safe_denom        # true contribution per det (what we want to sum)

    # Proposal
    if proposal == "uniform":
        q = np.full(M, 1.0 / M)
    elif proposal == "abs_t":
        w = np.abs(t) + 1e-30
        q = w / w.sum()
    elif proposal == "t_sq":
        w = t ** 2 + 1e-30
        q = w / w.sum()
    else:
        raise ValueError(f"Unknown proposal: {proposal}")

    idx = rng.choice(M, size=N_samples, replace=True, p=q)
    # IS estimator: mean of f[idx] / (M · q[idx]) ·  M   -- actually
    # E_PT2 = Σ_a f_a = Σ_a (f_a / q_a) q_a ≈ (1/N) Σ_{a ~ q} f_a / q_a
    estimator = (f[idx] / q[idx]).mean()
    # Variance diagnostic
    stderr = (f[idx] / q[idx]).std(ddof=1) / math.sqrt(N_samples)

    return float(estimator), float(stderr)


# --------------------------------------------------------------------------
# Helpers: HF seed + FCI reference
# --------------------------------------------------------------------------
def build_hf_seed(n_alpha, n_beta, norb):
    a0 = (1 << n_alpha) - 1
    b0 = (1 << n_beta) - 1
    return np.array([a0], dtype=np.int64), np.array([b0], dtype=np.int64)


def compute_fci_reference(h1e, eri, norb, n_alpha, n_beta, ecore):
    cisolver = direct_spin1.FCI()
    e, _ = cisolver.kernel(
        h1e, eri, norb, (n_alpha, n_beta), ecore=ecore, verbose=0,
    )
    return float(e)


def grow_basis_via_singles_doubles(V_alpha, V_beta, c_V, h1e, eri, norb,
                                    top_k: int = 200, eps: float = 1e-6):
    """Deterministic HCI-like growth: pick top-k external dets by |t_a|."""
    ext_alpha, ext_beta, t, _ = build_pt2_candidates(
        V_alpha, V_beta, c_V, h1e, eri, norb
    )
    if len(t) == 0:
        return V_alpha, V_beta
    abs_t = np.abs(t)
    keep = abs_t > eps
    if keep.sum() == 0:
        return V_alpha, V_beta
    idx = np.argsort(abs_t[keep])[::-1][:top_k]
    sel = np.where(keep)[0][idx]
    new_alpha = np.concatenate([V_alpha, ext_alpha[sel]])
    new_beta = np.concatenate([V_beta, ext_beta[sel]])
    return new_alpha, new_beta


# --------------------------------------------------------------------------
# Main: validate on small molecules
# --------------------------------------------------------------------------
def run_one_molecule(name: str, V_target_size: int = 50):
    print(f"\n{'='*84}")
    print(f"  {name}")
    print(f"{'='*84}")

    H, info = get_molecule(name)
    norb = H.n_orbitals
    n_alpha = H.n_alpha
    n_beta = H.n_beta
    h1e = np.asarray(H.integrals.h1e, dtype=np.float64)
    eri = np.asarray(H.integrals.h2e, dtype=np.float64)
    ecore = float(H.integrals.nuclear_repulsion)

    print(f"  norb={norb}, n_e=({n_alpha},{n_beta}), Hilbert(total)="
          f"{math.comb(norb, n_alpha) * math.comb(norb, n_beta):,}")

    # 1. FCI reference
    t0 = time.time()
    E_FCI = compute_fci_reference(h1e, eri, norb, n_alpha, n_beta, ecore)
    print(f"  FCI reference: E = {E_FCI:.10f} Ha  (t={time.time()-t0:.2f}s)")

    # 2. Build small V via HF + a few HCI growth steps to mimic NQS's state
    V_alpha, V_beta = build_hf_seed(n_alpha, n_beta, norb)
    result = solve_sparse_det_ci(V_alpha, V_beta, h1e, eri, norb, ecore=ecore)
    c_V = result.amplitudes
    E_var = result.energy

    growth_iters = 0
    while len(V_alpha) < V_target_size and growth_iters < 20:
        V_alpha, V_beta = grow_basis_via_singles_doubles(
            V_alpha, V_beta, c_V, h1e, eri, norb, top_k=20, eps=1e-6,
        )
        result = solve_sparse_det_ci(V_alpha, V_beta, h1e, eri, norb, ecore=ecore)
        c_V = result.amplitudes
        E_var = result.energy
        growth_iters += 1

    print(f"  Built variational V of size {len(V_alpha)} via HCI-like growth")
    print(f"  E_var             = {E_var:.10f} Ha  err vs FCI = "
          f"{(E_var-E_FCI)*1000:+.3f} mHa")

    # 3. M1: Epstein-Nesbet PT2 correction
    t0 = time.time()
    e_pt2_m1, n_m1 = method1_posthoc_pt2(
        V_alpha, V_beta, c_V, E_var, h1e, eri, norb, ecore, eps2=1e-12,
    )
    E_M1 = E_var + e_pt2_m1
    print(f"  [M1] E_var + E_PT2 = {E_M1:.10f} Ha  err vs FCI = "
          f"{(E_M1-E_FCI)*1000:+.3f} mHa  (n_ext={n_m1}, E_PT2={e_pt2_m1*1000:+.3f} mHa, "
          f"t={time.time()-t0:.2f}s)")

    # 4. M2: Two-threshold SHCI
    t0 = time.time()
    E_M2, n_var_added, n_pt2_M2, E_var_new = method2_two_threshold(
        V_alpha, V_beta, c_V, E_var, h1e, eri, norb, ecore,
        eps1=1e-3, eps2=1e-12,
    )
    print(f"  [M2] E_var'+E_PT2' = {E_M2:.10f} Ha  err vs FCI = "
          f"{(E_M2-E_FCI)*1000:+.3f} mHa  (+{n_var_added} into V, "
          f"E_var' err={((E_var_new-E_FCI)*1000):+.3f} mHa, t={time.time()-t0:.2f}s)")

    # 5. M3: Semistochastic PT2 (|t|-weighted IS, compare to uniform)
    for proposal, N in [("uniform", 2000), ("abs_t", 2000), ("abs_t", 8000)]:
        t0 = time.time()
        e_pt2_m3, se = method3_stochastic_pt2(
            V_alpha, V_beta, c_V, E_var, h1e, eri, norb, ecore,
            N_samples=N, proposal=proposal, seed=42,
        )
        E_M3 = E_var + e_pt2_m3
        err = (E_M3 - E_FCI) * 1000
        exact_err = abs(e_pt2_m3 - e_pt2_m1) * 1000
        print(f"  [M3:{proposal:>7}, N={N}] E = {E_M3:.10f}  err vs FCI = "
              f"{err:+.3f} mHa  |stderr|={se*1000:.3f} mHa  "
              f"|bias vs M1|={exact_err:.3f} mHa  (t={time.time()-t0:.2f}s)")

    return {
        "mol": name,
        "E_FCI": E_FCI,
        "E_var": E_var,
        "E_M1": E_M1,
        "E_M2": E_M2,
        "V_size": len(V_alpha),
    }


if __name__ == "__main__":
    results = []
    for mol_name in ["H2O", "NH3", "N2"]:
        try:
            r = run_one_molecule(mol_name, V_target_size=50)
            results.append(r)
        except Exception as ex:
            print(f"\n  [ERROR] {mol_name}: {ex}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*84}")
    print(f"  SUMMARY  (errors in mHa; negative = below FCI = impossible bug)")
    print(f"{'='*84}")
    print(f"  {'Mol':<6} {'V':>6} {'E_var err':>12} {'M1 err':>12} {'M2 err':>12}")
    for r in results:
        print(f"  {r['mol']:<6} {r['V_size']:>6} "
              f"{(r['E_var']-r['E_FCI'])*1000:>+10.3f}"
              f"{(r['E_M1']-r['E_FCI'])*1000:>+12.3f}"
              f"{(r['E_M2']-r['E_FCI'])*1000:>+12.3f}")
