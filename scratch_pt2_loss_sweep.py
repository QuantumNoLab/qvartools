"""PT2-augmented loss: hyperparameter sweep + 24Q validation.

Phase 1 — Sweep on N2 20Q (fast, max_basis=500, Hilbert=14.4k):
  Scan (λ_ext, top_K_ext) to find sweet spot and characterise how the gain
  depends on teacher-mixing strength and external-det count.

Phase 2 — Validation on N2 CAS(10,12) 24Q (max_basis=2000, Hilbert=627k):
  Confirm the improvement from best-of-sweep hyperparameters transfers to a
  larger, qualitatively different system (more orbitals, different FCI).
"""
from __future__ import annotations

import time
import numpy as np
import torch

from src.molecules import get_molecule
from src.methods.hi_nqs_sqd import HINQSSQDConfig, run_hi_nqs_sqd
import src.methods.hi_nqs_sqd as hi_mod

from scratch_pt2_methods import compute_fci_reference
from scratch_pt2_loss_exp import make_pt2_augmented_update


def run_one(mol_name, max_basis, top_k_per_iter, max_iter,
            variant: str, seed: int,
            h1e, eri, ecore, lambda_ext, top_K_ext,
            n_samples: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    H, info = get_molecule(mol_name)

    cfg = HINQSSQDConfig(
        max_iterations=max_iter,
        convergence_threshold=1e-6,
        convergence_window=3,
        n_samples=n_samples,
        top_k=top_k_per_iter,
        max_basis_size=max_basis,
        nqs_steps=10,
        nqs_lr=1e-3,
        teacher_weight=1.0,
        energy_weight=0.1,
        entropy_weight=0.05,
        initial_temperature=1.0,
        final_temperature=0.3,
        warm_start=True,
        use_incremental_sqd=False,
        use_sparse_det_solver=True,
        monotonic_basis=False,
    )

    original_update = hi_mod._update_nqs
    try:
        if variant == "pt2_aug":
            hi_mod._update_nqs = make_pt2_augmented_update(
                h1e=h1e, eri=eri, ecore=ecore,
                top_K_ext=top_K_ext, lambda_ext=lambda_ext, verbose=False,
            )
        t0 = time.time()
        result = run_hi_nqs_sqd(H, info, config=cfg)
        wall = time.time() - t0
    finally:
        hi_mod._update_nqs = original_update

    return result, wall


def phase1_sweep():
    MOL = "N2"  # 20Q, Hilbert=14400
    H, _ = get_molecule(MOL)
    h1e = np.asarray(H.integrals.h1e, dtype=np.float64)
    eri = np.asarray(H.integrals.h2e, dtype=np.float64)
    ecore = float(H.integrals.nuclear_repulsion)
    E_FCI = compute_fci_reference(h1e, eri, H.n_orbitals, H.n_alpha, H.n_beta, ecore)

    print(f"\n{'='*90}")
    print(f"  PHASE 1 — SWEEP on {MOL} 20Q  (FCI = {E_FCI:.6f} Ha, Hilbert=14,400)")
    print(f"  Config: max_basis=500, top_k=200, n_samples=2000, max_iter=20")
    print(f"{'='*90}")

    shared = dict(
        mol_name=MOL, max_basis=500, top_k_per_iter=200, max_iter=20,
        n_samples=2000, seed=2024, h1e=h1e, eri=eri, ecore=ecore,
    )

    # Baseline
    print(f"\n  [baseline: standard teacher]")
    result_bl, t_bl = run_one(variant="standard", lambda_ext=0.0, top_K_ext=0, **shared)
    print(f"    final E = {result_bl.energy:.8f}  err vs FCI = "
          f"{(result_bl.energy-E_FCI)*1000:+.3f} mHa  (t={t_bl:.1f}s)")

    # Sweep lambda × top_K
    lambdas = [0.1, 0.3, 0.5]
    top_Ks = [200, 1000]
    rows = []
    for lam in lambdas:
        for K in top_Ks:
            print(f"\n  [pt2_aug: λ={lam}, top_K={K}]")
            r, t = run_one(
                variant="pt2_aug", lambda_ext=lam, top_K_ext=K, **shared,
            )
            err = (r.energy - E_FCI) * 1000
            gain = (result_bl.energy - r.energy) * 1000
            print(f"    final E = {r.energy:.8f}  err vs FCI = {err:+.3f} mHa  "
                  f"gain = {gain:+.3f} mHa  (t={t:.1f}s)")
            rows.append(dict(lam=lam, K=K, E=r.energy, err=err, gain=gain, t=t))

    print(f"\n  {'='*90}")
    print(f"  Phase 1 SUMMARY  (baseline err = {(result_bl.energy-E_FCI)*1000:+.3f} mHa)")
    print(f"  {'='*90}")
    print(f"  {'λ':>5} {'K':>6} {'err (mHa)':>12} {'gain (mHa)':>12} {'t (s)':>8}")
    for r in rows:
        print(f"  {r['lam']:>5.1f} {r['K']:>6d} {r['err']:>+10.3f}   "
              f"{r['gain']:>+10.3f}   {r['t']:>6.1f}")

    best = max(rows, key=lambda r: r["gain"])
    print(f"\n  BEST (λ, K) = ({best['lam']}, {best['K']}) → gain {best['gain']:+.3f} mHa")
    return best, result_bl.energy, E_FCI


def phase2_validate(best_lam, best_K):
    MOL = "N2-CAS(10,12)"  # 24Q, Hilbert=627,264
    H, _ = get_molecule(MOL)
    h1e = np.asarray(H.integrals.h1e, dtype=np.float64)
    eri = np.asarray(H.integrals.h2e, dtype=np.float64)
    ecore = float(H.integrals.nuclear_repulsion)

    # FCI for 24Q — 627k may take a minute
    print(f"\n{'='*90}")
    print(f"  PHASE 2 — VALIDATE on {MOL}  (Hilbert=627,264, best λ={best_lam}, K={best_K})")
    print(f"{'='*90}")
    t0 = time.time()
    E_FCI = compute_fci_reference(h1e, eri, H.n_orbitals, H.n_alpha, H.n_beta, ecore)
    print(f"  FCI = {E_FCI:.8f} Ha  (t={time.time()-t0:.1f}s)")

    shared = dict(
        mol_name=MOL, max_basis=2000, top_k_per_iter=500, max_iter=15,
        n_samples=5000, seed=2024, h1e=h1e, eri=eri, ecore=ecore,
    )

    print(f"\n  [baseline: standard teacher]")
    r_bl, t_bl = run_one(variant="standard", lambda_ext=0.0, top_K_ext=0, **shared)
    err_bl = (r_bl.energy - E_FCI) * 1000
    print(f"    final E = {r_bl.energy:.8f}  err vs FCI = {err_bl:+.3f} mHa  "
          f"(t={t_bl:.1f}s)")

    print(f"\n  [pt2_aug: λ={best_lam}, top_K={best_K}]")
    r_aug, t_aug = run_one(
        variant="pt2_aug", lambda_ext=best_lam, top_K_ext=best_K, **shared,
    )
    err_aug = (r_aug.energy - E_FCI) * 1000
    gain = err_bl - err_aug
    print(f"    final E = {r_aug.energy:.8f}  err vs FCI = {err_aug:+.3f} mHa  "
          f"gain = {gain:+.3f} mHa  (t={t_aug:.1f}s)")

    print(f"\n  24Q validation: "
          f"{'CONFIRMED' if gain > 0 else 'FAILED'}  (gain {gain:+.3f} mHa)")

    # Iter-by-iter
    hA = r_bl.metadata.get("energy_history", [])
    hB = r_aug.metadata.get("energy_history", [])
    max_iter = max(len(hA), len(hB))
    print(f"\n  iter-by-iter (err vs FCI, mHa):")
    print(f"    {'iter':>4}  {'E_A err':>10}  {'E_B err':>10}  {'B-A':>10}")
    for i in range(max_iter):
        eA = f"{(hA[i]-E_FCI)*1000:+.3f}" if i < len(hA) else "  —  "
        eB = f"{(hB[i]-E_FCI)*1000:+.3f}" if i < len(hB) else "  —  "
        diff = f"{(hB[i]-hA[i])*1000:+.3f}" if (i < len(hA) and i < len(hB)) else "  —  "
        print(f"    {i:>4}  {eA:>10}  {eB:>10}  {diff:>10}")


def main():
    best, _, _ = phase1_sweep()
    phase2_validate(best["lam"], best["K"])


if __name__ == "__main__":
    main()
