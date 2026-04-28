#!/usr/bin/env python3
"""
N2 large-system benchmark: SCI vs HI+NQS+SQD v3
Systems: N2 20Q (reference), 40Q CAS(10,20), 52Q CAS(10,26)

SCI uses sparse eigsh (scipy ARPACK) for basis > 10000 — no OOM.
NQS v3 uses solve_fermion with top_k cap to keep basis manageable.
"""
import sys, time
import numpy as np
import torch
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.solvers.sci import CIPSISolver
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
print(f"Sparse SCI: no basis cap, eigsh for basis>10K", flush=True)
print(f"NQS v3: solve_fermion diagonalization", flush=True)
print(flush=True)

# (mol_name, nqs_top_k, nqs_samples, sci_expansion)
MOLECULES = [
    ("N2",              500,  5000,  500),   # 20Q  — FCI reference available
    ("N2-CAS(10,20)",  2000, 10000, 1000),   # 40Q  — no FCI
    ("N2-CAS(10,26)",  4000, 20000, 2000),   # 52Q  — no FCI
]

results = []

for mol_name, top_k, n_samples, sci_expansion in MOLECULES:
    print("=" * 70, flush=True)
    print(f"  {mol_name}", flush=True)
    print("=" * 70, flush=True)

    t_load = time.time()
    H, info = get_molecule(mol_name)
    n_orb = H.n_orbitals
    nq = info["n_qubits"]
    hilbert = comb(n_orb, H.n_alpha) * comb(n_orb, H.n_beta)
    print(f"{nq}Q, {n_orb} orb, ({H.n_alpha},{H.n_beta})e, Hilbert={hilbert:,}  "
          f"(load={time.time()-t_load:.1f}s)", flush=True)

    # ── FCI reference (only feasible for small systems) ──────────────────
    fci_E = None
    if hilbert <= 50_000:
        from src.solvers.fci import FCISolver
        fci = FCISolver().solve(H, info)
        fci_E = fci.energy
        print(f"FCI: {fci_E:.10f} Ha (basis={fci.diag_dim})", flush=True)

    # ── SCI — no basis cap, sparse eigsh kicks in at basis > 10K ─────────
    print(f"\nSCI (no basis cap, expansion={sci_expansion})...", flush=True)
    t0 = time.time()
    sci = CIPSISolver(
        max_iterations=200,
        max_basis_size=0,          # 0 = unlimited
        expansion_size=sci_expansion,
        convergence_threshold=1e-8,
    ).solve(H, info)
    sci_time = time.time() - t0
    ref_E = fci_E if fci_E else sci.energy
    sci_err = (sci.energy - ref_E) * 1000 if ref_E else 0.0
    print(f"SCI: {sci.energy:.10f} Ha, basis={sci.diag_dim}, "
          f"t={sci_time:.1f}s, err={sci_err:.6f} mHa", flush=True)

    # ── NQS v3 ───────────────────────────────────────────────────────────
    print(f"\nNQS v3 (top_k={top_k}, samples={n_samples}, no basis cap)...", flush=True)
    np.random.seed(42)
    torch.manual_seed(42)
    t0 = time.time()
    cfg = HINQSSQDConfig(
        n_samples=n_samples,
        top_k=top_k,
        max_basis_size=0,          # 0 = unlimited (PT2 controls quality)
        max_iterations=50,
        convergence_threshold=1e-8,
        convergence_window=3,
        nqs_steps=10,
    )
    r = run_hi_nqs_sqd(H, info, config=cfg)
    nqs_time = time.time() - t0
    nqs_err = (r.energy - ref_E) * 1000 if (r.energy and ref_E) else None

    # ── Per-molecule table ────────────────────────────────────────────────
    print(f"\n  {'Method':<20} {'Energy':>18} {'err(mHa)':>10} {'basis':>7} {'time':>8}")
    print(f"  {'-'*65}")
    if fci_E:
        print(f"  {'FCI':<20} {fci_E:>18.10f} {'0.000000':>10} {fci.diag_dim:>7} {fci.wall_time:>7.1f}s")
    err_str = f"{sci_err:>10.6f}"
    print(f"  {'SCI (sparse)':<20} {sci.energy:>18.10f} {err_str} {sci.diag_dim:>7} {sci_time:>7.1f}s")
    if r.energy:
        ne_str = f"{nqs_err:>10.6f}" if nqs_err is not None else f"{'N/A':>10}"
        print(f"  {'NQS v3':<20} {r.energy:>18.10f} {ne_str} {r.diag_dim:>7} {nqs_time:>7.1f}s")
    print(flush=True)

    results.append(dict(
        mol=mol_name, nq=nq,
        fci_E=fci_E,
        sci_E=sci.energy, sci_err=sci_err, sci_basis=sci.diag_dim, sci_t=sci_time,
        nqs_E=r.energy, nqs_err=nqs_err, nqs_basis=r.diag_dim, nqs_t=nqs_time,
    ))

# ── Final summary table ───────────────────────────────────────────────────
print("=" * 70, flush=True)
print("  FINAL COMPARISON (SCI sparse vs NQS v3)", flush=True)
print("=" * 70, flush=True)
hdr = f"  {'Mol':>16} {'Q':>3} | {'SCI E':>18} {'err':>10} {'basis':>6} {'time':>7} | {'NQS E':>18} {'err':>10} {'basis':>6} {'time':>7}"
print(hdr, flush=True)
print(f"  {'-'*100}", flush=True)
for res in results:
    se = f"{res['sci_err']:.6f}" if res['sci_err'] is not None else "N/A"
    ne = f"{res['nqs_err']:.6f}" if res['nqs_err'] is not None else "N/A"
    se_str = f"{res['sci_E']:.10f}" if res['sci_E'] else "FAIL"
    ne_str = f"{res['nqs_E']:.10f}" if res['nqs_E'] else "FAIL"
    print(f"  {res['mol']:>16} {res['nq']:>3} | {se_str:>18} {se:>10} {res['sci_basis']:>6} {res['sci_t']:>6.0f}s "
          f"| {ne_str:>18} {ne:>10} {res['nqs_basis']:>6} {res['nqs_t']:>6.0f}s", flush=True)
