#!/usr/bin/env python3
"""
Iron-sulfur cluster benchmarks — direct comparison with IBM SQD paper.

IBM (Robledo-Moreno et al., Science Advances 2024) benchmarked:
  - [2Fe-2S] CAS(30e,20o) = 40 qubits
  - [4Fe-4S] CAS(54e,36o) = 72 qubits

We use the same FCIDUMP integrals and solve_fermion backend.
The only difference: NQS sampling instead of quantum circuit sampling.

This is the "打臉 SQD" experiment:
  Same molecule, same diagonalization, classical NQS vs quantum hardware.
"""
import sys, time, json
import numpy as np
import torch
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

# [2Fe-2S] — 40 qubits, same as IBM benchmark
SYSTEMS = [
    ("2Fe2S", "40Q", HINQSSQDConfig(
        n_samples=100000,
        top_k=10000,
        max_basis_size=0,
        max_iterations=50,
        convergence_threshold=1e-8,
        convergence_window=3,
        nqs_steps=7,
        nqs_lr=3e-4,
        entropy_weight=0.15,
    )),
]

# Note: [4Fe-4S] = 72 qubits may be too large for single-GPU solve_fermion.
# Include only if memory allows.
try:
    H_test, _ = get_molecule("4Fe4S", device="cpu")
    SYSTEMS.append(("4Fe4S", "72Q", HINQSSQDConfig(
        n_samples=200000,
        top_k=15000,
        max_basis_size=0,
        max_iterations=50,
        convergence_threshold=1e-8,
        convergence_window=3,
        nqs_steps=7,
        nqs_lr=3e-4,
        entropy_weight=0.15,
    )))
    del H_test
    print("[4Fe-4S] 72Q: included", flush=True)
except Exception as ex:
    print(f"[4Fe-4S] 72Q: skipped ({ex})", flush=True)

all_results = []

for mol_name, qlabel, cfg in SYSTEMS:
    print(f"\n{'='*70}", flush=True)
    print(f"  {mol_name} ({qlabel})", flush=True)
    print(f"{'='*70}", flush=True)

    H, info = get_molecule(mol_name, device="cuda")
    n_orb = H.n_orbitals
    nq = info["n_qubits"]
    hilbert = comb(n_orb, H.n_alpha) * comb(n_orb, H.n_beta)
    print(f"  {nq}Q, {n_orb} orb, ({H.n_alpha},{H.n_beta})e, Hilbert={hilbert:,}", flush=True)

    # --- CIPSI ---
    print(f"\n  CIPSI:", flush=True)
    from src.solvers.sci import run_sci
    H_cpu = get_molecule(mol_name, device="cpu")[0]
    info_cpu = {"n_qubits": nq}
    t0 = time.time()
    r_sci = run_sci(H_cpu, info_cpu, expansion_size=1000, max_basis=0, max_iterations=200)
    t_sci = time.time() - t0
    print(f"    E={r_sci.energy:.10f}, basis={r_sci.diag_dim}, time={t_sci:.0f}s", flush=True)

    # --- HI-NQS ---
    print(f"\n  HI-NQS:", flush=True)
    np.random.seed(42)
    torch.manual_seed(42)
    t0 = time.time()
    r_nqs = run_hi_nqs_sqd(H, info, config=cfg)
    t_nqs = time.time() - t0
    print(f"    E={r_nqs.energy:.10f}, basis={r_nqs.diag_dim}, time={t_nqs:.0f}s, "
          f"converged={r_nqs.converged}", flush=True)

    all_results.append({
        "mol": mol_name, "qubits": nq,
        "cipsi_E": r_sci.energy, "cipsi_basis": r_sci.diag_dim, "cipsi_time": t_sci,
        "nqs_E": r_nqs.energy, "nqs_basis": r_nqs.diag_dim, "nqs_time": t_nqs,
        "nqs_converged": r_nqs.converged,
    })

# Summary
print(f"\n{'='*70}", flush=True)
print(f"  IRON-SULFUR CLUSTER SUMMARY (vs IBM SQD)", flush=True)
print(f"{'='*70}", flush=True)
for r in all_results:
    print(f"  {r['mol']} ({r['qubits']}Q):", flush=True)
    print(f"    CIPSI:  E={r['cipsi_E']:.10f}, basis={r['cipsi_basis']}, time={r['cipsi_time']:.0f}s", flush=True)
    print(f"    HI-NQS: E={r['nqs_E']:.10f}, basis={r['nqs_basis']}, time={r['nqs_time']:.0f}s", flush=True)

with open("fe_cluster_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nResults saved to fe_cluster_results.json", flush=True)
