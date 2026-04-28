"""Compare IncrementalSQDBackend vs SparseDetSQDBackend (HCI-style) end-to-end.

Runs the full HI+NQS+SQD loop on several molecules with each backend, using
the SAME seed and config, and reports:
  final energy, error vs FCI, final basis size, wall time, speedup.
"""
import time
import numpy as np
import torch

from src.molecules import get_molecule
from src.methods.hi_nqs_sqd import HINQSSQDConfig, run_hi_nqs_sqd
from pyscf.fci import direct_spin1


def compute_fci(H):
    hcore = np.asarray(H.integrals.h1e, dtype=np.float64)
    eri = np.asarray(H.integrals.h2e, dtype=np.float64)
    nuc = float(H.integrals.nuclear_repulsion)
    cisolver = direct_spin1.FCI()
    e, _ = cisolver.kernel(
        hcore, eri, H.n_orbitals, (H.n_alpha, H.n_beta), ecore=nuc, verbose=0,
    )
    return float(e)


def run_backend(mol_name, backend, n_samples=2000, top_k=500, max_basis=3000,
                max_iters=10, seed=2024):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    H, info = get_molecule(mol_name)
    cfg = HINQSSQDConfig(
        max_iterations=max_iters,
        n_samples=n_samples,
        top_k=top_k,
        max_basis_size=max_basis,
        nqs_steps=5,
        use_sparse_det_solver=(backend == "sparse_det"),
        use_incremental_sqd=(backend == "incremental"),
    )
    t0 = time.time()
    result = run_hi_nqs_sqd(H, info, config=cfg)
    wall = time.time() - t0
    return result, wall, H


def main():
    molecules = [
        ("H2O", 2000, 500, 3000, 6),
        ("BeH2", 2000, 500, 3000, 6),
        ("NH3", 2000, 600, 4000, 6),
        ("N2", 3000, 800, 15000, 8),
    ]
    rows = []
    for mol_name, ns, tk, mb, mi in molecules:
        print(f"\n{'='*80}\n  {mol_name}\n{'='*80}")

        # FCI reference (computed once)
        H, _ = get_molecule(mol_name)
        try:
            e_fci = compute_fci(H)
        except Exception as ex:
            print(f"  FCI failed: {ex}")
            e_fci = None

        # Run sparse_det (HCI-style)
        print(f"  [sparse_det]")
        r_sd, t_sd, _ = run_backend(mol_name, "sparse_det",
                                     n_samples=ns, top_k=tk, max_basis=mb, max_iters=mi)

        # Run incremental
        print(f"  [incremental]")
        r_in, t_in, _ = run_backend(mol_name, "incremental",
                                     n_samples=ns, top_k=tk, max_basis=mb, max_iters=mi)

        rows.append({
            "mol": mol_name,
            "e_fci": e_fci,
            "e_sd": r_sd.energy,
            "e_in": r_in.energy,
            "basis_sd": r_sd.diag_dim,
            "basis_in": r_in.diag_dim,
            "t_sd": t_sd,
            "t_in": t_in,
        })

    # ----- Summary table -----
    print("\n" + "=" * 98)
    print("  SUMMARY: sparse_det (HCI-style) vs incremental")
    print("=" * 98)
    header = (
        f"  {'Mol':<6} {'FCI (Ha)':>14} "
        f"{'HCI E (Ha)':>14} {'inc E (Ha)':>14} "
        f"{'HCI err':>10} {'inc err':>10} "
        f"{'HCI bs':>7} {'inc bs':>7} "
        f"{'HCI t':>8} {'inc t':>8} {'×':>6}"
    )
    print(header)
    print("  " + "-" * 96)
    for r in rows:
        fci = r["e_fci"]
        err_sd = (r["e_sd"] - fci) * 1000 if fci is not None else float("nan")
        err_in = (r["e_in"] - fci) * 1000 if fci is not None else float("nan")
        speedup = r["t_in"] / max(r["t_sd"], 1e-6)
        print(
            f"  {r['mol']:<6} {fci:>14.6f} "
            f"{r['e_sd']:>14.6f} {r['e_in']:>14.6f} "
            f"{err_sd:>+8.3f}mHa {err_in:>+8.3f}mHa "
            f"{r['basis_sd']:>7d} {r['basis_in']:>7d} "
            f"{r['t_sd']:>6.1f}s {r['t_in']:>6.1f}s {speedup:>5.1f}×"
        )


if __name__ == "__main__":
    main()
