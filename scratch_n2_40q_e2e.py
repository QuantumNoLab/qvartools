"""N2-CAS(10,20) 40Q: end-to-end comparison of sparse_det vs incremental.

Known references (from results/orbital_fair_results.json, HF orbitals):
  RHF                                      = -108.9830
  HCI  (17.4M dets, 285s)                  = -109.2147
  HI+NQS+SQD  (basis=43k, 4651s = 77 min)  = -109.2146
"""
import time
import gc

import numpy as np
import torch

from src.molecules import get_molecule
from src.methods.hi_nqs_sqd import HINQSSQDConfig, run_hi_nqs_sqd


REF_HCI_HF_ORBITALS = -109.21473332712169     # 17.4M dets, 285 s
REF_RHF             = -108.9830065322408


def run(backend_name, use_sparse, use_inc, max_iters=4,
        n_samples=5000, top_k=1000, max_basis=5000, seed=2024):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    H, info = get_molecule("N2-CAS(10,20)")
    cfg = HINQSSQDConfig(
        max_iterations=max_iters,
        n_samples=n_samples,
        top_k=top_k,
        max_basis_size=max_basis,
        nqs_steps=5,
        use_sparse_det_solver=use_sparse,
        use_incremental_sqd=use_inc,
    )
    print(f"\n{'='*90}\n  [{backend_name}]  max_iters={max_iters} samples={n_samples} "
          f"top_k={top_k} max_basis={max_basis}\n{'='*90}")
    t0 = time.time()
    result = run_hi_nqs_sqd(H, info, config=cfg)
    wall = time.time() - t0
    return result, wall


if __name__ == "__main__":
    print(f"\nReferences:")
    print(f"  RHF                              = {REF_RHF:.6f}")
    print(f"  HCI (HF orbs, 17.4M dets, 285s)  = {REF_HCI_HF_ORBITALS:.6f}")
    print(f"\n  (This benchmark: sparse_det vs incremental, same NQS, same seed)\n")

    # Start with very conservative: 4 iters, max_basis 5000
    rs, tss = run("sparse_det", use_sparse=True, use_inc=False,
                   max_iters=4, n_samples=5000, top_k=1000, max_basis=5000)

    ri, tii = run("incremental", use_sparse=False, use_inc=True,
                   max_iters=4, n_samples=5000, top_k=1000, max_basis=5000)

    print(f"\n{'='*90}")
    print(f"  SUMMARY")
    print(f"{'='*90}")
    print(f"  {'backend':<14} {'final E':>14} {'err vs HCI':>14} {'basis':>8} {'wall':>8}")
    print(f"  {'-'*14} {'-'*14} {'-'*14} {'-'*8} {'-'*8}")
    print(f"  {'sparse_det':<14} {rs.energy:>14.6f} "
          f"{(rs.energy - REF_HCI_HF_ORBITALS)*1000:>+11.3f}mHa "
          f"{rs.diag_dim:>8d} {tss:>7.1f}s")
    print(f"  {'incremental':<14} {ri.energy:>14.6f} "
          f"{(ri.energy - REF_HCI_HF_ORBITALS)*1000:>+11.3f}mHa "
          f"{ri.diag_dim:>8d} {tii:>7.1f}s")
