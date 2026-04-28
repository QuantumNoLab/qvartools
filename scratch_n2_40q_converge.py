"""N2-CAS(10,20) 40Q — let sparse_det converge naturally.

Config:
  n_samples    = 500,000 per iter
  top_k        = 40,000 per iter (4× previous 10k)
  max_basis    = 200,000 (4× previous 50k — was hitting the cap with churn)
  max_iters    = 50 (but convergence ΔE<1e-6 × 6 will stop it earlier)
"""
import time
import numpy as np
import torch

from src.molecules import get_molecule
from src.methods.hi_nqs_sqd import HINQSSQDConfig, run_hi_nqs_sqd
from src.nqs.transformer import AutoregressiveTransformer


# --- Monkey-patch nqs.sample to process large n_samples in GPU batches so the
# transformer's FFN intermediate doesn't blow 80 GB at 500k samples on 40Q. ---
_orig_sample = AutoregressiveTransformer.sample


@torch.no_grad()
def _batched_sample(self, n_samples, hard=True, temperature=1.0, batch_size=50_000):
    if n_samples <= batch_size:
        return _orig_sample(self, n_samples, hard=hard, temperature=temperature)
    cfgs = []
    lps = []
    for start in range(0, n_samples, batch_size):
        bn = min(batch_size, n_samples - start)
        c, lp = _orig_sample(self, bn, hard=hard, temperature=temperature)
        cfgs.append(c)
        lps.append(lp)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(cfgs, dim=0), torch.cat(lps, dim=0)


AutoregressiveTransformer.sample = _batched_sample


REF_HCI_HF_ORBITALS = -109.21473332712169   # HCI 17.4M dets, 285s
REF_RHF             = -108.9830065322408


def main():
    torch.manual_seed(2024)
    np.random.seed(2024)

    H, info = get_molecule("N2-CAS(10,20)")
    cfg = HINQSSQDConfig(
        max_iterations=50,
        convergence_threshold=1e-6,
        convergence_window=6,
        n_samples=500_000,
        top_k=40_000,
        max_basis_size=200_000,
        nqs_steps=5,
        use_sparse_det_solver=True,
        use_incremental_sqd=False,
    )

    print(f"Reference: HCI (HF orbs, 17.4M dets, 285s) = {REF_HCI_HF_ORBITALS:.6f}")
    print(f"           RHF = {REF_RHF:.6f}")
    print(f"\nConfig: n_samples=500k, top_k=40k, max_basis=200k, max_iters=50 "
          f"(ΔE<1e-6 × 6 convergence)")
    print(f"{'='*90}\n")

    t0 = time.time()
    result = run_hi_nqs_sqd(H, info, config=cfg)
    wall = time.time() - t0

    print(f"\n{'='*90}")
    print(f"  FINAL")
    print(f"{'='*90}")
    print(f"  final E          = {result.energy:.10f}")
    print(f"  err vs HCI ref   = {(result.energy - REF_HCI_HF_ORBITALS)*1000:+.3f} mHa")
    print(f"  err vs RHF       = {(result.energy - REF_RHF)*1000:+.3f} mHa "
          f"(negative = correlation recovered)")
    print(f"  final basis      = {result.diag_dim}")
    print(f"  converged        = {result.converged}")
    print(f"  wall             = {wall:.1f} s")

    hist = result.metadata.get("energy_history", [])
    bhist = result.metadata.get("basis_size_history", [])
    if hist:
        print(f"\n  iter-by-iter:")
        for i, (e, b) in enumerate(zip(hist, bhist)):
            print(f"    iter {i:>2}: E={e:.8f}  basis={b:>6}  "
                  f"err vs HCI = {(e - REF_HCI_HF_ORBITALS)*1000:+.3f} mHa")


if __name__ == "__main__":
    main()
