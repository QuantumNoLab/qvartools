"""Large-system v4 test: N2-CAS(10,26) 52Q and 4Fe4S 72Q.

The 77Q IBM SQD paper target is closest to our 4Fe4S (CAS(54,36)=72Q).
The 52Q N2 is the SQD-paper-accessible size.

Strategy: modest budgets (100k, 200k) since wall time may exceed 10h at
1M basis. Want to see:
  - 52Q: v4 vs HCI (HCI ref from job bench_hci_ndets_159284.log: -109.3289)
  - 72Q: v4 vs any existing reference (may be first attempt)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.molecules import get_molecule
from src.methods.hi_nqs_sqd_v4 import HINQSSQDv4Config, run_hi_nqs_sqd_v4
from src.nqs.transformer import AutoregressiveTransformer


REFERENCES = {
    "N2-CAS(10,26)": {
        "hci_energy": -109.3288968797, "hci_ndets": 119836809, "hci_time_s": 135311,
    },
    "4Fe4S": {"hci_energy": None, "hci_ndets": None, "hci_time_s": None},
}


def install_batched_sampler(batch_size=50_000):
    orig = AutoregressiveTransformer.sample
    @torch.no_grad()
    def _batched(self, n_samples, hard=True, temperature=1.0):
        if n_samples <= batch_size:
            return orig(self, n_samples, hard=hard, temperature=temperature)
        cfgs, lps = [], []
        for start in range(0, n_samples, batch_size):
            bn = min(batch_size, n_samples - start)
            c, lp = orig(self, bn, hard=hard, temperature=temperature)
            cfgs.append(c); lps.append(lp)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(cfgs, dim=0), torch.cat(lps, dim=0)
    AutoregressiveTransformer.sample = _batched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True,
                        choices=["N2-CAS(10,26)", "4Fe4S"])
    parser.add_argument("--budget", required=True,
                        choices=["50k", "100k", "200k", "500k"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_iterations", type=int, default=30)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    install_batched_sampler()

    BUDGETS = {
        "50k":  (300_000,   5_000,   50_000),
        "100k": (500_000,  10_000,  100_000),
        "200k": (1_000_000, 20_000, 200_000),
        "500k": (1_000_000, 50_000, 500_000),
    }
    n_samples, top_k, max_basis = BUDGETS[args.budget]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"System: {args.system}  Budget: {args.budget}", flush=True)
    print(f"  samples={n_samples:,}  top_k={top_k:,}  max_basis={max_basis:,}", flush=True)
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    H, info = get_molecule(args.system, device=device)
    from math import comb
    hilbert = comb(H.n_orbitals, H.n_alpha) * comb(H.n_orbitals, H.n_beta)
    print(f"n_orb={H.n_orbitals}, n_a={H.n_alpha}, n_b={H.n_beta}, hilbert={hilbert:,}",
          flush=True)

    cfg = HINQSSQDv4Config(
        n_samples=n_samples, top_k=top_k, max_basis_size=max_basis,
        max_iterations=args.max_iterations,
        convergence_threshold=1e-7, convergence_window=3,
        nqs_steps=5, nqs_lr=1e-3, entropy_weight=0.05,
        warm_start=True,
        use_incremental_sqd=False, use_sparse_det_solver=True,
        monotonic_basis=False,
        classical_seed=False, classical_expansion=True,
        classical_expansion_top_n=500, final_pt2_correction=True,
        pt2_top_n=2000,
        use_gpu_sparse_det=True, use_gpu_coupling=True,
    )

    t0 = time.time()
    result = run_hi_nqs_sqd_v4(H, info, config=cfg)
    wall = time.time() - t0

    meta = result.metadata
    e_var = meta.get("e_var"); e_pt2 = meta.get("e_pt2", 0.0)
    e_total = meta.get("e_total", result.energy)

    ref = REFERENCES.get(args.system, {}).get("hci_energy")
    if ref is not None and e_total is not None:
        err_var_mha = (e_var - ref) * 1000 if e_var is not None else None
        err_total_mha = (e_total - ref) * 1000
    else:
        err_var_mha = err_total_mha = None

    print(f"\n{'='*90}")
    print(f"  FINAL {args.system} @ {args.budget}")
    print(f"  E_var   = {e_var}")
    print(f"  E_PT2   = {e_pt2:+.10f}")
    print(f"  E_total = {e_total}")
    if err_total_mha is not None:
        print(f"  err_total vs HCI ref ({ref}) = {err_total_mha:+.3f} mHa")
    print(f"  basis   = {result.diag_dim}")
    print(f"  wall    = {wall:.1f}s = {wall/3600:.2f}h", flush=True)

    record = {
        "system": args.system, "budget": args.budget, "seed": args.seed,
        "hilbert": hilbert, "n_orbitals": H.n_orbitals,
        "n_alpha": H.n_alpha, "n_beta": H.n_beta,
        "n_samples": n_samples, "top_k": top_k, "max_basis_size": max_basis,
        "hci_ref": ref,
        "e_var": float(e_var) if e_var is not None and e_var < 1e10 else None,
        "e_pt2": float(e_pt2),
        "e_total": float(e_total) if e_total is not None else None,
        "err_var_mha": float(err_var_mha) if err_var_mha is not None else None,
        "err_total_mha": float(err_total_mha) if err_total_mha is not None else None,
        "basis": int(result.diag_dim),
        "n_externals": int(meta.get("n_pt2_externals", 0)),
        "var_wall_s": float(meta.get("var_wall_s", 0)),
        "pt2_wall_s": float(meta.get("pt2_wall_s", 0)),
        "total_wall_s": float(wall),
        "converged": bool(result.converged),
        "energy_history": [float(e) for e in meta.get("energy_history", [])],
        "basis_history": [int(b) for b in meta.get("basis_size_history", [])],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"-> saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
