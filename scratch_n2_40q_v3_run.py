"""N2-40Q HF v3 validation.

4 variants x 4 budgets = 16 runs:
  default : classical_expansion_top_n=1000, pt2_top_n=5000,  PT2=on
  nopt2   : classical_expansion_top_n=1000,                  PT2=off   (ablates PT2)
  small   : classical_expansion_top_n=200,   pt2_top_n=5000, PT2=on    (ablates top_n)
  big     : classical_expansion_top_n=2000,  pt2_top_n=10000, PT2=on

Budgets: (n_samples, top_k, max_basis) = (100k,10k,100k) (200k,20k,200k)
                                         (500k,50k,500k) (1M,100k,1M)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.molecules import get_molecule
from src.methods.hi_nqs_sqd_v3 import HINQSSQDv3Config, run_hi_nqs_sqd_v3
from src.nqs.transformer import AutoregressiveTransformer

REF_HCI_HF = -109.21473332712169


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


VARIANTS = {
    "default": {"classical_expansion_top_n": 1000, "pt2_top_n": 5000,  "final_pt2_correction": True},
    "nopt2":   {"classical_expansion_top_n": 1000, "pt2_top_n": 5000,  "final_pt2_correction": False},
    "small":   {"classical_expansion_top_n": 200,  "pt2_top_n": 5000,  "final_pt2_correction": True},
    "big":     {"classical_expansion_top_n": 2000, "pt2_top_n": 10000, "final_pt2_correction": True},
}

BUDGETS = {
    "100k":  (100_000,   10_000,   100_000),
    "200k":  (200_000,   20_000,   200_000),
    "500k":  (500_000,   50_000,   500_000),
    "1m":    (1_000_000, 100_000, 1_000_000),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    parser.add_argument("--budget", required=True, choices=list(BUDGETS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_iterations", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    install_batched_sampler()

    var = VARIANTS[args.variant]
    n_samples, top_k, max_basis = BUDGETS[args.budget]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Variant: {args.variant}  {var}", flush=True)
    print(f"Budget:  {args.budget}  samples={n_samples:,} top_k={top_k:,} "
          f"max_basis={max_basis:,}", flush=True)
    print(f"Device:  {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU:     {torch.cuda.get_device_name(0)}", flush=True)

    H, info = get_molecule("N2-CAS(10,20)", device=device)

    cfg = HINQSSQDv3Config(
        n_samples=n_samples, top_k=top_k, max_basis_size=max_basis,
        max_iterations=args.max_iterations,
        convergence_threshold=1e-9, convergence_window=5,
        nqs_steps=5, nqs_lr=1e-3, entropy_weight=0.05,
        initial_temperature=1.0, final_temperature=0.3,
        warm_start=True,
        use_incremental_sqd=False, use_sparse_det_solver=True,
        monotonic_basis=False,
        classical_seed=False,
        classical_expansion=True,
        **var,
    )

    t0 = time.time()
    result = run_hi_nqs_sqd_v3(H, info, config=cfg)
    wall = time.time() - t0

    meta = result.metadata
    e_var = meta.get("e_var")
    e_pt2 = meta.get("e_pt2", 0.0)
    e_total = meta.get("e_total", result.energy)
    err_var_mha = (e_var - REF_HCI_HF) * 1000 if e_var is not None and e_var < 1e10 else None
    err_total_mha = (e_total - REF_HCI_HF) * 1000 if e_total is not None else None

    hist = meta.get("energy_history", [])
    iter0_err = (hist[0] - REF_HCI_HF) * 1000 if hist else None

    print(f"\n{'='*90}")
    print(f"  FINAL ({args.variant} @ {args.budget})")
    print(f"{'='*90}")
    print(f"  iter 0 err    = {iter0_err:+.3f} mHa" if iter0_err is not None else "")
    print(f"  E_var         = {e_var}")
    print(f"  err_var       = {err_var_mha:+.3f} mHa" if err_var_mha is not None else "")
    print(f"  E_PT2         = {e_pt2:+.10f}")
    print(f"  E_total       = {e_total}")
    print(f"  err_total     = {err_total_mha:+.3f} mHa" if err_total_mha is not None else "")
    print(f"  final basis   = {result.diag_dim}")
    print(f"  n_externals   = {meta.get('n_pt2_externals', 0):,}")
    print(f"  var_wall      = {meta.get('var_wall_s', 0):.1f}s")
    print(f"  pt2_wall      = {meta.get('pt2_wall_s', 0):.1f}s")
    print(f"  total_wall    = {wall:.1f}s ({wall/60:.1f} min)", flush=True)

    record = {
        "variant": args.variant,
        "budget": args.budget,
        "seed": args.seed,
        "variant_params": var,
        "n_samples": n_samples, "top_k": top_k, "max_basis_size": max_basis,
        "ref_hci_hf": REF_HCI_HF,
        "e_var": float(e_var) if e_var is not None and e_var < 1e10 else None,
        "e_pt2": float(e_pt2),
        "e_total": float(e_total) if e_total is not None else None,
        "err_var_mha": float(err_var_mha) if err_var_mha is not None else None,
        "err_total_mha": float(err_total_mha) if err_total_mha is not None else None,
        "iter0_err_mha": float(iter0_err) if iter0_err is not None else None,
        "basis": int(result.diag_dim),
        "n_externals": int(meta.get("n_pt2_externals", 0)),
        "var_wall_s": float(meta.get("var_wall_s", 0)),
        "pt2_wall_s": float(meta.get("pt2_wall_s", 0)),
        "total_wall_s": float(wall),
        "converged": bool(result.converged),
        "energy_history": [float(e) for e in hist],
        "basis_history": [int(b) for b in meta.get("basis_size_history", [])],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"-> saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
