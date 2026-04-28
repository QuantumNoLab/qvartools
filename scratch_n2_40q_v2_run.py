"""N2-40Q HF v2 validation runs: ablate classical_seed and classical_expansion.

4 variants x 4 budgets = 16 runs:
  Variants: v1 (baseline), v2_seed_only, v2_expand_only, v2_full
  Budgets : (samples, top_k, max_basis) = (100k/10k/100k), (200k/20k/200k),
                                          (500k/50k/500k), (1M/100k/1M)

Key question: does classical_seed fix the +110 mHa iter-0 plateau?
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.molecules import get_molecule
from src.methods.hi_nqs_sqd_v2 import HINQSSQDv2Config, run_hi_nqs_sqd_v2
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
    # classical_seed, classical_expansion
    "v1":          (False, False),  # matches v1 behavior
    "v2_seed":     (True,  False),  # A only
    "v2_expand":   (False, True),   # B only
    "v2_full":     (True,  True),   # A + B
}

BUDGETS = {
    "100k":  (100_000,  10_000,   100_000),
    "200k":  (200_000,  20_000,   200_000),
    "500k":  (500_000,  50_000,   500_000),
    "1m":    (1_000_000, 100_000, 1_000_000),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    parser.add_argument("--budget", required=True, choices=list(BUDGETS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_iterations", type=int, default=25)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    install_batched_sampler()

    seed_on, expand_on = VARIANTS[args.variant]
    n_samples, top_k, max_basis = BUDGETS[args.budget]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Variant: {args.variant}  (classical_seed={seed_on}, expansion={expand_on})",
          flush=True)
    print(f"Budget:  {args.budget}  samples={n_samples:,}  top_k={top_k:,}  "
          f"max_basis={max_basis:,}", flush=True)
    print(f"Device:  {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU:     {torch.cuda.get_device_name(0)}", flush=True)

    H, info = get_molecule("N2-CAS(10,20)", device=device)

    cfg = HINQSSQDv2Config(
        n_samples=n_samples, top_k=top_k, max_basis_size=max_basis,
        max_iterations=args.max_iterations,
        convergence_threshold=1e-6, convergence_window=3,
        nqs_steps=5, nqs_lr=1e-3, entropy_weight=0.05,
        initial_temperature=1.0, final_temperature=0.3,
        warm_start=True,
        use_incremental_sqd=False, use_sparse_det_solver=True,
        monotonic_basis=False,
        classical_seed=seed_on,
        classical_expansion=expand_on,
        classical_expansion_top_n=100,
        classical_expansion_every_n_iters=1,
    )

    t0 = time.time()
    result = run_hi_nqs_sqd_v2(H, info, config=cfg)
    wall = time.time() - t0

    err_mha = (result.energy - REF_HCI_HF) * 1000 if result.energy is not None else None
    hist = result.metadata.get("energy_history", [])
    iter0_err = (hist[0] - REF_HCI_HF) * 1000 if hist else None

    print(f"\n{'='*90}")
    print(f"  variant     = {args.variant}")
    print(f"  budget      = {args.budget}")
    print(f"  iter-0 err  = {iter0_err:+.3f} mHa" if iter0_err is not None else "  iter-0 err = NaN")
    print(f"  final E     = {result.energy}")
    if err_mha is not None:
        print(f"  final err   = {err_mha:+.3f} mHa")
    print(f"  final basis = {result.diag_dim}")
    print(f"  wall        = {wall:.1f}s ({wall/60:.1f} min)", flush=True)

    record = {
        "variant": args.variant,
        "budget": args.budget,
        "seed": args.seed,
        "classical_seed": seed_on,
        "classical_expansion": expand_on,
        "n_samples": n_samples, "top_k": top_k, "max_basis_size": max_basis,
        "ref_hci_hf": REF_HCI_HF,
        "energy": float(result.energy) if result.energy is not None else None,
        "basis": int(result.diag_dim),
        "iter0_energy": float(hist[0]) if hist else None,
        "iter0_err_mha": float(iter0_err) if iter0_err is not None else None,
        "err_vs_hci_mha": float(err_mha) if err_mha is not None else None,
        "wall_time_s": wall,
        "converged": bool(result.converged),
        "energy_history": [float(e) for e in hist],
        "basis_history": [int(b) for b in result.metadata.get("basis_size_history", [])],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"-> saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
