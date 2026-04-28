"""N2-CAS(10,26) 52Q config scan: vary top_n and pt2_top_n at fixed budget=500k.

Goal: find the top_n × pt2_top_n setting that breaks past the +72 mHa plateau
seen in 26619 (where top_n=500 exhausted after iter 10, no new dets found).
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

REF_HCI = -109.3288968797


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
    parser.add_argument("--top_n", type=int, required=True)
    parser.add_argument("--pt2_top_n", type=int, required=True)
    parser.add_argument("--budget", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iterations", type=int, default=40)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    install_batched_sampler()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"N2-CAS(10,26) config scan: top_n={args.top_n}, pt2_top_n={args.pt2_top_n}, "
          f"budget={args.budget:,}, seed={args.seed}", flush=True)

    H, info = get_molecule("N2-CAS(10,26)", device=device)

    # n_samples scales with budget so NQS has enough exploration
    n_samples = max(1_000_000, args.budget * 2)
    top_k = max(10_000, args.budget // 10)

    cfg = HINQSSQDv4Config(
        n_samples=n_samples, top_k=top_k, max_basis_size=args.budget,
        max_iterations=args.max_iterations,
        convergence_threshold=1e-8, convergence_window=5,
        nqs_steps=5, nqs_lr=1e-3, entropy_weight=0.05,
        warm_start=True,
        use_sparse_det_solver=True, use_incremental_sqd=False,
        classical_seed=False, classical_expansion=True,
        classical_expansion_top_n=args.top_n,
        final_pt2_correction=True,
        pt2_top_n=args.pt2_top_n,
        use_gpu_sparse_det=True, use_gpu_coupling=True,
    )

    t0 = time.time()
    result = run_hi_nqs_sqd_v4(H, info, config=cfg)
    wall = time.time() - t0

    meta = result.metadata
    e_var = meta.get("e_var"); e_pt2 = meta.get("e_pt2", 0.0)
    e_total = meta.get("e_total", result.energy)
    err_var = (e_var - REF_HCI) * 1000 if e_var is not None and e_var < 1e10 else None
    err_total = (e_total - REF_HCI) * 1000 if e_total is not None else None

    print(f"\n===== FINAL =====", flush=True)
    print(f"  E_var   = {e_var}", flush=True)
    print(f"  E_PT2   = {e_pt2:+.6f}", flush=True)
    print(f"  E_total = {e_total}", flush=True)
    if err_total is not None:
        print(f"  err_total vs HCI = {err_total:+.2f} mHa", flush=True)
    print(f"  basis final = {result.diag_dim}", flush=True)
    print(f"  wall        = {wall/60:.1f} min", flush=True)

    record = {
        "system": "N2-CAS(10,26)", "seed": args.seed,
        "top_n": args.top_n, "pt2_top_n": args.pt2_top_n,
        "budget": args.budget, "n_samples": n_samples, "top_k": top_k,
        "ref_hci": REF_HCI,
        "e_var": float(e_var) if e_var is not None and e_var < 1e10 else None,
        "e_pt2": float(e_pt2),
        "e_total": float(e_total) if e_total is not None else None,
        "err_var_mha": float(err_var) if err_var is not None else None,
        "err_total_mha": float(err_total) if err_total is not None else None,
        "basis": int(result.diag_dim),
        "n_externals": int(meta.get("n_pt2_externals", 0)),
        "wall_s": float(wall),
        "converged": bool(result.converged),
        "energy_history": [float(e) for e in meta.get("energy_history", [])],
        "basis_history": [int(b) for b in meta.get("basis_size_history", [])],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
