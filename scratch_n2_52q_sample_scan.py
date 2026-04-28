"""52Q n_samples scan: test if more NQS samples (better |ψ|² coverage) breaks
the +71 mHa plateau.

Hilbert space = 4.3B dets; 1M samples covers only 0.023%. Hypothesis: more
samples per iter → broader basis seed candidates → deeper expansion → lower E.

Scan: 4 n_samples × 4 seeds = 16 tasks. Fixed budget=500k, top_n=2000,
top_k=50k, pt2_top_n=20k, max_iter=20.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch

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
        for s in range(0, n_samples, batch_size):
            bn = min(batch_size, n_samples - s)
            c, lp = orig(self, bn, hard=hard, temperature=temperature)
            cfgs.append(c); lps.append(lp)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        return torch.cat(cfgs, dim=0), torch.cat(lps, dim=0)
    AutoregressiveTransformer.sample = _batched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, required=True)
    p.add_argument("--top_k", type=int, default=50_000)
    p.add_argument("--budget", type=int, default=500_000)
    p.add_argument("--top_n", type=int, default=2000)
    p.add_argument("--pt2_top_n", type=int, default=20000)
    p.add_argument("--max_iterations", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    install_batched_sampler()

    print(f"[52Q sample scan] n_samples={args.n_samples:,}  top_k={args.top_k:,}  "
          f"budget={args.budget:,}  seed={args.seed}", flush=True)
    print(f"  Hilbert={4_327_008_400:,}  coverage = "
          f"{100 * args.n_samples / 4_327_008_400:.4f}%", flush=True)

    H, info = get_molecule("N2-CAS(10,26)", device="cuda")

    cfg = HINQSSQDv4Config(
        n_samples=args.n_samples, top_k=args.top_k,
        max_basis_size=args.budget,
        max_iterations=args.max_iterations,
        convergence_threshold=1e-7, convergence_window=3,
        nqs_steps=5, nqs_lr=1e-3, entropy_weight=0.05,
        warm_start=True, use_sparse_det_solver=True,
        classical_seed=False, classical_expansion=True,
        classical_expansion_top_n=args.top_n,
        final_pt2_correction=True, pt2_top_n=args.pt2_top_n,
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

    print(f"\n=== n_samples={args.n_samples} seed={args.seed} ===")
    if err_var is not None: print(f"  err_var   = {err_var:+.3f} mHa")
    if err_total is not None: print(f"  err_total = {err_total:+.3f} mHa")
    print(f"  basis = {result.diag_dim}  wall = {wall/60:.0f} min")

    record = {
        "system": "N2-CAS(10,26)",
        "n_samples": args.n_samples, "top_k": args.top_k,
        "budget": args.budget, "top_n": args.top_n, "pt2_top_n": args.pt2_top_n,
        "seed": args.seed, "ref": REF_HCI,
        "e_var": float(e_var) if e_var is not None and e_var < 1e10 else None,
        "e_pt2": float(e_pt2),
        "e_total": float(e_total) if e_total is not None else None,
        "err_var_mha": float(err_var) if err_var is not None else None,
        "err_total_mha": float(err_total) if err_total is not None else None,
        "basis": int(result.diag_dim),
        "wall_s": float(wall), "converged": bool(result.converged),
        "energy_history": [float(e) for e in meta.get("energy_history", [])],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f: json.dump(record, f, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
