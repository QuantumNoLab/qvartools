"""Smoke test for VMC + joint-supervised NQS update.

Runs BeH2 with both the original and VMC update, compares iter trajectory
+ final energy. Goal: VMC version reaches FCI (or matches baseline) without
breaking existing pipeline.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch

from src.molecules import get_molecule
from src.methods.hi_nqs_sqd_v4 import HINQSSQDv4Config, run_hi_nqs_sqd_v4
from src.methods.nqs_vmc_update import install_vmc_update_in_v3
from src.nqs.transformer import AutoregressiveTransformer


def install_batched_sampler(batch_size=20_000):
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
    p.add_argument("--mode", choices=["baseline", "vmc"], required=True)
    p.add_argument("--molecule", default="BeH2")
    p.add_argument("--max_iterations", type=int, default=10)
    p.add_argument("--top_k", type=int, default=2000)
    p.add_argument("--n_samples", type=int, default=50_000)
    p.add_argument("--budget", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    install_batched_sampler()

    if args.mode == "vmc":
        install_vmc_update_in_v3()

    H, info = get_molecule(args.molecule, device="cuda")

    cfg = HINQSSQDv4Config(
        n_samples=args.n_samples, top_k=args.top_k,
        max_basis_size=args.budget,
        max_iterations=args.max_iterations,
        convergence_threshold=1e-9, convergence_window=999,
        nqs_steps=5, nqs_lr=1e-3, entropy_weight=0.05,
        warm_start=True, use_sparse_det_solver=True,
        classical_seed=False, classical_expansion=True,
        classical_expansion_top_n=200,
        final_pt2_correction=True, pt2_top_n=500,
        use_gpu_sparse_det=True, use_gpu_coupling=True,
    )

    print(f"[smoke] mode={args.mode} molecule={args.molecule}", flush=True)

    t0 = time.time()
    result = run_hi_nqs_sqd_v4(H, info, config=cfg)
    wall = time.time() - t0

    meta = result.metadata
    e_var = meta.get("e_var")
    e_total = meta.get("e_total", result.energy)
    energy_history = [float(e) for e in meta.get("energy_history", [])]

    print(f"[smoke] e_var={e_var}  e_total={e_total}  wall={wall:.1f}s", flush=True)
    print(f"[smoke] energy history: {energy_history}", flush=True)

    record = {
        "mode": args.mode,
        "molecule": args.molecule,
        "e_var": float(e_var) if e_var is not None else None,
        "e_total": float(e_total) if e_total is not None else None,
        "energy_history": energy_history,
        "basis": int(result.diag_dim),
        "wall_s": float(wall),
        "converged": bool(result.converged),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f: json.dump(record, f, indent=2)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
