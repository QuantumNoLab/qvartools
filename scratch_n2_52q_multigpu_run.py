"""52Q big-budget run with multi-GPU on-the-fly Davidson.

Uses all 8 GPUs visible to the task: hamiltonian replicated, basis row-partitioned,
matvec parallelized. Targets budget 1M-2M (single GPU stalls due to per-iter
matvec time).
"""
import argparse, json, os, time
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
        for start in range(0, n_samples, batch_size):
            bn = min(batch_size, n_samples - start)
            c, lp = orig(self, bn, hard=hard, temperature=temperature)
            cfgs.append(c); lps.append(lp)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(cfgs, dim=0), torch.cat(lps, dim=0)
    AutoregressiveTransformer.sample = _batched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--top_n", type=int, default=2000)
    p.add_argument("--pt2_top_n", type=int, default=20000)
    p.add_argument("--max_iterations", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    install_batched_sampler()

    n_visible = torch.cuda.device_count()
    print(f"[52Q multi-GPU] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<all>')}",
          flush=True)
    print(f"  visible GPUs: {n_visible}", flush=True)
    print(f"  budget={args.budget:,}  top_n={args.top_n}  pt2_top_n={args.pt2_top_n}",
          flush=True)

    H, info = get_molecule("N2-CAS(10,26)", device="cuda")

    cfg = HINQSSQDv4Config(
        n_samples=max(2_000_000, args.budget * 2),
        top_k=max(20_000, args.budget // 10),
        max_basis_size=args.budget,
        max_iterations=args.max_iterations,
        convergence_threshold=1e-7, convergence_window=3,
        nqs_steps=5, nqs_lr=1e-3, entropy_weight=0.05,
        warm_start=True,
        use_sparse_det_solver=True,
        classical_seed=False, classical_expansion=True,
        classical_expansion_top_n=args.top_n,
        final_pt2_correction=True, pt2_top_n=args.pt2_top_n,
        use_gpu_sparse_det=True, use_gpu_coupling=True,
        use_multi_gpu_on_the_fly_davidson=True,  # ENABLE multi-GPU
        on_the_fly_chunk_size=4000,
    )

    t0 = time.time()
    result = run_hi_nqs_sqd_v4(H, info, config=cfg)
    wall = time.time() - t0

    meta = result.metadata
    e_var = meta.get("e_var"); e_pt2 = meta.get("e_pt2", 0.0)
    e_total = meta.get("e_total", result.energy)
    err_var = (e_var - REF_HCI) * 1000 if e_var is not None and e_var < 1e10 else None
    err_total = (e_total - REF_HCI) * 1000 if e_total is not None else None

    print(f"\n=== FINAL ===")
    print(f"  E_var   = {e_var}")
    print(f"  E_PT2   = {e_pt2:+.6f}")
    print(f"  E_total = {e_total}")
    if err_total is not None:
        print(f"  err_total vs HCI = {err_total:+.2f} mHa")
    print(f"  basis = {result.diag_dim}, wall = {wall/3600:.2f}h")

    record = {
        "system": "N2-CAS(10,26)", "seed": args.seed,
        "budget": args.budget, "top_n": args.top_n, "pt2_top_n": args.pt2_top_n,
        "ref_hci": REF_HCI,
        "n_gpus": n_visible,
        "use_multi_gpu_on_the_fly": True,
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
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
