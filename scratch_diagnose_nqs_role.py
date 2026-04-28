"""Ablation diagnostic: is NQS actually doing useful work in HI+NQS+SQD?

Four modes isolate different roles of NQS:
  baseline       : NQS sample + classical_expansion (reference)
  no_nqs         : ~no NQS samples, classical_seed=True + classical_expansion
                   -> if energy ≈ baseline, NQS sampling contributes nothing
  nqs_only       : full NQS samples, classical_expansion=False
                   -> measures what NQS alone can find (without SD-expansion safety net)
  random_sample  : uniform-random N-electron sampling (replaces NQS), expansion ON
                   -> compares NQS quality vs random

We also instrument NQS top-amplitude vs eigenvector top-|c| overlap to see
whether NQS is tracking the diagonalized state at all.

Use small budget so each mode finishes quickly (~30-45 min on 1 GPU):
  top_k=20000, n_samples=500000, max_iterations=5
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch

from src.molecules import get_molecule
from src.methods.hi_nqs_sqd_v4 import HINQSSQDv4Config, run_hi_nqs_sqd_v4
from src.nqs.transformer import AutoregressiveTransformer

REF_HCI = -109.3288968797


def install_random_uniform_sampler(n_alpha: int, n_beta: int, n_orb: int):
    """Replace NQS sample() with uniform-random N-electron sampling.

    Picks n_alpha alpha-orbitals and n_beta beta-orbitals uniformly at random.
    Returns (configs, log_probs) like the original; log_prob is a placeholder
    (downstream code only uses configs)."""
    @torch.no_grad()
    def _random_sample(self, n_samples, hard=True, temperature=1.0):
        device = next(self.parameters()).device
        configs = torch.zeros(n_samples, 2 * n_orb, device=device)
        # Pick n_alpha distinct positions per row, in [0, n_orb)
        for s in range(n_samples):
            ai = torch.randperm(n_orb, device=device)[:n_alpha]
            bi = torch.randperm(n_orb, device=device)[:n_beta]
            configs[s, ai] = 1
            configs[s, n_orb + bi] = 1
        log_probs = torch.zeros(n_samples, device=device)  # placeholder
        return configs, log_probs

    AutoregressiveTransformer.sample = _random_sample
    print(f"[diagnose] Installed random-uniform sampler "
          f"(n_alpha={n_alpha}, n_beta={n_beta}, n_orb={n_orb})", flush=True)


def install_batched_sampler(batch_size=50_000, force_temperature=None):
    """Standard batched wrapper for the original NQS sampler.

    If force_temperature is set, override caller's temperature with this value.
    For β-reshape (arXiv:2603.24728), use force_temperature=1/β.
    Bernoulli identity: P(n)^β / Σ P(n')^β = sigmoid(β·logit) = sigmoid(logit/(1/β)).
    """
    orig = AutoregressiveTransformer.sample
    @torch.no_grad()
    def _batched(self, n_samples, hard=True, temperature=1.0):
        T = force_temperature if force_temperature is not None else temperature
        if n_samples <= batch_size:
            return orig(self, n_samples, hard=hard, temperature=T)
        cfgs, lps = [], []
        for s in range(0, n_samples, batch_size):
            bn = min(batch_size, n_samples - s)
            c, lp = orig(self, bn, hard=hard, temperature=T)
            cfgs.append(c); lps.append(lp)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        return torch.cat(cfgs, dim=0), torch.cat(lps, dim=0)
    AutoregressiveTransformer.sample = _batched
    if force_temperature is not None:
        print(f"[diagnose] Installed β-reshape sampler "
              f"(T={force_temperature}, β={1/force_temperature:.3f})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["baseline", "no_nqs", "nqs_only", "random_sample",
                            "vmc_baseline", "vmc_nqs_only",
                            "nqs_only_beta04", "nqs_only_beta02"])
    p.add_argument("--beta", type=float, default=0.4,
                   help="β for conditional reshape (used by *_betaXX modes)")
    p.add_argument("--molecule", default="N2-CAS(10,26)")
    p.add_argument("--n_samples", type=int, default=500_000)
    p.add_argument("--top_k", type=int, default=20_000)
    p.add_argument("--budget", type=int, default=100_000)
    p.add_argument("--max_iterations", type=int, default=5)
    p.add_argument("--pt2_top_n", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    H, info = get_molecule(args.molecule, device="cuda")
    n_orb = H.n_orbitals
    n_alpha = H.n_alpha; n_beta = H.n_beta

    # Mode-specific configuration
    cfg_kwargs = dict(
        n_samples=args.n_samples, top_k=args.top_k,
        max_basis_size=args.budget, max_iterations=args.max_iterations,
        convergence_threshold=1e-9, convergence_window=999,  # don't early-exit
        nqs_steps=5, nqs_lr=1e-3, entropy_weight=0.05,
        warm_start=True, use_sparse_det_solver=True,
        classical_seed=False, classical_expansion=True,
        classical_expansion_top_n=2000,
        final_pt2_correction=True, pt2_top_n=args.pt2_top_n,
        use_gpu_sparse_det=True, use_gpu_coupling=True,
    )

    if args.mode == "baseline":
        install_batched_sampler()
    elif args.mode == "no_nqs":
        # Disable NQS contribution: tiny n_samples + classical_seed=True.
        # classical_seed=True initialises basis with HF + connected dets,
        # then classical_expansion takes over.
        cfg_kwargs["n_samples"] = 64
        cfg_kwargs["classical_seed"] = True
        install_batched_sampler()
    elif args.mode == "nqs_only":
        cfg_kwargs["classical_expansion"] = False
        install_batched_sampler()
    elif args.mode == "random_sample":
        install_random_uniform_sampler(n_alpha, n_beta, n_orb)
    elif args.mode == "vmc_baseline":
        install_batched_sampler()
        from src.methods.nqs_vmc_update import install_vmc_update_in_v3
        install_vmc_update_in_v3()
    elif args.mode == "vmc_nqs_only":
        cfg_kwargs["classical_expansion"] = False
        install_batched_sampler()
        from src.methods.nqs_vmc_update import install_vmc_update_in_v3
        install_vmc_update_in_v3()
    elif args.mode == "nqs_only_beta04":
        # arXiv:2603.24728: conditional β=0.4 reshape attacks mode collapse.
        # Equivalent to temperature=1/β for Bernoulli autoregressive.
        cfg_kwargs["classical_expansion"] = False
        install_batched_sampler(force_temperature=1.0 / 0.4)  # T=2.5
    elif args.mode == "nqs_only_beta02":
        cfg_kwargs["classical_expansion"] = False
        install_batched_sampler(force_temperature=1.0 / 0.2)  # T=5.0, more aggressive

    cfg = HINQSSQDv4Config(**cfg_kwargs)

    print(f"\n=== diagnostic mode={args.mode} on {args.molecule} ===", flush=True)
    print(f"  n_samples={cfg.n_samples}  top_k={cfg.top_k}  "
          f"budget={cfg.max_basis_size}  iters={cfg.max_iterations}", flush=True)
    print(f"  classical_seed={cfg.classical_seed}  "
          f"classical_expansion={cfg.classical_expansion}", flush=True)

    t0 = time.time()
    result = run_hi_nqs_sqd_v4(H, info, config=cfg)
    wall = time.time() - t0

    meta = result.metadata
    e_var = meta.get("e_var")
    e_pt2 = meta.get("e_pt2", 0.0)
    e_total = meta.get("e_total", result.energy)
    err_var = (e_var - REF_HCI) * 1000 if e_var is not None and e_var < 1e10 else None
    err_total = (e_total - REF_HCI) * 1000 if e_total is not None else None

    energy_history = [float(e) for e in meta.get("energy_history", [])]
    err_history = [(e - REF_HCI) * 1000 for e in energy_history]

    print(f"\n=== {args.mode} result ===", flush=True)
    print(f"  iter energies (mHa above HCI):", flush=True)
    for i, eh in enumerate(err_history):
        print(f"    iter {i}: {eh:+8.3f} mHa", flush=True)
    if err_var is not None:
        print(f"  err_var   = {err_var:+.3f} mHa", flush=True)
    if err_total is not None:
        print(f"  err_total = {err_total:+.3f} mHa", flush=True)
    print(f"  basis = {result.diag_dim}  wall = {wall/60:.1f} min", flush=True)

    record = {
        "mode": args.mode,
        "molecule": args.molecule,
        "n_samples": cfg.n_samples,
        "top_k": cfg.top_k,
        "budget": cfg.max_basis_size,
        "max_iterations": cfg.max_iterations,
        "classical_seed": cfg.classical_seed,
        "classical_expansion": cfg.classical_expansion,
        "seed": args.seed,
        "ref": REF_HCI,
        "e_var": float(e_var) if e_var is not None and e_var < 1e10 else None,
        "e_pt2": float(e_pt2),
        "e_total": float(e_total) if e_total is not None else None,
        "err_var_mha": float(err_var) if err_var is not None else None,
        "err_total_mha": float(err_total) if err_total is not None else None,
        "energy_history": energy_history,
        "err_history_mha": err_history,
        "basis": int(result.diag_dim),
        "wall_s": float(wall),
        "converged": bool(result.converged),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f: json.dump(record, f, indent=2)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
