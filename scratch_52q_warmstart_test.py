"""52Q cold-start vs warm-start comparison.

Critical question: was 52Q +73 mHa "plateau" actually a warm_start bug?
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch

from src.molecules import get_molecule
from src.methods.hi_nqs_sqd_v4 import HINQSSQDv4Config, run_hi_nqs_sqd_v4
from src.methods.sparse_det_backend import SparseDetSQDBackend
from src.nqs.transformer import AutoregressiveTransformer


def install_batched_sampler(force_temperature=None):
    orig = AutoregressiveTransformer.sample
    @torch.no_grad()
    def _batched(self, n_samples, hard=True, temperature=1.0):
        T = force_temperature if force_temperature is not None else temperature
        bs = 50_000
        if n_samples <= bs:
            return orig(self, n_samples, hard=hard, temperature=T)
        cfgs, lps = [], []
        for s in range(0, n_samples, bs):
            bn = min(bs, n_samples - s)
            c, lp = orig(self, bn, hard=hard, temperature=T)
            cfgs.append(c); lps.append(lp)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        return torch.cat(cfgs, dim=0), torch.cat(lps, dim=0)
    AutoregressiveTransformer.sample = _batched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm_start", type=int, default=0,
                   help="0 = cold-start (default), 1 = warm-start")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    H, info = get_molecule("N2-CAS(10,26)", device="cuda")
    install_batched_sampler(force_temperature=1.0/0.4)  # β=0.4

    cfg = HINQSSQDv4Config(
        n_samples=500_000, top_k=20_000, max_basis_size=100_000,
        max_iterations=8, convergence_threshold=1e-9, convergence_window=999,
        nqs_steps=5, nqs_lr=1e-3, entropy_weight=0.05,
        warm_start=bool(args.warm_start),  # ← key flag
        use_sparse_det_solver=True,
        classical_seed=False, classical_expansion=False,  # nqs_only mode
        final_pt2_correction=True, pt2_top_n=10000,
        use_gpu_sparse_det=True, use_gpu_coupling=True,
    )

    print(f"\n=== 52Q nqs_only β=0.4, warm_start={cfg.warm_start} ===\n", flush=True)
    t0 = time.time()
    result = run_hi_nqs_sqd_v4(H, info, config=cfg)
    wall = time.time() - t0

    meta = result.metadata
    e_var = meta.get("e_var")
    e_total = meta.get("e_total", result.energy)
    final_basis = meta.get("final_basis")
    REF_HCI = -109.3288968797
    err_var = (e_var - REF_HCI) * 1000 if e_var is not None else None
    err_total = (e_total - REF_HCI) * 1000 if e_total is not None else None

    record = {
        "warm_start": bool(args.warm_start),
        "seed": args.seed,
        "basis": int(result.diag_dim),
        "e_var": float(e_var) if e_var is not None else None,
        "e_total": float(e_total) if e_total is not None else None,
        "err_var_mha": float(err_var) if err_var is not None else None,
        "err_total_mha": float(err_total) if err_total is not None else None,
        "energy_history": [float(e) for e in meta.get("energy_history", [])],
        "wall_s": float(wall),
    }

    print(f"\n--- Result ---", flush=True)
    print(f"  warm_start: {cfg.warm_start}", flush=True)
    print(f"  basis: {result.diag_dim}", flush=True)
    print(f"  err_var:   {err_var:+.3f} mHa", flush=True)
    print(f"  err_total: {err_total:+.3f} mHa", flush=True)
    print(f"  wall: {wall/60:.1f} min", flush=True)

    # Cross-check: cold-start re-solve final basis (regardless of mode used)
    if final_basis is not None:
        print(f"\n--- Cross-check: fresh-backend re-solve same basis ---", flush=True)
        integrals = H.integrals
        be = SparseDetSQDBackend(
            hcore=np.asarray(integrals.h1e), eri=np.asarray(integrals.h2e),
            n_alpha=H.n_alpha, n_beta=H.n_beta,
        )
        t0 = time.time()
        e_cold, _ = be.solve(final_basis)
        e_cold += float(H.nuclear_repulsion)
        cold_wall = time.time() - t0
        err_cold = (e_cold - REF_HCI) * 1000
        print(f"  cold-start E_var: {e_cold:.10f} Ha", flush=True)
        print(f"  err vs HCI ref:   {err_cold:+.3f} mHa", flush=True)
        print(f"  wall: {cold_wall:.1f}s", flush=True)
        record["e_cold_var"] = float(e_cold)
        record["err_cold_var_mha"] = float(err_cold)
        record["cold_wall_s"] = float(cold_wall)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f: json.dump(record, f, indent=2)
    print(f"\n-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
