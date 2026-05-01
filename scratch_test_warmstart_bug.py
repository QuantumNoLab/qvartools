"""Reproduce aug_teacher's stretched-N2 failure, then cold-start re-diagonalize.

If cold-start Davidson on the SAME final basis gives FCI but warm-start gave
+110 mHa, confirms warm_start lock-in is the actual bug.
"""
import time
import numpy as np
import torch
from itertools import combinations

from src.hamiltonians.molecular import create_n2_hamiltonian
from src.methods.sparse_det_backend import SparseDetSQDBackend
from src.methods.hi_nqs_sqd_v4 import HINQSSQDv4Config, run_hi_nqs_sqd_v4
from src.methods.nqs_aug_teacher_update import install_aug_teacher_update_in_v3
from src.nqs.transformer import AutoregressiveTransformer


def install_batched_sampler(force_temperature=None):
    orig = AutoregressiveTransformer.sample
    @torch.no_grad()
    def _batched(self, n_samples, hard=True, temperature=1.0):
        T = force_temperature if force_temperature is not None else temperature
        bs_size = 20_000
        if n_samples <= bs_size:
            return orig(self, n_samples, hard=hard, temperature=T)
        cfgs, lps = [], []
        for s in range(0, n_samples, bs_size):
            bn = min(bs_size, n_samples - s)
            c, lp = orig(self, bn, hard=hard, temperature=T)
            cfgs.append(c); lps.append(lp)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        return torch.cat(cfgs, dim=0), torch.cat(lps, dim=0)
    AutoregressiveTransformer.sample = _batched


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--R", type=float, default=1.8)
    p.add_argument("--FCI", type=float, required=True, help="Ground-truth FCI for this R")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    R = args.R

    print(f"\n=== Reproduce aug_teacher on N2-20Q R={R} STO-3G ===\n", flush=True)

    torch.manual_seed(42); np.random.seed(42)

    H = create_n2_hamiltonian(bond_length=R, basis="sto-3g", device="cuda")
    info = {"name": f"N2_R{R}", "n_qubits": 2*H.n_orbitals,
            "n_orbitals": H.n_orbitals, "n_alpha": H.n_alpha, "n_beta": H.n_beta,
            "basis": "sto-3g", "is_cas": False,
            "description": f"N2 R={R}"}

    install_batched_sampler(force_temperature=1.0/0.4)
    install_aug_teacher_update_in_v3()

    cfg = HINQSSQDv4Config(
        n_samples=20_000, top_k=3000, max_basis_size=14_400,
        max_iterations=30, convergence_threshold=1e-9, convergence_window=999,
        nqs_steps=5, nqs_lr=1e-3, entropy_weight=0.05,
        warm_start=True, use_sparse_det_solver=True,
        classical_seed=False, classical_expansion=False,
        final_pt2_correction=True, pt2_top_n=5000,
        use_gpu_sparse_det=True, use_gpu_coupling=True,
    )

    print("\n--- Run v3 with warm_start=True (reproduce +110 mHa) ---", flush=True)
    t0 = time.time()
    result = run_hi_nqs_sqd_v4(H, info, config=cfg)
    wall = time.time() - t0
    e_warmstart = result.metadata.get("e_var")
    n_basis = result.diag_dim
    FCI = args.FCI
    err_warm = (e_warmstart - FCI) * 1000
    print(f"\n  WARM-START result:", flush=True)
    print(f"    basis size = {n_basis}", flush=True)
    print(f"    e_var      = {e_warmstart:.10f} Ha", flush=True)
    print(f"    err vs FCI = {err_warm:+.3f} mHa", flush=True)
    print(f"    wall: {wall/60:.1f} min", flush=True)

    # Now extract the final basis and cold-start re-solve
    final_bs = result.metadata.get("final_basis")
    if final_bs is None:
        # The metadata doesn't expose basis directly; reconstruct via run.
        # But at least we have e_var; let's also do a cold-start FCI on full Hilbert
        # for control.
        print("\n  WARNING: final_basis not in metadata, can't isolate warm vs cold on same basis", flush=True)
        print("  Falling back: full-Hilbert cold-start as control", flush=True)
        n_orb = H.n_orbitals
        n_alpha = H.n_alpha; n_beta = H.n_beta
        bs_full = np.zeros((14400, 2*n_orb), dtype=np.int8)
        k = 0
        a_combos = list(combinations(range(n_orb), n_alpha))
        b_combos = list(combinations(range(n_orb), n_beta))
        for a in a_combos:
            for b in b_combos:
                for i in a: bs_full[k, n_orb-1-i] = 1
                for i in b: bs_full[k, 2*n_orb-1-i] = 1
                k += 1
        integrals = H.integrals
        be = SparseDetSQDBackend(hcore=np.asarray(integrals.h1e),
                                  eri=np.asarray(integrals.h2e),
                                  n_alpha=n_alpha, n_beta=n_beta)
        e_cold_full, _ = be.solve(bs_full)
        e_cold_full += float(H.nuclear_repulsion)
        print(f"\n  COLD-START full Hilbert: e = {e_cold_full:.10f} Ha", flush=True)
        print(f"    err vs FCI = {(e_cold_full - FCI)*1000:+.3f} mHa", flush=True)
    else:
        # Cold-start: build fresh backend, no warm-start
        print("\n--- Cold-start re-diagonalize the SAME basis ---", flush=True)
        integrals = H.integrals
        cold_be = SparseDetSQDBackend(
            hcore=np.asarray(integrals.h1e),
            eri=np.asarray(integrals.h2e),
            n_alpha=H.n_alpha, n_beta=H.n_beta,
        )
        t0 = time.time()
        e_cold, _ = cold_be.solve(final_bs)
        e_cold += float(H.nuclear_repulsion)
        cold_wall = time.time() - t0
        err_cold = (e_cold - FCI) * 1000
        print(f"  COLD-START on basis size {len(final_bs)}:", flush=True)
        print(f"    e_var      = {e_cold:.10f} Ha", flush=True)
        print(f"    err vs FCI = {err_cold:+.3f} mHa", flush=True)
        print(f"    wall: {cold_wall:.1f}s", flush=True)
        print()
        print(f"  WARM vs COLD on same basis:", flush=True)
        print(f"    warm: {err_warm:+.3f} mHa", flush=True)
        print(f"    cold: {err_cold:+.3f} mHa", flush=True)
        print(f"    diff: {err_warm - err_cold:+.3f} mHa", flush=True)
        if abs(err_warm - err_cold) > 10:
            print("\n  CONCLUSION: warm_start CAUSES the error!", flush=True)

        if args.out:
            import json
            from pathlib import Path
            record = {
                "R": R, "basis_size": int(n_basis),
                "e_warm": float(e_warmstart),
                "e_cold": float(e_cold),
                "err_warm_mha": float(err_warm),
                "err_cold_mha": float(err_cold),
                "FCI": float(FCI),
            }
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w") as f: json.dump(record, f, indent=2)


if __name__ == "__main__":
    main()
