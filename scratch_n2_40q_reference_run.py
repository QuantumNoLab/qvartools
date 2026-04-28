"""N2-CAS(10,20) 40Q HF reference runs.

Presets:
  old_orig_incremental : reproduces the bench_orbital_fair_163794 "43k" run
                         (n_samples=100k, top_k=10k, max_basis=0, incremental backend).
                         Monkey-patches IncrementalSQDBackend.solve to log n_α × n_β per iter.

  new_* : NEW NQS + sparse_det configs for scan (see NEW_CONFIGS dict).

Called with --config <name> --seed N --out PATH.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.molecules import get_molecule
from src.methods.hi_nqs_sqd import HINQSSQDConfig, run_hi_nqs_sqd
from src.methods.incremental_sqd import IncrementalSQDBackend
from src.nqs.transformer import AutoregressiveTransformer
from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs


REF_HCI_HF = -109.21473332712169  # HCI 17.4M dets, 285s


# Shared NEW NQS baseline config
_NEW_BASE = dict(
    max_iterations=30,
    convergence_threshold=1e-6, convergence_window=3,
    nqs_steps=5, nqs_lr=1e-3, entropy_weight=0.05,
    initial_temperature=1.0, final_temperature=0.3,
    teacher_weight=1.0, energy_weight=0.1,
    warm_start=True,
    use_incremental_sqd=False, use_sparse_det_solver=True,
    monotonic_basis=False,
)


def _mk(base, **overrides):
    out = dict(base)
    out.update(overrides)
    return out


CONFIGS = {
    # === OLD: reproduces 43k basis run (with n_α × n_β diagnostic) ===
    "old_orig_incremental": dict(
        n_samples=100_000, top_k=10_000, max_basis_size=0,
        max_iterations=50, convergence_threshold=1e-8, convergence_window=3,
        nqs_steps=7, nqs_lr=3e-4, entropy_weight=0.15,
        warm_start=True,
        use_incremental_sqd=True, use_sparse_det_solver=False,
    ),

    # === A. max_basis scan (samples=1M, top_k=100K) — main axis ===
    "mb_100k":   _mk(_NEW_BASE, n_samples=1_000_000, top_k=100_000, max_basis_size=100_000),
    "mb_200k":   _mk(_NEW_BASE, n_samples=1_000_000, top_k=100_000, max_basis_size=200_000),
    "mb_500k":   _mk(_NEW_BASE, n_samples=1_000_000, top_k=100_000, max_basis_size=500_000),
    "mb_1m":     _mk(_NEW_BASE, n_samples=1_000_000, top_k=100_000, max_basis_size=1_000_000),   # user baseline
    "mb_1500k":  _mk(_NEW_BASE, n_samples=1_000_000, top_k=100_000, max_basis_size=1_500_000),

    # === B. n_samples scan (top_k = 10% samples, max_basis=1M) ===
    "s_200k":    _mk(_NEW_BASE, n_samples=  200_000, top_k= 20_000, max_basis_size=1_000_000),
    "s_500k":    _mk(_NEW_BASE, n_samples=  500_000, top_k= 50_000, max_basis_size=1_000_000),
    "s_2m":      _mk(_NEW_BASE, n_samples=2_000_000, top_k=200_000, max_basis_size=1_000_000),

    # === C. top_k scan (samples=1M, max_basis=1M) ===
    "tk_50k":    _mk(_NEW_BASE, n_samples=1_000_000, top_k= 50_000, max_basis_size=1_000_000),
    "tk_200k":   _mk(_NEW_BASE, n_samples=1_000_000, top_k=200_000, max_basis_size=1_000_000),

    # === D. NQS training variants (all 1M/100K/1M) ===
    "var_steps10": _mk(_NEW_BASE, n_samples=1_000_000, top_k=100_000, max_basis_size=1_000_000,
                       nqs_steps=10),
    "var_mono":    _mk(_NEW_BASE, n_samples=1_000_000, top_k=100_000, max_basis_size=1_000_000,
                       monotonic_basis=True),
    "var_hot":     _mk(_NEW_BASE, n_samples=1_000_000, top_k=100_000, max_basis_size=1_000_000,
                       initial_temperature=2.0),
    "var_cold":    _mk(_NEW_BASE, n_samples=1_000_000, top_k=100_000, max_basis_size=1_000_000,
                       initial_temperature=0.5),
    "var_slowlr":  _mk(_NEW_BASE, n_samples=1_000_000, top_k=100_000, max_basis_size=1_000_000,
                       nqs_lr=3e-4),
}


def install_incremental_diagnostic():
    """Patch IncrementalSQDBackend.solve to print n_α × n_β each call."""
    orig = IncrementalSQDBackend.solve
    state = {"iter": 0, "history": []}

    def patched(self, bitstring_matrix):
        n_bs = len(bitstring_matrix)
        ci_strs = bitstring_matrix_to_ci_strs(bitstring_matrix, open_shell=False)
        na = len(ci_strs[0])
        nb = len(ci_strs[1])
        davidson = na * nb
        blowup = davidson / max(n_bs, 1)
        print(f"    [DIAG iter={state['iter']}] bitstrings={n_bs:,}  "
              f"n_alpha={na:,}  n_beta={nb:,}  Davidson_dim={davidson:,}  "
              f"blowup={blowup:.2f}x", flush=True)
        state["history"].append({
            "iter": state["iter"],
            "bitstrings": n_bs,
            "n_alpha": na,
            "n_beta": nb,
            "davidson_dim": int(davidson),
        })
        state["iter"] += 1
        return orig(self, bitstring_matrix)

    IncrementalSQDBackend.solve = patched
    return state


def install_batched_sampler(batch_size=50_000):
    """Batch NQS sampling so the Transformer FFN doesn't blow GPU memory on 1M+ samples."""
    orig = AutoregressiveTransformer.sample

    @torch.no_grad()
    def _batched(self, n_samples, hard=True, temperature=1.0):
        if n_samples <= batch_size:
            return orig(self, n_samples, hard=hard, temperature=temperature)
        cfgs = []
        lps = []
        for start in range(0, n_samples, batch_size):
            bn = min(batch_size, n_samples - start)
            c, lp = orig(self, bn, hard=hard, temperature=temperature)
            cfgs.append(c)
            lps.append(lp)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(cfgs, dim=0), torch.cat(lps, dim=0)

    AutoregressiveTransformer.sample = _batched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    install_batched_sampler()
    diag_state = None
    cfg_dict = CONFIGS[args.config]
    if cfg_dict.get("use_incremental_sqd", False):
        diag_state = install_incremental_diagnostic()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Config: {args.config}", flush=True)
    print(f"Seed:   {args.seed}", flush=True)
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Params: {cfg_dict}", flush=True)
    print(f"HCI HF ref = {REF_HCI_HF:.10f}", flush=True)
    print(f"{'='*90}", flush=True)

    H, info = get_molecule("N2-CAS(10,20)", device=device)
    cfg = HINQSSQDConfig(**cfg_dict)

    t0 = time.time()
    result = run_hi_nqs_sqd(H, info, config=cfg)
    wall = time.time() - t0

    err_mha = (result.energy - REF_HCI_HF) * 1000 if result.energy is not None else None
    print(f"\n{'='*90}")
    print(f"  FINAL")
    print(f"{'='*90}")
    print(f"  config      = {args.config}")
    print(f"  seed        = {args.seed}")
    print(f"  final E     = {result.energy}")
    if err_mha is not None:
        print(f"  err vs HCI  = {err_mha:+.3f} mHa")
    print(f"  final basis = {result.diag_dim}")
    print(f"  converged   = {result.converged}")
    print(f"  wall        = {wall:.1f}s ({wall/60:.1f} min)", flush=True)

    record = {
        "config_name": args.config,
        "seed": args.seed,
        "config": cfg_dict,
        "system": "N2-CAS(10,20)",
        "orbitals": "HF canonical",
        "ref_hci_hf": REF_HCI_HF,
        "energy": float(result.energy) if result.energy is not None else None,
        "basis": int(result.diag_dim),
        "wall_time_s": wall,
        "converged": bool(result.converged),
        "err_vs_hci_mha": float(err_mha) if err_mha is not None else None,
        "energy_history": [float(e) for e in result.metadata.get("energy_history", [])],
        "basis_history": [int(b) for b in result.metadata.get("basis_size_history", [])],
    }
    if diag_state is not None:
        record["davidson_diagnostic"] = diag_state["history"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n-> saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
