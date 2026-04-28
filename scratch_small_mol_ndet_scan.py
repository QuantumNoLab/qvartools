"""Small-molecule N_det vs error benchmark.

Core research claim: given the same N_det budget, HI+NQS+SQD (sparse_det
backend) reaches lower energy than HCI and CIPSI.

For each small molecule with FCI tractable:
  - FCI baseline (ground truth)
  - HCI  scan: 5 select_cutoff values   -> (N_det = n_a * n_b, E)
  - CIPSI scan: 5 max_basis budgets     -> (N_det = basis size, E)
  - HI+NQS+SQD (sparse_det, evict): 5 budgets x 3 seeds

Designed to be called per (molecule, mode, seed) so jobs can be distributed
across many GPUs. Each run saves to results/smallmol_{tag}.json and the
aggregator merges them.
"""
import argparse
import gc
import json
import time
from math import comb
from pathlib import Path

import numpy as np
import torch
from pyscf.fci import selected_ci

from src.molecules import get_molecule
from src.solvers.fci import FCISolver
from src.solvers.sci import CIPSISolver
from src.methods.hi_nqs_sqd import HINQSSQDConfig, run_hi_nqs_sqd


MOLECULES = {
    "H2O":  {"budgets": [30, 60, 120, 250, 400]},        # hilbert 441
    "BeH2": {"budgets": [50, 100, 250, 500, 1000]},      # hilbert 1225
    "NH3":  {"budgets": [100, 250, 500, 1000, 2000]},    # hilbert 3136
    "CH4":  {"budgets": [200, 500, 1500, 5000, 10000]},  # hilbert 15876
    "N2":   {"budgets": [200, 500, 1500, 5000, 10000]},  # hilbert 14400
}

HCI_CUTOFFS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
ALL_SEEDS = [42, 2024, 777]


def run_fci(H, info):
    r = FCISolver().solve(H, info)
    return {"E": r.energy, "N_det": r.diag_dim, "time": r.wall_time}


def run_hci_scan(H, cutoffs):
    integrals = H.integrals
    hcore = np.asarray(integrals.h1e, dtype=np.float64)
    eri = np.asarray(integrals.h2e, dtype=np.float64)
    ecore = float(integrals.nuclear_repulsion)
    n_orb = H.n_orbitals
    nelec = (H.n_alpha, H.n_beta)

    out = []
    for eps in cutoffs:
        myci = selected_ci.SelectedCI()
        myci.select_cutoff = eps
        myci.ci_coeff_cutoff = eps
        myci.conv_tol = 1e-10
        myci.max_cycle = 100

        t0 = time.time()
        try:
            e_elec, civec = myci.kernel(hcore, eri, n_orb, nelec)
        except Exception as ex:
            out.append({"cutoff": eps, "error": str(ex)})
            continue
        elapsed = time.time() - t0

        strs = getattr(civec, "_strs", None)
        if strs is not None:
            try:
                n_a = len(strs[0])
                n_b = len(strs[1])
                n_det = n_a * n_b
            except Exception:
                n_a = n_b = None
                n_det = int(np.asarray(civec).size)
        else:
            n_a = n_b = None
            n_det = int(np.asarray(civec).size)

        out.append({
            "cutoff": eps,
            "E": float(e_elec) + ecore,
            "N_det": int(n_det),
            "n_a": n_a,
            "n_b": n_b,
            "time": elapsed,
        })
    return out


def run_cipsi_scan(H, info, budgets):
    out = []
    for budget in budgets:
        solver = CIPSISolver(
            max_iterations=50,
            max_basis_size=budget,
            convergence_threshold=1e-7,
            expansion_size=max(20, budget // 5),
        )
        t0 = time.time()
        try:
            res = solver.solve(H, info)
        except Exception as ex:
            out.append({"budget": budget, "error": str(ex)})
            continue
        elapsed = time.time() - t0
        out.append({
            "budget": budget,
            "E": res.energy,
            "N_det": res.diag_dim,
            "time": elapsed,
            "converged": res.converged,
        })
    return out


def run_nqs_single(H, info, budget, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    cfg = HINQSSQDConfig(
        max_iterations=30,
        convergence_threshold=1e-6,
        convergence_window=3,
        n_samples=max(5000, budget * 10),
        top_k=budget,
        max_basis_size=budget,
        nqs_steps=5,
        use_sparse_det_solver=True,
        use_incremental_sqd=False,
        monotonic_basis=False,  # evict mode caps at budget
    )
    t0 = time.time()
    try:
        res = run_hi_nqs_sqd(H, info, config=cfg)
    except Exception as ex:
        return {"seed": seed, "budget": budget, "error": str(ex)}
    elapsed = time.time() - t0
    return {
        "seed": seed,
        "budget": budget,
        "E": res.energy,
        "N_det": res.diag_dim,
        "time": elapsed,
        "converged": res.converged,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", type=str, required=True)
    parser.add_argument("--mode", choices=["classical", "nqs"], required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="NQS seed; None = all seeds")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    mol_name = args.molecule
    if mol_name not in MOLECULES:
        raise SystemExit(f"Unknown molecule '{mol_name}'")
    budgets = MOLECULES[mol_name]["budgets"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if (args.mode == "nqs" and torch.cuda.is_available()) else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print(f"\n{'='*90}")
    print(f"  {mol_name}   mode={args.mode}   seed={args.seed}   budgets={budgets}")
    print(f"{'='*90}", flush=True)

    H, info = get_molecule(mol_name, device=device)
    hilbert = comb(H.n_orbitals, H.n_alpha) * comb(H.n_orbitals, H.n_beta)
    print(f"  n_orb={H.n_orbitals}, n_a={H.n_alpha}, n_b={H.n_beta}, "
          f"hilbert={hilbert:,}", flush=True)

    record = {
        "molecule": mol_name,
        "mode": args.mode,
        "seed": args.seed,
        "hilbert": hilbert,
        "n_orbitals": H.n_orbitals,
        "n_alpha": H.n_alpha,
        "n_beta": H.n_beta,
    }

    if args.mode == "classical":
        print(f"\n  [FCI]", flush=True)
        fci = run_fci(H, info)
        print(f"    E_FCI = {fci['E']:.10f}   N={fci['N_det']:,}   "
              f"t={fci['time']:.2f}s", flush=True)
        record["fci"] = fci

        print(f"\n  [HCI]  cutoffs={HCI_CUTOFFS}", flush=True)
        hci_res = run_hci_scan(H, HCI_CUTOFFS)
        for r in hci_res:
            if "error" in r:
                print(f"    cutoff={r['cutoff']:.0e}  FAILED: {r['error']}",
                      flush=True)
            else:
                err = (r["E"] - fci["E"]) * 1000
                print(f"    cutoff={r['cutoff']:.0e}  E={r['E']:.10f}  "
                      f"N={r['N_det']:>6d} (n_a={r['n_a']} n_b={r['n_b']})  "
                      f"err={err:+8.3f} mHa  t={r['time']:.2f}s", flush=True)
        record["hci"] = hci_res

        print(f"\n  [CIPSI]  budgets={budgets}", flush=True)
        cipsi_res = run_cipsi_scan(H, info, budgets)
        for r in cipsi_res:
            if "error" in r:
                print(f"    budget={r['budget']:>5d}  FAILED: {r['error']}",
                      flush=True)
            else:
                err = (r["E"] - fci["E"]) * 1000
                print(f"    budget={r['budget']:>5d}  E={r['E']:.10f}  "
                      f"N={r['N_det']:>6d}  err={err:+8.3f} mHa  "
                      f"t={r['time']:.2f}s  converged={r['converged']}",
                      flush=True)
        record["cipsi"] = cipsi_res

    elif args.mode == "nqs":
        seeds = [args.seed] if args.seed is not None else ALL_SEEDS
        nqs_res = []
        for budget in budgets:
            per_b = []
            for sd in seeds:
                print(f"\n  [NQS]  budget={budget}  seed={sd}", flush=True)
                r = run_nqs_single(H, info, budget, sd)
                if "error" in r:
                    print(f"    FAILED: {r['error']}", flush=True)
                else:
                    print(f"    E={r['E']:.10f}  N={r['N_det']:>6d}  "
                          f"t={r['time']:.2f}s  converged={r['converged']}",
                          flush=True)
                per_b.append(r)
            nqs_res.append({"budget": budget, "runs": per_b})
        record["nqs"] = nqs_res

    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n-> saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
