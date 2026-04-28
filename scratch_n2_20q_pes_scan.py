"""N2 20Q STO-3G PES scan: test NQS on strongly correlated stretched geometries.

At equilibrium (R~1.1 Å) N2 is weakly correlated; CCSD(T) is near-exact.
At stretched bonds (R>2 Å) static correlation dominates; CCSD(T) overshoots
or fails. FCI is the ground truth (Hilbert=14,400, tractable). The test: does
v4 NQS+SQD track FCI across the full PES including strongly correlated region?

Methods:
  - FCI (PySCF direct_spin1) — exact reference, hilbert=14400
  - CCSD, CCSD(T) — standard single-reference baseline
  - CIPSI — deterministic selected CI
  - HI+NQS+SQD v4 — with classical_expansion + final PT2

Bond lengths: 0.8, 1.0, 1.1 (eq), 1.3, 1.5, 1.8, 2.2, 2.6, 3.0, 4.0 Å
  (eq + dissociation path, covering static correlation regime)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from pyscf import gto, scf, cc, fci, mcscf
from pyscf.fci import selected_ci

from src.hamiltonians.molecular import compute_molecular_integrals, MolecularHamiltonian
from src.methods.hi_nqs_sqd_v4 import HINQSSQDv4Config, run_hi_nqs_sqd_v4
from src.solvers.fci import FCISolver
from src.solvers.sci import CIPSISolver
from src.nqs.transformer import AutoregressiveTransformer


HCI_CUTOFFS = [1e-2, 1e-3, 1e-4, 1e-5]  # multiple cutoffs to trace N_det vs err curve


BOND_LENGTHS = [0.8, 1.0, 1.1, 1.3, 1.5, 1.8, 2.2, 2.6, 3.0, 4.0]


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


def make_n2(R, basis="sto-3g"):
    """Build N2 at bond length R."""
    geometry = [("N", (0, 0, 0)), ("N", (0, 0, R))]
    integrals = compute_molecular_integrals(geometry, basis=basis)
    return MolecularHamiltonian(integrals, device="cuda" if torch.cuda.is_available() else "cpu")


def run_ccsd(R):
    mol = gto.M(atom=f"N 0 0 0; N 0 0 {R}", basis="sto-3g", spin=0, verbose=0)
    mf = scf.RHF(mol).run()
    out = {"hf": float(mf.e_tot), "rhf_converged": bool(mf.converged)}
    try:
        mycc = cc.CCSD(mf).run()
        e_ccsd = float(mycc.e_tot)
        et = float(mycc.ccsd_t())
        out["ccsd"] = e_ccsd
        out["ccsdt"] = e_ccsd + et
        out["ccsd_converged"] = bool(mycc.converged)
    except Exception as ex:
        out["ccsd"] = None
        out["ccsdt"] = None
        out["ccsd_error"] = str(ex)
    return out


def run_fci(R, H):
    """FCI reference. N2 STO-3G hilbert=14,400 → trivial."""
    info = {"n_qubits": 2 * H.n_orbitals}
    t0 = time.time()
    r = FCISolver().solve(H, info)
    return {"energy": float(r.energy), "diag_dim": int(r.diag_dim),
            "time": time.time() - t0}


def run_hci(R):
    """PySCF HCI at multiple cutoffs. Returns list of {'cutoff','E','N_det','time'}."""
    mol = gto.Mole()
    mol.atom = f"N 0 0 0; N 0 0 {R}"
    mol.basis = "sto-3g"; mol.spin = 0; mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol); mf.kernel()

    n_orb = mol.nao_nr()
    n_elec = mol.nelectron
    out = []
    for eps in HCI_CUTOFFS:
        mc = mcscf.CASCI(mf, n_orb, n_elec)
        myci = selected_ci.SCI(mol)
        myci.select_cutoff = eps
        myci.ci_coeff_cutoff = eps
        mc.fcisolver = myci
        t0 = time.time()
        try:
            mc.kernel()
        except Exception as ex:
            out.append({"cutoff": eps, "error": str(ex)})
            continue
        elapsed = time.time() - t0
        strs = getattr(myci, "_strs", None)
        if strs is not None:
            n_a = len(strs[0]); n_b = len(strs[1])
            n_det = n_a * n_b
        else:
            n_det = int(np.asarray(mc.ci).size)
        out.append({
            "cutoff": float(eps), "E": float(mc.e_tot),
            "N_det": int(n_det), "time": elapsed,
        })
    return out


def run_cipsi(R, H):
    info = {"n_qubits": 2 * H.n_orbitals}
    solver = CIPSISolver(max_iterations=50, max_basis_size=5000,
                        convergence_threshold=1e-7, expansion_size=500)
    t0 = time.time()
    r = solver.solve(H, info)
    return {"energy": float(r.energy), "diag_dim": int(r.diag_dim),
            "time": time.time() - t0, "converged": bool(r.converged)}


def run_nqs(R, H, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    info = {"n_qubits": 2 * H.n_orbitals}
    cfg = HINQSSQDv4Config(
        n_samples=100_000, top_k=10_000, max_basis_size=10_000,
        max_iterations=30, convergence_threshold=1e-7, convergence_window=3,
        nqs_steps=5, nqs_lr=1e-3, entropy_weight=0.05,
        warm_start=True, use_sparse_det_solver=True, use_incremental_sqd=False,
        classical_seed=False, classical_expansion=True,
        classical_expansion_top_n=500, final_pt2_correction=True, pt2_top_n=2000,
        use_gpu_sparse_det=True, use_gpu_coupling=True,
    )
    t0 = time.time()
    r = run_hi_nqs_sqd_v4(H, info, config=cfg)
    wall = time.time() - t0
    m = r.metadata
    return {
        "energy_total": float(m.get("e_total", r.energy)),
        "energy_var": float(m.get("e_var", r.energy)) if m.get("e_var") is not None else None,
        "e_pt2": float(m.get("e_pt2", 0)),
        "diag_dim": int(r.diag_dim),
        "time": wall, "converged": bool(r.converged),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bond_length", type=float, required=True)
    parser.add_argument("--methods", type=str, default="fci,ccsd,hci,cipsi,nqs")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    install_batched_sampler()
    R = args.bond_length
    methods = args.methods.split(",")

    print(f"N2 STO-3G @ R={R:.3f} Å (20Q, hilbert=14,400)", flush=True)
    print(f"Methods: {methods}", flush=True)
    print(f"{'='*70}", flush=True)

    record = {"R": R, "basis": "sto-3g", "qubits": 20}

    if "ccsd" in methods:
        print(f"\n[CCSD / CCSD(T)]", flush=True)
        t0 = time.time()
        record["ccsd"] = run_ccsd(R)
        print(f"  HF      = {record['ccsd']['hf']:.10f}", flush=True)
        if record['ccsd'].get('ccsd') is not None:
            print(f"  CCSD    = {record['ccsd']['ccsd']:.10f}", flush=True)
            print(f"  CCSD(T) = {record['ccsd']['ccsdt']:.10f}", flush=True)
        else:
            print(f"  CCSD FAILED: {record['ccsd'].get('ccsd_error')}", flush=True)

    H = None
    if any(m in methods for m in ["fci", "cipsi", "nqs"]):
        H = make_n2(R)

    if "fci" in methods:
        print(f"\n[FCI]", flush=True)
        record["fci"] = run_fci(R, H)
        print(f"  E_FCI   = {record['fci']['energy']:.10f}  ({record['fci']['time']:.1f}s)",
              flush=True)

    if "hci" in methods:
        print(f"\n[HCI scan] cutoffs={HCI_CUTOFFS}", flush=True)
        record["hci"] = run_hci(R)
        for h in record["hci"]:
            if "error" in h:
                print(f"  cutoff={h['cutoff']:.0e}  FAILED: {h['error']}", flush=True)
            else:
                print(f"  cutoff={h['cutoff']:.0e}  E={h['E']:.10f}  "
                      f"N_det={h['N_det']:>7,}  t={h['time']:.1f}s", flush=True)

    if "cipsi" in methods:
        print(f"\n[CIPSI]", flush=True)
        record["cipsi"] = run_cipsi(R, H)
        print(f"  E_CIPSI = {record['cipsi']['energy']:.10f}  "
              f"basis={record['cipsi']['diag_dim']}  ({record['cipsi']['time']:.1f}s)",
              flush=True)

    if "nqs" in methods:
        print(f"\n[HI+NQS+SQD v4]", flush=True)
        record["nqs"] = run_nqs(R, H)
        print(f"  E_var   = {record['nqs']['energy_var']:.10f}", flush=True)
        print(f"  E_PT2   = {record['nqs']['e_pt2']:+.10f}", flush=True)
        print(f"  E_total = {record['nqs']['energy_total']:.10f}  "
              f"basis={record['nqs']['diag_dim']}  ({record['nqs']['time']:.1f}s)",
              flush=True)

    # Errors vs FCI
    if "fci" in record:
        ref = record["fci"]["energy"]
        print(f"\n[Errors vs FCI, mHa]", flush=True)
        if record.get("ccsd") and record["ccsd"].get("ccsd") is not None:
            print(f"  CCSD    err = {(record['ccsd']['ccsd'] - ref)*1000:+.3f}", flush=True)
            print(f"  CCSD(T) err = {(record['ccsd']['ccsdt'] - ref)*1000:+.3f}", flush=True)
        if record.get("hci"):
            for h in record["hci"]:
                if "error" in h: continue
                print(f"  HCI ε={h['cutoff']:.0e} err = "
                      f"{(h['E'] - ref)*1000:+.3f} mHa  (N={h['N_det']:,})", flush=True)
        if record.get("cipsi"):
            print(f"  CIPSI   err = {(record['cipsi']['energy'] - ref)*1000:+.3f}", flush=True)
        if record.get("nqs"):
            print(f"  NQS tot err = {(record['nqs']['energy_total'] - ref)*1000:+.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n-> saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
