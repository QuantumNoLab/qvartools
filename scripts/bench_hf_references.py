#!/usr/bin/env python3
"""
Reference benchmarks for all STO-3G small molecules in HF canonical orbital basis.

For each molecule, runs:
  - RHF           (baseline)
  - CCSD          (PySCF cc.CCSD)
  - CCSD(T)       (PySCF perturbative triples)
  - FCI           (PySCF direct_spin1, gated by Hilbert ≤ 1M)
  - HCI           (PySCF selected_ci.SCI — heat-bath Selected CI with tight cutoff)
  - CIPSI         (project's CIPSISolver — iterative PT2 selection)

Geometries match the factories in src/hamiltonians/molecular.py and src/molecules.py
so energies are directly comparable to rerun_small_mol.json (the HI-NQS results).

Stream-saves to results/hf_references.json after every molecule completes.
"""
import sys, time, json
from pathlib import Path
from math import comb

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyscf import gto, scf, cc, ao2mo
from pyscf.fci import direct_spin1, selected_ci

from src.molecules import get_molecule
from src.solvers.sci import CIPSISolver

RESULTS_PATH = Path(__file__).parent.parent / "results" / "hf_references.json"

# ---------------------------------------------------------------------------
# Build geometries matching the factories in src/
# ---------------------------------------------------------------------------
def _h2o():
    oh = 0.96
    ang = np.radians(104.5)
    return [("O", (0.0, 0.0, 0.0)),
            ("H", (oh, 0.0, 0.0)),
            ("H", (oh * np.cos(ang), oh * np.sin(ang), 0.0))]

def _nh3():
    nh = 1.01
    ang = np.radians(107.8)
    h = nh * np.cos(np.arcsin(np.sin(ang / 2) / np.sin(np.radians(60))))
    r = float(np.sqrt(nh**2 - h**2))
    return [("N", (0.0, 0.0, h)),
            ("H", (r, 0.0, 0.0)),
            ("H", (r * np.cos(np.radians(120)), r * np.sin(np.radians(120)), 0.0)),
            ("H", (r * np.cos(np.radians(240)), r * np.sin(np.radians(240)), 0.0))]

def _ch4():
    a = 1.09 / np.sqrt(3)
    return [("C", (0.0, 0.0, 0.0)),
            ("H", (a, a, a)),
            ("H", (a, -a, -a)),
            ("H", (-a, a, -a)),
            ("H", (-a, -a, a))]

def _h2s():
    sh = 1.34
    ang = np.radians(92.1)
    return [("S", (0.0, 0.0, 0.0)),
            ("H", (sh, 0.0, 0.0)),
            ("H", (sh * np.cos(ang), sh * np.sin(ang), 0.0))]

def _c2h4():
    cc_len = 1.33
    ch = 1.09
    ang = np.radians(121.3)
    h1 = (-ch * np.cos(np.pi - ang), ch * np.sin(np.pi - ang), 0.0)
    h2 = (-ch * np.cos(np.pi - ang), -ch * np.sin(np.pi - ang), 0.0)
    h3 = (cc_len + ch * np.cos(np.pi - ang), ch * np.sin(np.pi - ang), 0.0)
    h4 = (cc_len + ch * np.cos(np.pi - ang), -ch * np.sin(np.pi - ang), 0.0)
    return [("C", (0.0, 0.0, 0.0)),
            ("C", (cc_len, 0.0, 0.0)),
            ("H", h1), ("H", h2), ("H", h3), ("H", h4)]

def _benzene():
    cc_d = 1.40
    ch_d = 1.08
    atoms = []
    for i in range(6):
        ang = np.pi / 3 * i
        cx, cy = cc_d * np.cos(ang), cc_d * np.sin(ang)
        atoms.append(("C", (cx, cy, 0.0)))
        hx, hy = (cc_d + ch_d) * np.cos(ang), (cc_d + ch_d) * np.sin(ang)
        atoms.append(("H", (hx, hy, 0.0)))
    return atoms

MOLECULES = {
    # name:           (geometry builder, basis, charge, spin)
    "LiH":   (lambda: [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.6))],  "sto-3g", 0, 0),
    "H2O":   (_h2o,  "sto-3g", 0, 0),
    "BeH2":  (lambda: [("Be", (0, 0, 0)), ("H", (0, 0, 1.33)), ("H", (0, 0, -1.33))], "sto-3g", 0, 0),
    "NH3":   (_nh3,  "sto-3g", 0, 0),
    "CH4":   (_ch4,  "sto-3g", 0, 0),
    "N2":    (lambda: [("N", (0, 0, 0)), ("N", (0, 0, 1.10))], "sto-3g", 0, 0),
    "HCN":   (lambda: [("H", (0, 0, 0)), ("C", (0, 0, 1.06)), ("N", (0, 0, 1.06 + 1.16))], "sto-3g", 0, 0),
    "C2H2":  (lambda: [("H", (0, 0, 0)), ("C", (0, 0, 1.06)), ("C", (0, 0, 1.06 + 1.20)), ("H", (0, 0, 1.06 + 1.20 + 1.06))], "sto-3g", 0, 0),
    "H2S":   (_h2s,  "sto-3g", 0, 0),
    "C2H4":  (_c2h4, "sto-3g", 0, 0),
    "Benzene": (_benzene, "sto-3g", 0, 0),
}

# FCI gate: skip if Hilbert space exceeds this
FCI_LIMIT = 1_000_000
# HCI cutoff — tight enough to approach FCI closely on small molecules
HCI_CUTOFF = 1e-6


# ---------------------------------------------------------------------------
def run_methods_on(mol_name, geom_fn, basis, charge, spin):
    """Run all reference methods on one molecule. Returns dict."""
    print(f"\n{'='*78}", flush=True)
    print(f"  {mol_name}  ({basis})", flush=True)
    print(f"{'='*78}", flush=True)

    out = {"molecule": mol_name, "basis_set": basis}

    # 1. Build molecule + RHF
    t0 = time.time()
    mol = gto.Mole()
    mol.atom = geom_fn()
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.verbose = 0
    mol.symmetry = False
    mol.build()

    mf = scf.RHF(mol)
    mf.max_cycle = 200
    mf.kernel()
    t_hf = time.time() - t0

    out["n_orbitals"] = int(mol.nao)
    out["n_alpha"] = int(mol.nelec[0])
    out["n_beta"] = int(mol.nelec[1])
    out["n_qubits"] = 2 * int(mol.nao)
    hilbert = comb(mol.nao, mol.nelec[0]) * comb(mol.nao, mol.nelec[1])
    out["hilbert"] = int(hilbert)

    out["HF"] = {"energy": float(mf.e_tot), "time_s": float(t_hf), "converged": bool(mf.converged)}
    print(f"  HF:       E={mf.e_tot:.10f}   t={t_hf:.2f}s", flush=True)

    # 2. Integrals in MO basis (HF canonical)
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h2e = ao2mo.kernel(mol, mf.mo_coeff)
    h2e = ao2mo.restore(1, h2e, mol.nao)
    e_core = float(mol.energy_nuc())
    norb = int(mol.nao)
    nelec = (int(mol.nelec[0]), int(mol.nelec[1]))

    # 3. CCSD
    try:
        t0 = time.time()
        mycc = cc.CCSD(mf)
        mycc.max_cycle = 200
        mycc.verbose = 0
        mycc.kernel()
        t_ccsd = time.time() - t0
        out["CCSD"] = {
            "energy": float(mycc.e_tot),
            "time_s": float(t_ccsd),
            "converged": bool(mycc.converged),
        }
        print(f"  CCSD:     E={mycc.e_tot:.10f}   t={t_ccsd:.2f}s   conv={mycc.converged}", flush=True)
    except Exception as ex:
        out["CCSD"] = {"error": str(ex)}
        print(f"  CCSD:     FAILED ({ex})", flush=True)
        mycc = None

    # 4. CCSD(T)
    if mycc is not None and mycc.converged:
        try:
            t0 = time.time()
            et = mycc.ccsd_t()
            t_ccsdt = time.time() - t0
            out["CCSD(T)"] = {
                "energy": float(mycc.e_tot + et),
                "time_s": float(t_ccsd + t_ccsdt),
                "et": float(et),
            }
            print(f"  CCSD(T):  E={mycc.e_tot + et:.10f}   t={t_ccsd + t_ccsdt:.2f}s", flush=True)
        except Exception as ex:
            out["CCSD(T)"] = {"error": str(ex)}
            print(f"  CCSD(T):  FAILED ({ex})", flush=True)

    # 5. FCI (gated by Hilbert size)
    if hilbert <= FCI_LIMIT:
        try:
            t0 = time.time()
            fci_e, fci_ci = direct_spin1.kernel(h1e, h2e, norb, nelec)
            t_fci = time.time() - t0
            out["FCI"] = {
                "energy": float(fci_e + e_core),
                "time_s": float(t_fci),
                "basis": int(hilbert),
            }
            print(f"  FCI:      E={fci_e + e_core:.10f}   t={t_fci:.2f}s   basis={hilbert:,}", flush=True)
        except Exception as ex:
            out["FCI"] = {"error": str(ex)}
            print(f"  FCI:      FAILED ({ex})", flush=True)
    else:
        print(f"  FCI:      SKIPPED (Hilbert={hilbert:,} > {FCI_LIMIT:,})", flush=True)
        out["FCI"] = {"skipped": True, "reason": f"Hilbert {hilbert:,} > {FCI_LIMIT:,}"}

    # 6. HCI (PySCF selected_ci.SCI with tight heat-bath cutoff)
    try:
        t0 = time.time()
        myci = selected_ci.SCI()
        myci.select_cutoff = HCI_CUTOFF
        myci.ci_coeff_cutoff = HCI_CUTOFF
        myci.max_cycle = 50
        myci.verbose = 0
        hci_e, hci_vec = myci.kernel(h1e, h2e, norb, nelec)
        t_hci = time.time() - t0

        # Extract basis dimensions from the final CI vector
        try:
            ci_strs = getattr(hci_vec, "_strs", None)
            if ci_strs is not None:
                na_strs = len(np.asarray(ci_strs[0]))
                nb_strs = len(np.asarray(ci_strs[1]))
                hci_basis = na_strs * nb_strs
            else:
                hci_basis = int(np.asarray(hci_vec).size)
        except Exception:
            hci_basis = None

        out["HCI"] = {
            "energy": float(hci_e + e_core),
            "time_s": float(t_hci),
            "basis": int(hci_basis) if hci_basis is not None else None,
            "cutoff": HCI_CUTOFF,
        }
        print(f"  HCI:      E={hci_e + e_core:.10f}   t={t_hci:.2f}s   basis={hci_basis}", flush=True)
    except Exception as ex:
        out["HCI"] = {"error": str(ex)}
        print(f"  HCI:      FAILED ({ex})", flush=True)

    # 7. CIPSI (project solver, operates on MolecularHamiltonian API)
    try:
        H, info = get_molecule(mol_name)
        cipsi = CIPSISolver(
            max_iterations=30,
            max_basis_size=0,         # unlimited
            convergence_threshold=1e-8,
            expansion_size=500,
        )
        t0 = time.time()
        r = cipsi.solve(H, info)
        t_cipsi = time.time() - t0
        out["CIPSI"] = {
            "energy": float(r.energy),
            "time_s": float(t_cipsi),
            "basis": int(r.diag_dim),
            "converged": bool(r.converged),
        }
        print(f"  CIPSI:    E={r.energy:.10f}   t={t_cipsi:.2f}s   basis={r.diag_dim}   conv={r.converged}", flush=True)
    except Exception as ex:
        out["CIPSI"] = {"error": str(ex)}
        print(f"  CIPSI:    FAILED ({ex})", flush=True)

    return out


# ---------------------------------------------------------------------------
def main():
    all_results = []

    # Resume support
    done = set()
    if RESULTS_PATH.exists():
        try:
            with open(RESULTS_PATH) as f:
                all_results = json.load(f).get("runs", [])
            done = {r["molecule"] for r in all_results}
            print(f"[Resume] {len(done)} already done: {sorted(done)}", flush=True)
        except Exception:
            pass

    for name, (geom_fn, basis, charge, spin) in MOLECULES.items():
        if name in done:
            print(f"\n[Skip] {name}", flush=True)
            continue
        try:
            r = run_methods_on(name, geom_fn, basis, charge, spin)
        except Exception as ex:
            import traceback; traceback.print_exc()
            r = {"molecule": name, "error": str(ex)}
        all_results.append(r)

        # Stream save
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump({
                "description": "HF canonical reference benchmarks (CCSD, CCSD(T), FCI, HCI, CIPSI)",
                "hci_cutoff": HCI_CUTOFF,
                "fci_limit": FCI_LIMIT,
                "runs": all_results,
            }, f, indent=2)

    # Final summary
    print(f"\n{'='*78}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'='*78}", flush=True)
    hdr = f"  {'Mol':<10} {'Q':>3} {'Hilbert':>12}  {'HI-NQS':>14}  {'FCI':>14}  {'CCSD(T)':>14}  {'HCI':>14}  {'CIPSI':>14}"
    print(hdr, flush=True)
    print(f"  {'-'*len(hdr)}", flush=True)
    for r in all_results:
        if "error" in r:
            continue
        mol = r["molecule"]
        q = r["n_qubits"]
        hilb = r["hilbert"]
        fci_e = r.get("FCI", {}).get("energy", None)
        cct_e = r.get("CCSD(T)", {}).get("energy", None)
        hci_e = r.get("HCI", {}).get("energy", None)
        cip_e = r.get("CIPSI", {}).get("energy", None)
        def fmt(e):
            return f"{e:.10f}" if e is not None else "—"
        print(f"  {mol:<10} {q:>3} {hilb:>12,}  {'':>14}  {fmt(fci_e):>14}  {fmt(cct_e):>14}  {fmt(hci_e):>14}  {fmt(cip_e):>14}",
              flush=True)

    print(f"\n  Saved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
