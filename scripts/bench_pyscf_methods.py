#!/usr/bin/env python3
"""
PySCF multi-method benchmark: CASSCF, CCSD, CCSD(T), HCI (Selected CI)
4 methods run in parallel, each pinned to 1 GPU process.

Molecules: H2O, NH3, N2 (20Q), C2H2 (24Q), C2H4 (28Q),
           N2-CAS(10,20) (40Q), N2-CAS(10,26) (52Q)
"""
import os, sys, time, traceback
import multiprocessing as mp
from pathlib import Path

import numpy as np

# PySCF
from pyscf import gto, scf, cc, mcscf
from pyscf.fci import selected_ci

# ─────────────────────────────────────────────────────────────────────────────
# Molecule definitions (geometry + basis, matching src/molecules.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
def build_pyscf_mol(name):
    """Return (mol, n_elec_alpha, n_elec_beta, cas_nelec, cas_norb, label)."""
    mol = gto.Mole()
    mol.verbose = 3
    mol.output = f"/dev/null"  # suppress to stdout only

    if name == "H2O":
        import math
        oh = 0.96; ang = math.radians(104.5)
        mol.atom = [
            ("O", (0, 0, 0)),
            ("H", (oh, 0, 0)),
            ("H", (oh*math.cos(ang), oh*math.sin(ang), 0)),
        ]
        mol.basis = "sto-3g"
        mol.spin = 0; mol.charge = 0
        # 10e, 7 orb → CAS(8,6) valence
        return mol, 5, 5, 8, 6, "H2O 14Q STO-3G"

    elif name == "NH3":
        import math
        nh = 1.01; ang = math.radians(107.8)
        h = nh * math.cos(math.asin(math.sin(ang/2)/math.sin(math.radians(60))))
        r = math.sqrt(nh**2 - h**2)
        mol.atom = [
            ("N", (0, 0, h)),
            ("H", (r, 0, 0)),
            ("H", (r*math.cos(math.radians(120)), r*math.sin(math.radians(120)), 0)),
            ("H", (r*math.cos(math.radians(240)), r*math.sin(math.radians(240)), 0)),
        ]
        mol.basis = "sto-3g"
        mol.spin = 0; mol.charge = 0
        # 10e, 8 orb → CAS(8,6) valence
        return mol, 5, 5, 8, 6, "NH3 16Q STO-3G"

    elif name == "N2":
        mol.atom = [("N", (0, 0, 0)), ("N", (0, 0, 1.10))]
        mol.basis = "sto-3g"
        mol.spin = 0; mol.charge = 0
        # 14e, 10 orb → CAS(10,8)
        return mol, 7, 7, 10, 8, "N2 20Q STO-3G"

    elif name == "C2H2":
        mol.atom = [
            ("H", (0, 0, 0)),
            ("C", (0, 0, 1.06)),
            ("C", (0, 0, 1.06+1.20)),
            ("H", (0, 0, 1.06+1.20+1.06)),
        ]
        mol.basis = "sto-3g"
        mol.spin = 0; mol.charge = 0
        # 14e, 12 orb → CAS(10,10)
        return mol, 7, 7, 10, 10, "C2H2 24Q STO-3G"

    elif name == "C2H4":
        import math
        cc_l = 1.33; ch_l = 1.09
        ang = math.radians(121.3)
        c1 = (0, 0, 0); c2 = (cc_l, 0, 0)
        dx = ch_l * math.cos(math.pi - ang)
        dy = ch_l * math.sin(math.pi - ang)
        mol.atom = [
            ("C", c1), ("C", c2),
            ("H", (-dx,  dy, 0)),
            ("H", (-dx, -dy, 0)),
            ("H", (cc_l+dx,  dy, 0)),
            ("H", (cc_l+dx, -dy, 0)),
        ]
        mol.basis = "sto-3g"
        mol.spin = 0; mol.charge = 0
        # 16e, 14 orb → CAS(8,8)
        return mol, 8, 8, 8, 8, "C2H4 28Q STO-3G"

    elif name == "N2-CAS(10,20)":
        mol.atom = [("N", (0, 0, 0)), ("N", (0, 0, 1.10))]
        mol.basis = "cc-pvtz"
        mol.spin = 0; mol.charge = 0
        # CAS(10,20) = 40Q
        return mol, 7, 7, 10, 20, "N2-CAS(10,20) 40Q cc-pVTZ"

    elif name == "N2-CAS(10,26)":
        mol.atom = [("N", (0, 0, 0)), ("N", (0, 0, 1.10))]
        mol.basis = "cc-pvtz"
        mol.spin = 0; mol.charge = 0
        # CAS(10,26) = 52Q
        return mol, 7, 7, 10, 26, "N2-CAS(10,26) 52Q cc-pVTZ"

    else:
        raise ValueError(f"Unknown molecule: {name}")


MOLECULES = ["H2O", "NH3", "N2", "C2H2", "C2H4", "N2-CAS(10,20)", "N2-CAS(10,26)"]

# ─────────────────────────────────────────────────────────────────────────────
# Per-method runners
# ─────────────────────────────────────────────────────────────────────────────

def run_ccsd(result_dict, gpu_id):
    """CCSD for all molecules."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    method = "CCSD"
    print(f"[{method}] Starting on GPU slot {gpu_id}", flush=True)

    for name in MOLECULES:
        try:
            mol, na, nb, _, _, label = build_pyscf_mol(name)
            mol.build()
            mf = scf.RHF(mol); mf.max_cycle = 200
            t0 = time.time()
            mf.kernel()
            mycc = cc.CCSD(mf)
            mycc.max_cycle = 200
            mycc.kernel()
            E = mycc.e_tot
            elapsed = time.time() - t0
            n_occ = mol.nelectron // 2
            n_vir = mol.nao_nr() - n_occ
            basis_info = f"{mol.nao_nr()} AO, {n_occ}o/{n_vir}v"
            print(f"[{method}] {label}: E={E:.10f} Ha, {basis_info}, t={elapsed:.1f}s", flush=True)
            result_dict[f"{method}_{name}"] = dict(E=E, time=elapsed, basis=mol.nao_nr(), status="OK")
        except Exception as e:
            print(f"[{method}] {name} FAILED: {e}", flush=True)
            result_dict[f"{method}_{name}"] = dict(E=None, time=None, basis=None, status=str(e)[:80])

    print(f"[{method}] Done.", flush=True)


def run_ccsd_t(result_dict, gpu_id):
    """CCSD(T) for all molecules."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    method = "CCSD(T)"
    print(f"[{method}] Starting on GPU slot {gpu_id}", flush=True)

    for name in MOLECULES:
        try:
            mol, na, nb, _, _, label = build_pyscf_mol(name)
            mol.build()
            mf = scf.RHF(mol); mf.max_cycle = 200
            t0 = time.time()
            mf.kernel()
            mycc = cc.CCSD(mf)
            mycc.max_cycle = 200
            mycc.kernel()
            et = mycc.ccsd_t()
            E = mycc.e_tot + et
            elapsed = time.time() - t0
            basis_info = mol.nao_nr()
            print(f"[{method}] {label}: E={E:.10f} Ha, {basis_info} AO, t={elapsed:.1f}s", flush=True)
            result_dict[f"{method}_{name}"] = dict(E=E, time=elapsed, basis=mol.nao_nr(), status="OK")
        except Exception as e:
            print(f"[{method}] {name} FAILED: {e}", flush=True)
            result_dict[f"{method}_{name}"] = dict(E=None, time=None, basis=None, status=str(e)[:80])

    print(f"[{method}] Done.", flush=True)


def run_casscf(result_dict, gpu_id):
    """CASSCF for all molecules."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    method = "CASSCF"
    print(f"[{method}] Starting on GPU slot {gpu_id}", flush=True)

    for name in MOLECULES:
        try:
            mol, na, nb, cas_ne, cas_no, label = build_pyscf_mol(name)
            mol.build()
            mf = scf.RHF(mol); mf.max_cycle = 200

            t0 = time.time()
            mf.kernel()

            mc = mcscf.CASSCF(mf, cas_no, cas_ne)
            mc.max_cycle_macro = 100

            # For large CAS (>12 orb), use Selected CI as FCI solver
            if cas_no > 12:
                from pyscf.fci import selected_ci
                mc.fcisolver = selected_ci.SCI(mol)
                mc.fcisolver.select_cutoff = 1e-4
                mc.fcisolver.ci_coeff_cutoff = 1e-4
                print(f"[{method}] {name}: CAS({cas_ne},{cas_no}) using Selected CI solver", flush=True)

            mc.kernel()
            E = mc.e_tot
            elapsed = time.time() - t0
            from math import comb
            n_configs = comb(cas_no, cas_ne//2)**2
            print(f"[{method}] {label}: E={E:.10f} Ha, "
                  f"CAS({cas_ne},{cas_no})~{n_configs:,} configs, t={elapsed:.1f}s", flush=True)
            result_dict[f"{method}_{name}"] = dict(
                E=E, time=elapsed, basis=cas_no, cas=f"CAS({cas_ne},{cas_no})", status="OK"
            )
        except Exception as e:
            print(f"[{method}] {name} FAILED: {e}", flush=True)
            traceback.print_exc()
            result_dict[f"{method}_{name}"] = dict(E=None, time=None, basis=None, status=str(e)[:80])

    print(f"[{method}] Done.", flush=True)


def run_hci(result_dict, gpu_id):
    """HCI (Selected CI / Heat-bath CI) for all molecules."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    method = "HCI"
    print(f"[{method}] Starting on GPU slot {gpu_id}", flush=True)

    for name in MOLECULES:
        try:
            mol, na, nb, cas_ne, cas_no, label = build_pyscf_mol(name)
            mol.build()
            mf = scf.RHF(mol); mf.max_cycle = 200

            t0 = time.time()
            mf.kernel()

            # Run CASSCF to get optimized orbitals, then HCI in active space
            mc = mcscf.CASSCF(mf, cas_no, cas_ne)
            mc.max_cycle_macro = 30

            # Replace FCI solver with Selected CI (HCI)
            myci = selected_ci.SCI(mol)
            myci.select_cutoff = 1e-4   # amplitude cutoff for heat-bath selection
            myci.ci_coeff_cutoff = 1e-4
            mc.fcisolver = myci
            mc.kernel()

            E = mc.e_tot
            elapsed = time.time() - t0

            # Get number of determinants selected
            try:
                n_dets = myci.ci[0].size if hasattr(myci, 'ci') and myci.ci is not None else "?"
            except Exception:
                n_dets = "?"

            print(f"[{method}] {label}: E={E:.10f} Ha, "
                  f"CAS({cas_ne},{cas_no}), dets~{n_dets}, t={elapsed:.1f}s", flush=True)
            result_dict[f"{method}_{name}"] = dict(
                E=E, time=elapsed, basis=cas_no, cas=f"CAS({cas_ne},{cas_no})", status="OK"
            )
        except Exception as e:
            print(f"[{method}] {name} FAILED: {e}", flush=True)
            traceback.print_exc()
            result_dict[f"{method}_{name}"] = dict(E=None, time=None, basis=None, status=str(e)[:80])

    print(f"[{method}] Done.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main: launch 4 parallel workers
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    results = manager.dict()

    workers = [
        mp.Process(target=run_ccsd,   args=(results, 0), name="CCSD"),
        mp.Process(target=run_ccsd_t, args=(results, 1), name="CCSD(T)"),
        mp.Process(target=run_casscf, args=(results, 2), name="CASSCF"),
        mp.Process(target=run_hci,    args=(results, 3), name="HCI"),
    ]

    print("=" * 70, flush=True)
    print("  PySCF 4-method parallel benchmark", flush=True)
    print("  Methods: CCSD | CCSD(T) | CASSCF | HCI", flush=True)
    print("  Molecules: H2O NH3 N2(20Q) C2H2(24Q) C2H4(28Q) N2-40Q N2-52Q", flush=True)
    print("=" * 70, flush=True)

    t_global = time.time()
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    total_time = time.time() - t_global

    print(f"\n{'='*70}", flush=True)
    print(f"  SUMMARY TABLE  (wall time total: {total_time:.0f}s)", flush=True)
    print(f"{'='*70}", flush=True)

    methods = ["CCSD", "CCSD(T)", "CASSCF", "HCI"]
    mol_labels = {
        "H2O":           "H2O      14Q STO-3G",
        "NH3":           "NH3      16Q STO-3G",
        "N2":            "N2       20Q STO-3G",
        "C2H2":          "C2H2     24Q STO-3G",
        "C2H4":          "C2H4     28Q STO-3G",
        "N2-CAS(10,20)": "N2-CAS   40Q cc-pVTZ",
        "N2-CAS(10,26)": "N2-CAS   52Q cc-pVTZ",
    }

    hdr = f"  {'Molecule':<22}"
    for m in methods:
        hdr += f"  {m:>18}"
    print(hdr, flush=True)
    print(f"  {'-'*90}", flush=True)

    for mol_name, mol_label in mol_labels.items():
        row = f"  {mol_label:<22}"
        for m in methods:
            key = f"{m}_{mol_name}"
            r = results.get(key, {})
            if r.get("E") is not None:
                row += f"  {r['E']:>18.10f}"
            else:
                row += f"  {'FAIL':>18}"
        print(row, flush=True)

    print(f"\n  Timing:", flush=True)
    print(f"  {'Molecule':<22}", end="", flush=True)
    for m in methods:
        print(f"  {m+' t':>18}", end="", flush=True)
    print(flush=True)
    print(f"  {'-'*90}", flush=True)
    for mol_name, mol_label in mol_labels.items():
        row = f"  {mol_label:<22}"
        for m in methods:
            key = f"{m}_{mol_name}"
            r = results.get(key, {})
            t = r.get("time")
            row += f"  {f'{t:.1f}s' if t else 'FAIL':>18}"
        print(row, flush=True)
