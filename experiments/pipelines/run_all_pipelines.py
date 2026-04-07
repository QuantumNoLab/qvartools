"""Run all SKQD/SQD/VQE pipeline variants and compare results.

Usage:
    python run_all_pipelines.py h2
    python run_all_pipelines.py lih --skip-quantum   # skip CUDA-Q pipelines
    python run_all_pipelines.py h2 --only 001 002    # run only groups 001 and 002
    python run_all_pipelines.py h2 --skip-iterative  # skip slow iterative pipelines
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Make config_loader importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qvartools.molecules import get_molecule
from qvartools.solvers import FCISolver

CHEMICAL_ACCURACY_MHA = 1.6

# ---- All pipelines, organized by group ----
PIPELINES = [
    # (group, script_path, short_name, description)
    (
        "001_dci",
        "001_dci/dci_krylov_classical.py",
        "DCI+Krylov-C",
        "DCI → Classical Krylov",
    ),
    (
        "001_dci",
        "001_dci/dci_krylov_quantum.py",
        "DCI+Krylov-Q",
        "DCI → Quantum Krylov",
    ),
    ("001_dci", "001_dci/dci_sqd.py", "DCI+SQD", "DCI → SQD"),
    (
        "002_nf_dci",
        "002_nf_dci/nf_dci_krylov_classical.py",
        "NF+DCI+Krylov-C",
        "NF+DCI → Classical Krylov",
    ),
    (
        "002_nf_dci",
        "002_nf_dci/nf_dci_krylov_quantum.py",
        "NF+DCI+Krylov-Q",
        "NF+DCI → Quantum Krylov",
    ),
    ("002_nf_dci", "002_nf_dci/nf_dci_sqd.py", "NF+DCI+SQD", "NF+DCI → SQD"),
    (
        "003_nf_dci_pt2",
        "003_nf_dci_pt2/nf_dci_pt2_krylov_classical.py",
        "NF+DCI+PT2+Krylov-C",
        "NF+DCI+PT2 → Classical Krylov",
    ),
    (
        "003_nf_dci_pt2",
        "003_nf_dci_pt2/nf_dci_pt2_krylov_quantum.py",
        "NF+DCI+PT2+Krylov-Q",
        "NF+DCI+PT2 → Quantum Krylov",
    ),
    (
        "003_nf_dci_pt2",
        "003_nf_dci_pt2/nf_dci_pt2_sqd.py",
        "NF+DCI+PT2+SQD",
        "NF+DCI+PT2 → SQD",
    ),
    (
        "004_nf_only",
        "004_nf_only/nf_krylov_classical.py",
        "NF+Krylov-C",
        "NF-only → Classical Krylov",
    ),
    (
        "004_nf_only",
        "004_nf_only/nf_krylov_quantum.py",
        "NF+Krylov-Q",
        "NF-only → Quantum Krylov",
    ),
    ("004_nf_only", "004_nf_only/nf_sqd.py", "NF+SQD", "NF-only → SQD"),
    (
        "005_hf_only",
        "005_hf_only/hf_krylov_classical.py",
        "HF+Krylov-C",
        "HF-only → Classical Krylov",
    ),
    (
        "005_hf_only",
        "005_hf_only/hf_krylov_quantum.py",
        "HF+Krylov-Q",
        "HF-only → Quantum Krylov",
    ),
    ("005_hf_only", "005_hf_only/hf_sqd.py", "HF+SQD", "HF-only → SQD"),
    (
        "006_iterative_nqs",
        "006_iterative_nqs/iter_nqs_krylov_classical.py",
        "Iter+Krylov-C",
        "Iterative NQS → Classical Krylov",
    ),
    (
        "006_iterative_nqs",
        "006_iterative_nqs/iter_nqs_krylov_quantum.py",
        "Iter+Krylov-Q",
        "Iterative NQS → Quantum Krylov",
    ),
    (
        "006_iterative_nqs",
        "006_iterative_nqs/iter_nqs_sqd.py",
        "Iter+SQD",
        "Iterative NQS → SQD",
    ),
    (
        "007_iterative_nqs_dci",
        "007_iterative_nqs_dci/iter_nqs_dci_krylov_classical.py",
        "Iter+DCI+Krylov-C",
        "Iterative NQS+DCI → Classical Krylov",
    ),
    (
        "007_iterative_nqs_dci",
        "007_iterative_nqs_dci/iter_nqs_dci_krylov_quantum.py",
        "Iter+DCI+Krylov-Q",
        "Iterative NQS+DCI → Quantum Krylov",
    ),
    (
        "007_iterative_nqs_dci",
        "007_iterative_nqs_dci/iter_nqs_dci_sqd.py",
        "Iter+DCI+SQD",
        "Iterative NQS+DCI → SQD",
    ),
    (
        "008_iterative_nqs_dci_pt2",
        "008_iterative_nqs_dci_pt2/iter_nqs_dci_pt2_krylov_classical.py",
        "Iter+DCI+PT2+Krylov-C",
        "Iterative NQS+DCI+PT2 → Classical Krylov",
    ),
    (
        "008_iterative_nqs_dci_pt2",
        "008_iterative_nqs_dci_pt2/iter_nqs_dci_pt2_krylov_quantum.py",
        "Iter+DCI+PT2+Krylov-Q",
        "Iterative NQS+DCI+PT2 → Quantum Krylov",
    ),
    (
        "008_iterative_nqs_dci_pt2",
        "008_iterative_nqs_dci_pt2/iter_nqs_dci_pt2_sqd.py",
        "Iter+DCI+PT2+SQD",
        "Iterative NQS+DCI+PT2 → SQD",
    ),
    (
        "009_vqe",
        "009_vqe/vqe_uccsd.py",
        "VQE-UCCSD",
        "CUDA-QX VQE with UCCSD ansatz",
    ),
    (
        "009_vqe",
        "009_vqe/vqe_adapt.py",
        "ADAPT-VQE",
        "CUDA-QX ADAPT-VQE with GSD operator pool",
    ),
    # ---- Method-as-pipeline catalog (010-099) ----
    (
        "010_hi_nqs_sqd",
        "010_hi_nqs_sqd/default.py",
        "HI-NQS-SQD/default",
        "HI+NQS+SQD baseline (auto IBM detect)",
    ),
    (
        "010_hi_nqs_sqd",
        "010_hi_nqs_sqd/pt2.py",
        "HI-NQS-SQD/pt2",
        "HI+NQS+SQD with PT2 selection + temperature annealing",
    ),
    (
        "010_hi_nqs_sqd",
        "010_hi_nqs_sqd/ibm_off.py",
        "HI-NQS-SQD/ibm_off",
        "HI+NQS+SQD forced GPU fallback (no IBM)",
    ),
    (
        "011_hi_nqs_skqd",
        "011_hi_nqs_skqd/default.py",
        "HI-NQS-SKQD/default",
        "HI+NQS+SKQD baseline with Krylov expansion",
    ),
    (
        "011_hi_nqs_skqd",
        "011_hi_nqs_skqd/ibm_on.py",
        "HI-NQS-SKQD/ibm_on",
        "HI+NQS+SKQD with IBM solver + S-CORE",
    ),
    (
        "012_nqs_sqd",
        "012_nqs_sqd/default.py",
        "NQS-SQD/default",
        "Two-stage NQS+SQD (no iteration)",
    ),
    (
        "013_nqs_skqd",
        "013_nqs_skqd/default.py",
        "NQS-SKQD/default",
        "Two-stage NQS+SKQD with Krylov expansion",
    ),
]


def run_pipeline_subprocess(
    script_path: str,
    molecule: str,
    device: str,
) -> dict:
    """Run a pipeline script as a subprocess and capture results."""
    import subprocess

    pipelines_dir = Path(__file__).resolve().parent
    full_path = pipelines_dir / script_path

    if not full_path.exists():
        return {"status": "MISSING", "error": f"Script not found: {full_path}"}

    cmd = [
        sys.executable,
        str(full_path),
        molecule,
        "--device",
        device,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(pipelines_dir),
        )

        output = result.stdout + result.stderr

        # Parse energy from output
        energy = _parse_energy(output)
        error_mha = _parse_error(output)
        wall_time = _parse_wall_time(output)

        if result.returncode != 0:
            return {
                "status": "FAILED",
                "error": output[-500:] if len(output) > 500 else output,
                "energy": energy,
                "error_mha": error_mha,
                "wall_time": wall_time,
            }

        return {
            "status": "OK",
            "energy": energy,
            "error_mha": error_mha,
            "wall_time": wall_time,
        }

    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "error": "Exceeded 600s timeout"}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


def _parse_energy(output: str) -> float | None:
    """Extract final/best energy from script output."""
    import re

    for pattern in [
        r"Best energy\s*:\s*([-\d.]+)",
        r"Final energy\s*:\s*([-\d.]+)",
        r"energy\s*[:=]\s*([-\d.]+)",
    ]:
        match = re.search(pattern, output)
        if match:
            return float(match.group(1))
    return None


def _parse_error(output: str) -> float | None:
    """Extract error in mHa from script output."""
    import re

    match = re.search(r"Error\s*:\s*([-\d.]+)\s*mHa", output)
    if match:
        return float(match.group(1))
    return None


def _parse_wall_time(output: str) -> float | None:
    """Extract wall time from script output."""
    import re

    match = re.search(r"Wall time\s*:\s*([\d.]+)\s*s", output)
    if match:
        return float(match.group(1))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all SKQD/SQD/VQE pipelines and compare."
    )
    parser.add_argument(
        "molecule",
        nargs="?",
        default="h2",
        help="Molecule identifier (default: h2)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device (default: auto)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only these groups (e.g., --only 001 002 004)",
    )
    parser.add_argument(
        "--skip-quantum",
        action="store_true",
        help="Skip CUDA-Q-dependent pipelines (quantum Krylov and VQE)",
    )
    parser.add_argument(
        "--skip-iterative",
        action="store_true",
        help="Skip iterative NQS pipelines (slower)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    molecule = args.molecule

    # Compute exact energy first
    print(f"{'=' * 80}")
    print(f"  ALL PIPELINES COMPARISON: {molecule.upper()}")
    print(f"  Device: {device}")
    print(f"{'=' * 80}")

    hamiltonian, mol_info = get_molecule(molecule, device=device)
    fci_result = FCISolver().solve(hamiltonian, mol_info)
    exact_energy = fci_result.energy
    if exact_energy is not None:
        print(f"  Exact (FCI) energy: {exact_energy:.10f} Ha")
    else:
        print("  FCI reference unavailable for this system.")
    print(f"{'=' * 80}\n")

    # Filter pipelines
    pipelines_to_run = []
    for group, script, name, desc in PIPELINES:
        group_num = group.split("_")[0]

        if args.only and group_num not in args.only:
            continue
        if args.skip_quantum and ("Krylov-Q" in name or group == "009_vqe"):
            continue
        if args.skip_iterative and group.startswith(
            ("006", "007", "008", "010", "011")
        ):
            continue

        pipelines_to_run.append((group, script, name, desc))

    print(f"Running {len(pipelines_to_run)} of {len(PIPELINES)} pipelines...\n")

    # Run each pipeline
    results = []
    for i, (group, script, name, desc) in enumerate(pipelines_to_run):
        print(f"[{i + 1}/{len(pipelines_to_run)}] {name}: {desc}")
        print(f"  Script: {script}")

        t0 = time.perf_counter()
        result = run_pipeline_subprocess(script, molecule, device)
        elapsed = time.perf_counter() - t0

        status = result["status"]
        energy = result.get("energy")
        error_mha = result.get("error_mha")
        wall_time = result.get("wall_time", elapsed)

        if energy is not None and error_mha is None and exact_energy is not None:
            error_mha = (energy - exact_energy) * 1000.0

        status_icon = {
            "OK": "+",
            "FAILED": "X",
            "MISSING": "?",
            "TIMEOUT": "T",
            "ERROR": "!",
        }
        icon = status_icon.get(status, "?")

        if energy is not None:
            err_s = f"{error_mha:.4f}" if error_mha is not None else "?"
            t_s = f"{wall_time:.1f}" if wall_time is not None else "?"
            print(f"  [{icon}] E={energy:.8f} Ha, err={err_s} mHa, t={t_s}s")
        else:
            print(f"  [{icon}] {status}: {result.get('error', 'unknown')[:100]}")

        results.append(
            {
                "group": group,
                "name": name,
                "description": desc,
                "script": script,
                "status": status,
                "energy": energy,
                "error_mha": error_mha,
                "wall_time": wall_time,
                "exact_energy": exact_energy,
            }
        )
        print()

    # Print summary table
    print(f"\n{'=' * 100}")
    print(f"  COMPARISON TABLE: {molecule.upper()}")
    if exact_energy is not None:
        print(f"  Exact (FCI) energy: {exact_energy:.10f} Ha")
    else:
        print("  Exact (FCI) energy: N/A")
    print(f"  Chemical accuracy threshold: {CHEMICAL_ACCURACY_MHA} mHa")
    print(f"{'=' * 100}")
    print(
        f"  {'#':>2}  {'Pipeline':<30}  {'Status':>7}  {'Energy (Ha)':>16}  {'Error (mHa)':>12}  {'Chem.Acc':>8}  {'Time (s)':>9}"
    )
    print(f"  {'─' * 94}")

    for i, r in enumerate(results):
        status = r["status"]
        energy_str = f"{r['energy']:.10f}" if r["energy"] is not None else "---"
        error_str = f"{r['error_mha']:.4f}" if r["error_mha"] is not None else "---"
        time_str = f"{r['wall_time']:.1f}" if r["wall_time"] is not None else "---"

        chem_acc = "---"
        if r["error_mha"] is not None:
            chem_acc = "YES" if abs(r["error_mha"]) < CHEMICAL_ACCURACY_MHA else "NO"

        print(
            f"  {i + 1:>2}  {r['name']:<30}  {status:>7}  {energy_str:>16}  "
            f"{error_str:>12}  {chem_acc:>8}  {time_str:>9}"
        )

    print(f"{'=' * 100}")

    # Summary statistics
    ok_results = [r for r in results if r["status"] == "OK" and r["energy"] is not None]
    if ok_results:
        best = min(ok_results, key=lambda r: r["energy"])
        fastest = min(ok_results, key=lambda r: r["wall_time"] or float("inf"))
        n_chem_acc = sum(
            1
            for r in ok_results
            if r["error_mha"] is not None
            and abs(r["error_mha"]) < CHEMICAL_ACCURACY_MHA
        )

        print(f"\n  Completed: {len(ok_results)}/{len(results)}")
        print(f"  Chemical accuracy: {n_chem_acc}/{len(ok_results)}")
        if best["error_mha"] is not None:
            print(
                f"  Best energy: {best['name']} ({best['energy']:.10f} Ha, {best['error_mha']:.4f} mHa)"
            )
        else:
            print(
                f"  Best energy: {best['name']} ({best['energy']:.10f} Ha, error: N/A)"
            )
        print(f"  Fastest: {fastest['name']} ({fastest['wall_time']:.1f}s)")

    failed = [r for r in results if r["status"] != "OK"]
    if failed:
        print(f"\n  Failed/Missing: {len(failed)}")
        for r in failed:
            print(f"    [{r['status']}] {r['name']}")

    # Save to JSON
    if args.output:
        output_path = Path(args.output)
        with output_path.open("w") as f:
            json.dump(
                {
                    "molecule": molecule,
                    "exact_energy": exact_energy,
                    "device": device,
                    "results": results,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
