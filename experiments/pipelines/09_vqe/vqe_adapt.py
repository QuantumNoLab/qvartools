"""ADAPT-VQE pipeline using CUDA-QX solvers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config_loader import create_base_parser, load_config

from qvartools._ext.cudaq_vqe import run_cudaq_vqe
from qvartools.molecules import get_molecule_info

CHEMICAL_ACCURACY_MHA = 1.6


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = create_base_parser("ADAPT-VQE with CUDA-QX solvers.")
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        help="Optimizer name (e.g. cobyla).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum optimizer iterations.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="CUDA-Q target (e.g. nvidia, qpp-cpu).",
    )
    parser.add_argument("--verbose", action="store_true", default=None)
    _, config = load_config(parser)

    mol_info = get_molecule_info(config.get("molecule", "h2"))

    result = run_cudaq_vqe(
        geometry=mol_info["geometry"],
        basis=mol_info["basis"],
        charge=mol_info.get("charge", 0),
        spin=mol_info.get("spin", 0),
        method="adapt-vqe",
        optimizer=config.get("optimizer", "cobyla"),
        max_iterations=config.get("max_iterations", 200),
        target=config.get("target", "nvidia"),
        verbose=bool(config.get("verbose", False)),
    )

    error_mha = result.get("error_mha")
    within = (
        "YES"
        if error_mha is not None and abs(error_mha) < CHEMICAL_ACCURACY_MHA
        else "NO"
    )

    print("=" * 60)
    print("ADAPT-VQE RESULTS")
    print("=" * 60)
    print(f"Final energy : {result['energy']:.10f} Ha")
    if result.get("fci_energy") is not None:
        print(f"Exact energy : {result['fci_energy']:.10f} Ha")
    if error_mha is not None:
        print(f"Error        : {error_mha:.4f} mHa")
        print(f"Chemical acc.: {within}")
    print(f"Iterations   : {result['iterations']}")
    print(f"Wall time    : {result['wall_time']:.2f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
