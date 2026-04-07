"""Pipeline 011 hi_nqs_skqd / ibm_on — enable IBM solver + S-CORE recovery.

Wraps :func:`qvartools.methods.nqs.run_hi_nqs_skqd` with the ``[ibm_on]``
section of ``configs/011_hi_nqs_skqd.yaml``.

Usage::

    python experiments/pipelines/011_hi_nqs_skqd/ibm_on.py h2 --device cuda
    python experiments/pipelines/011_hi_nqs_skqd/ibm_on.py h2 \
        --config experiments/pipelines/configs/011_hi_nqs_skqd.yaml
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Make config_loader importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import torch  # noqa: E402
from config_loader import create_base_parser, load_config  # noqa: E402

from qvartools.methods.nqs import HINQSSKQDConfig, run_hi_nqs_skqd  # noqa: E402
from qvartools.molecules import get_molecule  # noqa: E402
from qvartools.solvers import FCISolver  # noqa: E402

VARIANT = "ibm_on"
METHOD = "hi_nqs_skqd"
PIPELINE_ID = "011"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = create_base_parser(
        f"Pipeline {PIPELINE_ID} {METHOD} ({VARIANT}): enable IBM solver + S-CORE recovery."
    )
    args, cfg = load_config(parser)

    # Extract variant section from multi-section YAML; fall back to flat cfg
    section_value = cfg.get(VARIANT)
    section: dict = section_value if isinstance(section_value, dict) else cfg

    # Resolve device: CLI override > section > auto
    device = args.device or section.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    molecule = args.molecule or section.get("molecule", "h2")
    hamiltonian, mol_info = get_molecule(molecule, device=device)
    n_qubits = mol_info["n_qubits"]
    print(f"Molecule : {molecule}")
    print(f"Qubits   : {n_qubits}")
    print(f"Pipeline : {PIPELINE_ID}_{METHOD}/{VARIANT}")
    print(f"Device   : {device}")
    print("=" * 60)

    fci = FCISolver().solve(hamiltonian, mol_info)
    if fci.energy is not None:
        print(f"Exact (FCI) energy: {fci.energy:.10f} Ha")
    print("-" * 60)

    valid_keys = HINQSSKQDConfig.__dataclass_fields__.keys()
    config_kwargs = {k: v for k, v in section.items() if k in valid_keys}
    config_kwargs["device"] = device
    config = HINQSSKQDConfig(**config_kwargs)

    t_start = time.perf_counter()
    result = run_hi_nqs_skqd(hamiltonian, mol_info, config=config)
    wall_time = time.perf_counter() - t_start

    err_mha = (result.energy - fci.energy) * 1000.0 if fci.energy is not None else None
    print("\n" + "=" * 60)
    print(f"PIPELINE {PIPELINE_ID}_{METHOD} ({VARIANT}) RESULTS")
    print("=" * 60)
    print(f"Best energy : {result.energy:.10f} Ha")
    print(f"Final energy: {result.energy:.10f} Ha")
    if err_mha is not None:
        print(f"Error       : {err_mha:.4f} mHa")
    print(f"Wall time   : {wall_time:.2f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
