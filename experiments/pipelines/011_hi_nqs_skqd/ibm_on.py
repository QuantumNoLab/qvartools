"""Pipeline 011 hi_nqs_skqd / ibm_on — enable IBM solver + S-CORE recovery.

Wraps :func:`qvartools.methods.nqs.run_hi_nqs_skqd` via ``METHODS_REGISTRY`` with
the ``[ibm_on]`` section of ``configs/011_hi_nqs_skqd.yaml``.

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
from config_loader import (  # noqa: E402
    create_base_parser,
    get_explicit_cli_args,
    load_config,
)

from qvartools.methods.nqs import METHODS_REGISTRY  # noqa: E402
from qvartools.molecules import get_molecule  # noqa: E402
from qvartools.solvers import FCISolver  # noqa: E402

METHOD_KEY = "hi_nqs_skqd"
VARIANT = "ibm_on"
PIPELINE_ID = "011"

# Reserved top-level keys handled outside the dataclass — filtered out
# before checking for unknown YAML keys so they don't produce false warnings.
_RESERVED_CFG_KEYS = {"molecule", "device", "config", "verbose"}


def _resolve_section(cfg: dict, variant: str) -> dict:
    """Pick the requested variant section from a multi-section YAML.

    Falls back to the flat cfg if the variant section is absent, but emits
    a warning when the YAML looks multi-section (has other dict values) so
    silent misreads don't hide stale configs.
    """
    section_value = cfg.get(variant)
    if isinstance(section_value, dict):
        return section_value
    other_sections = [k for k, v in cfg.items() if isinstance(v, dict) and k != variant]
    if other_sections:
        print(
            f"WARNING: --config YAML has sections {sorted(other_sections)} but "
            f"this script expected section '{variant}'. Falling back to "
            f"flat cfg; most keys may be silently dropped and the method will "
            f"run with dataclass defaults.",
            file=sys.stderr,
        )
    return cfg


def _build_config_kwargs(section: dict, config_cls, device: str) -> dict:
    """Filter a YAML section to the valid dataclass fields, warning on unknown keys."""
    valid_keys = set(config_cls.__dataclass_fields__.keys())
    section_scalar_keys = {k for k, v in section.items() if not isinstance(v, dict)}
    unknown = section_scalar_keys - valid_keys - _RESERVED_CFG_KEYS
    if unknown:
        print(
            f"WARNING: YAML section has keys that are not fields of "
            f"{config_cls.__name__} and will be silently ignored: "
            f"{sorted(unknown)}",
            file=sys.stderr,
        )
    config_kwargs = {k: v for k, v in section.items() if k in valid_keys}
    config_kwargs["device"] = device
    return config_kwargs


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = create_base_parser(
        f"Pipeline {PIPELINE_ID} {METHOD_KEY} ({VARIANT}): enable IBM solver + S-CORE recovery."
    )
    args, cfg = load_config(parser)
    # Detect which CLI args the user actually typed (vs merged defaults).
    # Needed because load_config mutates args with YAML/_DEFAULTS values.
    explicit_cli = get_explicit_cli_args(parser)

    section = _resolve_section(cfg, VARIANT)

    # Device precedence: explicit CLI > section > "auto"
    if "device" in explicit_cli:
        device = args.device
    else:
        device = section.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Molecule precedence: explicit CLI positional > section > "h2"
    if "molecule" in explicit_cli:
        molecule = args.molecule
    else:
        molecule = section.get("molecule", "h2")

    # Dispatch via registry (no hard imports of specific config/runner)
    method_info = METHODS_REGISTRY[METHOD_KEY]
    config_cls = method_info["config_cls"]
    run_fn = method_info["run_fn"]

    hamiltonian, mol_info = get_molecule(molecule, device=device)
    n_qubits = mol_info["n_qubits"]
    print(f"Molecule : {molecule}")
    print(f"Qubits   : {n_qubits}")
    print(f"Pipeline : {PIPELINE_ID}_{METHOD_KEY}/{VARIANT}")
    print(f"Device   : {device}")
    print("=" * 60)

    fci = FCISolver().solve(hamiltonian, mol_info)
    if fci.energy is not None:
        print(f"Exact (FCI) energy: {fci.energy:.10f} Ha")
    print("-" * 60)

    config_kwargs = _build_config_kwargs(section, config_cls, device)
    config = config_cls(**config_kwargs)

    t_start = time.perf_counter()
    result = run_fn(hamiltonian, mol_info, config=config)
    wall_time = time.perf_counter() - t_start

    err_mha = (result.energy - fci.energy) * 1000.0 if fci.energy is not None else None
    print("\n" + "=" * 60)
    print(f"PIPELINE {PIPELINE_ID}_{METHOD_KEY} ({VARIANT}) RESULTS")
    print("=" * 60)
    print(f"Best energy : {result.energy:.10f} Ha")
    print(f"Final energy: {result.energy:.10f} Ha")
    if err_mha is not None:
        print(f"Error       : {err_mha:.4f} mHa")
    print(f"Wall time   : {wall_time:.2f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
