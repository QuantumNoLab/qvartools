"""
registry --- Molecule registry and factory functions
====================================================

Defines a registry of standard molecular benchmarks used in quantum
chemistry.  Each entry provides a factory function that computes
molecular integrals and constructs a :class:`MolecularHamiltonian`.

The registry covers a range of system sizes from H2 (4 qubits) to
C2H4 (28 qubits), enabling systematic benchmarking of SQD/SKQD methods.

Constants
---------
MOLECULE_REGISTRY
    Dictionary mapping lowercase molecule names to factory metadata.

Functions
---------
get_molecule
    Instantiate a Hamiltonian for a named molecule.
list_molecules
    Return sorted list of registered molecule names.
"""

from __future__ import annotations

import logging
from typing import Any

from qvartools.hamiltonians.integrals import _FCI_CONFIG_LIMIT
from qvartools.hamiltonians.molecular import (
    MolecularHamiltonian,
    compute_molecular_integrals,
)

__all__ = [
    "MOLECULE_REGISTRY",
    "get_molecule",
    "get_molecule_info",
    "list_molecules",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry definitions
# ---------------------------------------------------------------------------

_H2_GEOMETRY: list[tuple[str, tuple[float, float, float]]] = [
    ("H", (0.0, 0.0, 0.0)),
    ("H", (0.0, 0.0, 0.74)),
]

_LIH_GEOMETRY: list[tuple[str, tuple[float, float, float]]] = [
    ("Li", (0.0, 0.0, 0.0)),
    ("H", (0.0, 0.0, 1.6)),
]

_BEH2_GEOMETRY: list[tuple[str, tuple[float, float, float]]] = [
    ("Be", (0.0, 0.0, 0.0)),
    ("H", (0.0, 0.0, 1.33)),
    ("H", (0.0, 0.0, -1.33)),
]

# H2O geometry from parametric (OH=0.96 Å, angle=104.5°)
import math as _math

_H2O_GEOMETRY: list[tuple[str, tuple[float, float, float]]] = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.96, 0.0, 0.0)),
    (
        "H",
        (
            0.96 * _math.cos(_math.radians(104.5)),
            0.96 * _math.sin(_math.radians(104.5)),
            0.0,
        ),
    ),
]

_NH3_GEOMETRY: list[tuple[str, tuple[float, float, float]]] = [
    ("N", (0.0, 0.0, 0.0)),
    ("H", (0.0, -0.9377, -0.3816)),
    ("H", (0.8121, 0.4689, -0.3816)),
    ("H", (-0.8121, 0.4689, -0.3816)),
]

_N2_GEOMETRY: list[tuple[str, tuple[float, float, float]]] = [
    ("N", (0.0, 0.0, 0.0)),
    ("N", (0.0, 0.0, 1.0977)),
]

_CH4_GEOMETRY: list[tuple[str, tuple[float, float, float]]] = [
    ("C", (0.0, 0.0, 0.0)),
    ("H", (0.6276, 0.6276, 0.6276)),
    ("H", (0.6276, -0.6276, -0.6276)),
    ("H", (-0.6276, 0.6276, -0.6276)),
    ("H", (-0.6276, -0.6276, 0.6276)),
]

_C2H4_GEOMETRY: list[tuple[str, tuple[float, float, float]]] = [
    ("C", (0.0, 0.0, 0.6695)),
    ("C", (0.0, 0.0, -0.6695)),
    ("H", (0.0, 0.9289, 1.2321)),
    ("H", (0.0, -0.9289, 1.2321)),
    ("H", (0.0, 0.9289, -1.2321)),
    ("H", (0.0, -0.9289, -1.2321)),
]

_CO_GEOMETRY: list[tuple[str, tuple[float, float, float]]] = [
    ("C", (0.0, 0.0, 0.0)),
    ("O", (0.0, 0.0, 1.13)),
]

_HCN_GEOMETRY: list[tuple[str, tuple[float, float, float]]] = [
    ("H", (0.0, 0.0, 0.0)),
    ("C", (0.0, 0.0, 1.06)),
    ("N", (0.0, 0.0, 2.22)),
]

_C2H2_GEOMETRY: list[tuple[str, tuple[float, float, float]]] = [
    ("H", (0.0, 0.0, 0.0)),
    ("C", (0.0, 0.0, 1.06)),
    ("C", (0.0, 0.0, 2.26)),
    ("H", (0.0, 0.0, 3.32)),
]

_H2S_GEOMETRY: list[tuple[str, tuple[float, float, float]]] = [
    ("S", (0.0, 0.0, 0.0)),
    ("H", (1.34, 0.0, 0.0)),
    ("H", (-0.0497, 1.3391, 0.0)),
]


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def _make_h2(device: str = "cpu") -> tuple[MolecularHamiltonian, dict[str, Any]]:
    """Create H2 Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_H2_GEOMETRY, basis="sto-3g", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("H2", 4, "sto-3g", _H2_GEOMETRY, 0, 0)
    return ham, info


def _make_lih(device: str = "cpu") -> tuple[MolecularHamiltonian, dict[str, Any]]:
    """Create LiH Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_LIH_GEOMETRY, basis="sto-3g", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("LiH", 12, "sto-3g", _LIH_GEOMETRY, 0, 0)
    return ham, info


def _make_beh2(device: str = "cpu") -> tuple[MolecularHamiltonian, dict[str, Any]]:
    """Create BeH2 Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_BEH2_GEOMETRY, basis="sto-3g", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("BeH2", 14, "sto-3g", _BEH2_GEOMETRY, 0, 0)
    return ham, info


def _make_h2o(device: str = "cpu") -> tuple[MolecularHamiltonian, dict[str, Any]]:
    """Create H2O Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_H2O_GEOMETRY, basis="sto-3g", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("H2O", 14, "sto-3g", _H2O_GEOMETRY, 0, 0)
    return ham, info


def _make_nh3(device: str = "cpu") -> tuple[MolecularHamiltonian, dict[str, Any]]:
    """Create NH3 Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_NH3_GEOMETRY, basis="sto-3g", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("NH3", 16, "sto-3g", _NH3_GEOMETRY, 0, 0)
    return ham, info


def _make_n2(device: str = "cpu") -> tuple[MolecularHamiltonian, dict[str, Any]]:
    """Create N2 Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_N2_GEOMETRY, basis="cc-pvdz", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("N2", 20, "cc-pvdz", _N2_GEOMETRY, 0, 0)
    return ham, info


def _make_ch4(device: str = "cpu") -> tuple[MolecularHamiltonian, dict[str, Any]]:
    """Create CH4 Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_CH4_GEOMETRY, basis="sto-3g", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("CH4", 18, "sto-3g", _CH4_GEOMETRY, 0, 0)
    return ham, info


def _make_c2h4(device: str = "cpu") -> tuple[MolecularHamiltonian, dict[str, Any]]:
    """Create C2H4 (ethylene) Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_C2H4_GEOMETRY, basis="sto-3g", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("C2H4", 28, "sto-3g", _C2H4_GEOMETRY, 0, 0)
    return ham, info


def _make_co(device: str = "cpu") -> tuple[MolecularHamiltonian, dict[str, Any]]:
    """Create CO (carbon monoxide) Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_CO_GEOMETRY, basis="sto-3g", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("CO", 20, "sto-3g", _CO_GEOMETRY, 0, 0)
    return ham, info


def _make_hcn(device: str = "cpu") -> tuple[MolecularHamiltonian, dict[str, Any]]:
    """Create HCN (hydrogen cyanide) Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_HCN_GEOMETRY, basis="sto-3g", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("HCN", 22, "sto-3g", _HCN_GEOMETRY, 0, 0)
    return ham, info


def _make_c2h2(device: str = "cpu") -> tuple[MolecularHamiltonian, dict[str, Any]]:
    """Create C2H2 (acetylene) Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_C2H2_GEOMETRY, basis="sto-3g", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("C2H2", 24, "sto-3g", _C2H2_GEOMETRY, 0, 0)
    return ham, info


def _make_h2s(device: str = "cpu") -> tuple[MolecularHamiltonian, dict[str, Any]]:
    """Create H2S (hydrogen sulfide) Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_H2S_GEOMETRY, basis="sto-3g", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("H2S", 26, "sto-3g", _H2S_GEOMETRY, 0, 0)
    return ham, info


# ---------------------------------------------------------------------------
# CAS molecule geometries
# ---------------------------------------------------------------------------

# N₂ CAS uses 1.10 Å bond length (standard for CAS benchmarks in Flow-Guided-Krylov),
# distinct from the full-space N₂ which uses 1.0977 Å (NIST equilibrium).
_N2_CAS_GEOMETRY: list[tuple[str, tuple[float, float, float]]] = [
    ("N", (0.0, 0.0, 0.0)),
    ("N", (0.0, 0.0, 1.10)),
]

_CR2_GEOMETRY: list[tuple[str, tuple[float, float, float]]] = [
    ("Cr", (0.0, 0.0, 0.0)),
    ("Cr", (0.0, 0.0, 1.68)),
]


def _benzene_geometry() -> list[tuple[str, tuple[float, float, float]]]:
    """Regular hexagon benzene geometry (C-C = 1.40 A, C-H = 1.08 A)."""
    import math

    cc, ch = 1.40, 1.08
    geom: list[tuple[str, tuple[float, float, float]]] = []
    for i in range(6):
        angle = math.pi / 3 * i
        geom.append(("C", (cc * math.cos(angle), cc * math.sin(angle), 0.0)))
        geom.append(
            ("H", ((cc + ch) * math.cos(angle), (cc + ch) * math.sin(angle), 0.0))
        )
    return geom


# ---------------------------------------------------------------------------
# CAS molecule factory functions
# ---------------------------------------------------------------------------


def _make_n2_cas(
    nelecas: int, ncas: int, basis: str, device: str = "cpu"
) -> tuple[MolecularHamiltonian, dict[str, Any]]:
    """Create N₂ CAS Hamiltonian with specified active space.

    Parameters
    ----------
    nelecas : int
        Number of active electrons.
    ncas : int
        Number of active orbitals.
    basis : str
        Gaussian basis set name (e.g. ``"cc-pvdz"``).
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)`` where ``info_dict["is_cas"]`` is ``True``.

    Raises
    ------
    ImportError
        If PySCF is not installed.
    """
    use_casci = ncas >= 15
    integrals = compute_molecular_integrals(
        _N2_CAS_GEOMETRY, basis=basis, cas=(nelecas, ncas), casci=use_casci
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info(
        f"N2-CAS({nelecas},{ncas})", 2 * ncas, basis, _N2_CAS_GEOMETRY, 0, 0
    )
    info["is_cas"] = True
    return ham, info


def _make_cr2(
    basis: str = "sto-3g",
    cas: tuple[int, int] = (12, 12),
    device: str = "cpu",
) -> tuple[MolecularHamiltonian, dict[str, Any]]:
    """Create Cr₂ CAS Hamiltonian with fix_spin_(ss=0) for singlet.

    Without fix_spin_, CASSCF converges to the septet (S=3) instead of the
    singlet ground state.  Uses ROHF fallback if RHF doesn't converge.
    Auto-CASCI for large active spaces (ncas >= 15).

    Parameters
    ----------
    basis : str, optional
        Gaussian basis set name (default ``"sto-3g"``).
    cas : tuple of (int, int), optional
        ``(nelecas, ncas)`` active-space specification (default ``(12, 12)``).
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)`` where ``info_dict["is_cas"]`` is ``True``.

    Raises
    ------
    ImportError
        If PySCF is not installed.
    """
    import warnings
    from math import comb as _comb

    try:
        from pyscf import ao2mo, fci, gto, mcscf, scf
    except ImportError as exc:
        raise ImportError("PySCF is required for Cr₂. pip install pyscf") from exc

    nelecas, ncas = cas
    mol = gto.M(
        atom="Cr 0 0 0; Cr 0 0 1.68", basis=basis, spin=0, symmetry=True, verbose=0
    )
    mf = scf.RHF(mol)
    mf.max_cycle = 300
    mf.kernel()
    if not mf.converged:
        warnings.warn("RHF did not converge for Cr₂, trying ROHF.", stacklevel=2)
        mf = scf.ROHF(mol)
        mf.max_cycle = 300
        mf.kernel()
        if not mf.converged:
            warnings.warn(
                "ROHF also did not converge for Cr₂. Results may be unreliable.",
                stacklevel=2,
            )

    # Estimate config count for auto-CASCI
    n_alpha_cas = nelecas // 2
    n_beta_cas = nelecas // 2
    n_configs = _comb(ncas, n_alpha_cas) * _comb(ncas, n_beta_cas)
    use_casci = ncas >= 15 or n_configs > _FCI_CONFIG_LIMIT

    if use_casci:
        mc = mcscf.CASCI(mf, ncas=ncas, nelecas=nelecas)
        mc.fix_spin_(ss=0)  # Enforce singlet (needed for CASCI too)
    else:
        mc = mcscf.CASSCF(mf, ncas=ncas, nelecas=nelecas)
        mc.fix_spin_(ss=0)  # Enforce singlet

    # Linear molecules need non-symmetry FCI solver
    if mol.symmetry and mol.topgroup in ("Dooh", "Coov"):
        mc.fcisolver = fci.direct_spin1.FCISolver(mol)

    if use_casci and n_configs > _FCI_CONFIG_LIMIT:
        logger.info(
            "Skipping FCI for Cr₂ CAS(%d,%d): %s configs.",
            nelecas,
            ncas,
            f"{n_configs:,}",
        )
    else:
        mc.kernel()
        if not use_casci and not mc.converged:
            warnings.warn(
                f"CASSCF did not converge for Cr₂ CAS({nelecas},{ncas}).", stacklevel=2
            )

    h1e_cas, e_core = mc.h1e_for_cas()
    active_mo = mc.mo_coeff[:, mc.ncore : mc.ncore + mc.ncas]
    h2e_cas = ao2mo.full(mol, active_mo)
    h2e_cas = ao2mo.restore(1, h2e_cas, ncas)

    import numpy as np

    from qvartools.hamiltonians.integrals import MolecularIntegrals

    integrals = MolecularIntegrals(
        h1e=np.asarray(h1e_cas, dtype=np.float64),
        h2e=np.asarray(h2e_cas, dtype=np.float64),
        nuclear_repulsion=float(e_core),
        n_electrons=nelecas,
        n_orbitals=ncas,
        n_alpha=n_alpha_cas,
        n_beta=n_beta_cas,
    )
    ham = MolecularHamiltonian(integrals, device=device)
    name = (
        "Cr2" if cas == (12, 12) and basis == "sto-3g" else f"Cr2-CAS({nelecas},{ncas})"
    )
    info = _build_info(name, 2 * ncas, basis, list(_CR2_GEOMETRY), 0, 0)
    info["is_cas"] = True
    return ham, info


def _make_benzene(device: str = "cpu") -> tuple[MolecularHamiltonian, dict[str, Any]]:
    """Create Benzene CAS(6,15) Hamiltonian.

    Uses CASCI (no orbital optimisation) because ncas >= 15.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)`` where ``info_dict["is_cas"]`` is ``True``.
    """
    geom = _benzene_geometry()
    integrals = compute_molecular_integrals(
        geom, basis="sto-3g", cas=(6, 15), casci=True
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("Benzene", 30, "sto-3g", geom, 0, 0)
    info["is_cas"] = True
    return ham, info


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_info(
    name: str,
    n_qubits: int,
    basis: str,
    geometry: list[tuple[str, tuple[float, float, float]]],
    charge: int,
    spin: int,
) -> dict[str, Any]:
    """Build a standardised molecule info dictionary.

    Parameters
    ----------
    name : str
        Molecule name.
    n_qubits : int
        Number of qubits (spin-orbitals).
    basis : str
        Gaussian basis set.
    geometry : list
        Atomic geometry.
    charge : int
        Net molecular charge.
    spin : int
        Spin multiplicity minus one (2S).

    Returns
    -------
    dict
        Info dictionary with keys ``name``, ``n_qubits``, ``basis``,
        ``geometry``, ``charge``, ``spin``.
    """
    return {
        "name": name,
        "n_qubits": n_qubits,
        "basis": basis,
        "geometry": geometry,
        "charge": charge,
        "spin": spin,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MOLECULE_REGISTRY: dict[str, dict[str, Any]] = {
    "h2": {
        "factory": _make_h2,
        "n_qubits": 4,
        "description": "Hydrogen molecule (minimal basis)",
        "basis": "sto-3g",
    },
    "lih": {
        "factory": _make_lih,
        "n_qubits": 12,
        "description": "Lithium hydride",
        "basis": "sto-3g",
    },
    "beh2": {
        "factory": _make_beh2,
        "n_qubits": 14,
        "description": "Beryllium dihydride",
        "basis": "sto-3g",
    },
    "h2o": {
        "factory": _make_h2o,
        "n_qubits": 14,
        "description": "Water molecule",
        "basis": "sto-3g",
    },
    "nh3": {
        "factory": _make_nh3,
        "n_qubits": 16,
        "description": "Ammonia",
        "basis": "sto-3g",
    },
    "n2": {
        "factory": _make_n2,
        "n_qubits": 20,
        "description": "Nitrogen molecule (cc-pVDZ basis)",
        "basis": "cc-pvdz",
    },
    "ch4": {
        "factory": _make_ch4,
        "n_qubits": 18,
        "description": "Methane",
        "basis": "sto-3g",
    },
    "c2h4": {
        "factory": _make_c2h4,
        "n_qubits": 28,
        "description": "Ethylene (minimal basis)",
        "basis": "sto-3g",
    },
    "co": {
        "factory": _make_co,
        "n_qubits": 20,
        "description": "Carbon monoxide (STO-3G)",
        "basis": "sto-3g",
    },
    "hcn": {
        "factory": _make_hcn,
        "n_qubits": 22,
        "description": "Hydrogen cyanide (STO-3G)",
        "basis": "sto-3g",
    },
    "c2h2": {
        "factory": _make_c2h2,
        "n_qubits": 24,
        "description": "Acetylene (STO-3G)",
        "basis": "sto-3g",
    },
    "h2s": {
        "factory": _make_h2s,
        "n_qubits": 26,
        "description": "Hydrogen sulfide (STO-3G)",
        "basis": "sto-3g",
    },
    # --- CAS active-space systems (24-58 qubits) ---
    "n2-cas(10,12)": {
        "factory": lambda device="cpu": _make_n2_cas(10, 12, "cc-pvdz", device),
        "n_qubits": 24,
        "description": "Nitrogen CAS(10,12) cc-pVDZ",
        "basis": "cc-pvdz",
        "is_cas": True,
    },
    "cr2": {
        "factory": lambda device="cpu": _make_cr2("sto-3g", (12, 12), device),
        "n_qubits": 24,
        "description": "Chromium dimer CAS(12,12) STO-3G",
        "basis": "sto-3g",
        "is_cas": True,
    },
    "n2-cas(10,15)": {
        "factory": lambda device="cpu": _make_n2_cas(10, 15, "cc-pvdz", device),
        "n_qubits": 30,
        "description": "Nitrogen CAS(10,15) cc-pVDZ",
        "basis": "cc-pvdz",
        "is_cas": True,
    },
    "benzene": {
        "factory": _make_benzene,
        "n_qubits": 30,
        "description": "Benzene CAS(6,15) STO-3G",
        "basis": "sto-3g",
        "is_cas": True,
    },
    "n2-cas(10,17)": {
        "factory": lambda device="cpu": _make_n2_cas(10, 17, "cc-pvdz", device),
        "n_qubits": 34,
        "description": "Nitrogen CAS(10,17) cc-pVDZ",
        "basis": "cc-pvdz",
        "is_cas": True,
    },
    "cr2-cas(12,18)": {
        "factory": lambda device="cpu": _make_cr2("cc-pvdz", (12, 18), device),
        "n_qubits": 36,
        "description": "Chromium dimer CAS(12,18) cc-pVDZ",
        "basis": "cc-pvdz",
        "is_cas": True,
    },
    "n2-cas(10,20)": {
        "factory": lambda device="cpu": _make_n2_cas(10, 20, "cc-pvtz", device),
        "n_qubits": 40,
        "description": "Nitrogen CAS(10,20) cc-pVTZ",
        "basis": "cc-pvtz",
        "is_cas": True,
    },
    "cr2-cas(12,20)": {
        "factory": lambda device="cpu": _make_cr2("cc-pvdz", (12, 20), device),
        "n_qubits": 40,
        "description": "Chromium dimer CAS(12,20) cc-pVDZ",
        "basis": "cc-pvdz",
        "is_cas": True,
    },
    "n2-cas(10,26)": {
        "factory": lambda device="cpu": _make_n2_cas(10, 26, "cc-pvtz", device),
        "n_qubits": 52,
        "description": "Nitrogen CAS(10,26) cc-pVTZ",
        "basis": "cc-pvtz",
        "is_cas": True,
    },
    "cr2-cas(12,26)": {
        "factory": lambda device="cpu": _make_cr2("cc-pvdz", (12, 26), device),
        "n_qubits": 52,
        "description": "Chromium dimer CAS(12,26) cc-pVDZ",
        "basis": "cc-pvdz",
        "is_cas": True,
    },
    "cr2-cas(12,28)": {
        "factory": lambda device="cpu": _make_cr2("cc-pvdz", (12, 28), device),
        "n_qubits": 56,
        "description": "Chromium dimer CAS(12,28) cc-pVDZ",
        "basis": "cc-pvdz",
        "is_cas": True,
    },
    "cr2-cas(12,29)": {
        "factory": lambda device="cpu": _make_cr2("cc-pvdz", (12, 29), device),
        "n_qubits": 58,
        "description": "Chromium dimer CAS(12,29) cc-pVDZ",
        "basis": "cc-pvdz",
        "is_cas": True,
    },
    # --- 64+ qubit systems ---
    "cr2-cas(12,32)": {
        "factory": lambda device="cpu": _make_cr2("cc-pvdz", (12, 32), device),
        "n_qubits": 64,
        "description": "Chromium dimer CAS(12,32) cc-pVDZ",
        "basis": "cc-pvdz",
        "is_cas": True,
    },
    "cr2-cas(12,36)": {
        "factory": lambda device="cpu": _make_cr2("cc-pvdz", (12, 36), device),
        "n_qubits": 72,
        "description": "Chromium dimer CAS(12,36) cc-pVDZ",
        "basis": "cc-pvdz",
        "is_cas": True,
    },
}

_MOLECULE_INFO_REGISTRY: dict[str, dict[str, Any]] = {
    "h2": _build_info("H2", 4, "sto-3g", _H2_GEOMETRY, 0, 0),
    "lih": _build_info("LiH", 12, "sto-3g", _LIH_GEOMETRY, 0, 0),
    "beh2": _build_info("BeH2", 14, "sto-3g", _BEH2_GEOMETRY, 0, 0),
    "h2o": _build_info("H2O", 14, "sto-3g", _H2O_GEOMETRY, 0, 0),
    "nh3": _build_info("NH3", 16, "sto-3g", _NH3_GEOMETRY, 0, 0),
    "n2": _build_info("N2", 20, "cc-pvdz", _N2_GEOMETRY, 0, 0),
    "ch4": _build_info("CH4", 18, "sto-3g", _CH4_GEOMETRY, 0, 0),
    "c2h4": _build_info("C2H4", 28, "sto-3g", _C2H4_GEOMETRY, 0, 0),
    "co": _build_info("CO", 20, "sto-3g", _CO_GEOMETRY, 0, 0),
    "hcn": _build_info("HCN", 22, "sto-3g", _HCN_GEOMETRY, 0, 0),
    "c2h2": _build_info("C2H2", 24, "sto-3g", _C2H2_GEOMETRY, 0, 0),
    "h2s": _build_info("H2S", 26, "sto-3g", _H2S_GEOMETRY, 0, 0),
    # --- CAS active-space systems ---
    "n2-cas(10,12)": {
        **_build_info("N2-CAS(10,12)", 24, "cc-pvdz", list(_N2_CAS_GEOMETRY), 0, 0),
        "is_cas": True,
    },
    "cr2": {
        **_build_info("Cr2", 24, "sto-3g", list(_CR2_GEOMETRY), 0, 0),
        "is_cas": True,
    },
    "n2-cas(10,15)": {
        **_build_info("N2-CAS(10,15)", 30, "cc-pvdz", list(_N2_CAS_GEOMETRY), 0, 0),
        "is_cas": True,
    },
    "benzene": {
        **_build_info("Benzene", 30, "sto-3g", _benzene_geometry(), 0, 0),
        "is_cas": True,
    },
    "n2-cas(10,17)": {
        **_build_info("N2-CAS(10,17)", 34, "cc-pvdz", list(_N2_CAS_GEOMETRY), 0, 0),
        "is_cas": True,
    },
    "cr2-cas(12,18)": {
        **_build_info("Cr2-CAS(12,18)", 36, "cc-pvdz", list(_CR2_GEOMETRY), 0, 0),
        "is_cas": True,
    },
    "n2-cas(10,20)": {
        **_build_info("N2-CAS(10,20)", 40, "cc-pvtz", list(_N2_CAS_GEOMETRY), 0, 0),
        "is_cas": True,
    },
    "cr2-cas(12,20)": {
        **_build_info("Cr2-CAS(12,20)", 40, "cc-pvdz", list(_CR2_GEOMETRY), 0, 0),
        "is_cas": True,
    },
    "n2-cas(10,26)": {
        **_build_info("N2-CAS(10,26)", 52, "cc-pvtz", list(_N2_CAS_GEOMETRY), 0, 0),
        "is_cas": True,
    },
    "cr2-cas(12,26)": {
        **_build_info("Cr2-CAS(12,26)", 52, "cc-pvdz", list(_CR2_GEOMETRY), 0, 0),
        "is_cas": True,
    },
    "cr2-cas(12,28)": {
        **_build_info("Cr2-CAS(12,28)", 56, "cc-pvdz", list(_CR2_GEOMETRY), 0, 0),
        "is_cas": True,
    },
    "cr2-cas(12,29)": {
        **_build_info("Cr2-CAS(12,29)", 58, "cc-pvdz", list(_CR2_GEOMETRY), 0, 0),
        "is_cas": True,
    },
    # --- 64+ qubit systems ---
    "cr2-cas(12,32)": {
        **_build_info("Cr2-CAS(12,32)", 64, "cc-pvdz", list(_CR2_GEOMETRY), 0, 0),
        "is_cas": True,
    },
    "cr2-cas(12,36)": {
        **_build_info("Cr2-CAS(12,36)", 72, "cc-pvdz", list(_CR2_GEOMETRY), 0, 0),
        "is_cas": True,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_molecule(
    name: str, device: str = "cpu"
) -> tuple[MolecularHamiltonian, dict[str, Any]]:
    """Create a Hamiltonian and info dict for a named molecule.

    Looks up the molecule in :data:`MOLECULE_REGISTRY`, runs the PySCF
    integral computation, and constructs a :class:`MolecularHamiltonian`.

    Parameters
    ----------
    name : str
        Molecule name (case-insensitive).  Must be a key in
        :data:`MOLECULE_REGISTRY`.
    device : str, optional
        Torch device for the Hamiltonian (default ``"cpu"``).

    Returns
    -------
    hamiltonian : MolecularHamiltonian
        The molecular Hamiltonian ready for diagonalisation.
    info : dict
        Metadata dictionary with keys ``name``, ``n_qubits``, ``basis``,
        ``geometry``, ``charge``, ``spin``.

    Raises
    ------
    KeyError
        If ``name`` is not found in the registry.

    Examples
    --------
    >>> ham, info = get_molecule("H2")
    >>> info["n_qubits"]
    4
    >>> ham.num_sites
    4
    """
    key = name.lower().strip()
    if key not in MOLECULE_REGISTRY:
        available = ", ".join(sorted(MOLECULE_REGISTRY.keys()))
        raise KeyError(f"Unknown molecule {name!r}. Available: {available}")

    entry = MOLECULE_REGISTRY[key]
    factory = entry["factory"]

    logger.info(
        "Creating molecule %r (%d qubits, %s basis)",
        key,
        entry["n_qubits"],
        entry["basis"],
    )

    return factory(device=device)


def get_molecule_info(name: str) -> dict[str, Any]:
    """Return molecule metadata without constructing a Hamiltonian.

    Parameters
    ----------
    name : str
        Molecule name (case-insensitive). Must be a key in
        :data:`MOLECULE_REGISTRY`.

    Returns
    -------
    dict
        Metadata dictionary with keys ``name``, ``n_qubits``, ``basis``,
        ``geometry``, ``charge``, ``spin``.

    Raises
    ------
    KeyError
        If ``name`` is not found in the registry.
    """
    key = name.lower().strip()
    if key not in MOLECULE_REGISTRY:
        available = ", ".join(sorted(MOLECULE_REGISTRY.keys()))
        raise KeyError(f"Unknown molecule {name!r}. Available: {available}")

    if key not in _MOLECULE_INFO_REGISTRY:
        raise KeyError(
            f"Molecule {name!r} is in MOLECULE_REGISTRY but missing from "
            f"_MOLECULE_INFO_REGISTRY. Both registries must be kept in sync."
        )
    # Return a copy to prevent callers from mutating global state
    info = dict(_MOLECULE_INFO_REGISTRY[key])
    if "geometry" in info:
        info["geometry"] = list(info["geometry"])
    return info


# Validate registry consistency at import time (survives python -O)
if set(MOLECULE_REGISTRY.keys()) != set(_MOLECULE_INFO_REGISTRY.keys()):
    raise RuntimeError(
        "MOLECULE_REGISTRY and _MOLECULE_INFO_REGISTRY are out of sync: "
        f"missing in info: {set(MOLECULE_REGISTRY) - set(_MOLECULE_INFO_REGISTRY)}, "
        f"extra in info: {set(_MOLECULE_INFO_REGISTRY) - set(MOLECULE_REGISTRY)}"
    )


def list_molecules() -> list[str]:
    """Return a sorted list of available molecule names.

    Returns
    -------
    list of str
        Registered molecule names in alphabetical order.

    Examples
    --------
    >>> names = list_molecules()
    >>> "h2" in names
    True
    """
    return sorted(MOLECULE_REGISTRY.keys())
