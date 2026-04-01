"""
Disk cache for PySCF-computed molecular integrals, FCI and SCI energies.

Cache directory: ~/.cache/molecular-krylov/
Format:
  - Integrals: <hash>.npz
  - FCI energy: <hash>_fci.json
  - SCI energy: <hash>_sci.json
  - Reference energy: <name>_ref.json  (keyed by molecule name for FCIDUMP systems)

The cache key is a SHA256 hash of (geometry, basis, charge, spin).
CASSCF integrals are NOT cached because orbital optimization is non-deterministic.

Usage::

    from src.utils.hamiltonian_cache import (
        load_integrals, save_integrals,
        load_fci_energy, save_fci_energy,
        load_sci_energy, save_sci_energy,
        load_reference_energy, save_reference_energy,
    )
"""

import hashlib
import json
from pathlib import Path

import numpy as np

CACHE_DIR = Path.home() / ".cache" / "molecular-krylov"


def _get_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def _geometry_hash(geometry, basis: str, charge: int, spin: int) -> str:
    """Compute a deterministic SHA256 hash for a given molecular specification."""
    # Normalise geometry: sort by atom symbol then coordinates for stability
    geo_str = json.dumps(
        [(sym, tuple(round(x, 8) for x in coords)) for sym, coords in geometry],
        sort_keys=True,
    )
    key = f"{geo_str}|{basis.lower()}|{charge}|{spin}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Integrals cache
# ---------------------------------------------------------------------------


def load_integrals(geometry, basis: str, charge: int, spin: int) -> dict | None:
    """Load cached MO integrals. Returns None if not cached.

    Returns a dict with keys:
        h1e, h2e, nuclear_repulsion, n_electrons, n_orbitals, n_alpha, n_beta
    """
    h = _geometry_hash(geometry, basis, charge, spin)
    cache_file = _get_cache_dir() / f"{h}.npz"
    if not cache_file.exists():
        return None
    try:
        data = np.load(str(cache_file), allow_pickle=False)
        return {
            "h1e": data["h1e"],
            "h2e": data["h2e"],
            "nuclear_repulsion": float(data["nuclear_repulsion"]),
            "n_electrons": int(data["n_electrons"]),
            "n_orbitals": int(data["n_orbitals"]),
            "n_alpha": int(data["n_alpha"]),
            "n_beta": int(data["n_beta"]),
        }
    except Exception:
        # Corrupted cache — delete and return None
        cache_file.unlink(missing_ok=True)
        return None


def save_integrals(
    geometry,
    basis: str,
    charge: int,
    spin: int,
    h1e: np.ndarray,
    h2e: np.ndarray,
    nuclear_repulsion: float,
    n_electrons: int,
    n_orbitals: int,
    n_alpha: int,
    n_beta: int,
) -> str:
    """Save MO integrals to disk cache. Returns the cache hash (for logging)."""
    h = _geometry_hash(geometry, basis, charge, spin)
    cache_file = _get_cache_dir() / f"{h}.npz"
    np.savez_compressed(
        str(cache_file),
        h1e=h1e.astype(np.float64),
        h2e=h2e.astype(np.float64),
        nuclear_repulsion=np.float64(nuclear_repulsion),
        n_electrons=np.int64(n_electrons),
        n_orbitals=np.int64(n_orbitals),
        n_alpha=np.int64(n_alpha),
        n_beta=np.int64(n_beta),
    )
    return h


# ---------------------------------------------------------------------------
# FCI energy cache
# ---------------------------------------------------------------------------


def load_fci_energy(geometry, basis: str, charge: int, spin: int) -> float | None:
    """Load cached FCI energy. Returns None if not cached."""
    h = _geometry_hash(geometry, basis, charge, spin)
    cache_file = _get_cache_dir() / f"{h}_fci.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file) as f:
            data = json.load(f)
        return float(data["fci_energy"])
    except Exception:
        cache_file.unlink(missing_ok=True)
        return None


def save_fci_energy(
    geometry, basis: str, charge: int, spin: int, energy: float
) -> None:
    """Save FCI energy to disk cache."""
    h = _geometry_hash(geometry, basis, charge, spin)
    cache_file = _get_cache_dir() / f"{h}_fci.json"
    with open(cache_file, "w") as f:
        json.dump({"fci_energy": energy, "hash": h}, f)


# ---------------------------------------------------------------------------
# SCI energy cache
# ---------------------------------------------------------------------------


def load_sci_energy(geometry, basis: str, charge: int, spin: int) -> dict | None:
    """Load cached SCI energy and metadata. Returns None if not cached.

    Returns a dict with keys: sci_energy, basis_size, wall_time
    """
    h = _geometry_hash(geometry, basis, charge, spin)
    cache_file = _get_cache_dir() / f"{h}_sci.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file) as f:
            return json.load(f)
    except Exception:
        cache_file.unlink(missing_ok=True)
        return None


def save_sci_energy(
    geometry,
    basis: str,
    charge: int,
    spin: int,
    energy: float,
    basis_size: int = 0,
    wall_time: float = 0.0,
) -> None:
    """Save SCI energy to disk cache."""
    h = _geometry_hash(geometry, basis, charge, spin)
    cache_file = _get_cache_dir() / f"{h}_sci.json"
    with open(cache_file, "w") as f:
        json.dump(
            {
                "sci_energy": energy,
                "basis_size": basis_size,
                "wall_time": wall_time,
                "hash": h,
            },
            f,
        )


# ---------------------------------------------------------------------------
# Generic reference energy cache (for FCIDUMP systems without geometry)
# ---------------------------------------------------------------------------


def load_reference_energy(name: str, method: str = "sci") -> dict | None:
    """Load cached reference energy by molecule name (e.g. '2Fe2S', 'C2H2').

    Useful for FCIDUMP-loaded systems that don't have geometry metadata.
    Returns a dict with keys: energy, method, basis_size, wall_time, or None.
    """
    cache_file = _get_cache_dir() / f"{name}_{method}_ref.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file) as f:
            return json.load(f)
    except Exception:
        cache_file.unlink(missing_ok=True)
        return None


def save_reference_energy(
    name: str,
    method: str,
    energy: float,
    basis_size: int = 0,
    wall_time: float = 0.0,
) -> None:
    """Save reference energy by molecule name."""
    cache_file = _get_cache_dir() / f"{name}_{method}_ref.json"
    with open(cache_file, "w") as f:
        json.dump(
            {
                "energy": energy,
                "method": method,
                "basis_size": basis_size,
                "wall_time": wall_time,
                "name": name,
            },
            f,
        )
