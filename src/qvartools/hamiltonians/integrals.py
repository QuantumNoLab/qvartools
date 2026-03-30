"""
integrals — Molecular integral container and PySCF computation
==============================================================

Provides the ``MolecularIntegrals`` frozen dataclass that holds one- and
two-electron integrals together with molecule metadata, and the
``compute_molecular_integrals`` helper that runs RHF via PySCF and
returns a populated ``MolecularIntegrals`` instance.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "MATRIX_ELEMENT_TOL",
    "MolecularIntegrals",
    "cached_compute_molecular_integrals",
    "clear_integral_cache",
    "compute_molecular_integrals",
    "get_integral_cache",
]

MATRIX_ELEMENT_TOL: float = 1e-12
"""float : Absolute tolerance below which matrix elements are treated as zero."""


# ---------------------------------------------------------------------------
# MolecularIntegrals dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MolecularIntegrals:
    """Container for molecular one- and two-electron integrals.

    All arrays use the *spatial-orbital* indexing convention produced by
    PySCF's ``ao2mo`` module.

    Parameters
    ----------
    h1e : np.ndarray
        One-electron integrals, shape ``(n_orb, n_orb)``, dtype ``float64``.
    h2e : np.ndarray
        Two-electron integrals in chemist's notation ``(pq|rs)``,
        shape ``(n_orb, n_orb, n_orb, n_orb)``, dtype ``float64``.
    nuclear_repulsion : float
        Nuclear repulsion energy in Hartree.
    n_electrons : int
        Total number of electrons.
    n_orbitals : int
        Number of spatial orbitals.
    n_alpha : int
        Number of alpha (spin-up) electrons.
    n_beta : int
        Number of beta (spin-down) electrons.

    Raises
    ------
    ValueError
        If array shapes are inconsistent with ``n_orbitals`` or if
        ``n_alpha + n_beta != n_electrons``.

    Examples
    --------
    >>> import numpy as np
    >>> h1 = np.zeros((2, 2))
    >>> h2 = np.zeros((2, 2, 2, 2))
    >>> mi = MolecularIntegrals(h1, h2, 0.7, 2, 2, 1, 1)
    >>> mi.n_orbitals
    2
    """

    h1e: np.ndarray
    h2e: np.ndarray
    nuclear_repulsion: float
    n_electrons: int
    n_orbitals: int
    n_alpha: int
    n_beta: int

    def __post_init__(self) -> None:
        """Validate shapes and dtypes."""
        n = self.n_orbitals
        if self.h1e.shape != (n, n):
            raise ValueError(
                f"h1e shape {self.h1e.shape} does not match n_orbitals={n}"
            )
        if self.h2e.shape != (n, n, n, n):
            raise ValueError(
                f"h2e shape {self.h2e.shape} does not match n_orbitals={n}"
            )
        if self.n_alpha + self.n_beta != self.n_electrons:
            raise ValueError(
                f"n_alpha ({self.n_alpha}) + n_beta ({self.n_beta}) "
                f"!= n_electrons ({self.n_electrons})"
            )


# ---------------------------------------------------------------------------
# compute_molecular_integrals (PySCF)
# ---------------------------------------------------------------------------


_FCI_CONFIG_LIMIT: int = 50_000_000
"""int : Auto-CASCI threshold — skip CASSCF orbital optimisation when the
active-space determinant count exceeds this limit."""


def compute_molecular_integrals(
    geometry: list[tuple[str, tuple[float, float, float]]],
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    cas: tuple[int, int] | None = None,
    casci: bool = False,
) -> MolecularIntegrals:
    """Run RHF (+ optional CASSCF/CASCI) and extract molecular integrals.

    Parameters
    ----------
    geometry : list of (str, (float, float, float))
        Molecular geometry.  Each element is ``(atom_symbol, (x, y, z))``
        with coordinates in **Angstroms**.
    basis : str, optional
        Gaussian basis set name (default ``"sto-3g"``).
    charge : int, optional
        Net charge of the molecule (default ``0``).
    spin : int, optional
        Spin multiplicity minus one, i.e. ``2S`` (default ``0`` for
        singlet).
    cas : tuple of (int, int) or None, optional
        ``(nelecas, ncas)`` for CAS active-space reduction.  When
        provided, runs CASSCF (or CASCI if *casci* is ``True``) after
        RHF and returns integrals in the active space only
        (``n_orbitals = ncas``).  ``None`` (default) uses the full MO
        space.
    casci : bool, optional
        If ``True`` and *cas* is not ``None``, use CASCI instead of
        CASSCF.  CASCI uses HF MOs directly (no orbital optimisation),
        which is faster for large active spaces where CASSCF's
        iterative FCI solver would be infeasible.  Also auto-enabled
        when ``ncas >= 15`` or when the determinant count exceeds
        ``_FCI_CONFIG_LIMIT``.

    Returns
    -------
    MolecularIntegrals
        Integrals and metadata needed by :class:`MolecularHamiltonian`.
        When *cas* is used, ``nuclear_repulsion`` is the frozen-core
        energy ``e_core`` (includes frozen-electron energy + nuclear
        repulsion), and ``n_orbitals`` equals ``ncas``.

    Raises
    ------
    ImportError
        If PySCF is not installed.

    Warns
    -----
    UserWarning
        If the RHF or CASSCF calculation does not converge (proceeds
        with unconverged orbitals).

    Examples
    --------
    >>> geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]
    >>> mi = compute_molecular_integrals(geometry, basis="sto-3g")  # doctest: +SKIP
    >>> mi.n_orbitals  # doctest: +SKIP
    2

    CAS example (N₂ with 10 active electrons in 8 orbitals):

    >>> n2 = [("N", (0, 0, 0)), ("N", (0, 0, 1.1))]
    >>> mi = compute_molecular_integrals(n2, "cc-pvdz", cas=(10, 8))  # doctest: +SKIP
    >>> mi.n_orbitals  # doctest: +SKIP
    8
    """
    try:
        import pyscf  # noqa: F401
        from pyscf import ao2mo, gto, scf
    except ImportError as exc:
        raise ImportError(
            "PySCF is required for compute_molecular_integrals. "
            "Install it with: pip install pyscf"
        ) from exc

    # Build molecule
    mol = gto.Mole()
    mol.atom = [(atom, coord) for atom, coord in geometry]
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.unit = "Angstrom"
    mol.build()

    # Run RHF
    mf = scf.RHF(mol)
    mf.kernel()
    if not mf.converged:
        import warnings

        warnings.warn(
            "RHF did not converge. Proceeding with unconverged orbitals.",
            stacklevel=2,
        )

    # --- CAS active-space path ---
    if cas is not None:
        return _compute_cas_integrals(mol, mf, cas, casci, spin)

    # --- Full-space path (original behaviour) ---
    n_orb = mf.mo_coeff.shape[1]
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h1e = np.asarray(h1e, dtype=np.float64)

    eri_mo = ao2mo.full(mol, mf.mo_coeff)
    h2e = ao2mo.restore(1, eri_mo, n_orb).astype(np.float64)

    n_electrons = mol.nelectron
    n_alpha = (n_electrons + spin) // 2
    n_beta = (n_electrons - spin) // 2

    return MolecularIntegrals(
        h1e=h1e,
        h2e=h2e,
        nuclear_repulsion=float(mol.energy_nuc()),
        n_electrons=n_electrons,
        n_orbitals=n_orb,
        n_alpha=n_alpha,
        n_beta=n_beta,
    )


def _compute_cas_integrals(
    mol: object,
    mf: object,
    cas: tuple[int, int],
    casci: bool,
    spin: int,
) -> MolecularIntegrals:
    """Extract active-space integrals via CASSCF or CASCI.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Built PySCF molecule.
    mf : pyscf.scf.RHF
        Converged (or attempted) RHF object.
    cas : tuple of (int, int)
        ``(nelecas, ncas)`` active space specification.
    casci : bool
        Force CASCI (no orbital optimisation).
    spin : int
        ``2S`` spin value.

    Returns
    -------
    MolecularIntegrals
        Active-space integrals with ``nuclear_repulsion = e_core``.

    Raises
    ------
    ValueError
        If active electron counts are negative or exceed ``ncas``.
    """
    from math import comb as _comb

    from pyscf import ao2mo, fci, mcscf

    nelecas, ncas = cas
    n_elec_cas = nelecas
    n_alpha_cas = (nelecas + spin) // 2
    n_beta_cas = (nelecas - spin) // 2

    # Validate active electron / orbital counts
    if n_alpha_cas < 0 or n_beta_cas < 0:
        raise ValueError(
            f"Invalid CAS electron counts: n_alpha={n_alpha_cas}, "
            f"n_beta={n_beta_cas} (from nelecas={nelecas}, spin={spin})"
        )
    if n_alpha_cas > ncas or n_beta_cas > ncas:
        raise ValueError(
            f"CAS electrons exceed orbitals: n_alpha={n_alpha_cas}, "
            f"n_beta={n_beta_cas}, ncas={ncas}"
        )

    # Estimate determinant count for auto-CASCI decision
    n_configs = _comb(ncas, n_alpha_cas) * _comb(ncas, n_beta_cas)
    use_casci = casci or ncas >= 15 or n_configs > _FCI_CONFIG_LIMIT

    if use_casci:
        mc = mcscf.CASCI(mf, ncas=ncas, nelecas=nelecas)
    else:
        mc = mcscf.CASSCF(mf, ncas=ncas, nelecas=nelecas)

    # Linear molecules need non-symmetry FCI solver
    if hasattr(mol, "symmetry") and mol.symmetry:
        topgroup = getattr(mol, "topgroup", "")
        if topgroup in ("Dooh", "Coov"):
            mc.fcisolver = fci.direct_spin1.FCISolver(mol)

    # Skip kernel for infeasibly large CAS (integrals-only mode)
    if use_casci and n_configs > _FCI_CONFIG_LIMIT:
        logger.info(
            "Skipping FCI solve for CAS(%s,%d): %s configs > %s limit. "
            "Extracting integrals only.",
            nelecas,
            ncas,
            f"{n_configs:,}",
            f"{_FCI_CONFIG_LIMIT:,}",
        )
    else:
        mc.kernel()
        if not use_casci and not mc.converged:
            import warnings

            warnings.warn(
                f"CASSCF did not converge for CAS({nelecas},{ncas}). "
                "Integrals may be unreliable.",
                stacklevel=3,
            )

    # Extract active-space integrals
    h1e_cas, e_core = mc.h1e_for_cas()
    active_mo = mc.mo_coeff[:, mc.ncore : mc.ncore + mc.ncas]
    h2e_cas = ao2mo.full(mol, active_mo)
    h2e_cas = ao2mo.restore(1, h2e_cas, ncas)

    h1e_cas = np.asarray(h1e_cas, dtype=np.float64)
    h2e_cas = np.asarray(h2e_cas, dtype=np.float64)

    return MolecularIntegrals(
        h1e=h1e_cas,
        h2e=h2e_cas,
        nuclear_repulsion=float(e_core),
        n_electrons=n_elec_cas,
        n_orbitals=ncas,
        n_alpha=n_alpha_cas,
        n_beta=n_beta_cas,
    )


# ---------------------------------------------------------------------------
# Persistent cache via joblib
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_DIR = os.path.join(
    os.environ.get("QVARTOOLS_CACHE_DIR", os.path.expanduser("~/.cache/qvartools")),
    "integrals",
)


def get_integral_cache(
    cache_dir: str | None = None,
) -> Callable[..., MolecularIntegrals]:
    """Return a cached version of :func:`compute_molecular_integrals`.

    Uses ``joblib.Memory`` for transparent disk-based caching of PySCF
    integral computations.  Repeated calls with the same arguments
    return instantly from disk.

    Parameters
    ----------
    cache_dir : str or None, optional
        Directory for cached results.  Defaults to
        ``~/.cache/qvartools/integrals`` (overridable via
        ``QVARTOOLS_CACHE_DIR`` environment variable).

    Returns
    -------
    callable
        A cached version of ``compute_molecular_integrals`` with the
        same signature.
    """
    try:
        from joblib import Memory
    except ImportError as exc:
        raise ImportError(
            "joblib is required for integral caching. "
            "Install it with: pip install joblib"
        ) from exc

    location = cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR
    memory = Memory(location, verbose=0)
    _joblib_cached = memory.cache(compute_molecular_integrals)
    logger.info("Integral cache enabled at %s", location)

    def _cas_aware_cached(
        geometry: list[tuple[str, tuple[float, float, float]]],
        basis: str = "sto-3g",
        charge: int = 0,
        spin: int = 0,
        cas: tuple[int, int] | None = None,
        casci: bool = False,
    ) -> MolecularIntegrals:
        if cas is not None:
            return compute_molecular_integrals(
                geometry, basis=basis, charge=charge, spin=spin, cas=cas, casci=casci
            )
        return _joblib_cached(geometry, basis=basis, charge=charge, spin=spin)

    return _cas_aware_cached


# Module-level default cached function (lazy init)
_default_cached_fn: Callable[..., MolecularIntegrals] | None = None


def cached_compute_molecular_integrals(
    geometry: list[tuple[str, tuple[float, float, float]]],
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    cas: tuple[int, int] | None = None,
    casci: bool = False,
) -> MolecularIntegrals:
    """Cached version of :func:`compute_molecular_integrals`.

    Identical interface, but results are persisted to disk via
    ``joblib.Memory``.  The default cache directory is
    ``~/.cache/qvartools/integrals``.

    CAS integrals are **not cached** because CASSCF orbital
    optimisation is non-deterministic.

    Parameters
    ----------
    geometry : list of (str, (float, float, float))
        Molecular geometry.
    basis : str, optional
        Basis set name (default ``"sto-3g"``).
    charge : int, optional
        Net charge (default ``0``).
    spin : int, optional
        2S (default ``0``).
    cas : tuple of (int, int) or None, optional
        ``(nelecas, ncas)`` for CAS. Bypasses cache when not ``None``.
    casci : bool, optional
        Force CASCI (default ``False``).

    Returns
    -------
    MolecularIntegrals
        Cached or freshly computed integrals.
    """
    # CAS integrals bypass cache (CASSCF is non-deterministic)
    if cas is not None:
        return compute_molecular_integrals(
            geometry, basis=basis, charge=charge, spin=spin, cas=cas, casci=casci
        )
    global _default_cached_fn  # noqa: PLW0603
    if _default_cached_fn is None:
        _default_cached_fn = get_integral_cache()
    return _default_cached_fn(geometry, basis=basis, charge=charge, spin=spin)


def clear_integral_cache(cache_dir: str | None = None) -> None:
    """Remove all cached integral data.

    Parameters
    ----------
    cache_dir : str or None, optional
        Cache directory to clear.  Defaults to the same directory
        used by :func:`get_integral_cache`.
    """
    location = cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR
    # Safety: refuse to delete directories that don't look like a cache
    if "qvartools" not in location and "cache" not in location.lower():
        raise ValueError(
            f"Refusing to delete '{location}' — path does not contain "
            f"'qvartools' or 'cache'. Pass an explicit cache directory."
        )
    if os.path.isdir(location):
        shutil.rmtree(location)
        logger.info("Integral cache cleared at %s", location)
