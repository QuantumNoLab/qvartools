"""
_shared --- Shared helpers for NQS method runners
==================================================

Internal utilities extracted from the four NQS method modules
(``nqs_sqd``, ``nqs_skqd``, ``hi_nqs_sqd``, ``hi_nqs_skqd``) to remove
duplication.  Behaviour-preserving extractions only — no API changes
exposed to callers of the public ``run_*`` functions.

Functions
---------
build_autoregressive_nqs
    Build an :class:`AutoregressiveTransformer` NQS on a target device.
extract_orbital_counts
    Extract ``(n_orb, n_alpha, n_beta, n_qubits)`` from ``mol_info``
    with fall-back to ``hamiltonian.integrals``.
validate_initial_basis
    Validate a user-provided ``initial_basis`` tensor (shape, dtype,
    binary values) and return a normalised long-tensor on the target
    device.
"""

from __future__ import annotations

from typing import Any

import torch

from qvartools.nqs.transformer.autoregressive import AutoregressiveTransformer

__all__ = [
    "build_autoregressive_nqs",
    "extract_orbital_counts",
    "validate_initial_basis",
]


def build_autoregressive_nqs(
    n_orb: int,
    n_alpha: int,
    n_beta: int,
    *,
    embed_dim: int,
    n_heads: int,
    n_layers: int,
    device: torch.device | str,
) -> AutoregressiveTransformer:
    """Build an autoregressive transformer NQS on the target device.

    Parameters
    ----------
    n_orb : int
        Spatial orbitals per spin channel.
    n_alpha, n_beta : int
        Particle counts in each spin channel.
    embed_dim : int
        Transformer embedding dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer layers per spin channel.
    device : torch.device or str
        Target device for the NQS parameters.

    Returns
    -------
    AutoregressiveTransformer
        The constructed NQS, moved to ``device``.
    """
    return AutoregressiveTransformer(
        n_orbitals=n_orb,
        n_alpha=n_alpha,
        n_beta=n_beta,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
    ).to(device)


def extract_orbital_counts(
    mol_info: dict[str, Any],
    hamiltonian: Any,
) -> tuple[int, int, int, int]:
    """Extract ``(n_orb, n_alpha, n_beta, n_qubits)`` from mol_info.

    Falls back to ``hamiltonian.integrals`` for any key missing from
    ``mol_info``.  Used by HI methods that accept partially-populated
    ``mol_info`` dicts.

    Parameters
    ----------
    mol_info : dict
        Molecular metadata.  May contain any subset of ``"n_orbitals"``,
        ``"n_alpha"``, ``"n_beta"``, ``"n_qubits"``.
    hamiltonian : Hamiltonian
        Molecular Hamiltonian; expected to expose ``.integrals`` with
        ``.n_orbitals``, ``.n_alpha``, ``.n_beta`` attributes when
        ``mol_info`` is missing those keys.

    Returns
    -------
    tuple of (int, int, int, int)
        ``(n_orbitals, n_alpha, n_beta, n_qubits)``.

    Raises
    ------
    ValueError
        If ``n_orbitals``, ``n_alpha``, or ``n_beta`` cannot be resolved
        from either ``mol_info`` or ``hamiltonian.integrals``.
    """
    _integrals = getattr(hamiltonian, "integrals", None)
    n_orb = mol_info.get("n_orbitals", _integrals.n_orbitals if _integrals else None)
    n_alpha = mol_info.get("n_alpha", _integrals.n_alpha if _integrals else None)
    n_beta = mol_info.get("n_beta", _integrals.n_beta if _integrals else None)
    if n_orb is None or n_alpha is None or n_beta is None:
        raise ValueError(
            "n_orbitals, n_alpha, and n_beta must be provided via mol_info "
            "or hamiltonian.integrals. Got: "
            f"n_orbitals={n_orb}, n_alpha={n_alpha}, n_beta={n_beta}"
        )
    n_qubits = mol_info.get("n_qubits", 2 * n_orb)
    return int(n_orb), int(n_alpha), int(n_beta), int(n_qubits)


def validate_initial_basis(
    initial_basis: torch.Tensor,
    n_qubits: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Validate and normalise a user-supplied ``initial_basis`` tensor.

    Performs fail-fast checks on the raw input (dtype, shape, binary
    values) before any cast, then returns a deduplicated long-tensor
    on the target device.

    Parameters
    ----------
    initial_basis : torch.Tensor
        Pre-computed configurations to seed a cumulative basis.
    n_qubits : int
        Expected second-axis dimensionality.
    device : torch.device or str
        Target device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Long-dtype tensor on ``device``, deduplicated along axis 0.

    Raises
    ------
    ValueError
        If ``initial_basis`` has floating-point or complex dtype, wrong
        shape, or non-binary values.
    """
    if initial_basis.is_floating_point() or initial_basis.is_complex():
        raise ValueError(
            f"initial_basis must be integer or bool dtype (binary occupations), "
            f"got {initial_basis.dtype}"
        )
    if initial_basis.ndim != 2 or initial_basis.shape[1] != n_qubits:
        raise ValueError(
            f"initial_basis must have shape (n_configs, {n_qubits}), "
            f"but got {tuple(initial_basis.shape)}"
        )
    if not torch.all((initial_basis == 0) | (initial_basis == 1)):
        raise ValueError("initial_basis must contain only binary values {0, 1}")
    out = initial_basis.to(dtype=torch.long, device=device)
    return torch.unique(out, dim=0)
