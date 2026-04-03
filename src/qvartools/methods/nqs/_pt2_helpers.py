"""PT2 selection helpers for HI-NQS v3 (ADR-005).

Standalone pure functions for:

- Epstein-Nesbet PT2 scoring
- EN-PT2 energy correction (E_PT2)
- Coefficient-based basis eviction (ASCI pattern)
- Linear temperature annealing

Functions
---------
compute_pt2_scores
    Score candidates by EN-PT2 importance.
compute_e_pt2
    Compute EN-PT2 energy correction.
evict_by_coefficient
    Keep highest-|c_i|² configs.
compute_temperature
    Linear temperature annealing.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch

from qvartools._utils.hashing.config_hash import config_integer_hash

logger = logging.getLogger(__name__)


def compute_pt2_scores(
    candidates: torch.Tensor,
    basis: torch.Tensor,
    coeffs: np.ndarray,
    hamiltonian: Any,
    e0: float,
) -> np.ndarray:
    """Score candidate configs by Epstein-Nesbet PT2 importance.

    For each candidate ``x`` NOT already in the basis, computes::

        score(x) = |⟨x|H|Φ₀⟩|² / |E₀ - H_xx|

    where ``Φ₀ = sum_i c_i |x_i⟩`` is the current ground-state estimate.

    Parameters
    ----------
    candidates : torch.Tensor
        Candidate configurations, shape ``(n_cand, n_qubits)``.
    basis : torch.Tensor
        Current basis configurations, shape ``(n_basis, n_qubits)``.
    coeffs : np.ndarray
        Eigenvector coefficients for the basis, shape ``(n_basis,)``.
    hamiltonian
        Hamiltonian with ``get_connections`` and ``diagonal_element`` methods.
    e0 : float
        Current ground-state energy estimate.

    Returns
    -------
    np.ndarray
        PT2 importance scores, shape ``(n_cand,)``, all non-negative.
    """
    n_cand = candidates.shape[0]
    scores = np.zeros(n_cand, dtype=np.float64)

    # Build hash → coefficient map for the basis
    # config_integer_hash returns int for <64 sites, tuple for >=64 sites;
    # both are hashable and usable as dict keys directly.
    basis_hash_list = config_integer_hash(basis)
    basis_coeff_map: dict = {}
    basis_hash_set: set = set()
    for i, h in enumerate(basis_hash_list):
        basis_coeff_map[h] = float(coeffs[i])
        basis_hash_set.add(h)

    # Hash candidates to skip those already in basis
    cand_hash_list = config_integer_hash(candidates)

    for idx in range(n_cand):
        cand_h = cand_hash_list[idx]
        if cand_h in basis_hash_set:
            # Candidate already in basis — not an external determinant
            continue

        config = candidates[idx]

        # Diagonal element H_xx
        h_xx = float(hamiltonian.diagonal_element(config))
        if not math.isfinite(h_xx):
            continue

        denom = abs(e0 - h_xx)
        if denom < 1e-14:
            denom = 1e-14

        # Coupling: ⟨x|H|Φ₀⟩ = sum_{y in basis} H_xy * c_y
        connected, h_elements = hamiltonian.get_connections(config)

        if connected is None or len(connected) == 0:
            continue

        conn_hashes = config_integer_hash(connected)
        coupling = 0.0
        for j in range(len(connected)):
            c_y = basis_coeff_map.get(conn_hashes[j], 0.0)
            if c_y != 0.0:
                coupling += float(h_elements[j]) * c_y

        scores[idx] = coupling**2 / denom

    return scores


def compute_e_pt2(
    basis: torch.Tensor,
    coeffs: np.ndarray,
    hamiltonian: Any,
    e0: float,
) -> float:
    r"""Compute Epstein-Nesbet second-order perturbation energy correction.

    Sums over all determinants connected to the basis but NOT in the basis::

        E_PT2 = Σ_{x ∉ V} |⟨x|H|Ψ₀⟩|² / (E₀ - H_xx)

    where ``Ψ₀ = Σ_i c_i |x_i⟩`` and the sum runs over external
    determinants reachable via single and double excitations.  For
    real-valued Hamiltonians, ``|⟨x|H|Ψ₀⟩|² = ⟨x|H|Ψ₀⟩²``.

    Parameters
    ----------
    basis : torch.Tensor
        **Full** variational basis, shape ``(n_basis, n_qubits)``.
        Must be the complete basis used for the diagonalisation that
        produced ``coeffs`` and ``e0``.
    coeffs : np.ndarray
        Ground-state eigenvector from diagonalising H in ``basis``,
        shape ``(n_basis,)``.
    hamiltonian
        Hamiltonian with ``get_connections`` and ``diagonal_element``.
    e0 : float
        Variational ground-state energy from the same diagonalisation
        as ``coeffs``.

    Returns
    -------
    float
        E_PT2 correction (typically negative).
    """
    basis_hash_list = config_integer_hash(basis)
    basis_hash_set: set = set(basis_hash_list)

    # Accumulate coupling ⟨x|H|Ψ₀⟩ for each external determinant x
    # and collect H_xx for the denominator.
    external_coupling: dict = {}  # hash -> coupling
    external_config: dict = {}  # hash -> config tensor (for H_xx lookup)

    for idx in range(basis.shape[0]):
        c_i = float(coeffs[idx])
        if abs(c_i) < 1e-14:
            continue

        connections, h_elements = hamiltonian.get_connections(basis[idx])
        if connections is None or len(connections) == 0:
            continue

        conn_hashes = config_integer_hash(connections)
        for j in range(len(connections)):
            h_conn = conn_hashes[j]
            if h_conn in basis_hash_set:
                continue
            # Accumulate coupling: ⟨x|H|Ψ₀⟩ += H_xy * c_y
            contrib = float(h_elements[j]) * c_i
            if h_conn in external_coupling:
                external_coupling[h_conn] += contrib
            else:
                external_coupling[h_conn] = contrib
                external_config[h_conn] = connections[j]

    # Compute E_PT2 = Σ coupling² / (e0 - H_xx)
    e_pt2 = 0.0
    for h_ext, coupling in external_coupling.items():
        config = external_config[h_ext]
        h_xx = float(hamiltonian.diagonal_element(config))
        if not math.isfinite(h_xx):
            continue
        denom = e0 - h_xx
        if abs(denom) < 1e-14:
            continue
        e_pt2 += coupling**2 / denom

    return e_pt2


def evict_by_coefficient(
    basis: torch.Tensor,
    coeffs: np.ndarray,
    max_size: int,
) -> tuple[torch.Tensor, np.ndarray]:
    """Keep only the highest-|c_i|² configs in the basis.

    Parameters
    ----------
    basis : torch.Tensor
        Basis configurations, shape ``(n_basis, n_qubits)``.
    coeffs : np.ndarray
        Eigenvector coefficients, shape ``(n_basis,)``.
    max_size : int
        Maximum number of configs to retain.

    Returns
    -------
    trimmed_basis : torch.Tensor
        Retained configurations, shape ``(min(n_basis, max_size), n_qubits)``.
        Original row ordering is preserved.
    trimmed_coeffs : np.ndarray
        Corresponding coefficients (same order as ``trimmed_basis``).

    Raises
    ------
    ValueError
        If ``max_size < 1``.
    """
    if max_size < 1:
        raise ValueError(f"max_size must be >= 1, got {max_size}")

    n = basis.shape[0]
    if n <= max_size:
        return basis, coeffs

    # Sort by |c_i|² descending, keep top max_size, then re-sort by index
    importance = np.abs(coeffs) ** 2
    top_indices = np.argsort(importance)[::-1][:max_size].copy()
    top_indices_sorted = np.sort(top_indices)

    return basis[top_indices_sorted], coeffs[top_indices_sorted]


def compute_temperature(
    iteration: int,
    max_iterations: int,
    t_init: float,
    t_final: float,
) -> float:
    """Compute linearly annealed temperature.

    Parameters
    ----------
    iteration : int
        Current iteration index (0-based).
    max_iterations : int
        Total number of iterations.
    t_init : float
        Temperature at iteration 0.
    t_final : float
        Temperature at the final iteration.

    Returns
    -------
    float
        Interpolated temperature, clamped to ``[min(t_init, t_final),
        max(t_init, t_final)]``.

    Raises
    ------
    ValueError
        If ``max_iterations < 1`` or ``iteration < 0``.
    """
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
    if iteration < 0:
        raise ValueError(f"iteration must be >= 0, got {iteration}")
    if max_iterations == 1:
        return t_init
    progress = min(iteration / (max_iterations - 1), 1.0)
    return t_init + progress * (t_final - t_init)
