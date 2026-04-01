"""
hi_nqs_sqd --- HI+NQS+SQD: iterative self-consistent NQS-SQD loop
====================================================================

Iterative pipeline that trains an autoregressive transformer NQS to
sample configurations, solves via subspace diagonalisation (SQD), feeds
the eigenvector back as a teacher signal, and repeats until convergence.

At each iteration the NQS samples are converted to IBM SQD format,
optionally processed through ``qiskit_addon_sqd`` configuration recovery,
and diagonalised with the internal GPU solver.

External dependencies (``qiskit_addon_sqd``) are optional.

Functions
---------
run_hi_nqs_sqd
    Execute the full HI+NQS+SQD pipeline.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from qvartools._utils.formatting.bitstring_format import (
    configs_to_ibm_format,
    vectorized_dedup,
)
from qvartools._utils.gpu.diagnostics import gpu_solve_fermion
from qvartools.nqs.transformer.autoregressive import AutoregressiveTransformer
from qvartools.solvers.solver import SolverResult

__all__ = [
    "HINQSSQDConfig",
    "run_hi_nqs_sqd",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from qiskit_addon_sqd.configuration_recovery import (
        recover_configurations,  # type: ignore[import-untyped]
    )
    from qiskit_addon_sqd.fermion import (
        solve_fermion as ibm_solve_fermion,  # type: ignore[import-untyped]
    )

    _IBM_SQD_AVAILABLE = True
except ImportError:
    recover_configurations = None  # type: ignore[assignment]
    ibm_solve_fermion = None  # type: ignore[assignment]
    _IBM_SQD_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HINQSSQDConfig:
    """Configuration for the HI+NQS+SQD pipeline.

    Parameters
    ----------
    n_iterations : int
        Number of outer self-consistent iterations (default ``10``).
    n_samples_per_iter : int
        NQS samples drawn per iteration (default ``10_000``).
    n_batches : int
        Configuration-recovery batches per iteration (default ``5``).
    max_configs_per_batch : int
        Maximum configs retained per batch (default ``5000``).
    energy_tol : float
        Convergence threshold in Hartree (default ``1e-5``).
    nqs_lr : float
        NQS optimiser learning rate (default ``1e-3``).
    nqs_train_epochs : int
        NQS training epochs per iteration (default ``50``).
    embed_dim : int
        Transformer embedding dimension (default ``64``).
    n_heads : int
        Number of attention heads (default ``4``).
    n_layers : int
        Number of transformer layers per channel (default ``4``).
    temperature : float
        NQS sampling temperature (default ``1.0``).  Ignored when
        ``use_pt2_selection`` is ``True`` (uses annealing schedule instead).
    use_ibm_solver : bool
        Use IBM ``solve_fermion`` when available (default ``False``).
        Set to ``True`` only if ``qiskit_addon_sqd`` is installed with a
        compatible API version.
    device : str
        Torch device string (default ``"cpu"``).
    use_pt2_selection : bool
        Enable PT2-based perturbative selection of NQS samples
        (default ``False``).  When ``True``, only the highest-scoring
        configs (by Epstein-Nesbet PT2 importance) are added to the basis
        each iteration, and low-coefficient configs are evicted.
    pt2_top_k : int
        Number of highest-PT2-scoring configs to keep per iteration
        (default ``2000``).  Only used when ``use_pt2_selection`` is ``True``.
    max_basis_size : int
        Maximum cumulative basis size (default ``10_000``).  Excess configs
        are evicted by lowest ``|c_i|²`` after each diagonalisation.
        Only used when ``use_pt2_selection`` is ``True``.
    convergence_window : int
        Number of consecutive iterations with ``|ΔE| < energy_tol``
        required before declaring convergence (default ``3``).
    initial_temperature : float
        NQS sampling temperature at iteration 0 (default ``1.0``).
        Linearly annealed to ``final_temperature``.
    final_temperature : float
        NQS sampling temperature at the final iteration (default ``0.3``).
    teacher_weight : float
        Weight for the KL-divergence teacher loss term (default ``1.0``).
    energy_weight : float
        Weight for the REINFORCE energy loss term (default ``0.1``).
    entropy_weight : float
        Weight for the entropy regularisation term (default ``0.05``).
    """

    n_iterations: int = 10
    n_samples_per_iter: int = 10_000
    n_batches: int = 5
    max_configs_per_batch: int = 5000
    energy_tol: float = 1e-5
    nqs_lr: float = 1e-3
    nqs_train_epochs: int = 50
    embed_dim: int = 64
    n_heads: int = 4
    n_layers: int = 4
    temperature: float = 1.0
    use_ibm_solver: bool = False
    device: str = "cpu"
    use_pt2_selection: bool = False
    pt2_top_k: int = 2000
    max_basis_size: int = 10_000
    convergence_window: int = 3
    initial_temperature: float = 1.0
    final_temperature: float = 0.3
    teacher_weight: float = 1.0
    energy_weight: float = 0.1
    entropy_weight: float = 0.05


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _train_nqs_teacher(
    nqs: AutoregressiveTransformer,
    configs: torch.Tensor,
    coeffs: np.ndarray,
    n_orb: int,
    lr: float,
    epochs: int,
    *,
    teacher_weight: float = 1.0,
    energy_weight: float = 0.0,
    entropy_weight: float = 0.0,
    hamiltonian: Any = None,
) -> list[float]:
    """Train NQS using eigenvector coefficients as teacher signal.

    Minimises a weighted combination of three loss terms:

    - **Teacher** (KL divergence): ``- sum_x p_teacher(x) * log q(x)``
    - **Energy** (REINFORCE): ``sum_x p_teacher(x) * advantage(x) * log q(x)``
    - **Entropy**: ``mean(log q(x))``

    where ``p_teacher(x) = |c_x|^2 / Z`` (full joint distribution, NOT
    α/β marginals) and ``advantage(x) = H_xx - E_0`` (diagonal energy).

    Parameters
    ----------
    nqs : AutoregressiveTransformer
        Transformer NQS.
    configs : torch.Tensor
        Basis configurations, shape ``(n_basis, 2*n_orb)``.
    coeffs : np.ndarray
        Eigenvector coefficients, shape ``(n_basis,)``.
    n_orb : int
        Spatial orbitals per spin channel.
    lr : float
        Learning rate.
    epochs : int
        Training epochs.
    teacher_weight : float, optional
        Weight for the KL teacher term (default ``1.0``).
    energy_weight : float, optional
        Weight for the REINFORCE energy term (default ``0.0``).
        Requires ``hamiltonian`` to be provided.
    entropy_weight : float, optional
        Weight for the entropy regularisation term (default ``0.0``).
    hamiltonian : Hamiltonian or None, optional
        Required when ``energy_weight > 0`` (for diagonal elements).

    Returns
    -------
    list of float
        Per-epoch loss values.
    """
    device = next(nqs.parameters()).device

    # Build teacher distribution: p(x) = |c_x|^2 / Z (full joint)
    weights = np.abs(coeffs) ** 2
    total = weights.sum()
    if total > 0:
        weights = weights / total
    weights_t = torch.from_numpy(weights).float().to(device)

    configs_dev = configs.to(device).long()
    alpha = configs_dev[:, :n_orb]
    beta = configs_dev[:, n_orb:]

    # Precompute energy advantage if needed
    advantage_t: torch.Tensor | None = None
    if energy_weight > 0 and hamiltonian is not None:
        with torch.no_grad():
            diag_e = hamiltonian.diagonal_elements_batch(configs_dev)
            e0_approx = float((weights_t * diag_e).sum())
            if math.isfinite(e0_approx):
                advantage_t = (diag_e - e0_approx).float()
            else:
                logger.warning(
                    "Non-finite e0_approx=%.4e; skipping energy term", e0_approx
                )
                advantage_t = None

    optimiser = torch.optim.Adam(nqs.parameters(), lr=lr)
    losses: list[float] = []

    nqs.train()
    for _epoch in range(epochs):
        optimiser.zero_grad()
        log_probs = nqs.log_prob(alpha, beta)

        # Teacher term: - sum_x p(x) * log q(x)
        loss = teacher_weight * (-(weights_t * log_probs).sum())

        # Energy term: sum_x p(x) * advantage(x) * log q(x)
        if energy_weight > 0 and advantage_t is not None:
            loss = loss + energy_weight * ((weights_t * advantage_t * log_probs).sum())

        # Entropy term: mean(log q(x))
        if entropy_weight > 0:
            loss = loss + entropy_weight * log_probs.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(nqs.parameters(), max_norm=1.0)
        optimiser.step()
        losses.append(float(loss.item()))

    nqs.eval()
    return losses


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_hi_nqs_sqd(
    hamiltonian: Any,
    mol_info: dict[str, Any],
    config: HINQSSQDConfig | None = None,
    *,
    initial_basis: torch.Tensor | None = None,
) -> SolverResult:
    """Execute the HI+NQS+SQD pipeline.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Molecular Hamiltonian.
    mol_info : dict
        Molecular metadata.  Required keys: ``"n_orbitals"``,
        ``"n_alpha"``, ``"n_beta"``, ``"n_qubits"``.
    config : HINQSSQDConfig or None
        Pipeline configuration.
    initial_basis : torch.Tensor or None, optional
        Pre-computed configurations to seed the cumulative basis
        (e.g., from NF+DCI Stage 1-2).  Shape ``(n_configs, n_qubits)``.
        If ``None`` (default), starts from an empty basis.

    Returns
    -------
    SolverResult
        Energy, timing, convergence, and per-iteration metadata.

    Raises
    ------
    ValueError
        If ``mol_info`` is missing required keys, or if ``initial_basis``
        has wrong shape, non-binary values, or floating-point/complex dtype.
    RuntimeError
        If all diagonalisation batches produce non-finite energies.
    """
    cfg = config or HINQSSQDConfig()

    # Support mol_info with or without orbital counts (fall back to hamiltonian)
    _integrals = getattr(hamiltonian, "integrals", None)
    n_orb: int = mol_info.get(
        "n_orbitals", _integrals.n_orbitals if _integrals else None
    )
    n_alpha: int = mol_info.get("n_alpha", _integrals.n_alpha if _integrals else None)
    n_beta: int = mol_info.get("n_beta", _integrals.n_beta if _integrals else None)
    if n_orb is None or n_alpha is None or n_beta is None:
        raise ValueError(
            "n_orbitals, n_alpha, and n_beta must be provided via mol_info "
            "or hamiltonian.integrals. Got: "
            f"n_orbitals={n_orb}, n_alpha={n_alpha}, n_beta={n_beta}"
        )
    n_qubits: int = mol_info.get("n_qubits", 2 * n_orb)
    device = torch.device(cfg.device)

    logger.info(
        "run_hi_nqs_sqd: %d orbitals, %d alpha, %d beta",
        n_orb,
        n_alpha,
        n_beta,
    )

    t_start = time.perf_counter()

    # --- Build NQS ---
    nqs = AutoregressiveTransformer(
        n_orbitals=n_orb,
        n_alpha=n_alpha,
        n_beta=n_beta,
        embed_dim=cfg.embed_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
    ).to(device)
    nqs.eval()

    # --- Occupancies (uniform prior) ---
    occ_alpha = np.full(n_orb, n_alpha / n_orb)
    occ_beta = np.full(n_orb, n_beta / n_orb)

    # --- Cumulative basis (warm-start from initial_basis if provided) ---
    if initial_basis is not None:
        # Validate raw input before any cast (fail-fast)
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
        cumulative_basis = initial_basis.to(dtype=torch.long, device=device)
        cumulative_basis = torch.unique(cumulative_basis, dim=0)
        logger.info(
            "Warm-starting with %d initial basis configs", cumulative_basis.shape[0]
        )
    else:
        cumulative_basis = torch.zeros(0, n_qubits, dtype=torch.long, device=device)

    energy_history: list[float] = []
    best_energy = float("inf")
    converged = False

    for iteration in range(cfg.n_iterations):
        logger.info("HI+NQS+SQD iteration %d / %d", iteration + 1, cfg.n_iterations)

        # --- NQS sampling ---
        with torch.no_grad():
            new_configs = nqs.sample(
                cfg.n_samples_per_iter, temperature=cfg.temperature
            ).to(device)

        # Deduplicate against cumulative basis (numpy for vectorized_dedup)
        if cumulative_basis.shape[0] > 0:
            cb_np = cumulative_basis.cpu().numpy()
            nc_np = new_configs.cpu().numpy()
            deduped_np = vectorized_dedup(cb_np, nc_np)
            unique_new = torch.from_numpy(deduped_np).long().to(device)
        else:
            unique_new = torch.unique(new_configs, dim=0)

        cumulative_basis = torch.cat([cumulative_basis, unique_new], dim=0)
        cumulative_basis = torch.unique(cumulative_basis, dim=0)

        logger.info(
            "  sampled %d, %d unique new, cumulative %d",
            cfg.n_samples_per_iter,
            unique_new.shape[0],
            cumulative_basis.shape[0],
        )

        # --- Batch diagonalisation ---
        batch_energies: list[float] = []
        best_coeffs: np.ndarray | None = None
        best_batch_configs: torch.Tensor | None = None
        best_batch_energy = float("inf")
        latest_occs: Any = None

        for _batch_idx in range(cfg.n_batches):
            if cumulative_basis.shape[0] > cfg.max_configs_per_batch:
                indices = torch.randperm(cumulative_basis.shape[0])[
                    : cfg.max_configs_per_batch
                ]
                batch_configs = cumulative_basis[indices]
            else:
                batch_configs = cumulative_basis

            # Optional IBM configuration recovery
            if _IBM_SQD_AVAILABLE and cfg.use_ibm_solver:
                ibm_data = configs_to_ibm_format(batch_configs, n_orb, n_qubits)
                n_samples = ibm_data.shape[0]
                uniform_probs = np.ones(n_samples) / n_samples
                refined_matrix, _ = recover_configurations(
                    ibm_data,
                    uniform_probs,
                    (occ_alpha, occ_beta),
                    num_elec_a=n_alpha,
                    num_elec_b=n_beta,
                )
                e_b, sci_state, occs_b, _ = ibm_solve_fermion(
                    refined_matrix,
                    hcore=hamiltonian.integrals.h1e,
                    eri=hamiltonian.integrals.h2e,
                )
                coeffs_b = sci_state.amplitudes
            else:
                e_b, coeffs_b, occs_b = gpu_solve_fermion(batch_configs, hamiltonian)

            e_b = float(e_b)
            if not math.isfinite(e_b):
                logger.warning(
                    "Non-finite energy %.4e in batch %d, skipping", e_b, _batch_idx
                )
                continue
            batch_energies.append(e_b)
            latest_occs = occs_b

            if e_b < best_batch_energy:
                best_batch_energy = e_b
                best_coeffs = np.asarray(coeffs_b)
                best_batch_configs = batch_configs

        if not batch_energies:
            raise RuntimeError(
                f"All {cfg.n_batches} batches produced non-finite energies "
                f"at iteration {iteration + 1}. Check Hamiltonian integrals."
            )
        else:
            iter_energy = float(np.min(batch_energies))
        energy_history.append(iter_energy)
        best_energy = min(best_energy, iter_energy)

        # --- Update occupancies ---
        if isinstance(latest_occs, tuple) and len(latest_occs) == 2:
            occ_alpha = np.clip(np.asarray(latest_occs[0], dtype=np.float64), 0.0, 1.0)
            occ_beta = np.clip(np.asarray(latest_occs[1], dtype=np.float64), 0.0, 1.0)

        # --- NQS teacher training ---
        if best_coeffs is not None and best_batch_configs is not None:
            _train_nqs_teacher(
                nqs,
                best_batch_configs,
                best_coeffs,
                n_orb,
                lr=cfg.nqs_lr,
                epochs=cfg.nqs_train_epochs,
            )

        logger.info(
            "  energy=%.8f best=%.8f basis=%d",
            iter_energy,
            best_energy,
            cumulative_basis.shape[0],
        )

        # --- Convergence ---
        if len(energy_history) >= 2:
            delta = abs(energy_history[-1] - energy_history[-2])
            if delta < cfg.energy_tol:
                converged = True
                logger.info("  converged: |dE|=%.2e", delta)
                break

    wall_time = time.perf_counter() - t_start

    return SolverResult(
        energy=best_energy,
        diag_dim=int(cumulative_basis.shape[0]),
        wall_time=wall_time,
        method="HI+NQS+SQD",
        converged=converged,
        metadata={
            "energy_history": energy_history,
            "n_iterations": len(energy_history),
            "final_basis_size": int(cumulative_basis.shape[0]),
        },
    )
