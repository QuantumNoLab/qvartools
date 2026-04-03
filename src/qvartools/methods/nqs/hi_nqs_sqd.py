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
from qvartools.methods.nqs._pt2_helpers import (
    compute_e_pt2,
    compute_pt2_scores,
    compute_temperature,
    evict_by_coefficient,
)
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
    from qiskit_addon_sqd.fermion import (
        solve_fermion as ibm_solve_fermion,  # type: ignore[import-untyped]
    )

    _IBM_SQD_AVAILABLE = True
except ImportError:
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
    use_ibm_solver : bool or None
        Control IBM ``solve_fermion`` usage (default ``None``).
        ``None``: auto-enable when ``qiskit_addon_sqd`` is installed.
        ``True``: force enable (fails if not installed).
        ``False``: force disable (use ``gpu_solve_fermion`` fallback).
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
        Weight for the REINFORCE energy loss term (default ``0.0``).
    entropy_weight : float
        Weight for the entropy regularisation term (default ``0.0``).
    compute_pt2_correction : bool
        Compute EN-PT2 energy correction after final iteration
        (default ``False``).  When ``True``, ``metadata`` includes
        ``e_pt2`` and ``corrected_energy = energy + e_pt2``.
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
    use_ibm_solver: bool | None = None
    device: str = "cpu"
    use_pt2_selection: bool = False
    pt2_top_k: int = 2000
    max_basis_size: int = 10_000
    convergence_window: int = 3
    initial_temperature: float = 1.0
    final_temperature: float = 0.3
    teacher_weight: float = 1.0
    energy_weight: float = 0.0
    entropy_weight: float = 0.0
    compute_pt2_correction: bool = False


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
    α/β marginals) and ``advantage(x) = H_xx - <H_xx>_{p_teacher}``, i.e. the
    diagonal energy minus its mean under the teacher distribution.

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
    if energy_weight > 0 and hamiltonian is None:
        raise ValueError(
            f"energy_weight={energy_weight} requires hamiltonian to be provided, "
            "but got hamiltonian=None"
        )
    if energy_weight > 0 and hamiltonian is not None:
        with torch.no_grad():
            diag_e = hamiltonian.diagonal_elements_batch(configs_dev).to(device)
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

    # Tri-state IBM solver control:
    #   None (default) = auto-enable when qiskit_addon_sqd is installed
    #   True = force enable (α×β Cartesian product, dramatically better accuracy)
    #   False = force disable (use gpu_solve_fermion fallback)
    if cfg.use_ibm_solver is None:
        use_ibm = _IBM_SQD_AVAILABLE
        if use_ibm:
            logger.info("Auto-enabling IBM solve_fermion (qiskit_addon_sqd available)")
    else:
        use_ibm = cfg.use_ibm_solver

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
    converge_count = 0
    # Persistent eigenvector state for PT2 scoring across iterations
    prev_coeffs: np.ndarray | None = None
    prev_batch_configs: torch.Tensor | None = None
    prev_energy: float = float("inf")

    for iteration in range(cfg.n_iterations):
        logger.info("HI+NQS+SQD iteration %d / %d", iteration + 1, cfg.n_iterations)

        # --- Temperature (annealing when PT2 is on, fixed otherwise) ---
        if cfg.use_pt2_selection:
            temp = compute_temperature(
                iteration,
                cfg.n_iterations,
                cfg.initial_temperature,
                cfg.final_temperature,
            )
        else:
            temp = cfg.temperature

        # --- NQS sampling ---
        with torch.no_grad():
            new_configs = nqs.sample(cfg.n_samples_per_iter, temperature=temp).to(
                device
            )

        # Deduplicate against cumulative basis (numpy for vectorized_dedup)
        if cumulative_basis.shape[0] > 0:
            cb_np = cumulative_basis.cpu().numpy()
            nc_np = new_configs.cpu().numpy()
            deduped_np = vectorized_dedup(cb_np, nc_np)
            unique_new = torch.from_numpy(deduped_np).long().to(device)
        else:
            unique_new = torch.unique(new_configs, dim=0)

        # --- PT2 filtering (only when enabled and we have a prior eigenvector) ---
        if (
            cfg.use_pt2_selection
            and unique_new.shape[0] > 0
            and prev_coeffs is not None
            and prev_batch_configs is not None
        ):
            scores = compute_pt2_scores(
                unique_new, prev_batch_configs, prev_coeffs, hamiltonian, prev_energy
            )
            n_keep = min(cfg.pt2_top_k, unique_new.shape[0])
            top_idx = torch.tensor(
                np.argsort(scores)[::-1][:n_keep].copy(),
                dtype=torch.long,
                device=unique_new.device,
            )
            unique_new = unique_new[top_idx]
            logger.info(
                "  PT2 filtered: %d → %d configs (top_k=%d)",
                len(scores),
                n_keep,
                cfg.pt2_top_k,
            )

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

            # IBM solve_fermion with α×β Cartesian product expansion
            if _IBM_SQD_AVAILABLE and use_ibm:
                ibm_data = configs_to_ibm_format(batch_configs, n_orb, n_qubits)
                # Skip recover_configurations (S-CORE) — it's designed for noisy
                # quantum hardware samples, not clean classical NQS samples.
                # PR #30's best results used "no rescore" mode (no S-CORE).
                # spin_sq = s(s+1) where s = spin/2. Default 0 = singlet.
                _spin = mol_info.get("spin", 0)
                _spin_sq = (_spin / 2) * (_spin / 2 + 1) if _spin else 0
                e_b, sci_state, occs_b, _ = ibm_solve_fermion(
                    ibm_data,
                    hcore=hamiltonian.integrals.h1e,
                    eri=hamiltonian.integrals.h2e,
                    spin_sq=_spin_sq,
                )
                # sci_state.amplitudes is 2D (n_alpha_strs × n_beta_strs).
                # Build per-config weights from α/β marginals for NQS teacher.
                # batch_configs stays unchanged (original sampled configs).
                amps_2d = np.abs(sci_state.amplitudes) ** 2
                alpha_marginal = amps_2d.sum(axis=1)  # sum over beta
                beta_marginal = amps_2d.sum(axis=0)  # sum over alpha
                alpha_marginal /= max(alpha_marginal.sum(), 1e-30)
                beta_marginal /= max(beta_marginal.sum(), 1e-30)

                # Map each sampled config to a teacher weight via marginals
                ci_a = sci_state.ci_strs_a
                ci_b = sci_state.ci_strs_b
                a_map = {int(s): float(v) for s, v in zip(ci_a, alpha_marginal)}
                b_map = {int(s): float(v) for s, v in zip(ci_b, beta_marginal)}

                coeffs_b = np.zeros(batch_configs.shape[0], dtype=np.float64)
                for i in range(batch_configs.shape[0]):
                    cfg_np = batch_configs[i].cpu().numpy()
                    a_int = sum(int(cfg_np[k]) << k for k in range(n_orb))
                    b_int = sum(int(cfg_np[k + n_orb]) << k for k in range(n_orb))
                    coeffs_b[i] = np.sqrt(a_map.get(a_int, 0.0) * b_map.get(b_int, 0.0))
                # solve_fermion returns electronic energy; add nuclear repulsion
                e_b = float(e_b) + float(hamiltonian.integrals.nuclear_repulsion)
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

        # --- Coefficient-based eviction (when PT2 is on) ---
        # ASCI pattern: diagonalise the FULL cumulative basis to get
        # coefficients for every config, then keep highest |c_i|².
        if cfg.use_pt2_selection and cumulative_basis.shape[0] > cfg.max_basis_size:
            n_full = cumulative_basis.shape[0]
            if n_full <= 50_000:
                h_full = hamiltonian.matrix_elements_fast(cumulative_basis)
                h_np = h_full.detach().cpu().numpy().astype(np.float64)
                h_np = 0.5 * (h_np + h_np.T)
                evals, evecs = np.linalg.eigh(h_np)
                full_coeffs = evecs[:, 0]
                evict_e0 = float(evals[0])
            else:
                from scipy.sparse.linalg import eigsh as sp_eigsh

                h_sp = hamiltonian.build_sparse_hamiltonian(cumulative_basis)
                evals, evecs = sp_eigsh(h_sp.tocsr(), k=1, which="SA")
                full_coeffs = evecs[:, 0]
                evict_e0 = float(evals[0])
            cumulative_basis, evicted_coeffs = evict_by_coefficient(
                cumulative_basis, full_coeffs, cfg.max_basis_size
            )
            # Use post-eviction state as PT2 reference (consistent basis + coeffs)
            prev_coeffs = evicted_coeffs.copy()
            prev_batch_configs = cumulative_basis.clone()
            prev_energy = evict_e0
            logger.info(
                "  evicted to %d configs (full-basis diag)", cumulative_basis.shape[0]
            )
        else:
            # --- Update persistent eigenvector state for next iteration's PT2 ---
            # Note: when cumulative_basis > max_configs_per_batch, best_batch_energy
            # is from a random sub-sample (not full-basis diag). This is an
            # approximation of E0 for PT2 scoring — acceptable because the eviction
            # path (above) uses full-basis diag when basis exceeds max_basis_size.
            if best_coeffs is not None and best_batch_configs is not None:
                prev_coeffs = best_coeffs.copy()
                prev_batch_configs = best_batch_configs.clone()
                prev_energy = best_batch_energy

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
                teacher_weight=cfg.teacher_weight,
                energy_weight=cfg.energy_weight,
                entropy_weight=cfg.entropy_weight,
                hamiltonian=hamiltonian,
            )

        logger.info(
            "  energy=%.8f best=%.8f basis=%d",
            iter_energy,
            best_energy,
            cumulative_basis.shape[0],
        )

        # --- Convergence (with window when PT2 is on) ---
        if len(energy_history) >= 2:
            delta = abs(energy_history[-1] - energy_history[-2])
            if delta < cfg.energy_tol:
                if cfg.use_pt2_selection:
                    converge_count += 1
                    if converge_count >= cfg.convergence_window:
                        converged = True
                        logger.info(
                            "  converged: |dE|=%.2e (window=%d)",
                            delta,
                            cfg.convergence_window,
                        )
                        break
                else:
                    converged = True
                    logger.info("  converged: |dE|=%.2e", delta)
                    break
            else:
                converge_count = 0

    if not converged:
        logger.warning(
            "HI+NQS+SQD did not converge after %d iterations (best=%.8f)",
            cfg.n_iterations,
            best_energy,
        )

    # --- Optional E_PT2 correction ---
    metadata: dict[str, Any] = {
        "energy_history": energy_history,
        "n_iterations": len(energy_history),
        "final_basis_size": int(cumulative_basis.shape[0]),
    }

    if (
        cfg.compute_pt2_correction
        and prev_coeffs is not None
        and prev_batch_configs is not None
        and prev_coeffs.shape[0] == prev_batch_configs.shape[0]
    ):
        e_pt2 = compute_e_pt2(prev_batch_configs, prev_coeffs, hamiltonian, best_energy)
        metadata["e_pt2"] = e_pt2
        metadata["corrected_energy"] = best_energy + e_pt2
        logger.info(
            "  E_PT2=%.6f Ha, corrected=%.8f Ha",
            e_pt2,
            best_energy + e_pt2,
        )

    wall_time = time.perf_counter() - t_start

    return SolverResult(
        energy=best_energy,
        diag_dim=int(cumulative_basis.shape[0]),
        wall_time=wall_time,
        method="HI+NQS+SQD",
        converged=converged,
        metadata=metadata,
    )
