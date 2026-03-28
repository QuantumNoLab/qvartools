"""
bitstring_format --- Format conversion between qvartools and IBM bitstring formats
===================================================================================

Provides vectorized conversion between the qvartools configuration format
``[alpha_0, ..., alpha_{n-1}, beta_0, ..., beta_{n-1}]`` and the IBM
bitstring format used by ``qiskit-addon-sqd``.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "cartesian_product_configs",
    "configs_to_ibm_format",
    "hash_config",
    "ibm_format_to_configs",
    "split_spin_strings",
    "vectorized_dedup",
]


def split_spin_strings(
    configs: torch.Tensor,
    n_orbitals: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split configurations into unique alpha and beta occupation strings.

    Given configs in qvartools format ``[alpha_0, ..., alpha_{n-1},
    beta_0, ..., beta_{n-1}]``, extracts the alpha and beta halves
    and returns only the unique strings for each spin channel.

    Parameters
    ----------
    configs : torch.Tensor
        Binary configurations, shape ``(n_configs, 2 * n_orbitals)``.
    n_orbitals : int or None, optional
        Number of spatial orbitals.  If ``None``, inferred as
        ``configs.shape[1] // 2``.

    Returns
    -------
    unique_alpha : torch.Tensor
        Unique alpha occupation strings, shape ``(n_alpha, n_orbitals)``.
    unique_beta : torch.Tensor
        Unique beta occupation strings, shape ``(n_beta, n_orbitals)``.
    """
    if configs.ndim != 2:
        raise ValueError(f"configs must be 2D, got {configs.ndim}D")

    if n_orbitals is None:
        if configs.shape[1] % 2 != 0:
            raise ValueError(
                f"configs has odd column count ({configs.shape[1]}); "
                f"cannot infer n_orbitals. Pass n_orbitals explicitly."
            )
        n_orbitals = configs.shape[1] // 2
    elif configs.shape[1] != 2 * n_orbitals:
        raise ValueError(
            f"configs has {configs.shape[1]} columns, expected "
            f"2 * n_orbitals = {2 * n_orbitals}"
        )

    if configs.shape[0] == 0:
        empty_a = configs[:, :n_orbitals]
        empty_b = configs[:, n_orbitals:]
        return empty_a, empty_b

    alpha = configs[:, :n_orbitals]
    beta = configs[:, n_orbitals:]

    unique_alpha = torch.unique(alpha, dim=0)
    unique_beta = torch.unique(beta, dim=0)

    return unique_alpha, unique_beta


def cartesian_product_configs(
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Form all alpha-beta combinations as full configurations.

    Given unique alpha strings and unique beta strings, produces the
    Cartesian product ``{[alpha_i, beta_j] : i, j}`` as concatenated
    configuration vectors.

    Parameters
    ----------
    alpha : torch.Tensor
        Alpha occupation strings, shape ``(n_alpha, n_orbitals)``.
    beta : torch.Tensor
        Beta occupation strings, shape ``(n_beta, n_orbitals)``.

    Returns
    -------
    torch.Tensor
        Cartesian product configurations, shape
        ``(n_alpha * n_beta, 2 * n_orbitals)``.

    Notes
    -----
    The output may contain configurations with incorrect particle
    numbers (e.g., ``sum(alpha_i) + sum(beta_j) != N_e``).  Callers
    should filter by particle number if the Hamiltonian requires it.
    """
    if alpha.ndim != 2 or beta.ndim != 2:
        raise ValueError(
            f"alpha and beta must be 2D; got alpha.ndim={alpha.ndim}, beta.ndim={beta.ndim}"
        )
    if alpha.shape[1] != beta.shape[1]:
        raise ValueError(
            f"alpha and beta must have same n_orbitals; "
            f"got alpha.shape[1]={alpha.shape[1]} vs beta.shape[1]={beta.shape[1]}"
        )

    n_alpha = alpha.shape[0]
    n_beta = beta.shape[0]
    n_orb = alpha.shape[1]

    if n_alpha == 0 or n_beta == 0:
        ref = alpha if n_alpha > 0 else beta
        return torch.zeros(0, 2 * n_orb, dtype=ref.dtype, device=ref.device)

    if alpha.device != beta.device or alpha.dtype != beta.dtype:
        raise ValueError(
            f"alpha and beta must have same dtype and device; got "
            f"alpha({alpha.dtype}, {alpha.device}) vs beta({beta.dtype}, {beta.device})"
        )

    # alpha_i repeated for each beta_j
    alpha_expanded = alpha.repeat_interleave(n_beta, dim=0)
    # beta_j tiled for each alpha_i
    beta_expanded = beta.repeat(n_alpha, 1)

    return torch.cat([alpha_expanded, beta_expanded], dim=1)


def hash_config(config: torch.Tensor) -> int:
    """Hash a binary configuration to a unique integer.

    Treats the configuration as a big-endian bitstring and computes
    its integer value.

    Parameters
    ----------
    config : torch.Tensor
        Binary vector, shape ``(n_sites,)`` with entries in ``{0, 1}``.

    Returns
    -------
    int
        Integer hash of the configuration.
    """
    n = config.shape[0]
    val = 0
    for i in range(n):
        val = val * 2 + int(config[i].item())
    return val


def configs_to_ibm_format(
    configs: torch.Tensor | np.ndarray,
    n_orb: int,
    n_qubits: int,
) -> np.ndarray:
    """Convert config tensor to IBM bitstring matrix.

    Parameters
    ----------
    configs : torch.Tensor or np.ndarray
        ``(n_configs, 2*n_orb)`` array in qvartools format.
    n_orb : int
        Number of spatial orbitals.
    n_qubits : int
        Total qubit count (``2 * n_orb``).

    Returns
    -------
    np.ndarray
        ``(n_configs, n_qubits)`` bool array in IBM format.
    """
    if isinstance(configs, torch.Tensor):
        configs_np = configs.cpu().numpy()
    else:
        configs_np = np.asarray(configs)

    n = len(configs_np)
    if n == 0:
        return np.zeros((0, n_qubits), dtype=bool)

    bs = np.zeros((n, n_qubits), dtype=bool)
    bs[:, :n_orb] = configs_np[:, :n_orb][:, ::-1].astype(bool)
    bs[:, n_orb:] = configs_np[:, n_orb:][:, ::-1].astype(bool)
    return bs


def ibm_format_to_configs(
    bs_matrix: np.ndarray,
    n_orb: int,
    n_qubits: int,
) -> torch.Tensor:
    """Convert IBM bitstring matrix back to qvartools config tensor.

    Parameters
    ----------
    bs_matrix : np.ndarray
        ``(n_configs, n_qubits)`` bool array in IBM format.
    n_orb : int
        Number of spatial orbitals.
    n_qubits : int
        Total qubit count (``2 * n_orb``).

    Returns
    -------
    torch.Tensor
        ``(n_configs, n_qubits)`` long tensor in qvartools format.
    """
    bs = np.asarray(bs_matrix)
    n = len(bs)
    if n == 0:
        return torch.zeros(0, n_qubits, dtype=torch.long)

    configs = np.zeros((n, n_qubits), dtype=np.int64)
    configs[:, :n_orb] = bs[:, :n_orb][:, ::-1].astype(np.int64)
    configs[:, n_orb:] = bs[:, n_orb:][:, ::-1].astype(np.int64)
    return torch.from_numpy(configs)


def vectorized_dedup(
    existing_bs: np.ndarray | None,
    new_bs: np.ndarray,
) -> np.ndarray:
    """Return rows in *new_bs* not present in *existing_bs*.

    Parameters
    ----------
    existing_bs : np.ndarray or None
        ``(n_existing, n_cols)`` bool array, or ``None`` if empty.
    new_bs : np.ndarray
        ``(n_new, n_cols)`` bool array.

    Returns
    -------
    np.ndarray
        Truly-new rows (preserving order from *new_bs*).
    """
    if len(new_bs) == 0:
        return new_bs

    n_cols = new_bs.shape[1]

    _, first_idx = np.unique(
        np.ascontiguousarray(new_bs).view(
            np.dtype((np.void, new_bs.dtype.itemsize * n_cols))
        ),
        return_index=True,
    )
    first_idx.sort()
    new_unique = new_bs[first_idx]

    if existing_bs is None or len(existing_bs) == 0:
        return new_unique

    existing_set = {row.tobytes() for row in np.ascontiguousarray(existing_bs)}
    mask = np.array(
        [row.tobytes() not in existing_set for row in np.ascontiguousarray(new_unique)],
        dtype=bool,
    )
    return new_unique[mask]
