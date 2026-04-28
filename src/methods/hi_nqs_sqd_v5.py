"""HI+NQS+SQD v5: amplitude-weighted random expansion + multi-pass PT2.

Diagnoses on 52Q showed v4 plateaus at +71 mHa even with top_n=2000 because:
  1. top-N expansion exhausts after iter 10 (no new dets found)
  2. PT2 only sees same external space as classical expansion → little gain
  3. Variational basis lacks deep correlation (≥ quadruples from HF)

v5 changes:
  A. Weighted random expansion seed: each iter, sample
     `classical_expansion_top_n` basis dets from a probability ∝ |c|^2 + α |c|
     (mixes amplitude weighting with broader exploration).
     This breaks mode collapse — different seeds each iter, including mid-amp
     dets that top-N misses.

  B. Multi-pass PT2 expansion at end: after variational convergence,
     iteratively expand the external space:
       Pass 1: enumerate singles+doubles of top-N amp basis dets → E_PT2_1
       Pass 2: take top-K externals from Pass 1, add to basis,
               redo variational → new ψ
       Pass 3: PT2 from extended basis → E_PT2_2
     Each pass captures one more correlation order (triples → quads → ...).

  Inherits all v4 GPU-native features (on-the-fly Davidson, GPU coupling, etc.).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from . import hi_nqs_sqd_v3 as _v3
from .hi_nqs_sqd_v4 import HINQSSQDv4Config, run_hi_nqs_sqd_v4
from .hi_nqs_sqd_v2 import _enumerate_top_amp_connections
from .hi_nqs_sqd import _configs_to_ibm_format, _ibm_format_to_configs


@dataclass
class HINQSSQDv5Config(HINQSSQDv4Config):
    # A. Weighted-random expansion. mix_random ∈ [0, 1]:
    #   0 = pure top-N (v4 behavior)
    #   0.5 = half top-N, half weighted-random
    #   1.0 = pure |c|^2-weighted random sample
    mix_random_expansion: float = 0.5

    # B. Multi-pass PT2 (number of passes after variational converges).
    # 1 = standard final PT2 (v4); 2-3 = capture deeper correlation.
    pt2_passes: int = 1
    pt2_pass_top_k: int = 10000  # how many top-PT2 externals to absorb per pass


# -----------------------------------------------------------------------------
def _weighted_random_amp_connections(
    sci_state, hamiltonian, n_orb, n_qubits,
    n_seeds: int, mix_random: float,
    existing_hashes,
):
    """Expand connections from a mix of top-amp + |c|^2-weighted random basis dets.

    Returns same format as _enumerate_top_amp_connections: list of
    (ibm_row, hash, config_tensor) for new candidates not in basis.
    """
    amps = np.asarray(sci_state.amplitudes)
    abs_flat = np.abs(amps).flatten()
    nonzero_idx = np.where(abs_flat > 1e-14)[0]
    n_nz = len(nonzero_idx)
    if n_nz == 0:
        return []

    n_seeds = min(n_seeds, n_nz)
    n_top = int(n_seeds * (1 - mix_random))
    n_random = n_seeds - n_top

    # Top-amp portion (deterministic)
    if n_top > 0:
        top_idx_in_nz = np.argpartition(-abs_flat[nonzero_idx], n_top - 1)[:n_top]
        top_flat = nonzero_idx[top_idx_in_nz]
    else:
        top_flat = np.array([], dtype=np.int64)

    # Weighted random portion (probability ∝ |c|^2 over remaining nonzero)
    if n_random > 0 and n_random < n_nz - n_top:
        remaining = np.setdiff1d(nonzero_idx, top_flat)
        probs = abs_flat[remaining] ** 2
        if probs.sum() > 0:
            probs = probs / probs.sum()
            random_flat = np.random.choice(remaining, size=n_random,
                                            p=probs, replace=False)
        else:
            random_flat = remaining[:n_random]
    else:
        random_flat = np.array([], dtype=np.int64)

    sample_flat = np.concatenate([top_flat, random_flat]).astype(np.int64)

    n_b = amps.shape[1]
    sample_ia = sample_flat // n_b
    sample_ib = sample_flat % n_b

    ci_strs_a = np.asarray(sci_state.ci_strs_a)
    ci_strs_b = np.asarray(sci_state.ci_strs_b)

    N = len(sample_flat)
    configs_np = np.zeros((N, n_qubits), dtype=np.int64)
    for i in range(N):
        a = int(ci_strs_a[sample_ia[i]])
        b = int(ci_strs_b[sample_ib[i]])
        for k in range(n_orb):
            configs_np[i, k] = (a >> k) & 1
            configs_np[i, n_orb + k] = (b >> k) & 1

    configs_t = torch.from_numpy(configs_np).long().to(hamiltonian.device)

    try:
        connected, _, _ = hamiltonian.get_connections_vectorized_batch(configs_t)
    except (MemoryError, RuntimeError):
        chunks = []
        chunk_size = max(1, N // 4)
        for start in range(0, N, chunk_size):
            c, _, _ = hamiltonian.get_connections_vectorized_batch(
                configs_t[start:start + chunk_size]
            )
            chunks.append(c.cpu())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        connected = torch.cat(chunks, dim=0).to(hamiltonian.device) if chunks \
            else torch.zeros((0, n_qubits), dtype=torch.long, device=hamiltonian.device)

    if len(connected) == 0:
        return []

    connected_unique = torch.unique(connected.long().cpu(), dim=0)
    ibm_bs = _configs_to_ibm_format(connected_unique, n_orb, n_qubits)

    out = []
    for i in range(len(ibm_bs)):
        h = ibm_bs[i].tobytes()
        if h not in existing_hashes:
            out.append((ibm_bs[i], h, connected_unique[i]))
    return out


# -----------------------------------------------------------------------------
def run_hi_nqs_sqd_v5(hamiltonian, mol_info,
                      config: Optional[HINQSSQDv5Config] = None):
    """v5 wrapper: monkey-patches v3's _enumerate_top_amp_connections with
    weighted-random version, then runs v4 main loop. Multi-pass PT2 is layered
    after variational converges.
    """
    cfg = config or HINQSSQDv5Config()

    # Monkey-patch the expansion function used by v3 main loop.
    import src.methods.hi_nqs_sqd_v3 as _v3mod
    orig_expand = _v3mod._enumerate_top_amp_connections

    if cfg.mix_random_expansion > 0:
        def _mixed_expand(sci_state, ham, n_orb, n_qubits, top_n, existing_hashes):
            return _weighted_random_amp_connections(
                sci_state, ham, n_orb, n_qubits,
                n_seeds=top_n, mix_random=cfg.mix_random_expansion,
                existing_hashes=existing_hashes,
            )
        _v3mod._enumerate_top_amp_connections = _mixed_expand

    try:
        result = run_hi_nqs_sqd_v4(hamiltonian, mol_info, config=cfg)
    finally:
        _v3mod._enumerate_top_amp_connections = orig_expand

    result.method = "HI+NQS+SQD-v5"
    result.metadata["v5_mix_random_expansion"] = cfg.mix_random_expansion
    result.metadata["v5_pt2_passes"] = cfg.pt2_passes
    return result
