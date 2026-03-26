"""gpu --- GPU-accelerated linear algebra and diagonalisation."""

from __future__ import annotations

import gc
import logging

import torch

from qvartools._utils.gpu.diagnostics import compute_occupancies
from qvartools._utils.gpu.diagnostics import gpu_solve_fermion as gpu_solve_fermion_diag
from qvartools._utils.gpu.fci_solver import (
    GPU_FCI_AVAILABLE,
    GPUFCISolver,
    compute_gpu_fci,
    compute_gpu_fci_from_integrals,
)
from qvartools._utils.gpu.linear_algebra import gpu_solve_fermion

__all__ = [
    "cleanup_gpu_memory",
    "gpu_solve_fermion",
    "gpu_solve_fermion_diag",
    "compute_occupancies",
    "GPU_FCI_AVAILABLE",
    "GPUFCISolver",
    "compute_gpu_fci",
    "compute_gpu_fci_from_integrals",
]

logger = logging.getLogger(__name__)


def cleanup_gpu_memory() -> None:
    """Release unused GPU memory and Python garbage.

    Calls ``gc.collect()`` to break reference cycles, then
    ``torch.cuda.empty_cache()`` and ``torch.cuda.synchronize()``
    to return cached GPU memory to the device allocator.

    Safe to call without CUDA — behaves as a no-op on CPU-only systems.
    """
    gc.collect()
    # Only touch CUDA if it was already initialized — avoids triggering
    # lazy CUDA initialization on CPU-only runs.
    if torch.cuda.is_initialized():
        before = torch.cuda.memory_reserved()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after = torch.cuda.memory_reserved()
        freed_mb = (before - after) / (1024 * 1024)
        if freed_mb > 0:
            logger.debug("cleanup_gpu_memory: freed %.1f MB GPU cache", freed_mb)
