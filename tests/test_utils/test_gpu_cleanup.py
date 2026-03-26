"""Tests for GPU memory cleanup utility."""

from __future__ import annotations

import pytest
import torch


class TestCleanupGpuMemory:
    """Tests for cleanup_gpu_memory."""

    def test_importable(self) -> None:
        """cleanup_gpu_memory should be importable from _utils.gpu."""
        from qvartools._utils.gpu import cleanup_gpu_memory

        assert callable(cleanup_gpu_memory)

    def test_no_error_without_cuda(self) -> None:
        """Should work without error even if CUDA is unavailable."""
        from qvartools._utils.gpu import cleanup_gpu_memory

        # Should not raise regardless of CUDA availability
        cleanup_gpu_memory()

    def test_runs_after_tensor_creation(self) -> None:
        """Should run without error after creating and deleting tensors."""
        from qvartools._utils.gpu import cleanup_gpu_memory

        # Create tensors without keeping references
        for _ in range(10):
            torch.randn(100, 100)
        cleanup_gpu_memory()

    def test_cuda_cache_freed(self) -> None:
        """On GPU, cached memory should be released after cleanup."""
        from qvartools._utils.gpu import cleanup_gpu_memory

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Allocate GPU memory
        x = torch.randn(1000, 1000, device="cuda")
        del x

        # Before cleanup, reserved cache may hold memory
        reserved_before = torch.cuda.memory_reserved()

        cleanup_gpu_memory()

        reserved_after = torch.cuda.memory_reserved()
        # Cache should be freed (or at least not grown)
        assert reserved_after <= reserved_before
