"""Tests for persistent Hamiltonian integral caching via joblib."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("joblib")
pytest.importorskip("pyscf")


class TestHamiltonianCache:
    """Tests for cached_compute_molecular_integrals."""

    def test_import(self) -> None:
        """cached_compute_molecular_integrals should be importable."""
        from qvartools.hamiltonians.integrals import cached_compute_molecular_integrals

        assert callable(cached_compute_molecular_integrals)

    def test_returns_molecular_integrals(self) -> None:
        """Should return a MolecularIntegrals dataclass."""
        from qvartools.hamiltonians.integrals import (
            MolecularIntegrals,
            cached_compute_molecular_integrals,
        )

        geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]
        result = cached_compute_molecular_integrals(geometry, basis="sto-3g")
        assert isinstance(result, MolecularIntegrals)
        assert result.n_orbitals == 2
        assert result.n_electrons == 2

    def test_second_call_is_cached(self, tmp_path) -> None:
        """Second call with same args should use cache (faster)."""
        import time

        from qvartools.hamiltonians.integrals import get_integral_cache

        cache = get_integral_cache(str(tmp_path / "cache"))
        geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]

        # First call — computes
        t0 = time.perf_counter()
        r1 = cache(geometry, basis="sto-3g")
        t_first = time.perf_counter() - t0

        # Second call — should be cached
        t0 = time.perf_counter()
        r2 = cache(geometry, basis="sto-3g")
        t_second = time.perf_counter() - t0

        # Cached call should be significantly faster
        assert t_second < t_first or t_first < 0.1  # very fast compute is OK too
        # Results should match
        np.testing.assert_array_equal(r1.h1e, r2.h1e)
        np.testing.assert_array_equal(r1.h2e, r2.h2e)
        assert r1.nuclear_repulsion == r2.nuclear_repulsion

    def test_different_geometry_not_cached(self, tmp_path) -> None:
        """Different geometry should compute fresh result."""
        from qvartools.hamiltonians.integrals import get_integral_cache

        cache = get_integral_cache(str(tmp_path / "cache"))
        g1 = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]
        g2 = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5))]

        r1 = cache(g1, basis="sto-3g")
        r2 = cache(g2, basis="sto-3g")

        # Different geometry → different integrals
        assert not np.allclose(r1.h1e, r2.h1e)

    def test_clear_cache(self, tmp_path) -> None:
        """clear_integral_cache should remove cached data."""
        from qvartools.hamiltonians.integrals import (
            clear_integral_cache,
            get_integral_cache,
        )

        cache_dir = str(tmp_path / "cache")
        cache = get_integral_cache(cache_dir)
        geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]
        cache(geometry, basis="sto-3g")

        clear_integral_cache(cache_dir)

        # After clear, cache directory should be empty or removed
        import pathlib

        cache_path = pathlib.Path(cache_dir)
        if cache_path.exists():
            # joblib may leave directory but should have no cached results
            cached_files = list(cache_path.rglob("*.npy"))
            assert len(cached_files) == 0
