"""Tests for ConnectionCache."""

from __future__ import annotations

import pytest
import torch

from qvartools._utils.hashing.connection_cache import ConnectionCache
from qvartools.hamiltonians.spin.heisenberg import HeisenbergHamiltonian


class TestConnectionCache:
    """Tests for ConnectionCache put, get, stats, eviction, etc."""

    @pytest.fixture()
    def cache(self) -> ConnectionCache:
        return ConnectionCache(max_size=5)

    @pytest.fixture()
    def config_a(self) -> torch.Tensor:
        return torch.tensor([1, 0, 1, 0])

    @pytest.fixture()
    def config_b(self) -> torch.Tensor:
        return torch.tensor([0, 1, 0, 1])

    def test_put_and_get_roundtrip(self, cache: ConnectionCache) -> None:
        config = torch.tensor([1, 0, 1, 0])
        connections = torch.tensor([[1, 0, 0, 1], [0, 1, 1, 0]])
        elements = torch.tensor([0.5, -0.3])

        cache.put(config, connections, elements)
        result = cache.get(config)

        assert result is not None
        retrieved_conn, retrieved_elem = result
        assert torch.equal(retrieved_conn, connections)
        assert torch.equal(retrieved_elem, elements)

    def test_cache_miss_returns_none(
        self, cache: ConnectionCache, config_a: torch.Tensor
    ) -> None:
        result = cache.get(config_a)
        assert result is None

    def test_stats_hits_misses(self, cache: ConnectionCache) -> None:
        config = torch.tensor([1, 0, 1, 0])
        connections = torch.tensor([[1, 1, 0, 0]])
        elements = torch.tensor([1.0])

        cache.put(config, connections, elements)
        cache.get(config)  # hit
        cache.get(config)  # hit
        cache.get(torch.tensor([0, 0, 0, 0]))  # miss

        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert abs(stats["hit_rate"] - 2.0 / 3.0) < 1e-6

    def test_max_size_eviction(self) -> None:
        cache = ConnectionCache(max_size=2)
        dummy_conn = torch.tensor([[1, 0]])
        dummy_elem = torch.tensor([1.0])

        cache.put(torch.tensor([1, 0]), dummy_conn, dummy_elem)
        cache.put(torch.tensor([0, 1]), dummy_conn, dummy_elem)
        assert len(cache) == 2

        # Adding a third should evict the oldest
        cache.put(torch.tensor([1, 1]), dummy_conn, dummy_elem)
        assert len(cache) == 2

    def test_clear_empties_cache(self, cache: ConnectionCache) -> None:
        config = torch.tensor([1, 0, 1, 0])
        cache.put(config, torch.tensor([[0, 0, 0, 0]]), torch.tensor([0.0]))
        cache.get(config)

        cache.clear()
        assert len(cache) == 0
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_len(self, cache: ConnectionCache) -> None:
        assert len(cache) == 0
        cache.put(
            torch.tensor([1, 0, 0, 0]),
            torch.tensor([[0, 0, 0, 0]]),
            torch.tensor([0.0]),
        )
        assert len(cache) == 1

    def test_contains(self, cache: ConnectionCache) -> None:
        config = torch.tensor([1, 0, 1, 0])
        assert config not in cache
        cache.put(config, torch.tensor([[0, 0, 0, 0]]), torch.tensor([0.0]))
        assert config in cache


class TestGetOrCompute:
    """Tests for get_or_compute (unified auto-compute interface)."""

    @pytest.fixture()
    def hamiltonian(self) -> HeisenbergHamiltonian:
        return HeisenbergHamiltonian(num_spins=4, Jx=1.0, Jy=1.0, Jz=1.0)

    @pytest.fixture()
    def cache(self) -> ConnectionCache:
        return ConnectionCache(max_size=100)

    def test_get_or_compute_returns_connections(
        self, cache: ConnectionCache, hamiltonian: HeisenbergHamiltonian
    ) -> None:
        """get_or_compute should return (connected, elements) tuple."""
        config = torch.tensor([1, 0, 1, 0])
        connected, elements = cache.get_or_compute(config, hamiltonian)
        assert connected.ndim == 2
        assert connected.shape[1] == 4
        assert elements.shape[0] == connected.shape[0]

    def test_get_or_compute_caches_result(
        self, cache: ConnectionCache, hamiltonian: HeisenbergHamiltonian
    ) -> None:
        """Second call should hit cache, not recompute."""
        config = torch.tensor([1, 0, 1, 0])
        cache.get_or_compute(config, hamiltonian)
        cache.get_or_compute(config, hamiltonian)
        stats = cache.stats()
        assert stats["hits"] >= 1

    def test_get_or_compute_matches_direct(
        self, cache: ConnectionCache, hamiltonian: HeisenbergHamiltonian
    ) -> None:
        """Cached result should match direct hamiltonian.get_connections()."""
        config = torch.tensor([0, 1, 0, 1])
        direct_conn, direct_elem = hamiltonian.get_connections(config)
        cached_conn, cached_elem = cache.get_or_compute(config, hamiltonian)
        assert torch.equal(cached_conn, direct_conn)
        assert torch.allclose(cached_elem, direct_elem)

    def test_get_or_compute_evicts_when_full(
        self, hamiltonian: HeisenbergHamiltonian
    ) -> None:
        """Cache should evict when full."""
        cache = ConnectionCache(max_size=2)
        configs = [
            torch.tensor([1, 0, 1, 0]),
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([1, 1, 0, 0]),
        ]
        for c in configs:
            cache.get_or_compute(c, hamiltonian)
        assert len(cache) == 2


class TestLRUEviction:
    """Tests for LRU eviction order (not FIFO)."""

    def test_lru_evicts_least_recently_used(self) -> None:
        """Access A after B, then add C at max_size=2 → B should be evicted, not A."""
        cache = ConnectionCache(max_size=2)
        dummy_conn = torch.tensor([[1, 0]])
        dummy_elem = torch.tensor([1.0])

        a = torch.tensor([1, 0])
        b = torch.tensor([0, 1])
        c = torch.tensor([1, 1])

        cache.put(a, dummy_conn, dummy_elem)  # insert A
        cache.put(b, dummy_conn, dummy_elem)  # insert B
        cache.get(a)  # access A → A is now most recent
        cache.put(c, dummy_conn, dummy_elem)  # insert C → should evict B (LRU)

        assert a in cache, "A should survive (most recently accessed)"
        assert c in cache, "C should be present (just inserted)"
        assert b not in cache, "B should be evicted (least recently used)"

    def test_lru_order_with_multiple_accesses(self) -> None:
        """Multiple get() calls should update recency."""
        cache = ConnectionCache(max_size=3)
        dummy_conn = torch.tensor([[0, 0, 0]])
        dummy_elem = torch.tensor([0.0])

        configs = [
            torch.tensor([0, 0, 1]),
            torch.tensor([0, 1, 0]),
            torch.tensor([1, 0, 0]),
        ]
        for c in configs:
            cache.put(c, dummy_conn, dummy_elem)

        # Access config[0] and config[2], leaving config[1] as LRU
        cache.get(configs[0])
        cache.get(configs[2])

        # Insert a new config → should evict config[1] (LRU)
        new_config = torch.tensor([1, 1, 0])
        cache.put(new_config, dummy_conn, dummy_elem)

        assert configs[0] in cache
        assert configs[2] in cache
        assert new_config in cache
        assert configs[1] not in cache, "config[1] should be evicted (LRU)"


class TestPowersCaching:
    """Tests for powers tensor reuse (not rebuilt per call)."""

    def test_powers_tensor_cached_across_calls(self) -> None:
        """Powers tensor should be created once and reused."""
        cache = ConnectionCache(max_size=10)
        config = torch.tensor([1, 0, 1, 0])
        dummy_conn = torch.tensor([[0, 0, 0, 0]])
        dummy_elem = torch.tensor([0.0])

        cache.put(config, dummy_conn, dummy_elem)
        # After first operation, _powers should exist
        assert hasattr(cache, "_powers"), "Cache should have a _powers attribute"
        powers_id = id(cache._powers)

        cache.get(config)
        assert id(cache._powers) == powers_id, "Powers tensor should be reused"

        cache.put(torch.tensor([0, 1, 0, 1]), dummy_conn, dummy_elem)
        assert id(cache._powers) == powers_id, "Powers tensor should be reused"


class TestBatchEncoding:
    """Tests for hash_batch and get_batch vectorized operations."""

    def test_hash_batch_shape(self) -> None:
        """hash_batch returns integer tensor of shape (n_configs,)."""
        cache = ConnectionCache(max_size=10)
        configs = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])
        hashes = cache.hash_batch(configs)
        assert hashes.shape == (3,)
        assert hashes.dtype == torch.int64

    def test_hash_batch_matches_individual(self) -> None:
        """Batch hash matches per-config _hash for each row."""
        cache = ConnectionCache(max_size=10)
        configs = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])
        batch_hashes = cache.hash_batch(configs)
        for i in range(configs.shape[0]):
            individual = cache._hash(configs[i])
            assert batch_hashes[i].item() == individual

    def test_hash_batch_unique_for_distinct_configs(self) -> None:
        """Distinct binary configs produce distinct hashes."""
        cache = ConnectionCache(max_size=10)
        configs = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
        hashes = cache.hash_batch(configs)
        assert len(set(hashes.tolist())) == 4

    def test_get_batch_returns_cached_entries(self) -> None:
        """get_batch returns results for configs that are in cache."""
        cache = ConnectionCache(max_size=10)
        configs = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]])
        conn_a = torch.tensor([[0, 1, 1, 0]])
        elem_a = torch.tensor([0.5])
        cache.put(configs[0], conn_a, elem_a)

        results = cache.get_batch(configs)
        # configs[0] should be found, configs[1] should be None
        assert results[0] is not None
        assert results[1] is None
        assert torch.equal(results[0][0], conn_a)

    def test_get_batch_all_cached(self) -> None:
        """get_batch returns all entries when all are cached."""
        cache = ConnectionCache(max_size=10)
        dummy_conn = torch.tensor([[0, 0]])
        dummy_elem = torch.tensor([1.0])
        configs = torch.tensor([[1, 0], [0, 1]])
        cache.put(configs[0], dummy_conn, dummy_elem)
        cache.put(configs[1], dummy_conn, dummy_elem)

        results = cache.get_batch(configs)
        assert all(r is not None for r in results)
