"""
connection_cache --- Hash-based cache for Hamiltonian connections
================================================================

Provides :class:`ConnectionCache`, a dictionary-backed cache that maps
configuration hashes to their Hamiltonian-connected configurations and
matrix elements.  This avoids redundant calls to
:meth:`Hamiltonian.get_connections` when the same configuration is
encountered multiple times during iterative basis expansion or sampling.

The hash function converts a binary occupation vector to a unique integer
by interpreting it as a binary number (via powers of 2).  A powers tensor
is computed once and reused across all lookups for efficiency.

Eviction follows **LRU** (least-recently-used) order: the entry that was
neither accessed nor inserted most recently is evicted first.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from qvartools.hamiltonians.hamiltonian import Hamiltonian

__all__ = [
    "ConnectionCache",
]

logger = logging.getLogger(__name__)


class ConnectionCache:
    """Hash-based LRU cache for Hamiltonian connections.

    Stores ``(connected_configs, matrix_elements)`` tuples keyed by the
    integer hash of each configuration.  Provides O(1) lookup and
    LRU eviction when the cache exceeds ``max_size``.

    Parameters
    ----------
    max_size : int, optional
        Maximum number of entries the cache may hold (default ``100000``).
        When the cache is full, the **least-recently-used** entry is
        evicted on the next :meth:`put` or :meth:`get_or_compute` call.

    Attributes
    ----------
    max_size : int
        Maximum cache capacity.

    Examples
    --------
    >>> cache = ConnectionCache(max_size=1000)
    >>> config = torch.tensor([1, 0, 1, 0])
    >>> cache.put(config, connected, elements)
    >>> result = cache.get(config)
    """

    def __init__(self, max_size: int = 100_000) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self.max_size: int = max_size
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._hits: int = 0
        self._misses: int = 0
        self._powers: torch.Tensor | None = None
        self._powers_n: int = 0

    def _get_powers(self, n: int, device: torch.device) -> torch.Tensor:
        """Return the cached powers-of-2 tensor for *n* sites.

        The tensor is created once and reused.  If *n* or *device* change,
        the tensor is rebuilt.
        """
        if self._powers is not None and self._powers_n == n:
            return self._powers.to(device)
        self._powers = torch.tensor(
            [1 << k for k in range(n - 1, -1, -1)],
            dtype=torch.int64,
        )
        self._powers_n = n
        return self._powers.to(device)

    def _hash(self, config: torch.Tensor) -> int:
        """Convert a binary configuration to an integer hash."""
        powers = self._get_powers(config.shape[0], config.device)
        return int((config.to(torch.int64) * powers).sum().item())

    def _touch(self, key: int) -> None:
        """Move *key* to the end of the dict (mark as most-recently-used)."""
        value = self._cache.pop(key)
        self._cache[key] = value

    def _evict(self) -> None:
        """Evict the least-recently-used entry (first in dict order)."""
        oldest_key = next(iter(self._cache))
        del self._cache[oldest_key]

    def get(self, config: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Look up cached connections for a configuration.

        On a cache hit the entry is promoted to most-recently-used.

        Parameters
        ----------
        config : torch.Tensor
            Binary configuration vector, shape ``(n_sites,)``.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor) or None
            ``(connected_configs, matrix_elements)`` if found, otherwise
            ``None``.
        """
        key = self._hash(config)
        if key in self._cache:
            self._hits += 1
            self._touch(key)
            return self._cache[key]
        self._misses += 1
        return None

    def put(
        self,
        config: torch.Tensor,
        connections: torch.Tensor,
        elements: torch.Tensor,
    ) -> None:
        """Store connections for a configuration in the cache.

        If the cache is at capacity, the least-recently-used entry is
        evicted.

        Parameters
        ----------
        config : torch.Tensor
            Binary configuration vector, shape ``(n_sites,)``.
        connections : torch.Tensor
            Connected configurations, shape ``(n_conn, n_sites)``.
        elements : torch.Tensor
            Matrix elements, shape ``(n_conn,)``.
        """
        key = self._hash(config)

        if key in self._cache:
            # Update existing entry and promote to most-recent
            self._cache.pop(key)
        elif len(self._cache) >= self.max_size:
            self._evict()

        self._cache[key] = (connections.clone(), elements.clone())

    def get_or_compute(
        self,
        config: torch.Tensor,
        hamiltonian: Hamiltonian,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve connections, computing and caching if absent.

        This unifies the lookup-or-compute pattern: if the configuration
        is already cached its entry is returned (and promoted to
        most-recently-used); otherwise ``hamiltonian.get_connections``
        is called, the result is cached, and then returned.

        Parameters
        ----------
        config : torch.Tensor
            Single configuration, shape ``(num_sites,)``.
        hamiltonian : Hamiltonian
            The Hamiltonian to query for connections on a cache miss.

        Returns
        -------
        connected_configs : torch.Tensor
            Connected configurations, shape ``(n_conn, num_sites)``.
        matrix_elements : torch.Tensor
            Corresponding matrix elements, shape ``(n_conn,)``.
        """
        result = self.get(config)
        if result is not None:
            return result

        connected, elements = hamiltonian.get_connections(config)
        self.put(config, connected, elements)
        return connected, elements

    def clear(self) -> None:
        """Remove all entries from the cache and reset statistics."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, Any]:
        """Return cache performance statistics.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``"hits"`` : int --- number of successful lookups.
            - ``"misses"`` : int --- number of failed lookups.
            - ``"hit_rate"`` : float --- fraction of lookups that were hits
              (0.0 if no lookups have been made).
            - ``"size"`` : int --- current number of cached entries.
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size": len(self._cache),
        }

    def __len__(self) -> int:
        """Return the number of cached entries."""
        return len(self._cache)

    def __contains__(self, config: torch.Tensor) -> bool:
        """Check whether a configuration is in the cache.

        Parameters
        ----------
        config : torch.Tensor
            Binary configuration vector, shape ``(n_sites,)``.

        Returns
        -------
        bool
            ``True`` if the configuration hash is in the cache.
        """
        key = self._hash(config)
        return key in self._cache

    def __repr__(self) -> str:
        return (
            f"ConnectionCache(size={len(self._cache)}, "
            f"max_size={self.max_size}, "
            f"hit_rate={self.stats()['hit_rate']:.2%})"
        )
