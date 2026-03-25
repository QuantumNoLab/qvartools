"""reference --- Exact and near-exact reference solvers."""

from __future__ import annotations

from qvartools.solvers.reference.ccsd import CCSDSolver
from qvartools.solvers.reference.fci import FCISolver

__all__ = ["FCISolver", "CCSDSolver"]
