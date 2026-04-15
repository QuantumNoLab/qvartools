"""nqs --- NQS-based method pipelines.

Public entry points for the four NQS methods plus a ``METHODS_REGISTRY``
that maps a stable method id to the runner, config class, and metadata.
The registry is used by the ``experiments/pipelines/010-013`` wrapper
scripts (and any future benchmark catalog tooling) so they can dispatch
by name without hard-importing each method module.
"""

from __future__ import annotations

from typing import Any

from qvartools.methods.nqs.hi_nqs_skqd import HINQSSKQDConfig, run_hi_nqs_skqd
from qvartools.methods.nqs.hi_nqs_sqd import HINQSSQDConfig, run_hi_nqs_sqd
from qvartools.methods.nqs.nqs_skqd import NQSSKQDConfig, run_nqs_skqd
from qvartools.methods.nqs.nqs_sqd import NQSSQDConfig, run_nqs_sqd

__all__ = [
    "NQSSQDConfig",
    "run_nqs_sqd",
    "NQSSKQDConfig",
    "run_nqs_skqd",
    "HINQSSQDConfig",
    "run_hi_nqs_sqd",
    "HINQSSKQDConfig",
    "run_hi_nqs_skqd",
    "METHODS_REGISTRY",
]


METHODS_REGISTRY: dict[str, dict[str, Any]] = {
    "nqs_sqd": {
        "run_fn": run_nqs_sqd,
        "config_cls": NQSSQDConfig,
        "iterative": False,
        "has_krylov_expansion": False,
        "has_ibm_solver": False,
        "has_pt2_selection": False,
        "supports_initial_basis": False,
        "description": "Two-stage: train NQS via VMC, sample, diagonalise.",
        "pipeline_folder": "012_nqs_sqd",
    },
    "nqs_skqd": {
        "run_fn": run_nqs_skqd,
        "config_cls": NQSSKQDConfig,
        "iterative": False,
        "has_krylov_expansion": True,
        "has_ibm_solver": False,
        "has_pt2_selection": False,
        "supports_initial_basis": False,
        "description": "Two-stage: train NQS, sample, Krylov expand, diagonalise.",
        "pipeline_folder": "013_nqs_skqd",
    },
    "hi_nqs_sqd": {
        "run_fn": run_hi_nqs_sqd,
        "config_cls": HINQSSQDConfig,
        "iterative": True,
        "has_krylov_expansion": False,
        "has_ibm_solver": True,
        "has_pt2_selection": True,
        "supports_initial_basis": True,
        "description": "Iterative HI loop: NQS sampling, batch diag, teacher feedback, optional PT2 selection.",
        "pipeline_folder": "010_hi_nqs_sqd",
    },
    "hi_nqs_skqd": {
        "run_fn": run_hi_nqs_skqd,
        "config_cls": HINQSSKQDConfig,
        "iterative": True,
        "has_krylov_expansion": True,
        "has_ibm_solver": True,
        "has_pt2_selection": False,
        "supports_initial_basis": True,
        "description": "Iterative HI loop: NQS sampling, Krylov expand, batch diag, teacher feedback.",
        "pipeline_folder": "011_hi_nqs_skqd",
    },
}
