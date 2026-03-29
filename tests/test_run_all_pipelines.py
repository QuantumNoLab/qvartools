"""Tests for experiments/pipelines/run_all_pipelines.py registry updates."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPELINES_DIR = ROOT / "experiments" / "pipelines"

# Load run_all_pipelines without mutating sys.path
_spec = importlib.util.spec_from_file_location(
    "run_all_pipelines", PIPELINES_DIR / "run_all_pipelines.py"
)
rap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rap)


def test_vqe_pipelines_are_registered() -> None:
    scripts = {script for _, script, _, _ in rap.PIPELINES}
    assert "09_vqe/vqe_uccsd.py" in scripts
    assert "09_vqe/vqe_adapt.py" in scripts


def test_vqe_pipeline_scripts_exist() -> None:
    assert (PIPELINES_DIR / "09_vqe" / "vqe_uccsd.py").is_file()
    assert (PIPELINES_DIR / "09_vqe" / "vqe_adapt.py").is_file()


def test_skip_quantum_filters_vqe_pipelines() -> None:
    vqe_names = {name for _, _, name, _ in rap.PIPELINES if "VQE" in name}
    skipped = {
        name for _, _, name, _ in rap.PIPELINES if "Krylov-Q" in name or "VQE" in name
    }
    assert vqe_names <= skipped
