"""Tests for experiments/pipelines/run_all_pipelines.py registry updates."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPELINES_DIR = ROOT / "experiments" / "pipelines"
sys.path.insert(0, str(PIPELINES_DIR))

import run_all_pipelines as rap


def test_vqe_pipelines_are_registered() -> None:
    scripts = {script for _, script, _, _ in rap.PIPELINES}
    assert "09_vqe/vqe_uccsd.py" in scripts
    assert "09_vqe/vqe_adapt.py" in scripts


def test_vqe_pipeline_scripts_exist() -> None:
    assert (PIPELINES_DIR / "09_vqe" / "vqe_uccsd.py").is_file()
    assert (PIPELINES_DIR / "09_vqe" / "vqe_adapt.py").is_file()
