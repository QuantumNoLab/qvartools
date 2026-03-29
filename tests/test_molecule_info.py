"""Tests for lightweight molecule metadata lookup."""

from __future__ import annotations

import pytest

from qvartools.molecules.registry import (
    MOLECULE_REGISTRY,
    get_molecule_info,
)


def test_get_molecule_info_has_expected_fields() -> None:
    info = get_molecule_info("h2")
    for key in ("name", "n_qubits", "basis", "geometry", "charge", "spin"):
        assert key in info
    assert info["name"] == "H2"
    assert info["basis"] == "sto-3g"


def test_get_molecule_info_is_case_insensitive() -> None:
    assert get_molecule_info("H2") == get_molecule_info("h2")


def test_get_molecule_info_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="Unknown molecule"):
        get_molecule_info("not-a-molecule")


def test_get_molecule_info_does_not_call_factory(monkeypatch) -> None:
    original_factory = MOLECULE_REGISTRY["h2"]["factory"]

    def _boom(*args, **kwargs):
        raise AssertionError("Factory should not be called by get_molecule_info")

    monkeypatch.setitem(MOLECULE_REGISTRY["h2"], "factory", _boom)
    info = get_molecule_info("h2")
    assert info["name"] == "H2"
    monkeypatch.setitem(MOLECULE_REGISTRY["h2"], "factory", original_factory)
