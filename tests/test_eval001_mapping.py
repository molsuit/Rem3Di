"""Unit tests for the EVAL-001 source→zarr mapping layer.

These functions decide *which molecules get scored*, so a regression here
silently corrupts every benchmark number. The duplicate-preserving disjoint
queue behaviour and the explicit standardisation resolver are the highest-risk
pieces and are covered directly here (no zarr / torch needed).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from threedscriptors.evaluation.eval001.mapping import (
    is_polar_standardisation,
    load_moleculenet_source_to_zarr,
    map_source_indices_to_zarr,
    map_tdc_frames_to_source_indices,
    reconstruct_moleculenet_source_to_zarr,
    reconstruct_tdc_source_to_zarr,
)


class _FakeDataset:
    """Minimal stand-in for MoleculeDataset (avoids needing a real zarr)."""

    def __init__(self, smiles: list[str], targets: np.ndarray):
        self._smiles = smiles
        self.targets_system = targets

    def get_smiles_per_structure(self):
        return self._smiles


def _patch_dataset(monkeypatch, smiles, targets):
    monkeypatch.setattr(
        "threedscriptors.data_handling.dataset.molecule_dataset."
        "MoleculeDataset.open_existing_dataset_from_dir",
        lambda *a, **k: _FakeDataset(smiles, targets),
    )


def test_is_polar_standardisation_explicit_overrides_path():
    # Explicit 'polar' wins even when the path looks like an OFF24 path.
    assert is_polar_standardisation(
        {"standardisation": "polar"},
        auto_path="/data/eval001_moleculenet/x",
        auto_token="moleculenet_polar",
    ) is True
    # Explicit 'off24' wins even when the path contains the polar token.
    assert is_polar_standardisation(
        {"standardisation": "off24"},
        auto_path="/data/eval001_moleculenet_polar/x",
        auto_token="moleculenet_polar",
    ) is False


def test_is_polar_standardisation_auto_falls_back_to_path_inference():
    # 'auto' (and unset) reproduces the legacy path-substring behaviour.
    for cfg in ({}, {"standardisation": "auto"}, {"standardisation": None}):
        assert is_polar_standardisation(
            cfg, auto_path="/d/eval001_moleculenet_polar/x", auto_token="moleculenet_polar"
        ) is True
        assert is_polar_standardisation(
            cfg, auto_path="/d/eval001_moleculenet/x", auto_token="moleculenet_polar"
        ) is False


def test_map_source_indices_to_zarr_keeps_order_and_mask():
    source_to_zarr = {0: 5, 2: 7, 3: 9}
    idx, mask = map_source_indices_to_zarr(np.array([0, 1, 2, 4, 3]), source_to_zarr)
    # Hits returned in input order; mask aligned to the input sequence.
    assert idx.tolist() == [5, 7, 9]
    assert mask.tolist() == [True, False, True, False, True]
    # The post-mask invariant the call sites assert on.
    assert len(idx) == int(mask.sum())


def _src_frame() -> pd.DataFrame:
    # Row 0 and row 2 are an exact duplicate (Drug_ID, Drug, Y).
    return pd.DataFrame(
        {
            "Drug_ID": ["A", "B", "A", "C", "D"],
            "Drug": ["sm1", "sm2", "sm1", "sm3", "sm4"],
            "Y": [1.0, 2.0, 1.0, 3.0, 4.0],
        }
    )


def test_map_tdc_frames_preserves_duplicates_disjointly():
    source = _src_frame()
    train = source.iloc[[0, 3]].reset_index(drop=True)  # A/sm1/1.0, C/sm3/3.0
    val = source.iloc[[2, 1]].reset_index(drop=True)  # A/sm1/1.0 (dup), B/sm2/2.0
    train_idx, val_idx = map_tdc_frames_to_source_indices([train, val], source)
    assert train_idx.tolist() == [0, 3]
    # The second occurrence of A/sm1/1.0 must go to source position 2, NOT reuse 0.
    assert val_idx.tolist() == [2, 1]
    # Disjoint split membership preserved across the duplicated key.
    assert set(train_idx).isdisjoint(set(val_idx))


def test_map_tdc_frames_applies_offset():
    source = _src_frame()
    frame = source.iloc[[1, 4]].reset_index(drop=True)
    (mapped,) = map_tdc_frames_to_source_indices([frame], source, offset=100)
    assert mapped.tolist() == [101, 104]


def test_map_tdc_frames_raises_on_unmappable_row():
    source = _src_frame()
    bogus = pd.DataFrame({"Drug_ID": ["Z"], "Drug": ["zzz"], "Y": [9.9]})
    with pytest.raises(KeyError):
        map_tdc_frames_to_source_indices([bogus], source)


def test_load_moleculenet_source_to_zarr(tmp_path):
    ds_dir = tmp_path / "BACE-S"
    ds_dir.mkdir()
    (ds_dir / "source_index_mapping.json").write_text(
        json.dumps({"source_smiles_to_zarr": {"CCO": 3, "c1ccccc1": 0}})
    )
    out = load_moleculenet_source_to_zarr(tmp_path, "BACE-S")
    assert out == {"CCO": 3, "c1ccccc1": 0}


def test_load_moleculenet_source_to_zarr_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_moleculenet_source_to_zarr(tmp_path, "Missing")
    ds_dir = tmp_path / "Bad"
    ds_dir.mkdir()
    (ds_dir / "source_index_mapping.json").write_text(json.dumps({"wrong_key": {}}))
    with pytest.raises(KeyError):
        load_moleculenet_source_to_zarr(tmp_path, "Bad")


def test_reconstruct_moleculenet_source_to_zarr_first_occurrence(tmp_path):
    zdir = tmp_path / "ESOL" / "zarr"
    zdir.mkdir(parents=True)
    # 'CCO' appears twice in the zarr; first occurrence (index 0) must win.
    (zdir / "isomeric_smiles.txt").write_text("CCO\nc1ccccc1\nCCO\nCCN\n")
    mapping = reconstruct_moleculenet_source_to_zarr(
        tmp_path, "ESOL", raw_smiles=["c1ccccc1", "CCO", "absent"]
    )
    assert mapping == {"c1ccccc1": 1, "CCO": 0}


def test_reconstruct_moleculenet_source_to_zarr_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        reconstruct_moleculenet_source_to_zarr(tmp_path, "Nope", raw_smiles=["CCO"])


def test_reconstruct_tdc_accepts_float32_large_target(tmp_path, monkeypatch):
    """C1 regression: a large-magnitude target whose float32 round-trip error
    exceeds the old hard 1e-3 bound must still match (no task-wide nuke)."""
    config = {"paths": {"tdc_root": str(tmp_path)}}
    source = pd.DataFrame(
        {"Drug_ID": ["a", "b"], "Drug": ["CCO", "c1ccccc1"], "Y": [123456.78, 0.5]}
    )
    # zarr persists targets as float32 — |float32(123456.78) - 123456.78| > 1e-3.
    zarr_targets = np.array([123456.78, 0.5], dtype=np.float32)
    assert abs(float(zarr_targets[0]) - 123456.78) > 1e-3  # old bound would reject
    _patch_dataset(monkeypatch, ["CCO", "c1ccccc1"], zarr_targets)

    mapping = reconstruct_tdc_source_to_zarr(config, "BigTask", source)
    assert mapping == {0: 0, 1: 1}  # both rows mapped, scale-aware tolerance


def test_reconstruct_tdc_rejects_genuinely_different_label(tmp_path, monkeypatch):
    config = {"paths": {"tdc_root": str(tmp_path)}}
    source = pd.DataFrame({"Drug_ID": ["a"], "Drug": ["CCO"], "Y": [1.0]})
    _patch_dataset(monkeypatch, ["CCO"], np.array([2.0], dtype=np.float32))
    # 1/1 unmatched > default 2% tolerance → systemic mismatch raises.
    with pytest.raises(KeyError):
        reconstruct_tdc_source_to_zarr(config, "T", source)


def test_reconstruct_tdc_tolerated_partial_vs_systemic_raise(tmp_path, monkeypatch):
    config = {"paths": {"tdc_root": str(tmp_path)}}
    source = pd.DataFrame({"Drug_ID": ["a"], "Drug": ["CCO"], "Y": [1.0]})
    # zarr has an extra unmatched row (CCN, no source candidate): 1 of 2 missing.
    _patch_dataset(monkeypatch, ["CCO", "CCN"], np.array([1.0, 9.0], dtype=np.float32))

    # Tolerated: drop the unmatched row, return the partial map (honest coverage).
    mapping = reconstruct_tdc_source_to_zarr(
        config, "T", source, max_unmatched_frac=0.5
    )
    assert mapping == {0: 0}
    # Not tolerated: same data, zero tolerance → raises instead of silent remap.
    with pytest.raises(KeyError):
        reconstruct_tdc_source_to_zarr(config, "T", source, max_unmatched_frac=0.0)
