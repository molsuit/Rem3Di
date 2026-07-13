"""Smoke tests for ConformerTimingRecord + embed_one_smiles instrumentation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from remedi.data_handling.dataset_creation.conformer_timing import (
    ConformerTimingRecord,
    write_timings_jsonl,
)
from remedi.data_handling.dataset_creation.utils import embed_one_smiles


def test_embed_one_smiles_ok_returns_timing_record():
    result = embed_one_smiles(
        isomeric_smiles="CCO",
        n_confs=1,
        max_embed_attempts=200,
        max_opt_iters=100,
    )

    assert result.timing.status == "ok"
    assert result.positions is not None
    assert result.atomic_numbers is not None
    assert result.positions.shape == (1, result.atomic_numbers.shape[0], 3)
    assert result.positions.dtype == np.float64
    assert result.timing.n_confs_emitted == 1
    assert result.timing.n_confs_requested == 1
    # Ethanol embedding + MMFF is fast but never instant on real hardware.
    assert result.timing.t_embed_s >= 0.0
    assert result.timing.t_mmff_s >= 0.0
    assert result.timing.n_atoms > 0
    assert result.timing.error_msg is None


def test_embed_one_smiles_bad_smiles_records_value_error():
    result = embed_one_smiles(
        isomeric_smiles="not_a_real_smiles",
        n_confs=1,
        max_embed_attempts=200,
        max_opt_iters=100,
    )

    assert result.timing.status == "value_error"
    assert result.positions is None
    assert result.atomic_numbers is None
    assert result.timing.n_confs_emitted == 0
    assert result.timing.error_msg is not None
    # Parse failure happens before any phase runs.
    assert result.timing.t_embed_s == 0.0
    assert result.timing.t_mmff_s == 0.0


def test_write_timings_jsonl_roundtrip(tmp_path: Path):
    records = [
        ConformerTimingRecord(
            isomeric_smiles="CCO",
            n_atoms=9,
            n_confs_requested=1,
            n_confs_emitted=1,
            t_embed_s=0.012,
            t_mmff_s=0.034,
            status="ok",
        ),
        ConformerTimingRecord(
            isomeric_smiles="not_a_real_smiles",
            n_atoms=-1,
            n_confs_requested=1,
            n_confs_emitted=0,
            t_embed_s=0.0,
            t_mmff_s=0.0,
            status="value_error",
            error_msg="Bad SMILES",
        ),
    ]
    out = tmp_path / "conformer_timings.jsonl"
    write_timings_jsonl(records, out)

    lines = out.read_text().splitlines()
    assert len(lines) == 2
    reloaded = [
        ConformerTimingRecord.model_validate(json.loads(line)) for line in lines
    ]
    assert reloaded[0].status == "ok"
    assert reloaded[0].t_mmff_s == pytest.approx(0.034)
    assert reloaded[1].status == "value_error"
    assert reloaded[1].error_msg == "Bad SMILES"
