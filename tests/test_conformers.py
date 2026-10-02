"""Conformer embedding: ``embed_one_smiles``, its timing records, and ``embed_many``."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from remedi.data_handling.chemistry import conformers as conformers_module
from remedi.data_handling.chemistry.conformers import (
    ConformerEmbeddingConfig,
    ConformerTimingRecord,
    EmbedResult,
    embed_many,
    embed_one_smiles,
    write_timings_jsonl,
)

#: The SMILES the crashing fake below kills its worker on.
CRASHING_SMILES = "CCCCCCO"
REAL_EMBED_ONE_SMILES = embed_one_smiles


def test_embed_one_smiles_ok_returns_timing_record():
    result = embed_one_smiles("CCO", ConformerEmbeddingConfig())

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
    result = embed_one_smiles("not_a_real_smiles", ConformerEmbeddingConfig())

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


def test_embed_many_in_process_yields_every_key() -> None:
    smiles_by_key = {"ethanol": "CCO", "bad": "not_a_real_smiles", "propane": "CCC"}

    results = dict(embed_many(smiles_by_key, ConformerEmbeddingConfig(), n_workers=1))

    assert set(results) == set(smiles_by_key)
    assert results["ethanol"].succeeded
    assert results["propane"].succeeded
    assert results["bad"].timing.status == "value_error"


def test_embed_many_in_a_pool_matches_the_in_process_outcome() -> None:
    smiles_by_key = {
        index: smiles for index, smiles in enumerate(["CCO", "CCN", "CC=O"])
    }

    results = dict(embed_many(smiles_by_key, ConformerEmbeddingConfig(), n_workers=2))

    assert set(results) == set(smiles_by_key)
    assert all(result.succeeded for result in results.values())


def kill_the_worker_on_one_molecule(
    isomeric_smiles: str, config: ConformerEmbeddingConfig
) -> EmbedResult:
    """Stand-in for a segfault inside RDKit: the worker process just dies."""
    if isomeric_smiles == CRASHING_SMILES:
        os._exit(1)
    return REAL_EMBED_ONE_SMILES(isomeric_smiles, config)


def test_embed_many_survives_a_worker_that_dies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One molecule killing its worker must not lose the others (open thread 5).

    Relies on the fork start method, so the forked workers inherit the patch.
    """
    monkeypatch.setattr(
        conformers_module, "embed_one_smiles", kill_the_worker_on_one_molecule
    )
    smiles = ["CCO", "CCN", CRASHING_SMILES, "CCC", "CC=O", "CCCl"]
    smiles_by_key = dict(enumerate(smiles))

    results = dict(embed_many(smiles_by_key, ConformerEmbeddingConfig(), n_workers=2))

    assert set(results) == set(smiles_by_key)
    crashed = smiles.index(CRASHING_SMILES)
    assert results[crashed].timing.status == "worker_crashed"
    assert all(result.succeeded for key, result in results.items() if key != crashed)
