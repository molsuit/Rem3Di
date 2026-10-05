"""Conformer embedding: ``embed_one_smiles``, its timing records, and ``embed_many``.

Writing the timing records is exercised by the build tests in
``test_dataset_build.py``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from remedi.data_handling.chemistry import conformers as conformers_module
from remedi.data_handling.chemistry.conformers import (
    ConformerEmbeddingConfig,
    EmbedResult,
    embed_many,
    embed_one_smiles,
)

#: The SMILES the crashing fake below kills its worker on.
CRASHING_SMILES = "CCCCCCO"
REAL_EMBED_ONE_SMILES = embed_one_smiles


def test_embed_many_in_process_records_a_timing_for_every_key() -> None:
    smiles_by_key = {"ethanol": "CCO", "bad": "not_a_real_smiles"}

    results = dict(embed_many(smiles_by_key, ConformerEmbeddingConfig(), n_workers=1))

    assert set(results) == set(smiles_by_key)
    ethanol = results["ethanol"]
    assert ethanol.succeeded and ethanol.timing.status == "ok"
    assert ethanol.positions is not None and ethanol.atomic_numbers is not None
    assert ethanol.positions.shape == (1, ethanol.atomic_numbers.shape[0], 3)
    assert ethanol.positions.dtype == np.float64
    assert ethanol.timing.n_confs_emitted == ethanol.timing.n_confs_requested == 1
    assert ethanol.timing.n_atoms == 9
    assert ethanol.timing.error_msg is None

    bad = results["bad"]
    assert bad.timing.status == "value_error"
    assert bad.positions is None and bad.atomic_numbers is None
    assert bad.timing.n_confs_emitted == 0
    assert bad.timing.error_msg is not None
    # parse failure happens before any phase runs
    assert bad.timing.t_embed_s == bad.timing.t_mmff_s == 0.0


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

    The pool bisects the unfinished molecules until the crashing one is alone;
    every other molecule comes back embedded, as it would in-process. Relies on
    the fork start method, so the forked workers inherit the patch.
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
