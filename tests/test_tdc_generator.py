"""TdcGenerator: official PyTDC split materialized, dedupe (first split wins).

PyTDC's network download is mocked: `_split_frames` is the only PyTDC touch
point, so stubbing it exercises the canonicalize/filter/dedupe/split-code
logic we own without fetching the leaderboard cache.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from threedscriptors.data_handling.benchmarks import get_benchmark
from threedscriptors.data_handling.dataset.tasks import Split
from threedscriptors.data_handling.dataset_creation.generators.tdc_generator import (
    TdcGenerator,
)


def _frames():
    train = pd.DataFrame({"Drug": ["CCO", "c1ccccc1"], "Y": [1.0, 2.0]})
    valid = pd.DataFrame({"Drug": ["CCN"], "Y": [3.0]})
    # CCO duplicates train (train must win); not_a_smiles + methane dropped.
    test = pd.DataFrame(
        {"Drug": ["CC(=O)O", "CCO", "not_a_smiles", "C"], "Y": [4.0, 9.0, 5.0, 9.0]}
    )
    return [
        (train, Split.train),
        (valid, Split.valid),
        (test, Split.test),
    ]


def test_official_split_materialized_and_dedup(monkeypatch):
    bench = get_benchmark("LD50_Zhu")  # single-task regression TdcBenchmark
    gen = TdcGenerator(bench, tdc_cache="/unused", batch_size=2)
    monkeypatch.setattr(gen, "_split_frames", _frames)

    batches = list(gen)
    smiles = [s.isomeric_smiles for b in batches for s in b.smiles]
    targets = np.concatenate([b.regression_data.targets_system for b in batches])
    split = np.concatenate([b.regression_data.split for b in batches])
    mask = np.concatenate([b.regression_data.mask_system for b in batches])

    # CCO, benzene, CCN, acetic acid — dup CCO/invalid/methane gone.
    assert len(smiles) == 4
    assert len(set(smiles)) == 4
    assert all(b.molecules is None for b in batches)  # conformer made later

    # First occurrence wins: CCO stays train (0), not test (2).
    code_by_smiles = dict(zip(smiles, split.tolist(), strict=True))
    cco = next(s for s in smiles if s == "CCO")
    assert code_by_smiles[cco] == Split.train.value
    assert set(split.tolist()) == {
        Split.train.value,
        Split.valid.value,
        Split.test.value,
    }
    assert split.dtype == np.uint8
    assert targets.shape == (4, 1)
    np.testing.assert_array_equal(mask.reshape(-1), np.ones(4, dtype=np.uint8))


def test_empty_dataset_yields_nothing(monkeypatch):
    gen = TdcGenerator(get_benchmark("LD50_Zhu"), tdc_cache="/unused")
    monkeypatch.setattr(
        gen,
        "_split_frames",
        lambda: [(pd.DataFrame({"Drug": [], "Y": []}), Split.train)],
    )
    assert list(gen) == []


def test_strip_salts_recovers_drug_half(monkeypatch):
    # Multi-fragment salt in the official train fold; without strip_salts
    # this row would be lost to the single-fragment gate.
    gen = TdcGenerator(get_benchmark("LD50_Zhu"), tdc_cache="/unused")
    monkeypatch.setattr(
        gen,
        "_split_frames",
        lambda: [
            (
                pd.DataFrame({"Drug": ["Oc1ccccc1.[Cl-]"], "Y": [1.0]}),
                Split.train,
            )
        ],
    )
    (batch,) = list(gen)
    assert [s.isomeric_smiles for s in batch.smiles] == ["Oc1ccccc1"]


def test_load_stats_count_each_rejection_class(monkeypatch):
    gen = TdcGenerator(get_benchmark("LD50_Zhu"), tdc_cache="/unused")
    monkeypatch.setattr(gen, "_split_frames", _frames)
    list(gen)

    s = gen.load_stats
    # _frames totals: 7 rows = 4 unique valid kept + 1 dup CCO + 1 invalid + 1 methane.
    assert s.n_raw_rows == 7
    assert s.n_invalid_smiles == 1
    assert s.n_filtered_out == 1
    assert s.n_duplicates == 1
    assert s.n_kept == 4
    assert (
        s.n_kept + s.n_invalid_smiles + s.n_filtered_out + s.n_duplicates
        == s.n_raw_rows
    )


def test_no_strip_drops_multi_fragment_salt(monkeypatch):
    gen = TdcGenerator(
        get_benchmark("LD50_Zhu"),
        tdc_cache="/unused",
        strip_salts=False,
        neutralize=False,
    )
    monkeypatch.setattr(
        gen,
        "_split_frames",
        lambda: [
            (
                pd.DataFrame(
                    {"Drug": ["Oc1ccccc1.[Cl-]", "CCC"], "Y": [1.0, 2.0]}
                ),
                Split.train,
            )
        ],
    )
    (batch,) = list(gen)
    assert [s.isomeric_smiles for s in batch.smiles] == ["CCC"]
