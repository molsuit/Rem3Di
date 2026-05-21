"""MoleculeNetGenerator: canonicalize/filter/dedupe + materialized split.

MoleculeNet still does its own filter (via ``apply_smiles_filter``) so that
the deterministic DeepChem scaffold split sees the same kept set the rest of
the pipeline writes. Filter knobs ride on a ``FilterMoleculeStageConfig``.
"""

from __future__ import annotations

import numpy as np

from threedscriptors.configuration.dataset_config import FilterMoleculeStageConfig
from threedscriptors.data_handling.benchmarks import (
    BenchmarkTask,
    MoleculeNetBenchmark,
)
from threedscriptors.data_handling.dataset.tasks import ElementSet, Split, TaskType
from threedscriptors.data_handling.dataset_creation.generators.moleculenet_generator import (
    MoleculeNetGenerator,
)


def _write_csv(path):
    # ethanol, benzene, acetic acid, ethylamine, a duplicate ethanol,
    # an unparseable SMILES, and methane (<3 atoms -> filtered).
    rows = [
        "smiles,y",
        "CCO,0.1",
        "c1ccccc1,0.2",
        "CC(=O)O,0.3",
        "CCN,0.4",
        "CCO,0.1",
        "not_a_smiles,0.5",
        "C,0.9",
    ]
    path.write_text("\n".join(rows) + "\n")


def _benchmark() -> MoleculeNetBenchmark:
    return MoleculeNetBenchmark(
        dataset_id="toy",
        csv_name="toy.csv",
        smiles_column="smiles",
        tasks=[BenchmarkTask(name="y", task_type=TaskType.regression)],
        metric="RMSE",
    )


def _filter(**overrides) -> FilterMoleculeStageConfig:
    return FilterMoleculeStageConfig(**overrides)


def test_generator_filters_dedupes_and_splits(tmp_path):
    _write_csv(tmp_path / "toy.csv")
    gen = MoleculeNetGenerator(_benchmark(), tmp_path, batch_size=2)

    batches = list(gen)
    smiles = [s.isomeric_smiles for b in batches for s in b.smiles]
    targets = np.concatenate([b.regression_data.targets_system for b in batches])
    split = np.concatenate([b.regression_data.split for b in batches])

    # 4 unique valid mols kept (dup ethanol collapsed; invalid + methane dropped)
    assert len(smiles) == 4
    assert len(set(smiles)) == 4
    assert targets.shape == (4, 1)

    # Every kept row is assigned a real partition (no unassigned).
    valid_codes = {Split.train.value, Split.valid.value, Split.test.value}
    assert set(split.tolist()).issubset(valid_codes)
    assert (split != Split.unassigned.value).all()
    assert split.dtype == np.uint8


def test_generator_mask_marks_missing_targets(tmp_path):
    (tmp_path / "toy.csv").write_text("smiles,y\nCCO,1.0\nCCN,\nc1ccccc1,3.0\n")
    gen = MoleculeNetGenerator(_benchmark(), tmp_path, batch_size=10)
    (batch,) = list(gen)
    mask = batch.regression_data.mask_system.reshape(-1)
    # The middle row has an empty target -> mask 0, others 1.
    np.testing.assert_array_equal(mask, np.array([1, 0, 1], dtype=np.uint8))


# Salt-form drug rows: ``Oc1ccccc1.[Cl-]`` is a multi-fragment SMILES with the
# neutral aromatic drug + a chloride counter-ion. Without salt-stripping the
# single-fragment gate in filter_mol drops the whole row; with strip_salts on
# the drug half (phenol) is recovered and ingested.
def test_strip_salts_recovers_drug_half(tmp_path):
    (tmp_path / "toy.csv").write_text(
        "smiles,y\nOc1ccccc1.[Cl-],1.5\nCCC,2.5\n"
    )
    gen = MoleculeNetGenerator(
        _benchmark(),
        tmp_path,
        batch_size=10,
        filter_config=_filter(strip_salts=True, neutralize=True),
    )
    (batch,) = list(gen)
    smiles = [s.isomeric_smiles for s in batch.smiles]
    assert "Oc1ccccc1" in smiles
    assert len(smiles) == 2


def test_no_strip_drops_multi_fragment_salt(tmp_path):
    (tmp_path / "toy.csv").write_text(
        "smiles,y\nOc1ccccc1.[Cl-],1.5\nCCC,2.5\n"
    )
    gen = MoleculeNetGenerator(
        _benchmark(),
        tmp_path,
        batch_size=10,
        filter_config=_filter(strip_salts=False, neutralize=False),
    )
    (batch,) = list(gen)
    smiles = [s.isomeric_smiles for s in batch.smiles]
    # Multi-fragment row rejected by allow_multifragment=False inside filter_mol.
    assert smiles == ["CCC"]


def test_neutralize_collapses_charged_form(tmp_path):
    # Acetate anion (single fragment, charge=-1) -> acetic acid when uncharged.
    (tmp_path / "toy.csv").write_text("smiles,y\nCC(=O)[O-],1.0\n")
    gen = MoleculeNetGenerator(
        _benchmark(),
        tmp_path,
        batch_size=10,
        filter_config=_filter(strip_salts=False, neutralize=True),
    )
    (batch,) = list(gen)
    assert [s.isomeric_smiles for s in batch.smiles] == ["CC(=O)O"]


def test_load_stats_count_each_rejection_class(tmp_path):
    _write_csv(tmp_path / "toy.csv")
    gen = MoleculeNetGenerator(_benchmark(), tmp_path, batch_size=10)
    list(gen)  # exhaust to populate load_stats

    s = gen.load_stats
    # _write_csv inserts: 4 valid uniques, 1 dup (CCO), 1 unparseable, 1 too-small.
    assert s.n_raw_rows == 7
    assert s.n_invalid_smiles == 1
    assert s.n_filtered_out == 1
    assert s.n_duplicates == 1
    assert s.n_kept == 4
    assert (
        s.n_kept + s.n_invalid_smiles + s.n_filtered_out + s.n_duplicates
        == s.n_raw_rows
    )


def test_element_set_mace_polar_accepts_silicon(tmp_path):
    # Ethylsilane (3 heavy atoms clears filter_mol's >=3 gate): Si is in
    # MACE_POLAR but not MACE_OFF.
    (tmp_path / "toy.csv").write_text("smiles,y\nCC[SiH3],1.0\nCCC,2.0\n")

    gen_off = MoleculeNetGenerator(
        _benchmark(),
        tmp_path,
        batch_size=10,
        filter_config=_filter(element_set=ElementSet.mace_off),
    )
    (batch_off,) = list(gen_off)
    assert [s.isomeric_smiles for s in batch_off.smiles] == ["CCC"]

    gen_polar = MoleculeNetGenerator(
        _benchmark(),
        tmp_path,
        batch_size=10,
        filter_config=_filter(element_set=ElementSet.mace_polar),
    )
    smiles_polar = [s.isomeric_smiles for b in gen_polar for s in b.smiles]
    assert any("Si" in s for s in smiles_polar)
    assert len(smiles_polar) == 2
