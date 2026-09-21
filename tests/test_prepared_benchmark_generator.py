"""The one benchmark generator: ``conformers``-stage bundle -> ``InputBatch``.

``BENCHMARK_DATA_FORMAT.md`` §2.2 / §2.3. What is pinned here is exactly what
replaced the five source-specific generators: the stage gate, the **verbatim**
id carry-through (the preparer's ids are authoritative, the loop counter is
not), the NaN -> mask re-derivation, the split-name -> code conversion, and the
charge / multiplicity extras.

The bundles are built in memory rather than through ``write_bundle`` so that a
case the format forbids (a ``smiles``-stage bundle reaching the generator) can
still be handed to it — that is the error path under test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ase import Atoms

from remedi.data_handling.bundle import (
    BenchmarkSpec,
    BenchmarkTask,
    Bundle,
    BundleProvenance,
    PreparerRecord,
)
from remedi.data_handling.dataset.tasks import Split, TaskType
from remedi.data_handling.dataset_creation.generators import (
    PreparedBenchmarkGenerator,
)

#: Two enantiomers of butan-2-ol, two conformers each: four rows whose three
#: ids all differ from one another, which is the case a loop counter gets wrong.
_ENANTIOMER_PAIR_SMILES = ["C[C@H](O)CC", "C[C@@H](O)CC"]


def make_spec(
    *,
    stage: str = "conformers",
    tasks: list[BenchmarkTask] | None = None,
    extra_columns: list[str] | None = None,
) -> BenchmarkSpec:
    return BenchmarkSpec(
        dataset_id="toy",
        tasks=tasks
        or [
            BenchmarkTask(name="activity", task_type=TaskType.classification),
            BenchmarkTask(name="logp", task_type=TaskType.regression),
        ],
        metrics=["AUROC"],
        stage=stage,  # type: ignore[arg-type]
        geometry_origin="etkdg_mmff" if stage == "conformers" else None,
        split_columns=["split"],
        default_split="split",
        split_group="molecule_id",
        extra_columns=extra_columns or [],
        source_kind="synthetic",
    )


def make_table(
    spec: BenchmarkSpec,
    n_rows: int = 4,
    extra_columns: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """Four rows: one constitution, two stereoisomers, two conformers each."""
    table = pd.DataFrame(
        {
            "structure_id": np.arange(n_rows, dtype=np.int64),
            "stereoisomer_id": np.arange(n_rows, dtype=np.int64) // 2,
            "molecule_id": np.zeros(n_rows, dtype=np.int64),
            "isomeric_smiles": [
                _ENANTIOMER_PAIR_SMILES[row // 2] for row in range(n_rows)
            ],
            "nonisomeric_smiles": ["CCC(C)O"] * n_rows,
            "enantiomer_of": pd.array([1, 1, 0, 0][:n_rows], dtype="Int64"),
        }
    )
    table["activity"] = np.array([1.0, np.nan, 0.0, 1.0][:n_rows])
    table["logp"] = np.array([0.1, 0.2, np.nan, 0.4][:n_rows])
    table["split"] = ["train", "train", "valid", "test"][:n_rows]
    for column_name, values in (extra_columns or {}).items():
        table[column_name] = values
    return table[spec.expected_columns()]


def make_bundle(
    *,
    stage: str = "conformers",
    with_structures: bool = True,
    extra_columns: dict[str, list[float]] | None = None,
    n_rows: int = 4,
) -> Bundle:
    spec = make_spec(stage=stage, extra_columns=list(extra_columns or {}))
    table = make_table(spec, n_rows=n_rows, extra_columns=extra_columns)
    structures = (
        [
            Atoms(numbers=[6, 1, 1, 1, 1], positions=np.zeros((5, 3)) + row)
            for row in range(n_rows)
        ]
        if with_structures
        else None
    )
    return Bundle(
        spec=spec,
        table=table,
        structures=structures,
        provenance=BundleProvenance(
            dataset_id="toy", preparer=PreparerRecord(repo="tests", script="x.py")
        ),
    )


# ------------------------------------------------------------- the stage gate


def test_a_smiles_stage_bundle_is_refused() -> None:
    with pytest.raises(ValueError, match="only a 'conformers'-stage bundle"):
        PreparedBenchmarkGenerator(make_bundle(stage="smiles", with_structures=False))


def test_a_conformers_bundle_without_structures_is_refused() -> None:
    with pytest.raises(ValueError, match="carries no structures"):
        PreparedBenchmarkGenerator(make_bundle(with_structures=False))


def test_a_frame_count_mismatch_is_refused() -> None:
    bundle = make_bundle()
    assert bundle.structures is not None
    bundle.structures = bundle.structures[:-1]
    with pytest.raises(ValueError, match="3 frames for 4 table rows"):
        PreparedBenchmarkGenerator(bundle)


# ------------------------------------------------------ the id carry-through


def test_the_three_ids_are_copied_verbatim_from_the_table() -> None:
    batches = list(PreparedBenchmarkGenerator(make_bundle(), batch_size=4))

    (batch,) = batches
    assert [sid.structure_id for sid in batch.structure_ids] == [0, 1, 2, 3]
    assert [sid.stereoisomer_id for sid in batch.structure_ids] == [0, 0, 1, 1]
    assert [sid.molecule_id for sid in batch.structure_ids] == [0, 0, 0, 0]


def test_structure_ids_continue_across_batches_rather_than_restarting() -> None:
    # The id must come from the column, never from the per-batch loop counter.
    first, second = list(PreparedBenchmarkGenerator(make_bundle(), batch_size=2))

    assert [sid.structure_id for sid in first.structure_ids] == [0, 1]
    assert [sid.structure_id for sid in second.structure_ids] == [2, 3]
    assert [sid.stereoisomer_id for sid in second.structure_ids] == [1, 1]


def test_the_smiles_pair_rides_along_per_row() -> None:
    (batch,) = list(PreparedBenchmarkGenerator(make_bundle(), batch_size=4))

    assert batch.smiles is not None
    assert [data.isomeric_smiles for data in batch.smiles] == [
        _ENANTIOMER_PAIR_SMILES[0],
        _ENANTIOMER_PAIR_SMILES[0],
        _ENANTIOMER_PAIR_SMILES[1],
        _ENANTIOMER_PAIR_SMILES[1],
    ]
    assert {data.nonisomeric_smiles for data in batch.smiles} == {"CCC(C)O"}


# ----------------------------------------------------------- targets + masks


def test_the_mask_is_exactly_the_non_null_pattern_of_the_table() -> None:
    (batch,) = list(PreparedBenchmarkGenerator(make_bundle(), batch_size=4))

    assert batch.regression_data is not None
    targets = batch.regression_data.targets_system
    mask = batch.regression_data.mask_system
    assert targets is not None and mask is not None
    np.testing.assert_array_equal(mask, (~np.isnan(targets)).astype(np.uint8))
    np.testing.assert_array_equal(
        mask, np.array([[1, 1], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    )


def test_split_names_become_split_codes() -> None:
    (batch,) = list(PreparedBenchmarkGenerator(make_bundle(), batch_size=4))

    assert batch.regression_data is not None
    np.testing.assert_array_equal(
        batch.regression_data.split,
        np.array(
            [
                Split.train.value,
                Split.train.value,
                Split.valid.value,
                Split.test.value,
            ],
            dtype=np.uint8,
        ),
    )


def test_an_unknown_split_name_is_refused_by_name() -> None:
    bundle = make_bundle()
    bundle.table["split"] = ["train", "train", "valid", "holdout"]
    with pytest.raises(ValueError, match="holdout"):
        PreparedBenchmarkGenerator(bundle)


# ------------------------------------------------- charge and multiplicity


def test_charge_and_multiplicity_default_to_a_neutral_singlet() -> None:
    (batch,) = list(PreparedBenchmarkGenerator(make_bundle(), batch_size=4))

    assert batch.total_charge == [0.0, 0.0, 0.0, 0.0]
    # Not 0.0: the writer's own default is unphysical, so the generator never
    # leaves the field for it to fill.
    assert batch.multiplicity == [1.0, 1.0, 1.0, 1.0]


def test_charge_and_multiplicity_come_from_the_extra_columns() -> None:
    bundle = make_bundle(
        extra_columns={
            "total_charge": [-1.0, 0.0, 1.0, 0.0],
            "multiplicity": [1.0, 2.0, 1.0, 3.0],
        }
    )

    (batch,) = list(PreparedBenchmarkGenerator(bundle, batch_size=4))

    assert batch.total_charge == [-1.0, 0.0, 1.0, 0.0]
    assert batch.multiplicity == [1.0, 2.0, 1.0, 3.0]


# ------------------------------------------------------------------ batching


def test_a_row_count_that_is_not_a_multiple_of_the_batch_size() -> None:
    generator = PreparedBenchmarkGenerator(make_bundle(n_rows=3), batch_size=2)

    batches = list(generator)

    assert [len(batch) for batch in batches] == [2, 1]
    assert [sid.structure_id for batch in batches for sid in batch.structure_ids] == [
        0,
        1,
        2,
    ]
    assert generator.load_stats.n_raw_rows == generator.load_stats.n_kept == 3


def test_the_task_set_is_the_specs_system_columns() -> None:
    task_set = PreparedBenchmarkGenerator(make_bundle()).task_set()

    assert [column.name for column in task_set.system_cols] == ["activity", "logp"]
    assert task_set.atom_cols == []
