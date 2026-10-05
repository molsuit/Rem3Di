"""Round trip: preparer -> bundle -> ``build_datasets`` -> zarr -> ``MoleculeDataset``.

``BENCHMARK_DATA_FORMAT.md`` §8 step 5. A bundle written by the shared SMILES
preparer path goes through the real dataset build (conformer embedding
included), and every structure of the resulting dataset must carry exactly the
ids, ``enantiomer_of``, labels (missing ones included) and split values of the
bundle row it was built from.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from remedi.data_handling.bundle import (
    SourceRecord,
    merge_source_rows,
    read_bundle,
)
from remedi.data_handling.bundle.preparation import (
    PreparerSettings,
    SourceLabel,
    smiles_bundle_spec,
    write_smiles_bundle,
)
from remedi.data_handling.bundle.provenance import PreparerRecord
from remedi.data_handling.bundle.spec import EvalMetric
from remedi.data_handling.chemistry.splits import fixed_test_seeded_split_columns
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.data_handling.dataset.tasks import Split, TaskType
from remedi.data_handling.dataset_build import DatasetBuildConfig, build_datasets

DATASET_ID = "round_trip"
LABELS = ("y", "z")
SPLIT_CODES = {split.name: split.value for split in Split}

#: Source rows: a replicate (ethanol twice), an enantiomer pair (alanine), a
#: molecule the filter drops (an isotope), and a sparse second label.
SOURCE_ROWS: list[tuple[str, float, float]] = [
    ("CCO", 1.0, math.nan),
    ("CCCO", 2.0, 0.5),
    ("OCC", 3.0, math.nan),
    ("C[C@@H](N)C(=O)O", 4.0, 1.5),
    ("C[C@H](N)C(=O)O", 5.0, math.nan),
    ("[13CH3]CO", 6.0, 2.0),
    ("CCCCO", 7.0, 2.5),
    ("c1ccccc1O", 8.0, math.nan),
    ("Cc1ccccc1", 9.0, 3.5),
    ("CC(=O)Nc1ccccc1", 10.0, 4.0),
]


class RoundTripSettings(PreparerSettings):
    raw_root: Path = Path("unused")
    bundle_root: Path = Path("bundles")

    def dataset_ids(self) -> list[str]:
        return [DATASET_ID]


@pytest.fixture(scope="module")
def round_trip(tmp_path_factory: pytest.TempPathFactory) -> tuple[pd.DataFrame, Path]:
    """Prepare the bundle, build the dataset; return the bundle table and dataset dir."""
    root = tmp_path_factory.mktemp("round_trip")
    settings = RoundTripSettings(bundle_root=root / "bundles", seeds=[1, 2])
    merged = merge_source_rows(
        [smiles for smiles, *_ in SOURCE_ROWS],
        {
            name: [row[position + 1] for row in SOURCE_ROWS]
            for position, name in enumerate(LABELS)
        },
        dict.fromkeys(LABELS, TaskType.regression),
        settings.smiles_filter,
    )
    is_test = np.zeros(len(merged), dtype=bool)
    is_test[-2:] = True
    write_smiles_bundle(
        spec=smiles_bundle_spec(
            dataset_id=DATASET_ID,
            description="round-trip test bundle",
            labels=[
                SourceLabel(name=name).label_column(TaskType.regression)
                for name in LABELS
            ],
            metrics=[EvalMetric.mae],
            split_columns=settings.split_columns(),
            source_kind="test",
        ),
        merged=merged,
        split_values=fixed_test_seeded_split_columns(
            merged.isomeric_smiles, is_test, list(settings.seeds), valid_fraction=0.25
        ),
        settings=settings,
        preparer=PreparerRecord(repo="molsuit/Rem3Di", script="test"),
        source=SourceRecord(),
    )
    report = build_datasets(
        DatasetBuildConfig(
            bundle_root=root / "bundles",
            zarr_root=root / "datasets",
            dataset_ids=[DATASET_ID],
            n_workers=1,
        )
    )
    assert report.n_failed == 0
    bundle_table = read_bundle(root / "bundles" / DATASET_ID).table
    return bundle_table, root / "datasets" / DATASET_ID


def test_the_bundle_merged_the_replicate_and_dropped_the_isotope(
    round_trip: tuple[pd.DataFrame, Path],
) -> None:
    bundle_table, _ = round_trip
    assert len(bundle_table) == 8
    ethanol = bundle_table[bundle_table["isomeric_smiles"] == "CCO"]
    assert ethanol["y"].tolist() == [2.0] and ethanol["n_measurements"].tolist() == [
        2.0
    ]
    assert bundle_table["enantiomer_of"].notna().sum() == 2


def test_every_structure_carries_its_bundle_row(
    round_trip: tuple[pd.DataFrame, Path],
) -> None:
    bundle_table, directory = round_trip
    dataset_table = read_bundle(directory).table
    assert len(dataset_table) == len(bundle_table)  # one conformer, none failed
    by_stereoisomer = bundle_table.set_index("stereoisomer_id")
    carried = [
        "molecule_id",
        "enantiomer_of",
        "isomeric_smiles",
        "nonisomeric_smiles",
        *LABELS,
        "split",
        "split__seed1",
        "split__seed2",
        "n_measurements",
    ]
    expected = by_stereoisomer.loc[dataset_table["stereoisomer_id"], carried]
    pd.testing.assert_frame_equal(
        dataset_table[carried].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )


def test_the_zarr_matches_the_table_row_for_row(
    round_trip: tuple[pd.DataFrame, Path],
) -> None:
    _, directory = round_trip
    table = read_bundle(directory).table
    dataset = MoleculeDataset.open_existing_dataset_from_dir(directory)
    try:
        assert np.array_equal(dataset.structure_ids[:], table["structure_id"])
        assert np.array_equal(dataset.isomer_ids[:], table["stereoisomer_id"])
        assert np.array_equal(dataset.molecule_ids[:], table["molecule_id"])
        labels = table[list(LABELS)].to_numpy(dtype=np.float64)
        present = ~np.isnan(labels)
        assert np.array_equal(np.asarray(dataset.mask_system[:]).astype(bool), present)
        targets = np.asarray(dataset.targets_system[:], dtype=np.float64)
        np.testing.assert_allclose(targets[present], labels[present], rtol=1e-6)
        assert np.asarray(dataset.split[:]).tolist() == [
            SPLIT_CODES[value] for value in table["split"]
        ]
        frames = dataset.get_all_molecules()
        assert len(frames) == len(table)
        assert dataset.get_smiles_per_structure() == table["isomeric_smiles"].tolist()
    finally:
        dataset.close()
