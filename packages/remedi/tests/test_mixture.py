"""Pretraining mixtures: selections, domain checks, deduplication and the split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pydantic_yaml
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from remedi.configuration.training_config import (
    ClassifierTrainingConfig,
    PretrainingConfig,
)
from remedi.training.data.mixture import (
    RECORD_NAME,
    DatasetComponent,
    DatasetGroupComponent,
    MixtureError,
    MoleculeFractionValidation,
    PretrainingDataConfig,
    _check_domain,
    resolve_mixture,
)
from remedi.training.data.sample import yield_molecules_collate_fn
from remedi.training.data.training_dataset import (
    TrainingMoleculeDataset,
    atoms_getitem,
)
from remedi_data.bundle import LabelColumn
from remedi_data.dataset.tasks import TaskType
from remedi_data.selection import RandomMolecules, SelectionConfig, select_structures
from remedi_data.testing import write_small_dataset

FIRST = [
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "CCN",
    "OC",
    "CC",
    "CCC",
    "CCCC",
    "CCCCC",
    "CCCCCC",
    "CCCCCCC",
    "CCCCCCCC",
]
# Three molecules in common with FIRST.
SECOND = ["CCO", "CCN", "CCC", "c1ccncc1", "CCOC", "CCCO", "CCCN", "CC(C)C"]


def corpus(root: Path, dataset_id: str, smiles: list[str], **labels) -> Path:
    names = list(labels)
    return write_small_dataset(
        root,
        dataset_id=dataset_id,
        labels=[LabelColumn(name=n, task_type=TaskType.regression) for n in names],
        metrics=None,
        targets=np.column_stack([labels[n] for n in names]) if names else None,
        smiles=smiles,
    )


def mixture(*components, fraction: float = 0.3, **options) -> PretrainingDataConfig:
    return PretrainingDataConfig(
        domain="drug_like",
        components=list(components),
        validation=MoleculeFractionValidation(fraction=fraction, seed=0),
        **options,
    )


def smiles_of(dataset: Path, structure_ids: np.ndarray) -> list[str]:
    table = pl.read_parquet(dataset / "table.parquet")
    return table.filter(
        pl.col("structure_id").is_in(pl.Series(structure_ids).implode())
    )["nonisomeric_smiles"].to_list()


def test_a_component_without_kind_is_a_dataset() -> None:
    config = PretrainingDataConfig.model_validate(
        {
            "domain": "drug_like",
            "components": [
                {"dataset": "/x/pcqm4mv2", "selection": "random_100k"},
                {"kind": "dataset_group", "root": "/x/benchmarks"},
            ],
            "validation": {"kind": "molecule_fraction", "fraction": 0.02},
        }
    )
    assert isinstance(config.components[0], DatasetComponent)
    assert isinstance(config.components[1], DatasetGroupComponent)
    assert config.components[0].name == "pcqm4mv2/random_100k"


def test_duplicates_are_kept_in_the_first_component_only(tmp_path: Path) -> None:
    first = corpus(tmp_path, "first", FIRST)
    second = corpus(tmp_path, "second", SECOND)
    data = resolve_mixture(
        mixture(DatasetComponent(dataset=first), DatasetComponent(dataset=second))
    )
    first_part, second_part = (c.parts[0] for c in data.components)
    assert len(first_part.train_structure_ids) + len(
        first_part.validation_structure_ids
    ) == len(FIRST)
    second_smiles = smiles_of(
        second,
        np.concatenate(
            [second_part.train_structure_ids, second_part.validation_structure_ids]
        ),
    )
    assert sorted(second_smiles) == sorted(set(SECOND) - set(FIRST))
    record = data.record.components[1].parts[0]
    assert record.duplicate_molecules_dropped == 3
    assert data.record.train_structures + data.record.validation_structures == len(
        set(FIRST) | set(SECOND)
    )

    kept = resolve_mixture(
        mixture(
            DatasetComponent(dataset=first),
            DatasetComponent(dataset=second),
            deduplicate_across_components="none",
        )
    )
    assert kept.record.components[1].parts[0].selected_structures == len(SECOND)
    assert kept.record.components[1].parts[0].duplicate_molecules_dropped == 0


def test_the_validation_split_is_by_molecule_and_nested(tmp_path: Path) -> None:
    dataset = corpus(tmp_path, "first", FIRST)
    select_structures(
        dataset,
        SelectionConfig(name="half", steps=[RandomMolecules(count=6, seed=0)]),
    )
    full = resolve_mixture(mixture(DatasetComponent(dataset=dataset), fraction=0.4))
    half = resolve_mixture(
        mixture(DatasetComponent(dataset=dataset, selection="half"), fraction=0.4)
    )
    full_part, half_part = full.components[0].parts[0], half.components[0].parts[0]
    assert not set(full_part.train_structure_ids) & set(
        full_part.validation_structure_ids
    )
    assert len(full_part.validation_structure_ids) > 0
    half_ids = set(half_part.train_structure_ids) | set(
        half_part.validation_structure_ids
    )
    assert len(half_ids) == 6
    # A structure of the smaller selection is held out exactly when it is held
    # out in the full dataset.
    assert set(half_part.validation_structure_ids) == (
        set(full_part.validation_structure_ids) & half_ids
    )
    assert half.record.components[0].parts[0].selection_sha256 is not None


def test_the_training_dataset_concatenates_the_components(tmp_path: Path) -> None:
    first = corpus(tmp_path, "first", FIRST)
    second = corpus(tmp_path, "second", SECOND)
    data = resolve_mixture(
        mixture(DatasetComponent(dataset=first), DatasetComponent(dataset=second))
    )
    datasets = data.open_datasets(atoms_getitem)
    training = datasets.training
    assert len(training) == data.record.train_structures
    counts = datasets.training_atom_counts
    assert len(counts) == len(training)
    expected = [
        TrainingMoleculeDataset(part.dataset, atoms_getitem)[int(structure_id)]
        for component in data.components
        for part in component.parts
        for structure_id in part.train_structure_ids
    ]
    for index in range(len(training)):
        sample = training[index]
        assert sample.atomic_numbers is not None
        assert len(sample.atomic_numbers) == counts[index]
        torch.testing.assert_close(
            sample.atomic_positions, expected[index].atomic_positions, rtol=0, atol=0
        )
    batches = list(
        DataLoader(
            training,
            batch_size=4,
            num_workers=2,
            collate_fn=yield_molecules_collate_fn,
        )
    )
    assert sum(len(batch.total_charge) for batch in batches) == len(training)

    validation = datasets.validation
    assert set(validation) <= {"first", "second"}
    assert sum(len(lengths) for _, lengths in validation.values()) == (
        data.record.validation_structures
    )
    record = yaml.safe_load(data.write_record(tmp_path).read_text())
    assert record["config"]["domain"] == "drug_like"
    assert (tmp_path / RECORD_NAME).is_file()


def test_a_dataset_group_takes_every_dataset_under_the_root(tmp_path: Path) -> None:
    pool = tmp_path / "pool"
    corpus(pool, "a", FIRST[:6])
    corpus(pool, "b", FIRST[3:])
    data = resolve_mixture(
        mixture(DatasetGroupComponent(kind="dataset_group", root=pool))
    )
    assert [c.name for c in data.components] == ["pool"]
    assert len(data.components[0].parts) == 2
    assert data.record.train_structures + data.record.validation_structures == len(
        FIRST
    )


def test_a_drug_like_mixture_refuses_other_elements() -> None:
    carbon, hydrogen, silicon = 6, 1, 14
    _check_domain("drug_like", np.array([carbon, hydrogen]), np.array([2]), "ok")
    with pytest.raises(MixtureError, match=r"\[14\], outside the drug-like"):
        _check_domain(
            "drug_like",
            np.array([carbon, silicon, hydrogen]),
            np.array([1, 2]),
            "silicones",
        )


def test_a_transition_metal_mixture_needs_a_metal_in_every_structure() -> None:
    iron, carbon, hydrogen = 26, 6, 1
    _check_domain(
        "transition_metal_complex",
        np.array([iron, carbon, hydrogen, carbon, iron]),
        np.array([3, 2]),
        "complexes",
    )
    with pytest.raises(MixtureError, match="1 of 2 structures"):
        _check_domain(
            "transition_metal_complex",
            np.array([iron, carbon, hydrogen, carbon, hydrogen]),
            np.array([3, 2]),
            "complexes",
        )


def test_a_missing_selection_is_reported(tmp_path: Path) -> None:
    dataset = corpus(tmp_path, "first", FIRST)
    with pytest.raises(MixtureError, match="no complete selection"):
        resolve_mixture(mixture(DatasetComponent(dataset=dataset, selection="none")))


def test_the_probe_set_reads_the_first_component_labels(tmp_path: Path) -> None:
    gaps = np.arange(len(FIRST), dtype=np.float64)
    dataset = corpus(tmp_path, "first", FIRST, gap=gaps)
    data = resolve_mixture(mixture(DatasetComponent(dataset=dataset), fraction=0.5))
    datasets = data.open_datasets(atoms_getitem)
    probe = datasets.probe_set(structures=100, seed=0, properties=None)
    assert probe is not None
    assert probe.names == ["gap"]
    part = data.components[0].parts[0]
    assert len(probe.dataset) == len(part.validation_structure_ids)
    table = pl.read_parquet(dataset / "table.parquet")
    expected = table.filter(
        pl.col("structure_id").is_in(pl.Series(part.validation_structure_ids).implode())
    ).sort("structure_id")["gap"]
    np.testing.assert_array_equal(probe.labels[:, 0], expected.to_numpy())
    # Sample i of the probe set is the structure label row i belongs to.
    whole = TrainingMoleculeDataset(dataset, atoms_getitem)
    for index, structure_id in enumerate(part.validation_structure_ids):
        torch.testing.assert_close(
            probe.dataset[index].atomic_positions,
            whole[int(structure_id)].atomic_positions,
            rtol=0,
            atol=0,
        )
    assert datasets.probe_set(structures=10, seed=0, properties=["logp"]) is None


REPOSITORY = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "path",
    sorted(
        path
        for path in (REPOSITORY / "configs/training").glob("**/*.yaml")
        if path.name != "architecture_config.yaml"
    ),
    ids=lambda path: path.parent.name,
)
def test_the_pretraining_configs_parse(path: Path) -> None:
    config = pydantic_yaml.parse_yaml_file_as(PretrainingConfig, path)
    assert config.data.domain in ("drug_like", "transition_metal_complex")


def test_a_pretraining_config_is_not_a_classifier_config() -> None:
    with pytest.raises(ValueError, match="dataset_path"):
        ClassifierTrainingConfig.model_validate(
            {
                "training_name": "x",
                "dataloader": {
                    "batch_sampling": {"kind": "random_shuffle", "batch_size": 2}
                },
                "epochs": 1,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "model_config_path": "x.yaml",
                "output_base": "runs",
                "noise_level": 0.3,
            }
        )
