"""Pretraining mixtures: several corpora, each restricted to a selection (§5.2.4).

A mixture names its components (one dataset, optionally a selection of it, or
every dataset under a root) and is the concatenation of them: no weights, every
selected structure is seen once per epoch. At start-up :func:`resolve_mixture`

1. reads each component's selection and checks it against the mixture's domain
   (``drug_like``: only MACE-OFF elements; ``transition_metal_complex``: a
   transition metal in every structure) and that charge and multiplicity are set,
2. drops molecules whose ``nonisomeric_smiles`` an earlier component already
   holds (components without SMILES are left alone),
3. splits each component into training and validation by ``molecule_id`` with a
   fixed key per molecule, so the validation molecules of a smaller nested
   selection are those of the larger one that it contains.

:meth:`PretrainingData.open_datasets` reads each dataset's selected structures
once and concatenates the training parts; validation stays per component so its
loss is reported per component. Ids are
never rewritten: a structure is ``(dataset, structure_id)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import polars as pl
import yaml
from ase.data import atomic_numbers as atomic_number_of_symbol
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag
from torch.utils.data import ConcatDataset, Dataset, Subset

from remedi.data_handling.dataset.training_dataset import (
    GetItemFn,
    TrainingMoleculeDataset,
)
from remedi_data.bundle import read_spec
from remedi_data.chemistry.elements import MACE_OFF_ELEMENTS
from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_data.selection import (
    SelectionError,
    dataset_identity,
    pool_datasets,
    read_selection,
    read_selection_record,
)
from remedi_data.selection.config import NAME_PATTERN
from remedi_data.selection.order import keys_by_id

logger = logging.getLogger(__name__)

Domain = Literal["drug_like", "transition_metal_complex"]

#: The random-key stream of the validation split; the selection streams are 0-2.
VALIDATION_STREAM = 10

#: Groups 3-12: Sc-Zn, Y-Cd, La and Hf-Hg, Ac and Rf-Cn.
TRANSITION_METAL_ATOMIC_NUMBERS = frozenset(
    [*range(21, 31), *range(39, 49), 57, *range(72, 81), 89, *range(104, 113)]
)
_MACE_OFF_ATOMIC_NUMBERS = frozenset(
    atomic_number_of_symbol[symbol] for symbol in MACE_OFF_ELEMENTS
)

RECORD_NAME = "pretraining_data.yaml"


class MixtureError(ValueError):
    """A mixture cannot be assembled from the data on disk."""


class DatasetComponent(BaseModel):
    """One dataset, all of it or one of its selections."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["dataset"] = "dataset"
    dataset: Path
    selection: str | None = Field(default=None, pattern=NAME_PATTERN)

    @property
    def name(self) -> str:
        suffix = f"/{self.selection}" if self.selection else ""
        return f"{self.dataset.name}{suffix}"


class DatasetGroupComponent(BaseModel):
    """Every dataset under ``root`` (the benchmark pool), all their structures.

    Its datasets are taken in name order; with ``deduplicate_across_components``
    a molecule in several of them is kept in the first only.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["dataset_group"]
    root: Path

    @property
    def name(self) -> str:
        return self.root.name


def _component_kind(value: Any) -> str:
    if isinstance(value, dict):
        return value.get("kind", "dataset")
    return getattr(value, "kind", "dataset")


MixtureComponent = Annotated[
    Annotated[DatasetComponent, Tag("dataset")]
    | Annotated[DatasetGroupComponent, Tag("dataset_group")],
    Discriminator(_component_kind),
]


class MoleculeFractionValidation(BaseModel):
    """A molecule is held out when its fixed random key is below ``fraction``."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["molecule_fraction"] = "molecule_fraction"
    fraction: float = Field(gt=0.0, lt=1.0)
    seed: int = 0


class PretrainingDataConfig(BaseModel):
    """The ``data`` block of a pretraining config."""

    model_config = ConfigDict(extra="forbid")

    domain: Domain
    components: list[MixtureComponent] = Field(min_length=1)
    validation: MoleculeFractionValidation
    deduplicate_across_components: Literal["nonisomeric_smiles", "none"] = (
        "nonisomeric_smiles"
    )


class PartRecord(BaseModel):
    """What one dataset of a component contributes."""

    model_config = ConfigDict(extra="forbid")

    dataset: Path
    dataset_id: str
    table_content_sha256: str
    structures_sha256: str
    selection: str | None
    selection_sha256: str | None
    selected_structures: int
    selected_molecules: int
    duplicate_molecules_dropped: int
    duplicate_structures_dropped: int
    train_structures: int
    train_molecules: int
    validation_structures: int
    validation_molecules: int


class ComponentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    parts: list[PartRecord]


class MixtureRecord(BaseModel):
    """The realised mixture, written into the run directory."""

    model_config = ConfigDict(extra="forbid")

    config: PretrainingDataConfig
    components: list[ComponentRecord]
    train_structures: int
    validation_structures: int
    created_at: datetime


@dataclass(frozen=True)
class MixturePart:
    """One dataset of a component, with its training and validation structures."""

    dataset: Path
    train_structure_ids: np.ndarray
    validation_structure_ids: np.ndarray
    train_atom_counts: np.ndarray
    validation_atom_counts: np.ndarray


@dataclass(frozen=True)
class MixtureComponentData:
    name: str
    parts: list[MixturePart]


@dataclass(frozen=True)
class ProbeSet:
    """Structures, their labels (NaN where missing) and the label names."""

    dataset: Dataset
    labels: np.ndarray
    masks: np.ndarray
    names: list[str]


@dataclass(frozen=True)
class OpenedPart:
    """A part's structures, read once; training and validation index into them."""

    part: MixturePart
    dataset: TrainingMoleculeDataset
    train_positions: np.ndarray
    validation_positions: np.ndarray


@dataclass(frozen=True)
class MixtureDatasets:
    """The training set, the validation sets per component, and the probe source."""

    training: Dataset
    training_atom_counts: np.ndarray
    #: component name -> (dataset, atoms per structure), for components with
    #: validation structures.
    validation: dict[str, tuple[Dataset, np.ndarray]]
    opened: list[list[OpenedPart]]

    def probe_set(
        self, *, structures: int, seed: int, properties: list[str] | None
    ) -> ProbeSet | None:
        """Validation structures of the first component with its labels.

        ``None`` when that component is not a single dataset with labels and
        validation structures, or none of ``properties`` is among its labels.
        """
        first = self.opened[0]
        if len(first) != 1 or not len(first[0].validation_positions):
            return None
        opened = first[0]
        dataset = opened.part.dataset
        label_names = [label.name for label in read_spec(dataset).labels]
        names = (
            label_names
            if properties is None
            else [name for name in properties if name in label_names]
        )
        if not names:
            return None
        generator = np.random.default_rng(seed)
        validation_ids = opened.part.validation_structure_ids
        count = min(structures, len(validation_ids))
        chosen = np.sort(
            generator.choice(len(validation_ids), size=count, replace=False)
        )
        chosen_ids = validation_ids[chosen]
        rows = (
            pl.read_parquet(dataset / "table.parquet", columns=["structure_id", *names])
            .filter(pl.col("structure_id").is_in(pl.Series(chosen_ids).implode()))
            .sort("structure_id")
        )
        labels = rows.select(names).fill_null(np.nan).to_numpy().astype(np.float64)
        return ProbeSet(
            dataset=Subset(
                opened.dataset, opened.validation_positions[chosen].tolist()
            ),
            labels=labels,
            masks=~np.isnan(labels),
            names=names,
        )


@dataclass(frozen=True)
class PretrainingData:
    """A resolved mixture: which structures of which datasets train and validate."""

    components: list[MixtureComponentData]
    record: MixtureRecord

    def open_datasets(self, get_item: GetItemFn) -> MixtureDatasets:
        """Read every part's structures once (in this process, before workers fork)."""
        opened: list[list[OpenedPart]] = []
        for component in self.components:
            opened_parts = []
            for part in component.parts:
                structure_ids = np.union1d(
                    part.train_structure_ids, part.validation_structure_ids
                )
                opened_parts.append(
                    OpenedPart(
                        part=part,
                        dataset=TrainingMoleculeDataset(
                            part.dataset, get_item, structure_ids=structure_ids
                        ).open(),
                        train_positions=np.searchsorted(
                            structure_ids, part.train_structure_ids
                        ),
                        validation_positions=np.searchsorted(
                            structure_ids, part.validation_structure_ids
                        ),
                    )
                )
            opened.append(opened_parts)

        def concatenate(parts: list[OpenedPart], positions: str) -> Dataset:
            return ConcatDataset(
                [
                    Subset(item.dataset, getattr(item, positions).tolist())
                    for item in parts
                    if len(getattr(item, positions))
                ]
            )

        everything = [item for parts in opened for item in parts]
        validation = {}
        for component, parts in zip(self.components, opened, strict=True):
            with_validation = [p for p in parts if len(p.validation_positions)]
            if with_validation:
                validation[component.name] = (
                    concatenate(with_validation, "validation_positions"),
                    np.concatenate(
                        [p.part.validation_atom_counts for p in with_validation]
                    ),
                )
        return MixtureDatasets(
            training=concatenate(everything, "train_positions"),
            training_atom_counts=np.concatenate(
                [item.part.train_atom_counts for item in everything]
            ),
            validation=validation,
            opened=opened,
        )

    def write_record(self, directory: Path) -> Path:
        path = Path(directory) / RECORD_NAME
        path.write_text(
            yaml.safe_dump(self.record.model_dump(mode="json"), sort_keys=False)
        )
        return path


def _component_datasets(component: DatasetComponent | DatasetGroupComponent):
    """``(dataset, selection)`` pairs a component stands for."""
    if isinstance(component, DatasetComponent):
        return [(component.dataset, component.selection)]
    try:
        return [(path, None) for path in pool_datasets([component.root])]
    except SelectionError as error:
        raise MixtureError(f"component {component.name}: {error}") from error


def _check_domain(
    domain: Domain, atomic_numbers: np.ndarray, atom_counts: np.ndarray, name: str
) -> None:
    """Raise if the structures (atoms concatenated) do not belong to ``domain``."""
    present = set(np.unique(atomic_numbers).tolist())
    if domain == "drug_like":
        outside = sorted(present - _MACE_OFF_ATOMIC_NUMBERS)
        if outside:
            raise MixtureError(
                f"{name} has atomic numbers {outside}, outside the drug-like "
                "(MACE-OFF) elements; is it a transition-metal corpus?"
            )
        return
    is_metal = np.isin(atomic_numbers, list(TRANSITION_METAL_ATOMIC_NUMBERS))
    starts = np.concatenate([[0], np.cumsum(atom_counts)[:-1]])
    metals_per_structure = np.add.reduceat(is_metal.astype(np.int64), starts)
    without = int(np.sum(metals_per_structure == 0))
    if without:
        raise MixtureError(
            f"{name}: {without} of {len(atom_counts)} structures hold no "
            "transition metal; is it a drug-like corpus?"
        )


def _selected_atoms(
    dataset: Path, structure_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Atomic numbers (concatenated) and atom counts of the given structures."""
    arrays = MoleculeDataset.open_structure_arrays(dataset)
    pointer = np.asarray(arrays.molecule_ptr[:], dtype=np.int64)
    counts = np.diff(pointer)
    selected = np.zeros(len(counts), dtype=bool)
    selected[structure_ids] = True
    numbers = np.asarray(arrays.atomic_numbers[:])
    return numbers[np.repeat(selected, counts)], counts[structure_ids]


def _resolve_part(
    dataset: Path,
    selection: str | None,
    config: PretrainingDataConfig,
    seen_smiles: set[str],
    component_name: str,
) -> tuple[MixturePart, PartRecord]:
    identity = dataset_identity(dataset)
    table = pl.read_parquet(Path(dataset) / "table.parquet")
    if selection is None:
        structure_ids = table["structure_id"].to_numpy().astype(np.int64)
        selection_sha256 = None
    else:
        structure_ids = read_selection(dataset, selection)
        selection_sha256 = read_selection_record(dataset, selection).selection_sha256
    rows = table.filter(
        pl.col("structure_id").is_in(pl.Series(structure_ids).implode())
    )
    name = f"{component_name} ({identity.dataset_id})"
    if len(rows) == 0:
        raise MixtureError(f"{name} selects no structures")
    for column in ("total_charge", "multiplicity"):
        if column not in rows.columns or rows[column].is_null().any():
            raise MixtureError(f"{name} has structures without {column}")
        if rows[column].is_nan().any():
            raise MixtureError(f"{name} has structures with a NaN {column}")

    atomic_numbers, atom_counts = _selected_atoms(dataset, structure_ids)
    _check_domain(config.domain, atomic_numbers, atom_counts, name)

    selected_molecules = rows["molecule_id"].n_unique()
    selected_structures = len(rows)
    deduplicate = (
        config.deduplicate_across_components == "nonisomeric_smiles"
        and "nonisomeric_smiles" in rows.columns
    )
    if deduplicate:
        repeated = rows["nonisomeric_smiles"].is_in(
            pl.Series(list(seen_smiles)).implode()
        )
        dropped = rows.filter(repeated)
        rows = rows.filter(~repeated)
        seen_smiles.update(rows["nonisomeric_smiles"].unique().to_list())
    else:
        dropped = rows.clear()
    if len(rows) == 0:
        raise MixtureError(f"{name}: every molecule is already in an earlier component")

    molecule_ids = rows["molecule_id"].to_numpy().astype(np.int64)
    keys = keys_by_id(config.validation.seed, VALIDATION_STREAM, molecule_ids)
    held_out = keys < config.validation.fraction
    structures = rows["structure_id"].to_numpy().astype(np.int64)
    train_ids = np.sort(structures[~held_out])
    validation_ids = np.sort(structures[held_out])

    def counts(ids: np.ndarray) -> np.ndarray:
        # structure_ids is sorted (a selection, or the table's row order).
        return atom_counts[np.searchsorted(structure_ids, ids)]

    part = MixturePart(
        dataset=Path(dataset),
        train_structure_ids=train_ids,
        validation_structure_ids=validation_ids,
        train_atom_counts=counts(train_ids),
        validation_atom_counts=counts(validation_ids),
    )
    record = PartRecord(
        dataset=Path(dataset),
        dataset_id=identity.dataset_id,
        table_content_sha256=identity.table_content_sha256,
        structures_sha256=identity.structures_sha256,
        selection=selection,
        selection_sha256=selection_sha256,
        selected_structures=selected_structures,
        selected_molecules=selected_molecules,
        duplicate_molecules_dropped=dropped["molecule_id"].n_unique(),
        duplicate_structures_dropped=len(dropped),
        train_structures=len(train_ids),
        train_molecules=len(np.unique(molecule_ids[~held_out])),
        validation_structures=len(validation_ids),
        validation_molecules=len(np.unique(molecule_ids[held_out])),
    )
    logger.info(
        "%s: %d structures selected, %d dropped as duplicates, %d train, %d validation",
        name,
        selected_structures,
        record.duplicate_structures_dropped,
        record.train_structures,
        record.validation_structures,
    )
    return part, record


def resolve_mixture(config: PretrainingDataConfig) -> PretrainingData:
    """Read, check, deduplicate and split every component of a mixture.

    Raises:
        MixtureError: if a component is missing, empty, outside the domain,
            lacks charge or multiplicity, or the mixture has no training or no
            validation structures.
    """
    seen_smiles: set[str] = set()
    components: list[MixtureComponentData] = []
    records: list[ComponentRecord] = []
    for component in config.components:
        parts: list[MixturePart] = []
        part_records: list[PartRecord] = []
        for dataset, selection in _component_datasets(component):
            try:
                part, record = _resolve_part(
                    dataset, selection, config, seen_smiles, component.name
                )
            except SelectionError as error:
                raise MixtureError(f"component {component.name}: {error}") from error
            parts.append(part)
            part_records.append(record)
        components.append(MixtureComponentData(name=component.name, parts=parts))
        records.append(ComponentRecord(name=component.name, parts=part_records))

    names = [component.name for component in components]
    if len(set(names)) != len(names):
        raise MixtureError(f"two components have the same name: {names}")
    train = sum(r.train_structures for c in records for r in c.parts)
    validation = sum(r.validation_structures for c in records for r in c.parts)
    if train == 0 or validation == 0:
        raise MixtureError(
            f"the mixture has {train} training and {validation} validation "
            "structures; both must be non-zero"
        )
    return PretrainingData(
        components=components,
        record=MixtureRecord(
            config=config,
            components=records,
            train_structures=train,
            validation_structures=validation,
            created_at=datetime.now(UTC),
        ),
    )
