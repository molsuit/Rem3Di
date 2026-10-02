"""Zenodo QM9-OR (record 13380412) -> the frozen QM9-OR molecule table plus DFT frames.

QM9-OR (Zhou et al.) ships one file, ``qm9-or.npy``: 121,416 dict entries, each
with a QM9 index, an InChI, a padded ``(27, 8)`` array of DFT-optimised
coordinates plus a one-hot H/C/N/O/F atom type, the per-centre CIP labels and
the optical rotations at 355, 589.3 and 633 nm.

This preparer rebuilds, from that file alone, the table the Rem3Di paper trained
on (formerly committed as ``qm9or_splits.csv``) and checks it byte for byte
against sha256 pins:

* one row per non-isomeric canonical SMILES, the **first** npy entry winning,
  ``molecule_id`` numbered in order of first appearance (121,416 -> 117,625);
* ``isomeric_smiles`` / ``n_chiral`` / ``rs`` / ``or_sign_589`` from that entry;
* the four seeded random splits of ``generators/make_splits.py`` and the three
  scaffold splits of ``generators/make_scaffold_splits.py --algo shuffle``.

Alongside the table it decodes the same first entry's geometry into an
``ase.Atoms`` per row. Nothing is written except the raw download; the dataset
format that will hold table plus frames does not exist yet.

Run from the repository root::

    uv run --no-sync prepare-qm9or [--raw-root DIR] [--no-download]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rdkit import Chem, rdBase
from rdkit.Chem.inchi import MolFromInchi
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger("preparers.qm9or")

#: The ``Rem3Di`` checkout this module lives in; the default raw path hangs off it.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_ROOT = REPOSITORY_ROOT / "benchmark_data" / "raw" / "qm9or"

#: Atomic numbers of the one-hot columns 3..7 of every ``xyz`` array.
ONE_HOT_ATOMIC_NUMBERS = (1, 6, 7, 8, 9)
#: Rows of every padded ``xyz`` array and its columns (x, y, z + five one-hot).
XYZ_SHAPE = (27, 8)
ENTRY_KEYS = frozenset({"index", "inchi", "xyz", "chiral_centers", "rotation"})
#: Position of the 589.3 nm value in each entry's ``rotation`` list.
ROTATION_589_POSITION = 1
HYDROGEN = 1

#: The five per-molecule columns, in table order (the old source table).
MOLECULE_COLUMNS = ("molecule_id", "isomeric_smiles", "n_chiral", "rs", "or_sign_589")
SPLIT_NAMES = ("train", "valid", "test")


class QM9ORPreparationError(RuntimeError):
    """Raised when the source file or the rebuilt table is not what it must be."""


# --------------------------------------------------------------- the source


class SourceFile(BaseModel):
    """A downloadable file and the checks it has to pass."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    md5: str = Field(pattern=r"^[0-9a-f]{32}$")


#: Zenodo record 13380412, its only file (version 1, published 2024-08-27).
QM9OR_SOURCE = SourceFile(
    url="https://zenodo.org/api/records/13380412/files/qm9-or.npy/content",
    filename="qm9-or.npy",
    size_bytes=237_092_369,
    sha256="cbe1c0def10a57e3d29c33e666b9fec30c04cf8cf3ddd8f59f642e4a1772e9ba",
    md5="d7a24d28e5f611acefb9a07ff9831efd",
)


# --------------------------------------------------------------- the splits


class RandomSplit(BaseModel):
    """``generators/make_splits.py``: a seeded permutation of the molecule ids.

    The first ``round(test_fraction * n)`` permuted rows are test, the next
    ``round(valid_fraction * n)`` valid, the rest train.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["random"] = "random"
    seed: int
    test_fraction: float = Field(default=0.2, gt=0, lt=1)
    valid_fraction: float = Field(default=0.1, gt=0, lt=1)

    @model_validator(mode="after")
    def check_fractions(self) -> RandomSplit:
        if self.test_fraction + self.valid_fraction >= 1:
            raise ValueError("test_fraction + valid_fraction must leave a train fold")
        return self

    @property
    def column_name(self) -> str:
        return f"random_s{self.seed}"


class ScaffoldSplit(BaseModel):
    """``generators/make_scaffold_splits.py --algo shuffle``.

    Bemis-Murcko scaffold groups (in order of first appearance) are visited in a
    seeded random order and packed whole: into train while it stays within
    ``train_fraction * n``, else into valid while train + valid stays within
    ``(train_fraction + valid_fraction) * n``, else into test.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["scaffold"] = "scaffold"
    seed: int
    train_fraction: float = Field(default=0.8, gt=0, lt=1)
    valid_fraction: float = Field(default=0.1, gt=0, lt=1)
    include_chirality: bool = True

    @model_validator(mode="after")
    def check_fractions(self) -> ScaffoldSplit:
        if self.train_fraction + self.valid_fraction >= 1:
            raise ValueError("train_fraction + valid_fraction must leave a test fold")
        return self

    @property
    def column_name(self) -> str:
        return f"scaffold_s{self.seed}"


SplitDefinition = Annotated[RandomSplit | ScaffoldSplit, Field(discriminator="kind")]

#: The seven frozen split columns, in table order.
FROZEN_SPLITS: tuple[RandomSplit | ScaffoldSplit, ...] = (
    *(RandomSplit(seed=seed) for seed in (42, 43, 44, 45)),
    *(ScaffoldSplit(seed=seed) for seed in (0, 1, 2)),
)


# ----------------------------------------------------------------- the pins


class SplitCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    train: int = Field(ge=0)
    valid: int = Field(ge=0)
    test: int = Field(ge=0)


class TableHashes(BaseModel):
    """sha256 of three serialisations of the table (``pandas.to_csv``, no index).

    * ``table``: every column, CRLF line endings — the bytes of the former
      ``qm9or_splits.csv``.
    * ``table_without_smiles``: every column but ``isomeric_smiles``, CRLF. When
      only this one matches, the RDKit canonicaliser is the difference.
    * ``molecule_columns``: the five :data:`MOLECULE_COLUMNS`, LF — the bytes of
      the author's lost ``qm9or_molecules-use.csv``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    table: str = Field(pattern=r"^[0-9a-f]{64}$")
    table_without_smiles: str = Field(pattern=r"^[0-9a-f]{64}$")
    molecule_columns: str = Field(pattern=r"^[0-9a-f]{64}$")


class TablePins(BaseModel):
    """What the rebuilt table must come out as."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    molecule_count: int = Field(gt=0)
    hashes: TableHashes
    #: Expected fold sizes per split column.
    split_counts: dict[str, SplitCounts]


_RANDOM_COUNTS = SplitCounts(train=82_338, valid=11_762, test=23_525)
_SCAFFOLD_COUNTS = SplitCounts(train=94_100, valid=11_762, test=11_763)

#: The table the paper trained on, verified against the committed CSV on 2026-10-02.
QM9OR_PINS = TablePins(
    molecule_count=117_625,
    hashes=TableHashes(
        table="b5592028060841535f8c7607d5ac00b6089b74c27987914285641130806149ac",
        table_without_smiles="dafdd77158ce2f86d1fbb9864b9ba76234432316849024448cae3450cbcb392f",
        molecule_columns="38386849432b81390454748dbda9ccb5ac11c260e621ba848b86a469266be2d6",
    ),
    split_counts={
        split.column_name: _RANDOM_COUNTS
        if split.kind == "random"
        else _SCAFFOLD_COUNTS
        for split in FROZEN_SPLITS
    },
)


# --------------------------------------------------------------- the config


class QM9ORPreparerConfig(BaseModel):
    """Knobs of one preparer run."""

    model_config = ConfigDict(extra="forbid")

    #: Directory that holds (or receives) ``qm9-or.npy``.
    raw_root: Path = DEFAULT_RAW_ROOT
    source: SourceFile = QM9OR_SOURCE
    #: Download the source when it is missing; otherwise a missing file is an error.
    download: bool = True
    splits: list[SplitDefinition] = Field(default_factory=lambda: list(FROZEN_SPLITS))
    #: ``None`` skips the table verification (only meant for synthetic inputs).
    pins: TablePins | None = QM9OR_PINS

    @model_validator(mode="after")
    def check_splits(self) -> QM9ORPreparerConfig:
        names = [split.column_name for split in self.splits]
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate split columns {duplicates}")
        if self.pins is not None and set(self.pins.split_counts) != set(names):
            raise ValueError(
                f"pins cover split columns {sorted(self.pins.split_counts)}, "
                f"the config defines {sorted(names)}"
            )
        return self

    @property
    def source_path(self) -> Path:
        return self.raw_root / self.source.filename


# ------------------------------------------------------- download and verify


def file_digests(path: Path) -> tuple[str, str]:
    """``(sha256, md5)`` of a file, read in 1 MiB chunks."""
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            sha256.update(chunk)
            md5.update(chunk)
    return sha256.hexdigest(), md5.hexdigest()


def verify_source_file(path: Path, source: SourceFile) -> None:
    """Raise unless ``path`` has the pinned size, sha256 and md5."""
    if not path.is_file():
        raise QM9ORPreparationError(f"source file not found: {path}")
    problems: list[str] = []
    size_bytes = path.stat().st_size
    if size_bytes != source.size_bytes:
        problems.append(f"size {size_bytes} != {source.size_bytes}")
    sha256, md5 = file_digests(path)
    if sha256 != source.sha256:
        problems.append(f"sha256 {sha256} != {source.sha256}")
    if md5 != source.md5:
        problems.append(f"md5 {md5} != {source.md5}")
    if problems:
        raise QM9ORPreparationError(
            f"{path} is not the pinned {source.filename} ({source.url}): "
            + "; ".join(problems)
        )


def download_source(source: SourceFile, destination: Path) -> None:
    """Fetch ``source`` to ``destination`` via a ``.part`` file, verified before the rename."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    logger.info("downloading %s -> %s", source.url, destination)
    try:
        with (
            urllib.request.urlopen(source.url, timeout=60) as response,
            partial.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle, length=1 << 20)
        verify_source_file(partial, source)
    except OSError as error:
        partial.unlink(missing_ok=True)
        raise QM9ORPreparationError(
            f"download of {source.url} failed: {error}"
        ) from error
    except QM9ORPreparationError:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(destination)


def ensure_source(config: QM9ORPreparerConfig) -> Path:
    """The verified source file, downloaded first when missing and allowed."""
    path = config.source_path
    if not path.exists():
        if not config.download:
            raise QM9ORPreparationError(
                f"{path} is missing and downloading is disabled"
            )
        download_source(config.source, path)
    verify_source_file(path, config.source)
    return path


def load_entries(path: Path) -> list[Any]:
    """The npy's entry dicts, as a list.

    Unpickling is only safe because :func:`verify_source_file` pinned the bytes first.
    """
    entries = np.load(path, allow_pickle=True)
    if entries.dtype != object or entries.ndim != 1 or len(entries) == 0:
        raise QM9ORPreparationError(
            f"{path}: expected a non-empty 1-d object array, got {entries.dtype} {entries.shape}"
        )
    return entries.tolist()


# ------------------------------------------------------------- one entry


def check_entry_layout(entry: Any, position: int) -> None:
    """Raise unless ``entry`` is a dict in the QM9-OR layout."""
    if not isinstance(entry, dict) or set(entry) != ENTRY_KEYS:
        found = sorted(entry) if isinstance(entry, dict) else type(entry).__name__
        raise QM9ORPreparationError(
            f"entry {position}: expected keys {sorted(ENTRY_KEYS)}, got {found}"
        )
    if np.shape(entry["xyz"]) != XYZ_SHAPE:
        raise QM9ORPreparationError(
            f"entry {position}: xyz has shape {np.shape(entry['xyz'])}, expected {XYZ_SHAPE}"
        )
    if len(entry["rotation"]) != 3:
        raise QM9ORPreparationError(
            f"entry {position}: rotation has {len(entry['rotation'])} values, expected 3"
        )


def decode_geometry(xyz: np.ndarray, position: int) -> Atoms:
    """Atomic numbers + positions of the real atoms of one padded ``xyz`` array.

    A real row has exactly one one-hot 1; padding rows are all zero and must
    come after every real row.
    """
    array = np.asarray(xyz, dtype=np.float64)
    one_hot = array[:, 3:]
    row_sums = one_hot.sum(axis=1)
    if not (np.isin(one_hot, (0.0, 1.0)).all() and np.isin(row_sums, (0.0, 1.0)).all()):
        raise QM9ORPreparationError(
            f"entry {position}: atom-type columns are not one-hot"
        )
    is_real = row_sums == 1.0
    atom_count = int(is_real.sum())
    if atom_count == 0 or not is_real[:atom_count].all():
        raise QM9ORPreparationError(
            f"entry {position}: padding rows must follow all {atom_count} real atoms"
        )
    if np.any(array[atom_count:] != 0.0):
        raise QM9ORPreparationError(f"entry {position}: padding rows are not all zero")
    numbers = np.asarray(ONE_HOT_ATOMIC_NUMBERS)[one_hot[:atom_count].argmax(axis=1)]
    return Atoms(numbers=numbers, positions=array[:atom_count, :3])


def molecule_from_inchi(inchi: str, position: int) -> Chem.Mol:
    molecule = MolFromInchi(inchi)
    if molecule is None:
        raise QM9ORPreparationError(f"entry {position}: RDKit cannot parse {inchi!r}")
    return molecule


@dataclass(frozen=True)
class CompositionComparison:
    """Element counts of a decoded geometry against its InChI-derived molecule."""

    heavy_atoms_agree: bool
    geometry_hydrogens: int
    molecule_hydrogens: int

    @property
    def hydrogens_agree(self) -> bool:
        return self.geometry_hydrogens == self.molecule_hydrogens


def compare_composition(atoms: Atoms, molecule: Chem.Mol) -> CompositionComparison:
    """Compare the geometry's elements to the InChI molecule with explicit hydrogens.

    Heavy atoms are a sound identity check: across all 121,416 source entries
    they agree. Hydrogen counts are not: for 1,571 entries the InChI (a radical
    or protonation layer RDKit completes with implicit hydrogens) and the DFT
    geometry differ by 1-5 hydrogens, so that disagreement is counted, not fatal.
    """
    geometry = Counter(int(number) for number in atoms.numbers)
    with_hydrogens = Chem.AddHs(molecule)
    expected = Counter(atom.GetAtomicNum() for atom in with_hydrogens.GetAtoms())
    geometry_hydrogens = geometry.pop(HYDROGEN, 0)
    molecule_hydrogens = expected.pop(HYDROGEN, 0)
    return CompositionComparison(
        heavy_atoms_agree=geometry == expected,
        geometry_hydrogens=geometry_hydrogens,
        molecule_hydrogens=molecule_hydrogens,
    )


@dataclass(frozen=True)
class EntryLabels:
    """The per-molecule labels of one source entry."""

    n_chiral: int
    rs: int
    or_sign_589: int


def entry_labels(entry: dict[str, Any]) -> EntryLabels:
    """``n_chiral`` counts every centre (pseudo-asymmetric ``r``/``s`` too).

    ``rs`` is 1 only when the centre labels are exactly ``["S"]`` (see the
    README's open question) and ``or_sign_589`` is ``numpy.sign`` with no
    threshold (the rotations are rounded to 0.01, ``-0.0`` gives 0).
    """
    centre_labels = [str(label) for _, label in entry["chiral_centers"]]
    rotation = float(entry["rotation"][ROTATION_589_POSITION])
    return EntryLabels(
        n_chiral=len(centre_labels),
        rs=int(centre_labels == ["S"]),
        or_sign_589=int(np.sign(rotation)),
    )


# ------------------------------------------------------------ the collapse


@dataclass(frozen=True)
class CollapsedEntries:
    """One kept entry per non-isomeric SMILES, in order of first appearance."""

    rows: list[dict[str, Any]]
    frames: list[Atoms]
    #: Every entry's 589 nm sign, per molecule_id (first entry first).
    signs_per_molecule: list[list[int]]
    #: Kept molecules whose geometry and InChI disagree on the hydrogen count.
    hydrogen_mismatch_molecule_ids: list[int]
    source_entry_count: int


def collapse_entries(entries: Sequence[Any]) -> CollapsedEntries:
    """Group entries by non-isomeric canonical SMILES; the first entry wins."""
    molecule_id_by_smiles: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    frames: list[Atoms] = []
    signs_per_molecule: list[list[int]] = []
    hydrogen_mismatches: list[int] = []
    for position, entry in enumerate(entries):
        check_entry_layout(entry, position)
        molecule = molecule_from_inchi(entry["inchi"], position)
        labels = entry_labels(entry)
        nonisomeric = Chem.MolToSmiles(molecule, isomericSmiles=False)
        if nonisomeric in molecule_id_by_smiles:
            signs_per_molecule[molecule_id_by_smiles[nonisomeric]].append(
                labels.or_sign_589
            )
            continue
        molecule_id = len(rows)
        molecule_id_by_smiles[nonisomeric] = molecule_id
        atoms = decode_geometry(entry["xyz"], position)
        comparison = compare_composition(atoms, molecule)
        if not comparison.heavy_atoms_agree:
            raise QM9ORPreparationError(
                f"entry {position} ({entry['inchi']}): heavy atoms of the geometry "
                f"{atoms.get_chemical_formula()} disagree with the InChI"
            )
        if not comparison.hydrogens_agree:
            hydrogen_mismatches.append(molecule_id)
        atoms.info.update(
            molecule_id=molecule_id,
            qm9_index=str(entry["index"]),
            source_entry=position,
        )
        rows.append(
            {
                "molecule_id": molecule_id,
                "isomeric_smiles": Chem.MolToSmiles(molecule, isomericSmiles=True),
                "n_chiral": labels.n_chiral,
                "rs": labels.rs,
                "or_sign_589": labels.or_sign_589,
            }
        )
        frames.append(atoms)
        signs_per_molecule.append([labels.or_sign_589])
    return CollapsedEntries(
        rows=rows,
        frames=frames,
        signs_per_molecule=signs_per_molecule,
        hydrogen_mismatch_molecule_ids=hydrogen_mismatches,
        source_entry_count=len(entries),
    )


# ----------------------------------------------------------------- splits


def random_split_values(molecule_count: int, split: RandomSplit) -> np.ndarray:
    permutation = np.random.default_rng(split.seed).permutation(molecule_count)
    test_count = round(split.test_fraction * molecule_count)
    valid_count = round(split.valid_fraction * molecule_count)
    values = np.full(molecule_count, "train", dtype=object)
    values[permutation[:test_count]] = "test"
    values[permutation[test_count : test_count + valid_count]] = "valid"
    return values


def scaffold_groups(smiles: Sequence[str], include_chirality: bool) -> list[list[int]]:
    """Row indices per Bemis-Murcko scaffold, groups in order of first appearance."""
    groups: dict[str, list[int]] = defaultdict(list)
    for row, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            raise QM9ORPreparationError(f"row {row}: RDKit cannot parse {text!r}")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=molecule, includeChirality=include_chirality
        )
        groups[scaffold].append(row)
    return list(groups.values())


def scaffold_split_values(
    groups: Sequence[Sequence[int]], molecule_count: int, split: ScaffoldSplit
) -> np.ndarray:
    train_limit = split.train_fraction * molecule_count
    valid_limit = (split.train_fraction + split.valid_fraction) * molecule_count
    values = np.empty(molecule_count, dtype=object)
    train_count = valid_count = 0
    for group_position in np.random.default_rng(split.seed).permutation(len(groups)):
        rows = list(groups[group_position])
        if train_count + len(rows) <= train_limit:
            values[rows] = "train"
            train_count += len(rows)
        elif train_count + valid_count + len(rows) <= valid_limit:
            values[rows] = "valid"
            valid_count += len(rows)
        else:
            values[rows] = "test"
    return values


def add_split_columns(
    table: pd.DataFrame, splits: Sequence[RandomSplit | ScaffoldSplit]
) -> int:
    """Append one column per split; returns the number of scaffold groups (0 if none)."""
    molecule_count = len(table)
    groups_by_chirality: dict[bool, list[list[int]]] = {}
    for split in splits:
        if isinstance(split, RandomSplit):
            table[split.column_name] = random_split_values(molecule_count, split)
            continue
        if split.include_chirality not in groups_by_chirality:
            groups_by_chirality[split.include_chirality] = scaffold_groups(
                table["isomeric_smiles"].tolist(), split.include_chirality
            )
        groups = groups_by_chirality[split.include_chirality]
        table[split.column_name] = scaffold_split_values(groups, molecule_count, split)
    return max((len(groups) for groups in groups_by_chirality.values()), default=0)


# ----------------------------------------------------------- verification


def _sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def table_hashes(table: pd.DataFrame) -> TableHashes:
    return TableHashes(
        table=_sha256_of_text(table.to_csv(index=False, lineterminator="\r\n")),
        table_without_smiles=_sha256_of_text(
            table.drop(columns="isomeric_smiles").to_csv(
                index=False, lineterminator="\r\n"
            )
        ),
        molecule_columns=_sha256_of_text(
            table[list(MOLECULE_COLUMNS)].to_csv(index=False, lineterminator="\n")
        ),
    )


def split_counts(table: pd.DataFrame, column: str) -> SplitCounts:
    counts = table[column].value_counts()
    unexpected = sorted(set(counts.index) - set(SPLIT_NAMES))
    if unexpected:
        raise QM9ORPreparationError(f"{column}: unexpected fold names {unexpected}")
    return SplitCounts(**{name: int(counts.get(name, 0)) for name in SPLIT_NAMES})


def verify_table(table: pd.DataFrame, hashes: TableHashes, pins: TablePins) -> None:
    """Raise with a diagnosis unless the table matches every pin."""
    problems: list[str] = []
    if len(table) != pins.molecule_count:
        problems.append(f"{len(table)} molecules, expected {pins.molecule_count}")
    for column, expected in pins.split_counts.items():
        found = split_counts(table, column)
        if found != expected:
            problems.append(
                f"{column} counts {found.model_dump()} != {expected.model_dump()}"
            )
    if hashes.table != pins.hashes.table:
        problems.append(f"table sha256 {hashes.table} != {pins.hashes.table}")
        if hashes.table_without_smiles == pins.hashes.table_without_smiles:
            problems.append(
                "every column except isomeric_smiles matches its pin, so only the "
                f"SMILES strings differ: RDKit {rdBase.rdkitVersion} canonicalises "
                "differently from the RDKit 2026.03.5 that produced the pins"
            )
        else:
            problems.append("columns other than isomeric_smiles differ too")
    if hashes.molecule_columns != pins.hashes.molecule_columns:
        problems.append(
            f"molecule columns sha256 {hashes.molecule_columns} != {pins.hashes.molecule_columns}"
        )
    if problems:
        raise QM9ORPreparationError(
            "rebuilt QM9-OR table does not match its pins: " + "; ".join(problems)
        )


# ------------------------------------------------------------------ build


class BuildSummary(BaseModel):
    """What one build came out as."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_entries: int
    molecules: int
    #: Source entries folded into an earlier molecule (first entry wins).
    duplicate_entries: int
    molecules_with_duplicates: int
    #: Molecules whose entries include both a positive and a negative 589 nm sign.
    opposite_sign_molecules: int
    #: Molecules whose entries do not all share one 589 nm sign (zero included).
    differing_sign_molecules: int
    #: Kept geometries whose hydrogen count differs from the InChI molecule's.
    hydrogen_mismatch_molecules: int
    scaffold_groups: int
    split_counts: dict[str, SplitCounts]
    hashes: TableHashes
    verified: bool
    rdkit_version: str


@dataclass(frozen=True)
class QM9ORDataset:
    """The rebuilt table and one DFT frame per row (``frames[i]`` is row ``i``)."""

    table: pd.DataFrame
    frames: list[Atoms]
    #: Molecules whose duplicate entries carry opposite nonzero 589 nm signs.
    opposite_sign_molecule_ids: list[int]
    hydrogen_mismatch_molecule_ids: list[int]
    summary: BuildSummary


def build_dataset_from_entries(
    entries: Sequence[Any], config: QM9ORPreparerConfig
) -> QM9ORDataset:
    """Collapse, split, hash and (when pinned) verify; nothing is written."""
    collapsed = collapse_entries(entries)
    table = pd.DataFrame(collapsed.rows, columns=list(MOLECULE_COLUMNS))
    scaffold_group_count = add_split_columns(table, config.splits)
    hashes = table_hashes(table)
    if config.pins is not None:
        verify_table(table, hashes, config.pins)
    opposite = [
        molecule_id
        for molecule_id, signs in enumerate(collapsed.signs_per_molecule)
        if 1 in signs and -1 in signs
    ]
    summary = BuildSummary(
        source_entries=collapsed.source_entry_count,
        molecules=len(table),
        duplicate_entries=collapsed.source_entry_count - len(table),
        molecules_with_duplicates=sum(
            len(signs) > 1 for signs in collapsed.signs_per_molecule
        ),
        opposite_sign_molecules=len(opposite),
        differing_sign_molecules=sum(
            len(set(signs)) > 1 for signs in collapsed.signs_per_molecule
        ),
        hydrogen_mismatch_molecules=len(collapsed.hydrogen_mismatch_molecule_ids),
        scaffold_groups=scaffold_group_count,
        split_counts={
            split.column_name: split_counts(table, split.column_name)
            for split in config.splits
        },
        hashes=hashes,
        verified=config.pins is not None,
        rdkit_version=rdBase.rdkitVersion,
    )
    return QM9ORDataset(
        table=table,
        frames=collapsed.frames,
        opposite_sign_molecule_ids=opposite,
        hydrogen_mismatch_molecule_ids=collapsed.hydrogen_mismatch_molecule_ids,
        summary=summary,
    )


def build_dataset(config: QM9ORPreparerConfig) -> QM9ORDataset:
    """Download if needed, verify the source, rebuild and verify the table."""
    path = ensure_source(config)
    rdBase.DisableLog("rdApp.*")  # InChI parsing warns on many QM9 entries
    try:
        entries = load_entries(path)
        logger.info("loaded %d entries from %s", len(entries), path)
        return build_dataset_from_entries(entries, config)
    finally:
        rdBase.EnableLog("rdApp.*")


# -------------------------------------------------------------------- CLI


def format_summary(summary: BuildSummary) -> str:
    lines = [
        f"source entries              {summary.source_entries:8d}",
        f"molecules                   {summary.molecules:8d}",
        f"duplicate entries dropped   {summary.duplicate_entries:8d}"
        f"  (in {summary.molecules_with_duplicates} molecules, first entry kept)",
        f"opposite 589 nm signs       {summary.opposite_sign_molecules:8d}"
        f"  (any differing sign: {summary.differing_sign_molecules})",
        f"hydrogen-count mismatches   {summary.hydrogen_mismatch_molecules:8d}"
        "  (kept geometry vs InChI; heavy atoms all agree)",
        f"scaffold groups             {summary.scaffold_groups:8d}",
        "",
        f"{'split':12s} {'train':>7s} {'valid':>7s} {'test':>7s}",
    ]
    for column, counts in summary.split_counts.items():
        lines.append(
            f"{column:12s} {counts.train:7d} {counts.valid:7d} {counts.test:7d}"
        )
    status = "verified against pins" if summary.verified else "NOT verified (no pins)"
    lines += [
        "",
        f"table sha256                {summary.hashes.table}",
        f"molecule columns sha256     {summary.hashes.molecule_columns}",
        f"RDKit {summary.rdkit_version}, {status}",
    ]
    return "\n".join(lines)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="fail instead of downloading when qm9-or.npy is missing",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    arguments = parse_arguments(argv)
    config = QM9ORPreparerConfig(
        raw_root=arguments.raw_root, download=not arguments.no_download
    )
    try:
        dataset = build_dataset(config)
    except QM9ORPreparationError as error:
        logger.error("%s", error)
        return 1
    print(format_summary(dataset.summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
