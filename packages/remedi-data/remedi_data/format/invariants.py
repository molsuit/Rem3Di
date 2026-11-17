"""The format's invariants (``BENCHMARK_DATA_FORMAT.md`` §4.4).

``validate_table`` checks the cheap, vectorised invariants every read asserts,
on a bundle table or a dataset table; ``structure_problems`` checks the
per-frame ones (9: geometry agrees with the SMILES, 10: geometry limits), which
run when a dataset is written and in ``verify``, not on every read. Every
message is prefixed with its invariant number; an empty list means valid.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from ase import Atoms
from rdkit import Chem
from rdkit.Geometry import Point3D

from remedi_data.chemistry.geometry import (
    GeometryLimits,
    frame_geometry_violations,
)
from remedi_data.format.spec import (
    SMILES_COLUMNS,
    SPLIT_VALUES,
    DatasetSpec,
    LabelColumn,
    TaskType,
)
from remedi_data.parallel import process_map

if TYPE_CHECKING:
    # Only annotations: remedi_data.dataset reads the format, so importing it
    # here at run time would be circular.
    from remedi_data.dataset.structures import Structures

#: How many offending rows a single message names before it summarises.
_MAX_REPORTED_ROWS = 5


def _summarise_rows(row_indices: list[int]) -> str:
    if len(row_indices) <= _MAX_REPORTED_ROWS:
        return f"rows {row_indices}"
    shown = row_indices[:_MAX_REPORTED_ROWS]
    return f"{len(row_indices)} rows, first {shown}"


def _summarise_ids(identifiers: list[int]) -> str:
    if len(identifiers) <= _MAX_REPORTED_ROWS:
        return str(identifiers)
    remaining = len(identifiers) - _MAX_REPORTED_ROWS
    return f"{identifiers[:_MAX_REPORTED_ROWS]} and {remaining} more"


def _is_string_column(series: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)


def count_stereoisomer_straddling_constitutions(
    table: pd.DataFrame, split_column: str
) -> int:
    """Constitutions whose stereoisomers land in more than one fold (§4.3).

    Counts ``molecule_id`` groups that hold more than one ``stereoisomer_id``
    *and* more than one distinct value of ``split_column``. This is zero by
    construction when ``split_group`` is ``molecule_id``; for the drug-property
    panel it is the published protocol's leakage, recorded so it stays visible.
    """
    for column_name in ("molecule_id", "stereoisomer_id", split_column):
        if column_name not in table.columns:
            raise KeyError(f"table has no column {column_name!r}")
    grouped = table.groupby("molecule_id")
    multi_stereoisomer = grouped["stereoisomer_id"].nunique() > 1
    multi_fold = grouped[split_column].nunique() > 1
    return int((multi_stereoisomer & multi_fold).sum())


def stereochemistry_from_frame(isomeric_smiles: str, atoms: Atoms) -> str | None:
    """:func:`stereochemistry_from_positions` for an ``ase.Atoms`` frame."""
    return stereochemistry_from_positions(
        isomeric_smiles, atoms.get_atomic_numbers(), atoms.get_positions()
    )


def stereochemistry_from_positions(
    isomeric_smiles: str, atomic_numbers: np.ndarray, positions: np.ndarray
) -> str | None:
    """Tetrahedral stereo perceived from one frame for invariant 9.

    The result is a canonical SMILES with tetrahedral centres only (see
    :func:`tetrahedral_stereo_smiles` for why double-bond stereo is dropped),
    restricted to the centres ``isomeric_smiles`` assigns, and is meant to be
    compared to ``tetrahedral_stereo_smiles(isomeric_smiles)``.

    The molecular graph comes from ``isomeric_smiles`` (with explicit hydrogens
    added) and only the *stereochemistry* is re-perceived from the coordinates.
    This assumes the frame's atom order equals the RDKit ``AddHs`` atom order
    of the isomeric SMILES: ETKDG produces that order, and the structure
    preparers reorder source geometries into it. Returns ``None`` when the
    molecule cannot be built that way; the caller then reports the row as
    failing invariant 9 rather than silently skipping it.
    """
    template = Chem.MolFromSmiles(isomeric_smiles)
    if template is None:
        return None
    molecule = Chem.AddHs(template)
    if molecule.GetNumAtoms() != len(atomic_numbers):
        return None
    expected_numbers = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
    if [int(number) for number in atomic_numbers] != expected_numbers:
        return None
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index, (x, y, z) in enumerate(np.asarray(positions, dtype=np.float64).tolist()):
        conformer.SetAtomPosition(index, Point3D(x, y, z))
    molecule.RemoveAllConformers()
    molecule.AddConformer(conformer, assignId=True)
    try:
        Chem.AssignStereochemistryFrom3D(molecule)
    except (ValueError, RuntimeError):
        return None
    # Only centres the SMILES actually assigns are compared: a source that
    # leaves a centre unspecified is not contradicted by whichever
    # configuration the embedding happened to pick for it. Heavy-atom indices
    # are shared between ``template`` and its ``AddHs`` copy.
    for template_atom in template.GetAtoms():
        if template_atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
            molecule.GetAtomWithIdx(template_atom.GetIdx()).SetChiralTag(
                Chem.ChiralType.CHI_UNSPECIFIED
            )
    return _tetrahedral_only_canonical_smiles(molecule)


def stereo_agrees_with_frame(
    isomeric_smiles: str | None, atomic_numbers: np.ndarray, positions: np.ndarray
) -> bool:
    """Invariant 9 for one row.

    False when ``isomeric_smiles`` is null, empty or does not parse: without a
    molecule there is nothing the frame could agree with. Otherwise true when
    the SMILES assigns no tetrahedral centre, or the frame's perceived centres
    equal the SMILES's.
    """
    if not isomeric_smiles:
        return False
    molecule = Chem.MolFromSmiles(isomeric_smiles)
    if molecule is None:
        return False
    if not _assigns_tetrahedral_centre(molecule):
        return True
    perceived = stereochemistry_from_positions(
        isomeric_smiles, atomic_numbers, positions
    )
    return perceived is not None and perceived == _tetrahedral_only_canonical_smiles(
        molecule
    )


def tetrahedral_stereo_smiles(isomeric_smiles: str) -> str | None:
    """Canonical SMILES of ``isomeric_smiles`` keeping tetrahedral stereo only.

    This is the reference that :func:`stereochemistry_from_frame` is compared
    against. Double-bond and imine geometry is deliberately dropped from both
    sides: a source SMILES usually leaves such bonds unspecified, while
    perceiving them from a 3D frame always yields E or Z, which would make a
    correct frame look like a mismatch. Returns ``None`` for an unparsable
    SMILES.
    """
    molecule = Chem.MolFromSmiles(isomeric_smiles)
    if molecule is None:
        return None
    return _tetrahedral_only_canonical_smiles(molecule)


def _tetrahedral_only_canonical_smiles(molecule: Chem.Mol) -> str:
    """Clear every double-bond stereo mark, then canonicalise without explicit H."""
    stripped = Chem.RWMol(molecule)
    for bond in stripped.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    return Chem.MolToSmiles(Chem.RemoveHs(stripped.GetMol()))


def _assigns_tetrahedral_centre(molecule: Chem.Mol) -> bool:
    """Whether ``molecule`` has at least one assigned tetrahedral centre.

    Invariant 9 compares a frame's perceived stereochemistry to the SMILES only
    for those molecules.
    """
    centres = Chem.FindMolChiralCenters(
        molecule, includeUnassigned=False, useLegacyImplementation=False
    )
    return len(centres) > 0


def _check_smiles_columns(table: pd.DataFrame, spec: DatasetSpec) -> list[str]:
    """Invariant 7 for the SMILES columns: every value present and non-empty.

    In a bundle every value must also parse. A dataset is not parsed here,
    because this check runs on every read and a corpus holds millions of rows:
    its ``isomeric_smiles`` is parsed row by row by invariant 9, where a value
    that does not parse fails, at write and in ``verify``.
    """
    problems: list[str] = []
    for column_name in SMILES_COLUMNS:
        if column_name not in table.columns or not _is_string_column(
            table[column_name]
        ):
            continue
        column = table[column_name]
        missing = column.isna() | (column.astype(object) == "")
        if missing.any():
            problems.append(
                f"7: {column_name} is null or empty in "
                f"{_summarise_rows(np.flatnonzero(missing.to_numpy()).tolist())}"
            )
        if spec.has_structures:
            continue
        present = column[~missing]
        unparsable = {
            text for text in present.unique() if Chem.MolFromSmiles(text) is None
        }
        if unparsable:
            rows = np.flatnonzero(column.isin(unparsable).to_numpy()).tolist()
            problems.append(
                f"7: {column_name} does not parse in {_summarise_rows(rows)}"
            )
    return problems


def _check_columns(table: pd.DataFrame, spec: DatasetSpec) -> list[str]:
    """Invariant 7: exactly the declared columns, in order, with the right dtypes."""
    problems: list[str] = []
    expected = spec.expected_columns()
    if list(table.columns) != expected:
        problems.append(f"7: columns {list(table.columns)} != expected {expected}")
    for column_name in ("structure_id", "stereoisomer_id", "molecule_id"):
        if column_name in table.columns and table[column_name].dtype != np.int64:
            problems.append(
                f"7: {column_name} is {table[column_name].dtype}, not int64"
            )
    if (
        "enantiomer_of" in table.columns
        and str(table["enantiomer_of"].dtype) != "Int64"
    ):
        problems.append(
            f"7: enantiomer_of is {table['enantiomer_of'].dtype}, not nullable Int64"
        )
    string_columns = [
        name
        for name in ("isomeric_smiles", "nonisomeric_smiles", *spec.split_columns())
        if name in table.columns
    ]
    for column_name in string_columns:
        if not _is_string_column(table[column_name]):
            problems.append(
                f"7: {column_name} is {table[column_name].dtype}, not a string dtype"
            )
    for column_name in ("total_charge", "multiplicity"):
        if column_name in table.columns and table[column_name].dtype != np.float64:
            problems.append(
                f"7: {column_name} is {table[column_name].dtype}, not float64"
            )
    problems += _check_smiles_columns(table, spec)
    for label in spec.labels:
        problems += _check_label_column(table, label)
    return problems


def _check_label_column(table: pd.DataFrame, label: LabelColumn) -> list[str]:
    """One label column: float64; 0/1 for classification; in range for multiclass."""
    if label.name not in table.columns:
        return []
    column = table[label.name]
    if column.dtype != np.float64:
        return [f"7: label column {label.name} is {column.dtype}, not float64"]
    values = column.to_numpy()
    present = values[~np.isnan(values)]
    if label.task_type is TaskType.classification:
        out_of_range = present[(present != 0.0) & (present != 1.0)]
        expected = "0 or 1"
    elif label.task_type is TaskType.multiclass and label.n_classes is not None:
        out_of_range = present[
            (present < 0)
            | (present > label.n_classes - 1)
            | (present != np.floor(present))
        ]
        expected = f"0 .. {label.n_classes - 1}"
    else:
        return []
    if len(out_of_range) == 0:
        return []
    return [
        f"7: {label.task_type.value} label {label.name} has {len(out_of_range)} "
        f"values outside {expected} (e.g. {out_of_range[0]})"
    ]


def _check_identity(table: pd.DataFrame, spec: DatasetSpec) -> list[str]:
    """Invariants 1-3: row identity, id-to-SMILES maps, nesting."""
    problems: list[str] = []
    if not set(spec.fixed_columns()).issubset(table.columns):
        return problems  # invariant 7 already reported the missing columns
    if spec.has_structures:
        if not np.array_equal(
            table["structure_id"].to_numpy(), np.arange(len(table), dtype="int64")
        ):
            problems.append("1: structure_id is not the dense row index 0 … N-1")
    elif table["stereoisomer_id"].duplicated().any():
        problems.append(
            "1: a bundle holds one row per stereoisomer, but stereoisomer_id repeats"
        )
    if spec.smiles:
        if table.groupby("stereoisomer_id")["isomeric_smiles"].nunique().max() > 1:
            problems.append(
                "2: a stereoisomer_id maps to more than one isomeric_smiles"
            )
        if table.groupby("molecule_id")["nonisomeric_smiles"].nunique().max() > 1:
            problems.append("2: a molecule_id maps to more than one nonisomeric_smiles")
    if table.groupby("stereoisomer_id")["molecule_id"].nunique().max() > 1:
        problems.append("3: a stereoisomer_id spans more than one molecule_id")
    return problems


def _check_enantiomer_relation(table: pd.DataFrame, spec: DatasetSpec) -> list[str]:
    """Invariant 4: one partner per stereoisomer; symmetric, irreflexive,
    resolvable, within one ``molecule_id``."""
    problems: list[str] = []
    if not set(spec.fixed_columns()).issubset(table.columns):
        return problems
    identity = table[["stereoisomer_id", "molecule_id", "enantiomer_of"]]
    partners_per_stereoisomer = identity.groupby("stereoisomer_id")[
        "enantiomer_of"
    ].nunique(dropna=False)
    inconsistent = partners_per_stereoisomer.index[partners_per_stereoisomer > 1]
    if len(inconsistent):
        problems.append(
            "4: enantiomer_of differs between the rows of stereoisomer "
            f"{_summarise_ids(inconsistent.tolist())}"
        )
    pairs = identity.drop_duplicates("stereoisomer_id")
    pairs = pairs[pairs["enantiomer_of"].notna()].astype({"enantiomer_of": "int64"})
    reflexive = pairs["enantiomer_of"] == pairs["stereoisomer_id"]
    if reflexive.any():
        problems.append(
            "4: enantiomer_of is reflexive for stereoisomer "
            f"{_summarise_ids(pairs.loc[reflexive, 'stereoisomer_id'].tolist())}"
        )
    partners = identity.drop_duplicates("stereoisomer_id").rename(
        columns={
            "stereoisomer_id": "enantiomer_of",
            "molecule_id": "partner_molecule_id",
            "enantiomer_of": "partner_points_at",
        }
    )
    matched = pairs[~reflexive].merge(
        partners, on="enantiomer_of", how="left", indicator=True
    )
    found = matched["_merge"] == "both"
    absent = matched[~found]
    for stereoisomer_id, partner_id in zip(
        absent["stereoisomer_id"], absent["enantiomer_of"], strict=True
    ):
        problems.append(
            f"4: enantiomer_of of {stereoisomer_id} points at stereoisomer "
            f"{partner_id}, which is absent"
        )
    matched = matched[found]
    asymmetric = matched[
        matched["partner_points_at"].isna()
        | (matched["partner_points_at"] != matched["stereoisomer_id"]).fillna(True)
    ]
    for stereoisomer_id, partner_id in zip(
        asymmetric["stereoisomer_id"], asymmetric["enantiomer_of"], strict=True
    ):
        problems.append(
            f"4: enantiomer_of is not symmetric for {stereoisomer_id} -> {partner_id}"
        )
    across = matched[matched["partner_molecule_id"] != matched["molecule_id"]]
    for stereoisomer_id, partner_id in zip(
        across["stereoisomer_id"], across["enantiomer_of"], strict=True
    ):
        problems.append(
            f"4: enantiomer pair {stereoisomer_id} <-> {partner_id} spans "
            "two molecule_ids"
        )
    if spec.evaluation is not None and spec.evaluation.require_enantiomer_pairs:
        orphans = int(table["enantiomer_of"].isna().sum())
        if orphans:
            problems.append(
                f"4: require_enantiomer_pairs is set but {orphans} rows have a "
                "null enantiomer_of"
            )
    return problems


def _check_splits(table: pd.DataFrame, spec: DatasetSpec) -> list[str]:
    """Invariants 5 and 6: allowed values, and no leakage across ``split_group``."""
    problems: list[str] = []
    evaluation = spec.evaluation
    if evaluation is None or evaluation.split_group not in table.columns:
        return problems
    for column_name in evaluation.split_columns:
        if column_name not in table.columns:
            continue
        values = set(table[column_name].dropna().unique()) - SPLIT_VALUES
        if values or table[column_name].isna().any():
            offending = sorted(str(value) for value in values)
            if table[column_name].isna().any():
                offending.append("<NA>")
            problems.append(
                f"5: split column {column_name} has values {offending} outside "
                f"{sorted(SPLIT_VALUES)}"
            )
        if table.groupby(evaluation.split_group)[column_name].nunique().max() > 1:
            problems.append(
                f"6: split column {column_name} is not constant within "
                f"{evaluation.split_group}"
            )
    return problems


def validate_table(spec: DatasetSpec, table: pd.DataFrame) -> list[str]:
    """The cheap invariants (1-7) of a bundle or dataset table; empty means valid."""
    return [
        *_check_columns(table, spec),
        *_check_identity(table, spec),
        *_check_enantiomer_relation(table, spec),
        *_check_splits(table, spec),
    ]


_GEOMETRY_REASONS = {
    "max_atoms": "more than max_atoms atoms",
    "no_heavy_atom": "no heavy atom",
    "zero_hydrogen": "no hydrogen while reject_zero_hydrogen is set",
    "hydrogen_heavy_ratio": "a hydrogen/heavy ratio below the limit",
    "element_gate": "an element outside the declared element set",
    "min_interatomic_distance": "two atoms closer than min_interatomic_distance",
}


def _frame_problems(
    isomeric_smiles: Sequence[str | None] | None,
    structures: Structures,
    limits: GeometryLimits,
    first_row: int,
) -> tuple[list[int], dict[str, list[int]]]:
    """Invariant 9 mismatches and invariant 10 offenders of one chunk of rows."""
    allowed = limits.allowed_atomic_numbers()
    mismatched: list[int] = []
    offenders: dict[str, list[int]] = {name: [] for name in _GEOMETRY_REASONS}
    for offset, (numbers, positions) in enumerate(structures.frames()):
        row_index = first_row + offset
        if isomeric_smiles is not None and not stereo_agrees_with_frame(
            isomeric_smiles[offset], numbers, positions
        ):
            mismatched.append(row_index)
        for violation in frame_geometry_violations(numbers, positions, limits, allowed):
            offenders[violation].append(row_index)
    return mismatched, offenders


def _frame_problems_of_chunk(
    arguments: tuple[Sequence[str | None] | None, Structures, GeometryLimits, int],
) -> tuple[list[int], dict[str, list[int]]]:
    return _frame_problems(*arguments)


def structure_problems(
    spec: DatasetSpec,
    table: pd.DataFrame,
    parts: Iterable[Structures],
    limits: GeometryLimits,
    *,
    workers: int = 1,
) -> list[str]:
    """The per-frame invariants of a dataset: 8 (one frame per row), 9 and 10.

    ``parts`` are the structures in record order, in any number of pieces
    (one :class:`Structures`, or the shards of a corpus streamed one at a
    time); :class:`StructureChecker` runs over them.
    """
    checker = StructureChecker(spec, table, limits, workers=workers)
    for part in parts:
        checker.check(part)
    return checker.problems()


class StructureChecker:
    """Invariants 8 (one frame per row), 9 and 10 over structures that arrive in parts.

    :meth:`check` takes the next part (its rows follow the previous parts'),
    :meth:`problems` reports. Invariant 9 applies to a dataset that declares
    ``smiles: true``: a row whose ``isomeric_smiles`` is null or does not parse
    fails it, and a row whose SMILES assigns a tetrahedral centre fails it when
    the frame disagrees. With ``workers > 1`` each part is checked in chunks in
    a process pool.
    """

    def __init__(
        self,
        spec: DatasetSpec,
        table: pd.DataFrame,
        limits: GeometryLimits,
        *,
        workers: int = 1,
    ) -> None:
        self._smiles = (
            [
                value if isinstance(value, str) else None
                for value in table["isomeric_smiles"]
            ]
            if spec.smiles
            else None
        )
        self._rows = len(table)
        self._limits = limits
        self._workers = workers
        self._checked = 0
        self._mismatched: list[int] = []
        self._offenders: dict[str, list[int]] = {name: [] for name in _GEOMETRY_REASONS}

    def check(self, part: Structures) -> None:
        """Check the next ``len(part)`` rows against their frames."""
        first_row = self._checked
        self._checked += len(part)
        if self._checked > self._rows:
            return  # more frames than rows; problems() reports the count
        count = len(part)
        chunk_size = max(1, -(-count // max(1, self._workers * 8)))
        arguments = [
            (
                None
                if self._smiles is None
                else self._smiles[first_row + start : first_row + stop],
                part.slice(start, stop),
                self._limits,
                first_row + start,
            )
            for start, stop in (
                (start, min(start + chunk_size, count))
                for start in range(0, count, chunk_size)
            )
        ]
        results = process_map(
            _frame_problems_of_chunk, arguments, workers=self._workers, chunk_size=1
        )
        for mismatched, offenders in results:
            self._mismatched += mismatched
            for name, rows in offenders.items():
                self._offenders[name] += rows

    def problems(self) -> list[str]:
        if self._checked != self._rows:
            return [f"8: {self._checked} structures for {self._rows} table rows"]
        problems: list[str] = []
        if self._mismatched:
            problems.append(
                "9: geometry disagrees with isomeric_smiles for "
                f"{_summarise_rows(self._mismatched)}"
            )
        for name, reason in _GEOMETRY_REASONS.items():
            if self._offenders[name]:
                problems.append(
                    f"10: {reason} in {_summarise_rows(self._offenders[name])}"
                )
        return problems
