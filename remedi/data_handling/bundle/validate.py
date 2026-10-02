"""The format's invariants (``BENCHMARK_DATA_FORMAT.md`` §1.1, made conditional by §10.2).

``validate_table`` checks the cheap, vectorised invariants every read asserts,
on a bundle table or a dataset table; ``structure_problems`` checks the
per-frame ones (9: geometry agrees with the SMILES, 10: geometry limits), which
run when a dataset is written and in ``verify``, not on every read. Every
message is prefixed with its invariant number; an empty list means valid.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from ase import Atoms
from rdkit import Chem

from remedi.data_handling.bundle.spec import (
    SPLIT_VALUES,
    DatasetSpec,
    LabelColumn,
)
from remedi.data_handling.chemistry.geometry import (
    GeometryLimits,
    geometry_violations,
)
from remedi.data_handling.dataset.tasks import TaskType

#: How many offending rows a single message names before it summarises.
_MAX_REPORTED_ROWS = 5


def _summarise_rows(row_indices: list[int]) -> str:
    if len(row_indices) <= _MAX_REPORTED_ROWS:
        return f"rows {row_indices}"
    shown = row_indices[:_MAX_REPORTED_ROWS]
    return f"{len(row_indices)} rows, first {shown}"


def _is_string_column(series: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)


def count_stereoisomer_straddling_constitutions(
    table: pd.DataFrame, split_column: str
) -> int:
    """Constitutions whose stereoisomers land in more than one fold (§1.1).

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
    """Tetrahedral stereo perceived from ``atoms`` for invariant 9.

    The result is a canonical SMILES with tetrahedral centres only (see
    :func:`tetrahedral_stereo_smiles` for why double-bond stereo is dropped),
    restricted to the centres ``isomeric_smiles`` assigns, and is meant to be
    compared to ``tetrahedral_stereo_smiles(isomeric_smiles)``.

    The molecular graph comes from ``isomeric_smiles`` (with explicit hydrogens
    added) and only the *stereochemistry* is re-perceived from the coordinates.
    This assumes the frame's atom order equals the RDKit ``AddHs`` atom order
    of the isomeric SMILES, which is exactly what ETKDG produces. Returns
    ``None`` when the molecule cannot be built that way; the caller then
    reports the row as failing invariant 9 rather than silently skipping it.
    """
    template = Chem.MolFromSmiles(isomeric_smiles)
    if template is None:
        return None
    molecule = Chem.AddHs(template)
    if molecule.GetNumAtoms() != len(atoms):
        return None
    expected_numbers = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
    if list(atoms.get_atomic_numbers()) != expected_numbers:
        return None
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index, position in enumerate(atoms.get_positions()):
        conformer.SetAtomPosition(index, [float(value) for value in position])
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


def has_assigned_tetrahedral_centre(isomeric_smiles: str) -> bool:
    """Whether ``isomeric_smiles`` declares at least one assigned tetrahedral centre.

    Invariant 9 only applies to those rows, so both the validator and the
    conformer generation gate on this before comparing a frame's
    perceived stereochemistry to the SMILES column.
    """
    molecule = Chem.MolFromSmiles(isomeric_smiles)
    if molecule is None:
        return False
    centres = Chem.FindMolChiralCenters(
        molecule, includeUnassigned=False, useLegacyImplementation=False
    )
    return len(centres) > 0


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
    """Invariant 4: symmetric, irreflexive, resolvable, within one molecule_id."""
    problems: list[str] = []
    if not set(spec.fixed_columns()).issubset(table.columns):
        return problems
    by_stereoisomer = table.drop_duplicates("stereoisomer_id").set_index(
        "stereoisomer_id"
    )
    partner_of = by_stereoisomer["enantiomer_of"]
    molecule_of = by_stereoisomer["molecule_id"]
    for stereoisomer_id, partner_id in partner_of.items():
        if pd.isna(partner_id):
            continue
        if partner_id == stereoisomer_id:
            problems.append(f"4: enantiomer_of is reflexive for {stereoisomer_id}")
            continue
        if partner_id not in partner_of.index:
            problems.append(
                f"4: enantiomer_of of {stereoisomer_id} points at stereoisomer "
                f"{partner_id}, which is absent"
            )
            continue
        back = partner_of[partner_id]
        if pd.isna(back) or back != stereoisomer_id:
            problems.append(
                f"4: enantiomer_of is not symmetric for {stereoisomer_id} -> {partner_id}"
            )
        if molecule_of[partner_id] != molecule_of[stereoisomer_id]:
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


def structure_problems(
    spec: DatasetSpec,
    table: pd.DataFrame,
    structures: Sequence[Atoms],
    limits: GeometryLimits,
) -> list[str]:
    """The per-frame invariants of a dataset: 8 (one frame per row), 9 and 10.

    Invariant 9 applies to rows whose ``isomeric_smiles`` assigns a tetrahedral
    centre, so only to a dataset that declares ``smiles: true``.
    """
    if len(structures) != len(table):
        return [f"8: {len(structures)} structures for {len(table)} table rows"]
    problems: list[str] = []
    if spec.smiles:
        mismatched = [
            row_index
            for row_index, (isomeric_smiles, atoms) in enumerate(
                zip(table["isomeric_smiles"], structures, strict=True)
            )
            if has_assigned_tetrahedral_centre(isomeric_smiles)
            and stereochemistry_from_frame(isomeric_smiles, atoms)
            != tetrahedral_stereo_smiles(isomeric_smiles)
        ]
        if mismatched:
            problems.append(
                "9: geometry disagrees with isomeric_smiles for "
                f"{_summarise_rows(mismatched)}"
            )
    allowed_symbols = limits.allowed_element_symbols()
    offenders: dict[str, list[int]] = {name: [] for name in _GEOMETRY_REASONS}
    for row_index, atoms in enumerate(structures):
        for violation in geometry_violations(atoms, limits, allowed_symbols):
            offenders[violation].append(row_index)
    problems += [
        f"10: {_GEOMETRY_REASONS[name]} in {_summarise_rows(rows)}"
        for name, rows in offenders.items()
        if rows
    ]
    return problems
