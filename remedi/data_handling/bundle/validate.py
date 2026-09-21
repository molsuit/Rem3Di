"""Invariants 1-10 of ``BENCHMARK_DATA_FORMAT.md`` §1.1.

``validate_bundle`` returns the list of violated invariants, each message
prefixed with its invariant number. An empty list means the bundle is valid.
These run on every ``write_bundle`` and every ``read_bundle``: they are cheap
and they are the whole point of having a format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from ase import Atoms
from rdkit import Chem

from remedi.data_handling.bundle.provenance import GeometryLimits
from remedi.data_handling.bundle.spec import (
    FIXED_COLUMNS,
    SPLIT_VALUES,
    BenchmarkSpec,
    BenchmarkTask,
)
from remedi.data_handling.dataset.tasks import TaskType

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from remedi.data_handling.bundle.bundle import Bundle

#: How many offending rows a single message names before it summarises.
_MAX_REPORTED_ROWS = 5


def _summarise_rows(row_indices: list[int]) -> str:
    if len(row_indices) <= _MAX_REPORTED_ROWS:
        return f"rows {row_indices}"
    shown = row_indices[:_MAX_REPORTED_ROWS]
    return f"{len(row_indices)} rows, first {shown}"


def _is_string_column(series: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)


def minimum_interatomic_distance(positions: np.ndarray) -> float:
    """Smallest distance between any two atoms, in Angstrom.

    Mirrors the guard ``FilterAtomsStage`` enforces today: MACE divides by a
    near-zero distance on overlapping atoms and emits NaN embeddings.
    """
    if len(positions) < 2:
        return float("inf")
    difference = positions[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(difference, axis=-1)
    np.fill_diagonal(distance, np.inf)
    return float(distance.min())


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
    This assumes the extxyz atom order equals the RDKit ``AddHs`` atom order of
    the isomeric SMILES, which is exactly what ETKDG produces and what the
    conformer stage writes. Returns ``None`` when the molecule cannot be built
    that way — the caller then reports the row as failing invariant 9 rather
    than silently skipping it.
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
    ``generate_conformers`` prepare task gate on this before comparing a frame's
    perceived stereochemistry to the SMILES column.
    """
    molecule = Chem.MolFromSmiles(isomeric_smiles)
    if molecule is None:
        return False
    centres = Chem.FindMolChiralCenters(
        molecule, includeUnassigned=False, useLegacyImplementation=False
    )
    return len(centres) > 0


def _check_fixed_column_dtypes(table: pd.DataFrame) -> list[str]:
    """The dtypes the §1.1 table declares for the six fixed columns."""
    problems: list[str] = []
    for column_name in ("structure_id", "stereoisomer_id", "molecule_id"):
        if column_name in table.columns and table[column_name].dtype != np.int64:
            problems.append(
                f"7: {column_name} is {table[column_name].dtype}, not int64"
            )
    for column_name in ("isomeric_smiles", "nonisomeric_smiles"):
        if column_name in table.columns and not _is_string_column(table[column_name]):
            problems.append(
                f"7: {column_name} is {table[column_name].dtype}, not a string dtype"
            )
    if (
        "enantiomer_of" in table.columns
        and str(table["enantiomer_of"].dtype) != "Int64"
    ):
        problems.append(
            f"7: enantiomer_of is {table['enantiomer_of'].dtype}, not nullable Int64"
        )
    return problems


def _check_task_column(table: pd.DataFrame, task: BenchmarkTask) -> list[str]:
    """One task column: float64, and in range when the task is multiclass."""
    if task.name not in table.columns:
        return []
    column = table[task.name]
    if column.dtype != np.float64:
        return [f"7: task column {task.name} is {column.dtype}, not float64"]
    if task.task_type is not TaskType.multiclass or task.n_classes is None:
        return []
    values = column.to_numpy()
    present = values[~np.isnan(values)]
    out_of_range = present[
        (present < 0) | (present > task.n_classes - 1) | (present != np.floor(present))
    ]
    if len(out_of_range) == 0:
        return []
    return [
        f"7: multiclass task {task.name} has {len(out_of_range)} values outside "
        f"0 .. {task.n_classes - 1} (e.g. {out_of_range[0]})"
    ]


def _check_columns(table: pd.DataFrame, spec: BenchmarkSpec) -> list[str]:
    """Invariant 7: exactly the declared columns, in order, with the right dtypes."""
    problems: list[str] = []
    expected = spec.expected_columns()
    if list(table.columns) != expected:
        problems.append(f"7: columns {list(table.columns)} != expected {expected}")
    problems += _check_fixed_column_dtypes(table)
    for task in spec.tasks:
        problems += _check_task_column(table, task)
    for column_name in spec.split_columns:
        if column_name in table.columns and not _is_string_column(table[column_name]):
            problems.append(
                f"7: split column {column_name} is {table[column_name].dtype}, "
                "not a string dtype"
            )
    return problems


def _check_identity(table: pd.DataFrame, spec: BenchmarkSpec) -> list[str]:
    """Invariants 1-4: dense row index, id-to-smiles maps, nesting, mirror relation."""
    problems: list[str] = []
    if not set(FIXED_COLUMNS).issubset(table.columns):
        return problems  # invariant 7 already reported the missing columns

    if not np.array_equal(
        table["structure_id"].to_numpy(), np.arange(len(table), dtype="int64")
    ):
        problems.append("1: structure_id is not the dense row index 0 … N-1")

    if table.groupby("stereoisomer_id")["isomeric_smiles"].nunique().max() > 1:
        problems.append("2: a stereoisomer_id maps to more than one isomeric_smiles")
    if table.groupby("molecule_id")["nonisomeric_smiles"].nunique().max() > 1:
        problems.append("2: a molecule_id maps to more than one nonisomeric_smiles")
    if table.groupby("stereoisomer_id")["molecule_id"].nunique().max() > 1:
        problems.append("3: a stereoisomer_id spans more than one molecule_id")

    problems += _check_enantiomer_relation(table, spec)
    return problems


def _check_enantiomer_relation(table: pd.DataFrame, spec: BenchmarkSpec) -> list[str]:
    """Invariant 4: symmetric, irreflexive, resolvable, within one molecule_id."""
    problems: list[str] = []
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
                f"{partner_id}, which is absent from the bundle"
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
    if spec.require_enantiomer_pairs:
        orphans = int(table["enantiomer_of"].isna().sum())
        if orphans:
            problems.append(
                f"4: require_enantiomer_pairs is set but {orphans} rows have a "
                "null enantiomer_of"
            )
    return problems


def _check_splits(table: pd.DataFrame, spec: BenchmarkSpec) -> list[str]:
    """Invariants 5 and 6: allowed values, and no leakage across ``split_group``."""
    problems: list[str] = []
    if spec.split_group not in table.columns:
        return problems
    for column_name in spec.split_columns:
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
        if table.groupby(spec.split_group)[column_name].nunique().max() > 1:
            problems.append(
                f"6: split column {column_name} is not constant within "
                f"{spec.split_group}"
            )
    return problems


def _check_structures(bundle: Bundle) -> list[str]:
    """Invariant 8: frame count and per-frame ``structure_id``."""
    problems: list[str] = []
    table = bundle.table
    if bundle.spec.stage == "smiles":
        if bundle.structures is not None:
            problems.append("8: a smiles-stage bundle must not carry structures")
        return problems
    if bundle.structures is None:
        problems.append("8: a conformers-stage bundle must carry structures")
        return problems
    if len(bundle.structures) != len(table):
        problems.append(f"8: {len(bundle.structures)} frames for {len(table)} rows")
    mislabelled = [
        index
        for index, atoms in enumerate(bundle.structures)
        if atoms.info.get("structure_id") != index
    ]
    if mislabelled:
        problems.append(
            f"8: frames not carrying their own structure_id in atoms.info: "
            f"{_summarise_rows(mislabelled)}"
        )
    return problems


def _check_geometry_matches_smiles(bundle: Bundle) -> list[str]:
    """Invariant 9: perceiving stereo from frame i reproduces row i's SMILES."""
    if bundle.structures is None or "isomeric_smiles" not in bundle.table.columns:
        return []
    n_rows = min(len(bundle.structures), len(bundle.table))
    has_centre: dict[str, bool] = {}
    reference: dict[str, str | None] = {}
    mismatched: list[int] = []
    for row_index in range(n_rows):
        isomeric_smiles = str(bundle.table["isomeric_smiles"].iloc[row_index])
        if isomeric_smiles not in has_centre:
            has_centre[isomeric_smiles] = has_assigned_tetrahedral_centre(
                isomeric_smiles
            )
            reference[isomeric_smiles] = tetrahedral_stereo_smiles(isomeric_smiles)
        if not has_centre[isomeric_smiles]:
            continue
        perceived = stereochemistry_from_frame(
            isomeric_smiles, bundle.structures[row_index]
        )
        if perceived is None or perceived != reference[isomeric_smiles]:
            mismatched.append(row_index)
    if mismatched:
        return [
            "9: geometry disagrees with isomeric_smiles for "
            f"{_summarise_rows(mismatched)}"
        ]
    return []


def geometry_limit_violations(
    atoms: Atoms, limits: GeometryLimits, allowed_symbols: set[str] | None
) -> list[str]:
    """Which of the invariant-10 guards one frame violates.

    Returns the guard names (``max_atoms``, ``element_gate``, ...), which double
    as ``counts.dropped`` keys: the ``generate_conformers`` prepare task calls
    this to drop a violating frame *before* ``write_bundle`` would refuse the
    whole bundle for it.
    """
    violations: list[str] = []
    if limits.max_atoms is not None and len(atoms) > limits.max_atoms:
        violations.append("max_atoms")
    atomic_numbers = atoms.get_atomic_numbers()
    n_hydrogen = int((atomic_numbers == 1).sum())
    n_heavy = int((atomic_numbers > 1).sum())
    if n_heavy == 0:
        violations.append("no_heavy_atom")
    elif limits.reject_zero_hydrogen and n_hydrogen == 0:
        violations.append("zero_hydrogen")
    elif (
        limits.min_hydrogen_heavy_ratio > 0.0
        and (n_hydrogen / n_heavy) < limits.min_hydrogen_heavy_ratio
    ):
        violations.append("hydrogen_heavy_ratio")
    if allowed_symbols is not None and not set(atoms.get_chemical_symbols()).issubset(
        allowed_symbols
    ):
        violations.append("element_gate")
    if (
        limits.min_interatomic_distance is not None
        and minimum_interatomic_distance(atoms.get_positions())
        < limits.min_interatomic_distance
    ):
        violations.append("min_interatomic_distance")
    return violations


def _check_geometry_limits(bundle: Bundle, limits: GeometryLimits) -> list[str]:
    """Invariant 10: the guards ``FilterAtomsStage`` enforces today."""
    if bundle.structures is None:
        return []
    allowed_symbols = limits.allowed_element_symbols()
    reasons = {
        "max_atoms": f"more than max_atoms={limits.max_atoms} atoms",
        "no_heavy_atom": "no heavy atom",
        "zero_hydrogen": "no hydrogen while reject_zero_hydrogen is set",
        "hydrogen_heavy_ratio": (
            f"a hydrogen/heavy ratio below {limits.min_hydrogen_heavy_ratio}"
        ),
        "element_gate": "an element outside the declared element set",
        "min_interatomic_distance": (
            "two atoms closer than min_interatomic_distance="
            f"{limits.min_interatomic_distance} A"
        ),
    }
    offenders: dict[str, list[int]] = {name: [] for name in reasons}
    for index, atoms in enumerate(bundle.structures):
        for violation in geometry_limit_violations(atoms, limits, allowed_symbols):
            offenders[violation].append(index)
    return [
        f"10: {reasons[name]} in {_summarise_rows(rows)}"
        for name, rows in offenders.items()
        if rows
    ]


def validate_bundle(bundle: Bundle) -> list[str]:
    """Return every violated invariant of §1.1; an empty list means valid."""
    problems: list[str] = []
    problems += _check_columns(bundle.table, bundle.spec)
    problems += _check_identity(bundle.table, bundle.spec)
    problems += _check_splits(bundle.table, bundle.spec)
    problems += _check_structures(bundle)
    problems += _check_geometry_matches_smiles(bundle)
    problems += _check_geometry_limits(bundle, bundle.provenance.geometry_limits)
    return problems
