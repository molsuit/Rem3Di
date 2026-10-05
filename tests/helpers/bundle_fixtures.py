"""Synthetic bundles and datasets for the format, build and evaluation tests.

Two families, both written through the real writers so every invariant of the
format is enforced on the fixture itself:

**Bundles** (``make_bundle`` / ``write_smiles_bundle``): six rows, one per
stereoisomer, which is what a SMILES preparer writes and what
``expand_to_structures`` and ``build_datasets`` read. The default six carry one
enantiomer pair, an achiral molecule, a meso compound, a lone chiral molecule
and a second achiral molecule; the ``require_enantiomer_pairs`` variant is three
enantiomer pairs, because invariant 4 refuses a null ``enantiomer_of`` when that
flag is set. ``smiles=False`` drops the SMILES columns and ``evaluation=False``
drops the evaluation block and its split columns (a corpus).

**Datasets** (``write_small_dataset``): a zarr with the three table files
beside it, written by ``write_dataset`` from one seeded ETKDG frame per SMILES
(or caller-supplied frames). Every evaluation test reads a dataset made this
way, so the writer and the evaluation side cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

from remedi.data_handling.bundle import (
    Bundle,
    BundleCounts,
    BundleProvenance,
    DatasetSpec,
    EvalMetric,
    EvaluationSpec,
    LabelColumn,
    PreparerRecord,
    SourceRecord,
    assign_identity,
    write_bundle,
)
from remedi.data_handling.chemistry.conformers import charge_and_multiplicity
from remedi.data_handling.chemistry.elements import ElementSet
from remedi.data_handling.chemistry.geometry import GeometryLimits
from remedi.data_handling.dataset.tasks import TaskType
from remedi.data_handling.dataset_build import ZarrLayout, write_dataset

#: One enantiomer pair, an achiral ring, a meso compound, a lone chiral
#: molecule and a small achiral molecule: six distinct stereoisomers.
UNPAIRED_SIX_SMILES: list[str] = [
    "C[C@H](N)C(=O)O",  # L-alanine
    "C[C@@H](N)C(=O)O",  # D-alanine
    "c1ccccc1O",  # phenol
    "O[C@@H]([C@@H](O)C(O)=O)C(O)=O",  # meso-tartaric acid
    "C[C@H](O)CC",  # (S)-2-butanol, partner absent
    "CCO",  # ethanol
]

#: Three enantiomer pairs, so every row has a partner.
PAIRED_SIX_SMILES: list[str] = [
    "C[C@H](N)C(=O)O",  # L-alanine
    "C[C@@H](N)C(=O)O",  # D-alanine
    "C[C@H](O)CC",  # (S)-2-butanol
    "C[C@@H](O)CC",  # (R)-2-butanol
    "OC[C@@H](O)C=O",  # D-glyceraldehyde
    "OC[C@H](O)C=O",  # L-glyceraldehyde
]

GEOMETRY_LIMITS = GeometryLimits(
    max_atoms=100,
    elements=ElementSet.mace_off,
    reject_zero_hydrogen=True,
    min_hydrogen_heavy_ratio=0.0,
    min_interatomic_distance=0.5,
)

#: Small chunks and shards, so a ten-structure zarr already spans several
#: chunks and writer batches.
SMALL_ZARR_LAYOUT = ZarrLayout(
    atom_chunk=8,
    molecule_chunk=4,
    atom_chunks_per_shard=4,
    molecule_chunks_per_shard=4,
    batch_size=4,
)

#: Which split each ``molecule_id`` lands in, so invariant 6 holds by
#: construction for ``split_group: molecule_id``.
_SPLIT_BY_MOLECULE = ["train", "valid", "test", "train", "unassigned", "train"]
_ALTERNATIVE_SPLIT_BY_MOLECULE = ["test", "train", "train", "valid", "train", "valid"]

#: The two label columns every bundle fixture carries.
BUNDLE_LABELS: list[LabelColumn] = [
    LabelColumn(name="activity", task_type=TaskType.classification),
    LabelColumn(name="logp", task_type=TaskType.regression),
]


def make_provenance(
    dataset_id: str,
    *,
    source_molecules: int = 0,
    geometry_limits: GeometryLimits = GEOMETRY_LIMITS,
) -> BundleProvenance:
    """A provenance block as a preparer would fill it, before the writer runs."""
    return BundleProvenance(
        dataset_id=dataset_id,
        preparer=PreparerRecord(
            repo="molsuit/Rem3Di", script="tests/helpers/bundle_fixtures.py"
        ),
        source=SourceRecord(package_versions={"synthetic": "1.0"}),
        geometry_limits=geometry_limits,
        counts=BundleCounts(
            source_molecules=source_molecules, dropped={"element_gate": 2}
        ),
        notices=["synthetic fixture"],
    )


def make_bundle_spec(
    *,
    dataset_id: str = "synthetic6",
    require_enantiomer_pairs: bool = False,
    smiles: bool = True,
    evaluation: bool = True,
) -> DatasetSpec:
    """A bundle spec with one classification and one regression label."""
    return DatasetSpec(
        dataset_id=dataset_id,
        description="Six synthetic stereoisomers.",
        smiles=smiles,
        labels=list(BUNDLE_LABELS),
        evaluation=(
            EvaluationSpec(
                metrics=[EvalMetric.auroc, EvalMetric.rmse],
                split_columns=["split", "split__random_s1"],
                default_split="split",
                split_group="molecule_id",
                require_enantiomer_pairs=require_enantiomer_pairs,
            )
            if evaluation
            else None
        ),
        source_kind="synthetic",
    )


def make_bundle(
    *,
    dataset_id: str = "synthetic6",
    require_enantiomer_pairs: bool = False,
    smiles: bool = True,
    evaluation: bool = True,
) -> Bundle:
    """Six rows, one per stereoisomer, with labels, two splits and provenance."""
    spec = make_bundle_spec(
        dataset_id=dataset_id,
        require_enantiomer_pairs=require_enantiomer_pairs,
        smiles=smiles,
        evaluation=evaluation,
    )
    smiles_values = (
        PAIRED_SIX_SMILES if require_enantiomer_pairs else UNPAIRED_SIX_SMILES
    )
    table = assign_identity(smiles_values).to_frame()
    table["activity"] = np.array([1.0, 0.0, np.nan, 1.0, np.nan, 0.0])
    table["logp"] = np.array([-3.0, -3.0, 1.5, np.nan, 0.6, -0.3])
    table["split"] = [
        _SPLIT_BY_MOLECULE[int(molecule_id)] for molecule_id in table["molecule_id"]
    ]
    table["split__random_s1"] = [
        _ALTERNATIVE_SPLIT_BY_MOLECULE[int(molecule_id)]
        for molecule_id in table["molecule_id"]
    ]
    return Bundle(
        spec=spec,
        table=table[spec.expected_columns()],
        provenance=make_provenance(dataset_id, source_molecules=len(smiles_values)),
    )


def write_smiles_bundle(
    root: Path, *, dataset_id: str = "synthetic6", **variant: bool
) -> Path:
    """Write :func:`make_bundle` to ``root/<dataset_id>``; returns that directory.

    ``variant`` takes the keyword flags of :func:`make_bundle`.
    """
    directory = Path(root) / dataset_id
    write_bundle(make_bundle(dataset_id=dataset_id, **variant), directory)
    return directory


# ------------------------------------------------------------------ datasets

#: Ten small achiral molecules. Achiral on purpose: invariant 9 (perceive the
#: stereochemistry back out of the frame) only runs on a SMILES with an
#: assigned tetrahedral centre, so an achiral fixture is immune to the ETKDG
#: seed and stays a fast fixture rather than a chemistry test.
ACHIRAL_TEN_SMILES: list[str] = [
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
]

#: Six train, two valid, two test: enough for a learner fit with a held-out
#: fold, and the shape the evaluation tests assert their row counts against.
TEN_ROW_SPLIT: list[str] = ["train"] * 6 + ["valid"] * 2 + ["test"] * 2


def embed_frame(isomeric_smiles: str, seed: int = 0xF00D) -> Atoms:
    """One seeded ETKDG + MMFF frame in RDKit ``AddHs`` atom order.

    A fixed ETKDG seed, unlike production, because a fixture must not move
    between runs; the "no reproduction contract" of generated coordinates
    applies to published data, not to ten test molecules.
    """
    molecule = Chem.AddHs(Chem.MolFromSmiles(isomeric_smiles))
    parameters = rdDistGeom.ETKDGv3()
    parameters.randomSeed = seed
    if rdDistGeom.EmbedMolecule(molecule, parameters) != 0:
        raise ValueError(f"the fixture SMILES {isomeric_smiles!r} failed to embed")
    rdForceFieldHelpers.MMFFOptimizeMolecule(molecule, maxIters=100)
    return Atoms(
        numbers=[atom.GetAtomicNum() for atom in molecule.GetAtoms()],
        positions=molecule.GetConformer().GetPositions(),
        pbc=[0, 0, 0],
    )


def make_dataset_spec(
    *,
    dataset_id: str,
    labels: list[LabelColumn],
    metrics: list[EvalMetric] | None,
    split_columns: list[str] | None = None,
    extra_columns: list[str] | None = None,
) -> DatasetSpec:
    """A dataset spec with SMILES; ``metrics=None`` makes it a corpus (no evaluation)."""
    split_columns = split_columns or ["split"]
    return DatasetSpec(
        dataset_id=dataset_id,
        description="A synthetic dataset for the evaluation tests.",
        smiles=True,
        geometry_origin="etkdg_mmff",
        labels=labels,
        extra_columns=extra_columns or [],
        evaluation=(
            EvaluationSpec(
                metrics=metrics,
                split_columns=split_columns,
                default_split=split_columns[0],
                split_group="stereoisomer_id",
            )
            if metrics is not None
            else None
        ),
        source_kind="synthetic",
    )


def make_dataset_table(
    spec: DatasetSpec,
    smiles_values: Sequence[str],
    targets: np.ndarray | None,
    splits: dict[str, list[str]] | None,
) -> pd.DataFrame:
    """One row per SMILES (one structure each), in ``spec``'s column order.

    Args:
        targets: ``(n_rows, n_labels)``; ``NaN`` marks a missing label.
        splits: split column name -> one value per row.
    """
    table = assign_identity(list(smiles_values)).to_frame()
    table.insert(0, "structure_id", np.arange(len(table), dtype="int64"))
    charges = [charge_and_multiplicity(smiles) for smiles in table["isomeric_smiles"]]
    table["total_charge"] = [charge for charge, _ in charges]
    table["multiplicity"] = [multiplicity for _, multiplicity in charges]
    if spec.labels:
        assert targets is not None
        label_values = np.asarray(targets, dtype=np.float64).reshape(len(table), -1)
        for column, label in enumerate(spec.labels):
            table[label.name] = label_values[:, column]
    for column_name, values in (splits or {}).items():
        table[column_name] = list(values)
    return table[spec.expected_columns()]


def write_small_dataset(
    root: Path,
    *,
    dataset_id: str,
    labels: list[LabelColumn],
    metrics: list[EvalMetric] | None,
    targets: np.ndarray | None,
    smiles: Sequence[str] = tuple(ACHIRAL_TEN_SMILES),
    splits: dict[str, list[str]] | None = None,
    structures: Sequence[Atoms] | None = None,
) -> Path:
    """Write one dataset to ``root/<dataset_id>`` through ``write_dataset``.

    Args:
        metrics: the evaluation metrics; ``None`` writes a corpus without an
            evaluation block (and without split columns).
        targets: ``(n_rows, n_labels)``; ``NaN`` marks a missing label.
        splits: split column name -> one value per row. Defaults to a single
            ``split`` column with the 6/2/2 partition when ``metrics`` is set.
        structures: one frame per SMILES; seeded ETKDG frames by default.
    """
    if metrics is not None and splits is None:
        splits = {"split": list(TEN_ROW_SPLIT)}
    spec = make_dataset_spec(
        dataset_id=dataset_id,
        labels=labels,
        metrics=metrics,
        split_columns=list(splits) if splits else None,
    )
    smiles_values = list(smiles)
    table = make_dataset_table(spec, smiles_values, targets, splits)
    directory = Path(root) / dataset_id
    write_dataset(
        spec,
        table,
        (
            list(structures)
            if structures is not None
            else [embed_frame(smiles) for smiles in smiles_values]
        ),
        make_provenance(dataset_id, source_molecules=len(smiles_values)),
        directory,
        layout=SMALL_ZARR_LAYOUT,
    )
    return directory
