"""Synthetic prepared-benchmark bundles for the prepare- and eval-side tests.

Two families, both real bundles written through ``write_bundle`` so every
invariant of §1.1 is enforced on the fixture itself:

**``smiles``-stage** (``write_smiles_bundle``) — six rows, one per
stereoisomer, the shape ``expand_to_conformers`` requires. The default six
carry one enantiomer pair, an achiral molecule, a meso compound, a lone chiral
molecule and a second achiral molecule; the ``require_enantiomer_pairs``
variant is three enantiomer pairs, because invariant 4 refuses a null
``enantiomer_of`` when that flag is set.

**``conformers``-stage** (``write_conformers_bundle`` + ``ingest_tiny_bundle``)
— ten achiral rows with caller-supplied tasks, labels and split columns, plus
the real ``ingest_benchmark`` task that turns one into a zarr. Every eval test
builds its zarr this way, so the ingest path and the eval path cannot drift.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem

from remedi.data_handling.bundle import (
    BenchmarkSpec,
    BenchmarkTask,
    Bundle,
    BundleCounts,
    BundleProvenance,
    GeometryLimits,
    PreparerRecord,
    SourceRecord,
    assign_identity,
    write_bundle,
)
from remedi.data_handling.dataset.tasks import ElementSet, TaskType

#: One enantiomer pair, an achiral ring, a meso compound, a lone chiral
#: molecule and a small achiral molecule — six distinct stereoisomers.
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

#: Which split each ``molecule_id`` lands in, so invariant 6 holds by
#: construction for ``split_group: molecule_id``.
_SPLIT_BY_MOLECULE = ["train", "valid", "test", "train", "unassigned", "train"]
_ALTERNATIVE_SPLIT_BY_MOLECULE = ["test", "train", "train", "valid", "train", "valid"]


def make_smiles_spec(
    *, dataset_id: str = "synthetic6", require_enantiomer_pairs: bool = False
) -> BenchmarkSpec:
    """A ``smiles``-stage spec with one classification and one regression task."""
    return BenchmarkSpec(
        dataset_id=dataset_id,
        description="Six synthetic stereoisomers for the prepare-side tests.",
        tasks=[
            BenchmarkTask(name="activity", task_type=TaskType.classification),
            BenchmarkTask(name="logp", task_type=TaskType.regression),
        ],
        metrics=["AUROC", "RMSE"],
        stage="smiles",
        split_columns=["split", "split__random_s1"],
        default_split="split",
        split_group="molecule_id",
        require_enantiomer_pairs=require_enantiomer_pairs,
        source_kind="synthetic",
    )


def make_smiles_bundle(
    *, dataset_id: str = "synthetic6", require_enantiomer_pairs: bool = False
) -> Bundle:
    """Six rows, one per stereoisomer, with labels, two splits and provenance."""
    spec = make_smiles_spec(
        dataset_id=dataset_id, require_enantiomer_pairs=require_enantiomer_pairs
    )
    smiles = PAIRED_SIX_SMILES if require_enantiomer_pairs else UNPAIRED_SIX_SMILES
    table = assign_identity(smiles).to_frame()
    table["activity"] = np.array([1.0, 0.0, np.nan, 1.0, np.nan, 0.0])
    table["logp"] = np.array([-3.0, -3.0, 1.5, np.nan, 0.6, -0.3])
    table["split"] = [
        _SPLIT_BY_MOLECULE[int(molecule_id)] for molecule_id in table["molecule_id"]
    ]
    table["split__random_s1"] = [
        _ALTERNATIVE_SPLIT_BY_MOLECULE[int(molecule_id)]
        for molecule_id in table["molecule_id"]
    ]
    provenance = BundleProvenance(
        dataset_id=dataset_id,
        preparer=PreparerRecord(
            repo="molsuit/remedi-data", script="tests/helpers/bundle_fixtures.py"
        ),
        source=SourceRecord(package_versions={"synthetic": "1.0"}),
        geometry_limits=GEOMETRY_LIMITS,
        counts=BundleCounts(source_molecules=len(smiles), dropped={"element_gate": 2}),
        notices=["synthetic fixture — carries forward to the conformers stage"],
    )
    return Bundle(
        spec=spec,
        table=table[spec.expected_columns()],
        structures=None,
        provenance=provenance,
    )


def write_smiles_bundle(
    root: Path,
    *,
    dataset_id: str = "synthetic6",
    require_enantiomer_pairs: bool = False,
) -> Path:
    """Write the six-row ``smiles``-stage bundle to ``root/<dataset_id>``."""
    bundle = make_smiles_bundle(
        dataset_id=dataset_id, require_enantiomer_pairs=require_enantiomer_pairs
    )
    return write_bundle(bundle, Path(root) / dataset_id)


# ------------------------------------------------- conformers-stage fixtures

#: Ten small achiral molecules. Achiral on purpose: invariant 9 (perceive the
#: stereochemistry back out of the frame) only runs on a SMILES with an
#: assigned tetrahedral centre, so an achiral fixture is immune to the ETKDG
#: seed and stays a *fast* fixture rather than a chemistry test.
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

#: Six train, two valid, two test — enough for a learner fit with a held-out
#: fold, and the shape the eval tests assert their row counts against.
TEN_ROW_SPLIT: list[str] = ["train"] * 6 + ["valid"] * 2 + ["test"] * 2


def embed_frames(smiles_values: Sequence[str]) -> list[Atoms]:
    """One ETKDG + MMFF frame per SMILES, stamped with its ``structure_id``.

    A fixed ETKDG seed, unlike the production task, because a fixture must not
    move between runs; §1.2's "no reproduction contract" applies to published
    coordinates, not to ten test molecules.
    """
    frames: list[Atoms] = []
    for structure_id, smiles in enumerate(smiles_values):
        molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed = 0xF00D
        if AllChem.EmbedMolecule(molecule, parameters) != 0:
            raise ValueError(f"the fixture SMILES {smiles!r} failed to embed")
        AllChem.MMFFOptimizeMolecule(molecule, maxIters=100)
        frames.append(
            Atoms(
                numbers=[atom.GetAtomicNum() for atom in molecule.GetAtoms()],
                positions=molecule.GetConformer().GetPositions(),
                pbc=[0, 0, 0],
                info={"structure_id": structure_id},
            )
        )
    return frames


def make_conformers_spec(
    *,
    dataset_id: str,
    tasks: list[BenchmarkTask],
    metrics: list[str],
    split_columns: list[str] | None = None,
    extra_columns: list[str] | None = None,
) -> BenchmarkSpec:
    """A ``conformers``-stage spec over caller-supplied tasks and metrics."""
    return BenchmarkSpec(
        dataset_id=dataset_id,
        description="A synthetic conformers-stage bundle for the eval tests.",
        tasks=tasks,
        metrics=metrics,
        stage="conformers",
        geometry_origin="etkdg_mmff",
        split_columns=split_columns or ["split"],
        default_split=(split_columns or ["split"])[0],
        split_group="stereoisomer_id",
        extra_columns=extra_columns or [],
        source_kind="synthetic",
    )


def make_conformers_bundle(
    *,
    dataset_id: str,
    tasks: list[BenchmarkTask],
    metrics: list[str],
    targets: np.ndarray,
    smiles: Sequence[str] = tuple(ACHIRAL_TEN_SMILES),
    splits: dict[str, list[str]] | None = None,
) -> Bundle:
    """A conformers-stage bundle with one embedded frame per SMILES.

    Args:
        targets: ``(n_rows, n_tasks)``; ``NaN`` marks a missing label, exactly
            as the format's only missingness encoding does.
        splits: split column name -> one value per row. Defaults to a single
            ``split`` column with the 6/2/2 partition.
    """
    splits = splits or {"split": list(TEN_ROW_SPLIT)}
    spec = make_conformers_spec(
        dataset_id=dataset_id,
        tasks=tasks,
        metrics=metrics,
        split_columns=list(splits),
    )
    smiles_values = list(smiles)
    table = assign_identity(smiles_values).to_frame()
    targets = np.asarray(targets, dtype=np.float64).reshape(len(smiles_values), -1)
    for column, task in enumerate(tasks):
        table[task.name] = targets[:, column]
    for column_name, values in splits.items():
        table[column_name] = list(values)
    provenance = BundleProvenance(
        dataset_id=dataset_id,
        preparer=PreparerRecord(
            repo="molsuit/remedi-data", script="tests/helpers/bundle_fixtures.py"
        ),
        source=SourceRecord(package_versions={"synthetic": "1.0"}),
        counts=BundleCounts(source_molecules=len(smiles_values)),
    )
    return Bundle(
        spec=spec,
        table=table[spec.expected_columns()],
        structures=embed_frames(smiles_values),
        provenance=provenance,
    )


def write_conformers_bundle(root: Path, **kwargs) -> Path:
    """Write :func:`make_conformers_bundle` to ``root/<dataset_id>``."""
    bundle = make_conformers_bundle(**kwargs)
    return write_bundle(bundle, Path(root) / bundle.spec.dataset_id)


def ingest_tiny_bundle(
    benchmark_root: Path, output_root: Path, dataset_id: str
) -> Path:
    """Run the real ``ingest_benchmark`` task on one bundle. Returns the zarr path.

    Every eval test builds its zarr through this rather than hand-writing zarr
    arrays, so the ingest path and the eval path cannot drift apart.
    """
    from remedi.data_handling.prepare import IngestBenchmarkConfig, PrepareContext

    task = IngestBenchmarkConfig(
        dataset_ids=[dataset_id],
        batch_size=4,
        atom_chunk=8,
        molecule_chunk=4,
        atom_chunks_per_shard=4,
        molecule_chunks_per_shard=4,
    )
    context = PrepareContext(
        smiles_bundle_root=Path(benchmark_root),
        benchmark_root=Path(benchmark_root),
        output_root=Path(output_root),
    )
    list(task.run(context))
    return Path(output_root) / dataset_id
