"""Synthetic prepared-benchmark bundles for the prepare-side tests.

The same idea as ``tests/test_bundle_format.py``'s six-row fixture, but with
**one row per stereoisomer** — the shape a ``smiles``-stage bundle actually has
and the shape ``expand_to_conformers`` requires. Two variants:

* the default six rows carry one enantiomer pair, an achiral molecule, a meso
  compound, a lone chiral molecule and a second achiral molecule;
* the ``require_enantiomer_pairs`` variant is three enantiomer pairs, because
  invariant 4 refuses a null ``enantiomer_of`` when that flag is set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
