"""The ``generate_conformers`` prepare task (``BENCHMARK_DATA_FORMAT.md`` §8 step 3).

bundle(``smiles``) -> bundle(``conformers``): frame/row alignment, id
carry-through, invariant 9 after MMFF, the orphan rule at the stage boundary,
and the idempotent skip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from rdkit import Chem

from remedi.data_handling.bundle import (
    CONFORMER_EMBEDDING_FAILED,
    ENANTIOMER_PARTNER_FAILED,
    canonical_smiles_pair,
    content_hash_of_table,
    has_assigned_tetrahedral_centre,
    read_bundle,
    stereochemistry_from_frame,
    tetrahedral_stereo_smiles,
)
from remedi.data_handling.dataset_creation.conformer_timing import (
    ConformerTimingRecord,
    EmbedResult,
)
from remedi.data_handling.dataset_creation.utils import embed_one_smiles
from remedi.data_handling.prepare import (
    STEREO_MISMATCH_AFTER_EMBEDDING,
    TIMINGS_FILENAME,
    GenerateConformersConfig,
    PrepareManifest,
    prepare,
)
from remedi.data_handling.prepare.tasks import generate_conformers as task_module

from .helpers.bundle_fixtures import write_smiles_bundle

DATASET_ID = "synthetic6"


# --------------------------------------------------------------------- helpers


def build_manifest(
    tmp_path: Path,
    *,
    require_enantiomer_pairs: bool = False,
    overwrite: bool = False,
    dataset_ids: list[str] | None = None,
) -> PrepareManifest:
    """A one-task manifest over one written ``smiles``-stage bundle."""
    smiles_root = tmp_path / "bundles"
    write_smiles_bundle(
        smiles_root,
        dataset_id=DATASET_ID,
        require_enantiomer_pairs=require_enantiomer_pairs,
    )
    return PrepareManifest(
        smiles_bundle_root=smiles_root,
        benchmark_root=tmp_path / "benchmark_bundles",
        output_root=tmp_path / "prepare_out",
        tasks=[
            GenerateConformersConfig(
                dataset_ids=dataset_ids,
                n_conformers=1,
                n_workers=1,
                overwrite=overwrite,
            )
        ],
    )


def _canonical(smiles: str) -> str:
    return canonical_smiles_pair(smiles).isomeric


def patch_embedding_to_fail_for(
    monkeypatch: pytest.MonkeyPatch, failing_smiles: str
) -> None:
    """Make the embedding of exactly one stereoisomer report ``embed_failed``.

    ``n_workers=1`` keeps the run in-process, so the patched callable is the
    one the task actually calls (it would not survive pickling into a pool).
    """
    target = _canonical(failing_smiles)

    def fake_embed_one_smiles(
        isomeric_smiles: str,
        n_confs: int,
        max_embed_attempts: int,
        max_opt_iters: int,
        mmff_non_bonded_thresh: float = 100.0,
    ) -> EmbedResult:
        if _canonical(isomeric_smiles) == target:
            return EmbedResult(
                nonisomeric_smiles=None,
                positions=None,
                atomic_numbers=None,
                timing=ConformerTimingRecord(
                    isomeric_smiles=isomeric_smiles,
                    n_atoms=-1,
                    n_confs_requested=n_confs,
                    n_confs_emitted=0,
                    t_embed_s=0.0,
                    t_mmff_s=0.0,
                    status="embed_failed",
                    error_msg="injected failure",
                ),
            )
        return embed_one_smiles(
            isomeric_smiles,
            n_confs,
            max_embed_attempts,
            max_opt_iters,
            mmff_non_bonded_thresh,
        )

    monkeypatch.setattr(task_module, "embed_one_smiles", fake_embed_one_smiles)


# ------------------------------------------------------------------ happy path


def test_generate_conformers_writes_a_valid_conformers_bundle(tmp_path: Path) -> None:
    manifest = build_manifest(tmp_path)
    parent = read_bundle(manifest.smiles_bundle_root / DATASET_ID)

    report = prepare(manifest)

    assert report.n_failed == 0
    assert report.n_tasks == 1

    bundle = read_bundle(manifest.benchmark_root / DATASET_ID)
    assert bundle.spec.stage == "conformers"
    assert bundle.spec.geometry_origin == "etkdg_mmff"
    assert bundle.structures is not None

    # One frame per row, densely re-numbered, frames carrying their own id.
    assert len(bundle.structures) == len(bundle.table) == 6
    assert np.array_equal(bundle.table["structure_id"].to_numpy(), np.arange(6))
    assert [atoms.info["structure_id"] for atoms in bundle.structures] == list(range(6))

    # The four identity columns carry through unchanged (§1.2).
    for column in (
        "stereoisomer_id",
        "molecule_id",
        "isomeric_smiles",
        "nonisomeric_smiles",
        "enantiomer_of",
        "activity",
        "logp",
        "split",
        "split__random_s1",
    ):
        pd.testing.assert_series_equal(
            bundle.table[column], parent.table[column], check_names=False
        )

    # Invariant 9 holds for every row with an assigned tetrahedral centre.
    for row_index, isomeric_smiles in enumerate(bundle.table["isomeric_smiles"]):
        if not has_assigned_tetrahedral_centre(str(isomeric_smiles)):
            continue
        perceived = stereochemistry_from_frame(
            str(isomeric_smiles), bundle.structures[row_index]
        )
        assert perceived == tetrahedral_stereo_smiles(str(isomeric_smiles))

    record = bundle.provenance.conformers
    assert record is not None
    assert record.rdkit_version == Chem.rdBase.rdkitVersion
    assert record.n_conformers_requested == 1
    assert record.etkdg.version == "ETKDGv3"
    assert record.etkdg.max_iterations == 200
    assert record.mmff.max_iterations == 100
    assert record.mmff.non_bonded_threshold == 100.0
    assert record.parent_bundle_content_sha256 == content_hash_of_table(parent.table)

    # geometry_limits, notices, source and preparer all reach the far side.
    assert bundle.provenance.geometry_limits.max_atoms is None
    assert bundle.provenance.geometry_limits.min_interatomic_distance == 0.5
    assert bundle.provenance.notices == parent.provenance.notices
    assert bundle.provenance.source == parent.provenance.source
    assert bundle.provenance.preparer == parent.provenance.preparer
    # The parent's own drop reasons survive the merge; nothing was lost here.
    dropped = bundle.provenance.counts.dropped
    assert dropped["element_gate"] == 2
    assert dropped[CONFORMER_EMBEDDING_FAILED] == 0
    assert dropped[ENANTIOMER_PARTNER_FAILED] == 0

    timings_path = manifest.benchmark_root / DATASET_ID / TIMINGS_FILENAME
    assert timings_path.is_file()
    assert len(timings_path.read_text().strip().splitlines()) == 6

    summary = yaml.safe_load(
        (
            manifest.output_root / "generate_conformers" / f"{DATASET_ID}.yaml"
        ).read_text()
    )
    assert summary["dataset_id"] == DATASET_ID
    assert summary["rows_in"] == 6
    assert summary["rows_out"] == 6
    assert summary["frames_out"] == 6
    assert summary["skipped"] is False


def test_multiple_conformers_multiply_the_rows(tmp_path: Path) -> None:
    manifest = build_manifest(tmp_path)
    manifest.tasks[0] = manifest.tasks[0].model_copy(update={"n_conformers": 3})

    assert prepare(manifest).n_failed == 0

    bundle = read_bundle(manifest.benchmark_root / DATASET_ID)
    assert bundle.structures is not None
    # Survivors are multiplied by their conformer count; structure_id is
    # re-assigned densely over the expanded rows (§1.2).
    assert len(bundle.table) == len(bundle.structures) > 6
    assert bundle.table["stereoisomer_id"].value_counts().max() <= 3
    assert np.array_equal(
        bundle.table["structure_id"].to_numpy(), np.arange(len(bundle.table))
    )
    assert bundle.provenance.conformers is not None
    assert bundle.provenance.conformers.n_conformers_requested == 3


# ------------------------------------------------------------- the orphan rule


def test_embedding_failure_nullifies_the_surviving_partner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = build_manifest(tmp_path)
    patch_embedding_to_fail_for(monkeypatch, "C[C@@H](N)C(=O)O")  # D-alanine
    parent = read_bundle(manifest.smiles_bundle_root / DATASET_ID)
    d_alanine_id = int(parent.table.loc[1, "stereoisomer_id"])
    l_alanine_id = int(parent.table.loc[0, "stereoisomer_id"])

    assert prepare(manifest).n_failed == 0

    bundle = read_bundle(manifest.benchmark_root / DATASET_ID)
    assert len(bundle.table) == 5
    assert d_alanine_id not in set(bundle.table["stereoisomer_id"])
    orphan = bundle.table[bundle.table["stereoisomer_id"] == l_alanine_id]
    assert len(orphan) == 1
    assert orphan["enantiomer_of"].isna().all()

    dropped = bundle.provenance.counts.dropped
    assert dropped[CONFORMER_EMBEDDING_FAILED] == 1
    assert dropped[ENANTIOMER_PARTNER_FAILED] == 1
    assert dropped[STEREO_MISMATCH_AFTER_EMBEDDING] == 0


def test_embedding_failure_drops_the_orphan_when_pairs_are_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = build_manifest(tmp_path, require_enantiomer_pairs=True)
    patch_embedding_to_fail_for(monkeypatch, "C[C@@H](N)C(=O)O")  # D-alanine

    assert prepare(manifest).n_failed == 0

    bundle = read_bundle(manifest.benchmark_root / DATASET_ID)
    # Both alanine rows are gone: the failure and its now-partnerless mirror.
    assert len(bundle.table) == 4
    assert bundle.table["enantiomer_of"].notna().all()
    dropped = bundle.provenance.counts.dropped
    assert dropped[CONFORMER_EMBEDDING_FAILED] == 1
    assert dropped[ENANTIOMER_PARTNER_FAILED] == 1


def test_a_stereo_mismatch_after_mmff_is_dropped_and_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A frame whose perceived stereochemistry disagrees is dropped, not written.

    The mismatch is injected at the perception step: MMFF walking a centre
    through planarity is exactly what invariant 9 exists to catch, and it is
    not reproducible on demand.
    """
    manifest = build_manifest(tmp_path)
    target = _canonical("C[C@@H](N)C(=O)O")  # D-alanine

    def fake_stereochemistry_from_frame(isomeric_smiles: str, atoms) -> str | None:
        if _canonical(isomeric_smiles) == target:
            return "C[C@H](N)C(=O)O"  # the mirror image: a silent L/D flip
        return stereochemistry_from_frame(isomeric_smiles, atoms)

    monkeypatch.setattr(
        task_module, "stereochemistry_from_frame", fake_stereochemistry_from_frame
    )

    assert prepare(manifest).n_failed == 0

    bundle = read_bundle(manifest.benchmark_root / DATASET_ID)
    assert len(bundle.table) == 5
    dropped = bundle.provenance.counts.dropped
    assert dropped[STEREO_MISMATCH_AFTER_EMBEDDING] == 1
    # The frame was rejected for its stereochemistry, not miscounted as a
    # failed embedding.
    assert dropped[CONFORMER_EMBEDDING_FAILED] == 0
    assert dropped[ENANTIOMER_PARTNER_FAILED] == 1


# ------------------------------------------------------------- guards and skip


def test_overwrite_false_skips_an_existing_output(tmp_path: Path) -> None:
    manifest = build_manifest(tmp_path)
    assert prepare(manifest).n_failed == 0
    written = manifest.benchmark_root / DATASET_ID
    before = (written / "table.parquet").stat().st_mtime_ns

    assert prepare(manifest).n_failed == 0

    assert (written / "table.parquet").stat().st_mtime_ns == before
    summary = yaml.safe_load(
        (
            manifest.output_root / "generate_conformers" / f"{DATASET_ID}.yaml"
        ).read_text()
    )
    assert summary["skipped"] is True
    assert summary["rows_out"] == 0


def test_overwrite_true_regenerates(tmp_path: Path) -> None:
    manifest = build_manifest(tmp_path)
    assert prepare(manifest).n_failed == 0
    manifest.tasks[0] = manifest.tasks[0].model_copy(update={"overwrite": True})

    assert prepare(manifest).n_failed == 0

    summary = yaml.safe_load(
        (
            manifest.output_root / "generate_conformers" / f"{DATASET_ID}.yaml"
        ).read_text()
    )
    assert summary["skipped"] is False
    assert summary["rows_out"] == 6


def test_a_conformers_stage_input_is_refused(tmp_path: Path) -> None:
    """A source-supplied 3D bundle needs no generation and must not be re-embedded."""
    manifest = build_manifest(tmp_path)
    assert prepare(manifest).n_failed == 0

    # Point a second run at the conformers-stage bundle the first one wrote.
    second = PrepareManifest(
        smiles_bundle_root=manifest.benchmark_root,
        benchmark_root=tmp_path / "second_bundles",
        output_root=tmp_path / "second_out",
        tasks=[GenerateConformersConfig(dataset_ids=[DATASET_ID], n_workers=1)],
        keep_going=False,
    )
    with pytest.raises(ValueError, match=r"conformers.*-stage bundle"):
        prepare(second)


def test_frames_outside_the_geometry_limits_are_dropped(tmp_path: Path) -> None:
    manifest = build_manifest(tmp_path)
    manifest.tasks[0] = manifest.tasks[0].model_copy(
        update={
            "geometry_limits": manifest.tasks[0].geometry_limits.model_copy(
                update={"max_atoms": 10}
            )
        }
    )

    assert prepare(manifest).n_failed == 0

    bundle = read_bundle(manifest.benchmark_root / DATASET_ID)
    assert bundle.structures is not None
    assert all(len(atoms) <= 10 for atoms in bundle.structures)
    assert bundle.provenance.counts.dropped["max_atoms"] > 0
    assert len(bundle.table) < 6
