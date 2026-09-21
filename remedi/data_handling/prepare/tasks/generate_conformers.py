"""``generate_conformers``: bundle(``smiles``) -> bundle(``conformers``).

Build-order step 3 of ``BENCHMARK_DATA_FORMAT.md``. ETKDG + MMFF94 embedding
runs *here*, in Rem3Di, as a named prepare task writing into the bundle — not
in ``remedi-data`` (the knobs were tuned here) and not buried inside zarr
construction (that is what made the conformers invisible). §1.2 spells out
why; §6b records that no ETKDG seed is pinned, so a re-run yields equivalent,
not identical, coordinates. **The published coordinates are the artifact, not
the recipe.**

The embedding itself is
:func:`remedi.data_handling.dataset_creation.utils.embed_one_smiles`, verbatim
— the same function ``ConformerGenerationStage`` drives for the pretraining
corpora, with the same ``max_embed_attempts`` / ``max_mmff_steps`` /
``mmff_non_bonded_threshold`` defaults validated by the CYP timing experiment.

What this task adds on top of the embedding is the bookkeeping the stage
boundary needs:

* invariant 9 after MMFF — a relaxation that walks a centre through planarity
  silently turns L into D while the SMILES column still says L. Frames whose
  perceived stereochemistry disagrees with ``isomeric_smiles`` are dropped into
  ``counts.dropped.stereo_mismatch_after_embedding``.
* invariant 10 — frames outside ``geometry_limits`` are dropped by guard name
  rather than sinking the whole dataset in ``write_bundle``.
* the row expansion and the orphan rule, delegated to
  :func:`remedi.data_handling.bundle.expand_to_conformers`.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import rdkit
from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field

from remedi.data_handling.bundle import (
    CONFORMER_EMBEDDING_FAILED,
    ENANTIOMER_PARTNER_FAILED,
    Bundle,
    ConformerGenerationRecord,
    EtkdgParameters,
    GeometryLimits,
    MmffParameters,
    content_hash_of_table,
    expand_to_conformers,
    geometry_limit_violations,
    has_assigned_tetrahedral_centre,
    read_bundle,
    stereochemistry_from_frame,
    tetrahedral_stereo_smiles,
    write_bundle,
)
from remedi.data_handling.bundle.bundle import SPEC_FILENAME
from remedi.data_handling.dataset.tasks import ElementSet
from remedi.data_handling.dataset_creation.conformer_timing import (
    ConformerTimingRecord,
    write_timings_jsonl,
)
from remedi.data_handling.dataset_creation.utils import embed_one_smiles
from remedi.data_handling.prepare.context import BundleRoots, PrepareContext
from remedi.data_handling.prepare.tasks.per_dataset import PerDatasetPrepareTask
from remedi.evaluation.results import EvalResult, PydanticResult

logger = logging.getLogger(__name__)

#: ``counts.dropped`` key for a frame whose geometry disagrees with its SMILES.
STEREO_MISMATCH_AFTER_EMBEDDING = "stereo_mismatch_after_embedding"

#: Written next to the bundle files, as ``ConformerGenerationStage`` writes it
#: next to the zarr today.
TIMINGS_FILENAME = "conformer_timings.jsonl"

#: The element gate the SMILES-stage filter already applied, plus the two
#: geometry guards ``FilterAtomsStage`` adds.
# No total-atom cap: the smiles-stage filter already bounds *heavy* atoms
# (recorded in provenance ``smiles_filter``), and a 100-heavy-atom molecule
# carries up to ~150 atoms once hydrogens are added. Today's benchmark builds
# apply no atom limit on the geometry side either.
DEFAULT_GEOMETRY_LIMITS = GeometryLimits(
    max_atoms=None,
    elements=ElementSet.mace_off,
    reject_zero_hydrogen=True,
    min_hydrogen_heavy_ratio=0.0,
    min_interatomic_distance=0.5,
)


# --------------------------------------------------------------------- results


class GenerateConformersSummary(BaseModel):
    """What one dataset's conformer generation did — the task's artifact."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    bundle_path: Path
    skipped: bool = False
    rows_in: int = 0
    rows_out: int = 0
    frames_out: int = 0
    n_conformers_requested: int = 1
    dropped: dict[str, int] = Field(default_factory=dict)
    parent_bundle_content_sha256: str | None = None
    content_sha256: str | None = None
    elapsed_seconds: float = 0.0


@dataclass
class _EmbeddingOutcome:
    """Frames kept per stereoisomer plus the reasons the others were dropped."""

    frames_by_stereoisomer: dict[int, list[Atoms]] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)
    timings: list[ConformerTimingRecord] = field(default_factory=list)

    def count_drop(self, reason: str, amount: int = 1) -> None:
        if amount:
            self.dropped[reason] = self.dropped.get(reason, 0) + amount


# ------------------------------------------------------------------- embedding


def _frames_from_embedding(positions, atomic_numbers) -> list[Atoms]:
    """One clean :class:`ase.Atoms` per embedded conformer.

    ``atoms.info`` is left empty; ``expand_to_conformers`` stamps the dense
    ``structure_id`` that invariant 8 checks.
    """
    return [
        Atoms(numbers=atomic_numbers, positions=positions[index], pbc=[0, 0, 0])
        for index in range(positions.shape[0])
    ]


def _keep_frame(
    isomeric_smiles: str,
    atoms: Atoms,
    *,
    canonical_isomeric: str | None,
    check_stereochemistry: bool,
    limits: GeometryLimits,
    allowed_symbols: set[str] | None,
) -> str | None:
    """``None`` to keep the frame, else the ``counts.dropped`` key to drop it under."""
    if check_stereochemistry:
        perceived = stereochemistry_from_frame(isomeric_smiles, atoms)
        if perceived is None or perceived != canonical_isomeric:
            return STEREO_MISMATCH_AFTER_EMBEDDING
    violations = geometry_limit_violations(atoms, limits, allowed_symbols)
    return violations[0] if violations else None


class GenerateConformersConfig(PerDatasetPrepareTask):
    """ETKDG + MMFF94 over every row of one or more ``smiles``-stage bundles.

    ``dataset_ids`` defaults to every bundle under
    ``PrepareContext.smiles_bundle_root``.
    """

    kind: Literal["generate_conformers"] = "generate_conformers"

    n_conformers: int = Field(default=1, ge=1)
    # RDKit's ``params.maxIterations``. 200 covers essentially anything ETKDG
    # can embed; raising it inflates wall time on pathological molecules
    # without improving yield (§1.2, slurm-4686108).
    max_embed_attempts: int = Field(default=200, ge=1)
    max_mmff_steps: int = Field(default=100, ge=0)
    mmff_non_bonded_threshold: float = Field(default=100.0, gt=0.0)

    #: ``None`` means ``os.cpu_count()``. ``1`` runs in-process, no pool.
    n_workers: int | None = Field(default=None, ge=1)
    #: False skips a dataset whose ``conformers``-stage bundle already exists.
    overwrite: bool = False

    geometry_limits: GeometryLimits = DEFAULT_GEOMETRY_LIMITS

    # ----------------------------------------------------------------- running

    def discovery_root(self, roots: BundleRoots) -> Path:
        """This task consumes ``smiles``-stage bundles."""
        return roots.smiles_bundle_root

    def run(self, ctx: PrepareContext) -> Iterator[EvalResult]:
        """Generate conformers for every resolved dataset id.

        Raises:
            FileNotFoundError: if a named bundle directory does not exist.
            ValueError: if a bundle is not at the ``smiles`` stage, or if the
                expanded bundle fails the frame/row alignment gate.
        """
        for dataset_id in self.resolve_dataset_ids(ctx.roots()):
            yield self._run_one_dataset(ctx, dataset_id)

    def _run_one_dataset(self, ctx: PrepareContext, dataset_id: str) -> EvalResult:
        started = time.monotonic()
        output_directory = Path(ctx.benchmark_root) / dataset_id
        summary_path = Path("generate_conformers") / f"{dataset_id}.yaml"

        if (output_directory / SPEC_FILENAME).is_file() and not self.overwrite:
            logger.info(
                "%s: %s already exists, skipping (overwrite is False)",
                dataset_id,
                output_directory,
            )
            return PydanticResult(
                file_name=summary_path,
                obj=GenerateConformersSummary(
                    dataset_id=dataset_id,
                    bundle_path=output_directory,
                    skipped=True,
                    n_conformers_requested=self.n_conformers,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )

        source_directory = Path(ctx.smiles_bundle_root) / dataset_id
        bundle_smiles = read_bundle(source_directory)
        if bundle_smiles.spec.stage != "smiles":
            raise ValueError(
                f"{source_directory} is a {bundle_smiles.spec.stage!r}-stage bundle; "
                "generate_conformers consumes a 'smiles'-stage bundle "
                "(a source-supplied conformers bundle needs no generation)"
            )

        outcome = self._embed_bundle(bundle_smiles)
        expanded = expand_to_conformers(
            bundle_smiles,
            outcome.frames_by_stereoisomer,
            geometry_origin="etkdg_mmff",
            conformer_record=self._conformer_record(bundle_smiles),
        )
        result_bundle = expanded.bundle
        result_bundle.provenance.geometry_limits = self.geometry_limits
        result_bundle.provenance.counts.dropped = self._merged_dropped_counts(
            bundle_smiles, outcome, expanded.dropped_counts
        )
        _assert_alignment(result_bundle, bundle_smiles)

        write_bundle(result_bundle, output_directory)
        write_timings_jsonl(outcome.timings, output_directory / TIMINGS_FILENAME)
        logger.info(
            "%s: %d stereoisomers -> %d conformer rows (dropped %s)",
            dataset_id,
            len(bundle_smiles.table),
            len(result_bundle.table),
            result_bundle.provenance.counts.dropped,
        )

        return PydanticResult(
            file_name=summary_path,
            obj=GenerateConformersSummary(
                dataset_id=dataset_id,
                bundle_path=output_directory,
                rows_in=len(bundle_smiles.table),
                rows_out=len(result_bundle.table),
                frames_out=len(result_bundle.structures or []),
                n_conformers_requested=self.n_conformers,
                dropped=dict(result_bundle.provenance.counts.dropped),
                parent_bundle_content_sha256=content_hash_of_table(bundle_smiles.table),
                content_sha256=content_hash_of_table(result_bundle.table),
                elapsed_seconds=time.monotonic() - started,
            ),
        )

    # ------------------------------------------------------------- internals

    def _conformer_record(self, bundle_smiles: Bundle) -> ConformerGenerationRecord:
        return ConformerGenerationRecord(
            rdkit_version=rdkit.__version__,
            n_conformers_requested=self.n_conformers,
            etkdg=EtkdgParameters(
                version="ETKDGv3", max_iterations=self.max_embed_attempts
            ),
            mmff=MmffParameters(
                max_iterations=self.max_mmff_steps,
                non_bonded_threshold=self.mmff_non_bonded_threshold,
            ),
            parent_bundle_content_sha256=content_hash_of_table(bundle_smiles.table),
        )

    def _merged_dropped_counts(
        self,
        bundle_smiles: Bundle,
        outcome: _EmbeddingOutcome,
        expansion_counts: dict[str, int],
    ) -> dict[str, int]:
        """The parent's drop reasons plus this stage's, without double counting.

        ``expand_to_conformers`` attributes *every* stereoisomer that produced
        no frame to ``conformer_embedding_failed``, including the ones whose
        frames this task rejected for stereochemistry or geometry. Those are
        already counted under their own reason here, so only the orphan count
        is taken from the expansion.
        """
        merged = dict(bundle_smiles.provenance.counts.dropped)
        stage_counts = dict(outcome.dropped)
        # The three stage-boundary reasons are always reported, zero included,
        # so a reviewer sees that they were checked (§1.4's counts block).
        stage_counts.setdefault(CONFORMER_EMBEDDING_FAILED, 0)
        stage_counts.setdefault(STEREO_MISMATCH_AFTER_EMBEDDING, 0)
        stage_counts[ENANTIOMER_PARTNER_FAILED] = expansion_counts.get(
            ENANTIOMER_PARTNER_FAILED, 0
        )
        for reason, count in stage_counts.items():
            merged[reason] = merged.get(reason, 0) + count
        return merged

    def _embed_bundle(self, bundle_smiles: Bundle) -> _EmbeddingOutcome:
        """Embed every row, then gate each frame on invariants 9 and 10."""
        table = bundle_smiles.table
        smiles_by_stereoisomer = {
            int(stereoisomer_id): str(isomeric_smiles)
            for stereoisomer_id, isomeric_smiles in zip(
                table["stereoisomer_id"], table["isomeric_smiles"], strict=True
            )
        }
        outcome = _EmbeddingOutcome()
        allowed_symbols = self.geometry_limits.allowed_element_symbols()
        for stereoisomer_id, embedding in self._embed_all(smiles_by_stereoisomer):
            isomeric_smiles = smiles_by_stereoisomer[stereoisomer_id]
            outcome.timings.append(embedding.timing)
            if (
                embedding.timing.status != "ok"
                or embedding.positions is None
                or embedding.atomic_numbers is None
            ):
                outcome.frames_by_stereoisomer[stereoisomer_id] = []
                outcome.count_drop(CONFORMER_EMBEDDING_FAILED)
                logger.debug(
                    "embedding failed for %s: %s (%s)",
                    isomeric_smiles,
                    embedding.timing.status,
                    embedding.timing.error_msg,
                )
                continue
            outcome.frames_by_stereoisomer[stereoisomer_id] = self._gate_frames(
                isomeric_smiles,
                _frames_from_embedding(embedding.positions, embedding.atomic_numbers),
                outcome=outcome,
                allowed_symbols=allowed_symbols,
            )
        return outcome

    def _gate_frames(
        self,
        isomeric_smiles: str,
        frames: list[Atoms],
        *,
        outcome: _EmbeddingOutcome,
        allowed_symbols: set[str] | None,
    ) -> list[Atoms]:
        canonical_isomeric = tetrahedral_stereo_smiles(isomeric_smiles)
        check_stereochemistry = has_assigned_tetrahedral_centre(isomeric_smiles)
        kept: list[Atoms] = []
        for atoms in frames:
            reason = _keep_frame(
                isomeric_smiles,
                atoms,
                canonical_isomeric=canonical_isomeric,
                check_stereochemistry=check_stereochemistry,
                limits=self.geometry_limits,
                allowed_symbols=allowed_symbols,
            )
            if reason is None:
                kept.append(atoms)
            else:
                outcome.count_drop(reason)
        return kept

    def _embed_all(self, smiles_by_stereoisomer: dict[int, str]):
        """Yield ``(stereoisomer_id, EmbedResult)`` for every stereoisomer.

        ``n_workers=1`` runs in-process — that keeps a single-dataset debug run
        (and the tests, which monkeypatch the embedding function) out of the
        pool, where a patched callable would not survive pickling.
        """
        arguments = (
            self.n_conformers,
            self.max_embed_attempts,
            self.max_mmff_steps,
            self.mmff_non_bonded_threshold,
        )
        n_workers = self.n_workers or os.cpu_count() or 1
        if n_workers == 1:
            for stereoisomer_id, isomeric_smiles in smiles_by_stereoisomer.items():
                yield stereoisomer_id, embed_one_smiles(isomeric_smiles, *arguments)
            return
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(embed_one_smiles, isomeric_smiles, *arguments): (
                    stereoisomer_id
                )
                for stereoisomer_id, isomeric_smiles in smiles_by_stereoisomer.items()
            }
            for future in as_completed(futures):
                yield futures[future], future.result()


def _assert_alignment(result_bundle: Bundle, bundle_smiles: Bundle) -> None:
    """The §8 step-3 gate: frame/row alignment and id carry-through.

    ``write_bundle`` validates all of §1.1 too, but asserting it here names the
    failure as *this task's* bug rather than as a format violation found three
    layers down.

    Raises:
        ValueError: on any misalignment.
    """
    table = result_bundle.table
    structures = result_bundle.structures
    problems: list[str] = []
    if structures is None:
        problems.append("the expanded bundle carries no structures")
    elif len(structures) != len(table):
        problems.append(f"{len(structures)} frames for {len(table)} rows")
    else:
        misaligned = [
            row_index
            for row_index, atoms in enumerate(structures)
            if int(atoms.info.get("structure_id", -1)) != row_index
            or int(table["structure_id"].iloc[row_index]) != row_index
        ]
        if misaligned:
            problems.append(
                f"frame/row structure_id disagrees at rows {misaligned[:5]}"
            )

    parent = bundle_smiles.table.set_index("stereoisomer_id")
    for column in ("molecule_id", "isomeric_smiles", "nonisomeric_smiles"):
        carried = table.drop_duplicates("stereoisomer_id").set_index("stereoisomer_id")[
            column
        ]
        if not carried.equals(parent.loc[carried.index, column]):
            problems.append(f"{column} did not carry through the stage boundary")
    if problems:
        raise ValueError(
            "generate_conformers produced a misaligned bundle for "
            f"{result_bundle.spec.dataset_id}:\n  " + "\n  ".join(problems)
        )
