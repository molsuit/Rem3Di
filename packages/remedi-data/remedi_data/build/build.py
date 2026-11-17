"""``dataset_build``: SMILES bundles -> datasets, then verify every dataset.

One YAML (:class:`DatasetBuildConfig`) drives one run. Each bundle under
``bundle_root`` is one task: conformers are generated with
:func:`~remedi_data.chemistry.conformers.embed_many`, every frame is gated on
MMFF having run, invariant 9 (stereochemistry survived MMFF) and invariant 10
(geometry limits), the bundle is expanded to one row per structure, and
:func:`~remedi_data.build.write.write_dataset` writes the zarr with the table
beside it. Each dataset under ``zarr_root`` (built here or
written directly by a preparer) is then verified. One failing dataset is one
failed row in ``status.yaml``; the rest still finish.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import rdkit
from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field
from rdkit import Chem

from remedi_data.build.expansion import ENANTIOMER_PARTNER_FAILED, expand_to_structures
from remedi_data.build.write import write_dataset
from remedi_data.chemistry.charge import charge_and_multiplicity
from remedi_data.chemistry.conformers import (
    ConformerEmbeddingConfig,
    ConformerGeometryOrigin,
    ConformerTimingRecord,
    embed_many,
    write_timings_jsonl,
)
from remedi_data.chemistry.elements import ElementSet
from remedi_data.chemistry.geometry import GeometryLimits, frame_geometry_violations
from remedi_data.dataset.config import DatasetConfig
from remedi_data.dataset.packed_structures import PackedStructures
from remedi_data.format import (
    Bundle,
    ConformerGenerationRecord,
    discover_bundles,
    discover_datasets,
    read_bundle,
    read_provenance,
    stereo_agrees_with_frame,
)
from remedi_data.format.verify import verify_dataset
from remedi_data.parallel import default_workers
from remedi_data.runs.results import EvalResult, PydanticResult
from remedi_data.runs.task_runner import (
    RunReport,
    run_and_record,
)

logger = logging.getLogger(__name__)

#: ``counts.dropped`` key for a stereoisomer whose embedding failed outright.
CONFORMER_EMBEDDING_FAILED = "conformer_embedding_failed"
#: ``counts.dropped`` key for a frame whose geometry disagrees with its SMILES.
STEREO_MISMATCH_AFTER_EMBEDDING = "stereo_mismatch_after_embedding"
#: ``counts.dropped`` key for a frame MMFF left as ETKDG embedded it (MMFF94 has
#: no parameters for the molecule): the dataset declares ``etkdg_mmff``.
MMFF_NOT_APPLIED = "mmff_not_applied"
#: The ``counts.dropped`` keys the build owns; each is recorded, also at zero.
BUILD_DROP_REASONS = (
    CONFORMER_EMBEDDING_FAILED,
    STEREO_MISMATCH_AFTER_EMBEDDING,
    ENANTIOMER_PARTNER_FAILED,
    MMFF_NOT_APPLIED,
)
#: Written next to each generated dataset.
TIMINGS_FILENAME = "conformer_timings.jsonl"

#: The element gate the SMILES filter already applied, plus the zero-hydrogen
#: and atom-overlap guards. No total-atom cap: the SMILES filter bounds heavy
#: atoms, and hydrogens push the total well past that.
DEFAULT_GEOMETRY_LIMITS = GeometryLimits(
    max_atoms=None,
    elements=ElementSet.mace_off,
    reject_zero_hydrogen=True,
    min_hydrogen_heavy_ratio=0.0,
    min_interatomic_distance=0.5,
)


class DatasetBuildConfig(BaseModel):
    """One build run: which bundles, where the datasets go, how to embed."""

    model_config = ConfigDict(extra="forbid")

    #: Where the SMILES bundles are read.
    bundle_root: Path
    #: Where the datasets are written, and ``manifest.yaml`` / ``status.yaml``.
    zarr_root: Path
    #: ``None`` means every bundle under ``bundle_root``.
    dataset_ids: list[str] | None = None
    conformers: ConformerEmbeddingConfig = Field(
        default_factory=ConformerEmbeddingConfig
    )
    geometry_limits: GeometryLimits = DEFAULT_GEOMETRY_LIMITS
    zarr: DatasetConfig = Field(default_factory=DatasetConfig)
    #: ``None`` means every CPU; ``1`` embeds in-process.
    n_workers: int | None = Field(default=None, ge=1)
    #: False skips a dataset already built from the same bundle content with the
    #: same conformer settings.
    overwrite: bool = False
    #: Verify every dataset under ``zarr_root`` after building.
    verify: bool = True
    #: False re-raises the first failure instead of recording it.
    keep_going: bool = True

    def worker_count(self) -> int:
        """``n_workers``, with ``None`` resolved to the CPUs this process may use."""
        return self.n_workers or default_workers()

    def resolve_dataset_ids(self) -> list[str]:
        if self.dataset_ids is not None:
            return list(self.dataset_ids)
        root = Path(self.bundle_root)
        return [path.relative_to(root).as_posix() for path in discover_bundles(root)]


class BuildSummary(BaseModel):
    """What one dataset's build did: the task's artifact."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    dataset_path: Path
    skipped: bool = False
    stereoisomers_in: int = 0
    structures_out: int = 0
    dropped: dict[str, int] = Field(default_factory=dict)
    #: Kept frames whose MMFF relaxation stopped at ``max_mmff_steps``.
    mmff_not_converged: int = 0
    bundle_content_sha256: str | None = None
    table_content_sha256: str | None = None
    elapsed_seconds: float = 0.0


@dataclass
class _EmbeddingOutcome:
    """Frames kept per stereoisomer, and why the others were dropped."""

    frames_by_stereoisomer: dict[int, list[Atoms]] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)
    timings: list[ConformerTimingRecord] = field(default_factory=list)
    #: Kept frames whose MMFF relaxation stopped at ``max_mmff_steps``.
    mmff_not_converged: int = 0

    def count_drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1


def _frame_drop_reason(
    isomeric_smiles: str,
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    origin: ConformerGeometryOrigin,
    limits: GeometryLimits,
    allowed_atomic_numbers: frozenset[int] | None,
) -> str | None:
    """``None`` to keep a generated frame, else the ``counts.dropped`` key."""
    if origin != "etkdg_mmff":
        return MMFF_NOT_APPLIED
    if not stereo_agrees_with_frame(isomeric_smiles, atomic_numbers, positions):
        return STEREO_MISMATCH_AFTER_EMBEDDING
    violations = frame_geometry_violations(
        atomic_numbers, positions, limits, allowed_atomic_numbers
    )
    return violations[0] if violations else None


def embed_bundle(
    bundle: Bundle,
    conformers: ConformerEmbeddingConfig,
    limits: GeometryLimits,
    n_workers: int | None,
) -> _EmbeddingOutcome:
    """Embed every stereoisomer of ``bundle`` and gate each frame."""
    table = bundle.table
    smiles_by_stereoisomer = {
        int(stereoisomer_id): str(isomeric_smiles)
        for stereoisomer_id, isomeric_smiles in zip(
            table["stereoisomer_id"], table["isomeric_smiles"], strict=True
        )
    }
    outcome = _EmbeddingOutcome()
    allowed_atomic_numbers = limits.allowed_atomic_numbers()
    for stereoisomer_id, embedding in embed_many(
        smiles_by_stereoisomer, conformers, n_workers=n_workers
    ):
        outcome.timings.append(embedding.timing)
        if not embedding.succeeded:
            outcome.count_drop(CONFORMER_EMBEDDING_FAILED)
            continue
        assert embedding.positions is not None and embedding.atomic_numbers is not None
        isomeric_smiles = smiles_by_stereoisomer[stereoisomer_id]
        kept: list[Atoms] = []
        for positions, origin, mmff_outcome in zip(
            embedding.positions,
            embedding.geometry_origins,
            embedding.timing.mmff_outcomes,
            strict=True,
        ):
            reason = _frame_drop_reason(
                isomeric_smiles,
                embedding.atomic_numbers,
                positions,
                origin,
                limits,
                allowed_atomic_numbers,
            )
            if reason is not None:
                outcome.count_drop(reason)
                continue
            kept.append(
                Atoms(
                    numbers=embedding.atomic_numbers,
                    positions=positions,
                    pbc=[0, 0, 0],
                )
            )
            if mmff_outcome == "not_converged":
                outcome.mmff_not_converged += 1
        outcome.frames_by_stereoisomer[stereoisomer_id] = kept
    return outcome


def _already_built(
    dataset_path: Path,
    bundle_content_sha256: str,
    conformers: ConformerEmbeddingConfig,
    limits: GeometryLimits,
) -> bool:
    """Whether ``dataset_path`` was built from this bundle content with these settings.

    The zarr layout is not compared: it changes how the data is stored, not
    which data it is.
    """
    try:
        provenance = read_provenance(dataset_path)
    except FileNotFoundError:
        return False
    record = provenance.conformers
    return (
        record is not None
        and record.parent_bundle_content_sha256 == bundle_content_sha256
        and record.settings == conformers
        and provenance.geometry_limits == limits
    )


class BuildDatasetTask(BaseModel):
    """Build one dataset from one SMILES bundle."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["build_dataset"] = "build_dataset"
    dataset_id: str
    config: DatasetBuildConfig

    @property
    def status_label(self) -> str:
        return self.dataset_id.replace("/", "__")

    def run(self, ctx: object) -> Iterator[EvalResult]:
        """Raises whatever reading, embedding or writing raised; the runner records it."""
        yield PydanticResult(
            file_name=Path("build") / f"{self.status_label}.yaml", obj=self.build()
        )

    def build(self) -> BuildSummary:
        started = time.monotonic()
        config = self.config
        bundle = read_bundle(Path(config.bundle_root) / self.dataset_id)
        dataset_path = Path(config.zarr_root) / self.dataset_id
        # ``read_bundle`` checked the recorded hash against the table.
        bundle_hash = bundle.content_sha256
        if not config.overwrite and _already_built(
            dataset_path, bundle_hash, config.conformers, config.geometry_limits
        ):
            logger.info("%s: already built from this bundle, skipping", self.dataset_id)
            return BuildSummary(
                dataset_id=self.dataset_id,
                dataset_path=dataset_path,
                skipped=True,
                bundle_content_sha256=bundle_hash,
                elapsed_seconds=time.monotonic() - started,
            )
        if not bundle.spec.smiles:
            raise ValueError(
                f"{self.dataset_id}: conformers are generated from SMILES, but the "
                "bundle declares smiles: false"
            )

        outcome = embed_bundle(
            bundle, config.conformers, config.geometry_limits, config.n_workers
        )
        surviving = [
            stereoisomer_id
            for stereoisomer_id, frames in outcome.frames_by_stereoisomer.items()
            if frames
        ]
        smiles_by_stereoisomer = dict(
            zip(
                bundle.table["stereoisomer_id"],
                bundle.table["isomeric_smiles"],
                strict=True,
            )
        )
        expanded = expand_to_structures(
            bundle,
            outcome.frames_by_stereoisomer,
            {
                stereoisomer_id: charge_and_multiplicity(
                    Chem.MolFromSmiles(smiles_by_stereoisomer[stereoisomer_id])
                )
                for stereoisomer_id in surviving
            },
            geometry_origin="etkdg_mmff",
        )
        provenance = expanded.provenance
        provenance.geometry_limits = config.geometry_limits
        provenance.conformers = ConformerGenerationRecord(
            rdkit_version=rdkit.__version__,
            settings=config.conformers,
            parent_bundle_content_sha256=bundle_hash,
        )
        # The build's own reasons (each frame or failed embedding under the
        # first reason that applies) and the expansion's orphan rule, merged
        # once onto the preparer's counts.
        stage_counts = dict.fromkeys(BUILD_DROP_REASONS, 0)
        for counts in (outcome.dropped, expanded.dropped_counts):
            for reason, count in counts.items():
                stage_counts[reason] = stage_counts.get(reason, 0) + count
        dropped = dict(bundle.provenance.counts.dropped)
        for reason, count in stage_counts.items():
            dropped[reason] = dropped.get(reason, 0) + count
        provenance.counts.dropped = dropped

        workers = config.worker_count()
        written = write_dataset(
            expanded.spec,
            expanded.table,
            PackedStructures.from_atoms(expanded.structures),
            provenance,
            dataset_path,
            layout=config.zarr,
            workers=workers,
        )
        write_timings_jsonl(outcome.timings, dataset_path / TIMINGS_FILENAME)
        logger.info(
            "%s: %d stereoisomers -> %d structures (dropped %s)",
            self.dataset_id,
            len(bundle.table),
            len(expanded.table),
            stage_counts,
        )
        assert written.outputs is not None
        return BuildSummary(
            dataset_id=self.dataset_id,
            dataset_path=dataset_path,
            stereoisomers_in=len(bundle.table),
            structures_out=len(expanded.table),
            dropped=dropped,
            mmff_not_converged=outcome.mmff_not_converged,
            bundle_content_sha256=bundle_hash,
            table_content_sha256=written.outputs.table_parquet.content_sha256,
            elapsed_seconds=time.monotonic() - started,
        )


class VerifyDatasetTask(BaseModel):
    """Re-check one dataset directory: table, structures and zarr."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["verify_dataset"] = "verify_dataset"
    dataset_path: Path
    #: Processes that check invariants 9 and 10.
    workers: int = Field(default=1, ge=1)

    @property
    def status_label(self) -> str:
        return self.dataset_path.name

    def run(self, ctx: object) -> Iterator[EvalResult]:
        """Yields the report, then raises ``ValueError`` if it found problems."""
        report = verify_dataset(self.dataset_path, workers=self.workers)
        yield PydanticResult(
            file_name=Path("verify") / f"{self.status_label}.yaml", obj=report
        )
        if report.problems:
            raise ValueError(
                f"{self.dataset_path} failed verification:\n  "
                + "\n  ".join(report.problems)
            )


def build_datasets(config: DatasetBuildConfig) -> RunReport:
    """Build every selected bundle, then verify every dataset under ``zarr_root``.

    Always writes ``manifest.yaml`` and ``status.yaml`` under ``zarr_root``,
    also when a task fails and ``keep_going`` is False.
    """
    output_root = Path(config.zarr_root)

    def task_groups() -> Iterator[list[BuildDatasetTask] | list[VerifyDatasetTask]]:
        yield [
            BuildDatasetTask(dataset_id=dataset_id, config=config)
            for dataset_id in config.resolve_dataset_ids()
        ]
        if config.verify:
            # Discovered only now, so the datasets just built are verified too.
            yield [
                VerifyDatasetTask(dataset_path=path, workers=config.worker_count())
                for path, _ in discover_datasets(output_root)
            ]

    return run_and_record(
        task_groups(), None, output_root, config, keep_going=config.keep_going
    )
