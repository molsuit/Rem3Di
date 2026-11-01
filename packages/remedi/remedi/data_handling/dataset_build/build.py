"""``dataset_build``: SMILES bundles -> datasets, then verify every dataset.

One YAML (:class:`DatasetBuildConfig`) drives one run. Each bundle under
``bundle_root`` is one task: conformers are generated with
:func:`~remedi.data_handling.chemistry.conformers.embed_many`, every frame is
gated on invariant 9 (stereochemistry survived MMFF) and invariant 10 (geometry
limits), the bundle is expanded to one row per structure, and
:func:`~remedi.data_handling.dataset_build.write.write_dataset` writes the
zarr with the table beside it. Each dataset under ``zarr_root`` (built here or
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

import pydantic_yaml
import rdkit
from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field

from remedi.data_handling.bundle import (
    Bundle,
    ConformerGenerationRecord,
    content_hash_of_table,
    discover_bundles,
    discover_datasets,
    expand_to_structures,
    has_assigned_tetrahedral_centre,
    read_bundle,
    read_provenance,
    stereochemistry_from_frame,
    tetrahedral_stereo_smiles,
)
from remedi.data_handling.bundle.expansion import (
    CONFORMER_EMBEDDING_FAILED,
    ENANTIOMER_PARTNER_FAILED,
)
from remedi.data_handling.chemistry.conformers import (
    ConformerEmbeddingConfig,
    ConformerTimingRecord,
    charge_and_multiplicity,
    embed_many,
    write_timings_jsonl,
)
from remedi.data_handling.chemistry.elements import ElementSet
from remedi.data_handling.chemistry.geometry import GeometryLimits, geometry_violations
from remedi.data_handling.dataset_build.verify import verify_dataset
from remedi.data_handling.dataset_build.write import ZarrLayout, write_dataset
from remedi.evaluation.framework.task import TaskStatus
from remedi.evaluation.framework.task_runner import (
    RunReport,
    make_run_report,
    run_tasks,
)
from remedi.evaluation.results import EvalResult, PydanticResult

logger = logging.getLogger(__name__)

#: ``counts.dropped`` key for a frame whose geometry disagrees with its SMILES.
STEREO_MISMATCH_AFTER_EMBEDDING = "stereo_mismatch_after_embedding"
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
    zarr: ZarrLayout = Field(default_factory=ZarrLayout)
    #: ``None`` means every CPU; ``1`` embeds in-process.
    n_workers: int | None = Field(default=None, ge=1)
    #: False skips a dataset already built from the same bundle content with the
    #: same conformer settings.
    overwrite: bool = False
    #: Verify every dataset under ``zarr_root`` after building.
    verify: bool = True
    #: False re-raises the first failure instead of recording it.
    keep_going: bool = True

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
    bundle_content_sha256: str | None = None
    table_content_sha256: str | None = None
    elapsed_seconds: float = 0.0


@dataclass
class _EmbeddingOutcome:
    """Frames kept per stereoisomer, and why the others were dropped."""

    frames_by_stereoisomer: dict[int, list[Atoms]] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)
    timings: list[ConformerTimingRecord] = field(default_factory=list)

    def count_drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1


def _frame_drop_reason(
    isomeric_smiles: str,
    atoms: Atoms,
    limits: GeometryLimits,
    allowed_symbols: frozenset[str] | None,
) -> str | None:
    """``None`` to keep a generated frame, else the ``counts.dropped`` key."""
    if has_assigned_tetrahedral_centre(isomeric_smiles):
        perceived = stereochemistry_from_frame(isomeric_smiles, atoms)
        if perceived is None or perceived != tetrahedral_stereo_smiles(isomeric_smiles):
            return STEREO_MISMATCH_AFTER_EMBEDDING
    violations = geometry_violations(atoms, limits, allowed_symbols)
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
    allowed_symbols = limits.allowed_element_symbols()
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
        for positions in embedding.positions:
            atoms = Atoms(
                numbers=embedding.atomic_numbers, positions=positions, pbc=[0, 0, 0]
            )
            reason = _frame_drop_reason(isomeric_smiles, atoms, limits, allowed_symbols)
            if reason is None:
                kept.append(atoms)
            else:
                outcome.count_drop(reason)
        outcome.frames_by_stereoisomer[stereoisomer_id] = kept
    return outcome


def _already_built(
    dataset_path: Path,
    bundle: Bundle,
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
        and record.parent_bundle_content_sha256 == content_hash_of_table(bundle.table)
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
        bundle_hash = content_hash_of_table(bundle.table)
        if not config.overwrite and _already_built(
            dataset_path, bundle, config.conformers, config.geometry_limits
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
                    smiles_by_stereoisomer[stereoisomer_id]
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
        # ``expand_to_structures`` counts every stereoisomer without a frame as
        # an embedding failure, including those whose frames were rejected here
        # under their own reason, so only its orphan count is merged.
        dropped = dict(bundle.provenance.counts.dropped)
        stage_counts = dict(outcome.dropped)
        stage_counts.setdefault(CONFORMER_EMBEDDING_FAILED, 0)
        stage_counts.setdefault(STEREO_MISMATCH_AFTER_EMBEDDING, 0)
        stage_counts.setdefault(ENANTIOMER_PARTNER_FAILED, 0)
        for reason, count in expanded.dropped_counts.items():
            if reason != CONFORMER_EMBEDDING_FAILED:
                stage_counts[reason] = stage_counts.get(reason, 0) + count
        for reason, count in stage_counts.items():
            dropped[reason] = dropped.get(reason, 0) + count
        provenance.counts.dropped = dropped

        written = write_dataset(
            expanded.spec,
            expanded.table,
            expanded.structures,
            provenance,
            dataset_path,
            layout=config.zarr,
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
            bundle_content_sha256=bundle_hash,
            table_content_sha256=written.outputs.table_parquet.content_sha256,
            elapsed_seconds=time.monotonic() - started,
        )


class VerifyDatasetTask(BaseModel):
    """Re-check one dataset directory: table, structures and zarr."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["verify_dataset"] = "verify_dataset"
    dataset_path: Path

    @property
    def status_label(self) -> str:
        return self.dataset_path.name

    def run(self, ctx: object) -> Iterator[EvalResult]:
        """Yields the report, then raises ``ValueError`` if it found problems."""
        report = verify_dataset(self.dataset_path)
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
    output_root.mkdir(parents=True, exist_ok=True)
    statuses: list[TaskStatus] = []
    # The statuses of the group that is running; ``run_tasks`` re-raises on
    # ``keep_going=False`` before returning them, so they are mirrored here.
    running: list[TaskStatus] = []
    planned = 0

    def flush(current: list[TaskStatus]) -> None:
        running[:] = current
        _write_run_files(output_root, config, planned, [*statuses, *running])

    def run_group(tasks: list[BuildDatasetTask] | list[VerifyDatasetTask]) -> None:
        nonlocal planned
        planned += len(tasks)
        running.clear()
        try:
            run_tasks(
                tasks,
                None,
                output_root,
                keep_going=config.keep_going,
                flush=flush,
                first_index=len(statuses),
            )
        finally:
            statuses.extend(running)
            running.clear()

    try:
        run_group(
            [
                BuildDatasetTask(dataset_id=dataset_id, config=config)
                for dataset_id in config.resolve_dataset_ids()
            ]
        )
        if config.verify:
            run_group(
                [
                    VerifyDatasetTask(dataset_path=path)
                    for path, _ in discover_datasets(output_root)
                ]
            )
    finally:
        _write_run_files(output_root, config, planned, statuses)
    report = make_run_report(planned, statuses)
    if report.n_failed:
        logger.warning("%d/%d task(s) failed", report.n_failed, len(statuses))
    return report


def _write_run_files(
    output_root: Path,
    config: DatasetBuildConfig,
    planned: int,
    statuses: list[TaskStatus],
) -> None:
    pydantic_yaml.to_yaml_file(output_root / "manifest.yaml", config)
    pydantic_yaml.to_yaml_file(
        output_root / "status.yaml", make_run_report(planned, statuses)
    )
