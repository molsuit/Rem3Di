"""``provenance.yaml``: where a bundle or dataset came from, what it hashes to, what it counts.

See ``BENCHMARK_DATA_FORMAT.md`` §1.4 and §10.2. The preparer fills
``preparer``, ``source``, ``geometry_limits``, ``smiles_filter``, ``notices`` and
the parts of ``counts`` only it knows (``source_molecules`` and ``dropped``);
the writer (:func:`remedi.data_handling.bundle.bundle.write_bundle` or
``dataset_build.write_dataset``) fills everything derivable from the data: the
output hashes, ``final_rows``, ``per_split``, ``per_label_non_null``,
``stereoisomer_straddling_constitutions`` and ``prepared_at``.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from remedi.data_handling.chemistry.conformers import ConformerEmbeddingConfig
from remedi.data_handling.chemistry.geometry import GeometryLimits
from remedi.data_handling.chemistry.smiles_filter import SmilesFilterConfig


class FileHash(BaseModel):
    """The sha256 of one pinned source file."""

    model_config = ConfigDict(extra="forbid")

    sha256: str


logger = logging.getLogger(__name__)


def git_head_sha(directory: Path) -> str | None:
    """``git rev-parse HEAD`` in ``directory``, or ``None`` when unavailable."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        logger.warning("could not read the git sha of %s: %s", directory, error)
        return None
    return completed.stdout.strip() or None


class PreparerRecord(BaseModel):
    """Which script in which repository produced this bundle."""

    model_config = ConfigDict(extra="forbid")

    repo: str
    script: str
    git_sha: str | None = None

    @classmethod
    def for_script(cls, repo: str, script_path: Path) -> PreparerRecord:
        """Record ``script_path`` and the git sha of the checkout it lives in."""
        script_path = Path(script_path).resolve()
        repository_root = next(
            (parent for parent in script_path.parents if (parent / ".git").exists()),
            script_path.parent,
        )
        script = script_path.relative_to(repository_root).as_posix()
        if not (repository_root / ".git").exists():
            logger.warning("%s is not inside a git checkout", script_path)
        return cls(repo=repo, script=script, git_sha=git_head_sha(repository_root))


class SourceRecord(BaseModel):
    """What the preparer read."""

    model_config = ConfigDict(extra="forbid")

    #: filename -> sha256 of the pinned raw inputs.
    files: dict[str, FileHash] = Field(default_factory=dict)
    #: Free text: where the coordinates came from, if the source supplied any.
    geometry: str | None = None
    #: Package name -> version, when the source is a package rather than a file.
    package_versions: dict[str, str] = Field(default_factory=dict)


class TableOutputRecord(BaseModel):
    """The two hashes of ``table.parquet`` (§1.4).

    ``file_sha256`` is a *writer* hash — compression and row-group size change
    it. ``content_sha256`` is over the logical content and is the key every
    downstream comparability check uses.
    """

    model_config = ConfigDict(extra="forbid")

    file_sha256: str
    content_sha256: str
    rows: int


class BundleOutputs(BaseModel):
    """The ``outputs:`` block: what the writer wrote, as hashes."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    table_parquet: TableOutputRecord = Field(alias="table.parquet")
    #: sha256 over the zarr's ``atomic_numbers``, ``positions`` and ``ptr``; set
    #: on a dataset only. It is the data identity a descriptor cache keys on.
    structures_sha256: str | None = None


class ConformerGenerationRecord(BaseModel):
    """Present only on a dataset with ``geometry_origin: etkdg_mmff``.

    Descriptive, not a reproduction contract: no ETKDG seed is pinned (§1.2).
    """

    model_config = ConfigDict(extra="forbid")

    rdkit_version: str
    settings: ConformerEmbeddingConfig
    #: ``content_sha256`` of the bundle the dataset was built from.
    parent_bundle_content_sha256: str


class BundleCounts(BaseModel):
    """The ``counts:`` block. Everything a reviewer wants without opening the table."""

    model_config = ConfigDict(extra="forbid")

    #: Rows the preparer started from, before any filtering.
    source_molecules: int = 0
    #: reason -> how many were dropped for it.
    dropped: dict[str, int] = Field(default_factory=dict)
    #: Filled by the writer.
    final_rows: int = 0
    stereoisomer_straddling_constitutions: int = 0
    per_split: dict[str, int] = Field(default_factory=dict)
    per_label_non_null: dict[str, int] = Field(default_factory=dict)


class BundleProvenance(BaseModel):
    """``provenance.yaml`` in full (§1.4)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    dataset_id: str
    prepared_at: datetime | None = None
    preparer: PreparerRecord
    source: SourceRecord = Field(default_factory=SourceRecord)
    #: Filled by the writer; ``None`` until written.
    outputs: BundleOutputs | None = None
    geometry_limits: GeometryLimits = Field(default_factory=GeometryLimits)
    #: The SMILES filter the preparer ran; ``None`` when it filtered nothing.
    smiles_filter: SmilesFilterConfig | None = None
    conformers: ConformerGenerationRecord | None = None
    counts: BundleCounts = Field(default_factory=BundleCounts)
    #: Retraction notices carried forward.
    notices: list[str] = Field(default_factory=list)
