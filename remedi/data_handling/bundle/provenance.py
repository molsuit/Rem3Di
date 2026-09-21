"""``provenance.yaml`` — where a bundle came from, what it hashes to, what it counts.

See ``BENCHMARK_DATA_FORMAT.md`` §1.4. The preparer fills ``preparer``,
``source``, ``geometry_limits``, ``notices`` and the parts of ``counts`` only it
knows (``source_molecules`` and ``dropped``);
:func:`remedi.data_handling.bundle.bundle.write_bundle` fills everything that is
derivable from the data — the output hashes, ``final_rows``, ``per_split``,
``per_task_non_null``, ``stereoisomer_straddling_constitutions`` and
``prepared_at``.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from remedi.data_handling.dataset.tasks import ElementSet


class FileHash(BaseModel):
    """The sha256 of one pinned source file."""

    model_config = ConfigDict(extra="forbid")

    sha256: str


class PreparerRecord(BaseModel):
    """Which script in which repository produced this bundle."""

    model_config = ConfigDict(extra="forbid")

    repo: str
    script: str
    git_sha: str | None = None


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


class StructuresOutputRecord(BaseModel):
    """The hash and frame count of ``structures.extxyz``, the geometry of record."""

    model_config = ConfigDict(extra="forbid")

    file_sha256: str
    frames: int


class BundleOutputs(BaseModel):
    """The ``outputs:`` block, keyed by the on-disk filenames."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    table_parquet: TableOutputRecord = Field(alias="table.parquet")
    structures_extxyz: StructuresOutputRecord | None = Field(
        default=None, alias="structures.extxyz"
    )


class GeometryLimits(BaseModel):
    """The guards invariant 10 enforces, mirroring ``FilterAtomsStage`` today.

    ``elements`` is either a named :class:`ElementSet` preset or an explicit list
    of element symbols; ``None`` disables the element gate (a curated source may
    legitimately carry anything).
    """

    model_config = ConfigDict(extra="forbid")

    max_atoms: int | None = None
    elements: ElementSet | list[str] | None = None
    reject_zero_hydrogen: bool = False
    min_hydrogen_heavy_ratio: float = 0.0
    min_interatomic_distance: float | None = None

    def allowed_element_symbols(self) -> set[str] | None:
        """Resolve ``elements`` to a set of symbols, or ``None`` if ungated."""
        if self.elements is None:
            return None
        if isinstance(self.elements, ElementSet):
            # Imported lazily: the generators package pulls in torch through its
            # ``__init__`` chain, and this package must stay torch-free.
            from remedi.data_handling.dataset_creation.generators.utils import (
                resolve_element_set,
            )

            return resolve_element_set(self.elements)
        return set(self.elements)


class SmilesFilterRecord(BaseModel):
    """The SMILES-side cleaning a preparer applied before assigning identity.

    Mirrors ``FilterMoleculeStageConfig`` field for field, so a bundle records
    exactly which parse -> standardize -> filter -> canonicalize -> dedupe knobs
    produced its row set. ``element_set`` is either a named :class:`ElementSet`
    preset or an explicit list of element symbols; ``None`` means the element
    gate was off.
    """

    model_config = ConfigDict(extra="forbid")

    max_atoms: int | None = 100
    element_set: ElementSet | list[str] | None = ElementSet.mace_off

    allow_charged: bool = True
    allow_radicals: bool = True
    allow_isotopes: bool = False
    allow_multifragment: bool = False

    strip_salts: bool = True
    neutralize: bool = True

    dedupe: bool = True


class EtkdgParameters(BaseModel):
    """ETKDG knobs, recorded descriptively — not a reproduction contract (§1.2)."""

    model_config = ConfigDict(extra="forbid")

    version: str = "ETKDGv3"
    max_iterations: int


class MmffParameters(BaseModel):
    """MMFF94 relaxation knobs."""

    model_config = ConfigDict(extra="forbid")

    max_iterations: int
    non_bonded_threshold: float


class ConformerGenerationRecord(BaseModel):
    """Present only on a ``conformers``-stage bundle with ``geometry_origin: etkdg_mmff``."""

    model_config = ConfigDict(extra="forbid")

    rdkit_version: str
    n_conformers_requested: int
    etkdg: EtkdgParameters
    mmff: MmffParameters
    #: ``content_sha256`` of the smiles-stage bundle this one was expanded from.
    parent_bundle_content_sha256: str | None = None


class BundleCounts(BaseModel):
    """The ``counts:`` block. Everything a reviewer wants without opening the table."""

    model_config = ConfigDict(extra="forbid")

    #: Rows the preparer started from, before any filtering.
    source_molecules: int = 0
    #: reason -> how many were dropped for it.
    dropped: dict[str, int] = Field(default_factory=dict)
    #: Filled by ``write_bundle``.
    final_rows: int = 0
    stereoisomer_straddling_constitutions: int = 0
    per_split: dict[str, int] = Field(default_factory=dict)
    per_task_non_null: dict[str, int] = Field(default_factory=dict)


class BundleProvenance(BaseModel):
    """``provenance.yaml`` in full (§1.4)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    dataset_id: str
    prepared_at: datetime | None = None
    preparer: PreparerRecord
    source: SourceRecord = Field(default_factory=SourceRecord)
    #: Filled by ``write_bundle``; ``None`` on a bundle that has not been written.
    outputs: BundleOutputs | None = None
    geometry_limits: GeometryLimits = Field(default_factory=GeometryLimits)
    #: The SMILES filter the preparer ran; ``None`` when it filtered nothing.
    smiles_filter: SmilesFilterRecord | None = None
    conformers: ConformerGenerationRecord | None = None
    counts: BundleCounts = Field(default_factory=BundleCounts)
    #: Retraction notices carried forward.
    notices: list[str] = Field(default_factory=list)
