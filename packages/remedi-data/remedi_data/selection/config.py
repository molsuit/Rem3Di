"""Selection steps, configs and the record written beside every selection (§5.2.3).

A selection is a sorted array of ``structure_id``s of one dataset. It is made by
applying :data:`SelectionStep`s in order to the dataset's table; each step sees
only the rows the previous one kept.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Selection and cluster names become file names.
NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"


class SelectionError(ValueError):
    """A selection cannot be made or read as configured."""


class _SizedDraw(BaseModel):
    """A draw of ``count`` molecules or a ``fraction`` of them; exactly one is set."""

    model_config = ConfigDict(extra="forbid")

    count: int | None = Field(default=None, gt=0)
    fraction: float | None = Field(default=None, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _exactly_one_size(self) -> _SizedDraw:
        if (self.count is None) == (self.fraction is None):
            raise ValueError("set exactly one of count and fraction")
        return self

    def molecules_to_draw(self, available: int) -> int:
        """How many of ``available`` molecules the draw takes.

        Raises:
            SelectionError: if ``count`` asks for more molecules than are left.
        """
        if self.fraction is not None:
            return math.floor(self.fraction * available)
        assert self.count is not None
        if self.count > available:
            raise SelectionError(
                f"asked for {self.count} molecules but only {available} are left "
                "after the previous steps"
            )
        return self.count


class RandomMolecules(_SizedDraw):
    """A seeded draw of molecules, each with all its structures.

    Every molecule gets a random key from ``(seed, molecule_id)`` alone, and the
    draw takes the molecules with the smallest keys. So draws of different sizes
    with one seed are nested, and removing molecules in an earlier step removes
    them from the draw without reshuffling the rest.
    """

    kind: Literal["random_molecules"] = "random_molecules"
    seed: int = 0


ClusterWeighting = Literal["uniform", "square_root", "proportional"]


class HierarchicalClusters(_SizedDraw):
    """A nested, diversity-first draw over a two-level clustering of the molecules.

    The clustering is ``<dataset>/clusters/<clusters>.parquet``. The draw goes
    round-robin over coarse clusters, inside a coarse cluster over its fine
    clusters, inside a fine cluster in a seeded order. A cluster of size *s*
    comes up in proportion to 1 (``uniform``), sqrt(*s*) (``square_root``) or
    *s* (``proportional``). Draws of different sizes with one seed are nested.
    """

    kind: Literal["hierarchical_clusters"] = "hierarchical_clusters"
    clusters: str = Field(pattern=NAME_PATTERN)
    seed: int = 0
    cluster_weighting: ClusterWeighting = "square_root"


class ConformersPerMolecule(BaseModel):
    """Keep the best conformers of each group, ranked by a table column.

    ``group_by`` names what a "molecule" is: for GEOM-Drugs it is
    ``geom_smiles`` (one GEOM entry), since conformers of one entry can carry
    different stereo and so different ``stereoisomer_id``s. Rows with a ``null``
    rank are dropped. Ties are broken by ``structure_id``.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["conformers_per_molecule"] = "conformers_per_molecule"
    rank_by: str
    group_by: str = "molecule_id"
    #: ``False`` keeps the smallest values (``conformer_rank``), ``True`` the
    #: largest (``boltzmann_weight``).
    descending: bool = False
    count: int | None = Field(default=None, gt=0)
    #: Keep only rows whose ``rank_by`` value is at least this.
    minimum: float | None = None

    @model_validator(mode="after")
    def _some_limit(self) -> ConformersPerMolecule:
        if self.count is None and self.minimum is None:
            raise ValueError("set count, minimum or both")
        return self


OverlapKey = Literal["nonisomeric_smiles", "murcko_scaffold"]


class ExcludeOverlap(BaseModel):
    """Drop molecules that also appear in other datasets (the benchmark pool).

    Each path in ``datasets`` is a dataset directory or a directory whose direct
    children are dataset directories; all their rows, every fold, labels
    ignored, form one pool. ``murcko_scaffold`` drops every molecule whose
    scaffold is in the pool; an acyclic molecule has no scaffold and is matched
    on its ``nonisomeric_smiles`` instead.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["exclude_overlap"] = "exclude_overlap"
    datasets: list[Path] = Field(min_length=1)
    match_on: OverlapKey = "nonisomeric_smiles"


class Filter(BaseModel):
    """Keep the rows where a SQL expression over table columns holds.

    Besides the table's own columns the expression can use ``atom_count``.
    Example: ``atom_count <= 50 AND total_charge = 0``.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["filter"] = "filter"
    expression: str = Field(min_length=1)


SelectionStep = Annotated[
    RandomMolecules
    | HierarchicalClusters
    | ConformersPerMolecule
    | ExcludeOverlap
    | Filter,
    Field(discriminator="kind"),
]


class SelectionConfig(BaseModel):
    """One named selection of one dataset."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=NAME_PATTERN)
    description: str = ""
    steps: list[SelectionStep] = Field(min_length=1)


class StepCounts(BaseModel):
    """Rows and molecules before and after one step."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    structures_in: int
    structures_out: int
    molecules_in: int
    molecules_out: int


class SelectionInput(BaseModel):
    """A file a step read besides the dataset, with the hash it had."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    content_sha256: str


class SelectionRecord(BaseModel):
    """``selections/<name>.yaml``: how a selection was made and what it saw."""

    model_config = ConfigDict(extra="forbid")

    name: str
    dataset_id: str
    #: The dataset this selection is valid for; reading it checks both.
    table_content_sha256: str
    structures_sha256: str
    config: SelectionConfig
    steps: list[StepCounts]
    inputs: list[SelectionInput] = Field(default_factory=list)
    structures: int
    molecules: int
    atoms: int
    selection_sha256: str
    created_at: datetime
    elapsed_seconds: float


class CorpusSelections(BaseModel):
    """Every selection to make of one dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset: Path
    selections: list[SelectionConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_names(self) -> CorpusSelections:
        names = [selection.name for selection in self.selections]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"selection names repeat: {duplicates}")
        return self


class SelectionBatchConfig(BaseModel):
    """The yaml of ``scripts/dataset_creation/select_structures.py``."""

    model_config = ConfigDict(extra="forbid")

    corpora: list[CorpusSelections] = Field(min_length=1)
    #: Processes for the Murcko scaffolds of ``exclude_overlap``.
    workers: int = Field(default=1, gt=0)
    #: Replace an existing selection whose config or dataset changed; an
    #: unchanged one is always kept.
    overwrite: bool = False
    keep_going: bool = True
