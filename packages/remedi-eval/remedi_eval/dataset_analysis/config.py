"""Configs for ``MoleculeDatasetAnalysis`` and ``DatasetComparison``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BitBirchUmapConfig(BaseModel):
    """UMAP projection of BitBIRCH fingerprints, colored by cluster id."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    sample_size: int | None = 200_000
    """Cap on fingerprints fed to UMAP. UMAP scales poorly past a few 100k."""
    n_neighbors: int = Field(50, ge=2)
    min_dist: float = Field(0.1, ge=0.0, le=1.0)
    metric: Literal["jaccard", "hamming", "euclidean", "cosine"] = "jaccard"
    random_state: int = 42
    top_clusters_colored: int = Field(20, ge=1)
    """How many largest clusters get distinct colors. Smaller ones go grey."""
    plot_size: int = Field(1600, ge=100)
    """Size of the square UMAP image in pixels (the datashader canvas, or the
    matplotlib scatter of the dataset comparison)."""


class BitBirchConfig(BaseModel):
    """BitBIRCH clustering configuration.

    The fingerprint is the Morgan fingerprint of
    :func:`remedi_data.chemistry.fingerprints.morgan_fingerprints` (``radius``
    2 is ECFP4); it is also the one the dataset comparison scores
    nearest-neighbour Tanimoto on, with either backend.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    radius: int = Field(2, ge=0)
    fingerprint_size: int = Field(2048, gt=0, multiple_of=8)
    threshold: float = Field(0.65, ge=0.0, le=1.0)
    """Hard-coded threshold. Ignored when ``auto_threshold`` is true."""
    auto_threshold: bool = False
    """If true, derive the threshold at fit time from the union fingerprints
    via ``bblean.bitbirch.guess_threshold`` (mean Tanimoto + factor·std).
    The resolved value is recorded in ``BitBirchSummary``."""
    auto_threshold_factor: float = Field(3.0, ge=0.0)
    """``factor`` passed to ``guess_threshold``; bblean's default is 3.0."""
    branching_factor: int = Field(50, gt=1)
    merge_criterion: Literal["radius", "diameter", "tolerance"] = "diameter"
    tolerance: float | None = None
    max_molecules: int | None = Field(1_000_000, ge=1)
    """Cap on the number of molecules clustered (a uniform random subsample;
    for the comparison, of the union of all datasets). None = use all."""
    umap: BitBirchUmapConfig = Field(default_factory=BitBirchUmapConfig)


class MoleculeDatasetAnalysisConfig(BaseModel):
    """Knobs for `MoleculeDatasetAnalysis`.

    Defaults are tuned for datasets up to ~10M structures on a single workstation.
    """

    model_config = ConfigDict(extra="forbid")

    rdkit_subsample: int | None = 200_000
    """Subsample size for expensive per-molecule RDKit operations.
    None disables subsampling. The same RNG seed is used for reproducibility."""

    rdkit_n_workers: int = Field(default=8, ge=1)
    """Number of processes for RDKit work (joblib/multiprocessing)."""

    atom_chunk_size: int = 1_000_000
    """How many atoms to stream from zarr per chunk for atom-level metrics."""

    structure_chunk_size: int = 500_000
    """How many structures to process per chunk for structure-level metrics."""

    max_example_molecules: int = 100
    """How many example structures to render in the example-molecules grid."""

    top_n_scaffolds: int = 25
    """How many top Murcko scaffolds to keep in summary statistics."""

    random_seed: int = 1234

    bitbirch: BitBirchConfig = Field(default_factory=BitBirchConfig)


class DatasetEntry(BaseModel):
    """One named on-disk MoleculeDataset, tagged by role for the comparison."""

    model_config = ConfigDict(extra="forbid")

    name: str
    """Human-readable id used in plots and the contingency table."""

    role: Literal["pretrain", "eval"]
    """`pretrain` datasets are pooled into the reference; `eval` datasets are
    queried against the pooled pretrain set for nearest-neighbor Tanimoto."""

    path: Path
    """Directory of the zarr-backed MoleculeDataset
    (``MoleculeDataset.open_existing_dataset_from_dir``)."""

    max_molecules: int | None = None
    """Optional uniform-random subsample cap per dataset (deduped SMILES).
    None = use all unique SMILES."""


class CpuNnTanimotoBackend(BaseModel):
    """bblean's packed Tanimoto kernel, one query at a time against the reference."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["cpu"] = "cpu"


class NvmolkitNnTanimotoBackend(BaseModel):
    """``nvmolkit.similarity.crossTanimotoSimilarity`` on the GPU.

    Requires nvMolKit and a CUDA PyTorch, and a ``bitbirch.fingerprint_size``
    that is a multiple of 32 (nvMolKit takes 32-bit words).
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["nvmolkit"] = "nvmolkit"
    chunk_size: int = Field(2048, ge=1)
    """How many query (eval) fingerprints to score against the whole pretrain
    reference per GPU call (clamped so one similarity block stays under ~8 GB)."""


NnTanimotoBackend = Annotated[
    CpuNnTanimotoBackend | NvmolkitNnTanimotoBackend, Field(discriminator="kind")
]


class NnTanimotoConfig(BaseModel):
    """Eval -> pretrain nearest-neighbor Tanimoto search.

    Both backends score the ``bitbirch`` fingerprint (kind and size), so the
    numbers do not depend on the backend.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    backend: NnTanimotoBackend = Field(default_factory=CpuNnTanimotoBackend)


class ScaffoldOverlapConfig(BaseModel):
    """Bemis-Murcko scaffold set overlap configuration.

    Runs on the comparison's ``rdkit_n_workers`` processes.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    include_generic: bool = True
    """Also compute the framework-only (atom-types stripped) scaffold."""


class DatasetComparisonConfig(BaseModel):
    """Top-level config for ``DatasetComparison``.

    Mirrors the BitBIRCH-side knobs in ``MoleculeDatasetAnalysisConfig`` so a
    user can reuse the same fingerprint settings."""

    model_config = ConfigDict(extra="forbid")

    output_dir: Path
    datasets: list[DatasetEntry] = Field(min_length=2)

    random_seed: int = 1234
    rdkit_n_workers: int = Field(8, ge=1)
    """Processes for the per-molecule RDKit work (scaffolds)."""

    bitbirch: BitBirchConfig = Field(default_factory=BitBirchConfig)
    nn_tanimoto: NnTanimotoConfig = Field(default_factory=NnTanimotoConfig)
    scaffold_overlap: ScaffoldOverlapConfig = Field(
        default_factory=ScaffoldOverlapConfig
    )

    top_clusters_plotted: int = Field(20, ge=1)
    """How many largest clusters the composition bar plot shows. The UMAP
    colours ``bitbirch.umap.top_clusters_colored`` clusters."""

    @model_validator(mode="after")
    def _nvmolkit_needs_whole_words(self) -> DatasetComparisonConfig:
        backend = self.nn_tanimoto.backend
        if (
            self.nn_tanimoto.enabled
            and isinstance(backend, NvmolkitNnTanimotoBackend)
            and self.bitbirch.fingerprint_size % 32
        ):
            raise ValueError(
                "nn_tanimoto backend nvmolkit takes 32-bit fingerprint words: "
                "bitbirch.fingerprint_size must be a multiple of 32 "
                f"(got {self.bitbirch.fingerprint_size})"
            )
        return self
