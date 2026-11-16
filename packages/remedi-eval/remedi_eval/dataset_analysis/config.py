"""Configs for ``MoleculeDatasetAnalysis`` and ``DatasetComparison``."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
    plot_size: int = 1600
    """Datashader canvas size (square pixels)."""


class BitBirchConfig(BaseModel):
    """BitBIRCH clustering configuration (used when bblean is installed)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    fingerprint_kind: Literal["ecfp4", "ecfp6", "rdkit", "maccs"] = "ecfp4"
    n_features: int = 2048
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
    max_molecules: int | None = 1_000_000
    """Cap on number of unique SMILES fed to BitBIRCH. None = use all."""
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


class NnTanimotoConfig(BaseModel):
    """Eval -> pretrain nearest-neighbor Tanimoto search."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    fingerprint_kind: Literal["ecfp4", "ecfp6", "rdkit", "maccs"] = "ecfp4"
    """Re-used when the BitBIRCH fingerprints can't be shared (e.g. when the
    GPU backend wants its own packing). Kept consistent with
    ``BitBirchConfig.fingerprint_kind`` by default."""
    n_features: int = 2048
    chunk_size: int = Field(2048, ge=1)
    """How many query (eval) fingerprints to score against the full pretrain
    set per inner loop iteration. Trades RAM for Python overhead."""
    backend: Literal["cpu", "nvmolkit"] = "cpu"
    """`cpu` uses bblean's packed Tanimoto kernel. `nvmolkit` uses
    ``nvmolkit.similarity.crossTanimotoSimilarity`` on the GPU (requires
    nvMolKit + a CUDA-capable PyTorch install)."""
    histogram_bins: int = Field(50, ge=4)


class ScaffoldOverlapConfig(BaseModel):
    """Bemis-Murcko scaffold set overlap configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    include_generic: bool = True
    """Also compute the framework-only (atom-types stripped) scaffold."""
    n_rdkit_workers: int = Field(8, ge=1)


class DatasetComparisonConfig(BaseModel):
    """Top-level config for ``DatasetComparison``.

    Mirrors the BitBIRCH-side knobs in ``MoleculeDatasetAnalysisConfig`` so a
    user can reuse the same fingerprint settings."""

    model_config = ConfigDict(extra="forbid")

    output_dir: Path
    datasets: list[DatasetEntry] = Field(min_length=2)

    random_seed: int = 1234
    rdkit_n_workers: int = Field(8, ge=1)

    bitbirch: BitBirchConfig = Field(default_factory=BitBirchConfig)
    nn_tanimoto: NnTanimotoConfig = Field(default_factory=NnTanimotoConfig)
    scaffold_overlap: ScaffoldOverlapConfig = Field(
        default_factory=ScaffoldOverlapConfig
    )

    top_clusters_plotted: int = Field(20, ge=1)
    """How many largest clusters get distinct colors in the composition plot."""

    @property
    def umap(self) -> BitBirchUmapConfig:
        return self.bitbirch.umap
