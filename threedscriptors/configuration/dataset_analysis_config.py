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
