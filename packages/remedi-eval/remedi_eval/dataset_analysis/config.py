"""Configs for ``MoleculeDatasetAnalysis`` and ``DatasetComparison``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: The bblean fingerprint kinds. ``ecfp4`` / ``ecfp6`` are RDKit Morgan
#: fingerprints of radius 2 / 3 folded to ``n_features`` bits, the definition
#: the nvMolKit nearest-neighbour backend reproduces on the GPU.
FingerprintKind = Literal["ecfp4", "ecfp6", "rdkit", "topological", "ap"]

#: Morgan radius of each fingerprint kind the nvMolKit backend can compute.
MORGAN_RADIUS: dict[str, int] = {"ecfp4": 2, "ecfp6": 3}


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

    The fingerprint (``fingerprint_kind``, ``n_features``) is also the one the
    dataset comparison scores nearest-neighbour Tanimoto on, with either
    backend.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    fingerprint_kind: FingerprintKind = "ecfp4"
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

    Requires nvMolKit and a CUDA PyTorch, and a Morgan fingerprint
    (``bitbirch.fingerprint_kind`` ``ecfp4`` or ``ecfp6``).
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
    def _nvmolkit_needs_a_morgan_fingerprint(self) -> DatasetComparisonConfig:
        backend = self.nn_tanimoto.backend
        if (
            self.nn_tanimoto.enabled
            and isinstance(backend, NvmolkitNnTanimotoBackend)
            and self.bitbirch.fingerprint_kind not in MORGAN_RADIUS
        ):
            raise ValueError(
                "nn_tanimoto backend nvmolkit computes Morgan fingerprints only: "
                f"set bitbirch.fingerprint_kind to one of {sorted(MORGAN_RADIUS)} "
                f"(got {self.bitbirch.fingerprint_kind!r})"
            )
        return self
