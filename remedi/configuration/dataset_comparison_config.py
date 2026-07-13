from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from remedi.configuration.dataset_analysis_config import (
    BitBirchConfig,
    BitBirchUmapConfig,
)


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
