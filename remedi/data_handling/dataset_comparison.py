from __future__ import annotations

import logging
import os
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from remedi.configuration.dataset_comparison_config import (
    DatasetComparisonConfig,
    DatasetEntry,
)
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.data_handling.dataset_analysis import (
    BitBirchSummary,
    BitBirchUmapSummary,
    DistributionStats,
    resolve_bitbirch_threshold,
)
from remedi.evaluation.results import (
    EvalResult,
    FigureResult,
    PydanticResult,
)

RDLogger.DisableLog("rdApp.*")

log = logging.getLogger(__name__)


# ---------- Result helpers ----------


class _NpzResult(EvalResult):
    result_type: str = "npz"
    arrays: dict[str, np.ndarray]

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def serialize_to(self, directory: Path) -> dict:
        output_path = directory / self.file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = dict(self.arrays)
        np.savez_compressed(
            str(output_path), **arrays
        )  # ty: ignore[invalid-argument-type]
        return {"path": str(output_path)}


# ---------- Pydantic summary models ----------


class PerDatasetCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    role: str
    n_smiles_unique: int
    n_fingerprints: int
    n_in_union: int


class ClusterCompositionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_clusters: int
    n_singletons: int
    largest_cluster_size: int
    largest_cluster_fraction: float
    cluster_size_distribution: DistributionStats
    scaffold_purity_distribution: DistributionStats
    """Per-cluster fraction of the most common Murcko scaffold within the
    cluster. 1.0 = single-scaffold cluster; ~1/k = perfectly mixed."""

    shared_clusters: int
    """Clusters containing molecules from >=2 distinct datasets."""
    exclusive_clusters: int

    pretrain_eval_shared_clusters: int
    """Clusters that contain at least one pretrain molecule AND at least one
    eval molecule. The molecule counts in those clusters are tabulated in
    `pretrain_in_eval_clusters_fraction`."""
    pretrain_molecules_in_eval_clusters: int
    pretrain_in_eval_clusters_fraction: float | None = None
    """`pretrain_molecules_in_eval_clusters / n_pretrain_molecules` -- how
    much of the pretraining set lives near eval clusters under this BitBIRCH
    threshold."""

    eval_clusters_with_pretrain_fraction: dict[str, float] = Field(default_factory=dict)
    """Per eval dataset: fraction of its non-empty clusters that contain at
    least one pretrain molecule. Closer to 1.0 means the eval set is
    well-covered by pretrain in scaffold space."""

    mutual_info: float
    normalized_mutual_info: float
    """MI between cluster id and dataset-name label. NMI ∈ [0, 1]; 0 = labels
    are independent of clustering, 1 = clusters perfectly predict labels."""


class NnTanimotoPerDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    n_queries: int
    distribution: DistributionStats
    fraction_above_0p4: float
    fraction_above_0p7: float


class NnTanimotoSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str
    n_pretrain_reference: int
    seconds_total: float | None = None
    per_dataset: list[NnTanimotoPerDataset] = Field(default_factory=list)


class ScaffoldOverlapPerDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    role: str
    n_molecules: int
    n_scaffolds_specific: int
    n_scaffolds_generic: int | None = None


class ScaffoldOverlapSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    per_dataset: list[ScaffoldOverlapPerDataset] = Field(default_factory=list)
    pairwise_jaccard_specific: dict[str, float] = Field(default_factory=dict)
    """Key is "<a>__<b>" with a<b lexicographically."""
    pairwise_jaccard_generic: dict[str, float] = Field(default_factory=dict)
    eval_specific_coverage_by_pretrain: dict[str, float] = Field(default_factory=dict)
    """For each eval dataset: |eval_scaffolds ∩ union_pretrain_scaffolds| /
    |eval_scaffolds|."""
    eval_generic_coverage_by_pretrain: dict[str, float] = Field(default_factory=dict)


class DatasetComparisonSummary(BaseModel):
    """Top-level summary serialized as a single yaml."""

    model_config = ConfigDict(extra="forbid")

    datasets: list[PerDatasetCounts]
    bitbirch: BitBirchSummary
    composition: ClusterCompositionSummary | None = None
    nn_tanimoto: NnTanimotoSummary | None = None
    scaffold_overlap: ScaffoldOverlapSummary | None = None
    timings_seconds: dict[str, float] = Field(default_factory=dict)


# ---------- Module-level worker fns (picklable) ----------


def _murcko_specific(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return None


def _murcko_generic(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        scaf = MurckoScaffold.MakeScaffoldGeneric(scaf)
        return Chem.MolToSmiles(scaf)
    except Exception:
        return None


def _parallel_map(
    fn,
    items: Sequence[str],
    n_workers: int,
    chunksize_div: int = 32,
) -> list:
    if not items:
        return []
    if n_workers <= 1 or len(items) < 1000:
        return [fn(s) for s in items]
    chunksize = max(64, len(items) // (n_workers * chunksize_div) or 1)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(fn, items, chunksize=chunksize))


# ---------- Dataset loading ----------


@dataclass
class _LoadedDataset:
    entry: DatasetEntry
    smiles: list[str]
    fps_packed: np.ndarray  # (n, n_features // 8), uint8
    kept_smiles: list[str]  # one per row of fps_packed (post skip_invalid)


def _unique_smiles_from_dataset(ds: MoleculeDataset) -> list[str]:
    store = ds.isomeric_smiles or ds.smiles
    if store is None:
        return []
    seen: set[str] = set()
    unique: list[str] = []
    for s in store:
        if s and s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


# ---------- NN Tanimoto backends ----------


def _nn_tanimoto_cpu(
    query_packed: np.ndarray,
    ref_packed: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """Max Tanimoto of each query packed-uint8 fingerprint vs the reference
    set. Returns shape (n_query,) float64."""
    import bblean

    n_q = int(query_packed.shape[0])
    out = np.zeros(n_q, dtype=np.float64)
    if n_q == 0 or ref_packed.shape[0] == 0:
        return out
    # bblean.similarity.jt_sim_packed handles (F,) vs (N,F). Loop queries.
    for i in range(n_q):
        sims = bblean.similarity.jt_sim_packed(query_packed[i], ref_packed)
        out[i] = float(np.max(sims))
        if (i + 1) % max(1, chunk_size) == 0:
            log.debug("NN-Tanimoto (cpu): %d/%d", i + 1, n_q)
    return out


def _nvmolkit_radius(fingerprint_kind: str) -> int:
    radius = {"ecfp4": 2, "ecfp6": 3}.get(fingerprint_kind)
    if radius is None:
        raise ValueError(
            f"nvmolkit backend requires Morgan FPs (ecfp4/ecfp6); got {fingerprint_kind}."
        )
    return radius


def _nvmolkit_compute_ref_fps(
    ref_smiles: list[str],
    fingerprint_kind: str,
    n_features: int,
):
    """Parse SMILES → RDKit Mols → Morgan FPs on GPU. Returns (n_ref, fpSize/32)
    torch tensor on cuda:0. Skips SMILES that fail to parse."""
    try:
        from nvmolkit.fingerprints import MorganFingerprintGenerator
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "backend='nvmolkit' selected but nvmolkit is not importable. "
            "Install per https://nvidia-digital-bio.github.io/nvMolKit/."
        ) from exc

    radius = _nvmolkit_radius(fingerprint_kind)
    fpgen = MorganFingerprintGenerator(radius=radius, fpSize=n_features)
    mols = [m for m in (Chem.MolFromSmiles(s) for s in ref_smiles) if m is not None]
    return fpgen.GetFingerprints(mols).torch()


def _nn_tanimoto_nvmolkit(
    query_smiles: list[str],
    ref_fps,
    fingerprint_kind: str,
    n_features: int,
    chunk_size: int,
):
    """GPU NN Tanimoto against a pre-fingerprinted reference (a torch tensor on
    the GPU). Returns ``(n_query,)`` float64 numpy array. Query SMILES are
    parsed + fingerprinted here, then scored in ``chunk_size`` slices so the
    full (Q, R) similarity matrix never has to fit on the GPU at once."""
    try:
        from nvmolkit.fingerprints import MorganFingerprintGenerator
        from nvmolkit.similarity import crossTanimotoSimilarity
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "backend='nvmolkit' selected but nvmolkit is not importable. "
            "Install per https://nvidia-digital-bio.github.io/nvMolKit/."
        ) from exc

    import torch

    radius = _nvmolkit_radius(fingerprint_kind)
    fpgen = MorganFingerprintGenerator(radius=radius, fpSize=n_features)
    q_mols = [m for m in (Chem.MolFromSmiles(s) for s in query_smiles) if m is not None]
    if not q_mols or ref_fps.shape[0] == 0:
        return np.zeros(len(q_mols), dtype=np.float64)
    q_fps = fpgen.GetFingerprints(q_mols).torch()
    n_q = int(q_fps.shape[0])
    n_r = int(ref_fps.shape[0])
    out = np.zeros(n_q, dtype=np.float64)
    # The similarity matrix is float64, so chunk RAM cost is
    # 8 * chunk_size * n_r bytes -- on a 40GB A100 we want this well under
    # ~10GB to leave headroom for the ref FPs (~0.9GB at 3.5M x 64 int32) and
    # whatever nvmolkit allocates internally.
    safe_chunk = max(
        1, min(int(chunk_size), max(1, 8_000_000_000 // (8 * max(1, n_r))))
    )
    if safe_chunk != chunk_size:
        log.info(
            "nvmolkit chunk_size clamped: %d -> %d (n_ref=%d)",
            chunk_size,
            safe_chunk,
            n_r,
        )
    for start in range(0, n_q, safe_chunk):
        end = min(start + safe_chunk, n_q)
        sim = crossTanimotoSimilarity(q_fps[start:end], ref_fps)
        torch.cuda.synchronize()
        sim_t = sim.torch() if hasattr(sim, "torch") else sim
        out[start:end] = sim_t.max(dim=1).values.detach().cpu().numpy()
        del sim_t
        torch.cuda.empty_cache()
    return out


# ---------- Orchestrator ----------


@contextmanager
def _timed(name: str, timings: dict[str, float]) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        timings[name] = elapsed
        log.info("%s took %.2fs", name, elapsed)


@dataclass
class _UnionFps:
    fps_packed: np.ndarray  # (N, F/8) uint8
    dataset_index: np.ndarray  # (N,) int64, into self._entries
    smiles: list[str]  # length N, kept-aligned


class DatasetComparison:
    """Compare a set of role-tagged molecular datasets in scaffold space.

    The three analyses requested by the user are:
      1. BitBIRCH clustering on the *union* of fingerprints, with health
         checks (cluster count, singletons, largest fraction, scaffold purity
         per cluster), a per-cluster shared/exclusive contingency between
         datasets, the fraction of pretrain molecules residing in eval
         clusters, and normalized mutual information between cluster id and
         dataset name.
      2. Per-eval-dataset nearest-neighbor Tanimoto distribution against the
         pooled pretrain reference. CPU (bblean packed) backend is the
         default; ``nn_tanimoto.backend='nvmolkit'`` switches to GPU.
      3. Bemis-Murcko scaffold-set overlaps (pairwise Jaccards plus
         eval-coverage-by-pretrain). Both specific and generic Murcko
         scaffolds are reported.
    """

    def __init__(self, config: DatasetComparisonConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.random_seed)
        self.results: list[EvalResult] = []
        self._timings: dict[str, float] = {}
        self._entries: list[DatasetEntry] = list(config.datasets)
        self._name_to_idx: dict[str, int] = {
            e.name: i for i, e in enumerate(self._entries)
        }
        if len(self._name_to_idx) != len(self._entries):
            raise ValueError("DatasetComparisonConfig.datasets must have unique names")
        self._loaded: list[_LoadedDataset] = []

    # ----- Load + fingerprint -----

    def load_datasets(self) -> None:
        for entry in self._entries:
            ds = MoleculeDataset.open_existing_dataset_from_dir(entry.path)
            smiles = _unique_smiles_from_dataset(ds)
            if entry.max_molecules is not None and len(smiles) > entry.max_molecules:
                idx = self._rng.choice(
                    len(smiles), size=entry.max_molecules, replace=False
                )
                idx.sort()
                smiles = [smiles[int(i)] for i in idx]
            fps_packed, kept_smiles = self._fps_for(smiles)
            log.info(
                "Loaded %s (%s): %d unique SMILES, %d fingerprints",
                entry.name,
                entry.role,
                len(smiles),
                fps_packed.shape[0],
            )
            self._loaded.append(
                _LoadedDataset(
                    entry=entry,
                    smiles=smiles,
                    fps_packed=fps_packed,
                    kept_smiles=kept_smiles,
                )
            )

    def _fps_for(self, smiles: list[str]) -> tuple[np.ndarray, list[str]]:
        import bblean

        if not smiles:
            return np.zeros((0, 0), dtype=np.uint8), []
        result = bblean.fps_from_smiles(
            smiles,
            kind=self.config.bitbirch.fingerprint_kind,
            n_features=self.config.bitbirch.n_features,
            pack=True,
            skip_invalid=True,
        )
        if isinstance(result, tuple):
            # bblean returns (fps, dropped_indices) -- the second element is
            # the indices of SMILES that failed fingerprinting, not the kept
            # ones (the public docs are misleading; verified empirically).
            fps, dropped_idx = result
            dropped = {int(i) for i in dropped_idx}
            kept_smiles = [s for i, s in enumerate(smiles) if i not in dropped]
            assert len(kept_smiles) == int(fps.shape[0])
        else:
            fps = result
            kept_smiles = list(smiles)
        return fps, kept_smiles

    def _build_union(self) -> _UnionFps:
        fps_list = [ld.fps_packed for ld in self._loaded if ld.fps_packed.shape[0] > 0]
        if not fps_list:
            raise RuntimeError("No fingerprints produced for any dataset")
        fps = np.vstack(fps_list)
        idx_list: list[np.ndarray] = []
        smiles: list[str] = []
        for di, ld in enumerate(self._loaded):
            n = ld.fps_packed.shape[0]
            if n == 0:
                continue
            idx_list.append(np.full(n, di, dtype=np.int64))
            smiles.extend(ld.kept_smiles)
        return _UnionFps(
            fps_packed=fps,
            dataset_index=np.concatenate(idx_list),
            smiles=smiles,
        )

    # ----- BitBIRCH on the union -----

    def run_bitbirch_union(
        self, union: _UnionFps
    ) -> tuple[np.ndarray, BitBirchSummary]:
        """Run BitBIRCH on the union FP array. Returns (row->cluster_id, summary).

        Reuses the histograms-and-counts shape from
        ``MoleculeDatasetAnalysis.run_bitbirch`` (but operates on a
        pre-fingerprinted matrix instead of SMILES, so we can keep the
        dataset-origin alignment)."""
        import bblean

        cfg = self.config.bitbirch
        n_fps = int(union.fps_packed.shape[0])
        if n_fps == 0:
            return np.zeros(0, dtype=np.int64), BitBirchSummary(
                enabled=True, notes="empty union"
            )

        threshold, source, mean_isim, std_isim = resolve_bitbirch_threshold(
            cfg, union.fps_packed
        )
        log.info(
            "BitBIRCH union: threshold=%.3f (source=%s, mean_iSIM=%s, std=%s)",
            threshold,
            source,
            f"{mean_isim:.3f}" if mean_isim is not None else "n/a",
            f"{std_isim:.3f}" if std_isim is not None else "n/a",
        )
        tree = bblean.BitBirch(
            threshold=threshold,
            branching_factor=cfg.branching_factor,
            merge_criterion=cfg.merge_criterion,
            tolerance=cfg.tolerance,
        )
        tree.fit(union.fps_packed)
        clusters = tree.get_cluster_mol_ids()
        sizes = np.asarray([len(c) for c in clusters], dtype=np.int64)
        row_cluster = np.full(n_fps, -1, dtype=np.int64)
        for cid, rows in enumerate(clusters):
            row_cluster[np.asarray(rows, dtype=np.int64)] = cid

        # Cluster size histogram
        fig, ax = plt.subplots(figsize=(8, 4))
        if sizes.size:
            ax.hist(
                sizes, bins=min(50, max(5, int(np.sqrt(sizes.size)))), color="#8172b3"
            )
            ax.set_yscale("log")
        ax.set_xlabel("Molecules per cluster")
        ax.set_ylabel("Cluster count (log)")
        ax.set_title(
            f"BitBIRCH union clusters (n={sizes.size},"
            f" threshold={threshold:.2f} [{source}], kind={cfg.fingerprint_kind})"
        )
        fig.tight_layout()
        self.results.append(
            FigureResult(figure=fig, file_name=Path("bitbirch_union_cluster_sizes.png"))
        )

        summary = BitBirchSummary(
            enabled=True,
            n_input_smiles=n_fps,
            n_fingerprints=n_fps,
            n_clusters=int(sizes.size),
            singleton_clusters=int((sizes == 1).sum()),
            largest_cluster=int(sizes.max()) if sizes.size else 0,
            cluster_size_distribution=DistributionStats.from_array(sizes),
            threshold_used=threshold,
            threshold_source=source,
            mean_isim=mean_isim,
            std_isim=std_isim,
            umap=self._bitbirch_umap(union, row_cluster, sizes),
        )
        return row_cluster, summary

    def _bitbirch_umap(
        self,
        union: _UnionFps,
        row_cluster: np.ndarray,
        cluster_sizes: np.ndarray,
    ) -> BitBirchUmapSummary | None:
        cfg = self.config.bitbirch.umap
        if not cfg.enabled:
            return BitBirchUmapSummary(enabled=False, notes="disabled by config")
        n_fps = int(union.fps_packed.shape[0])
        if n_fps == 0:
            return BitBirchUmapSummary(enabled=True, notes="no fingerprints")

        cap = cfg.sample_size if cfg.sample_size is not None else n_fps
        cap = min(cap, n_fps)
        if cap < n_fps:
            sample_rows = self._rng.choice(n_fps, size=cap, replace=False)
            sample_rows.sort()
        else:
            sample_rows = np.arange(n_fps, dtype=np.int64)

        try:
            import umap
        except Exception as exc:  # pragma: no cover - optional
            return BitBirchUmapSummary(
                enabled=True, notes=f"umap-learn import failed: {exc!r}"
            )

        import bblean

        unpacked = bblean.unpack_fingerprints(
            union.fps_packed[sample_rows], n_features=self.config.bitbirch.n_features
        )
        t0 = time.perf_counter()
        try:
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=cfg.n_neighbors,
                min_dist=cfg.min_dist,
                metric=cfg.metric,
                random_state=cfg.random_state,
            )
            coords = reducer.fit_transform(unpacked.astype(np.float32, copy=False))
        except Exception as exc:
            return BitBirchUmapSummary(
                enabled=True,
                n_points=int(sample_rows.size),
                notes=f"UMAP fit failed: {exc!r}",
            )
        elapsed = time.perf_counter() - t0

        # Two scatter plots: one colored by dataset origin, one by top-N clusters.
        self._plot_umap_by_origin(coords, union.dataset_index[sample_rows])
        top_n = min(self.config.top_clusters_plotted, int(cluster_sizes.size))
        top_cids = (
            np.argsort(cluster_sizes)[-top_n:][::-1]
            if top_n > 0
            else np.asarray([], dtype=np.int64)
        )
        self._plot_umap_by_cluster(coords, row_cluster[sample_rows], top_cids)

        self.results.append(
            _NpzResult(
                file_name=Path("dataset_comparison_umap_coords.npz"),
                arrays={
                    "coords": coords.astype(np.float32),
                    "cluster_id": row_cluster[sample_rows].astype(np.int64),
                    "dataset_index": union.dataset_index[sample_rows].astype(np.int64),
                    "row_index": sample_rows.astype(np.int64),
                },
            )
        )
        return BitBirchUmapSummary(
            enabled=True,
            n_points=int(sample_rows.size),
            n_neighbors=cfg.n_neighbors,
            min_dist=cfg.min_dist,
            metric=cfg.metric,
            top_clusters_colored=int(top_n),
            fit_seconds=float(elapsed),
        )

    def _plot_umap_by_origin(
        self, coords: np.ndarray, dataset_index: np.ndarray
    ) -> None:
        cmap = plt.colormaps.get_cmap("tab20")
        fig, ax = plt.subplots(figsize=(8, 8))
        for di, ld in enumerate(self._loaded):
            m = dataset_index == di
            if not np.any(m):
                continue
            label = f"{ld.entry.name} ({ld.entry.role})"
            ax.scatter(
                coords[m, 0],
                coords[m, 1],
                s=4,
                alpha=0.5,
                color=cmap(di % cmap.N),
                label=label,
                linewidths=0,
            )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("Union BitBIRCH FPs (colored by dataset)")
        ax.legend(markerscale=2, frameon=False, fontsize=8, loc="best")
        fig.tight_layout()
        self.results.append(
            FigureResult(figure=fig, file_name=Path("umap_by_origin.png"))
        )

    def _plot_umap_by_cluster(
        self, coords: np.ndarray, cluster_ids: np.ndarray, top_cids: np.ndarray
    ) -> None:
        top_set = set(int(c) for c in top_cids)
        cmap = plt.colormaps.get_cmap("tab20")
        fig, ax = plt.subplots(figsize=(8, 8))
        mask_other = np.asarray([int(c) not in top_set for c in cluster_ids])
        ax.scatter(
            coords[mask_other, 0],
            coords[mask_other, 1],
            s=3,
            alpha=0.25,
            color="#cccccc",
            linewidths=0,
            label="other",
        )
        for i, c in enumerate(top_cids):
            m = cluster_ids == int(c)
            if not np.any(m):
                continue
            ax.scatter(
                coords[m, 0],
                coords[m, 1],
                s=5,
                alpha=0.7,
                color=cmap(i % cmap.N),
                linewidths=0,
                label=f"c{int(c)}",
            )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(f"Union BitBIRCH FPs (top-{len(top_cids)} clusters colored)")
        ax.legend(markerscale=2, frameon=False, fontsize=8, loc="best", ncol=2)
        fig.tight_layout()
        self.results.append(
            FigureResult(figure=fig, file_name=Path("umap_by_cluster.png"))
        )

    # ----- Cluster composition analyses -----

    def analyze_clusters(
        self, union: _UnionFps, row_cluster: np.ndarray
    ) -> ClusterCompositionSummary | None:
        n = int(row_cluster.shape[0])
        if n == 0:
            return None

        n_clusters = int(row_cluster.max()) + 1 if row_cluster.size else 0
        sizes = np.bincount(row_cluster, minlength=n_clusters)

        # contingency[c, d] = count of dataset d in cluster c
        contingency = np.zeros((n_clusters, len(self._entries)), dtype=np.int64)
        np.add.at(contingency, (row_cluster, union.dataset_index), 1)

        nonempty_per_cluster = (contingency > 0).sum(axis=1)
        shared = int((nonempty_per_cluster >= 2).sum())
        exclusive = int((nonempty_per_cluster == 1).sum())

        pretrain_idx = [i for i, e in enumerate(self._entries) if e.role == "pretrain"]
        eval_idx = [i for i, e in enumerate(self._entries) if e.role == "eval"]
        n_pretrain_total = (
            int(contingency[:, pretrain_idx].sum()) if pretrain_idx else 0
        )
        has_pretrain = (
            (contingency[:, pretrain_idx].sum(axis=1) > 0)
            if pretrain_idx
            else np.zeros(n_clusters, dtype=bool)
        )
        has_eval = (
            (contingency[:, eval_idx].sum(axis=1) > 0)
            if eval_idx
            else np.zeros(n_clusters, dtype=bool)
        )
        pretrain_eval_shared = int((has_pretrain & has_eval).sum())
        pretrain_mols_in_eval_clusters = (
            int(contingency[has_eval][:, pretrain_idx].sum()) if pretrain_idx else 0
        )
        pretrain_fraction = (
            pretrain_mols_in_eval_clusters / n_pretrain_total
            if n_pretrain_total
            else None
        )

        eval_clusters_with_pretrain: dict[str, float] = {}
        for di in eval_idx:
            name = self._entries[di].name
            mask = contingency[:, di] > 0
            if mask.sum() == 0:
                eval_clusters_with_pretrain[name] = 0.0
                continue
            have_pre = has_pretrain & mask
            eval_clusters_with_pretrain[name] = float(have_pre.sum()) / float(
                mask.sum()
            )

        # Scaffold purity per cluster (compute Murcko on union SMILES once).
        with _timed("scaffolds_for_purity", self._timings):
            scaffolds = _parallel_map(
                _murcko_specific, union.smiles, self.config.rdkit_n_workers
            )
        purity = np.zeros(n_clusters, dtype=np.float64)
        bucket: dict[int, Counter[str]] = defaultdict(Counter)
        for sc, cid in zip(scaffolds, row_cluster.tolist(), strict=True):
            if sc is None:
                continue
            bucket[int(cid)][sc] += 1
        for cid in range(n_clusters):
            counts = bucket.get(cid)
            if not counts:
                purity[cid] = float("nan")
                continue
            top = max(counts.values())
            total = sum(counts.values())
            purity[cid] = top / total if total else float("nan")

        # Mutual information between cluster id and dataset name.
        from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

        mi = float(mutual_info_score(row_cluster, union.dataset_index))
        nmi = float(
            normalized_mutual_info_score(
                row_cluster, union.dataset_index, average_method="arithmetic"
            )
        )

        # Plots: composition stacked bar + scaffold purity hist
        self._plot_cluster_composition(contingency)
        self._plot_scaffold_purity(purity[~np.isnan(purity)])

        # Save the contingency for downstream inspection.
        self.results.append(
            _NpzResult(
                file_name=Path("cluster_contingency.npz"),
                arrays={
                    "contingency": contingency,
                    "cluster_sizes": sizes,
                    "scaffold_purity": purity,
                    "dataset_names": np.asarray(
                        [e.name for e in self._entries], dtype=object
                    ),
                    "dataset_roles": np.asarray(
                        [e.role for e in self._entries], dtype=object
                    ),
                },
            )
        )

        largest_size = int(sizes.max()) if sizes.size else 0
        return ClusterCompositionSummary(
            n_clusters=int(n_clusters),
            n_singletons=int((sizes == 1).sum()),
            largest_cluster_size=largest_size,
            largest_cluster_fraction=float(largest_size) / float(n) if n else 0.0,
            cluster_size_distribution=DistributionStats.from_array(sizes),
            scaffold_purity_distribution=DistributionStats.from_array(purity),
            shared_clusters=shared,
            exclusive_clusters=exclusive,
            pretrain_eval_shared_clusters=pretrain_eval_shared,
            pretrain_molecules_in_eval_clusters=pretrain_mols_in_eval_clusters,
            pretrain_in_eval_clusters_fraction=pretrain_fraction,
            eval_clusters_with_pretrain_fraction=eval_clusters_with_pretrain,
            mutual_info=mi,
            normalized_mutual_info=nmi,
        )

    def _plot_cluster_composition(self, contingency: np.ndarray) -> None:
        sizes = contingency.sum(axis=1)
        top_n = min(self.config.top_clusters_plotted, int(sizes.size))
        if top_n == 0:
            return
        order = np.argsort(sizes)[::-1][:top_n]
        cmap = plt.colormaps.get_cmap("tab20")
        fig, ax = plt.subplots(figsize=(max(8, 0.5 * top_n), 5))
        bottom = np.zeros(top_n, dtype=np.float64)
        x = np.arange(top_n)
        for di, ld in enumerate(self._loaded):
            counts = contingency[order, di].astype(np.float64)
            ax.bar(
                x,
                counts,
                bottom=bottom,
                color=cmap(di % cmap.N),
                label=f"{ld.entry.name} ({ld.entry.role})",
                width=0.85,
            )
            bottom += counts
        ax.set_xticks(x)
        ax.set_xticklabels([f"c{int(c)}" for c in order], rotation=60, ha="right")
        ax.set_xlabel("Cluster (top by size)")
        ax.set_ylabel("Molecules")
        ax.set_title(f"Per-cluster composition by dataset (top-{top_n} by size)")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        fig.tight_layout()
        self.results.append(
            FigureResult(figure=fig, file_name=Path("cluster_composition.png"))
        )

    def _plot_scaffold_purity(self, purity: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        if purity.size:
            ax.hist(purity, bins=40, range=(0.0, 1.0), color="#4c72b0")
        ax.set_xlabel("Top-scaffold fraction in cluster")
        ax.set_ylabel("Cluster count")
        ax.set_title(
            f"BitBIRCH scaffold purity (n_clusters_with_scaffolds={purity.size})"
        )
        fig.tight_layout()
        self.results.append(
            FigureResult(figure=fig, file_name=Path("cluster_scaffold_purity.png"))
        )

    # ----- NN Tanimoto -----

    def nn_tanimoto(self, union: _UnionFps) -> NnTanimotoSummary | None:
        cfg = self.config.nn_tanimoto
        if not cfg.enabled:
            return None
        pretrain_loaded = [ld for ld in self._loaded if ld.entry.role == "pretrain"]
        eval_loaded = [ld for ld in self._loaded if ld.entry.role == "eval"]
        if not pretrain_loaded or not eval_loaded:
            log.warning("NN Tanimoto skipped: need >=1 pretrain and >=1 eval dataset")
            return NnTanimotoSummary(
                backend=cfg.backend,
                n_pretrain_reference=0,
            )

        ref_packed = np.vstack([ld.fps_packed for ld in pretrain_loaded])
        ref_smiles: list[str] = []
        for ld in pretrain_loaded:
            ref_smiles.extend(ld.kept_smiles)
        log.info(
            "NN Tanimoto: %d eval datasets vs %d pretrain molecules (backend=%s)",
            len(eval_loaded),
            ref_packed.shape[0],
            cfg.backend,
        )

        # GPU backend amortizes ref fingerprinting across eval datasets --
        # parsing 3.5M pretrain SMILES once is the difference between the
        # NN-Tanimoto step finishing in minutes vs. dominating the run.
        ref_gpu_fps = None
        if cfg.backend == "nvmolkit":
            with _timed("nn_tanimoto_ref_fps", self._timings):
                ref_gpu_fps = _nvmolkit_compute_ref_fps(
                    ref_smiles,
                    fingerprint_kind=cfg.fingerprint_kind,
                    n_features=cfg.n_features,
                )
            log.info(
                "nvmolkit: ref FPs on %s, shape=%s",
                ref_gpu_fps.device,
                tuple(ref_gpu_fps.shape),
            )

        t0 = time.perf_counter()
        per_dataset: list[NnTanimotoPerDataset] = []
        per_dataset_values: dict[str, np.ndarray] = {}
        for ld in eval_loaded:
            if cfg.backend == "cpu":
                values = _nn_tanimoto_cpu(
                    ld.fps_packed, ref_packed, chunk_size=cfg.chunk_size
                )
            elif cfg.backend == "nvmolkit":
                values = _nn_tanimoto_nvmolkit(
                    ld.kept_smiles,
                    ref_gpu_fps,
                    fingerprint_kind=cfg.fingerprint_kind,
                    n_features=cfg.n_features,
                    chunk_size=cfg.chunk_size,
                )
            else:  # pragma: no cover - exhaustive Literal
                raise ValueError(f"unknown backend {cfg.backend!r}")
            per_dataset_values[ld.entry.name] = values
            per_dataset.append(
                NnTanimotoPerDataset(
                    name=ld.entry.name,
                    n_queries=int(values.size),
                    distribution=DistributionStats.from_array(values),
                    fraction_above_0p4=float((values >= 0.4).mean())
                    if values.size
                    else 0.0,
                    fraction_above_0p7=float((values >= 0.7).mean())
                    if values.size
                    else 0.0,
                )
            )
        elapsed = time.perf_counter() - t0

        self._plot_nn_tanimoto(per_dataset_values, cfg.histogram_bins)
        self.results.append(
            _NpzResult(
                file_name=Path("nn_tanimoto_values.npz"),
                arrays={
                    name: vals.astype(np.float32)
                    for name, vals in per_dataset_values.items()
                },
            )
        )
        return NnTanimotoSummary(
            backend=cfg.backend,
            n_pretrain_reference=int(ref_packed.shape[0]),
            seconds_total=float(elapsed),
            per_dataset=per_dataset,
        )

    def _plot_nn_tanimoto(
        self, per_dataset_values: dict[str, np.ndarray], bins: int
    ) -> None:
        # `bins` is kept for API stability; the figure is now a violin plot
        # which gives a cleaner side-by-side view of per-dataset distributions
        # than overlaid histograms.
        del bins
        items = [(name, vals) for name, vals in per_dataset_values.items() if vals.size]
        if not items:
            return
        values = [vals for _, vals in items]
        cmap = plt.colormaps.get_cmap("tab10")
        positions = np.arange(1, len(items) + 1)

        fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(items)), 5))
        parts = ax.violinplot(
            values,
            positions=positions,
            showmedians=True,
            showextrema=True,
            widths=0.8,
        )
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(cmap(i % cmap.N))
            body.set_edgecolor("black")
            body.set_alpha(0.6)
        for key in ("cmedians", "cmaxes", "cmins", "cbars"):
            if key in parts:
                parts[key].set_color("black")
                parts[key].set_linewidth(1.0)
        # Overlay mean markers.
        means = [float(np.mean(v)) for v in values]
        ax.scatter(
            positions,
            means,
            marker="D",
            color="white",
            edgecolor="black",
            s=30,
            zorder=3,
            label="mean",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{name}\n(n={v.size})" for name, v in items], rotation=0)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Nearest-neighbor Tanimoto vs pretrain pool")
        ax.set_title("Eval -> pretrain NN Tanimoto (per dataset)")
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        ax.legend(loc="upper right", frameon=False, fontsize=9)
        fig.tight_layout()
        self.results.append(
            FigureResult(figure=fig, file_name=Path("nn_tanimoto_distributions.png"))
        )

    # ----- Bemis-Murcko scaffold overlap -----

    def _collect_scaffold_sets(
        self, include_generic: bool, n_workers: int
    ) -> tuple[
        dict[str, set[str]],
        dict[str, set[str]],
        list[ScaffoldOverlapPerDataset],
    ]:
        specific_sets: dict[str, set[str]] = {}
        generic_sets: dict[str, set[str]] = {}
        per_dataset: list[ScaffoldOverlapPerDataset] = []
        for ld in self._loaded:
            with _timed(f"scaffolds_specific[{ld.entry.name}]", self._timings):
                spec = _parallel_map(_murcko_specific, ld.kept_smiles, n_workers)
            specific_sets[ld.entry.name] = {s for s in spec if s}
            generic_set: set[str] | None = None
            if include_generic:
                with _timed(f"scaffolds_generic[{ld.entry.name}]", self._timings):
                    gen = _parallel_map(_murcko_generic, ld.kept_smiles, n_workers)
                generic_set = {s for s in gen if s}
                generic_sets[ld.entry.name] = generic_set
            per_dataset.append(
                ScaffoldOverlapPerDataset(
                    name=ld.entry.name,
                    role=ld.entry.role,
                    n_molecules=len(ld.kept_smiles),
                    n_scaffolds_specific=len(specific_sets[ld.entry.name]),
                    n_scaffolds_generic=(
                        len(generic_set) if generic_set is not None else None
                    ),
                )
            )
        return specific_sets, generic_sets, per_dataset

    def _pairwise_jaccard(
        self, sets: dict[str, set[str]], names: list[str]
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                sa, sb = sets[a], sets[b]
                u = len(sa | sb)
                out[f"{a}__{b}"] = float(len(sa & sb)) / float(u) if u else 0.0
        return out

    def _eval_coverage(
        self, eval_sets: dict[str, set[str]], pretrain_union: set[str]
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for ld in self._loaded:
            if ld.entry.role != "eval":
                continue
            es = eval_sets[ld.entry.name]
            out[ld.entry.name] = (
                float(len(es & pretrain_union)) / float(len(es)) if es else 0.0
            )
        return out

    def scaffold_overlap(self) -> ScaffoldOverlapSummary | None:
        cfg = self.config.scaffold_overlap
        if not cfg.enabled:
            return None

        specific_sets, generic_sets, per_dataset = self._collect_scaffold_sets(
            include_generic=cfg.include_generic, n_workers=cfg.n_rdkit_workers
        )
        names = [e.name for e in self._entries]

        pairwise_specific = self._pairwise_jaccard(specific_sets, names)
        pairwise_generic = (
            self._pairwise_jaccard(generic_sets, names) if cfg.include_generic else {}
        )

        pretrain_union_specific: set[str] = set()
        pretrain_union_generic: set[str] = set()
        for ld in self._loaded:
            if ld.entry.role != "pretrain":
                continue
            pretrain_union_specific |= specific_sets[ld.entry.name]
            if cfg.include_generic:
                pretrain_union_generic |= generic_sets[ld.entry.name]

        eval_coverage_specific = self._eval_coverage(
            specific_sets, pretrain_union_specific
        )
        eval_coverage_generic = (
            self._eval_coverage(generic_sets, pretrain_union_generic)
            if cfg.include_generic
            else {}
        )

        self._plot_scaffold_jaccard_heatmap(pairwise_specific, names, "specific")
        if cfg.include_generic:
            self._plot_scaffold_jaccard_heatmap(pairwise_generic, names, "generic")

        return ScaffoldOverlapSummary(
            per_dataset=per_dataset,
            pairwise_jaccard_specific=pairwise_specific,
            pairwise_jaccard_generic=pairwise_generic,
            eval_specific_coverage_by_pretrain=eval_coverage_specific,
            eval_generic_coverage_by_pretrain=eval_coverage_generic,
        )

    def _plot_scaffold_jaccard_heatmap(
        self,
        pairwise: dict[str, float],
        names: list[str],
        kind: str,
    ) -> None:
        n = len(names)
        M = np.zeros((n, n), dtype=np.float64)
        for i, a in enumerate(names):
            M[i, i] = 1.0
            for j in range(i + 1, n):
                b = names[j]
                v = pairwise.get(f"{a}__{b}", 0.0)
                M[i, j] = v
                M[j, i] = v
        fig, ax = plt.subplots(figsize=(0.6 * n + 3, 0.6 * n + 2))
        im = ax.imshow(M, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(names, rotation=60, ha="right")
        ax.set_yticklabels(names)
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if M[i, j] < 0.5 else "black",
                    fontsize=8,
                )
        fig.colorbar(im, ax=ax, label="Jaccard")
        ax.set_title(f"Bemis-Murcko scaffold Jaccard ({kind})")
        fig.tight_layout()
        self.results.append(
            FigureResult(figure=fig, file_name=Path(f"scaffold_jaccard_{kind}.png"))
        )

    # ----- Orchestration -----

    def run(self) -> DatasetComparisonSummary:
        with _timed("load_datasets", self._timings):
            self.load_datasets()

        with _timed("build_union", self._timings):
            union = self._build_union()

        per_dataset_counts = [
            PerDatasetCounts(
                name=ld.entry.name,
                role=ld.entry.role,
                n_smiles_unique=len(ld.smiles),
                n_fingerprints=int(ld.fps_packed.shape[0]),
                n_in_union=int(ld.fps_packed.shape[0]),
            )
            for ld in self._loaded
        ]

        with _timed("bitbirch_union", self._timings):
            row_cluster, bb_summary = self.run_bitbirch_union(union)

        with _timed("analyze_clusters", self._timings):
            composition = self.analyze_clusters(union, row_cluster)

        nn = None
        if self.config.nn_tanimoto.enabled:
            with _timed("nn_tanimoto", self._timings):
                nn = self.nn_tanimoto(union)

        overlap = None
        if self.config.scaffold_overlap.enabled:
            with _timed("scaffold_overlap", self._timings):
                overlap = self.scaffold_overlap()

        summary = DatasetComparisonSummary(
            datasets=per_dataset_counts,
            bitbirch=bb_summary,
            composition=composition,
            nn_tanimoto=nn,
            scaffold_overlap=overlap,
            timings_seconds=dict(self._timings),
        )
        self.results.append(
            PydanticResult(
                file_name=Path("dataset_comparison_summary.yaml"), obj=summary
            )
        )
        return summary

    def output(self, output_dir: Path | str | None = None) -> Path:
        out = Path(output_dir) if output_dir is not None else self.config.output_dir
        os.makedirs(out, exist_ok=True)
        for r in self.results:
            r.serialize_to(out)
        return out
