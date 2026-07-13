from __future__ import annotations

import logging
import os
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from ase.data import chemical_symbols
from ase.visualize.plot import plot_atoms
from pydantic import BaseModel, ConfigDict, Field
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem.Scaffolds import MurckoScaffold

from remedi.configuration.dataset_analysis_config import (
    BitBirchConfig,
    MoleculeDatasetAnalysisConfig,
)
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.data_handling.physchem import DESCRIPTORS_2D
from remedi.evaluation.results import (
    EvalResult,
    FigureResult,
    PydanticResult,
)


class _NpzResult(EvalResult):
    """Serializable bundle of numpy arrays written as a single .npz."""

    result_type: str = "npz"
    arrays: dict[str, np.ndarray]

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def serialize_to(self, directory: Path) -> dict:
        output_path = directory / self.file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Build positional kwarg dict; np.savez_compressed accepts **kwds: ArrayLike.
        arrays: dict[str, np.ndarray] = dict(self.arrays)
        np.savez_compressed(
            str(output_path), **arrays
        )  # ty: ignore[invalid-argument-type]
        return {"path": str(output_path)}


class _DatashaderImageResult(EvalResult):
    """Datashader tf.Image saved as PNG."""

    result_type: str = "datashader_image"
    image: object  # tf.Image; kept loose to avoid datashader import at module load

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def serialize_to(self, directory: Path) -> dict:
        from datashader.utils import export_image

        output_path = directory / self.file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_image(
            self.image,
            filename=output_path.stem,
            fmt=".png",
            background="white",
            export_path=str(output_path.parent),
        )
        return {"path": str(output_path)}


log = logging.getLogger(__name__)


# ---------- Descriptor workers (top-level for pickle-ability) ----------

# Disable verbose RDKit warnings in workers.
RDLogger.DisableLog("rdApp.*")


class _MolDescriptors(BaseModel):
    """RDKit-derived per-molecule descriptors."""

    model_config = ConfigDict(extra="forbid")

    mw: float | None = None
    logp: float | None = None
    tpsa: float | None = None
    hbd: int | None = None
    hba: int | None = None
    rot_bonds: int | None = None
    n_rings: int | None = None
    n_aromatic_rings: int | None = None
    n_heavy_atoms: int | None = None
    has_stereo: bool | None = None
    scaffold: str | None = None
    valid: bool = True


def _compute_descriptors(smiles: str) -> _MolDescriptors:
    """Single-SMILES descriptor pass. Top-level so multiprocessing can pickle it.

    The numeric fields come from the shared :data:`physchem.DESCRIPTORS_2D`
    registry (one source of truth with the dataset-creation probe stage); the
    scaffold / stereo fields stay analysis-only.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _MolDescriptors(valid=False)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        scaffold = ""
    d = DESCRIPTORS_2D
    return _MolDescriptors(
        mw=d["mw"](mol),
        logp=d["logp"](mol),
        tpsa=d["tpsa"](mol),
        hbd=int(d["hbd"](mol)),
        hba=int(d["hba"](mol)),
        rot_bonds=int(d["rot_bonds"](mol)),
        n_rings=int(d["n_rings"](mol)),
        n_aromatic_rings=int(d["n_aromatic_rings"](mol)),
        n_heavy_atoms=int(d["n_heavy_atoms"](mol)),
        has_stereo=bool(rdMD.CalcNumAtomStereoCenters(mol) > 0),
        scaffold=scaffold,
        valid=True,
    )


# ---------- Summary stats Pydantic results ----------


class DistributionStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    p05: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p95: float | None = None
    max: float | None = None

    @classmethod
    def from_array(cls, x: np.ndarray) -> DistributionStats:
        x = np.asarray(x, dtype=np.float64).ravel()
        x = x[np.isfinite(x)]
        if x.size == 0:
            return cls(n=0)
        q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
        return cls(
            n=int(x.size),
            mean=float(x.mean()),
            std=float(x.std()),
            min=float(x.min()),
            p05=float(q[0]),
            p25=float(q[1]),
            p50=float(q[2]),
            p75=float(q[3]),
            p95=float(q[4]),
            max=float(x.max()),
        )


class TargetColumnStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column_index: int
    name: str | None = None
    coverage: float
    distribution: DistributionStats


class BitBirchUmapSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    n_points: int = 0
    n_neighbors: int = 0
    min_dist: float = 0.0
    metric: str = ""
    top_clusters_colored: int = 0
    fit_seconds: float | None = None
    notes: str | None = None


class BitBirchSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    n_input_smiles: int = 0
    n_fingerprints: int = 0
    n_clusters: int = 0
    singleton_clusters: int = 0
    largest_cluster: int = 0
    cluster_size_distribution: DistributionStats = Field(
        default_factory=lambda: DistributionStats(n=0)
    )
    threshold_used: float | None = None
    """Actual threshold passed to BitBIRCH (resolved from auto-calibration
    when ``BitBirchConfig.auto_threshold`` is true, else the config value)."""
    threshold_source: Literal["config", "auto"] | None = None
    mean_isim: float | None = None
    """Mean pairwise Tanimoto (iSIM) of the fingerprint set, when computed."""
    std_isim: float | None = None
    """Estimated std of pairwise Tanimoto, when computed."""
    umap: BitBirchUmapSummary | None = None
    notes: str | None = None


def resolve_bitbirch_threshold(
    cfg: BitBirchConfig,
    fps_packed: np.ndarray,
) -> tuple[float, str, float | None, float | None]:
    """Decide the BitBIRCH threshold for a given (packed-uint8) FP matrix.

    Returns ``(threshold, source, mean_isim, std_isim)``. When
    ``cfg.auto_threshold`` is false the source is ``"config"`` and the
    iSIM/std fields are ``None``; otherwise the helper calls
    ``bblean.bitbirch.guess_threshold`` (mean iSIM + factor·std) and the
    returned numbers are filled in so callers can record what was used."""
    if not cfg.auto_threshold:
        return float(cfg.threshold), "config", None, None
    import bblean.bitbirch as bb

    threshold, mean, std = bb.guess_threshold(
        fps_packed,
        input_is_packed=True,
        n_features=cfg.n_features,
        factor=cfg.auto_threshold_factor,
        return_mean_std=True,
    )
    threshold = float(np.clip(threshold, 0.0, 1.0))
    return threshold, "auto", float(mean), float(std)


class DatasetSummary(BaseModel):
    """Top-level analysis summary serialized as a single yaml."""

    model_config = ConfigDict(extra="forbid")

    n_structures: int
    n_molecules: int
    n_atoms: int
    stereocentre_ratio: float | None = None
    atoms_per_structure: DistributionStats
    heteroatoms_per_structure: DistributionStats
    atom_species_counts: dict[str, int]
    charge_distribution: DistributionStats
    multiplicity_distribution: DistributionStats
    drug_likeness: dict[str, DistributionStats] = Field(default_factory=dict)
    lipinski_ro5_pass_fraction: float | None = None
    n_unique_scaffolds: int | None = None
    top_scaffolds: list[tuple[str, int]] = Field(default_factory=list)
    target_columns_system: list[TargetColumnStats] = Field(default_factory=list)
    target_columns_atom: list[TargetColumnStats] = Field(default_factory=list)
    bitbirch: BitBirchSummary | None = None
    timings_seconds: dict[str, float] = Field(default_factory=dict)


# ---------- Utilities ----------


@contextmanager
def _timed(name: str, timings: dict[str, float]) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        timings[name] = elapsed
        log.info("%s took %.2fs", name, elapsed)


def _iter_smiles(smiles_iter: Iterable[str]) -> Iterator[str]:
    """Yield non-empty SMILES strings."""
    for s in smiles_iter:
        if s:
            yield s


def _bblean_available() -> bool:
    try:
        import bblean  # noqa: F401
    except Exception:
        return False
    return True


# ---------- MoleculeDatasetAnalysis ----------


class MoleculeDatasetAnalysis:
    def __init__(
        self,
        dataset: MoleculeDataset,
        config: MoleculeDatasetAnalysisConfig | None = None,
    ):
        self.dataset = dataset
        self.config = config or MoleculeDatasetAnalysisConfig()
        self.results: list[EvalResult] = []
        self._timings: dict[str, float] = {}
        self._rng = np.random.default_rng(self.config.random_seed)

    # ----- Atom-/structure-level metrics (chunked) -----

    def molecule_sizes(self) -> np.ndarray:
        """Per-structure atom counts streamed from ragged pointer."""
        n_struct = self.dataset.N_structures
        if n_struct == 0:
            return np.asarray([], dtype=np.int64)

        ptr = self.dataset.ptr
        chunk_len = getattr(ptr, "chunks", (n_struct + 1,))[0]
        chunk_len = max(1, min(chunk_len, n_struct))

        sizes = np.empty(n_struct, dtype=np.int64)
        offset = 0
        while offset < n_struct:
            end = min(offset + chunk_len, n_struct)
            ptr_chunk = np.asarray(ptr[offset : end + 1], dtype=np.int64)
            sizes[offset:end] = np.diff(ptr_chunk)
            offset = end
        return sizes

    def atom_species(self) -> dict[int, int]:
        n_atoms = self.dataset.N_atoms
        if n_atoms == 0:
            return {}

        atomic_numbers = self.dataset.atomic_numbers
        chunk_len = max(
            1,
            min(self.config.atom_chunk_size, n_atoms),
        )

        counts: Counter[int] = Counter()
        offset = 0
        while offset < n_atoms:
            end = min(offset + chunk_len, n_atoms)
            chunk = np.asarray(atomic_numbers[offset:end], dtype=np.int64)
            if chunk.size:
                unique, c = np.unique(chunk, return_counts=True)
                counts.update(dict(zip(unique.tolist(), c.tolist(), strict=False)))
            offset = end
        return {int(z): int(c) for z, c in counts.items()}

    def heteroatom_counts_per_structure(self) -> np.ndarray:
        """Stream atomic_numbers + ptr in chunks so we never hold all atoms in RAM."""
        n_struct = self.dataset.N_structures
        if n_struct == 0:
            return np.asarray([], dtype=np.int64)

        ptr_arr = self.dataset.ptr
        atomic_numbers = self.dataset.atomic_numbers
        atom_chunk = max(1, self.config.atom_chunk_size)

        out = np.zeros(n_struct, dtype=np.int64)
        struct_chunk = max(1, self.config.structure_chunk_size)

        offset = 0
        while offset < n_struct:
            end = min(offset + struct_chunk, n_struct)
            ptr_chunk = np.asarray(ptr_arr[offset : end + 1], dtype=np.int64)
            atom_start = int(ptr_chunk[0])
            atom_end = int(ptr_chunk[-1])
            if atom_end == atom_start:
                offset = end
                continue
            # Stream atoms for this structure chunk in atom_chunk slices,
            # then combine with reduceat over the local ptr.
            local_ptr = ptr_chunk - atom_start
            mask = np.empty(atom_end - atom_start, dtype=np.int8)
            sub = 0
            while sub < atom_end - atom_start:
                sub_end = min(sub + atom_chunk, atom_end - atom_start)
                a_chunk = np.asarray(
                    atomic_numbers[atom_start + sub : atom_start + sub_end]
                )
                mask[sub:sub_end] = ((a_chunk != 1) & (a_chunk != 6)).astype(np.int8)
                sub = sub_end
            out[offset:end] = np.add.reduceat(mask.astype(np.int64), local_ptr[:-1])
            offset = end
        return out

    # ----- SMILES / RDKit metrics -----

    def _unique_smiles(self) -> list[str]:
        """Return deduplicated isomeric SMILES (one per molecule_id)."""
        store = self.dataset.isomeric_smiles
        if store is None:
            return []
        smiles = list(_iter_smiles(store))
        # Dedup while preserving order.
        seen: set[str] = set()
        unique: list[str] = []
        for s in smiles:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique

    def _sample_smiles(self, smiles: list[str]) -> list[str]:
        cap = self.config.rdkit_subsample
        if cap is None or len(smiles) <= cap:
            return smiles
        idx = self._rng.choice(len(smiles), size=cap, replace=False)
        idx.sort()
        return [smiles[int(i)] for i in idx]

    def compute_rdkit_descriptors(self, smiles: list[str]) -> list[_MolDescriptors]:
        if not smiles:
            return []
        workers = max(1, self.config.rdkit_n_workers)
        if workers == 1 or len(smiles) < 1000:
            return [_compute_descriptors(s) for s in smiles]
        # chunksize tuned to keep IPC overhead low for many small jobs
        chunksize = max(64, len(smiles) // (workers * 32) or 1)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_compute_descriptors, smiles, chunksize=chunksize))

    # ----- Plots -----

    def plot_molecule_size_distribution(self, sizes: np.ndarray) -> None:
        fig, ax = plt.subplots()
        if sizes.size:
            ax.hist(sizes, bins="auto", color="#4c72b0")
        ax.set_xlabel("Atoms per structure")
        ax.set_ylabel("Frequency")
        ax.set_title("Molecule size distribution")
        fig.tight_layout()
        self.results.append(
            FigureResult(figure=fig, file_name=Path("molecule_size_distribution.png"))
        )

    def plot_atom_species_histogram(self, species_counts: dict[int, int]) -> None:
        if not species_counts:
            return
        items = sorted(species_counts.items(), key=lambda kv: kv[1], reverse=True)
        labels = [chemical_symbols[z] for z, _ in items]
        counts = [c for _, c in items]
        fig, ax = plt.subplots()
        ax.bar(labels, counts, color="#dd8452")
        ax.set_xlabel("Atomic species")
        ax.set_ylabel("Frequency")
        ax.set_yscale("log")
        ax.set_title("Atom species frequency")
        fig.tight_layout()
        self.results.append(
            FigureResult(figure=fig, file_name=Path("atom_species_histogram.png"))
        )

    def plot_heteroatom_distribution(self, hetero_counts: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(12, 4))
        if hetero_counts.size:
            unique, freq = np.unique(hetero_counts, return_counts=True)
            ax.bar(unique, freq, align="center", width=0.8, color="#55a868")
            ax.set_xlabel("Heteroatoms per structure")
            ax.set_ylabel("Number of structures")
            ax.set_title("Heteroatom distribution")
            ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        else:
            ax.text(0.5, 0.5, "No structures available", ha="center", va="center")
            ax.axis("off")
        fig.tight_layout()
        self.results.append(
            FigureResult(figure=fig, file_name=Path("heteroatom_distribution.png"))
        )

    def plot_charge_multiplicity(self) -> None:
        n_struct = self.dataset.N_structures
        if n_struct == 0:
            return
        charges = np.asarray(self.dataset.total_charge[:n_struct])
        mults = np.asarray(self.dataset.multiplicity[:n_struct])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        if charges.size:
            u, c = np.unique(np.round(charges).astype(np.int64), return_counts=True)
            axes[0].bar(u, c, color="#4c72b0")
        axes[0].set_xlabel("Total charge")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Charge distribution")

        if mults.size:
            u, c = np.unique(np.round(mults).astype(np.int64), return_counts=True)
            axes[1].bar(u, c, color="#c44e52")
        axes[1].set_xlabel("Multiplicity")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Multiplicity distribution")
        fig.tight_layout()
        self.results.append(
            FigureResult(figure=fig, file_name=Path("charge_multiplicity.png"))
        )

    def plot_descriptor_distributions(
        self, descriptors: list[_MolDescriptors]
    ) -> dict[str, DistributionStats]:
        """Plot drug-likeness panels and return per-descriptor stats."""
        if not descriptors:
            return {}
        fields = [
            ("mw", "Molecular weight (Da)"),
            ("logp", "LogP"),
            ("tpsa", "TPSA (Å²)"),
            ("hbd", "H-bond donors"),
            ("hba", "H-bond acceptors"),
            ("rot_bonds", "Rotatable bonds"),
            ("n_rings", "Ring count"),
            ("n_aromatic_rings", "Aromatic rings"),
            ("n_heavy_atoms", "Heavy atoms"),
        ]

        stats: dict[str, DistributionStats] = {}
        fig, axes = plt.subplots(3, 3, figsize=(14, 10))
        for ax, (attr, label) in zip(axes.ravel(), fields, strict=False):
            values = np.asarray(
                [getattr(d, attr) for d in descriptors if getattr(d, attr) is not None],
                dtype=np.float64,
            )
            stats[attr] = DistributionStats.from_array(values)
            if values.size:
                ax.hist(values, bins=50, color="#4c72b0")
            ax.set_title(label)
            ax.set_ylabel("Count")
        fig.suptitle(f"Drug-likeness descriptors (n={len(descriptors)} unique mols)")
        fig.tight_layout()
        self.results.append(
            FigureResult(figure=fig, file_name=Path("drug_likeness_distributions.png"))
        )
        return stats

    def plot_relaxed_atoms(self, n_max: int | None = None) -> None:
        n_max = n_max or self.config.max_example_molecules
        molecules = self.dataset.get_all_molecules(N_molecules=n_max)
        if not molecules:
            return
        n_horizontal = 3
        n_vertical = (len(molecules) + n_horizontal - 1) // n_horizontal
        fig, axarr = plt.subplots(n_vertical, n_horizontal, squeeze=False)
        fig.set_figheight(4 * n_vertical)
        fig.set_figwidth(4 * n_horizontal)
        for i, mol in enumerate(molecules):
            plot_atoms(mol, axarr[i // n_horizontal, i % n_horizontal])
        for j in range(len(molecules), n_vertical * n_horizontal):
            fig.delaxes(axarr[j // n_horizontal, j % n_horizontal])
        self.results.append(
            FigureResult(figure=fig, file_name=Path("example_molecules.png"))
        )

    # ----- Targets EDA -----

    def _target_column_stats(
        self,
        targets: np.ndarray,
        masks: np.ndarray | None,
        prefix: str,
    ) -> list[TargetColumnStats]:
        if targets is None or targets.shape[0] == 0:
            return []
        n_rows, n_cols = targets.shape
        if masks is None:
            masks = np.ones_like(targets, dtype=bool)
        else:
            masks = masks.astype(bool, copy=False)

        out: list[TargetColumnStats] = []
        for j in range(n_cols):
            col = targets[:, j]
            mask_col = masks[:, j]
            valid = col[mask_col]
            coverage = float(mask_col.sum()) / float(n_rows) if n_rows else 0.0
            out.append(
                TargetColumnStats(
                    column_index=j,
                    name=f"{prefix}_{j}",
                    coverage=coverage,
                    distribution=DistributionStats.from_array(valid),
                )
            )
        return out

    def target_stats(self) -> tuple[list[TargetColumnStats], list[TargetColumnStats]]:
        n_struct = self.dataset.N_structures
        n_atoms = self.dataset.N_atoms
        sys_stats: list[TargetColumnStats] = []
        atom_stats: list[TargetColumnStats] = []
        if self.dataset.targets_system is not None and n_struct:
            t = np.asarray(self.dataset.targets_system[:n_struct])
            m = (
                np.asarray(self.dataset.mask_system[:n_struct])
                if self.dataset.mask_system is not None
                else None
            )
            sys_stats = self._target_column_stats(t, m, "sys")
        if self.dataset.targets_atom is not None and n_atoms:
            t = np.asarray(self.dataset.targets_atom[:n_atoms])
            m = (
                np.asarray(self.dataset.mask_atom[:n_atoms])
                if self.dataset.mask_atom is not None
                else None
            )
            atom_stats = self._target_column_stats(t, m, "atom")
        return sys_stats, atom_stats

    # ----- BitBIRCH clustering -----

    def run_bitbirch(self, smiles: list[str]) -> BitBirchSummary:
        cfg = self.config.bitbirch
        if not cfg.enabled:
            return BitBirchSummary(enabled=False, notes="disabled by config")
        if not smiles:
            return BitBirchSummary(enabled=True, notes="no SMILES available")
        if not _bblean_available():
            return BitBirchSummary(
                enabled=True,
                notes="bblean not installed (install with `uv sync --extra cluster`)",
            )

        import bblean  # local import keeps it optional

        if cfg.max_molecules is not None and len(smiles) > cfg.max_molecules:
            idx = self._rng.choice(len(smiles), size=cfg.max_molecules, replace=False)
            idx.sort()
            smi_subset = [smiles[int(i)] for i in idx]
        else:
            smi_subset = smiles

        log.info(
            "BitBIRCH: fingerprinting %d SMILES (kind=%s, n_features=%d)",
            len(smi_subset),
            cfg.fingerprint_kind,
            cfg.n_features,
        )
        result = bblean.fps_from_smiles(
            smi_subset,
            kind=cfg.fingerprint_kind,
            n_features=cfg.n_features,
            pack=True,
            skip_invalid=True,
        )
        # `skip_invalid=True` returns a tuple (fps, indices); else just fps.
        if isinstance(result, tuple):
            fps, _kept_idx = result
            n_fps = int(fps.shape[0]) if fps is not None else 0
            log.info(
                "BitBIRCH: %d/%d SMILES survived fingerprinting", n_fps, len(smi_subset)
            )
        else:
            fps = result
            n_fps = int(fps.shape[0]) if fps is not None else 0

        if fps is None or n_fps == 0:
            return BitBirchSummary(
                enabled=True,
                n_input_smiles=len(smi_subset),
                notes="no valid fingerprints produced",
            )

        threshold, source, mean_isim, std_isim = resolve_bitbirch_threshold(cfg, fps)
        log.info(
            "BitBIRCH: threshold=%.3f (source=%s, mean_iSIM=%s, std=%s)",
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
        tree.fit(fps)
        clusters = tree.get_cluster_mol_ids()
        sizes = np.asarray([len(c) for c in clusters], dtype=np.int64)

        # row -> cluster id assignment (row indices into the fps array)
        row_cluster = np.full(n_fps, -1, dtype=np.int64)
        for cid, rows in enumerate(clusters):
            row_cluster[np.asarray(rows, dtype=np.int64)] = cid

        # Cluster-size histogram figure
        fig, ax = plt.subplots(figsize=(8, 4))
        if sizes.size:
            ax.hist(
                sizes, bins=min(50, max(5, int(np.sqrt(sizes.size)))), color="#8172b3"
            )
            ax.set_yscale("log")
        ax.set_xlabel("Molecules per cluster")
        ax.set_ylabel("Cluster count (log)")
        ax.set_title(
            f"BitBIRCH clusters (n={sizes.size}, threshold={threshold:.2f}"
            f" [{source}], kind={cfg.fingerprint_kind})"
        )
        fig.tight_layout()
        self.results.append(
            FigureResult(figure=fig, file_name=Path("bitbirch_cluster_sizes.png"))
        )

        umap_summary = self._bitbirch_umap(fps, row_cluster, sizes)

        return BitBirchSummary(
            enabled=True,
            n_input_smiles=len(smi_subset),
            n_fingerprints=n_fps,
            n_clusters=int(sizes.size),
            singleton_clusters=int((sizes == 1).sum()),
            largest_cluster=int(sizes.max()) if sizes.size else 0,
            cluster_size_distribution=DistributionStats.from_array(sizes),
            threshold_used=threshold,
            threshold_source=source,
            mean_isim=mean_isim,
            std_isim=std_isim,
            umap=umap_summary,
        )

    def _bitbirch_umap(
        self,
        fps: np.ndarray,
        row_cluster: np.ndarray,
        cluster_sizes: np.ndarray,
    ) -> BitBirchUmapSummary:
        """Project BitBIRCH fingerprints to 2D with UMAP and datashade by cluster id.

        - Stratified subsample: keep all members of the top-N largest clusters,
          fill the remaining budget uniformly from the rest. This guarantees the
          biggest clusters are visible even when sample_size << n_fps.
        - UMAP runs on unpacked binary features with Jaccard distance by default.
        - Datashader renders categorical colors for top-N; everything else is grey.
        """
        cfg = self.config.bitbirch.umap
        if not cfg.enabled:
            return BitBirchUmapSummary(enabled=False, notes="disabled by config")

        n_fps = int(fps.shape[0])
        if n_fps == 0:
            return BitBirchUmapSummary(enabled=True, notes="no fingerprints")

        top_n = min(cfg.top_clusters_colored, int(cluster_sizes.size))
        top_cids = (
            np.argsort(cluster_sizes)[-top_n:][::-1]
            if top_n > 0
            else np.asarray([], dtype=np.int64)
        )
        top_set = set(int(c) for c in top_cids)

        cap = cfg.sample_size if cfg.sample_size is not None else n_fps
        cap = min(cap, n_fps)

        top_rows = (
            np.where(np.isin(row_cluster, np.asarray(list(top_set), dtype=np.int64)))[0]
            if top_set
            else np.asarray([], dtype=np.int64)
        )
        # Cap top-cluster contribution to half of the budget to keep balance.
        top_budget = min(top_rows.size, max(1, cap // 2))
        if top_rows.size > top_budget:
            top_rows = self._rng.choice(top_rows, size=top_budget, replace=False)
        rest_rows = np.setdiff1d(np.arange(n_fps), top_rows, assume_unique=False)
        rest_budget = max(0, cap - int(top_rows.size))
        if rest_rows.size > rest_budget:
            rest_rows = self._rng.choice(rest_rows, size=rest_budget, replace=False)
        sample_rows = np.concatenate([top_rows, rest_rows]).astype(np.int64)
        sample_rows.sort()

        import bblean  # local import; we only get here if bblean was available

        n_features = self.config.bitbirch.n_features
        unpacked = bblean.unpack_fingerprints(fps[sample_rows], n_features=n_features)

        try:
            import umap
        except Exception as exc:
            return BitBirchUmapSummary(
                enabled=True, notes=f"umap-learn import failed: {exc!r}"
            )

        log.info(
            "BitBIRCH UMAP: fitting %d points (n_neighbors=%d, metric=%s)",
            sample_rows.size,
            cfg.n_neighbors,
            cfg.metric,
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

        self._render_bitbirch_umap(
            coords, row_cluster[sample_rows], top_cids, cfg.plot_size
        )

        # Save coords so users can re-render without refitting.
        coords_path = Path("bitbirch_umap_coords.npz")
        self.results.append(
            _NpzResult(
                file_name=coords_path,
                arrays={
                    "coords": coords.astype(np.float32),
                    "cluster_id": row_cluster[sample_rows].astype(np.int64),
                    "row_index": sample_rows.astype(np.int64),
                    "top_cluster_ids": top_cids.astype(np.int64),
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

    def _render_bitbirch_umap(
        self,
        coords: np.ndarray,
        cluster_ids: np.ndarray,
        top_cids: np.ndarray,
        plot_size: int,
    ) -> None:
        import datashader as ds
        import datashader.transfer_functions as tf
        import matplotlib.colors as mcolors
        import pandas as pd

        # Label points: top clusters keep their numeric id (as str), rest get "other".
        top_lookup = {int(c): f"c{int(c)}" for c in top_cids}
        labels = np.asarray(
            [
                "other" if int(c) not in top_lookup else top_lookup[int(c)]
                for c in cluster_ids
            ]
        )

        categories = ["other", *(top_lookup[int(c)] for c in top_cids)]
        cat = pd.Categorical(labels, categories=categories)

        cmap = plt.colormaps.get_cmap("tab20")
        color_key: dict[str, str] = {"other": "#cccccc"}
        for i, c in enumerate(top_cids):
            color_key[top_lookup[int(c)]] = mcolors.to_hex(cmap(i % cmap.N))

        df = pd.DataFrame(
            {
                "x": coords[:, 0].astype(np.float32),
                "y": coords[:, 1].astype(np.float32),
                "cat": cat,
            }
        )
        x_min, x_max = float(df["x"].min()), float(df["x"].max())
        y_min, y_max = float(df["y"].min()), float(df["y"].max())
        pad_x = 0.03 * (x_max - x_min + 1e-9)
        pad_y = 0.03 * (y_max - y_min + 1e-9)
        canvas = ds.Canvas(
            plot_width=plot_size,
            plot_height=plot_size,
            x_range=(x_min - pad_x, x_max + pad_x),
            y_range=(y_min - pad_y, y_max + pad_y),
        )
        agg = canvas.points(df, "x", "y", ds.count_cat("cat"))
        img = tf.shade(agg, color_key=color_key, how="eq_hist", min_alpha=180)
        img = tf.dynspread(img, threshold=0.6, max_px=4)
        img = tf.set_background(img, "white")

        self.results.append(
            _DatashaderImageResult(
                file_name=Path("bitbirch_umap.png"),
                image=img,
            )
        )

        # Legend figure (small, separate file) so users can identify clusters.
        legend_fig, ax = plt.subplots(figsize=(4, max(2.0, 0.25 * (len(top_cids) + 1))))
        handles = [
            plt.Line2D(
                [0], [0], marker="o", color=color_key[name], linestyle="", label=name
            )
            for name in categories
        ]
        ax.legend(handles=handles, loc="center", frameon=False, ncol=1)
        ax.axis("off")
        legend_fig.tight_layout()
        self.results.append(
            FigureResult(figure=legend_fig, file_name=Path("bitbirch_umap_legend.png"))
        )

    # ----- Orchestration -----

    def run(self) -> DatasetSummary:
        timings = self._timings
        ds = self.dataset

        n_struct = ds.N_structures
        n_mols = ds.N_molecules
        n_atoms = ds.N_atoms
        log.info(
            "Analysing dataset: N_structures=%d, N_molecules=%d, N_atoms=%d",
            n_struct,
            n_mols,
            n_atoms,
        )

        with _timed("molecule_sizes", timings):
            sizes = self.molecule_sizes()
        self.plot_molecule_size_distribution(sizes)

        with _timed("atom_species", timings):
            species_counts = self.atom_species()
        self.plot_atom_species_histogram(species_counts)

        with _timed("heteroatom_counts", timings):
            hetero = self.heteroatom_counts_per_structure()
        self.plot_heteroatom_distribution(hetero)

        with _timed("charge_multiplicity", timings):
            self.plot_charge_multiplicity()

        with _timed("example_molecules", timings):
            self.plot_relaxed_atoms()

        # SMILES-derived metrics
        descriptors: list[_MolDescriptors] = []
        drug_stats: dict[str, DistributionStats] = {}
        stereo_ratio: float | None = None
        top_scaffolds: list[tuple[str, int]] = []
        n_unique_scaffolds: int | None = None
        ro5_pass_fraction: float | None = None
        bb_summary: BitBirchSummary | None = None

        unique_smiles: list[str] = []
        if ds.isomeric_smiles is not None:
            with _timed("collect_unique_smiles", timings):
                unique_smiles = self._unique_smiles()
            log.info("Collected %d unique SMILES", len(unique_smiles))

        if unique_smiles:
            with _timed("rdkit_descriptors", timings):
                sample = self._sample_smiles(unique_smiles)
                descriptors = self.compute_rdkit_descriptors(sample)
            valid = [d for d in descriptors if d.valid]
            if valid:
                drug_stats = self.plot_descriptor_distributions(valid)
                stereo_ratio = float(sum(1 for d in valid if d.has_stereo) / len(valid))
                scaffold_counter: Counter[str] = Counter(
                    d.scaffold for d in valid if d.scaffold
                )
                n_unique_scaffolds = len(scaffold_counter)
                top_scaffolds = scaffold_counter.most_common(
                    self.config.top_n_scaffolds
                )
                ro5_pass_fraction = float(
                    sum(
                        1
                        for d in valid
                        if (d.mw or 0) <= 500
                        and (d.logp or 0) <= 5
                        and (d.hbd or 0) <= 5
                        and (d.hba or 0) <= 10
                    )
                    / len(valid)
                )

            with _timed("bitbirch", timings):
                bb_summary = self.run_bitbirch(unique_smiles)

        with _timed("targets", timings):
            sys_stats, atom_stats = self.target_stats()

        charges = np.asarray(ds.total_charge[:n_struct]) if n_struct else np.asarray([])
        mults = np.asarray(ds.multiplicity[:n_struct]) if n_struct else np.asarray([])

        summary = DatasetSummary(
            n_structures=n_struct,
            n_molecules=n_mols,
            n_atoms=n_atoms,
            stereocentre_ratio=stereo_ratio,
            atoms_per_structure=DistributionStats.from_array(sizes),
            heteroatoms_per_structure=DistributionStats.from_array(hetero),
            atom_species_counts={
                chemical_symbols[z]: c for z, c in species_counts.items()
            },
            charge_distribution=DistributionStats.from_array(charges),
            multiplicity_distribution=DistributionStats.from_array(mults),
            drug_likeness=drug_stats,
            lipinski_ro5_pass_fraction=ro5_pass_fraction,
            n_unique_scaffolds=n_unique_scaffolds,
            top_scaffolds=top_scaffolds,
            target_columns_system=sys_stats,
            target_columns_atom=atom_stats,
            bitbirch=bb_summary,
            timings_seconds=dict(timings),
        )

        self.results.append(
            PydanticResult(file_name=Path("dataset_summary.yaml"), obj=summary)
        )
        return summary

    def output(self, output_dir: Path | str) -> None:
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        for result in self.results:
            result.serialize_to(output_dir)
