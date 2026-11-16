from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from ase.data import chemical_symbols
from ase.visualize.plot import plot_atoms
from pydantic import BaseModel, ConfigDict, Field
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors as rdMD

from remedi_data.bundle import SPEC_FILENAME, read_spec, read_table
from remedi_data.chemistry.fingerprints import morgan_fingerprints
from remedi_data.chemistry.physchem import DESCRIPTORS_2D
from remedi_data.chemistry.splits import murcko_scaffold
from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_data.parallel import process_map
from remedi_data.runs.results import (
    ArrayResult,
    EvalResult,
    PydanticResult,
)
from remedi_eval.dataset_analysis.bitbirch import (
    BitBirchSummary,
    fit_bitbirch,
    fit_umap,
)
from remedi_eval.dataset_analysis.config import MoleculeDatasetAnalysisConfig
from remedi_eval.dataset_analysis.statistics import DistributionStats, timed
from remedi_eval.results import FigureResult

#: The analyses group molecules by the scaffold the scaffold splits use
#: (:func:`remedi_data.chemistry.splits.murcko_scaffold`, chirality kept), so a
#: scaffold count here means what a scaffold split would see.
SCAFFOLD_INCLUDE_CHIRALITY = True

#: SMILES per process-pool task for the per-molecule RDKit work.
RDKIT_CHUNK_SIZE = 1_000


class _DatashaderImageResult(EvalResult):
    """Datashader tf.Image saved as PNG."""

    result_type: str = "datashader_image"
    image: object  # tf.Image; kept loose to avoid datashader import at module load

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def serialize_to(self, directory: Path) -> dict[str, Any]:
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
        return self._manifest_entry()


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
    scaffold / stereo fields stay analysis-only. ``scaffold`` is the splits'
    Bemis-Murcko scaffold, ``""`` for an acyclic molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _MolDescriptors(valid=False)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    scaffold = murcko_scaffold(smiles, include_chirality=SCAFFOLD_INCLUDE_CHIRALITY)
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


class TargetColumnStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column_index: int
    name: str
    coverage: float
    distribution: DistributionStats


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
    target_columns: list[TargetColumnStats] = Field(default_factory=list)
    bitbirch: BitBirchSummary | None = None
    timings_seconds: dict[str, float] = Field(default_factory=dict)


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

    def _sample_smiles(self, smiles: list[str]) -> list[str]:
        cap = self.config.rdkit_subsample
        if cap is None or len(smiles) <= cap:
            return smiles
        idx = self._rng.choice(len(smiles), size=cap, replace=False)
        idx.sort()
        return [smiles[int(i)] for i in idx]

    def compute_rdkit_descriptors(self, smiles: list[str]) -> list[_MolDescriptors]:
        return process_map(
            _compute_descriptors,
            smiles,
            workers=self.config.rdkit_n_workers,
            chunk_size=RDKIT_CHUNK_SIZE,
        )

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

    def target_stats(self) -> list[TargetColumnStats]:
        """Coverage and distribution of every label ``dataset.yaml`` declares.

        The values are the label columns of ``table.parquet`` (NaN = missing).
        A dataset without ``dataset.yaml`` declares no labels and gets none.
        """
        directory = Path(self.dataset.path)
        if not (directory / SPEC_FILENAME).is_file():
            return []
        names = read_spec(directory).label_names()
        if not names:
            return []
        values = read_table(directory, columns=names).to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        present = ~np.isnan(values)
        n_rows = values.shape[0]
        return [
            TargetColumnStats(
                column_index=index,
                name=name,
                coverage=float(present[:, index].sum()) / n_rows if n_rows else 0.0,
                distribution=DistributionStats.from_array(
                    values[present[:, index], index]
                ),
            )
            for index, name in enumerate(names)
        ]

    # ----- BitBIRCH clustering -----

    def run_bitbirch(self, smiles: list[str]) -> BitBirchSummary:
        cfg = self.config.bitbirch
        if not cfg.enabled:
            return BitBirchSummary(enabled=False, notes="disabled by config")
        if not smiles:
            return BitBirchSummary(enabled=True, notes="no SMILES available")

        if cfg.max_molecules is not None and len(smiles) > cfg.max_molecules:
            idx = self._rng.choice(len(smiles), size=cfg.max_molecules, replace=False)
            idx.sort()
            smi_subset = [smiles[int(i)] for i in idx]
        else:
            smi_subset = smiles

        log.info(
            "BitBIRCH: fingerprinting %d SMILES (Morgan radius %d, %d bits)",
            len(smi_subset),
            cfg.radius,
            cfg.fingerprint_size,
        )
        fps = morgan_fingerprints(
            smi_subset,
            layout="packed_bytes",
            radius=cfg.radius,
            size=cfg.fingerprint_size,
            workers=self.config.rdkit_n_workers,
        )
        clusters = fit_bitbirch(fps, cfg)
        self.results.append(
            FigureResult(
                figure=clusters.size_histogram(
                    f"BitBIRCH clusters, Morgan radius {cfg.radius}"
                ),
                file_name=Path("bitbirch_cluster_sizes.png"),
            )
        )
        fit, umap_summary = fit_umap(
            fps, clusters, cfg.umap, cfg.fingerprint_size, self._rng
        )
        if fit is not None:
            cluster_ids = clusters.row_cluster[fit.rows]
            self._render_bitbirch_umap(
                fit.coordinates, cluster_ids, fit.top_cluster_ids, cfg.umap.plot_size
            )
            # Save coords so users can re-render without refitting.
            self.results.append(
                ArrayResult(
                    file_name=Path("bitbirch_umap_coords.npz"),
                    arrays={
                        "coords": fit.coordinates.astype(np.float32),
                        "cluster_id": cluster_ids.astype(np.int64),
                        "row_index": fit.rows,
                        "top_cluster_ids": fit.top_cluster_ids.astype(np.int64),
                    },
                )
            )
        return clusters.summary(len(smi_subset), umap_summary)

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

        with timed("molecule_sizes", timings):
            sizes = self.molecule_sizes()
        self.plot_molecule_size_distribution(sizes)

        with timed("atom_species", timings):
            species_counts = self.atom_species()
        self.plot_atom_species_histogram(species_counts)

        with timed("heteroatom_counts", timings):
            hetero = self.heteroatom_counts_per_structure()
        self.plot_heteroatom_distribution(hetero)

        with timed("charge_multiplicity", timings):
            self.plot_charge_multiplicity()

        with timed("example_molecules", timings):
            self.plot_relaxed_atoms()

        # SMILES-derived metrics
        descriptors: list[_MolDescriptors] = []
        drug_stats: dict[str, DistributionStats] = {}
        stereo_ratio: float | None = None
        top_scaffolds: list[tuple[str, int]] = []
        n_unique_scaffolds: int | None = None
        ro5_pass_fraction: float | None = None
        bb_summary: BitBirchSummary | None = None

        with timed("collect_unique_smiles", timings):
            unique_smiles = ds.unique_smiles()
            log.info("Collected %d unique SMILES", len(unique_smiles))

        if unique_smiles:
            with timed("rdkit_descriptors", timings):
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

            with timed("bitbirch", timings):
                bb_summary = self.run_bitbirch(unique_smiles)

        with timed("targets", timings):
            label_stats = self.target_stats()

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
            target_columns=label_stats,
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
