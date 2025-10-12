import os
from itertools import chain, combinations, groupby
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.data import chemical_symbols
from ase.visualize.plot import plot_atoms
from matplotlib.figure import Figure
from pydantic import BaseModel, Field, ConfigDict

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset


class FigureResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    result_type: Literal["figure"] = "figure"
    file_name: str
    figure: Figure
    save_kwargs: dict[str, Any] = Field(default_factory=dict)

    def serialize_to(self, directory: Path) -> dict[str, Any]:
        output_path = directory / self.file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(output_path, **self.save_kwargs)
        plt.close(self.figure)


class MoleculeDatasetAnalysis:

    def __init__(self, dataset: MoleculeDataset):

        self.dataset = dataset
        self.results :list[FigureResult] = []

    def molecule_sizes(self) -> np.ndarray:
        """Per-structure atom counts from the ragged pointer."""
        # Only take the active part of ptr (up to sentinel)
        ptr = np.asarray(self.dataset.ptr[: self.dataset.N_structures + 1])
        return np.diff(ptr)  # shape: (n_structures,)

    def atom_species(self) -> np.ndarray:

        atomic_numbers, counts = np.unique_counts(self.dataset.atomic_numbers[:self.dataset.N_atoms])

        return {n: c for n, c in zip(atomic_numbers, counts, strict=False)}


    def get_descriptor_mean_std(self):
        
        embeddings = self.dataset.atomic_embeddings

        data = np.asarray(embeddings)
        reduce_axes = tuple(range(data.ndim - 1))
        mean = np.mean(data, axis=reduce_axes)
        print(mean)
        std = np.std(data, axis=reduce_axes)
        return mean, std

    def get_descriptor_norm(self):
        embeddings = self.dataset.atomic_embeddings
        data = np.asarray(embeddings)
        norm = np.linalg.norm(data,axis = -1)

        return norm 

    def get_conformer_distance_distribution(self):
        pass

    def plot_descriptor_mean_std_distribution(self):
        mean, std = self.get_descriptor_mean_std()
        mean = np.asarray(mean, dtype=float).ravel()
        std = np.asarray(std, dtype=float).ravel()

        fig, ax = plt.subplots(figsize=(12, 4))
        if mean.size > 0:
            indices = np.arange(mean.size)
            error_kw = {"ecolor": "0.3", "alpha": 0.7, "elinewidth": 0.8, "capsize": 2}
            ax.bar(
                indices,
                mean,
                yerr=std,
                align="center",
                color="#4c72b0",
                edgecolor="none",
                width=0.9,
                error_kw=error_kw,
            )
            ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.grid(axis="y", alpha=0.2, linewidth=0.5)
            ax.set_xlim(-0.5, mean.size - 0.5)
            ax.set_xlabel("Descriptor channel")
            ax.set_ylabel("Mean value")
            ax.set_title("Descriptor channel statistics")
            if mean.size > 20:
                step = max(mean.size // 10, 1)
                ax.set_xticks(indices[::step])
                ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        else:
            ax.text(0.5, 0.5, "No descriptor channels available", ha="center", va="center")
            ax.axis("off")
        fig.tight_layout()
        result = FigureResult(figure=fig, file_name="descriptor_channel_mean_std.png")
        self.results.append(result)
        return result

    def plot_descriptor_norm_distribution(self):

        norm = self.get_descriptor_norm()
        fig, ax = plt.subplots()
        flat_norm = np.asarray(norm, dtype=float).ravel()
        if flat_norm.size > 0:
            counts, bin_edges = np.histogram(flat_norm, bins="auto")
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            bar_widths = np.diff(bin_edges)
            ax.bar(bin_centers, counts, width=bar_widths, align="center")
            ax.set_xlabel("Descriptor norm")
            ax.set_ylabel("Frequency")
            ax.set_title("Descriptor norm distribution")
        else:
            ax.text(0.5, 0.5, "No descriptor norms available", ha="center", va="center")
            ax.axis("off")
        fig.tight_layout()
        result = FigureResult(figure=fig, file_name="descriptor_norm_distribution.png")
        self.results.append(result)

    def plot_atom_species_histogram(self):

        atom_species = self.atom_species()

        if not atom_species:
            return

        # Sort species by descending frequency for readability
        sorted_species = sorted(atom_species.items(), key=lambda item: item[1], reverse=True)
        labels = [chemical_symbols[atomic_number] for atomic_number, _ in sorted_species]
        counts = [count for _, count in sorted_species]

        fig, ax = plt.subplots()
        ax.bar(labels, counts)
        ax.set_xlabel("Atomic species")
        ax.set_ylabel("Frequency")
        ax.set_title("Atom species frequency")
        fig.tight_layout()

        self.results.append(FigureResult(figure=fig, file_name="atom_species_histogram.png"))



    def plot_molecule_size_distribution(self) -> Figure:
        sizes = self.molecule_sizes()
        fig = plt.figure()
        plt.hist(sizes, bins="auto")
        plt.xlabel("Atoms per structure")
        plt.ylabel("Frequency")

        self.results.append(FigureResult(figure=fig, file_name="molecule_size_distribution.png"))


    def plot_regression_task_distribution(self):
        pass

    def plot_relaxed_atoms(self, N_max_molecules: int | None = None):

        molecules = self.dataset.get_all_molecules()

        N_horizontal = 3
        N_vertical = (len(molecules) // 3 )+1

        fig, axarr = plt.subplots(N_vertical, N_horizontal)

        fig.set_figheight(4*N_vertical)
        fig.set_figwidth(4*N_horizontal)

        for i, mol in enumerate(molecules):
            plot_atoms(mol, axarr[i // 3 , i % 3])


    def run(self):
        self.plot_molecule_size_distribution()
        self.plot_atom_species_histogram()
        self.plot_descriptor_norm_distribution()
        self.plot_descriptor_mean_std_distribution()

    def output(self, output_dir: Path):

        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        for result in self.results:
            result.serialize_to(output_dir)

class DatasetPostLoadAnalysis:


    def __init__(self, dataset, output_dir):
        self.dataset = dataset
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

    def calculate_atomic_descriptor_norms(atomic_descriptors, padding_masks):
        pass

    def get_invariants_std(self):

        irreps = get_mace_calculator_irrep_signature(self.dataset.dataset_config.embedding_model_config.mace_calc)
        invariant_indices, _ = get_invariant_indices(irreps)


        invariants = self.dataset.embeddings[:,:,invariant_indices]

        std = torch.std(invariants, dim = (0,1))

        plt.figure()
        plt.hist(std)
        plt.savefig(f"{self.output_dir}/invariants_std.png")

        bar_std = plt.figure(figsize=(12, 4), dpi=100)
        plt.bar(np.arange(len(std)),std, width=1,align="edge")
        plt.xlim([0, len(std)])
        plt.xlabel("MACE feature dimension")
        plt.ylabel("Std of each invariant MACE feature dimension")
        bar_std.savefig(f"{self.output_dir}/embeddings_std.png")


    def count_samples_per_task(self):
        num_samples = self.dataset.regression_masks.sum(0).tolist()
        samples_per_task = dict(zip(self.dataset.dataset_config.get_task_names(),num_samples, strict=False))
        return samples_per_task

    def mean_and_std(self):
        task_names = self.dataset.dataset_config.get_task_names()

        rt = self.dataset.regression_targets.cpu().numpy()
        rm = self.dataset.regression_masks.bool().cpu().numpy()

        means = np.mean(rt, axis = 0, where = rm)
        stds = np.std(rt, axis = 0, where = rm)

        return dict(zip(task_names, means, strict=False)), dict(zip(task_names, stds, strict=False))


    def check_conformer_distance(self):

        if self.dataset.dataset_config.N_conformers == 1:
            return

        mol_id_chunks = [list(g) for _, g in groupby(range(len(self.dataset.mol_ids)), key = lambda i : self.dataset.mol_ids[i])]

        global_rmsds = []

        for mol_indices in mol_id_chunks:
            mols =[ self.dataset.molecules[i] for i in mol_indices]
            positions = [m.get_positions() for m in mols]


            rmsds = []

            for pos_conf_A, pos_conf_B in combinations(positions,2):
                rmsds.append(rmsd(pos_conf_A, pos_conf_B))

            global_rmsds.append(rmsds)



        flattend_rmsds = list(chain.from_iterable(global_rmsds))
        rmsd_fig = plt.figure()

        plt.hist(flattend_rmsds)
        rmsd_fig.savefig(f"{self.output_dir}/rmsd_distribution.png")
