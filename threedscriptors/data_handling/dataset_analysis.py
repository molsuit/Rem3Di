import os
from collections import Counter
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

from threedscriptors.evaluation.results import FigureResult
from rdkit.Chem import rdMolDescriptors as rdMD

from rdkit import Chem  



def has_stereocenter(iso_smi):
    # make sure stereochem is perceived from 2D/SMILES
    
    mol = Chem.MolFromSmiles(iso_smi)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    n_assigned = rdMD.CalcNumAtomStereoCenters(mol)
    return n_assigned > 0


class MoleculeDatasetAnalysis:

    def __init__(self, dataset: MoleculeDataset):

        self.dataset = dataset
        self.results :list[FigureResult] = []

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
            # include sentinel at end for diff
            ptr_chunk = np.asarray(ptr[offset : end + 1], dtype=np.int64)
            sizes[offset:end] = np.diff(ptr_chunk)
            offset = end

        return sizes

    def atom_species(self, chunk_len: int | None = None) -> np.ndarray:

        n_atoms = self.dataset.N_atoms
        if n_atoms == 0:
            return {}

        atomic_numbers = self.dataset.atomic_numbers
        if chunk_len is None:
            chunk_len = getattr(atomic_numbers, "chunks", (n_atoms,))[0]
        chunk_len = max(1, min(chunk_len, n_atoms))

        counts: Counter[int] = Counter()
        offset = 0
        while offset < n_atoms:
            end = min(offset + chunk_len, n_atoms)
            chunk = np.asarray(atomic_numbers[offset:end], dtype=np.int64)
            if chunk.size == 0:
                offset = end
                continue
            unique, chunk_counts = np.unique(chunk, return_counts=True)
            counts.update(dict(zip(unique.tolist(), chunk_counts.tolist(), strict=False)))
            offset = end

        return {int(n): int(c) for n, c in counts.items()}


    def get_descriptor_mean_std(self):
        embeddings = self.dataset.atomic_embeddings
        n_atoms = self.dataset.N_atoms
        embedding_dim = embeddings.shape[-1]
        if n_atoms == 0 or embedding_dim == 0:
            nan_array = np.full((embedding_dim,), np.nan, dtype=np.float64)
            print(nan_array)
            return nan_array, nan_array

        chunk_len = getattr(embeddings, "chunks", (n_atoms, embedding_dim))[0]
        chunk_len = max(1, min(chunk_len, n_atoms))

        count = 0
        mean = np.zeros(embedding_dim, dtype=np.float64)
        m2 = np.zeros(embedding_dim, dtype=np.float64)

        offset = 0
        while offset < n_atoms:
            end = min(offset + chunk_len, n_atoms)
            chunk = np.asarray(embeddings[offset:end], dtype=np.float64)
            chunk_count = chunk.shape[0]
            if chunk_count == 0:
                offset = end
                continue

            chunk_mean = chunk.mean(axis=0)
            chunk_var = chunk.var(axis=0, ddof=0)

            if count == 0:
                mean = chunk_mean
                m2 = chunk_var * chunk_count
                count = chunk_count
            else:
                total_count = count + chunk_count
                delta = chunk_mean - mean
                mean = mean + delta * (chunk_count / total_count)
                m2 = (
                    m2
                    + chunk_var * chunk_count
                    + (delta**2) * count * chunk_count / total_count
                )
                count = total_count
            offset = end

        variance = m2 / count if count > 0 else np.full_like(mean, np.nan)
        std = np.sqrt(variance)
        return mean, std

    def get_heteroatom_count_per_molecule_distribution(self):
        n_struct = self.dataset.N_structures
        if n_struct == 0:
            return np.asarray([], dtype=np.int64)

        ptr = np.asarray(self.dataset.ptr[: n_struct + 1], dtype=np.int64)
        total_atoms = int(ptr[-1]) if ptr.size else 0
        if total_atoms == 0:
            return np.zeros(n_struct, dtype=np.int64)

        atomic_numbers = np.asarray(self.dataset.atomic_numbers[:total_atoms])
        hetero_mask = (atomic_numbers != 1) & (atomic_numbers != 6)
        hetero_counts = np.add.reduceat(hetero_mask.astype(np.int64, copy=False), ptr[:-1])
        return hetero_counts
    

    def get_fraction_molecules_with_stereocentres(self):
        dataset = self.dataset
        n_struct = dataset.N_structures
        
        molecules_with_stereo = 0
        for smi in dataset.isomeric_smiles:
            if has_stereocenter(smi):

                molecules_with_stereo += 1

        return molecules_with_stereo / n_struct

    def get_descriptor_norm(self):
        embeddings = self.dataset.atomic_embeddings
        n_atoms = self.dataset.N_atoms
        if n_atoms == 0:
            return np.asarray([], dtype=np.float32)

        chunk_len = getattr(embeddings, "chunks", (n_atoms, embeddings.shape[-1]))[0]
        chunk_len = max(1, min(chunk_len, n_atoms))

        norms = np.empty(n_atoms, dtype=np.float32)
        offset = 0
        while offset < n_atoms:
            end = min(offset + chunk_len, n_atoms)
            chunk = np.asarray(embeddings[offset:end], dtype=np.float64)
            if chunk.size == 0:
                offset = end
                continue
            norms[offset:end] = np.linalg.norm(chunk, axis=-1, ord=2)
            offset = end

        return norms

    def plot_heteroatom_distribution(self):

        hetero_counts = self.get_heteroatom_count_per_molecule_distribution()
        hetero_counts = np.asarray(hetero_counts, dtype=np.int64).ravel()

        fig, ax = plt.subplots(figsize=(12, 4))
        if hetero_counts.size > 0:
            unique_counts, frequencies = np.unique(hetero_counts, return_counts=True)
            ax.bar(unique_counts, frequencies, align="center", width=0.8, color="#55a868")
            ax.set_xlabel("Heteroatoms per molecule")
            ax.set_ylabel("Number of molecules")
            ax.set_title("Distribution of heteroatoms per molecule")
            ax.set_xticks(unique_counts)
            ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        else:
            ax.text(0.5, 0.5, "No molecules available", ha="center", va="center")
            ax.axis("off")

        fig.tight_layout()
        result = FigureResult(figure=fig, file_name="heteroatom_distribution.png")
        self.results.append(result)
        return result

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

        atom_species = self.atom_species(chunk_len=100_000)

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

        self.results.append(FigureResult(figure=fig, file_name="example_molecules.png"))


    def print_dataset_properties(self):
        n_molecules = self.dataset.N_molecules
        n_structures = self.dataset.N_structures
        n_atoms = self.dataset.N_atoms

        stereocentre_ratio = self.get_fraction_molecules_with_stereocentres()

        print(f"N_molecules={n_molecules}, N_structures={n_structures}, N_atoms={n_atoms}, stereocentre_ratio={stereocentre_ratio}")


    def run(self):

        self.print_dataset_properties()
        self.plot_molecule_size_distribution()
        print(1)
        self.plot_atom_species_histogram()
        print(2)
        #self.plot_descriptor_norm_distribution()
        #print(3)
        #self.plot_descriptor_mean_std_distribution()

        self.plot_relaxed_atoms(100)
        self.plot_heteroatom_distribution()

    def output(self, output_dir: Path):

        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        for result in self.results:
            result.serialize_to(output_dir)
