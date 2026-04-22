import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.data import chemical_symbols
from ase.visualize.plot import plot_atoms
from matplotlib.figure import Figure
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.evaluation.results import FigureResult


def has_stereocenter(iso_smi):
    # make sure stereochem is perceived from 2D/SMILES

    mol = Chem.MolFromSmiles(iso_smi)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    n_assigned = rdMD.CalcNumAtomStereoCenters(mol)
    return n_assigned > 0


class MoleculeDatasetAnalysis:
    def __init__(self, dataset: MoleculeDataset):
        self.dataset = dataset
        self.results: list[FigureResult] = []

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
            counts.update(
                dict(zip(unique.tolist(), chunk_counts.tolist(), strict=False))
            )
            offset = end

        return {int(n): int(c) for n, c in counts.items()}

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
        hetero_counts = np.add.reduceat(
            hetero_mask.astype(np.int64, copy=False), ptr[:-1]
        )
        return hetero_counts

    def get_fraction_molecules_with_stereocentres(self):
        dataset = self.dataset
        n_struct = dataset.N_structures

        if dataset.isomeric_smiles is None or n_struct == 0:
            return None

        molecules_with_stereo = 0
        for smi in dataset.isomeric_smiles:
            if has_stereocenter(smi):
                molecules_with_stereo += 1

        return molecules_with_stereo / n_struct

    def plot_heteroatom_distribution(self):
        hetero_counts = self.get_heteroatom_count_per_molecule_distribution()
        hetero_counts = np.asarray(hetero_counts, dtype=np.int64).ravel()

        fig, ax = plt.subplots(figsize=(12, 4))
        if hetero_counts.size > 0:
            unique_counts, frequencies = np.unique(hetero_counts, return_counts=True)
            ax.bar(
                unique_counts, frequencies, align="center", width=0.8, color="#55a868"
            )
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

    def plot_atom_species_histogram(self):
        atom_species = self.atom_species(chunk_len=100_000)

        if not atom_species:
            return

        # Sort species by descending frequency for readability
        sorted_species = sorted(
            atom_species.items(), key=lambda item: item[1], reverse=True
        )
        labels = [
            chemical_symbols[atomic_number] for atomic_number, _ in sorted_species
        ]
        counts = [count for _, count in sorted_species]

        fig, ax = plt.subplots()
        ax.bar(labels, counts)
        ax.set_xlabel("Atomic species")
        ax.set_ylabel("Frequency")
        ax.set_title("Atom species frequency")
        fig.tight_layout()

        self.results.append(
            FigureResult(figure=fig, file_name="atom_species_histogram.png")
        )

    def plot_molecule_size_distribution(self) -> Figure:
        sizes = self.molecule_sizes()
        fig = plt.figure()
        plt.hist(sizes, bins="auto")
        plt.xlabel("Atoms per structure")
        plt.ylabel("Frequency")

        self.results.append(
            FigureResult(figure=fig, file_name="molecule_size_distribution.png")
        )

    def plot_regression_task_distribution(self):
        pass

    def plot_relaxed_atoms(self, N_max_molecules: int | None = None):
        molecules = self.dataset.get_all_molecules(N_molecules=N_max_molecules)
        if not molecules:
            return

        N_horizontal = 3
        N_vertical = (len(molecules) + N_horizontal - 1) // N_horizontal

        fig, axarr = plt.subplots(N_vertical, N_horizontal, squeeze=False)
        fig.set_figheight(4 * N_vertical)
        fig.set_figwidth(4 * N_horizontal)

        for i, mol in enumerate(molecules):
            plot_atoms(mol, axarr[i // N_horizontal, i % N_horizontal])
        for j in range(len(molecules), N_vertical * N_horizontal):
            fig.delaxes(axarr[j // N_horizontal, j % N_horizontal])

        self.results.append(FigureResult(figure=fig, file_name="example_molecules.png"))

    def print_dataset_properties(self):
        n_molecules = self.dataset.N_molecules
        n_structures = self.dataset.N_structures
        n_atoms = self.dataset.N_atoms

        stereocentre_ratio = self.get_fraction_molecules_with_stereocentres()

        print(
            f"N_molecules={n_molecules}, N_structures={n_structures}, N_atoms={n_atoms}, stereocentre_ratio={stereocentre_ratio}"
        )

    def run(self):
        self.print_dataset_properties()
        self.plot_molecule_size_distribution()
        self.plot_atom_species_histogram()
        self.plot_relaxed_atoms(100)
        self.plot_heteroatom_distribution()

    def output(self, output_dir: Path):
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        for result in self.results:
            result.serialize_to(output_dir)
