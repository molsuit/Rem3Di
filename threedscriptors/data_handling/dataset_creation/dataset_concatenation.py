from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from pathlib import Path
from typing import Sequence
import numpy as np


class DatasetConcatenation():
    def __init__(self, datasets : Sequence[MoleculeDataset], new_dataset_dir: Path):


        self.datasets = datasets
        self.new_dataset_dir = Path(new_dataset_dir)


    def concatenate_datasets(self):
        # Concatenate multiple MoleculeDatasets into a new one at new_dataset_dir.
        # Arrays are backed by zarr; we stream per-dataset to keep it fast.

        if len(self.datasets) == 0:
            raise ValueError("No datasets provided for concatenation.")

        # Validate config compatibility across datasets; use the first as template
        ref_cfg = self.datasets[0].config
        for ds in self.datasets[1:]:
            cfg = ds.config
            if (
                cfg.embedding_dim != ref_cfg.embedding_dim
                or cfg.contains_smiles != ref_cfg.contains_smiles
                or cfg.irreps != ref_cfg.irreps
            ):
                raise ValueError("Incompatible dataset configs; cannot concatenate.")

        # Create the output dataset
        out_ds = MoleculeDataset.create_empty_dataset(self.new_dataset_dir, ref_cfg)

        # For non-smiles case, maintain running id offsets to avoid collisions
        next_mol_base = 0
        next_iso_base = 0

        for src in self.datasets:
            # Load per-atom arrays
            E = src.atomic_embeddings[:]
            P = src.positions[:]
            Z = src.atomic_numbers[:]

            # Build cumulative ends per structure from ptr
            src_ptr = src.ptr[:]
            if src_ptr.shape[0] <= 1:
                # Empty dataset; skip
                continue
            lengths = (src_ptr[1:] - src_ptr[:-1]).astype("i8", copy=False)
            C = lengths.cumsum(dtype="i8")

            # Remap molecule and isomer ids appropriately
            if ref_cfg.contains_smiles:
                # Simple remap: ids -> strings -> insert -> ids
                old_mol_ids = src.molecule_ids[:]
                old_iso_ids = src.isomer_ids[:]

                mol_strings = [src.smiles.id_to_string(int(i)) for i in old_mol_ids]
                iso_strings = [src.isomeric_smiles.id_to_string(int(i)) for i in old_iso_ids]

                mol_map = out_ds.smiles.append_new_lines(mol_strings)
                iso_map = out_ds.isomeric_smiles.append_new_lines(iso_strings)

                new_mol_ids = np.asarray([mol_map[s] for s in mol_strings], dtype="i8")
                new_iso_ids = np.asarray([iso_map[s] for s in iso_strings], dtype="i8")
            else:

                old_mol_ids = src.molecule_ids[:].astype("i8", copy=False)
                old_iso_ids = src.isomer_ids[:].astype("i8", copy=False)

                new_mol_ids = old_mol_ids + next_mol_base
                new_iso_ids = old_iso_ids + next_iso_base

                # Advance bases to keep ids distinct across datasets
                if old_mol_ids.size > 0:
                    next_mol_base += int(old_mol_ids.max()) + 1
                if old_iso_ids.size > 0:
                    next_iso_base += int(old_iso_ids.max()) + 1

            # Append to the target dataset
            out_ds.append_batch(E, P, Z, C, new_mol_ids, new_iso_ids)

        # Finalize storage files
        if ref_cfg.contains_smiles:
            out_ds.smiles.close()
            out_ds.isomeric_smiles.close()

        out_ds.shrink_to_fit()

        return out_ds
