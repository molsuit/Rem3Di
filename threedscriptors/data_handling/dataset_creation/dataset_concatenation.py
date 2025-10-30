import numpy as np
from pathlib import Path
from typing import Sequence

from threedscriptors.configuration.dataset_config import DatasetConfig
from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset.tasks import TaskConfig, TaskSet, TaskType

class DatasetConcatenation():
    def __init__(self, datasets : Sequence[MoleculeDataset], new_dataset_dir: Path):

        self.datasets = datasets
        self.new_dataset_dir = Path(new_dataset_dir)


    def concatenate_datasets(self):
        # Concatenate multiple MoleculeDatasets into a new one at new_dataset_dir.
        # Arrays are backed by zarr; we stream per-dataset to keep it fast.

        out_ds, ref_cfg = self._create_target_dataset()

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

    def concatenate_datasets_chunked(
        self, structures_per_chunk: int = 50_000, smiles_batch: int = 50_000
    ) -> MoleculeDataset:
        """
        Concatenate datasets while streaming atoms/molecules in chunks. Suitable for
        multi-million molecule datasets without loading everything into memory.
        """

        out_ds, ref_cfg = self._create_target_dataset()

        next_mol_base = 0
        next_iso_base = 0

        for src in self.datasets:
            src_ptr = np.asarray(src.ptr[:], dtype=np.int64)
            n_structs = src_ptr.shape[0] - 1
            if n_structs <= 0:
                continue

            if ref_cfg.contains_smiles:
                mol_lut = self._build_smiles_mapping(
                    src.smiles, out_ds.smiles, smiles_batch
                )
                iso_lut = self._build_smiles_mapping(
                    src.isomeric_smiles, out_ds.isomeric_smiles, smiles_batch
                )
            else:
                mol_lut = None
                iso_lut = None
                mol_max = -1
                iso_max = -1

            for s0 in range(0, n_structs, structures_per_chunk):
                s1 = min(s0 + structures_per_chunk, n_structs)

                a0 = int(src_ptr[s0])
                a1 = int(src_ptr[s1])

                if a1 == a0 and s1 > s0:
                    # All structures empty in this chunk; skip append to avoid zero-atom writes
                    continue

                E = np.asarray(src.atomic_embeddings[a0:a1, :])
                P = np.asarray(src.positions[a0:a1, :])
                Z = np.asarray(src.atomic_numbers[a0:a1])

                # ptr cumulative ends for chunk
                chunk_ptr = src_ptr[s0 : s1 + 1]
                lens = np.diff(chunk_ptr, dtype=np.int64)
                C = lens.cumsum(dtype=np.int64)

                mol_ids_chunk = np.asarray(src.molecule_ids[s0:s1], dtype=np.int64)
                iso_ids_chunk = np.asarray(src.isomer_ids[s0:s1], dtype=np.int64)

                if ref_cfg.contains_smiles:
                    new_mol_ids = np.take(mol_lut, mol_ids_chunk)
                    new_iso_ids = np.take(iso_lut, iso_ids_chunk)
                else:
                    new_mol_ids = mol_ids_chunk + next_mol_base
                    new_iso_ids = iso_ids_chunk + next_iso_base

                    if mol_ids_chunk.size > 0:
                        mol_max = max(mol_max, int(mol_ids_chunk.max()))
                    if iso_ids_chunk.size > 0:
                        iso_max = max(iso_max, int(iso_ids_chunk.max()))

                out_ds.append_batch(E, P, Z, C, new_mol_ids, new_iso_ids)

            if not ref_cfg.contains_smiles:
                if mol_max >= 0:
                    next_mol_base += mol_max + 1
                if iso_max >= 0:
                    next_iso_base += iso_max + 1

        if ref_cfg.contains_smiles:
            out_ds.smiles.close()
            out_ds.isomeric_smiles.close()

        out_ds.shrink_to_fit()
        return out_ds

    def _create_target_dataset(self) -> tuple[MoleculeDataset, DatasetConfig]:
        
        assert self._check_dataset_compatible()
        
        ref_cfg = self.datasets[0].config

        out_ds = MoleculeDataset.create_empty_dataset(self.new_dataset_dir, ref_cfg)
        return out_ds, ref_cfg
    

    def _check_dataset_compatible(self):

        if len(self.datasets) == 0:
            return False

        ref_cfg = self.datasets[0].config
        for ds in self.datasets[1:]:
            cfg = ds.config
            if (
                cfg.embedding_dim != ref_cfg.embedding_dim
                or cfg.contains_smiles != ref_cfg.contains_smiles
                or cfg.irreps != ref_cfg.irreps
            ):
                return False
        return True

    @staticmethod
    def _build_smiles_mapping(storage, target_storage, batch_size: int) -> np.ndarray:
        from threedscriptors.data_handling.dataset.smiles_storage import SmilesStorage

        if storage is None or target_storage is None:
            raise ValueError("Smiles storage missing; configuration mismatch.")

        if not isinstance(storage, SmilesStorage) or not isinstance(
            target_storage, SmilesStorage
        ):
            raise TypeError("Unexpected storage type for SMILES mapping.")

        total = len(storage)
        mapping = np.empty(total, dtype=np.int64)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            strings = [storage.id_to_string(i) for i in range(start, end)]
            new_ids = target_storage.append_new_lines(strings)
            mapping[start:end] = [new_ids[s] for s in strings]

        return mapping


class LabeldDatasetConcatenation(DatasetConcatenation):
    """
    This datasets concatenation is for labeld datasets, and concatenates the system labels and masks, as well as all the embeddings and positions
    """

    
    def __init__(self, datasets : Sequence[MoleculeDataset], new_dataset_dir: Path):

        self.datasets = datasets
        self.new_dataset_dir = Path(new_dataset_dir)

    
    def _create_target_dataset(self) -> tuple[MoleculeDataset, DatasetConfig]:

        assert self._check_dataset_compatible()

        system_tasks: list[TaskConfig] = []
        atom_tasks: list[TaskConfig] = []
        system_index: dict[str, TaskConfig] = {}
        atom_index: dict[str, TaskConfig] = {}

        for dataset in self.datasets:
            cfg = dataset.config
            if cfg.tasks is None:
                continue

            for task in cfg.tasks.system_cols:
                system_tasks.append(task)

            for task in cfg.tasks.atom_cols:
                atom_tasks.append(task)


        combined_tasks = TaskSet(
            system_cols=system_tasks,
            atom_cols=atom_tasks,
        ).finalize()

        ref_cfg = self.datasets[0].config
        new_config = ref_cfg.model_copy(deep=True)
        new_config.tasks = combined_tasks

        out_ds = MoleculeDataset.create_empty_dataset(self.new_dataset_dir, new_config)

        return out_ds, new_config

    def concatenate_datasets(self):

        
        out_ds, ref_cfg = self._create_target_dataset()

        N_total_systems_tasks = len(ref_cfg.tasks.system_cols)

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

            # Get all the prediction targets
            # Build mapping for old dataset column to new dataset_column
            column_map_systems = {
                src_col_idx: ref_cfg.tasks.system_map[src_task_name]
                for src_task_name, src_col_idx in src.config.tasks.system_map.items()
            }

            n_structures = C.shape[0]
            if N_total_systems_tasks > 0:
                system_targets = np.zeros(
                    (n_structures, N_total_systems_tasks), dtype="f4", order="C"
                )
                system_masks = np.zeros(
                    (n_structures, N_total_systems_tasks), dtype="u1", order="C"
                )

                if (
                    column_map_systems
                    and src.targets_system is not None
                    and src.mask_system is not None
                ):
                    src_indices = np.fromiter(
                        column_map_systems.keys(),
                        dtype=np.int64,
                        count=len(column_map_systems),
                    )
                    dst_indices = np.fromiter(
                        column_map_systems.values(),
                        dtype=np.int64,
                        count=len(column_map_systems),
                    )

                    src_targets = np.asarray(
                        src.targets_system[:n_structures], dtype="f4", order="C"
                    )
                    src_masks = np.asarray(
                        src.mask_system[:n_structures], dtype="u1", order="C"
                    )

                    system_targets[:, dst_indices] = src_targets[:, src_indices]
                    system_masks[:, dst_indices] = src_masks[:, src_indices]
            else:
                system_targets = None
                system_masks = None

            # Create the correct masking for all other targets

            out_ds.append_batch(
                E,
                P,
                Z,
                C,
                new_mol_ids,
                new_iso_ids,
                system_targets,
                system_masks,
                None,
                None,
            )
