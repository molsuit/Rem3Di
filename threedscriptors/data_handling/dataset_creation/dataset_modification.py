from __future__ import annotations

from pathlib import Path

import numpy as np

from threedscriptors.configuration.dataset_config import DatasetConfig
from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset.smiles_storage import SmilesStorage


class DatasetReconfigurator:
    """
    Rewrites an existing `MoleculeDataset` into a new directory using an updated
    `DatasetConfig`. All stored arrays (atoms, structures, optional tasks) are copied
    so that downstream pipelines keep working with the new configuration.
    """

    def __init__(
        self,
        source_dir: Path,
        target_dir: Path,
        new_config: DatasetConfig,
        structures_per_chunk: int = 50_000,
        smiles_batch: int = 50_000,
        structure_limit: int | None = None,
    ) -> None:
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.new_config = new_config
        self.structures_per_chunk = structures_per_chunk
        self.smiles_batch = smiles_batch
        if structure_limit is not None and structure_limit < 0:
            raise ValueError("structure_limit must be non-negative.")
        self.structure_limit = structure_limit

    def run(self) -> MoleculeDataset:
        src = MoleculeDataset.open_existing_dataset_from_dir(self.source_dir)
        self._validate_compatibility(src)

        dst = MoleculeDataset.create_empty_dataset(self.target_dir, self.new_config)

        mol_lut: np.ndarray | None = None
        iso_lut: np.ndarray | None = None
        if self.new_config.contains_smiles:
            mol_lut = self._build_smiles_mapping(src.smiles, dst.smiles)
            iso_lut = self._build_smiles_mapping(
                src.isomeric_smiles, dst.isomeric_smiles
            )

        self._copy_contents(src, dst, mol_lut, iso_lut)

        if dst.smiles is not None:
            dst.smiles.close()
        if dst.isomeric_smiles is not None:
            dst.isomeric_smiles.close()

        dst.shrink_to_fit()
        return dst

    def _validate_compatibility(self, src: MoleculeDataset) -> None:
        src_cfg = src.config
        if src_cfg.contains_smiles != self.new_config.contains_smiles:
            raise ValueError(
                "SMILES configuration mismatch between source and requested config."
            )
        if self.new_config.contains_smiles and (
            src.smiles is None or src.isomeric_smiles is None
        ):
            raise ValueError("Source dataset is missing SMILES storage.")

        src_sys_cols = (
            0 if src.targets_system is None else int(src.targets_system.shape[1])
        )
        src_atom_cols = (
            0 if src.targets_atom is None else int(src.targets_atom.shape[1])
        )
        new_sys_cols = (
            0
            if self.new_config.tasks is None
            else len(self.new_config.tasks.system_cols)
        )
        new_atom_cols = (
            0 if self.new_config.tasks is None else len(self.new_config.tasks.atom_cols)
        )
        if src_sys_cols != new_sys_cols or src_atom_cols != new_atom_cols:
            raise ValueError(
                "Task dimensions differ between source dataset and new configuration."
            )

    def _copy_contents(
        self,
        src: MoleculeDataset,
        dst: MoleculeDataset,
        mol_lut: np.ndarray | None,
        iso_lut: np.ndarray | None,
    ) -> None:
        src_ptr = np.asarray(src.ptr[:], dtype=np.int64)
        total_structs = src_ptr.shape[0] - 1
        if self.structure_limit is not None:
            n_structs = min(total_structs, self.structure_limit)
            src_ptr = src_ptr[: n_structs + 1]
        else:
            n_structs = total_structs

        if n_structs <= 0:
            return

        for s0 in range(0, n_structs, self.structures_per_chunk):
            s1 = min(s0 + self.structures_per_chunk, n_structs)

            a0 = int(src_ptr[s0])
            a1 = int(src_ptr[s1])

            if a1 == a0 and s1 > s0:
                continue

            positions = np.asarray(src.positions[a0:a1, :])
            atomic_numbers = np.asarray(src.atomic_numbers[a0:a1])
            total_charge = np.asarray(src.total_charge[s0:s1])
            total_spin = np.asarray(src.total_spin[s0:s1])

            chunk_ptr = src_ptr[s0 : s1 + 1]
            lens = np.diff(chunk_ptr)
            cum_atoms = lens.cumsum()

            mol_ids = np.asarray(src.molecule_ids[s0:s1], dtype=np.int64)
            iso_ids = np.asarray(src.isomer_ids[s0:s1], dtype=np.int64)

            if mol_lut is not None and iso_lut is not None:
                new_mol_ids = np.take(mol_lut, mol_ids)
                new_iso_ids = np.take(iso_lut, iso_ids)
            else:
                new_mol_ids = mol_ids
                new_iso_ids = iso_ids

            system_targets = (
                None
                if src.targets_system is None
                else np.asarray(src.targets_system[s0:s1])
            )
            system_masks = (
                None if src.mask_system is None else np.asarray(src.mask_system[s0:s1])
            )

            atom_targets = (
                None
                if src.targets_atom is None
                else np.asarray(src.targets_atom[a0:a1])
            )
            atom_masks = (
                None if src.mask_atom is None else np.asarray(src.mask_atom[a0:a1])
            )

            dst.append_batch(
                positions,
                atomic_numbers,
                cum_atoms,
                new_mol_ids,
                new_iso_ids,
                total_charge,
                total_spin,
                system_targets,
                system_masks,
                atom_targets,
                atom_masks,
            )

    def _build_smiles_mapping(
        self, storage: SmilesStorage | None, target_storage: SmilesStorage | None
    ) -> np.ndarray | None:
        if storage is None or target_storage is None:
            raise ValueError("SMILES storage missing during mapping.")

        total = len(storage)
        mapping = np.empty(total, dtype=np.int64)

        for start in range(0, total, self.smiles_batch):
            end = min(start + self.smiles_batch, total)
            strings = [storage.id_to_string(i) for i in range(start, end)]
            new_ids = target_storage.append_new_lines(strings)
            mapping[start:end] = [new_ids[s] for s in strings]

        return mapping
