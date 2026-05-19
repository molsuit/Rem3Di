import os
from pathlib import Path

import numpy as np
import pydantic_yaml as pyd_yaml
import zarr
from ase import Atoms
from zarr import Array
from zarr.codecs import BloscCodec, BloscShuffle
from zarr.storage import LocalStore

from threedscriptors.configuration.dataset_config import DatasetConfig
from threedscriptors.data_handling.dataset.smiles_storage import SmilesStorage


class MoleculeDataset:
    # On-disk storage of all data related to the molecular systems.

    def __init__(
        self,
        positions: Array,
        atomic_numbers: Array,
        ptr: Array,
        structure_ids: Array,
        molecule_ids: Array,
        isomer_ids: Array,
        total_charge: Array,
        multiplicity: Array,
        smiles: SmilesStorage | None,
        isomeric_smiles: SmilesStorage | None,
        targets_system: Array | None,
        mask_system: Array | None,
        targets_atom: Array | None,
        mask_atom: Array | None,
        config: DatasetConfig,
        root: Path,
    ):
        self.path = Path(root)
        self.positions = positions
        self.atomic_numbers = atomic_numbers
        self.ptr = ptr

        self.structure_ids = structure_ids
        self.molecule_ids = molecule_ids
        self.isomer_ids = isomer_ids

        self.total_charge = total_charge
        self.multiplicity = multiplicity

        self.smiles = smiles
        self.isomeric_smiles = isomeric_smiles

        self.targets_system = targets_system
        self.mask_system = mask_system
        self.targets_atom = targets_atom
        self.mask_atom = mask_atom

        self.config = config

    @classmethod
    def open_existing_dataset_from_dir(cls, path: Path):
        """Open an existing on-disk dataset created by create_empty_dataset."""
        path = Path(path)
        store = LocalStore(path)
        g = zarr.open_group(store=store, mode="r+")

        positions = g["positions"]
        atomic_numbers = g["atomic_numbers"]
        ptr = g["molecule_ptr"]
        total_charge = g["total_charge"]
        multiplicity = g["multiplicity"]

        ids = g["ids"]
        mol_id = ids["molecule_id"]
        stereo_id = ids["stereoisomer_id"]
        struct_id = ids["structure_id"]

        smiles = None
        isomeric_smiles = None
        smiles_text_path = os.path.join(path, "smiles.txt")
        smiles_index_path = os.path.join(path, "smiles_idx")
        isomeric_smiles_path = os.path.join(path, "isomeric_smiles.txt")
        isomeric_index_path = os.path.join(path, "isomeric_smiles_idx")

        if os.path.exists(smiles_text_path) and os.path.exists(smiles_index_path):
            smiles = SmilesStorage(
                text_path=smiles_text_path, index_path=smiles_index_path
            )
        if os.path.exists(isomeric_smiles_path) and os.path.exists(isomeric_index_path):
            isomeric_smiles = SmilesStorage(
                text_path=isomeric_smiles_path, index_path=isomeric_index_path
            )

        config = pyd_yaml.parse_yaml_file_as(
            DatasetConfig, file=path / "dataset_config.yaml"
        )

        tasks_grp = g["tasks"] if "tasks" in g else None

        if tasks_grp is not None:
            targets_system = (
                tasks_grp["targets_system"] if "targets_system" in tasks_grp else None
            )
            mask_system = (
                tasks_grp["mask_system"] if "mask_system" in tasks_grp else None
            )
            targets_atom = (
                tasks_grp["targets_atom"] if "targets_atom" in tasks_grp else None
            )
            mask_atom = tasks_grp["mask_atom"] if "mask_atom" in tasks_grp else None

        else:
            targets_system, targets_atom, mask_atom, mask_system = (
                None,
                None,
                None,
                None,
            )

        return cls(
            positions,
            atomic_numbers,
            ptr,
            struct_id,
            mol_id,
            stereo_id,
            total_charge,
            multiplicity,
            smiles=smiles,
            isomeric_smiles=isomeric_smiles,
            targets_system=targets_system,
            mask_system=mask_system,
            targets_atom=targets_atom,
            mask_atom=mask_atom,
            config=config,
            root=path,
        )

    def _structure_atom_span(self, i: int) -> tuple[int, int]:
        """Return [a0, a1) atom indices for structure i."""
        a0 = int(self.ptr[i])
        a1 = int(self.ptr[i + 1])
        return a0, a1

    @classmethod
    def create_empty_dataset(cls, path: Path, config: DatasetConfig):
        path = Path(path)
        os.makedirs(path, exist_ok=True)

        compressor = BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)

        store = LocalStore(path)
        g = zarr.create_group(store=store, overwrite=True)

        a_cps = config.atom_chunks_per_shard
        m_cps = config.molecule_chunks_per_shard

        def _mk(grp, name, shape, chunk0, cps, dtype, axis1: int | None = None):
            """Create a sharded array.

            ``chunk0`` is the chunk size along the growable (first) axis; the
            shard groups ``cps`` such chunks into one on-disk file. zarr
            requires the shard shape to be a whole multiple of the chunk
            shape, which holds by construction here (cps on axis 0, 1x on the
            fixed axis 1).
            """
            if axis1 is None:
                chunks, shards = (chunk0,), (chunk0 * cps,)
            else:
                chunks, shards = (chunk0, axis1), (chunk0 * cps, axis1)
            return grp.create_array(
                name=name,
                shape=shape,
                chunks=chunks,
                shards=shards,
                dtype=dtype,
                compressors=compressor,
            )

        positions = _mk(g, "positions", (0, 3), config.atom_chunk, a_cps, "f4", 3)
        atomic_numbers = _mk(
            g, "atomic_numbers", (0,), config.atom_chunk, a_cps, "u1"
        )

        # ragged pointer
        ptr = _mk(g, "molecule_ptr", (1,), config.molecule_chunk, m_cps, "i8")
        ptr[:] = 0

        # per-structure scalars
        total_charge = _mk(
            g, "total_charge", (0,), config.molecule_chunk, m_cps, "f4"
        )
        multiplicity = _mk(
            g, "multiplicity", (0,), config.molecule_chunk, m_cps, "f4"
        )

        # ID section
        ids = g.require_group("ids")
        mol_id = _mk(ids, "molecule_id", (0,), config.molecule_chunk, m_cps, "i8")
        stereo_id = _mk(
            ids, "stereoisomer_id", (0,), config.molecule_chunk, m_cps, "i8"
        )
        struct_id = _mk(
            ids, "structure_id", (0,), config.molecule_chunk, m_cps, "i8"
        )

        smiles = None
        isomeric_smiles = None
        if config.contains_smiles:
            smiles_text_path = os.path.join(path, "smiles.txt")
            smiles_index_path = os.path.join(path, "smiles_idx")
            isomeric_smiles_path = os.path.join(path, "isomeric_smiles.txt")
            isomeric_index_path = os.path.join(path, "isomeric_smiles_idx")

            SmilesStorage.create_files(smiles_text_path, smiles_index_path)
            SmilesStorage.create_files(isomeric_smiles_path, isomeric_index_path)

            smiles = SmilesStorage(
                text_path=smiles_text_path, index_path=smiles_index_path
            )
            isomeric_smiles = SmilesStorage(
                text_path=isomeric_smiles_path, index_path=isomeric_index_path
            )

        targets_system = None
        mask_system = None
        targets_atom = None
        mask_atom = None

        if config.tasks is not None:
            tasks_grp = g.require_group("tasks")

            Nsys = len(config.tasks.system_cols)
            Natom = len(config.tasks.atom_cols)

            if Nsys > 0:
                targets_system = _mk(
                    tasks_grp, "targets_system", (0, Nsys),
                    config.molecule_chunk, m_cps, "f4", max(1, Nsys),
                )
                mask_system = _mk(
                    tasks_grp, "mask_system", (0, Nsys),
                    config.molecule_chunk, m_cps, "u1", max(1, Nsys),
                )

            if Natom > 0:
                targets_atom = _mk(
                    tasks_grp, "targets_atom", (0, Natom),
                    config.atom_chunk, a_cps, "f4", max(1, Natom),
                )
                mask_atom = _mk(
                    tasks_grp, "mask_atom", (0, Natom),
                    config.atom_chunk, a_cps, "u1", max(1, Natom),
                )

        pyd_yaml.to_yaml_file(path / "dataset_config.yaml", config)

        return cls(
            positions,
            atomic_numbers,
            ptr,
            struct_id,
            mol_id,
            stereo_id,
            total_charge,
            multiplicity,
            smiles,
            isomeric_smiles,
            targets_system,
            mask_system,
            targets_atom,
            mask_atom,
            config,
            root=path,
        )

    @property
    def N_structures(self) -> int:
        # ptr has length N_structures + 1 (leading 0 sentinel). Valid once
        # the dataset has been finalized (ShardAlignedWriter.finalize).
        return int(self.ptr.shape[0]) - 1

    @property
    def N_atoms(self) -> int:
        # ptr is cumulative; the last entry is the total atom count.
        return int(np.asarray(self.ptr[self.ptr.shape[0] - 1]))

    @property
    def N_molecules(self) -> int:
        n = int(self.molecule_ids.shape[0])
        if n == 0:
            return 0
        return int(np.asarray(self.molecule_ids[:]).max()) + 1

    def __len__(self):
        return self.N_structures

    def get_structure_ids_from_molecule_ids(self, molecule_ids: set):
        if not molecule_ids:
            return np.asarray([], dtype="i8")

        mol_ids = np.asarray(self.molecule_ids[:])
        mask = np.isin(mol_ids, list(molecule_ids))
        struct_ids = np.asarray(self.structure_ids[:])
        retrieved_structure_ids = struct_ids[mask]
        return retrieved_structure_ids

    def get_smiles_per_structure(self):
        if self.smiles is None:
            return []

        n_struct = self.N_structures
        if n_struct == 0:
            return []

        mol_ids = np.asarray(self.molecule_ids[:n_struct], dtype=np.int64)
        smiles_array = np.asarray(self.isomeric_smiles.to_list(), dtype=object)
        return smiles_array[mol_ids].tolist()

    def get_all_molecules(self, N_molecules: int | None = None) -> list[Atoms]:
        n_struct = self.N_structures
        if n_struct == 0:
            return []

        if N_molecules is not None:
            if N_molecules <= 0:
                return []
            n_struct = min(n_struct, N_molecules)
            if n_struct == 0:
                return []

        ptr = np.asarray(self.ptr[: n_struct + 1], dtype=np.int64, order="C")
        atomic_numbers = np.asarray(
            self.atomic_numbers[: ptr[-1]], dtype=np.int64, order="C"
        )
        positions = np.asarray(self.positions[: ptr[-1]], dtype=np.float32, order="C")
        total_charge = np.asarray(self.total_charge[:n_struct])
        multiplicity = np.asarray(self.multiplicity[:n_struct])

        molecules: list[Atoms] = []
        append = molecules.append
        for idx in range(n_struct):
            start = ptr[idx]
            end = ptr[idx + 1]
            append(
                Atoms(
                    numbers=atomic_numbers[start:end],
                    positions=positions[start:end],
                    info={
                        "total_charge": float(total_charge[idx]),
                        "multiplicity": float(multiplicity[idx]),
                    },
                )
            )

        return molecules
