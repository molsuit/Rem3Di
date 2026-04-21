import os
from pathlib import Path

import numpy as np
import pydantic_yaml as pyd_yaml
import zarr
from ase import Atoms
from numcodecs import Blosc
from zarr import Array
from zarr.convenience import consolidate_metadata

from threedscriptors.configuration.dataset_config import DatasetConfig
from threedscriptors.data_handling.dataset.smiles_storage import SmilesStorage


class MoleculeDataset:
    # This is the ondisk storage of all data related to the molecular systems we store

    def __init__(
        self,
        atomic_embeddings: Array,
        positions: Array,
        atomic_numbers: Array,
        ptr: Array,
        structure_ids: Array,
        molecule_ids: Array,
        isomer_ids: Array,
        smiles: SmilesStorage | None,
        isomeric_smiles: SmilesStorage | None,
        targets_system: Array | None,
        mask_system: Array | None,
        targets_atom: Array | None,
        mask_atom: Array | None,
        config: DatasetConfig,
    ):
        # creates all required fields in the zarr dataset

        self.atomic_embeddings = atomic_embeddings
        self.positions = positions
        self.atomic_numbers = atomic_numbers
        self.ptr = ptr

        self.structure_ids = structure_ids
        self.molecule_ids = molecule_ids
        self.isomer_ids = isomer_ids

        self.smiles = smiles
        self.isomeric_smiles = isomeric_smiles

        self.targets_system = targets_system
        self.mask_system = mask_system
        self.targets_atom = targets_atom
        self.mask_atom = mask_atom

        self.config = config

        self._mol_cursor = int(self.ptr.shape[0]) - 1
        self._atom_cursor = int(np.asarray(self.ptr[self._mol_cursor]))

    @classmethod
    def open_existing_dataset_from_dir(cls, path: Path):
        """
        Open an existing on-disk dataset created by create_empty_dataset.
        """
        store = zarr.DirectoryStore(str(path))
        g = zarr.open_group(store=store, mode="r+")

        if os.path.exists(path / "atomic_embeddings"):
            atomic_embeddings = g["atomic_embeddings"]
        else:
            atomic_embeddings = None

        positions = g["positions"]
        atomic_numbers = g["atomic_numbers"]
        ptr = g["molecule_ptr"]

        ids = g.require_group("ids")
        mol_id = ids["molecule_id"]
        stereo_id = ids["stereoisomer_id"]
        struct_id = ids["structure_id"]

        # Optional smiles stores
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
            atomic_embeddings,
            positions,
            atomic_numbers,
            ptr,
            struct_id,
            mol_id,
            stereo_id,
            smiles=smiles,
            isomeric_smiles=isomeric_smiles,
            targets_system=targets_system,
            mask_system=mask_system,
            targets_atom=targets_atom,
            mask_atom=mask_atom,
            config=config,
        )

    def _structure_atom_span(self, i: int) -> tuple[int, int]:
        """Return [a0, a1) atom indices for structure i."""
        a0 = int(self.ptr[i])
        a1 = int(self.ptr[i + 1])
        return a0, a1

    @classmethod
    def create_empty_dataset(cls, path: Path, config: DatasetConfig):
        # Configure compressor
        os.makedirs(path, exist_ok=True)

        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

        store = zarr.DirectoryStore(path)
        g = zarr.group(store=store, overwrite=True)

        # per-atom
        if config.contains_embeddings:
            atomic_embeddings = g.create(
                "atomic_embeddings",
                shape=(0, config.embedding_dim),
                chunks=(config.atom_chunk, config.embedding_dim),
                dtype="f4",
                compressor=compressor,
            )
        else:
            atomic_embeddings = None

        positions = g.create(
            "positions",
            shape=(0, 3),
            chunks=(config.atom_chunk, 3),
            dtype="f4",
            compressor=compressor,
        )
        atomic_numbers = g.create(
            "atomic_numbers",
            shape=(0,),
            chunks=(config.atom_chunk,),
            dtype="u1",
            compressor=compressor,
        )

        # ragged pointer
        ptr = g.create(
            "molecule_ptr",
            shape=(1,),
            chunks=(config.molecule_chunk,),
            dtype="i8",
            compressor=compressor,
        )
        ptr[:] = 0

        # ID section
        ids = g.require_group("ids")
        mol_id = ids.create(
            "molecule_id",
            shape=(0,),
            chunks=(config.molecule_chunk,),
            dtype="i8",
            compressor=compressor,
        )
        stereo_id = ids.create(
            "stereoisomer_id",
            shape=(0,),
            chunks=(config.molecule_chunk,),
            dtype="i8",
            compressor=compressor,
        )

        struct_id = ids.create(
            "structure_id",
            shape=(0,),
            chunks=(config.molecule_chunk,),
            dtype="i8",
            compressor=compressor,
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
        else:
            smiles = None
            isomeric_smiles = None

        targets_system = None
        mask_system = None
        targets_atom = None
        mask_atom = None

        if config.tasks is not None:
            tasks_grp = g.require_group("tasks")

            Nsys = len(config.tasks.system_cols)
            Natom = len(config.tasks.atom_cols)

            if Nsys > 0:
                targets_system = tasks_grp.create(
                    "targets_system",
                    shape=(0, Nsys),
                    chunks=(config.molecule_chunk, max(1, Nsys)),
                    dtype="f4",
                    compressor=compressor,
                )
                mask_system = tasks_grp.create(
                    "mask_system",
                    shape=(0, Nsys),
                    chunks=(config.molecule_chunk, max(1, Nsys)),
                    dtype="u1",
                    compressor=compressor,
                )

            if Natom > 0:
                targets_atom = tasks_grp.create(
                    "targets_atom",
                    shape=(0, Natom),
                    chunks=(config.atom_chunk, max(1, Natom)),
                    dtype="f4",
                    compressor=compressor,
                )
                mask_atom = tasks_grp.create(
                    "mask_atom",
                    shape=(0, Natom),
                    chunks=(config.atom_chunk, max(1, Natom)),
                    dtype="u1",
                    compressor=compressor,
                )

        # Write config to disk
        pyd_yaml.to_yaml_file(path / "dataset_config.yaml", config)

        consolidate_metadata(str(path))

        return cls(
            atomic_embeddings,
            positions,
            atomic_numbers,
            ptr,
            struct_id,
            mol_id,
            stereo_id,
            smiles,
            isomeric_smiles,
            targets_system,
            mask_system,
            targets_atom,
            mask_atom,
            config,
        )

    @property
    def N_structures(self) -> int:
        """Number of stored structures (conformers)."""
        return int(self._mol_cursor)

    @property
    def N_atoms(self) -> int:
        """Total number of stored atoms."""
        return int(self._atom_cursor)

    @property
    def N_molecules(self) -> int:
        """
        Number of unique molecules, assuming 0-based contiguous `molecule_id`s.
        """
        n = int(self.molecule_ids.shape[0])
        if n == 0:
            return 0
        return int(np.asarray(self.molecule_ids[:]).max()) + 1

    def __len__(self):
        return self.N_structures

    def _ensure_capacity_atoms(
        self, extra_atoms: int, growth: float = 1.5, min_slack: int = 100_000
    ):
        need = self._atom_cursor + extra_atoms
        cur = int(self.positions.shape[0])
        if need > cur:
            new_cap = max(need, int(cur * growth) + min_slack)
            if self.config.contains_embeddings:
                self.atomic_embeddings.resize((new_cap, self.config.embedding_dim))
            self.positions.resize((new_cap, 3))
            self.atomic_numbers.resize((new_cap,))
            if self.targets_atom is not None:
                ncols = self.targets_atom.shape[1]
                self.targets_atom.resize((new_cap, ncols))
                self.mask_atom.resize((new_cap, ncols))

    def _ensure_capacity_mols(
        self, extra_mols: int, growth: float = 1.5, min_slack: int = 10_000
    ):
        need_ptr = self._mol_cursor + extra_mols + 1
        cur_ptr = int(self.ptr.shape[0])
        if need_ptr > cur_ptr:
            new_ptr_cap = max(need_ptr, int(cur_ptr * growth) + min_slack)
            self.ptr.resize((new_ptr_cap,))
        need_ids = self._mol_cursor + extra_mols
        cur_ids = int(self.structure_ids.shape[0])
        if need_ids > cur_ids:
            new_ids_cap = max(need_ids, int(cur_ids * growth) + (min_slack - 1))
            self.structure_ids.resize((new_ids_cap,))
            self.molecule_ids.resize((new_ids_cap,))
            self.isomer_ids.resize((new_ids_cap,))
            if self.targets_system is not None:
                ncols = self.targets_system.shape[1]
                self.targets_system.resize((new_ids_cap, ncols))
                self.mask_system.resize((new_ids_cap, ncols))

    def append_batch(
        self,
        embeddings,
        positions,
        atomic_numbers,
        batch_ptr_cumsum,  # length = n_mols, cumulative ends (no leading 0)
        molecule_ids,
        stereoisomer_ids,
        system_targets,
        system_masks,
        atom_targets,
        atom_masks,
    ):
        # Normalize dtype & layout once, here (avoids per-element casting inside zarr)
        if self.config.contains_embeddings and embeddings is not None:
            E = np.asarray(embeddings, dtype="f4", order="C")
        elif not self.config.contains_embeddings and embeddings is None:
            E = None

        else:
            raise ValueError(
                "Dataset Config and batch append disagree on whether there should be embeddings here or not"
            )

        P = np.asarray(positions, dtype="f4", order="C")
        Z = np.asarray(atomic_numbers, dtype="u1", order="C")
        M = np.asarray(molecule_ids, dtype="i8", order="C")
        R = np.asarray(stereoisomer_ids, dtype="i8", order="C")
        C = np.asarray(batch_ptr_cumsum, dtype="i8", order="C")

        assert C.shape[0] == M.shape[0] == R.shape[0]

        n_atoms = P.shape[0]
        n_mols = int(C.shape[0])

        # If not, we still attempt to write, but this hints at upstream issues
        # (e.g., inconsistent system_idx vs embeddings sizing).
        self._ensure_capacity_atoms(n_atoms, growth=4)
        self._ensure_capacity_mols(n_mols, growth=4)

        a0, a1 = self._atom_cursor, self._atom_cursor + n_atoms
        m0, m1 = self._mol_cursor, self._mol_cursor + n_mols

        if atom_targets is not None:
            AT = np.asarray(atom_targets, dtype="f4", order="C")
            AM = np.asarray(atom_masks, dtype="i8", order="C")

            self.mask_atom[a0:a1] = AM
            self.targets_atom[a0:a1] = AT

        # per-atom writes
        if E is not None:
            self.atomic_embeddings[a0:a1, :] = E

        self.positions[a0:a1, :] = P
        self.atomic_numbers[a0:a1] = Z

        # ptr: extend ends relative to existing sentinel at m0
        self.ptr[m0 + 1 : m1 + 1] = self.ptr[m0] + C

        # Get a new array of structure ids, starting at molecule_cursor
        S = np.arange(0, n_mols) + m0

        # ids
        self.structure_ids[m0:m1] = S

        self.molecule_ids[m0:m1] = M
        self.isomer_ids[m0:m1] = R

        if system_targets is not None:
            ST = np.asarray(system_targets, dtype="f4", order="C")
            SM = np.asarray(system_masks, dtype="i8", order="C")

            self.targets_system[m0:m1] = ST
            self.mask_system[m0:m1] = SM

        self._atom_cursor = a1
        self._mol_cursor = m1

    def get_structure_ids_from_molecule_ids(self, molecule_ids: set):
        # retrieve all structure_ids, for which the molecule_id is in the molecules ids set
        if not molecule_ids:
            return np.asarray([], dtype="i8")

        mol_ids = np.asarray(self.molecule_ids[:])
        mask = np.isin(mol_ids, list(molecule_ids))
        struct_ids = np.asarray(self.structure_ids[:])
        retrieved_structure_ids = struct_ids[mask]
        return retrieved_structure_ids

    def shrink_to_fit(self):
        if self.config.contains_embeddings:
            self.atomic_embeddings.resize(
                (self._atom_cursor, self.config.embedding_dim)
            )
        self.positions.resize((self._atom_cursor, 3))
        self.atomic_numbers.resize((self._atom_cursor,))
        self.ptr.resize((self._mol_cursor + 1,))
        self.structure_ids.resize((self._mol_cursor,))
        self.molecule_ids.resize((self._mol_cursor,))
        self.isomer_ids.resize((self._mol_cursor,))

        if self.targets_system is not None:
            ncols = self.targets_system.shape[1]
            self.targets_system.resize((self._mol_cursor, ncols))
            self.mask_system.resize((self._mol_cursor, ncols))
        if self.targets_atom is not None:
            ncols = self.targets_atom.shape[1]
            self.targets_atom.resize((self._atom_cursor, ncols))
            self.mask_atom.resize((self._atom_cursor, ncols))

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

        molecules: list[Atoms] = []
        append = molecules.append
        for idx in range(n_struct):
            start = ptr[idx]
            end = ptr[idx + 1]
            append(
                Atoms(numbers=atomic_numbers[start:end], positions=positions[start:end])
            )

        return molecules
