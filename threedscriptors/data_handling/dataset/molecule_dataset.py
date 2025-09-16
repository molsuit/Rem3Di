import os
from pathlib import Path

import numpy as np
import pydantic_yaml as pyd_yaml
import zarr
from numcodecs import Blosc
from zarr import Array

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

        atomic_embeddings = g["atomic_embeddings"]
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
            config=config,
        )

    def _structure_atom_span(self, i: int) -> tuple[int, int]:
        """Return [a0, a1) atom indices for structure i."""
        a0 = int(self._ptr[i])
        a1 = int(self._ptr[i + 1])
        return a0, a1

    @classmethod
    def create_empty_dataset(cls, path: Path, config: DatasetConfig):

        # Configure compressor
        os.makedirs(path, exist_ok=True)

        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

        store = zarr.DirectoryStore(path)
        g = zarr.group(store=store, overwrite=True)

        # per-atom
        atomic_embeddings = g.create(
            "atomic_embeddings",
            shape=(0, config.embedding_dim),
            chunks=(config.atom_chunk, config.embedding_dim),
            dtype="f4",
            compressor=compressor,
        )
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

        # Write config to disk
        pyd_yaml.to_yaml_file(path / "dataset_config.yaml", config)

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

    def _ensure_capacity_atoms(
        self, extra_atoms: int, growth: float = 1.5, min_slack: int = 100_000
    ):
        need = self._atom_cursor + extra_atoms
        cur = int(self.atomic_embeddings.shape[0])
        if need > cur:
            new_cap = max(need, int(cur * growth) + min_slack)
            self.atomic_embeddings.resize((new_cap, self.config.embedding_dim))
            self.positions.resize((new_cap, 3))
            self.atomic_numbers.resize((new_cap,))

    def _ensure_capacity_mols(
        self, extra_mols: int, growth: float = 1.5, min_slack: int = 10_000
    ):
        need_ptr = self._mol_cursor + extra_mols + 1  # +1 for sentinel
        cur_ptr = int(self.ptr.shape[0])
        if need_ptr > cur_ptr:
            new_ptr_cap = max(need_ptr, int(cur_ptr * growth) + min_slack)
            self.ptr.resize((new_ptr_cap,))
            # ids are length == _mol_cursor
        need_ids = self._mol_cursor + extra_mols
        cur_ids = int(self.structure_ids.shape[0])
        if need_ids > cur_ids:
            new_ids_cap = max(need_ids, int(cur_ids * growth) + (min_slack - 1))
            self.structure_ids.resize((new_ids_cap,))
            self.molecule_ids.resize((new_ids_cap,))
            self.isomer_ids.resize((new_ids_cap,))

    def append_batch(
        self,
        embeddings,
        positions,
        atomic_numbers,
        batch_ptr_cumsum,  # length = n_mols, cumulative ends (no leading 0)
        molecule_ids,
        stereoisomer_ids,
    ):

        # Normalize dtype & layout once, here (avoids per-element casting inside zarr)
        E = np.asarray(embeddings, dtype="f4", order="C")
        P = np.asarray(positions, dtype="f4", order="C")
        Z = np.asarray(atomic_numbers, dtype="u1", order="C")
        M = np.asarray(molecule_ids, dtype="i8", order="C")
        R = np.asarray(stereoisomer_ids, dtype="i8", order="C")
        C = np.asarray(batch_ptr_cumsum, dtype="i8", order="C")

        assert C.shape[0] == M.shape[0] == R.shape[0]

        n_atoms = E.shape[0]
        n_mols = int(C.shape[0])

        # If not, we still attempt to write, but this hints at upstream issues
        # (e.g., inconsistent system_idx vs embeddings sizing).
        self._ensure_capacity_atoms(n_atoms)
        self._ensure_capacity_mols(n_mols)

        a0, a1 = self._atom_cursor, self._atom_cursor + n_atoms
        m0, m1 = self._mol_cursor, self._mol_cursor + n_mols

        # per-atom writes
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

        print(retrieved_structure_ids)
        return retrieved_structure_ids

    def shrink_to_fit(self):

        self.atomic_embeddings.resize((self._atom_cursor, self.config.embedding_dim))
        self.positions.resize((self._atom_cursor, 3))
        self.atomic_numbers.resize((self._atom_cursor,))
        self.ptr.resize((self._mol_cursor + 1,))
        self.structure_ids.resize((self._mol_cursor,))
        self.molecule_ids.resize((self._mol_cursor,))
        self.isomer_ids.resize((self._mol_cursor,))
