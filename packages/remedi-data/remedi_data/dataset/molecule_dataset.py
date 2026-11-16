from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from ase import Atoms
from zarr import Array
from zarr.codecs import BloscCodec
from zarr.storage import LocalStore

from remedi_data.bundle import read_table_polars, table_columns
from remedi_data.dataset.config import DatasetConfig
from remedi_data.dataset.packed_structures import PackedStructures, gather_index
from remedi_data.yaml_io import read_model, write_model


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

        self.config = config

    @classmethod
    def open_existing_dataset_from_dir(cls, path: Path):
        """Open an existing on-disk dataset written by :meth:`write`, read-only.

        Every array refuses writes (``ValueError``). The only writer is
        :meth:`write`; a finished dataset is replaced as a whole, never edited
        in place.

        Raises:
            FileNotFoundError: if ``path`` holds no zarr group or no
                ``dataset_config.yaml``.
            KeyError: if one of the arrays is missing.
            pydantic.ValidationError: if ``dataset_config.yaml`` holds a key
                :class:`DatasetConfig` does not know, such as the ``tasks`` of
                the layout before labels left the zarr
                (``scripts/dataset_creation/migrate_dataset_identity.py``).
        """
        path = Path(path)
        store = LocalStore(path, read_only=True)
        g = zarr.open_group(store=store, mode="r")

        positions = g["positions"]
        atomic_numbers = g["atomic_numbers"]
        ptr = g["molecule_ptr"]
        total_charge = g["total_charge"]
        multiplicity = g["multiplicity"]

        ids = g["ids"]
        mol_id = ids["molecule_id"]
        stereo_id = ids["stereoisomer_id"]
        struct_id = ids["structure_id"]

        config = read_model(path / "dataset_config.yaml", DatasetConfig)

        return cls(
            positions,
            atomic_numbers,
            ptr,
            struct_id,
            mol_id,
            stereo_id,
            total_charge,
            multiplicity,
            config=config,
            root=path,
        )

    @classmethod
    def write(
        cls,
        path: Path,
        structures: PackedStructures,
        *,
        structure_ids: np.ndarray,
        molecule_ids: np.ndarray,
        stereoisomer_ids: np.ndarray,
        config: DatasetConfig,
    ) -> MoleculeDataset:
        """Write a new zarr at ``path`` (replacing one there) and open it read-only.

        ``structures`` must carry ``total_charge`` and ``multiplicity``. Every
        array is written in one piece through ``create_array(data=...)``, so
        each shard file is written exactly once. Chunks and shards follow
        ``config`` along the first axis: atoms in ``atom_chunk`` chunks of
        ``atom_chunks_per_shard`` per shard, structures in ``molecule_chunk``
        chunks of ``molecule_chunks_per_shard``.

        Raises:
            ValueError: if ``structures`` has no charges or multiplicities, or
                an id array does not have one entry per structure.
        """
        if structures.total_charge is None or structures.multiplicity is None:
            raise ValueError("the structures to write need charges and multiplicities")
        for name, ids in (
            ("structure_ids", structure_ids),
            ("molecule_ids", molecule_ids),
            ("stereoisomer_ids", stereoisomer_ids),
        ):
            if np.shape(ids) != (len(structures),):
                raise ValueError(
                    f"{name} has shape {np.shape(ids)}, expected ({len(structures)},)"
                )
        path = Path(path)
        group = zarr.create_group(store=LocalStore(path), overwrite=True)
        compressor = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")

        def write_array(
            parent, name: str, data: np.ndarray, chunk: int, per_shard: int
        ):
            parent.create_array(
                name=name,
                data=data,
                chunks=(chunk, *data.shape[1:]),
                shards=(chunk * per_shard, *data.shape[1:]),
                compressors=compressor,
            )

        atoms = (config.atom_chunk, config.atom_chunks_per_shard)
        per_structure = (config.molecule_chunk, config.molecule_chunks_per_shard)
        write_array(group, "positions", structures.positions, *atoms)
        write_array(group, "atomic_numbers", structures.atomic_numbers, *atoms)
        write_array(group, "molecule_ptr", structures.ptr, *per_structure)
        write_array(group, "total_charge", structures.total_charge, *per_structure)
        write_array(group, "multiplicity", structures.multiplicity, *per_structure)
        ids = group.require_group("ids")
        for name, values in (
            ("molecule_id", molecule_ids),
            ("stereoisomer_id", stereoisomer_ids),
            ("structure_id", structure_ids),
        ):
            write_array(ids, name, np.asarray(values, dtype=np.int64), *per_structure)
        write_model(path / "dataset_config.yaml", config)
        return cls.open_existing_dataset_from_dir(path)

    def read_structures(
        self, structure_ids: np.ndarray | None = None
    ) -> PackedStructures:
        """The structures (with charges and multiplicities) in memory.

        ``structure_ids`` (sorted, unique, e.g. from ``read_selection``) reads
        only those structures, renumbered ``0..len(structure_ids) - 1`` in that
        order, through zarr's orthogonal indexing (only the shards that hold
        them are read); ``None`` reads every structure.

        Raises:
            ValueError: if ``structure_ids`` is not sorted, unique and in range.
        """
        pointer = np.asarray(self.ptr[:], dtype=np.int64)
        if structure_ids is None:
            return PackedStructures.from_arrays(
                self.positions[:],
                self.atomic_numbers[:],
                pointer,
                self.total_charge[:],
                self.multiplicity[:],
            )
        structure_ids = np.asarray(structure_ids).astype(np.int64, copy=False)
        if structure_ids.ndim != 1 or np.any(np.diff(structure_ids) <= 0):
            raise ValueError("structure_ids must be a sorted 1-d array without repeats")
        if len(structure_ids) and (
            structure_ids[0] < 0 or structure_ids[-1] >= len(pointer) - 1
        ):
            raise ValueError(
                f"structure_ids reach outside the {len(pointer) - 1} structures "
                f"of {self.path}"
            )
        new_pointer, atom_index = gather_index(pointer, structure_ids)
        return PackedStructures.from_arrays(
            self.positions.oindex[atom_index],
            self.atomic_numbers.oindex[atom_index],
            new_pointer,
            self.total_charge.oindex[structure_ids],
            self.multiplicity.oindex[structure_ids],
        )

    @property
    def N_structures(self) -> int:
        # ptr has length N_structures + 1 (leading 0 sentinel).
        return int(self.ptr.shape[0]) - 1

    @property
    def N_atoms(self) -> int:
        # ptr is cumulative; the last entry is the total atom count.
        return int(np.asarray(self.ptr[self.ptr.shape[0] - 1]))

    @property
    def N_molecules(self) -> int:
        # Distinct ids, not max + 1: ids keep the bundle's numbering, so a
        # molecule dropped at embedding leaves a gap.
        return int(np.unique(np.asarray(self.molecule_ids[:])).size)

    def __len__(self):
        return self.N_structures

    def get_smiles_per_structure(self) -> list[str]:
        """One isomeric SMILES per structure, in structure order, from ``table.parquet``.

        Raises:
            FileNotFoundError: if the dataset has no table.
            KeyError: if the table has no ``isomeric_smiles`` (a corpus without
                SMILES, such as tmQM).
        """
        if "isomeric_smiles" not in table_columns(self.path):
            raise KeyError(f"{self.path} has no isomeric_smiles")
        table = read_table_polars(
            self.path, columns=["structure_id", "isomeric_smiles"]
        )
        return table.sort("structure_id")["isomeric_smiles"].to_list()

    def unique_smiles(self) -> list[str]:
        """The distinct isomeric SMILES of ``table.parquet``, in first-seen order.

        Empty for a dataset without SMILES (such as tmQM).
        """
        import polars as pl

        if "isomeric_smiles" not in table_columns(self.path):
            return []
        smiles = read_table_polars(self.path, columns=["isomeric_smiles"])
        present = pl.col("isomeric_smiles").is_not_null() & (
            pl.col("isomeric_smiles") != ""
        )
        return (
            smiles.filter(present)["isomeric_smiles"]
            .unique(maintain_order=True)
            .to_list()
        )

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
