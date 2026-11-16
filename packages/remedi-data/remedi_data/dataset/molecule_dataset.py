import os
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pydantic_yaml as pyd_yaml
import zarr
from ase import Atoms
from zarr import Array
from zarr.codecs import BloscCodec
from zarr.storage import LocalStore

from remedi_data.dataset.config import DatasetConfig


@dataclass(frozen=True)
class StructureArrays:
    """The five structure arrays a training loader reads, by their zarr names.

    This is the contract between the zarr layout (owned here) and the training
    code in ``remedi``: field names are the array names in the zarr group.
    Structure ``i`` owns atoms ``molecule_ptr[i]:molecule_ptr[i + 1]`` of
    ``positions`` (float32, ``(n_atoms, 3)``) and ``atomic_numbers`` (uint8);
    ``total_charge`` and ``multiplicity`` (float32) are per structure. Each
    field is a numpy array (``in_memory=True`` or a ``structure_ids`` selection)
    or a read-only zarr array whose slices are numpy arrays.
    """

    molecule_ptr: np.ndarray | Array
    positions: np.ndarray | Array
    atomic_numbers: np.ndarray | Array
    total_charge: np.ndarray | Array
    multiplicity: np.ndarray | Array


def _gather_atoms(array: Array, atom_index: np.ndarray) -> np.ndarray:
    """``array[atom_index]`` for a sorted index, reading one shard at a time."""
    gathered = np.empty((len(atom_index), *array.shape[1:]), dtype=array.dtype)
    block = int((array.shards or array.chunks)[0])
    for block_start in range(0, int(array.shape[0]), block):
        first, last = np.searchsorted(
            atom_index, [block_start, block_start + block], side="left"
        )
        if first == last:
            continue
        block_stop = int(atom_index[last - 1]) + 1
        values = np.asarray(array[block_start:block_stop])
        gathered[first:last] = values[atom_index[first:last] - block_start]
    return gathered


def _gather_structures(
    arrays: dict[str, Array], structure_ids: np.ndarray, path: Path
) -> StructureArrays:
    pointer = np.asarray(arrays["molecule_ptr"][:], dtype=np.int64)
    structure_count = len(pointer) - 1
    structure_ids = structure_ids.astype(np.int64, copy=False)
    if structure_ids.ndim != 1 or np.any(np.diff(structure_ids) <= 0):
        raise ValueError("structure_ids must be a sorted 1-d array without repeats")
    if len(structure_ids) and (
        structure_ids[0] < 0 or structure_ids[-1] >= structure_count
    ):
        raise ValueError(
            f"structure_ids reach outside the {structure_count} structures of {path}"
        )
    starts = pointer[structure_ids]
    counts = pointer[structure_ids + 1] - starts
    new_pointer = np.zeros(len(structure_ids) + 1, dtype=np.int64)
    np.cumsum(counts, out=new_pointer[1:])
    atom_index = np.repeat(starts - new_pointer[:-1], counts) + np.arange(
        new_pointer[-1], dtype=np.int64
    )
    return StructureArrays(
        molecule_ptr=new_pointer.astype(arrays["molecule_ptr"].dtype),
        positions=_gather_atoms(arrays["positions"], atom_index),
        atomic_numbers=_gather_atoms(arrays["atomic_numbers"], atom_index),
        total_charge=np.asarray(arrays["total_charge"][:])[structure_ids],
        multiplicity=np.asarray(arrays["multiplicity"][:])[structure_ids],
    )


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
        targets_system: Array | None,
        mask_system: Array | None,
        targets_atom: Array | None,
        mask_atom: Array | None,
        split: Array | None,
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

        self.targets_system = targets_system
        self.mask_system = mask_system
        self.targets_atom = targets_atom
        self.mask_atom = mask_atom
        # Per-structure literature split (uint8 Split codes). None when the
        # dataset was created without the tasks group.
        self.split = split

        self.config = config

    @staticmethod
    def open_structure_arrays(
        path: Path,
        in_memory: bool = False,
        structure_ids: np.ndarray | None = None,
    ) -> StructureArrays:
        """Open a dataset's structure arrays read-only.

        Takes a path rather than an open dataset so a data-loader worker can
        reopen it after pickling. ``in_memory=True`` loads all five arrays into
        numpy; otherwise they stay lazy zarr arrays (sharded reads pull byte
        ranges; the OS page cache covers repeated access to hot shards).

        ``structure_ids`` (sorted, unique, e.g. from ``read_selection``) gathers
        only those structures into compact in-memory arrays, renumbered
        ``0..len(structure_ids) - 1`` in that order; ``in_memory`` is then
        ignored. The atom arrays are read one zarr shard at a time, so memory
        stays near the size of the selection.

        Raises:
            KeyError: if an array is missing.
            ValueError: if ``structure_ids`` is not sorted, unique and in range.
        """
        group = zarr.open_group(store=LocalStore(Path(path), read_only=True), mode="r")
        arrays: dict[str, Array] = {}
        for field in fields(StructureArrays):
            if field.name not in group:
                raise KeyError(
                    f"{path} has no {field.name!r} array; not a structure dataset"
                )
            array = group[field.name]
            if not isinstance(array, Array):
                raise TypeError(
                    f"{path}/{field.name} is a zarr group, expected an array"
                )
            arrays[field.name] = array
        if structure_ids is not None:
            return _gather_structures(arrays, np.asarray(structure_ids), path)
        if in_memory:
            return StructureArrays(
                **{name: np.array(array) for name, array in arrays.items()}
            )
        return StructureArrays(**arrays)

    @classmethod
    def open_existing_dataset_from_dir(cls, path: Path):
        """Open an existing on-disk dataset created by create_empty_dataset, read-only.

        Every array refuses writes (``ValueError``). The only writer is
        :meth:`create_empty_dataset` followed by ``ShardAlignedWriter``; a
        finished dataset is replaced as a whole, never edited in place.

        Raises:
            FileNotFoundError: if ``path`` holds no zarr group or no
                ``dataset_config.yaml``.
            KeyError: if one of the arrays is missing.
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
            split = tasks_grp["split"] if "split" in tasks_grp else None

        else:
            targets_system, targets_atom, mask_atom, mask_system, split = (
                None,
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
            targets_system=targets_system,
            mask_system=mask_system,
            targets_atom=targets_atom,
            mask_atom=mask_atom,
            split=split,
            config=config,
            root=path,
        )

    @classmethod
    def create_empty_dataset(cls, path: Path, config: DatasetConfig):
        path = Path(path)
        os.makedirs(path, exist_ok=True)

        compressor = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")

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
        atomic_numbers = _mk(g, "atomic_numbers", (0,), config.atom_chunk, a_cps, "u1")

        # ragged pointer
        ptr = _mk(g, "molecule_ptr", (1,), config.molecule_chunk, m_cps, "i8")
        ptr[:] = 0

        # per-structure scalars
        total_charge = _mk(g, "total_charge", (0,), config.molecule_chunk, m_cps, "f4")
        multiplicity = _mk(g, "multiplicity", (0,), config.molecule_chunk, m_cps, "f4")

        # ID section
        ids = g.require_group("ids")
        mol_id = _mk(ids, "molecule_id", (0,), config.molecule_chunk, m_cps, "i8")
        stereo_id = _mk(
            ids, "stereoisomer_id", (0,), config.molecule_chunk, m_cps, "i8"
        )
        struct_id = _mk(ids, "structure_id", (0,), config.molecule_chunk, m_cps, "i8")

        targets_system = None
        mask_system = None
        targets_atom = None
        mask_atom = None
        split = None

        if config.tasks is not None:
            tasks_grp = g.require_group("tasks")

            # Per-structure literature split; written for every molecule by the
            # ShardAlignedWriter (defaults to Split.unassigned when a generator
            # supplies no split).
            split = _mk(tasks_grp, "split", (0,), config.molecule_chunk, m_cps, "u1")

            Nsys = len(config.tasks.system_cols)
            Natom = len(config.tasks.atom_cols)

            if Nsys > 0:
                targets_system = _mk(
                    tasks_grp,
                    "targets_system",
                    (0, Nsys),
                    config.molecule_chunk,
                    m_cps,
                    "f4",
                    max(1, Nsys),
                )
                mask_system = _mk(
                    tasks_grp,
                    "mask_system",
                    (0, Nsys),
                    config.molecule_chunk,
                    m_cps,
                    "u1",
                    max(1, Nsys),
                )

            if Natom > 0:
                targets_atom = _mk(
                    tasks_grp,
                    "targets_atom",
                    (0, Natom),
                    config.atom_chunk,
                    a_cps,
                    "f4",
                    max(1, Natom),
                )
                mask_atom = _mk(
                    tasks_grp,
                    "mask_atom",
                    (0, Natom),
                    config.atom_chunk,
                    a_cps,
                    "u1",
                    max(1, Natom),
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
            targets_system,
            mask_system,
            targets_atom,
            mask_atom,
            split,
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
        import polars as pl

        table = pl.read_parquet(self.path / "table.parquet")
        if "isomeric_smiles" not in table.columns:
            raise KeyError(f"{self.path} has no isomeric_smiles")
        return table.sort("structure_id")["isomeric_smiles"].to_list()

    def unique_smiles(self) -> list[str]:
        """The distinct isomeric SMILES of ``table.parquet``, in first-seen order.

        Empty for a dataset without SMILES (such as tmQM).
        """
        import polars as pl

        table_path = self.path / "table.parquet"
        if "isomeric_smiles" not in pl.read_parquet_schema(table_path):
            return []
        smiles = pl.read_parquet(table_path, columns=["isomeric_smiles"])
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
