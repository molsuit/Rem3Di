"""Shard-aligned, buffered writer for a :class:`MoleculeDataset`.

zarr v3 sharding does a full read-modify-write of a shard file for any write
that does not cover whole shards. Writing one append-batch at a time (batch
<< shard) would therefore rewrite the in-progress shard once per batch.

This writer owns the build-time concern of *when* to write: it buffers
appended rows in memory and flushes them to the dataset's zarr arrays only in
blocks aligned to the absolute shard grid, so each shard file is written
exactly once (a single store ``set``, no read-modify-write ``get``).
``finalize`` writes the final partial shard once and trims the arrays to the
exact size.

The writer is the only append API: ``MoleculeDataset`` is pure storage.
"""

from __future__ import annotations

import numpy as np

from remedi_data.dataset.molecule_dataset import MoleculeDataset


class ShardAlignedWriter:
    def __init__(self, dataset: MoleculeDataset):
        self.ds = dataset
        cfg = dataset.config

        self._atom_shard = int(cfg.atom_chunk * cfg.atom_chunks_per_shard)
        self._mol_shard = int(cfg.molecule_chunk * cfg.molecule_chunks_per_shard)

        # Pick up any rows already on disk. ptr is cumulative; ptr[k] is the
        # atom count before molecule k, so ptr[-1] == total atoms.
        ptr = dataset.ptr
        self._mol_cursor = int(ptr.shape[0]) - 1
        self._atom_cursor = int(np.asarray(ptr[self._mol_cursor]))
        self._atom_flushed = self._atom_cursor
        self._mol_flushed = self._mol_cursor
        self._ptr_base = self._atom_cursor

        # in-memory tails holding rows in [flushed, cursor)
        self._buf_pos: list[np.ndarray] = []
        self._buf_anum: list[np.ndarray] = []
        self._buf_sizes: list[np.ndarray] = []  # per-molecule atom counts
        self._buf_mid: list[np.ndarray] = []
        self._buf_iso: list[np.ndarray] = []
        self._buf_sid: list[np.ndarray] = []
        self._buf_q: list[np.ndarray] = []
        self._buf_mult: list[np.ndarray] = []

    # -- logical progress --
    @property
    def n_structures(self) -> int:
        return int(self._mol_cursor)

    @property
    def n_atoms(self) -> int:
        return int(self._atom_cursor)

    # -- append / finalize -------------------------------------------------
    def append_batch(
        self,
        positions,
        atomic_numbers,
        batch_ptr_cumsum,  # length = n_mols, cumulative ends (no leading 0)
        molecule_ids,
        stereoisomer_ids,
        total_charge,
        multiplicity,
        *,
        structure_ids=None,
    ) -> None:
        """Buffer one batch of structures for the next shard-aligned flush.

        Args:
            structure_ids: the ``ids/structure_id`` values of this batch;
                ``None`` derives them from the shard cursor as ``arange``.

        Raises:
            ValueError: if ``structure_ids`` does not have one entry per
                structure in the batch.
        """
        P = np.asarray(positions, dtype="f4", order="C")
        Z = np.asarray(atomic_numbers, dtype="u1", order="C")
        M = np.asarray(molecule_ids, dtype="i8", order="C")
        R = np.asarray(stereoisomer_ids, dtype="i8", order="C")
        C = np.asarray(batch_ptr_cumsum, dtype="i8", order="C")
        Q = np.asarray(total_charge, dtype="f4", order="C")
        S_mult = np.asarray(multiplicity, dtype="f4", order="C")

        assert C.shape[0] == M.shape[0] == R.shape[0]
        assert Q.shape[0] == M.shape[0] and S_mult.shape[0] == M.shape[0]

        n_atoms = int(P.shape[0])
        n_mols = int(C.shape[0])
        if n_mols == 0:
            return

        # per-molecule atom counts; cumsum reconstructs the global ptr later
        sizes = np.diff(C, prepend=np.int64(0)).astype("i8", copy=False)
        if int(sizes.sum()) != n_atoms:
            raise ValueError(
                "batch_ptr_cumsum inconsistent with positions: "
                f"sizes sum {int(sizes.sum())} != n_atoms {n_atoms}"
            )

        self._buf_pos.append(P)
        self._buf_anum.append(Z)
        self._buf_sizes.append(sizes)
        self._buf_mid.append(M)
        self._buf_iso.append(R)
        self._buf_q.append(Q)
        self._buf_mult.append(S_mult)
        self._buffer_structure_ids(structure_ids, n_mols)

        self._atom_cursor += n_atoms
        self._mol_cursor += n_mols

        self._flush(final=False)

    def _buffer_structure_ids(self, structure_ids, n_mols: int) -> None:
        if structure_ids is None:
            start = self._mol_cursor
            self._buf_sid.append(np.arange(start, start + n_mols, dtype="i8"))
            return
        array = np.asarray(structure_ids, dtype="i8", order="C")
        if array.shape != (n_mols,):
            raise ValueError(
                f"structure_ids must have length {n_mols}, got shape {array.shape}"
            )
        self._buf_sid.append(array)

    def finalize(self) -> None:
        """Flush every remaining buffered row (the final, partial shards are
        written here, exactly once each), then trim arrays to the exact
        size."""
        self._flush(final=True)
        if self._atom_flushed != self._atom_cursor or (
            self._mol_flushed != self._mol_cursor
        ):
            raise RuntimeError(
                "finalize: buffer not fully flushed "
                f"(atoms {self._atom_flushed}/{self._atom_cursor}, "
                f"mols {self._mol_flushed}/{self._mol_cursor})"
            )

        ds = self.ds
        ds.positions.resize((self._atom_cursor, 3))
        ds.atomic_numbers.resize((self._atom_cursor,))
        ds.ptr.resize((self._mol_cursor + 1,))
        ds.structure_ids.resize((self._mol_cursor,))
        ds.molecule_ids.resize((self._mol_cursor,))
        ds.isomer_ids.resize((self._mol_cursor,))
        ds.total_charge.resize((self._mol_cursor,))
        ds.multiplicity.resize((self._mol_cursor,))

    # -- flushing ----------------------------------------------------------
    def _next_stop(
        self, flushed: int, cursor: int, shard: int, final: bool
    ) -> int | None:
        """End index of the next flush block, aligned to the *absolute*
        shard grid. ``None`` when no whole shard is ready and this is not
        the final flush (so the rows stay buffered)."""
        boundary = ((flushed // shard) + 1) * shard
        if cursor >= boundary:
            return boundary
        if final:
            return cursor
        return None

    def _flush(self, *, final: bool) -> None:
        self._flush_atom_axis(final)
        self._flush_mol_axis(final)

    def _flush_atom_axis(self, final: bool) -> None:
        if self._atom_flushed >= self._atom_cursor:
            return
        if (
            self._next_stop(
                self._atom_flushed, self._atom_cursor, self._atom_shard, final
            )
            is None
        ):
            return

        ds = self.ds
        pos = np.concatenate(self._buf_pos, axis=0)
        anum = np.concatenate(self._buf_anum, axis=0)
        off = 0
        while self._atom_flushed < self._atom_cursor:
            stop = self._next_stop(
                self._atom_flushed, self._atom_cursor, self._atom_shard, final
            )
            if stop is None:
                break
            b = self._atom_flushed
            n = stop - b
            ds.positions.resize((stop, 3))
            ds.atomic_numbers.resize((stop,))
            ds.positions[b:stop, :] = pos[off : off + n]
            ds.atomic_numbers[b:stop] = anum[off : off + n]
            off += n
            self._atom_flushed = stop
        self._buf_pos = [pos[off:]]
        self._buf_anum = [anum[off:]]

    def _flush_mol_axis(self, final: bool) -> None:
        if self._mol_flushed >= self._mol_cursor:
            return
        if (
            self._next_stop(self._mol_flushed, self._mol_cursor, self._mol_shard, final)
            is None
        ):
            return

        ds = self.ds
        sizes = np.concatenate(self._buf_sizes, axis=0)
        mid = np.concatenate(self._buf_mid, axis=0)
        iso = np.concatenate(self._buf_iso, axis=0)
        sid = np.concatenate(self._buf_sid, axis=0)
        q = np.concatenate(self._buf_q, axis=0)
        mult = np.concatenate(self._buf_mult, axis=0)
        off = 0
        while self._mol_flushed < self._mol_cursor:
            stop = self._next_stop(
                self._mol_flushed, self._mol_cursor, self._mol_shard, final
            )
            if stop is None:
                break
            b = self._mol_flushed
            n = stop - b
            # ptr is cumulative atom offsets, length N_mol+1, with a leading
            # 0 sentinel at index 0 (set at creation). The +1 index shift vs
            # the molecule shard grid is harmless: ptr is i8 and tiny next
            # to positions.
            ends = self._ptr_base + np.cumsum(sizes[off : off + n], dtype="i8")
            ds.ptr.resize((stop + 1,))
            ds.ptr[b + 1 : stop + 1] = ends
            self._ptr_base = int(ends[-1])

            ds.structure_ids.resize((stop,))
            ds.structure_ids[b:stop] = sid[off : off + n]
            ds.molecule_ids.resize((stop,))
            ds.isomer_ids.resize((stop,))
            ds.total_charge.resize((stop,))
            ds.multiplicity.resize((stop,))
            ds.molecule_ids[b:stop] = mid[off : off + n]
            ds.isomer_ids[b:stop] = iso[off : off + n]
            ds.total_charge[b:stop] = q[off : off + n]
            ds.multiplicity[b:stop] = mult[off : off + n]
            off += n
            self._mol_flushed = stop
        self._buf_sizes = [sizes[off:]]
        self._buf_mid = [mid[off:]]
        self._buf_iso = [iso[off:]]
        self._buf_sid = [sid[off:]]
        self._buf_q = [q[off:]]
        self._buf_mult = [mult[off:]]
