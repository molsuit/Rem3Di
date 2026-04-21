from __future__ import annotations

import mmap
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np

_UINT64 = np.uint64
_UINT64_SIZE = 8


class SmilesStorage:
    """
    Minimal, append-only line store with O(1) id->string via memory maps.

    Layout:
      text_file:  UTF-8 lines, each terminated with '\n'
      index_file: uint64 offsets of length N+1:
                  offsets[i]   = byte start of line i in text_file
                  offsets[N]   = total byte size of text_file (sentinel)
    """

    # ---------- Construction ----------

    @staticmethod
    def create_files(
        text_path: os.PathLike | str,
        index_path: os.PathLike | str,
        overwrite: bool = False,
    ) -> tuple[Path, Path]:
        text_path = Path(text_path).absolute()
        index_path = Path(index_path).absolute()

        if not overwrite:
            for p in (text_path, index_path):
                if p.exists():
                    raise FileExistsError(f"File exists: {p}")

        # Create/truncate both files. Index starts with sentinel 0.
        text_path.write_bytes(b"")
        with open(index_path, "wb") as fidx:
            fidx.write(_UINT64(0).tobytes())
        return text_path, index_path

    def __init__(
        self, text_path: os.PathLike | str, index_path: os.PathLike | str
    ) -> None:
        self._text_path = Path(text_path).absolute()
        self._index_path = Path(index_path).absolute()

        if not (self._text_path.exists() and self._index_path.exists()):
            raise FileNotFoundError(
                "Both text and index files must exist; call create_files() first."
            )

        self._text_file = open(self._text_path, "rb")
        self._text_mmap: mmap.mmap | None = (
            None  # lazily created if file is empty at open
        )

        self._offsets_memmap = np.memmap(self._index_path, dtype=np.uint64, mode="r")
        self._verify_integrity_or_raise()

        self._str2id: dict[str, int] | None = None

        # Create text mmap if file is non-empty
        if self._text_path.stat().st_size > 0:
            self._text_mmap = mmap.mmap(
                self._text_file.fileno(), 0, access=mmap.ACCESS_READ
            )

    def to_list(self):
        n = len(self)
        if n == 0:
            return []

        if self._text_mmap is None:
            size = self._text_path.stat().st_size
            if size == 0:
                return []
            self._text_mmap = mmap.mmap(
                self._text_file.fileno(), 0, access=mmap.ACCESS_READ
            )

        mm = self._text_mmap
        if mm is None:
            return []

        offsets = self._offsets_memmap
        starts = offsets[:-1]
        ends = offsets[1:]
        result: list[str] = [""] * n

        for i, (start, end) in enumerate(zip(starts, ends, strict=False)):
            a = int(start)
            b = int(end)
            if b > a and mm[b - 1] == 10:  # strip trailing newline
                result[i] = mm[a : b - 1].decode("utf-8")
            else:
                result[i] = mm[a:b].decode("utf-8")

        return result

    def __len__(self) -> int:
        return max(0, self._offsets_memmap.shape[0] - 1)

    def close(self) -> None:
        try:
            if self._text_mmap is not None:
                self._text_mmap.close()
        except Exception:
            pass
        try:
            self._text_file.close()
        except Exception:
            pass
        try:
            del self._offsets_memmap
        except Exception:
            pass

    def __enter__(self):  # context manager support
        return self

    def __iter__(self):
        n = len(self)
        if n == 0:
            return iter(())

        # Lazily create text mmap if needed
        if self._text_mmap is None:
            size = self._text_path.stat().st_size
            if size == 0:
                return iter(())
            self._text_mmap = mmap.mmap(
                self._text_file.fileno(), 0, access=mmap.ACCESS_READ
            )

        mm = self._text_mmap
        offsets = self._offsets_memmap
        starts = offsets[:-1]
        ends = offsets[1:]

        def _gen():
            for start, end in zip(starts, ends, strict=False):
                a = int(start)
                b = int(end)
                # strip trailing '\n' (byte 10) if present
                if b > a and mm[b - 1] == 10:
                    yield mm[a : b - 1].decode("utf-8")
                else:
                    yield mm[a:b].decode("utf-8")

        return _gen()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _norm(self, s: str) -> str:
        return s.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")

    # ---------- Public API ----------

    def id_to_string(self, entry_id: int) -> str:
        n = len(self)
        if not (0 <= entry_id < n):
            raise IndexError(f"id {entry_id} out of range [0, {n}).")

        # Create text mmap lazily if file was empty at construction time.
        if self._text_mmap is None:
            if self._text_path.stat().st_size == 0:
                raise RuntimeError("Text file empty but index reports entries.")
            self._text_mmap = mmap.mmap(
                self._text_file.fileno(), 0, access=mmap.ACCESS_READ
            )

        start = int(self._offsets_memmap[entry_id])
        end = int(self._offsets_memmap[entry_id + 1])
        if end < start:
            raise RuntimeError("Corrupt index: non-monotonic offsets.")

        # Trim trailing newline if present
        if end > start and self._text_mmap[end - 1 : end] == b"\n":
            return self._text_mmap[start : end - 1].decode("utf-8")
        return self._text_mmap[start:end].decode("utf-8")

    def append_lines(self, lines: Iterable[str]) -> int:
        """
        Append strings as lines. Returns the starting id for this batch (old size).
        Newlines inside strings are replaced with spaces.
        """
        # Normalize & collect
        batch: list[str] = []
        for s in lines:
            s_norm = self._norm(s)
            batch.append(s_norm)
        if not batch:
            return len(self)

        start_id = len(self)

        with open(self._text_path, "ab") as ftxt, open(self._index_path, "r+b") as fidx:
            # Check current end and index sentinel agree
            ftxt.seek(0, os.SEEK_END)
            current_end = ftxt.tell()

            fidx.seek(-_UINT64_SIZE, os.SEEK_END)
            (sentinel_before,) = np.frombuffer(
                fidx.read(_UINT64_SIZE), dtype=np.uint64, count=1
            )
            if int(sentinel_before) != current_end:
                raise RuntimeError(
                    f"Index/text out of sync: sentinel={int(sentinel_before)} vs file_size={current_end}"
                )

            # Prepare offsets and text bytes
            next_offset = current_end
            offsets = np.empty(len(batch), dtype=np.uint64)
            text_chunks: list[bytes] = []

            for i, s in enumerate(batch):
                offsets[i] = next_offset
                b = (s + "\n").encode("utf-8")
                text_chunks.append(b)
                next_offset += len(b)

            # 1) Overwrite old sentinel with new start offsets
            fidx.seek(-_UINT64_SIZE, os.SEEK_END)
            fidx.write(offsets.tobytes())

            # 2) Append all text
            ftxt.write(b"".join(text_chunks))

            # 3) Append new sentinel (final size)
            fidx.write(_UINT64(next_offset).tobytes())

        # Refresh mmaps so this instance sees the new data
        self._refresh_mmaps()

        # If we already built a reverse map, update it incrementally
        if self._str2id is not None:
            for i, s in enumerate(batch):  # 'batch' already normalized above
                self._str2id[s] = start_id + i

        return start_id

    def append_new_lines(self, lines: Iterable[str]) -> dict[str, int]:
        """
        Append only strings not already present. Returns {string: id}.
        Uses a cached string->id map (built lazily on first call) and
        updates it incrementally after appends.
        """

        if self._str2id is None:
            self._str2id = self.build_string_to_id_map()

        new_unique: list[str] = []
        out: dict[str, int] = {}

        for s in lines:
            s_norm = self._norm(s)
            if s_norm in self._str2id:
                out[s_norm] = self._str2id[s_norm]
            else:
                self._str2id[
                    s_norm
                ] = -1  # reserved to keep order uniqueness for this call
                new_unique.append(s_norm)

        if new_unique:
            self.append_lines(new_unique)
            for s in new_unique:
                out[s] = self._str2id[s]

        return out

    def build_string_to_id_map(self) -> dict[str, int]:
        """Build a dict mapping each stored string to its 0-based id."""
        n = len(self)
        if n == 0:
            return {}

        # Ensure mmap exists
        if self._text_mmap is None:
            if self._text_path.stat().st_size == 0:
                return {}
            self._text_mmap = mmap.mmap(
                self._text_file.fileno(), 0, access=mmap.ACCESS_READ
            )

        result: dict[str, int] = {}
        starts = self._offsets_memmap[:-1]
        ends = self._offsets_memmap[1:]
        for i in range(n):
            a, b = int(starts[i]), int(ends[i])
            if b > a and self._text_mmap[b - 1 : b] == b"\n":
                raw = self._text_mmap[a : b - 1]
            else:
                raw = self._text_mmap[a:b]

            s = raw.decode("utf-8")
            if s not in result:
                result[s] = i
        return result

    def string_to_id(self, s: str) -> int | None:
        """
        Look up the id for a given string. Returns None if not present.
        Lazily builds (and then reuses) a string->id cache. The cache is
        incrementally updated by append_lines, so it stays in sync for
        appends done through this instance.
        """
        if self._str2id is None:
            self._str2id = self.build_string_to_id_map()
        return self._str2id.get(self._norm(s))

    # ---------- Internals ----------

    def verify_integrity(self, raise_on_error: bool = False) -> bool:
        """
        Public integrity check mirroring constructor validation.
        Returns True if OK, False otherwise (or raises if raise_on_error is True).
        """
        try:
            self._verify_integrity_or_raise()
            return True
        except Exception:
            if raise_on_error:
                raise
            return False

    def _verify_integrity_or_raise(self) -> None:
        if self._offsets_memmap.ndim != 1 or self._offsets_memmap.dtype != np.uint64:
            raise RuntimeError("Index memmap must be a 1D uint64 array.")
        # Sentinel must match text file size
        sentinel = int(self._offsets_memmap[-1]) if self._offsets_memmap.size > 0 else 0
        file_size = self._text_path.stat().st_size
        if sentinel != file_size:
            raise RuntimeError(
                f"Index sentinel ({sentinel}) != text file size ({file_size})."
            )
        # Offsets non-decreasing
        if self._offsets_memmap.size >= 2:
            if np.any(np.diff(self._offsets_memmap) < 0):
                raise RuntimeError("Offsets must be non-decreasing.")

    def _refresh_mmaps(self) -> None:
        # Refresh text mmap
        try:
            if self._text_mmap is not None:
                self._text_mmap.close()
        except Exception:
            pass
        try:
            self._text_file.close()
        except Exception:
            pass

        self._text_file = open(self._text_path, "rb")
        size = self._text_path.stat().st_size
        self._text_mmap = (
            mmap.mmap(self._text_file.fileno(), 0, access=mmap.ACCESS_READ)
            if size > 0
            else None
        )

        # Refresh offsets memmap
        del self._offsets_memmap
        self._offsets_memmap = np.memmap(self._index_path, dtype=np.uint64, mode="r")

        self._verify_integrity_or_raise()
