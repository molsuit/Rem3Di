"""``structures.atp``: the streaming writer, the memory-mapped reader and the hash."""

from __future__ import annotations

import pickle
import tracemalloc
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from remedi_data.dataset import (
    StructureFile,
    Structures,
    structures_sha256,
    write_structures,
)
from remedi_data.dataset.structure_file import partial_path


def random_structures(
    count: int, seed: int, atoms: tuple[int, int] = (1, 40)
) -> Structures:
    """``count`` structures with random atom counts in ``atoms``, random frames."""
    generator = np.random.default_rng(seed)
    counts = generator.integers(*atoms, size=count)
    pointer = np.concatenate([[0], np.cumsum(counts)])
    return Structures.from_arrays(
        generator.normal(size=(int(pointer[-1]), 3)) * 3.0,
        generator.integers(1, 90, size=int(pointer[-1])),
        pointer,
    )


def parts_of(structures: Structures, part_size: int) -> list[Structures]:
    return [
        structures.slice(start, start + part_size)
        for start in range(0, len(structures), part_size)
    ]


def test_a_round_trip_is_exact(tmp_path: Path) -> None:
    structures = random_structures(200, seed=0)
    path = tmp_path / "structures.atp"
    assert write_structures(path, [structures]) == 200
    read = StructureFile.open(path).read()
    np.testing.assert_array_equal(read.ptr, structures.ptr)
    np.testing.assert_array_equal(read.atomic_numbers, structures.atomic_numbers)
    np.testing.assert_array_equal(read.positions, structures.positions)
    assert read.positions.dtype == np.float32
    assert read.atomic_numbers.dtype == np.uint8
    assert read.total_charge is None and read.multiplicity is None


def test_records_keep_their_order_across_parts_with_mixed_atom_counts(
    tmp_path: Path,
) -> None:
    # One-atom and 300-atom structures interleaved, split into uneven parts.
    structures = Structures.concatenate(
        [
            random_structures(1, seed, atoms=(1, 2) if seed % 2 else (299, 300))
            for seed in range(30)
        ]
    )
    path = tmp_path / "structures.atp"
    write_structures(
        path, [structures.slice(0, 7), structures.slice(7, 8), structures.slice(8, 30)]
    )
    structure_file = StructureFile.open(path)
    assert len(structure_file) == 30
    for index in range(30):
        numbers, positions = structure_file.frame(index)
        expected_numbers, expected_positions = structures.frame(index)
        np.testing.assert_array_equal(numbers, expected_numbers)
        np.testing.assert_array_equal(positions, expected_positions)
    np.testing.assert_array_equal(structure_file.atom_counts(), structures.atom_counts)
    chosen = np.array([0, 1, 13, 29])
    np.testing.assert_array_equal(
        structure_file.read(chosen).positions, structures.take(chosen).positions
    )
    np.testing.assert_array_equal(
        structure_file.slice(25, 40).positions, structures.slice(25, 30).positions
    )


def test_many_parts_flush_once(tmp_path: Path) -> None:
    """Atompack #45: a flush per part would leave a stale index per part."""
    structures = random_structures(2000, seed=1)
    single = tmp_path / "single.atp"
    streamed = tmp_path / "streamed.atp"
    write_structures(single, [structures])
    write_structures(streamed, parts_of(structures, 10))
    size = streamed.stat().st_size
    assert abs(size - single.stat().st_size) <= 0.02 * single.stat().st_size
    # A stale index per part would add 20 bytes per structure per flush.
    np.testing.assert_array_equal(
        StructureFile.open(streamed).read().positions, structures.positions
    )


def test_an_interrupted_write_leaves_no_file(tmp_path: Path) -> None:
    path = tmp_path / "structures.atp"

    def failing_parts() -> Iterator[Structures]:
        yield random_structures(10, seed=2)
        raise RuntimeError("the source died")

    with pytest.raises(RuntimeError, match="the source died"):
        write_structures(path, failing_parts())
    assert not path.exists()
    assert not partial_path(path).exists()


def test_a_finished_file_is_never_written_again(tmp_path: Path) -> None:
    path = tmp_path / "structures.atp"
    write_structures(path, [random_structures(5, seed=3)])
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_structures(path, [random_structures(5, seed=4)])
    assert path.read_bytes() == before


def test_a_stale_partial_file_is_replaced(tmp_path: Path) -> None:
    path = tmp_path / "structures.atp"
    partial_path(path).write_bytes(b"left by a write that died")
    write_structures(path, [random_structures(3, seed=5)])
    assert len(StructureFile.open(path)) == 3
    assert not partial_path(path).exists()


def test_streaming_holds_one_part_in_memory(tmp_path: Path) -> None:
    """Python-side peak while streaming stays near one part, not the whole set."""
    part_size, parts = 2_000, 40
    part_bytes = random_structures(part_size, seed=0).positions.nbytes * 2

    def generated_parts() -> Iterator[Structures]:
        for seed in range(parts):
            yield random_structures(part_size, seed=seed)

    tracemalloc.start()
    try:
        write_structures(tmp_path / "structures.atp", generated_parts())
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    total_bytes = part_bytes * parts
    assert peak < 6 * part_bytes < total_bytes / 4
    assert len(StructureFile.open(tmp_path / "structures.atp")) == part_size * parts


def test_the_hash_covers_frames_order_and_charges(tmp_path: Path) -> None:
    structures = random_structures(50, seed=6)
    charges, multiplicities = np.zeros(50), np.ones(50)
    path = tmp_path / "structures.atp"
    write_structures(path, parts_of(structures, 7))
    from_file = structures_sha256(StructureFile.open(path), charges, multiplicities)
    assert from_file == structures_sha256(structures, charges, multiplicities)

    order = np.arange(50)
    order[[3, 4]] = order[[4, 3]]
    swapped = structures.take(order)
    assert structures_sha256(swapped, charges, multiplicities) != from_file
    moved = Structures.from_arrays(
        structures.positions + np.float32(1e-3) * (np.arange(3) == 0),
        structures.atomic_numbers,
        structures.ptr,
    )
    assert structures_sha256(moved, charges, multiplicities) != from_file
    other_charges = charges.copy()
    other_charges[10] = 1.0
    assert structures_sha256(structures, other_charges, multiplicities) != from_file
    with pytest.raises(ValueError, match="for 50 structures"):
        structures_sha256(structures, charges[:-1], multiplicities)


def test_the_hash_is_the_documented_byte_sequence() -> None:
    import hashlib

    structures = Structures.from_frames(
        [
            (np.array([8, 1]), np.arange(6.0).reshape(2, 3)),
            (np.array([6]), np.ones((1, 3))),
        ]
    )
    expected = hashlib.sha256(
        np.array([0, 2, 3], "<i8").tobytes()
        + np.array([8, 1, 6], "u1").tobytes()
        + np.concatenate([np.arange(6.0).reshape(2, 3), np.ones((1, 3))])
        .astype("<f4")
        .tobytes()
        + np.array([0.0, -1.0], "<f4").tobytes()
        + np.array([1.0, 2.0], "<f4").tobytes()
    ).hexdigest()
    assert structures_sha256(structures, [0.0, -1.0], [1.0, 2.0]) == expected


def test_a_structure_file_pickles_as_its_path(tmp_path: Path) -> None:
    path = tmp_path / "structures.atp"
    structures = random_structures(4, seed=7)
    write_structures(path, [structures])
    reopened = pickle.loads(pickle.dumps(StructureFile.open(path)))
    assert reopened.path == path
    np.testing.assert_array_equal(reopened.read().positions, structures.positions)


def test_an_empty_structure_file(tmp_path: Path) -> None:
    path = tmp_path / "structures.atp"
    assert write_structures(path, []) == 0
    structure_file = StructureFile.open(path)
    assert len(structure_file) == 0
    assert len(structure_file.read()) == 0
    assert structure_file.atom_counts().tolist() == []
