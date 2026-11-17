"""Pinned source files: a missing one is fetched only when allowed, an
existing one is never fetched again, and every one is checked against its pin."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from remedi_data.format import sha256_of_file
from remedi_data.preparation.smiles import (
    PinnedFile,
    PinnedFileError,
    ensure_pinned_file,
)


def _origin(tmp_path: Path) -> tuple[Path, str]:
    origin = tmp_path / "origin"
    origin.mkdir()
    (origin / "toy.csv").write_text("smiles\nCCO\n")
    return origin, hashlib.sha256(b"smiles\nCCO\n").hexdigest()


def test_sha256_of_file_is_the_sha256_of_the_bytes(tmp_path: Path) -> None:
    path = tmp_path / "bytes.bin"
    path.write_bytes(bytes(range(256)) * 5000)
    assert sha256_of_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_an_existing_file_is_checked_against_its_pin(tmp_path: Path) -> None:
    origin, pin = _origin(tmp_path)
    assert (
        ensure_pinned_file(
            PinnedFile(file_name="toy.csv", sha256=pin), origin, download=False
        ).sha256
        == pin
    )
    with pytest.raises(PinnedFileError, match="is pinned"):
        ensure_pinned_file(
            PinnedFile(file_name="toy.csv", sha256="0" * 64), origin, download=False
        )


def test_a_missing_file_is_fetched_only_when_allowed(tmp_path: Path) -> None:
    origin, pin = _origin(tmp_path)
    pinned = PinnedFile(
        file_name="toy.csv", sha256=pin, url=(origin / "toy.csv").as_uri()
    )
    raw = tmp_path / "raw"
    with pytest.raises(PinnedFileError, match="enable the download"):
        ensure_pinned_file(pinned, raw, download=False)
    assert ensure_pinned_file(pinned, raw, download=True).sha256 == pin
    with pytest.raises(PinnedFileError, match="placed by hand"):
        ensure_pinned_file(
            PinnedFile(file_name="other.csv", sha256=pin), raw, download=True
        )


def test_an_existing_file_is_never_fetched_again(tmp_path: Path) -> None:
    origin, pin = _origin(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "toy.csv").write_text("kept as is\n")
    pinned = PinnedFile(
        file_name="toy.csv", sha256=pin, url=(origin / "toy.csv").as_uri()
    )
    with pytest.raises(PinnedFileError, match="is pinned"):
        ensure_pinned_file(pinned, raw, download=True)
    assert (raw / "toy.csv").read_text() == "kept as is\n"


def test_a_failed_download_leaves_nothing(tmp_path: Path) -> None:
    origin, pin = _origin(tmp_path)
    raw = tmp_path / "raw"
    pinned = PinnedFile(
        file_name="absent.csv", sha256=pin, url=(origin / "absent.csv").as_uri()
    )
    with pytest.raises(PinnedFileError, match="downloading"):
        ensure_pinned_file(pinned, raw, download=True)
    assert list(raw.iterdir()) == []
