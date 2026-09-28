"""Reading the source pickles into (smiles, class, mol) triples."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

from rdkit import Chem

from .chemistry import load_mols
from .config import SourceConfig, TypedSource

SourceRecord = tuple[str, str, Chem.Mol]


class ChecksumMismatchError(ValueError):
    """A pinned input file does not have the sha256 the config expects."""


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str | None) -> None:
    """Raise ``ChecksumMismatchError`` unless ``path`` hashes to ``expected``.

    ``expected=None`` skips the check, for configs that do not pin a file.
    """
    if expected is None:
        return
    observed = sha256_of_file(path)
    if observed != expected:
        raise ChecksumMismatchError(
            f"{path}: sha256 {observed} does not match the pinned {expected}"
        )


def iter_source(source: SourceConfig, data_dir: Path) -> list[SourceRecord]:
    """Yield ``(raw_smiles, raw_class, mol)`` triples for one source file.

    Raises ``FileNotFoundError`` if the pickle is missing, ``ChecksumMismatchError``
    if it differs from the pinned sha256 and ``ValueError`` if its parallel
    arrays disagree in length, so a malformed source fails the run
    rather than silently truncating the dataset.
    """
    path = data_dir / source.path
    if not path.is_file():
        raise FileNotFoundError(f"Source file not found: {path}")
    verify_sha256(path, source.sha256)
    with open(path, "rb") as handle:
        data = pickle.load(handle)

    for key in (source.smiles_key, source.mol_key):
        if key not in data:
            raise KeyError(f"{source.path}: missing key {key!r}")

    smiles_list = data[source.smiles_key]
    mols = load_mols(data[source.mol_key])
    if len(mols) != len(smiles_list):
        raise ValueError(
            f"{source.path}: {len(mols)} mols vs {len(smiles_list)} SMILES "
            "(length mismatch)"
        )

    if isinstance(source, TypedSource):
        if source.type_key not in data:
            raise KeyError(f"{source.path}: missing key {source.type_key!r}")
        types = data[source.type_key]
        if len(types) != len(smiles_list):
            raise ValueError(
                f"{source.path}: {len(types)} chiral types vs {len(smiles_list)} "
                "SMILES (length mismatch)"
            )
        return list(zip(smiles_list, types, mols, strict=True))
    return [
        (smiles, source.label, mol) for smiles, mol in zip(smiles_list, mols, strict=True)
    ]
