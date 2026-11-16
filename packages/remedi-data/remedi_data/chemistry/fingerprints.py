"""Morgan fingerprints for many SMILES: the one fingerprint definition.

Every fingerprint in the workspace is RDKit's ``GetMorganGenerator(radius,
fpSize)`` bit vector of the parsed (fully sanitised) SMILES, chirality off. The
ECFP descriptor, the dataset analyses (BitBIRCH, nearest-neighbour Tanimoto)
and the pretraining clustering all come through :func:`morgan_fingerprints`;
they differ only in how the bits are stored (:data:`FingerprintLayout`):

- ``dense``: ``(n, size)`` ``uint8`` of 0/1, one column per bit (the ECFP
  descriptor matrix, after a cast).
- ``packed_bytes``: ``(n, size // 8)`` ``uint8``, ``np.packbits`` with the
  most significant bit first, the layout ``bblean`` takes (``pack=True``).
- ``packed_words``: ``(n, size // 32)`` ``uint32``, bit ``i`` in word
  ``i // 32`` at bit ``i % 32``, the layout ``nvmolkit``'s similarity and
  clustering kernels take.

Tanimoto similarity counts set bits, so it is the same in every packed layout.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Literal

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

FingerprintLayout = Literal["dense", "packed_bytes", "packed_words"]

#: Bits per stored element of each layout; ``size`` must be a multiple of it.
_BITS_PER_ELEMENT: dict[str, int] = {"dense": 1, "packed_bytes": 8, "packed_words": 32}

_CHUNK = 20_000


class FingerprintError(ValueError):
    """A SMILES could not be fingerprinted."""


def _stored(bits: np.ndarray, layout: FingerprintLayout) -> np.ndarray:
    if layout == "dense":
        return bits
    if layout == "packed_bytes":
        return np.packbits(bits, axis=1)
    packed = np.packbits(bits, axis=1, bitorder="little")
    return packed.view("<u4").astype(np.uint32, copy=False)


def _fingerprint_chunk(
    smiles: list[str], radius: int, size: int, layout: FingerprintLayout
) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=size)
    bits = np.zeros((len(smiles), size), dtype=np.uint8)
    for row, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            raise FingerprintError(f"cannot parse {text!r}")
        bits[row] = generator.GetFingerprintAsNumPy(molecule)
    return _stored(bits, layout)


def morgan_fingerprints(
    smiles: list[str],
    *,
    layout: FingerprintLayout,
    radius: int = 2,
    size: int = 2048,
    workers: int = 1,
) -> np.ndarray:
    """Morgan fingerprints of ``smiles``, one row each, stored as ``layout``.

    Raises:
        FingerprintError: if a SMILES does not parse.
        ValueError: if ``size`` does not fill whole elements of ``layout``.
    """
    per_element = _BITS_PER_ELEMENT[layout]
    if size <= 0 or size % per_element:
        raise ValueError(
            f"fingerprint size {size} is not a positive multiple of {per_element} "
            f"(layout {layout})"
        )
    if not smiles:
        return _stored(np.zeros((0, size), dtype=np.uint8), layout)
    chunks = [smiles[start : start + _CHUNK] for start in range(0, len(smiles), _CHUNK)]
    if workers <= 1 or len(chunks) == 1:
        return np.concatenate(
            [_fingerprint_chunk(chunk, radius, size, layout) for chunk in chunks]
        )
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("forkserver")
    ) as executor:
        results = executor.map(
            _fingerprint_chunk,
            chunks,
            [radius] * len(chunks),
            [size] * len(chunks),
            [layout] * len(chunks),
        )
        return np.concatenate(list(results))


def tanimoto_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Tanimoto similarity of two packed fingerprints (1.0 for two empty ones)."""
    both = int(np.bitwise_count(left & right).sum())
    either = int(np.bitwise_count(left | right).sum())
    return 1.0 if either == 0 else both / either
