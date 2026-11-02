"""Packed Morgan fingerprints for many SMILES, computed in a process pool.

Rows are ``uint32`` words, bit ``i`` of the fingerprint in word ``i // 32`` at
bit ``i % 32``: the layout ``nvmolkit``'s similarity and clustering kernels
take, and a compact one for anything else that needs popcount Tanimoto.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

_CHUNK = 20_000


class FingerprintError(ValueError):
    """A SMILES could not be fingerprinted."""


def _fingerprint_chunk(smiles: list[str], radius: int, size: int) -> np.ndarray:
    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=size)
    bits = np.zeros((len(smiles), size), dtype=np.uint8)
    for row, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            raise FingerprintError(f"cannot parse {text!r}")
        bits[row] = generator.GetFingerprintAsNumPy(molecule)
    packed = np.packbits(bits, axis=1, bitorder="little")
    return packed.view("<u4").astype(np.uint32, copy=False)


def packed_morgan_fingerprints(
    smiles: list[str], *, radius: int = 2, size: int = 2048, workers: int = 1
) -> np.ndarray:
    """``(len(smiles), size // 32)`` packed Morgan fingerprints.

    Raises:
        FingerprintError: if a SMILES does not parse.
        ValueError: if ``size`` is not a multiple of 32.
    """
    if size % 32:
        raise ValueError(f"fingerprint size {size} is not a multiple of 32")
    if not smiles:
        return np.zeros((0, size // 32), dtype=np.uint32)
    chunks = [smiles[start : start + _CHUNK] for start in range(0, len(smiles), _CHUNK)]
    if workers <= 1 or len(chunks) == 1:
        return np.concatenate([_fingerprint_chunk(c, radius, size) for c in chunks])
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("forkserver")
    ) as executor:
        results = executor.map(
            _fingerprint_chunk, chunks, [radius] * len(chunks), [size] * len(chunks)
        )
        return np.concatenate(list(results))


def tanimoto_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Tanimoto similarity of two packed fingerprints (1.0 for two empty ones)."""
    both = int(np.bitwise_count(left & right).sum())
    either = int(np.bitwise_count(left | right).sum())
    return 1.0 if either == 0 else both / either
