"""Morgan fingerprint + Tanimoto helpers for the retrieval Tanimoto task.

Kept separate from the task so the chemistry can be unit-tested without a model
or a built store.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import ExplicitBitVect


def morgan_fingerprints(
    smiles: list[str], *, radius: int = 2, n_bits: int = 2048
) -> list[ExplicitBitVect | None]:
    """Morgan (ECFP-like) fingerprints aligned to ``smiles``; ``None`` if unparseable."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps: list[ExplicitBitVect | None] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if smi else None
        fps.append(gen.GetFingerprint(mol) if mol is not None else None)
    return fps


def bulk_tanimoto(query: ExplicitBitVect, others: list[ExplicitBitVect]) -> np.ndarray:
    """Tanimoto of ``query`` against every fingerprint in ``others``."""
    return np.asarray(
        DataStructs.BulkTanimotoSimilarity(query, others), dtype=np.float64
    )


def tanimoto(a: ExplicitBitVect, b: ExplicitBitVect) -> float:
    return float(DataStructs.TanimotoSimilarity(a, b))
