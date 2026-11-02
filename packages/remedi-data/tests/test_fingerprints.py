"""Packed Morgan fingerprints agree with RDKit's own bit vectors and Tanimoto."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from remedi_data.chemistry.fingerprints import (
    FingerprintError,
    packed_morgan_fingerprints,
    tanimoto_similarity,
)

SMILES = ["c1ccccc1O", "c1ccccc1N", "CCO", "CC(=O)Oc1ccccc1C(=O)O", "C1CCCCC1"]


def test_packed_bits_are_rdkit_bits_in_little_endian_words() -> None:
    packed = packed_morgan_fingerprints(SMILES, radius=2, size=2048)
    assert packed.shape == (len(SMILES), 64)
    assert packed.dtype == np.uint32
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for row, text in enumerate(SMILES):
        on_bits = list(generator.GetFingerprint(Chem.MolFromSmiles(text)).GetOnBits())
        unpacked = np.flatnonzero(
            (packed[row][:, None] >> np.arange(32, dtype=np.uint32)) & 1
        )
        assert unpacked.tolist() == on_bits


def test_tanimoto_matches_rdkit() -> None:
    packed = packed_morgan_fingerprints(SMILES)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    vectors = [generator.GetFingerprint(Chem.MolFromSmiles(text)) for text in SMILES]
    for left in range(len(SMILES)):
        for right in range(len(SMILES)):
            assert tanimoto_similarity(packed[left], packed[right]) == pytest.approx(
                DataStructs.TanimotoSimilarity(vectors[left], vectors[right])
            )


def test_a_process_pool_gives_the_same_fingerprints(monkeypatch) -> None:
    import remedi_data.chemistry.fingerprints as fingerprints

    monkeypatch.setattr(fingerprints, "_CHUNK", 2)
    np.testing.assert_array_equal(
        packed_morgan_fingerprints(SMILES, workers=2),
        packed_morgan_fingerprints(SMILES),
    )


def test_an_unparsable_smiles_is_an_error() -> None:
    with pytest.raises(FingerprintError, match="cannot parse"):
        packed_morgan_fingerprints(["C1CC"])
