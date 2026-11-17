"""Morgan fingerprints: RDKit's bits in every layout, and the same bits as
bblean's ``ecfp4``/``ecfp6`` (molfeat's ECFP, used before, matched bit for bit
until it left the workspace)."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from remedi_data.chemistry.fingerprints import (
    FingerprintError,
    morgan_fingerprints,
    tanimoto_similarity,
)

SMILES = ["c1ccccc1O", "c1ccccc1N", "CCO", "CC(=O)Oc1ccccc1C(=O)O", "C1CCCCC1"]

#: Stereo (tetrahedral, double bond, sulfoxide), charges, salts, isotopes,
#: metals, boron, silicon and phosphorus: where fingerprint definitions differ.
VARIED_SMILES = [
    "CCO",
    "c1ccccc1O",
    "C[C@H](N)C(=O)O",
    "C[C@@H](N)C(=O)O",
    "F/C=C/F",
    "F/C=C\\F",
    "[NH4+].[Cl-]",
    "CC(=O)[O-].[Na+]",
    "C[N+](C)(C)C",
    "O=C([O-])c1ccccc1",
    "[O-][N+](=O)c1ccccc1",
    "CC1=CC(=O)C=CC1=O",
    "N#Cc1ccncc1",
    "C1CC2CCC1C2",
    "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
    "c1ccc2c(c1)[nH]c1ccccc12",
    "[2H]C([2H])([2H])O",
    "Clc1ccc(Br)cc1I",
    "CS(=O)(=O)N",
    "C1=CC=CN=C1.Cl",
    "[Fe+2]",
    "[Cu+].[Cl-]",
    "CC(C)(C)OC(=O)N[C@@H](Cc1ccccc1)C(=O)O",
    "C[S@](=O)c1ccccc1",
    "C[C@]12CC[C@H]3[C@@H](CC=C4C[C@@H](O)CC[C@@]43C)[C@@H]1CC[C@@H]2O",
    "O=C1NC(=O)C(=O)N1",
    "B(O)(O)c1ccccc1",
    "P(=O)(O)(O)O",
    "[Si](C)(C)(C)C",
    "C#C",
    "c1ccsc1",
    "C1CCCCC1",
    "CC[N-]CC.[Li+]",
    "[O-]c1ccccc1.[K+]",
]


def _rdkit_bits(smiles: list[str], radius: int, size: int) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=size)
    return np.stack(
        [generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(text)) for text in smiles]
    ).astype(np.uint8)


def test_packed_words_are_rdkit_bits_in_little_endian_words() -> None:
    packed = morgan_fingerprints(SMILES, layout="packed_words", radius=2, size=2048)
    assert packed.shape == (len(SMILES), 64)
    assert packed.dtype == np.uint32
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for row, text in enumerate(SMILES):
        on_bits = list(generator.GetFingerprint(Chem.MolFromSmiles(text)).GetOnBits())
        unpacked = np.flatnonzero(
            (packed[row][:, None] >> np.arange(32, dtype=np.uint32)) & 1
        )
        assert unpacked.tolist() == on_bits


@pytest.mark.parametrize("radius", [2, 3])
def test_every_layout_stores_the_same_bits(radius: int) -> None:
    bits = _rdkit_bits(VARIED_SMILES, radius, 1024)
    dense = morgan_fingerprints(VARIED_SMILES, layout="dense", radius=radius, size=1024)
    assert dense.dtype == np.uint8
    np.testing.assert_array_equal(dense, bits)
    packed_bytes = morgan_fingerprints(
        VARIED_SMILES, layout="packed_bytes", radius=radius, size=1024
    )
    np.testing.assert_array_equal(np.unpackbits(packed_bytes, axis=1), bits)
    packed_words = morgan_fingerprints(
        VARIED_SMILES, layout="packed_words", radius=radius, size=1024
    )
    np.testing.assert_array_equal(
        np.unpackbits(packed_words.view(np.uint8), axis=1, bitorder="little"), bits
    )


@pytest.mark.parametrize(("kind", "radius"), [("ecfp4", 2), ("ecfp6", 3)])
def test_packed_bytes_are_bbleans_fingerprints(kind: str, radius: int) -> None:
    bblean = pytest.importorskip("bblean")
    reference = bblean.fps_from_smiles(
        VARIED_SMILES, kind=kind, n_features=2048, pack=True
    )
    ours = morgan_fingerprints(
        VARIED_SMILES, layout="packed_bytes", radius=radius, size=2048
    )
    np.testing.assert_array_equal(ours, reference)


def test_tanimoto_matches_rdkit_in_both_packed_layouts() -> None:
    words = morgan_fingerprints(SMILES, layout="packed_words")
    packed_bytes = morgan_fingerprints(SMILES, layout="packed_bytes")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    vectors = [generator.GetFingerprint(Chem.MolFromSmiles(text)) for text in SMILES]
    for left in range(len(SMILES)):
        for right in range(len(SMILES)):
            expected = DataStructs.TanimotoSimilarity(vectors[left], vectors[right])
            assert tanimoto_similarity(words[left], words[right]) == pytest.approx(
                expected
            )
            assert tanimoto_similarity(
                packed_bytes[left], packed_bytes[right]
            ) == pytest.approx(expected)


def test_a_process_pool_gives_the_same_fingerprints(monkeypatch) -> None:
    import remedi_data.chemistry.fingerprints as fingerprints

    monkeypatch.setattr(fingerprints, "_CHUNK", 2)
    np.testing.assert_array_equal(
        morgan_fingerprints(SMILES, layout="packed_bytes", workers=2),
        morgan_fingerprints(SMILES, layout="packed_bytes"),
    )


def test_no_smiles_gives_an_empty_matrix_of_the_layouts_width() -> None:
    assert morgan_fingerprints([], layout="packed_words", size=64).shape == (0, 2)
    assert morgan_fingerprints([], layout="packed_bytes", size=64).shape == (0, 8)
    assert morgan_fingerprints([], layout="dense", size=64).shape == (0, 64)


def test_an_unparsable_smiles_is_an_error() -> None:
    with pytest.raises(FingerprintError, match="cannot parse"):
        morgan_fingerprints(["C1CC"], layout="dense")


def test_a_size_that_does_not_fill_the_layout_is_refused() -> None:
    with pytest.raises(ValueError, match="multiple of 32"):
        morgan_fingerprints(SMILES, layout="packed_words", size=100)
