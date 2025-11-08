import pytest
from threedscriptors.data_handling.mol_id import StructureID


@pytest.mark.parametrize(
    "instance",
    [
        # no enantiomer_id
        StructureID(
            structure_id=1,
            canonical_smiles="CC",
            molecule_id=99,
            conformer_id=0,
            enantiomer_id=None,
        ),
        # with enantiomer_id
        StructureID(
            structure_id=2,
            canonical_smiles="C[C@H](F)O",
            molecule_id=100,
            conformer_id=3,
            enantiomer_id=1,
        ),
    ],
)
def test_roundtrip(instance: StructureID):
    """
    1. Serialise the dataclass to a human-readable ID string.
    2. Deserialise it back with from_id_string.
    3. The reconstructed object must equal the original.
    """
    id_str = instance.to_id_string()
    print(id_str)
    rebuilt = StructureID.from_id_string(id_str, smiles=instance.canonical_smiles)

    # Using == works because dataclasses generate an __eq__ that
    # compares every field.
    assert rebuilt == instance, f"Round-trip failed for {instance}"
