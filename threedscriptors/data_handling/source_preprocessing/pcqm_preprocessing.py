from pathlib import Path
from random import shuffle

from ase import Atoms
from rdkit import Chem
from rdkit.Chem import Mol
from tqdm import tqdm

from threedscriptors.data_handling.mol_id import StructureID

MACE_OFF_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}


def load_pcqm(
    pcqm_file: Path, N_molecules: int
) -> tuple[list[str], list[Atoms], list[StructureID]]:
    mols = filter_mols(pcqm_file, N_molecules)

    # convert mols to atoms

    molecules = convert_to_ase(mols)

    smiles = get_canon_smiles(mols)

    structure_ids = [
        StructureID(sid, canonical_smiles=smi, molecule_id=sid, conformer_id=0)
        for sid, smi in enumerate(smiles)
    ]

    return smiles, molecules, structure_ids


def get_canon_smiles(mols: list[Mol]):
    return [Chem.MolToSmiles(mol) for mol in mols]


def convert_to_ase(mols):
    all_atoms = []

    for mol in mols:
        try:
            symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
            positions = mol.GetConformer().GetPositions()

            all_atoms.append(
                Atoms(
                    symbols=symbols,
                    positions=positions,
                    info={"smiles": Chem.MolToSmiles(mol)},
                )
            )

        except Exception as e:
            print(e)
            continue

    return all_atoms


def filter_mols(pcqm_file: Path, N_max: int):
    mols = []
    suppl = Chem.SDMolSupplier(pcqm_file, removeHs=False)

    indices = list(range(len(suppl)))
    shuffle(indices)
    for idx in tqdm(indices):
        mol = suppl[idx]
        try:
            if mol is None:
                continue

            if len(Chem.GetMolFrags(mol, asMols=True)) > 1:
                continue

            if any(a.GetSymbol() not in MACE_OFF_ELEMENTS for a in mol.GetAtoms()):
                continue

            if any(
                a.GetNumRadicalElectrons() != 0
                or a.GetIsotope() != 0
                or a.GetFormalCharge() != 0
                for a in mol.GetAtoms()
            ):
                continue
        except Exception as e:
            print(f"Failed reading {Chem.MolToSmiles(mol)}: {e}")
            continue

        mols.append(mol)
        if len(mols) >= N_max:
            break

    return mols
