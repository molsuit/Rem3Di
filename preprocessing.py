

from ase.optimize import BFGS
from rdkit2ase import rdkit2ase

import rdkit.Chem as Chem
from rdkit.Chem import AllChem

from ase import Atoms
import math

from model.model import TransformerEncoder
import torch


def get_mace_descriptors(smiles,calculator,BFGS_tol= 0.05):
    atoms = get_ase_atoms(smiles)
    atoms.calc = calculator
    dyn = BFGS(atoms,logfile=None)
    dyn.run(fmax=BFGS_tol)
    descriptors = calculator.get_descriptors(atoms)
    return descriptors


def get_ase_atoms(smiles) -> Atoms:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, useBasicKnowledge=True, useExpTorsionAnglePrefs=True, randomSeed=-1)
    atoms = rdkit2ase(mol)
    return atoms


def get_max_molecule_size(smiles_path: str, max_num_molecules = math.inf) -> int:
    max_atoms = 0
    
    with open(smiles_path, "r") as f:
        for i, smiles in enumerate(f):
            smiles = f.readline()
            smiles = smiles[:-1] # Remove newline character
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            num_atoms = mol.GetNumAtoms()
            if num_atoms > max_atoms:
                max_atoms = num_atoms

            if i >= max_num_molecules:
                break

    return max_atoms


def get_global_descriptor(smiles : str, encoder: TransformerEncoder,calculator):
    # TODO: Maybe check SMILES validity?
    mace_des = get_mace_descriptors(smiles,calculator)
    mace_des = torch.tensor(mace_des).unsqueeze(0).float()
    encoder.eval()
    with torch.no_grad():
        global_descriptor = encoder(mace_des)

    return global_descriptor