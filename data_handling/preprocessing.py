

from ase.optimize import BFGSLineSearch, BFGS
from rdkit2ase import rdkit2ase

import rdkit.Chem as Chem
from rdkit.Chem import AllChem

from ase import Atoms
import math

from model.model import TransformerEncoder
import torch
from data_handling.SmilesIterator import SmilesIterator


def get_mace_descriptors(atoms: Atoms,calculator,BFGS_tol= 0.05,max_steps=100):
    atoms.calc = calculator
    dyn = BFGS(atoms,logfile=None)
    converged = dyn.run(fmax=BFGS_tol,steps=max_steps)
    if not converged:
        raise ValueError("BFGS did not converge")
    
    descriptors = calculator.get_descriptors(atoms)
    return descriptors


def get_ase_atoms(smiles) -> Atoms:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, useBasicKnowledge=True, useExpTorsionAnglePrefs=True, randomSeed=-1)
    atoms = rdkit2ase(mol)
    return atoms


def get_max_molecule_size(smiles_iterator: SmilesIterator, max_num_molecules = math.inf) -> int:
    max_atoms = 0
    for i, smiles in enumerate(smiles_iterator):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        num_atoms = mol.GetNumAtoms()
        if num_atoms > max_atoms:
            max_atoms = num_atoms
        if i >= max_num_molecules:
            break
    return max_atoms


def get_atom_species_in_smiles(smiles_iterator: SmilesIterator):
    atom_species_set = set("H")
    for smiles in smiles_iterator:
        mol = Chem.MolFromSmiles(smiles)
        atom_species_set.update([atom.GetSymbol() for atom in mol.GetAtoms()])
    return atom_species_set


def get_global_descriptor(smiles : str, encoder: TransformerEncoder,calculator):
    # TODO: Maybe check SMILES validity?
    atoms = get_ase_atoms(smiles)
    mace_des = get_mace_descriptors(atoms,calculator)
    mace_des = torch.tensor(mace_des).unsqueeze(0).float()
    encoder.eval()
    with torch.no_grad():
        global_descriptor = encoder(mace_des)

    return global_descriptor