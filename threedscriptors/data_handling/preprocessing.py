import math

import rdkit.Chem as Chem
import torch
from ase import Atoms
from ase.optimize import LBFGS
from mace.calculators import MACECalculator
from rdkit.Chem import AllChem
from rdkit.Chem.rdDistGeom import EmbedMultipleConfs
from rdkit2ase import rdkit2ase

from threedscriptors.configuration.data_config import DatasetConfig
from threedscriptors.data_handling.smiles_iterator import SmilesIterator
from threedscriptors.model.transformer_components import TransformerEncoder


def relax_atoms(atoms: Atoms, calculator: MACECalculator, BFGS_tol=0.05, max_steps=100):
    atoms.calc = calculator
    dyn = LBFGS(atoms, logfile=None)
    converged = dyn.run(fmax=BFGS_tol, steps=max_steps)
    if not converged:
        raise ValueError("LBFGS did not converge")


def get_ase_atoms(smiles) -> Atoms:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    returncode = AllChem.EmbedMolecule(
        mol, useBasicKnowledge=True, useExpTorsionAnglePrefs=True, randomSeed=-1
    )

    if returncode == -1:
        AllChem.EmbedMolecule(
            mol,
            useRandomCoords=True,
            randomSeed=-1,
        )

    atoms = rdkit2ase(mol)
    return atoms


def get_ase_atoms_with_conformers(smiles, N_conformers: int) -> list[Atoms]:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    EmbedMultipleConfs(
        mol, numConfs=N_conformers, numThreads=N_conformers, maxAttempts=5000
    )

    confs = [
        Atoms(
            positions=conf.GetPositions(),
            numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
        )
        for conf in mol.GetConformers()
    ]

    # if mol.GetNumConformers() != N_conformers:
    #    print("Failed Embedding Multi Confs, trying again with random coords")
    #
    #    EmbedMultipleConfs(mol, numConfs=N_conformers, numThreads=N_conformers,maxAttempts=100000, useRandomCoords= True, forceTol=1)

    return confs


def get_relaxed_conformers(
    smiles,
    mace_calculator: MACECalculator,
    dataset_config: DatasetConfig,
    N_conformers: int,
):
    if N_conformers == 1:
        confs = [get_ase_atoms(smiles)]
    else:
        confs = get_ase_atoms_with_conformers(smiles, N_conformers)

    molecules = []

    for atoms in confs:
        try:
            relax_atoms(
                atoms,
                mace_calculator,
                dataset_config.BFGS_tol,
                dataset_config.BFGS_max_steps,
            )
        except ValueError:
            print("Molecule did not relax.")
            continue
        else:
            molecules.append(atoms)
    return molecules


def get_max_molecule_size(
    smiles_iterator: SmilesIterator, max_num_molecules=math.inf
) -> int:
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


def get_global_descriptor(
    smiles: str, encoder: TransformerEncoder, calculator: MACECalculator
):
    # TODO: Maybe check SMILES validity?
    atoms = get_ase_atoms(smiles)
    mace_des = calculator.get_descriptors(atoms, invariants_only=True)
    mace_des = torch.tensor(mace_des).unsqueeze(0).float()
    encoder.eval()
    with torch.no_grad():
        global_descriptor = encoder(mace_des)

    return global_descriptor
