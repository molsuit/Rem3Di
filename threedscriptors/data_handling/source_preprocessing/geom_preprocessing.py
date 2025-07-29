from ase import Atoms
import numpy as np
import os
import json
import random
import pickle
from rdkit import Chem
from pathlib import Path
from threedscriptors.data_handling.mol_id import StructureID
import yaml

from typing import Tuple, List
MACE_OFF_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}


def load_geom(geom_dir: str, boltzmann_weight_threshold: float, N_molecules: int, max_atoms : int= 100) -> Tuple[List[str], List[Atoms], List[StructureID]]:
    
    drugs_file = os.path.join(geom_dir, "rdkit_folder/summary_drugs.json")
    with open(drugs_file, "r") as f:
        drugs_summ = json.load(f)
    print("Loaded json")    

    mol_paths = []
    for smiles, sub_dic in drugs_summ.items():
        pickle_path = os.path.join(
            geom_dir, "rdkit_folder", sub_dic.get("pickle_path", "")
        )
        if os.path.isfile(pickle_path):
            mol_paths.append(pickle_path)

    if N_molecules is None:
        N_molecules = len(mol_paths)

    structure_ids = []
    molecules = []
    smiles = []
    running_structure_id = 0
    running_mol_id = 0

    for mol_id, mol_path in enumerate(mol_paths):
        with open(mol_path, "rb") as f:
            dic = pickle.load(f)

        if dic["charge"] != 0:
            continue
        

        mol = Chem.AddHs(Chem.MolFromSmiles(dic["smiles"]))

        if len(Chem.GetMolFrags(mol, asMols=True)) > 1:
            continue
        
        if mol.GetNumAtoms() > max_atoms:
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

        conformers = [
                c
                for c in dic["conformers"]
                if c["boltzmannweight"] > boltzmann_weight_threshold
            ]
            # Get all conformers with a boltzmann weight larger then 0.2
        if len(conformers) == 0:
            continue

        can_smiles = Chem.CanonSmiles(dic["smiles"])

        for conf_id, conf in enumerate(conformers):
        
            molecules.append(mol_to_ase(conf["rd_mol"], can_smi= can_smiles))
            structure_ids.append(
                StructureID(
                    structure_id=running_structure_id,
                    conformer_id=conf_id,
                    molecule_id=mol_id,
                    canonical_smiles=can_smiles,
                )
            )
            smiles.append(can_smiles)
            running_structure_id += 1
        running_mol_id += 1


        if len(smiles) > N_molecules:
            break


    print(f"Got to molId {mol_id} with {running_mol_id} = filter % {running_mol_id/mol_id} ")

    return smiles, molecules, structure_ids


def mol_to_ase(mol: Chem.Mol, can_smi : str):

    numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
    conf = mol.GetConformer()
    
    atoms = Atoms(numbers=numbers, positions=conf.GetPositions(), info = {"smiles" : can_smi})
    return atoms
