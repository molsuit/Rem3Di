import json
import os
import pickle
from pathlib import Path

import numpy as np
from ase import Atoms
from rdkit import Chem
from tqdm import tqdm

from threedscriptors.data_handling.mol_id import StructureID

MACE_OFF_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}



def get_all_mol_paths(geom_dir):

    drugs_file = os.path.join(geom_dir, "rdkit_folder/summary_drugs.json")
    with open(drugs_file) as f:
        drugs_summ = json.load(f)


    rdkit_dir = Path(geom_dir) / "rdkit_folder/"
    existing = {
        str(p.relative_to(rdkit_dir)): p  # key: "drug123/drug123.pkl"
        for p in rdkit_dir.rglob("*.pickle")  # finds files in sub‑directories too
    }

    mol_paths = [
        existing[path]
        for sub in drugs_summ.values()
        if (path := sub.get("pickle_path")) in existing
    ]

    return mol_paths

def load_geom(
    geom_dir: str,
    boltzmann_weight_threshold: float,
    N_molecules: int,
    max_atoms: int = 100,
) -> tuple[list[str], list[Atoms], list[StructureID]]:

    mol_paths = get_all_mol_paths(geom_dir)


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

            molecules.append(mol_to_ase(conf["rd_mol"], can_smi=can_smiles))
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

    print(
        f"Got to molId {mol_id} with {running_mol_id} = filter % {running_mol_id/mol_id} "
    )

    return smiles, molecules, structure_ids


def mol_to_ase(mol: Chem.Mol, can_smi: str):

    numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
    conf = mol.GetConformer()

    atoms = Atoms(
        numbers=numbers, positions=conf.GetPositions(), info={"smiles": can_smi}
    )
    return atoms


from concurrent.futures import ProcessPoolExecutor, as_completed

from ase import Atoms
from rdkit import Chem

# assume these exist in your module
# from your_module import StructureID, MACE_OFF_ELEMENTS


def _process_one_file(args):
    """
    Worker: read one pickle, run all filters, and return a list of tuples:
    (mol_id, conf_id, canonical_smiles, atomic_numbers (list[int]), positions (ndarray))
    """
    mol_id, mol_path, boltzmann_weight_threshold, max_atoms = args

    with open(mol_path, "rb") as f:
        dic = pickle.load(f)

    # quick rejects
    if dic.get("charge", 0) != 0:
        return []

    mol = Chem.AddHs(Chem.MolFromSmiles(dic["smiles"]))
    if len(Chem.GetMolFrags(mol, asMols=True)) > 1:
        return []
    if mol.GetNumAtoms() > max_atoms:
        return []
    if any(a.GetSymbol() not in MACE_OFF_ELEMENTS for a in mol.GetAtoms()):
        return []
    if any(
        a.GetNumRadicalElectrons() != 0
        or a.GetIsotope() != 0
        or a.GetFormalCharge() != 0
        for a in mol.GetAtoms()
    ):
        return []

    conformers = [
        c
        for c in dic["conformers"]
        if c.get("boltzmannweight", 0.0) > boltzmann_weight_threshold
    ]
    if not conformers:
        return []

    can_smiles = Chem.CanonSmiles(dic["smiles"])

    results = []
    for conf_id, conf in enumerate(conformers):
        rd = conf["rd_mol"]  # RDKit Mol with a conformer
        nums = [a.GetAtomicNum() for a in rd.GetAtoms()]
        pos = np.asarray(rd.GetConformer().GetPositions(), dtype=float)
        results.append((mol_id, conf_id, can_smiles, nums, pos))

    return results





def load_geom_parallel(
    geom_dir: str,
    boltzmann_weight_threshold: float,
    N_structures: int | None,
    max_atoms: int = 100,
    max_workers: int | None = None,
) -> tuple[list[str], list[Atoms], list["StructureID"]]:

    # 1) read summary and collect existing pickle paths (single-threaded)

    mol_paths = get_all_mol_paths(geom_dir)

    print("All mol paths")
    # 2) dispatch work
    args = [
        (i, p, boltzmann_weight_threshold, max_atoms) for i, p in enumerate(mol_paths)
    ]

    raw_results = []  # list of (mol_id, conf_id, can_smiles, nums, pos)
    submitted = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_one_file, a) for a in args]
        submitted = len(futures)

        for fut in tqdm(as_completed(futures)):
            chunk = fut.result()
            if chunk:
                raw_results.extend(chunk)
            # soft early stop once we have enough structures

            if len(raw_results) >= N_structures:
                # cancel pending tasks (those not yet running)
                for f in futures:
                    f.cancel()
                break

    # 3) make output deterministic and build final objects
    raw_results.sort(key=lambda t: (t[0], t[1]))  # (molecule_id, conformer_id)

    smiles: list[str] = []
    molecules: list[Atoms] = []
    structure_ids: list[StructureID] = []

    for structure_idx, (mol_id, conf_id, can_smi, nums, pos) in enumerate(
        raw_results[:N_structures]
    ):
        atoms = Atoms(numbers=nums, positions=pos, info={"smiles": can_smi})
        molecules.append(atoms)
        structure_ids.append(
            StructureID(
                structure_id=structure_idx,
                conformer_id=conf_id,
                molecule_id=mol_id,
                canonical_smiles=can_smi,
            )
        )
        smiles.append(can_smi)

    kept_mols = len({mid for (mid, *_rest) in raw_results[:N_structures]})
    print(
        f"Processed {submitted} files; built {len(smiles)} structures from {kept_mols} molecules."
    )

    return smiles, molecules, structure_ids
