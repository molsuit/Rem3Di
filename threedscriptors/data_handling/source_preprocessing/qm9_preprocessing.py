import random
import re
from enum import IntEnum
from pathlib import Path

import numpy as np
from ase import Atoms
from rdkit import Chem
from tqdm import tqdm

from threedscriptors.configuration.data_config import TaskConfig
from threedscriptors.data_handling.mol_id import StructureID


class QM9PropertyNames(IntEnum):
    A              = 0  # rotational constant [GHz]
    B              = 1  # rotational constant [GHz]
    C              = 2  # rotational constant [GHz]
    mu             = 3  # dipole moment [D]
    alpha          = 4  # isotropic polarizability [a0^3]
    epsilon_HOMO   = 5  # orbital energy [Ha]
    epsilon_LUMO   = 6  # orbital energy [Ha]
    gap            = 7  # HOMO–LUMO gap [Ha]
    r2             = 8  # ⟨R²⟩ [a0²]
    zpve           = 9  # zero‑point vibrational energy [Ha]
    U0             = 10 # internal energy at 0 K [Ha]
    U              = 11 # internal energy [Ha]
    H              = 12 # enthalpy [Ha]
    G              = 13 # Gibbs free energy [Ha]
    Cv             = 14 # heat capacity [cal·mol⁻¹·K⁻¹]



_scientific = re.compile(r'([+-]?\d*\.?\d*)\*\^([+-]?\d+)')   # x*^y → xey
def _safe_float(tok: str) -> float:
    """
    Converts '2.1997*^-6', '1.23D+03', plain '0.12', etc. to float.
    """
    tok = tok.replace('D', 'e').replace('d', 'e')
    tok = _scientific.sub(r'\1e\2', tok)      # Mathematica *^ exponent
    return float(tok)



def parse_qm9_xyz(path: Path):
    """
    Parse a single QM9 .xyz file.

    Returns
    -------
    atoms : ase.Atoms
        Cartesian geometry (Å); Mulliken charges are stored in
        `atoms.arrays["initial_charges"]` if present.
    props : np.ndarray, shape=(15,)
        The 15 floating-point properties listed in PROPERTY_NAMES.
    smiles : str
        SMILES string on the line right after the vibrational frequencies.
    """
    with path.open() as fh:
        n_atoms   = int(fh.readline())            # line 1
        info_line = fh.readline().split()         # line 2

        tag, gdb_idx = info_line[:2]
        props = np.array([_safe_float(t) for t in info_line[2:]], dtype=float)

        symbols, coords = [], []
        for _ in range(n_atoms):                  # atom lines
            s, x, y, z, q = fh.readline().split()
            symbols.append(s)
            coords.append([_safe_float(x), _safe_float(y), _safe_float(z)])

        fh.readline()                   # vibrational frequencies
        smiles = fh.readline().split()[0]
        smiles = Chem.CanonSmiles(smiles, useChiral=False)



    atoms = Atoms(symbols=symbols, positions=np.asarray(coords), info = {"smiles" : smiles})
    atoms.info["gdb_index"] = int(gdb_idx)

    return atoms, props, smiles




def load_qm9(qm9_dir: Path, N_molecules: int | None = None, tasks_to_load = list[QM9PropertyNames] | None, shuffle: bool = True):



    xyz_files = sorted(qm9_dir.glob("**/*.xyz"))

    if shuffle:
        random.shuffle(xyz_files)


    if N_molecules is not None:
        xyz_files = xyz_files[:N_molecules]                # limit to N_molecules

    all_props = []
    all_smiles = []
    molecules : list[Atoms] = []


    for f in tqdm(xyz_files):
        atoms, props, smiles = parse_qm9_xyz(f)
        if any([a.GetFormalCharge() != 0 for a in Chem.MolFromSmiles(smiles).GetAtoms()]):
            continue

        molecules.append(atoms)
        all_props.append(props)
        all_smiles.append(smiles)


    print(all_props)
    regression_targets  = np.stack(all_props)
    print(regression_targets.shape)

    if tasks_to_load is not None:
        task_col_indices = [t.value for t in tasks_to_load]

        regression_targets= regression_targets[:,task_col_indices]
        task_names = [t.name for t in tasks_to_load]

    else:
        task_names = [p.name for p in QM9PropertyNames]

    print(task_names)

    regression_masks = np.ones_like(regression_targets, dtype=bool)



    task_configs = [TaskConfig(task_name=task_name) for task_name in task_names]

    structure_ids = [StructureID(structure_id= idx, molecule_id=idx, smiles_id=idx, conformer_id=0, enantiomer_id= 0, canonical_smiles=smi) for idx, smi in enumerate(all_smiles)]

    return all_smiles, molecules, structure_ids, regression_targets, regression_masks, task_configs
