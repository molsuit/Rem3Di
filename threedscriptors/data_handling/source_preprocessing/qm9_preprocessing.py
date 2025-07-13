import numpy as np
from pathlib import Path
from ase import Atoms
from threedscriptors.configuration.data_config import TaskConfig
from tqdm import tqdm
from rdkit import Chem
import re

# ----------------------------------------------------------------------
# Column labels in the order they appear after the identifier
# (taken from the original QM9 paper, cf. Table 3 in your screenshot)
PROPERTY_NAMES = [
    "A", "B", "C",                       # rotational constants  [GHz]
    "mu",                                # dipole moment         [D]
    "alpha",                             # isotropic polarizab.  [a0^3]
    "epsilon_HOMO", "epsilon_LUMO",      # orbital energies      [Ha]
    "gap",                               # HOMO–LUMO gap         [Ha]
    "r2",                                # ⟨R²⟩                  [a0²]
    "zpve",                              # zero‑point vibr. E    [Ha]
    "U0", "U", "H", "G",                 # thermochemistry       [Ha]
    "Cv"                                 # heat capacity         [cal mol⁻¹ K⁻¹]
]



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

    atoms = Atoms(symbols=symbols, positions=np.asarray(coords))
    atoms.info["gdb_index"] = int(gdb_idx)

    return atoms, props, smiles




def load_qm9(qm9_dir: Path, N_molecules: int | None = None):
    


    xyz_files = sorted(qm9_dir.glob("**/*.xyz"))  
    if N_molecules is not None:
        xyz_files = xyz_files[:N_molecules]                # limit to N_molecules

    all_props = []
    all_smiles = []
    molecules  = []

    for f in tqdm(xyz_files):
        atoms, props, smiles = parse_qm9_xyz(f)
        if any([a.GetFormalCharge() != 0 for a in Chem.MolFromSmiles(smiles).GetAtoms()]):
            continue

        molecules.append(atoms)
        all_props.append(props)
        all_smiles.append(smiles)

    regression_targets  = np.stack(all_props)

    regression_masks = np.ones_like(regression_targets, dtype=bool)

    task_names = PROPERTY_NAMES

    task_configs = [TaskConfig(task_name=task_name) for task_name in task_names]


    return all_smiles, molecules, regression_targets, regression_masks, task_configs
