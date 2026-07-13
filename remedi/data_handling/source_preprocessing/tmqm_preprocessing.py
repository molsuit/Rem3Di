import glob
import os
from enum import Enum

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read

from remedi.configuration.data_config import TaskConfig
from remedi.data_handling.mol_id import StructureID


class TmqmTask(Enum):
    ELECTRONIC_E = "Electronic_E"
    DISPERSION_E = "Dispersion_E"
    DIPOLE_M = "Dipole_M"
    METAL_Q = "Metal_q"
    HL_GAP = "HL_Gap"
    HOMO_ENERGY = "HOMO_Energy"
    LUMO_ENERGY = "LUMO_Energy"
    POLARIZABILITY = "Polarizability"


def load_tmqm_dataset(
    directory: str, tasks: list[str], N_structures=50000, max_atoms=80
) -> tuple[
    list[StructureID], list[int], list[Atoms], np.ndarray, np.ndarray, list[TaskConfig]
]:
    """
    Load the tmQM dataset.

    Parameters:
    - directory: path to the folder containing .xyz/.extxyz files and tmQM_y.csv
    - N_molecules: total number of molecules to load
    - tasks: list of target properties to extract

    Returns:
    - csd_ids_int: integer-encoded CSD IDs for each molecule
    - molecules: list of ASE Atoms objects
    - regression_targets: (N_molecules x len(tasks)) array
    - regression_masks: same shape, with 1 where target is present, 0 for missing
    """
    # Find all xyz or extxyz files
    xyz_files = sorted(glob.glob(os.path.join(directory, "*.xyz")))

    if not xyz_files:
        raise FileNotFoundError(f"No .xyz files found in {directory}")

    molecules: list[Atoms] = []
    csd_ids: list[str] = []

    # Load molecules until we reach the requested count
    for file in xyz_files:
        new_mols, new_ids = load_molecules(file, max_atoms)
        molecules.extend(new_mols)
        csd_ids.extend(new_ids)

    N_molecules = len(molecules)

    # Load regression targets
    regression_file = os.path.join(directory, "tmQM_y.csv")
    regression_targets, regression_masks = load_regression_targets(
        regression_file, tasks, csd_ids
    )

    if regression_targets.shape[0] != N_molecules:
        raise ValueError(
            f"Target rows {regression_targets.shape[0]} != number of molecules {N_molecules}"
        )

    tasks = get_task_configs(tasks)

    return (
        csd_ids_to_structure_id(csd_ids),
        molecules,
        regression_targets,
        regression_masks,
        tasks,
    )


def csd_ids_to_structure_id(csd_ids) -> list[StructureID]:
    assert len(csd_ids) == len(set(csd_ids))  # uniqueness check

    structure_ids = []
    for index, csd_id in enumerate(csd_ids):
        structure_ids.append(
            StructureID(
                structure_id=index,
                molecule_id=index,
                canonical_smiles=csd_id,
                conformer_id=0,
                smiles_id=index,
            )
        )

    return structure_ids


def get_task_configs(tasks):
    task_configs = []
    for task in tasks:
        task = TaskConfig(
            task_name=task, has_auxillary_data=False, auxillary_data_dimension=0
        )
        task_configs.append(task)
    return task_configs


def load_molecules(file: str, max_atoms: int) -> tuple[list[Atoms], list[str]]:
    mol = read(file, index=":")

    csd_ids = [m.info["CSD_code"] for m in mol]

    filter = [m.info["q"] == 0 and m.info["S"] == 0 and len(m) < max_atoms for m in mol]

    filtered_csd_ids = [id for id, keep in zip(csd_ids, filter, strict=False) if keep]
    filtered_mols = [m for m, keep in zip(mol, filter, strict=False) if keep]

    return filtered_mols, filtered_csd_ids


def load_regression_targets(
    regression_target_file: str, tasks: list[str], csd_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load regression targets and masks for specified tasks from tmQM_y.csv.

    Returns:
    - targets: (N_samples x N_tasks) numpy array, NaNs replaced by 0
    - masks: same shape, 1 where data present, 0 where missing
    """

    for t in tasks:
        if t not in set(TmqmTask):
            raise ValueError(f"Unknown task '{t}'")

    df = pd.read_csv(regression_target_file, sep=";")

    task_names = [t.value for t in tasks]

    missing_cols = set(task_names) - set(df.columns)

    if missing_cols:
        raise ValueError(f"Tasks {missing_cols} not found in CSV columns")

    # 2. Make CSD_code the index so we can re‑index by `csd_ids`
    df = df.set_index("CSD_code")

    # 3. Keep only desired rows/columns, preserving the order of `csd_ids`
    sub = df.loc[csd_ids, task_names]

    # 4. Build mask before we overwrite NaNs
    masks = (~sub.isna()).astype(int).to_numpy()

    # 5. Replace NaNs with 0 for the model
    targets = sub.fillna(0).to_numpy(dtype=float)

    return targets, masks
