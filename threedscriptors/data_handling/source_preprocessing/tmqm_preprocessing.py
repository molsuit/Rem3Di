import glob
import os

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read

from threedscriptors.configuration.data_config import TaskConfig


def load_tmqm_dataset(
    directory: str, tasks: list[str]
) -> tuple[list[int], list[Atoms], np.ndarray, np.ndarray]:
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
        new_mols, new_ids = load_molecules(file)
        molecules.extend(new_mols)
        csd_ids.extend(new_ids)

    N_molecules = len(molecules)

    assert len(csd_ids) == len(set(csd_ids))
    id_map = {cid: idx for idx, cid in enumerate(csd_ids)}
    csd_ids_int = [id_map[cid] for cid in csd_ids]

    # Load regression targets
    regression_file = os.path.join(directory, "tmQM_y.csv")
    regression_targets, regression_masks = load_regression_targets(
        regression_file, tasks
    )

    if regression_targets.shape[0] != N_molecules:
        raise ValueError(
            f"Target rows {regression_targets.shape[0]} != number of molecules {N_molecules}"
        )

    tasks = get_task_configs(tasks)

    return np.array(csd_ids_int), molecules, regression_targets, regression_masks, tasks


def get_task_configs(tasks):
    task_configs = []
    for task in tasks:
        task = TaskConfig(
            task_name=task, has_auxillary_data=False, auxillary_data_dimension=0
        )
        task_configs.append(task)
    return task_configs


def load_molecules(file: str) -> tuple[list[Atoms], list[str]]:
    mol = read(file, index=":")
    csd_ids = [m.info["CSD_code"] for m in mol]

    return mol, csd_ids


def load_regression_targets(
    regression_target_file: str, tasks: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load regression targets and masks for specified tasks from tmQM_y.csv.

    Returns:
    - targets: (N_samples x N_tasks) numpy array, NaNs replaced by 0
    - masks: same shape, 1 where data present, 0 where missing
    """
    allowed = [
        "Electronic_E",
        "Dispersion_E",
        "Dipole_M",
        "Metal_q",
        "HL_Gap",
        "HOMO_Energy",
        "LUMO_Energy",
        "Polarizability",
    ]

    for t in tasks:
        if t not in allowed:
            raise ValueError(f"Unknown task '{t}'. Allowed: {allowed}")

    df = pd.read_csv(regression_target_file, sep=";", index_col=0)

    missing_cols = set(tasks) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Tasks {missing_cols} not found in CSV columns")

    targets = df[tasks].values.astype(float)

    if np.any(np.isnan(targets)):
        raise ValueError

    masks = (~np.isnan(targets)).astype(int)
    # Replace NaNs with zero to keep array numeric
    targets = np.nan_to_num(targets, nan=0.0)

    return targets, masks
