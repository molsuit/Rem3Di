import glob
import os

import numpy as np
import pandas as pd


def load_similarity_screening_data(dir_path, N_target_classes=None):
    """
    Load similarity screening data from .dat files in a directory,
    optionally limiting to a specified number of target classes.

    Args:
        dir_path (str): Path to the directory containing .dat files.
        N_target_classes (int, optional): Maximum number of distinct classes to load. If None, load all classes.

    Returns:
        smiles (list of str): List of SMILES strings across selected files.
        class_labels (np.ndarray): Array of integer class indices for each SMILES.
        activity_labels (np.ndarray): Array of binary activity labels (1=active, 0=decoy).
        target_class_dict (dict): Mapping from class name to integer index for loaded classes.
    """
    # Find all cmp_list_*.dat files, exclude ChEMBL
    all_files = glob.glob(os.path.join(dir_path, "cmp_list_*.dat"))
    files_list = [f for f in all_files if "ChEMBL" not in os.path.basename(f)]

    # Group files by class name
    class_to_files = {}
    for file_path in files_list:
        fname = os.path.basename(file_path)
        class_name = get_class_name(fname)
        class_to_files.setdefault(class_name, []).append(file_path)

    # Optionally limit to N_target_classes
    if N_target_classes is not None:
        # preserve insertion order
        selected = list(class_to_files)[:N_target_classes]
        class_to_files = {cls: class_to_files[cls] for cls in selected}

    smiles = []
    activity_labels = []
    class_labels = []
    target_class_dict = {}

    # Iterate over classes and their files
    for class_idx, (class_name, file_paths) in enumerate(class_to_files.items()):
        target_class_dict[class_name] = class_idx
        for file_path in file_paths:
            fname = os.path.basename(file_path)
            activity = get_activity_class_from_filename(fname)
            # Read SMILES
            new_smiles = extract_smiles_from_dat_file(file_path)
            smiles.extend(new_smiles)
            activity_labels.extend([activity] * len(new_smiles))
            class_labels.extend([class_idx] * len(new_smiles))

    # Convert to numpy arrays
    activity_labels = np.array(activity_labels, dtype=np.int64)
    class_labels = np.array(class_labels, dtype=np.int64)

    return smiles, class_labels, activity_labels, target_class_dict


def extract_smiles_from_dat_file(filename):
    """
    Read a .dat file and extract the SMILES column as a list.
    """
    df = pd.read_csv(filename, sep="\t", header=0, usecols=["SMILES"])
    return df["SMILES"].astype(str).tolist()


def get_activity_class_from_filename(filename):
    """
    Determine activity class from filename suffix.
    Returns 1 for '_actives', 0 for '_decoys'.
    """
    if "_actives" in filename:
        return 1
    elif "_decoys" in filename:
        return 0
    else:
        raise ValueError(
            f"Filename '{filename}' does not specify activity class (_actives or _decoys)"
        )


def get_class_name(file_name):
    """
    Extract the target class name from the filename by stripping the prefix and suffix.
    """
    base = os.path.splitext(file_name)[0]
    if base.startswith("cmp_list_"):
        base = base[len("cmp_list_") :]
    for suffix in ["_actives", "_decoys"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base
