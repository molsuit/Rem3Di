import glob
import os
from enum import Enum
from itertools import chain

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import iread

from threedscriptors.data_handling.dataset_creation.generators.molecule_generator import (
    MoleculeGenerator,
)
from threedscriptors.data_handling.dataset_creation.loading_batch import (
    InputBatch,
    RegressionData,
)
from threedscriptors.data_handling.dataset_creation.structure_ids import StructureID


class TmqmTask(Enum):
    ELECTRONIC_E = "Electronic_E"
    DISPERSION_E = "Dispersion_E"
    DIPOLE_M = "Dipole_M"
    METAL_Q = "Metal_q"
    HL_GAP = "HL_Gap"
    HOMO_ENERGY = "HOMO_Energy"
    LUMO_ENERGY = "LUMO_Energy"
    POLARIZABILITY = "Polarizability"


class TmqmGenerator(MoleculeGenerator):
    def __init__(
        self, tmqm_dir: str, batch_size: int, tasks: TmqmTask | list[TmqmTask]
    ):
        self.dir = tmqm_dir
        self.loading_batch_size = batch_size
        self.tasks = [tasks] if isinstance(TmqmTask) else tasks

    @staticmethod
    def filter_systems(
        mol: Atoms, max_atoms: int, can_model_spin_and_charge: bool = False
    ):
        if not can_model_spin_and_charge and mol.info["q"] == 0 and mol.info["S"] == 0:
            return False
        if max_atoms is not None and len(mol) < max_atoms:
            return False

        return True

    def open_regression_labels(self):
        label_file = os.path.join(self.dir, "tmQM_y.csv")
        df = pd.read_csv(label_file, sep=";", columns=self.tasks, index_col="CSD_code")
        return df

    def __iter__(self):
        # Opens the regression_dataset

        xyz_files = sorted(glob.glob(os.path.join(self.dir, "*.xyz")))

        if not xyz_files:
            raise FileNotFoundError(f"No .xyz files found in {self.dir}")

        suppl = chain.from_iterable([iread(p) for p in xyz_files])

        batch_atoms: list[Atoms] = []
        batch_structure_ids: list[StructureID] = []
        targets_buffer: list[float] = []

        regression_df = self.open_regression_labels()

        for idx, atoms in enumerate(suppl):
            if self.filter_systems(atoms):
                batch_atoms.append(atoms)
                batch_structure_ids.append(
                    StructureID(structure_id=idx, molecule_id=idx, stereoisomer_id=idx)
                )

                targets_buffer.append(regression_df[atoms.info["CSD_code"], :])

            if len(batch_atoms) >= self.loading_batch_size:
                regression_targets = np.array(targets_buffer)
                regression_masks = np.ones_like(regression_targets)

                yield InputBatch(
                    molecules=batch_atoms,
                    smiles=None,
                    structure_ids=batch_structure_ids,
                    regression_data=RegressionData(
                        targets_system=regression_targets, mask_system=regression_masks
                    ),
                )
                batch_atoms, batch_structure_ids, targets_buffer = [], [], []

        # flush tail
        if batch_atoms:
            regression_targets = np.array(targets_buffer)
            regression_masks = np.ones_like(regression_targets)

            yield InputBatch(
                molecules=batch_atoms,
                smiles=None,
                structure_ids=batch_structure_ids,
                regression_data=RegressionData(
                    targets_system=regression_targets, mask_system=regression_masks
                ),
            )

