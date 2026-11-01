import glob
import os
from enum import Enum
from itertools import chain

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import iread

from remedi.data_handling.dataset_creation.generators.molecule_generator import (
    MoleculeGenerator,
)
from remedi.data_handling.dataset_creation.loading_batch import (
    InputBatch,
    RegressionData,
)
from remedi.data_handling.dataset_creation.structure_ids import StructureID


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
    """Stream raw tmQM Atoms; size gate lives in ``FilterAtomsStage``."""

    def __init__(
        self,
        tmqm_dir: str,
        batch_size: int,
        tasks: TmqmTask | list[TmqmTask],
    ):
        self.dir = tmqm_dir
        self.loading_batch_size = batch_size
        self.tasks = [tasks] if isinstance(tasks, TmqmTask) else tasks

    def open_regression_labels(self):
        label_file = os.path.join(self.dir, "tmQM_y.csv")
        df = pd.read_csv(label_file, sep=";", columns=self.tasks, index_col="CSD_code")
        return df

    def __iter__(self):
        xyz_files = sorted(glob.glob(os.path.join(self.dir, "*.xyz")))

        if not xyz_files:
            raise FileNotFoundError(f"No .xyz files found in {self.dir}")

        suppl = chain.from_iterable([iread(p, index=":") for p in xyz_files])

        batch_atoms: list[Atoms] = []
        batch_structure_ids: list[StructureID] = []
        batch_charges: list[float] = []
        batch_multiplicities: list[float] = []
        targets_buffer: list[float] = []

        regression_df = self.open_regression_labels()

        for idx, atoms in enumerate(suppl):
            batch_atoms.append(atoms)
            batch_structure_ids.append(
                StructureID(structure_id=idx, molecule_id=idx, stereoisomer_id=idx)
            )
            batch_charges.append(float(atoms.info["q"]))
            # tmQM xyz headers carry total spin angular momentum S, but the
            # downstream MACE / PolarMACE input expects spin multiplicity
            # 2S+1. Store the multiplicity so a tmQM singlet maps to 1.0 (the
            # MACE closed-shell default) rather than 0.0.
            batch_multiplicities.append(2.0 * float(atoms.info["S"]) + 1.0)

            targets_buffer.append(regression_df[atoms.info["CSD_code"], :])

            if len(batch_atoms) >= self.loading_batch_size:
                regression_targets = np.array(targets_buffer)
                regression_masks = np.ones_like(regression_targets)

                yield InputBatch(
                    molecules=batch_atoms,
                    smiles=None,
                    structure_ids=batch_structure_ids,
                    total_charge=batch_charges,
                    multiplicity=batch_multiplicities,
                    regression_data=RegressionData(
                        targets_system=regression_targets, mask_system=regression_masks
                    ),
                )
                batch_atoms, batch_structure_ids = [], []
                batch_charges, batch_multiplicities, targets_buffer = [], [], []

        if batch_atoms:
            regression_targets = np.array(targets_buffer)
            regression_masks = np.ones_like(regression_targets)

            yield InputBatch(
                molecules=batch_atoms,
                smiles=None,
                structure_ids=batch_structure_ids,
                total_charge=batch_charges,
                multiplicity=batch_multiplicities,
                regression_data=RegressionData(
                    targets_system=regression_targets, mask_system=regression_masks
                ),
            )
