from enum import Enum
from functools import reduce
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem

from threedscriptors.data_handling.dataset.tasks import TaskConfig, TaskScope, TaskType
from threedscriptors.data_handling.dataset_creation.generators.molecule_generator import (
    MoleculeGenerator,
)
from threedscriptors.data_handling.dataset_creation.generators.utils import (
    filter_mol,
)
from threedscriptors.data_handling.dataset_creation.loading_batch import (
    InputBatch,
    RegressionData,
    SmilesData,
)
from threedscriptors.data_handling.dataset_creation.structure_ids import StructureID


class MoleculeNetTask(Enum):
    LIPOPHILICITY = "lipophilicity"
    BACE = "bace"
    BBBP = "bbbp"
    FREE_SOLVE = "free_solv"
    ESOL = "esol"
    HIV = "hiv"


task_column_registry = {
    "lipophilicity": ["lipophilicity"],
    "bace": ["bace_active", "bace_pIC50"],
    "bbbp": ["bbbp_permeable"],
    "free_solv": ["hydration_free_energy"],
    "esol": ["solubility"],
    "hiv": ["hiv_active"],
}

task_type_registry = {
    "lipophilicity": TaskType.regression,
    "bace_pIC50": TaskType.regression,
    "bace_active": TaskType.classification,
    "bbbp_permeable": TaskType.classification,
    "hydration_free_energy": TaskType.regression,
    "solubility": TaskType.regression,
    "hiv_active": TaskType.classification,
}


class MoleculeNetTaskConfig(TaskConfig):
    name: str
    file_path: Path
    task_type: TaskType
    scope: TaskScope = TaskScope.system


def convert_tasks_to_configs(
    tasks: list[MoleculeNetTask],
) -> list[MoleculeNetTaskConfig]:
    configs = []

    for mnet_task in tasks:
        task_name_str = mnet_task.value
        task_columns = [
            f"{task_name_str}.{t}" for t in task_column_registry[task_name_str]
        ]

        file_path = f"{task_name_str}.csv"
        target_types = [
            task_type_registry[t] for t in task_column_registry[task_name_str]
        ]

        for col, target_typ in zip(task_columns, target_types, strict=False):
            configs.append(
                MoleculeNetTaskConfig(
                    name=col,
                    file_path=file_path,
                    auxillary_dim=None,
                    scope=TaskScope.system,
                    task_type=target_typ,
                )
            )

    return configs


class MoleculeNetGenerator(MoleculeGenerator):
    def __init__(
        self,
        processed_mnet_dir: Path,
        batch_size: int,
        tasks: MoleculeNetTaskConfig | list[MoleculeNetTaskConfig],
        max_atoms,
    ):
        self.mnet_dir = processed_mnet_dir
        self.loading_batch_size = batch_size
        self.max_atoms = max_atoms
        self.tasks = tasks

    @staticmethod
    def _load_renamed_moleculenet_lf(path: Path) -> pl.LazyFrame:
        lf = pl.scan_csv(path, dtypes={"smiles": pl.Utf8}).unique(subset=["smiles"])
        rename_map = {c: f"{path.stem}.{c}" for c in lf.columns if c != "smiles"}
        return lf.rename(rename_map)

    def load_regression_data(self):
        csvs = set([t.file_path for t in self.tasks])
        lfs = [
            self._load_renamed_moleculenet_lf(self.mnet_dir / file_path)
            for file_path in csvs
        ]

        # set of keys
        smiles_series = pl.concat([lf.select("smiles") for lf in lfs]).unique()

        lf = reduce(
            lambda acc, lf: acc.join(lf, on="smiles", how="left"), lfs, smiles_series
        )

        self.target_cols = lf.columns
        self.target_cols.remove("smiles")
        return lf

    def __iter__(self):
        """
        Generator yielding 'Ligand SMILES' values from a TSV file in streaming batches.
        """
        idx = 0

        data_source = self.load_regression_data()

        # 2) Execute in the streaming engine (memory‐bounded)
        df = data_source.collect()

        batch_smiles = []
        batch_structure_ids = []
        targets_buffer = []
        masks_buffer = []

        # 3) Slice into batches and yield one SMILES at a time
        for batch_df in df.iter_slices(n_rows=self.loading_batch_size):
            accept_indices = []

            for i, smi in enumerate(batch_df["smiles"]):
                if smi is None:
                    continue

                mol = Chem.MolFromSmiles(smi)
                if filter_mol(mol, max_atoms=self.max_atoms):
                    smiles = Chem.MolToSmiles(
                        Chem.RemoveAllHs(mol), isomericSmiles=True, canonical=True
                    )

                    batch_smiles.append(
                        SmilesData(
                            nonisomeric_smiles=Chem.CanonSmiles(
                                smiles, useChiral=False
                            ),
                            isomeric_smiles=smiles,
                        )
                    )

                    batch_structure_ids.append(
                        StructureID(
                            structure_id=idx, molecule_id=idx, stereoisomer_id=idx
                        )
                    )
                    idx += 1

                    accept_indices.append(i)

            if accept_indices:
                accept_indices = np.array(accept_indices)
                sub = batch_df.select(self.target_cols)
                sub = sub.with_columns(pl.all().cast(pl.Float32))
                t = (
                    sub.fill_null(np.nan).to_numpy().reshape(-1, len(self.target_cols))
                )  # (k, n_targets), floats with NaN
                m = (
                    sub.select(pl.all().is_not_null())
                    .to_numpy()
                    .astype(np.int8)
                    .reshape(-1, len(self.target_cols))
                )  # (k, n_targets), 0/1

                targets_buffer.extend(t[accept_indices, :])
                masks_buffer.extend(m[accept_indices, :])

            if len(batch_smiles) >= self.loading_batch_size:
                regression_targets = np.vstack(targets_buffer)
                regression_masks = np.vstack(masks_buffer)

                yield InputBatch(
                    molecules=None,
                    smiles=batch_smiles,
                    structure_ids=batch_structure_ids,
                    regression_data=RegressionData(
                        targets_system=regression_targets, mask_system=regression_masks
                    ),
                )

                batch_smiles = batch_smiles[self.loading_batch_size :]
                batch_structure_ids = batch_structure_ids[self.loading_batch_size :]
                targets_buffer = [regression_targets[self.loading_batch_size :, :]]
                masks_buffer = [regression_masks[self.loading_batch_size :, :]]

        if batch_smiles:
            regression_targets = np.concatenate(targets_buffer)
            regression_masks = np.concatenate(masks_buffer)

            yield InputBatch(
                molecules=None,
                smiles=batch_smiles,
                structure_ids=batch_structure_ids,
                regression_data=RegressionData(
                    targets_system=regression_targets, mask_system=regression_masks
                ),
            )
