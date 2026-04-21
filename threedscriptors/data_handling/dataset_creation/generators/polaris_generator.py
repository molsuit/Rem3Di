from typing import Literal

import numpy as np
import polaris as po
from polaris.dataset import DatasetV1, DatasetV2
from rdkit import Chem

from threedscriptors.configuration.data_config import TaskConfig
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

POLARIS_DATASET = Literal["antiviral_admet", "adme_fang", "antiviral_potency"]

DATASET_REGISTRY = {
    "antiviral_admet": "asap-discovery/antiviral-admet-2025-unblinded",
    "adme_fang": "biogen/adme-fang-v1",
    "antiviral_potency": "asap-discovery/antiviral-potency-2025-unblinded",
}

SMILES_COLUMN = {
    "antiviral_admet": "CXSMILES",
    "adme_fang": "MOL_smiles",
    "antiviral_potency": "CXSMILES",
}

NON_TASK_COLUMNS = {
    "antiviral_admet": ["Molecule Name", "Set", "CXSMILES"],
    "adme_fang": ["UNIQUE_ID", "MOL_smiles", "SMILES"],
    "antiviral_potency": ["Molecule Name", "Set", "CXSMILES"],
}


def get_polaris_task_configs(dataset_name: POLARIS_DATASET) -> list[TaskConfig]:
    configs = []
    dataset = po.load_dataset(DATASET_REGISTRY[dataset_name])
    columns = dataset.columns

    target_cols = [c for c in columns if c not in NON_TASK_COLUMNS[dataset_name]]

    for task in target_cols:
        configs.append(
            TaskConfig(name=task, task_type=TaskType.regression, scope=TaskScope.system)
        )

    return configs


def load_polaris_benchmark(benchmark_name: str):
    benchmark = po.load_benchmark(benchmark_name)

    train, test = benchmark.get_train_test_split()

    data = train.as_dataframe()
    smiles = data["CXSMILES"]
    target_cols = train.target_cols
    regression_targets = data[target_cols].to_numpy()
    regression_masks = np.where(np.isnan(regression_targets), 0, 1)

    return smiles, regression_targets, regression_masks


def load_polaris_dataset(dataset_name: POLARIS_DATASET, datasplit="Train"):
    dataset = po.load_dataset(DATASET_REGISTRY[dataset_name])
    columns = dataset.columns

    target_cols = [c for c in columns if c not in NON_TASK_COLUMNS[dataset_name]]

    if isinstance(dataset, DatasetV1):
        data_dict = dataset.table[:]

    elif isinstance(dataset, DatasetV2):
        set_col = dataset.zarr_data["Set"][:]

        row_indices = np.argwhere(set_col == datasplit)

        data_dict = {name: arr[row_indices] for name, arr in dataset.zarr_data.items()}

    smiles = data_dict[SMILES_COLUMN[dataset_name]].squeeze().tolist()

    regression_targets = np.array([data_dict[task] for task in target_cols]).T.squeeze()

    regression_masks = np.where(np.isnan(regression_targets), 0, 1)

    assert np.all(np.any(regression_targets, axis=0))

    return smiles, regression_targets, regression_masks


class PolarisGenerator(MoleculeGenerator):
    def __init__(self, dataset_name: str, batch_size: int, max_atoms: int):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.max_atoms = max_atoms

    def __iter__(self):
        idx = 0
        smiles, regression_targets, regression_masks = load_polaris_dataset(
            dataset_name=self.dataset_name, datasplit="Train"
        )

        batch_smiles = []
        batch_structure_ids = []
        accept_indices = []

        for i, smi in enumerate(smiles):
            if smi is None:
                continue

            mol = Chem.MolFromSmiles(smi)
            if filter_mol(mol, max_atoms=self.max_atoms):
                smiles = Chem.MolToSmiles(
                    Chem.RemoveAllHs(mol), isomericSmiles=True, canonical=True
                )
                batch_smiles.append(
                    SmilesData(
                        nonisomeric_smiles=Chem.CanonSmiles(smiles, useChiral=False),
                        isomeric_smiles=smiles,
                    )
                )
                batch_structure_ids.append(
                    StructureID(structure_id=idx, molecule_id=idx, stereoisomer_id=idx)
                )
                idx += 1

                accept_indices.append(i)

            if len(batch_smiles) >= self.batch_size:
                batch_regression_targets = regression_targets[accept_indices, :]
                batch_regression_masks = regression_masks[accept_indices, :]

                yield InputBatch(
                    molecules=None,
                    smiles=batch_smiles,
                    structure_ids=batch_structure_ids,
                    regression_data=RegressionData(
                        targets_system=batch_regression_targets,
                        mask_system=batch_regression_masks,
                    ),
                )

                accept_indices = []
                batch_smiles = []
                batch_structure_ids = []

        if batch_smiles:
            batch_regression_targets = regression_targets[accept_indices, :]
            batch_regression_masks = regression_masks[accept_indices, :]

            yield InputBatch(
                molecules=None,
                smiles=batch_smiles,
                structure_ids=batch_structure_ids,
                regression_data=RegressionData(
                    targets_system=batch_regression_targets,
                    mask_system=batch_regression_masks,
                ),
            )
