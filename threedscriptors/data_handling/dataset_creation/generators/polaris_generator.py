from typing import Literal

import numpy as np
import polaris as po
from polaris.dataset import DatasetV1, DatasetV2

from threedscriptors.configuration.data_config import TaskConfig
from threedscriptors.data_handling.dataset.tasks import TaskConfig, TaskScope, TaskType
from threedscriptors.data_handling.dataset_creation.generators.molecule_generator import (
    MoleculeGenerator,
)
from threedscriptors.data_handling.dataset_creation.loading_batch import (
    InputBatch,
    RegressionData,
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
    """Yield raw Polaris rows; ``FilterMoleculeStage`` handles cleanup."""

    def __init__(self, dataset_name: str, batch_size: int):
        self.dataset_name = dataset_name
        self.batch_size = batch_size

    def __iter__(self):
        smiles, regression_targets, regression_masks = load_polaris_dataset(
            dataset_name=self.dataset_name, datasplit="Train"
        )
        bs = self.batch_size
        n = len(smiles)
        for start in range(0, n, bs):
            end = min(start + bs, n)
            raw = list(smiles[start:end])
            structure_ids = [
                StructureID(structure_id=j, molecule_id=j, stereoisomer_id=j)
                for j in range(start, end)
            ]
            yield InputBatch(
                smiles=None,
                molecules=None,
                raw_smiles=raw,
                structure_ids=structure_ids,
                regression_data=RegressionData(
                    targets_system=regression_targets[start:end, :],
                    mask_system=regression_masks[start:end, :],
                ),
            )
