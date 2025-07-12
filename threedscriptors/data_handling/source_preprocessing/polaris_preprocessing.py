from dataclasses import dataclass

import numpy as np
import polaris as po
from polaris.dataset import DatasetV1, DatasetV2

from threedscriptors.configuration.data_config import TaskConfig


@dataclass
class RegressionData:
    task_names: list[str]
    smiles: list[str]
    regression_targets: np.ndarray
    regression_masks: np.ndarray


def remove_molecules_with_no_data(regression_targets, regression_masks, smiles):
    datapoints_per_smile = np.sum(regression_masks, axis=1)
    no_data_smiles = np.where(datapoints_per_smile == 0)[0].tolist()

    idx_with_data = [idx for idx, smi in enumerate(
        smiles) if idx not in no_data_smiles]

    smiles = [smiles[idx] for idx in idx_with_data]
    regression_targets = regression_targets[idx_with_data, :]
    regression_masks = regression_masks[idx_with_data, :]

    return regression_targets, regression_masks, smiles


def pretreat_polaris_dataset(smiles: list[str], regression_targets):
    # Pretreat polaris benchmark data for regression. Creates the Masks required for multitask training and removes nans. Also removes the smiles from the subset that do not have any data for the required target.

    regression_masks = np.where(np.isnan(regression_targets), False, True)
    
    if regression_masks.ndim == 1:
        regression_masks = regression_masks[:, np.newaxis]
    if regression_targets.ndim == 1:
        regression_targets = regression_targets[:, np.newaxis]

    regression_targets, regression_masks, smiles = remove_molecules_with_no_data(
        regression_targets, regression_masks, smiles
    )

    regression_targets = np.where(
        np.isnan(regression_targets), 0, regression_targets)

    return smiles, regression_targets, regression_masks


def create_task_configs(tasks: list[str]) -> list[TaskConfig]:
    for task in tasks:
        print(task)
    return [TaskConfig(task_name=task) for task in tasks]


def load_polaris_benchmark(benchmark_name: str):
    benchmark = po.load_benchmark(benchmark_name)

    train, test = benchmark.get_train_test_split()

    data = train.as_dataframe()
    smiles = data["CXSMILES"]
    target_cols = train.target_cols
    targets = data[target_cols].to_numpy()

    smiles, regression_targets, regression_masks = pretreat_polaris_dataset(
        smiles, targets
    )

    tasks = create_task_configs(target_cols)

    return smiles, regression_targets, regression_masks, tasks


def load_polaris_dataset(dataset_name: str, smiles_column, non_task_columns, datasplit="Train"):

    dataset = po.load_dataset(dataset_name)
    print(dataset)
    columns = dataset.columns

    target_cols = [c for c in columns if c not in non_task_columns]

    if isinstance(dataset, DatasetV1):
        data_dict = dataset.table[:]

    elif isinstance(dataset, DatasetV2):
        # if "Set" in non_task_columns and datasplit is not None:
        #    data_dict = dataset[dataset["Set"] == datasplit]
        # else:
        

        set_col = dataset.zarr_data["Set"][:]

        row_indices = np.argwhere(set_col ==  datasplit)

        data_dict = {name : arr[row_indices] for name, arr in dataset.zarr_data.items()}

    smiles = data_dict[smiles_column].squeeze().tolist()

    regression_targets = np.array([data_dict[task] for task in target_cols]).T.squeeze()


    smiles, regression_targets, regression_masks = pretreat_polaris_dataset(
        smiles, regression_targets
    )

    assert np.all(np.any(regression_targets, axis=0))

    tasks = create_task_configs(target_cols)

    return smiles, regression_targets, regression_masks, tasks
