import numpy as np
import pandas as pd

from threedscriptors.configuration.data_config import LabelScalingType, TaskConfig


def get_one_hot_columns_encodings(columns: list):
    # Gets the one hot encoded chromatography column type
    unique_columns = set(columns)

    one_hot_dict = {column: i for i, column in enumerate(unique_columns)}

    column_array = np.zeros(shape=(len(columns), len(unique_columns)))

    for i, col in enumerate(columns):
        column_array[i, one_hot_dict[col]] = 1

    return column_array, one_hot_dict


def build_auxillary_data(column_type: list, proh_proportion: np.ndarray, speed : np.ndarray):
    column_array, _ = get_one_hot_columns_encodings(column_type)

    proh_proportion = proh_proportion.reshape(-1, 1)

    speed = speed.reshape(-1, 1)
    # aux_data = {"column_type": column_array, "proh_proportion": proh_proportion}

    arr = np.hstack((column_array, proh_proportion, speed)).astype(np.float32)
    aux_data = {"cmrt": arr}
    return aux_data


def get_task_configs(aux_data: dict) -> TaskConfig:
    aux_data_dim = sum([v.shape[1] if v.ndim == 2 else 1 for v in aux_data.values()])

    task = TaskConfig(
        task_name="cmrt", has_auxillary_data=True, auxillary_data_dimension=aux_data_dim, scaling= LabelScalingType.LOG_Z
    )

    return [task]


def load_cmrt_data(dataset_filepath: str, single_column_type=False):
    # Load the CSV data. Replace 'data.csv' with your CSV file path.
    df = pd.read_csv(
        dataset_filepath,
        index_col="index",
    )

    # 1. Drop the "Literature" column.
    df = df.drop(columns=["Literature"])
    df = df[df["RT"] != 0]

    if single_column_type:
        df = df[df["Column"] == "ADH"]

    pair_indices = df["pair_index"].to_numpy()

    unique_pair_indices, counts = np.unique(pair_indices, return_counts=True)
    classes_appearing_twice = unique_pair_indices[counts == 2]
    df = df[df["pair_index"].isin(classes_appearing_twice)]



    aux_data = build_auxillary_data(
        column_type=df["Column"].tolist(),
        proh_proportion=df["i-PrOH_proportion"].to_numpy(),
        speed = df["Speed"].to_numpy()
    )

    smiles = df["SMILES"]

    regression_targets = (df["RT"].to_numpy()).reshape(
        -1, 1
    )
    regression_masks = np.ones_like(
        regression_targets, dtype=bool
    )  # Validate that this is he correct way to mask

    tasks = get_task_configs(aux_data)
    return smiles, regression_targets, regression_masks, aux_data, tasks
