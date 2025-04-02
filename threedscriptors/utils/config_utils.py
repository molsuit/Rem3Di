from dataclasses import asdict, fields

import numpy as np
import torch
import yaml

from threedscriptors.data_handling.data_config import DatasetConfig
from threedscriptors.model.architecture_config import (
    ArchitectureConfig,
    EmbeddingPreprocessConfig,
    GlobalAggregatorConfig,
    RegressionHeadConfig,
)
from threedscriptors.training.training_config import TrainingConfig


def dataclass_from_dict(data: dict, cls):
    """Convert a dictionary to a dataclass instance."""

    fieldtypes = {f.name: f.type for f in fields(cls)}
    kwargs = {}
    for field, field_type in fieldtypes.items():
        value = data.get(field)
        if isinstance(value, dict) and hasattr(field_type, "__dataclass_fields__"):
            # If the value is a dictionary and the field type is a dataclass, recursively call from_dict
            kwargs[field] = dataclass_from_dict(value, field_type)
        else:
            kwargs[field] = value
    return cls(**kwargs)


def from_yaml(yaml_file: str, cls):
    with open(yaml_file) as file:
        data = yaml.safe_load(file)
    return dataclass_from_dict(data, cls)


def to_yaml(yaml_file, data):
    data_in_dict = asdict(data)
    with open(yaml_file, "w") as file:
        yaml.dump(data_in_dict, file)


def get_global_config(
    training_config: TrainingConfig,
    dataset_config: DatasetConfig,
    architecture_config: ArchitectureConfig,
    regression_head_config: RegressionHeadConfig,
    global_aggregator_config: GlobalAggregatorConfig,
    preprocessor_config: EmbeddingPreprocessConfig,
):
    config_dict = {
        "training_config": asdict(training_config),
        "dataset_config": asdict(dataset_config),
        "architecture_config": asdict(architecture_config),
        "regression_head_config": asdict(regression_head_config),
        "preprocessor_config": preprocessor_config.serialize(),
        "global_aggregator_config": asdict(global_aggregator_config),
    }

    return convert_leaf_tensors_to_list(config_dict)


def convert_leaf_tensors_to_list(tree):
    """
    Recursively convert torch.Tensor leaves in a nested tree structure to numpy arrays.

    Args:
        tree: A nested structure (dict, list, tuple, set, etc.) whose leaves might be torch.Tensor objects.

    Returns:
        A new nested structure with the same shape where each torch.Tensor is replaced by its corresponding
        numpy array.
    """
    if isinstance(tree, torch.Tensor):
        # Ensure tensor is on CPU before calling numpy()
        return tree.cpu().numpy().tolist()
    elif isinstance(tree, np.ndarray):
        return tree.tolist()
    elif isinstance(tree, dict):
        return {key: convert_leaf_tensors_to_list(value) for key, value in tree.items()}
    elif isinstance(tree, list):
        return [convert_leaf_tensors_to_list(item) for item in tree]
    elif isinstance(tree, tuple):
        return tuple(convert_leaf_tensors_to_list(item) for item in tree)
    elif isinstance(tree, set):
        return {convert_leaf_tensors_to_list(item) for item in tree}
    else:
        # Base case: if tree is not a container or a tensor, return it unchanged.
        return tree
