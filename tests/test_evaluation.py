from importlib import resources

import pydantic_yaml as pyaml
import pytest
from mace.calculators import mace_mp

from remedi.configuration.architecture_config import (
    RegressionArchitectureConfig,
)
from remedi.configuration.data_config import (
    DatasetConfig,
    DatasetTypes,
    MaceCalculatorConfig,
    TaskConfig,
)
from remedi.data_handling.pipelines import regression_training_pipeline
from remedi.evaluation.clustering import UMAPCalculator
from remedi.evaluation.evaluation_pipeline import RegressionHeadPCATask
from remedi.evaluation.evaluation_utils import (
    evaluate_molecular_descriptor_on_dataset,
    evaluate_regression_model_on_dataset,
)


def test_regression_evaluation(sample_smiles, regression_targets, regression_masks):
    dataset_config = DatasetConfig(
        N_molecules=5,
        dataset_type=DatasetTypes.REGRESSION_DATASET,
        BFGS_tol=0.2,
        BFGS_max_steps=500,
        N_conformers=1,
        embedding_model_config=MaceCalculatorConfig(
            mace_calc=mace_mp("medium"), model_name="medium"
        ),
        tasks=[
            TaskConfig(task_name="HLM"),
            TaskConfig(task_name="LogD"),
            TaskConfig(task_name="KSOL"),
        ],
    )

    config_path = resources.files("tests") / "test_architecture_config.yaml"
    architecture_config = pyaml.parse_yaml_file_as(
        RegressionArchitectureConfig, config_path
    )
    model = architecture_config.build().eval()

    dataset = regression_training_pipeline(
        dataset_config, sample_smiles, regression_targets, regression_masks
    ).build()

    _ = evaluate_regression_model_on_dataset(model, dataset, device="cpu")


def test_regression_evaluation_negative(
    sample_smiles, regression_targets, regression_masks
):
    dataset_config = DatasetConfig(
        N_molecules=5,
        dataset_type=DatasetTypes.REGRESSION_DATASET,
        BFGS_tol=0.2,
        BFGS_max_steps=500,
        N_conformers=1,
        embedding_model_config=MaceCalculatorConfig(
            mace_calc=mace_mp("medium"), model_name="medium"
        ),
        tasks=[TaskConfig(task_name="cmrt")],
    )

    config_path = resources.files("tests") / "test_architecture_config.yaml"
    architecture_config = pyaml.parse_yaml_file_as(
        RegressionArchitectureConfig, config_path
    )
    model = architecture_config.build().eval()

    dataset = regression_training_pipeline(
        dataset_config, sample_smiles, regression_targets, regression_masks
    ).build()

    with pytest.raises(AssertionError):
        _ = evaluate_regression_model_on_dataset(model, dataset, device="cpu")


def test_descriptor_evaluation(sample_smiles, regression_targets, regression_masks):
    dataset_config = DatasetConfig(
        N_molecules=5,
        dataset_type=DatasetTypes.REGRESSION_DATASET,
        BFGS_tol=0.2,
        BFGS_max_steps=500,
        N_conformers=1,
        embedding_model_config=MaceCalculatorConfig(
            mace_calc=mace_mp("medium"), model_name="medium"
        ),
        tasks=[
            TaskConfig(task_name="HLM"),
            TaskConfig(task_name="LogD"),
            TaskConfig(task_name="KSOL"),
        ],
    )

    config_path = resources.files("tests") / "test_architecture_config.yaml"
    architecture_config = pyaml.parse_yaml_file_as(
        RegressionArchitectureConfig, config_path
    )
    model = architecture_config.build().eval()

    dataset = regression_training_pipeline(
        dataset_config, sample_smiles, regression_targets, regression_masks
    ).build()

    _ = evaluate_molecular_descriptor_on_dataset(model, dataset, device="cpu")


def test_activation_clustering_in_fully_connected_regression_heads(
    sample_smiles, regression_targets, regression_masks
):
    dataset_config = DatasetConfig(
        N_molecules=5,
        dataset_type=DatasetTypes.REGRESSION_DATASET,
        BFGS_tol=0.2,
        BFGS_max_steps=500,
        N_conformers=1,
        embedding_model_config=MaceCalculatorConfig(
            mace_calc=mace_mp("medium"), model_name="medium"
        ),
        tasks=[
            TaskConfig(task_name="KSOL"),
        ],
    )

    config_path = resources.files("tests") / "test_architecture_config.yaml"
    architecture_config = pyaml.parse_yaml_file_as(
        RegressionArchitectureConfig, config_path
    )
    model = architecture_config.build().eval().cuda()
    dataset = regression_training_pipeline(
        dataset_config, sample_smiles, regression_targets, regression_masks
    ).build()

    umap_calc = UMAPCalculator()

    task = RegressionHeadPCATask(dataset=dataset, clustering_calculator=umap_calc)
    activations = task.run(model)
    print(activations)


def test_activation_clustering_in_residual_regression_heads(
    sample_smiles, regression_targets, regression_masks
):
    dataset_config = DatasetConfig(
        N_molecules=5,
        dataset_type=DatasetTypes.REGRESSION_DATASET,
        BFGS_tol=0.2,
        BFGS_max_steps=500,
        N_conformers=1,
        embedding_model_config=MaceCalculatorConfig(
            mace_calc=mace_mp("medium"), model_name="medium"
        ),
        tasks=[
            TaskConfig(task_name="cmrt"),
        ],
    )

    config_path = resources.files("tests") / "architecture_config_ps.yaml"
    architecture_config = pyaml.parse_yaml_file_as(
        RegressionArchitectureConfig, config_path
    )

    architecture_config.regression_head_config[
        0
    ].input_dimensions = 384  # To allow us not passing any auxillary data.

    model = architecture_config.build().eval().cuda()

    print(regression_masks)
    dataset = regression_training_pipeline(
        dataset_config, sample_smiles, regression_targets, regression_masks
    ).build()

    umap_calc = UMAPCalculator()
    task = RegressionHeadPCATask(dataset=dataset, clustering_calculator=umap_calc)
    activations = task.run(model)
    print(activations)
