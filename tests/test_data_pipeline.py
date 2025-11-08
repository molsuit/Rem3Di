import numpy as np
from mace.calculators import mace_mp
from threedscriptors.data_handling.data_build_pipeline import (
    ChiralConformalEmbeddingStage,
    InitializeBuildPipeline,
    InsertSmilesStage,
    PipelineOrchestrator,
)
from threedscriptors.data_handling.pipelines import (
    regression_training_pipeline,
    regression_training_with_pos_pipeline,
)

from threedscriptors.configuration.data_config import (
    DatasetConfig,
    MaceCalculatorConfig,
)
from threedscriptors.data_handling.dataset import RegressionDataset


def test_regression_training_pipeline(
    sample_smiles, regression_targets, regression_masks, sample_dataset_config
):
    try:
        regression_pipeline = regression_training_pipeline(
            sample_dataset_config, sample_smiles, regression_targets, regression_masks
        )
        regression_pipeline.build()
    except Exception as err:
        raise AssertionError from err


def test_chiral_pipeline():
    dataset_config = DatasetConfig(
        N_molecules=2,
        dataset_type=RegressionDataset,
        BFGS_tol=0.2,
        BFGS_max_steps=500,
        N_conformers=1,
        embedding_model_config=MaceCalculatorConfig(
            mace_calc=mace_mp("medium"), model_name="medium_mp"
        ),
    )

    chiral_smiles = [
        "Cc1ccc(cc1)[C@@]2(C)CC(C)(C)CN2C(=S)Nc3cc(cc(c3)C(F)(F)F)C(F)(F)F",
        "Cc1ccc(cc1)[C@]2(C)CC(C)(C)CN2C(=S)Nc3cc(cc(c3)C(F)(F)F)C(F)(F)F",
    ]

    stages = [
        InitializeBuildPipeline(dataset_config, RegressionDataset),
        InsertSmilesStage(smiles=chiral_smiles),
        ChiralConformalEmbeddingStage(),
    ]

    dataset = PipelineOrchestrator(stages).build()

    assert np.all(
        dataset.molecules[0].get_positions() == -dataset.molecules[1].get_positions()
    )

def test_regression_training_with_positions_pipeline(
    sample_smiles, regression_targets, regression_masks, sample_dataset_config
):

    regression_pipeline = regression_training_with_pos_pipeline(
        sample_dataset_config, sample_smiles, regression_targets,regression_masks
    )
    dataset =  regression_pipeline.build()

    sample = dataset[0]

    print(sample)

    assert sample.atomic_positions is not None


