import torch
from mace.calculators import mace_mp

from threedscriptors.configuration.data_config import (
    DatasetConfig,
    DatasetTypes,
    MaceCalculatorConfig,
)
from threedscriptors.data_handling.pipelines import (
    regression_training_with_transition_probs_pipeline,
)


def test_random_walk_probabilities(regression_targets, regression_masks):


    smiles = ["C", "CC" , "CCC" , "CO", "COO"]


    dataset_config = DatasetConfig(
        N_molecules=5,
        dataset_type=DatasetTypes.REGRESSION_DATATSET_WITH_RANDOMWALK,
        BFGS_tol=0.5,
        BFGS_max_steps=500,
        N_conformers=1,
        embedding_model_config=MaceCalculatorConfig(
            mace_calc=mace_mp("medium"), model_name="medium"
        ),
        load_adjacency_matrix= True
    )

    dataset = regression_training_with_transition_probs_pipeline(dataset_config, smiles, regression_targets, regression_masks).build()

    assert torch.isfinite(dataset.random_walk_transition_matrix).all()

    for T in dataset.random_walk_transition_matrix:
        print(T)


