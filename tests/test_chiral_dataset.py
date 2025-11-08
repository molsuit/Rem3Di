from importlib import resources

import numpy as np
import torch
from mace.calculators import mace_mp
from threedscriptors.data_handling.pipelines import chiral_regression_training_pipeline

from threedscriptors.configuration.architecture_config import (
    ArchitectureConfig,
)
from threedscriptors.configuration.config_utils import from_yaml
from threedscriptors.configuration.data_config import (
    DatasetConfig,
    DatasetTypes,
    MaceCalculatorConfig,
)
from threedscriptors.data_handling.source_preprocessing.cmrt_preprocessing import (
    load_cmrt_data,
)
from threedscriptors.model.model_builder import ModelBuilder

data_file = resources.files("tests") / "cmrt_raw_test_data.csv"
smiles, regression_targets, regression_masks, aux_data, tasks = load_cmrt_data(
    data_file
)

print(regression_masks)
print(regression_targets)

embedding_model_config = MaceCalculatorConfig(
    mace_calc=mace_mp("medium", enable_cueq=False, device="cpu"),
    model_name="MACE-MP0 medium",
    enable_cueq=False,
    device="cpu",
)  # I know, not very elegant...

# Get the train and test data-loaders
dataset_config = DatasetConfig(
    N_molecules=2,
    dataset_type=DatasetTypes.REGRESSION_WITH_AUX_DATASET,
    BFGS_tol=0.2,
    BFGS_max_steps=500,
    N_conformers=2,
    embedding_model_config=embedding_model_config,
    max_atoms=None,
    tasks=tasks,
)

dataset = chiral_regression_training_pipeline(
    dataset_config=dataset_config,
    smiles=smiles,
    regression_targets=regression_targets,
    regression_masks=regression_masks,
    auxillary_data=aux_data,
).build()

print(type(dataset.auxillary_data["cmrt"]))


dataset.auxillary_data["cmrt"] = torch.Tensor(dataset.auxillary_data["cmrt"])

mol_0 = dataset.molecules[0]
mol_1 = dataset.molecules[1]


def test_molecule_creation():
    assert (mol_1.get_atomic_numbers() == mol_0.get_atomic_numbers()).all()
    assert (mol_0.get_positions() == -mol_1.get_positions()).all()


def test_different_predictions_nops():
    yaml_file = resources.files("tests") / "architecture_config_nops.yaml"

    architecture_config = from_yaml(yaml_file, ArchitectureConfig)

    mb = ModelBuilder(architecture_config=architecture_config)
    model = mb.build_model()
    model.eval()

    invariant_embeddings = model.preprocessor(dataset.embeddings)

    print(torch.sum(invariant_embeddings[0] - invariant_embeddings[1]))

    assert torch.allclose(invariant_embeddings[0], invariant_embeddings[1], rtol=0.001)

    pred = model(
        dataset.embeddings,
        padding_mask=dataset.padding_mask,
        auxillary_data=dataset.auxillary_data,
    )

    assert torch.allclose(pred[0], pred[1], atol=0.01)


def test_different_predictions_with_ps():
    yaml_file = resources.files("tests") / "architecture_config_ps.yaml"

    architecture_config = from_yaml(yaml_file, ArchitectureConfig)

    mb = ModelBuilder(architecture_config=architecture_config)
    model = mb.build_model()
    model.eval()

    embeddings_w_ps = model.preprocessor(dataset.embeddings)
    assert not np.allclose(
        embeddings_w_ps[0].detach().cpu().numpy(),
        -embeddings_w_ps[1].detach().cpu().numpy(),
    )

    mol_des = model.get_molecular_descriptor(
        dataset.embeddings, padding_mask=dataset.padding_mask
    )

    assert not np.allclose(
        mol_des[0].detach().cpu().numpy(), mol_des[1].detach().cpu().numpy()
    )

    pred = model(
        dataset.embeddings,
        padding_mask=dataset.padding_mask,
        auxillary_data=dataset.auxillary_data,
    )

    assert pred[0].detach().cpu().numpy() != pred[1].detach().cpu().numpy()
