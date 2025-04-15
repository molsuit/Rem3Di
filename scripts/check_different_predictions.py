import numpy.testing as npt
import pydantic_yaml as pyaml
import torch

from threedscriptors.configuration.architecture_config import (
    ArchitectureConfig,
)
from threedscriptors.configuration.data_config import DatasetConfig, DatasetTypes
from threedscriptors.data_handling.cmrt_preprocessing import load_cmrt_data
from threedscriptors.data_handling.dataset_builder import (
    DatasetBuildingDirector,
)
from threedscriptors.data_handling.smiles_iterator import ListSmilesIterator
from threedscriptors.model.model_builder import ModelBuilder

smiles, regression_targets, regression_masks, aux_data, tasks = load_cmrt_data()


MACE_PATH = (
    "/data/fast-pc-06/snw30/projects/models/2023-12-03-mace-128-L1_epoch-199.model"
)
# Get the train and test data-loaders

dataset_config = DatasetConfig(
    N_molecules=2,
    dataset_type=DatasetTypes.REGRESSION,
    BFGS_tol=0.2,
    BFGS_max_steps=500,
    N_conformers=2,
    embedding_model=MACE_PATH,
    max_atoms=None,
    tasks=tasks,
)
iterator = ListSmilesIterator(smiles)
db_director, dataset = DatasetBuildingDirector.build_chiral_dataset(
    iterator=iterator,
    dataset_config=dataset_config,
    regression_targets=regression_targets,
    regression_masks=regression_masks,
    auxillary_data=aux_data,
    return_normalized_targets=True,
    return_normalized_inputs=True,
)


mol_0 = dataset.molecules[0]
mol_1 = dataset.molecules[1]
assert (mol_0.get_positions() == -mol_1.get_positions()).all()

dataset.auxillary_data["cmrt"] = torch.Tensor(dataset.auxillary_data["cmrt"])


def test_different_predictions_nops():
    architecture_config = pyaml.parse_yaml_file_as(
        ArchitectureConfig,
        "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/transformer_model/cmrt_nops/architecture_config.yaml",
    )

    mb = ModelBuilder(architecture_config=architecture_config)
    model = mb.build_model()
    model.eval()

    invariant_embeddings = model.preprocessor(dataset.embeddings)

    npt.assert_array_almost_equal(invariant_embeddings[0], invariant_embeddings[1])

    pred = model(
        dataset.embeddings,
        padding_mask=dataset.padding_mask,
        auxillary_data=dataset.auxillary_data,
    )

    print(pred)
    print(dataset.regression_targets)


def test_different_predictions_with_ps():
    architecture_config = pyaml.parse_yaml_file_as(
        ArchitectureConfig,
        "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/transformer_model/cmrt_ps/architecture_config.yaml",
    )

    mb = ModelBuilder(architecture_config=architecture_config)
    model = mb.build_model()
    model.eval()

    embeddings_w_ps = model.preprocessor(dataset.embeddings)
    print(embeddings_w_ps[0] - embeddings_w_ps[1])

    mol_des = model.get_molecular_descriptor(
        dataset.embeddings, padding_mask=dataset.padding_mask
    )

    print(mol_des[0] - mol_des[1])

    pred = model(
        dataset.embeddings,
        padding_mask=dataset.padding_mask,
        auxillary_data=dataset.auxillary_data,
    )

    print(pred)
    print(dataset.regression_targets)


# test_different_predictions_nops()

test_different_predictions_with_ps()
