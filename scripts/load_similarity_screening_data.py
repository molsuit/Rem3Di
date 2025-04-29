from mace.calculators import MACECalculator

from threedscriptors.configuration.data_config import (
    DatasetConfig,
    DatasetTypes,
    MaceCalculatorConfig,
)
from threedscriptors.data_handling.dataset_io import store_data_to_disk
from threedscriptors.data_handling.pipelines import similarity_screening_pipeline
from threedscriptors.data_handling.smiles_iterator import ListSmilesIterator
from threedscriptors.data_handling.source_preprocessing.similarity_screening_preprocessing import (
    load_similarity_screening_data,
)

directory = "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/similarity_screening_datasets"

N_classes = 1

smiles, class_labels, activity_labels, target_class_dict = (
    load_similarity_screening_data(dir_path=directory, N_target_classes=N_classes)
)

iterator = ListSmilesIterator(smiles)

MACE_PATH = (
    "/data/fast-pc-06/snw30/projects/models/2023-12-03-mace-128-L1_epoch-199.model"
)
embedding_model_config = MaceCalculatorConfig(
    mace_calc=MACECalculator(model_paths=MACE_PATH, enable_cueq=True, device="cuda"),
    model_name="mace_mp_medium",
    model_path=MACE_PATH,
    enable_cueq=True,
    device="cuda",
)

print(len(smiles))

dataset_config = DatasetConfig(
    N_molecules=None,
    dataset_type=DatasetTypes.REGRESSION,
    BFGS_tol=0.5,
    BFGS_max_steps=100,
    N_conformers=1,
    embedding_model_config=embedding_model_config,
    max_atoms=None,
)


dataset = similarity_screening_pipeline(
    dataset_config=dataset_config,
    smiles=smiles,
    target_class_labels=class_labels,
    activity_decoy_labels=activity_labels,
).build()

store_data_to_disk(
    dataset,
    "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/virtual_screening",
)
