from mace.calculators import MACECalculator
from threedscriptors.data_handling.dataset import SimilarityScreeningDataset
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

directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/raw_data/virtual_screening"

N_classes = 1

smiles, class_labels, activity_labels, target_class_dict = (
    load_similarity_screening_data(dir_path=directory, N_target_classes=N_classes)
)


print(set(activity_labels))
print(target_class_dict)

iterator = ListSmilesIterator(smiles)


MACE_PATH = (
    "/share/snw30/projects/mace_model/mace_agnesi_medium.model"
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
    dataset_type=SimilarityScreeningDataset,
    BFGS_tol=0.01,
    BFGS_max_steps=100,
    N_conformers=1,
    embedding_model_config=embedding_model_config,
    max_atoms=None,
)


dataset = similarity_screening_pipeline(
    dataset_config=dataset_config,
    smiles=smiles,
    target_class_labels=class_labels,
    active_decoy_labels=activity_labels,
).build()

store_data_to_disk(
    dataset,
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/virtual_screening",
)
