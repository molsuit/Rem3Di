from threedscriptors.data_handling.source_preprocessing.geom_preprocessing import load_geom, load_geom_parallel


from threedscriptors.data_handling.dataset_builder import DatasetBuilder

from threedscriptors.data_handling.dataset import AtomicEmbeddingWithPositionsDataset
from threedscriptors.data_handling.dataset_io import store_data_to_disk

from threedscriptors.data_handling.pipelines import (
   pretraining_pipeline_from_structures
)

from mace.calculators import MACECalculator

from threedscriptors.configuration.data_config import (
    DatasetConfig,
    MaceCalculatorConfig,
    DatasetSplit,
)
# change to where you untarred the rdkit folder
base_path = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/raw_data"

# Where the dataset will be stored
dataset_directory = (
    f"/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/geom"
)

# The mace embedded model
MACE_PATH = "/share/snw30/projects/mace_model/MACE-OFF24_medium.model"


smiles_list, molecules, structure_ids = load_geom_parallel(base_path, boltzmann_weight_threshold=0.15, max_atoms= 100, N_structures=75000, max_workers=None, )


embedding_model_config = MaceCalculatorConfig(
    mace_calc=MACECalculator(model_paths=MACE_PATH, enable_cueq=True, device="cuda"),
    model_name="mace_off_24_medium",
    model_path=MACE_PATH,
    enable_cueq=True,
    device="cuda",
)

dataset_config = DatasetConfig(
    N_molecules=structure_ids[-1].molecule_id,
    dataset_type=AtomicEmbeddingWithPositionsDataset,
    BFGS_tol=0.1,
    BFGS_max_steps=500,
    N_conformers=1,
    embedding_model_config=embedding_model_config,
    max_atoms=None,
    tasks=None,
    only_heavy_atoms=False,
    dataset_name="geom",
)

dataset = pretraining_pipeline_from_structures(
    dataset_config, molecules, structure_ids).build()


store_data_to_disk(dataset, dataset_directory +"full")