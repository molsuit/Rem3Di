from mace.calculators import MACECalculator

from threedscriptors.configuration.data_config import (
   DatasetConfig,
   MaceCalculatorConfig,
)
from threedscriptors.data_handling.dataset import AtomicEmbeddingWithPositionsDataset
from threedscriptors.data_handling.dataset_io import store_data_to_disk
from threedscriptors.data_handling.pipelines import pretraining_pipeline_from_structures
from threedscriptors.data_handling.source_preprocessing.pcqm_preprocessing import (
   load_pcqm,
)

pcqm_file = "/home/snw30/rds/hpc-work/3DMolecularDescriptors/data/raw_data/pcqm4m-v2-train.sdf"


N_molecules = 400_000


smiles, molecules, structure_ids = load_pcqm(pcqm_file=pcqm_file, N_molecules = N_molecules)



dataset_directory = (
    "/home/snw30/rds/hpc-work/3DMolecularDescriptors/data/pcqm"
)

MACE_PATH = "/home/snw30/rds/hpc-work/models/MACE-OFF24_medium.model"

embedding_model_config = MaceCalculatorConfig(
    mace_calc=MACECalculator(model_paths=MACE_PATH, enable_cueq=True, device="cuda"),
    model_name="mace_off_24_medium",
    model_path=MACE_PATH,
    enable_cueq=True,
    device="cuda",
    default_dtype="float32",
)

dataset_config = DatasetConfig(
    N_molecules=N_molecules,
    dataset_type=AtomicEmbeddingWithPositionsDataset,
    BFGS_tol=0.1,
    BFGS_max_steps=500,
    N_conformers=1,
    embedding_model_config=embedding_model_config,
    max_atoms=None,
    tasks=None,
    only_heavy_atoms=False,
    dataset_name="pcqm4m",
    rw_transition_matrix_from_3D=False,
)

dataset = pretraining_pipeline_from_structures(
    dataset_config, molecules, structure_ids).build()


store_data_to_disk(dataset, dataset_directory)
