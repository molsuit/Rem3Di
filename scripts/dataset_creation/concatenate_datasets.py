from threedscriptors.data_handling.data_build_pipeline import (
    AtomicPositionsStage,
    PipelineOrchestrator,
    ReloadFromDiskStage,
    ReduceMoleculeSize
)
from threedscriptors.data_handling.dataset_concatenation import DatasetConcatenation
from threedscriptors.data_handling.dataset_io import (
    store_data_to_disk,
)
from threedscriptors.data_handling.dataset import AtomicEmbeddingWithPositionsDataset




def reload_fn(directory):
    stages = [
        ReloadFromDiskStage(directory),
        AtomicPositionsStage(),
        ReduceMoleculeSize(max_atoms = 100)]
    
    dataset = PipelineOrchestrator(stages).build()
    dataset.convert_to_dataset_type(AtomicEmbeddingWithPositionsDataset)
    return dataset


#data_dir = "/home/snw30/rds/hpc-work/3DMolecularDescriptors/data"
data_dir="/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data"

admet_antiviral = reload_fn(
    f"{data_dir}/antiviral_admet_test",
)

antiviral_potency = reload_fn(f"{data_dir}/antiviral_potency_test")


print(antiviral_potency.embeddings.shape)
adme_fang = reload_fn(f"{data_dir}/adme_fang_test")
#
#geom_200k = reload_fn(f"{data_dir}/geom_200k")



concatenation = DatasetConcatenation(datasets=[admet_antiviral, antiviral_potency, adme_fang])

new_dataset = concatenation.concatenate()

dataset_dir = f"{data_dir}/{new_dataset.dataset_config.dataset_name}"

if new_dataset.dataset_config.dataset_split is not None:
    dataset_dir = dataset_dir + "_" +str(new_dataset.dataset_config.dataset_split.name).lower()

store_data_to_disk(
    new_dataset,
    dataset_dir
)

print(max([len(mol) for mol in new_dataset.molecules]))