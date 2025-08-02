from threedscriptors.data_handling.data_build_pipeline import (
    AtomicPositionsStage,
    PipelineOrchestrator,
    ReloadFromDiskStage,
)
from threedscriptors.data_handling.dataset_concatenation import DatasetConcatenation
from threedscriptors.data_handling.dataset_io import (
    store_data_to_disk,
)


def reload_fn(directory):
    stages = [
        ReloadFromDiskStage(directory),
        AtomicPositionsStage()]
    return PipelineOrchestrator(stages)





admet_antiviral = reload_fn(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/antiviral_admet_valid",
).build()



antiviral_potency = reload_fn("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/antiviral_potency_valid").build()


concatenation = DatasetConcatenation(datasets=[admet_antiviral, antiviral_potency])

new_dataset = concatenation.concatenate()

dataset_dir = f"/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/{new_dataset.dataset_config.dataset_name}"

if new_dataset.dataset_config.dataset_split is not None:
    dataset_dir = dataset_dir + "_" +str(new_dataset.dataset_config.dataset_split.name).lower()

store_data_to_disk(
    new_dataset,
    dataset_dir
)
