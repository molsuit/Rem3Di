from threedscriptors.data_handling.dataset import (
    RegressionDataset,
    RegressionWithAuxDataset,
)
from threedscriptors.data_handling.dataset_concatenation import DatasetConcatenation
from threedscriptors.data_handling.dataset_io import (
    load_data_from_disk,
    store_data_to_disk,
)

adme_fang_dataset = load_data_from_disk(
    "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/adme_fang",
    RegressionDataset,
    load_molecules=True,
)
admet_antiviral = load_data_from_disk(
    "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/antiviral_admet",
    RegressionDataset,
    load_molecules=True,
)
cmrt = load_data_from_disk(
    "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/cmrt",
    RegressionWithAuxDataset,
    load_molecules=True,
)


concatenation = DatasetConcatenation(datasets=[adme_fang_dataset, cmrt])

new_dataset = concatenation.concatenate()

store_data_to_disk(
    new_dataset,
    "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/multitask_dataset",
)
