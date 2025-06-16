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
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/adme_fang",
    RegressionDataset,
    load_molecules=True,
)
admet_antiviral = load_data_from_disk(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/antiviral_admet",
    RegressionDataset,
    load_molecules=True,
)
cmrt = load_data_from_disk(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/cmrt",
    RegressionWithAuxDataset,
    load_molecules=True,
)


concatenation = DatasetConcatenation(datasets=[adme_fang_dataset, admet_antiviral, cmrt])

new_dataset = concatenation.concatenate()

store_data_to_disk(
    new_dataset,
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/multitask",
)
