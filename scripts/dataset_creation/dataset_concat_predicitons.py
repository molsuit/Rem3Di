from pathlib import Path

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset_creation.dataset_concatenation import (
    LabeldDatasetConcatenation,
)

adme_fang_path = Path("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/adme_fang")
antiviral_potency_path = Path("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/antiviral_potency")
moleculenet_path = Path("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/molecule_net")
antiviral_admet_path = Path("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/antiviral_admet")

dirs = [adme_fang_path, antiviral_potency_path, antiviral_admet_path,moleculenet_path]

datasets = [MoleculeDataset.open_existing_dataset_from_dir(data_dir) for data_dir in dirs]

dc = LabeldDatasetConcatenation(datasets = datasets, new_dataset_dir=Path("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/multi_pharma_dataset"))

dc.concatenate_datasets()

