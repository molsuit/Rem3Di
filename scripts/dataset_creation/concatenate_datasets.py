from pathlib import Path

from remedi_data.dataset.dataset_concatenation import (
    DatasetConcatenation,
)
from remedi_data.dataset.molecule_dataset import MoleculeDataset

path_0 = Path("/path/to/datasets/omol25_tmcs")
path_1 = Path("/path/to/datasets/tmqm")

dataset_0 = MoleculeDataset.open_existing_dataset_from_dir(path_0)
dataset_1 = MoleculeDataset.open_existing_dataset_from_dir(path_1)


dc = DatasetConcatenation(
    datasets=[dataset_0, dataset_1],
    new_dataset_dir=Path("/path/to/datasets/tmcs"),
)
#
dc.concatenate_datasets_copy_first()
