from pathlib import Path

from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.data_handling.dataset_creation.dataset_concatenation import (
    DatasetConcatenation,
)

path_0 = Path("/scratch/s5f/wedigs.s5f/datasets/omol25_tmcs")
path_1 = Path("/scratch/s5f/wedigs.s5f/datasets/tmqm")

dataset_0 = MoleculeDataset.open_existing_dataset_from_dir(path_0)
dataset_1 = MoleculeDataset.open_existing_dataset_from_dir(path_1)


dc = DatasetConcatenation(
    datasets=[dataset_0, dataset_1],
    new_dataset_dir=Path("/scratch/s5f/wedigs.s5f/datasets/tmcs"),
)
#
dc.concatenate_datasets_copy_first()
