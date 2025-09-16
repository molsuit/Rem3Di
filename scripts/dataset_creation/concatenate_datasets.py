from threedscriptors.data_handling.dataset_creation.dataset_concatenation import DatasetConcatenation
from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset

from pathlib import Path 
pcqm_path = Path("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/pcqm")
geom_path = Path("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/geom_drugs")
dataset_pcqm = MoleculeDataset.open_existing_dataset_from_dir(pcqm_path)
dataset_geom = MoleculeDataset.open_existing_dataset_from_dir(geom_path)


dc = DatasetConcatenation(datasets = [dataset_geom, dataset_pcqm], new_dataset_dir=Path("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/concat_dataset"))

dc.concatenate_datasets()