from threedscriptors.data_handling.dataset_creation.dataset_concatenation import DatasetConcatenation
from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset

from pathlib import Path 


pcqm_path = Path("/scratch/public/snw30/dataset/pcqm/pcqm_only_structures_3_5_M")
geom_path = Path("/scratch/public/snw30/dataset/geom_drugs")
pharma_path = Path("/scratch/public/snw30/dataset/multi_pharma_dataset")
dataset_pcqm = MoleculeDataset.open_existing_dataset_from_dir(pcqm_path)
dataset_geom = MoleculeDataset.open_existing_dataset_from_dir(geom_path)
dataset_pharma = MoleculeDataset.open_existing_dataset_from_dir(pharma_path)

print(len(dataset_pcqm))
print(len(dataset_geom))

dc = DatasetConcatenation(datasets = [dataset_pcqm, dataset_geom, dataset_pharma], new_dataset_dir=Path("/scratch/public/snw30/dataset/concat_dataset"))
#
dc.concatenate_datasets_copy_first()

##from pathlib import Path
#
#from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
#from threedscriptors.data_handling.dataset_creation.dataset_concatenation import (
#    LabeldDatasetConcatenation,
#)
#
#adme_fang_path = Path("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/adme_fang")
#antiviral_potency_path = Path("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/antiviral_potency")
#moleculenet_path = Path("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/molecule_net")
#antiviral_admet_path = Path("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/antiviral_admet")
#
#dirs = [adme_fang_path, antiviral_potency_path, antiviral_admet_path,moleculenet_path]
#
#datasets = [MoleculeDataset.open_existing_dataset_from_dir(data_dir) for data_dir in dirs]
#
#dc = LabeldDatasetConcatenation(datasets = datasets, new_dataset_dir=Path("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/multi_pharma_dataset"))
#
#dc.concatenate_datasets()


