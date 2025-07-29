
from mace.calculators import MACECalculator

from threedscriptors.configuration.data_config import (
    DatasetConfig,
    MaceCalculatorConfig,
    DatasetSplit
)
from threedscriptors.data_handling.dataset import (
    RegressionDatasetwithPositions,RegressionDatasetwithRandomWalks
)
from threedscriptors.data_handling.dataset_analysis import DatasetPostLoadAnalysis

from threedscriptors.data_handling.smiles_iterator import ListSmilesIterator
from threedscriptors.data_handling.dataset_io import (
    store_data_to_disk,
)
#from threedscriptors.data_handling.pipelines import (
#    regression_training_with_pos_pipeline
#)
#from threedscriptors.data_handling.source_preprocessing.polaris_preprocessing import (
#    load_polaris_dataset,
#)
#
#
#
#
#
#dataset_registry = {
#    "antiviral_admet": "asap-discovery/antiviral-admet-2025-unblinded",
#    "adme_fang": "biogen/adme-fang-v1",
#    "antiviral_potency" : "asap-discovery/antiviral-potency-2025-unblinded"
#}
#
#smiles_column = {
#    "antiviral_admet": "CXSMILES",
#    "adme_fang": "MOL_smiles",
#    "antiviral_potency": "CXSMILES"
#}
#
#non_task_columns = {
#    "antiviral_admet": ["Molecule Name","Set", "CXSMILES"],
#    "adme_fang": ["UNIQUE_ID", "MOL_smiles", "SMILES"],
#    "antiviral_potency": ["Molecule Name","Set", "CXSMILES"],
#
#}
#
#
#load_dataset = "antiviral_admet"
## Load the benchmark from polarishub
#smiles, _,_,_ = load_polaris_dataset(
#    dataset_registry[load_dataset],
#    smiles_column=smiles_column[load_dataset],
#    non_task_columns=non_task_columns[load_dataset],datasplit="Train"
#)

from threedscriptors.data_handling.smiles_iterator import FileSmilesIterator
from threedscriptors.data_handling.source_preprocessing.cmrt_preprocessing import load_cmrt_data

dataset_directory = (
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/cmrt"
)

smiles, regression_targets, regression_masks, aux_data, tasks = load_cmrt_data(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/raw_data/cmrt_raw_data.csv",
    single_column_type=False,
)

from rdkit import Chem

atom_counts =  []

for smi in smiles:
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    atom_counts.append(mol.GetNumAtoms())

import matplotlib.pyplot as plt 

import numpy as np 
print(np.mean(np.array(atom_counts)))
print(np.std(np.array(atom_counts)))
print(np.max(np.array(atom_counts)))
fig = plt.figure()
plt.hist(atom_counts)
plt.savefig("atom_count_hist.png")

