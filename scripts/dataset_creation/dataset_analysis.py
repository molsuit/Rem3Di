from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from threedscriptors.data_handling.source_preprocessing.cmrt_preprocessing import (
    load_cmrt_data,
)

# from threedscriptors.data_handling.pipelines import (
#    regression_training_with_pos_pipeline
# )
# from threedscriptors.data_handling.source_preprocessing.polaris_preprocessing import (
#    load_polaris_dataset,
# )
#
#
#
#
#
# dataset_registry = {
#    "antiviral_admet": "asap-discovery/antiviral-admet-2025-unblinded",
#    "adme_fang": "biogen/adme-fang-v1",
#    "antiviral_potency" : "asap-discovery/antiviral-potency-2025-unblinded"
# }
#
# smiles_column = {
#    "antiviral_admet": "CXSMILES",
#    "adme_fang": "MOL_smiles",
#    "antiviral_potency": "CXSMILES"
# }
#
# non_task_columns = {
#    "antiviral_admet": ["Molecule Name","Set", "CXSMILES"],
#    "adme_fang": ["UNIQUE_ID", "MOL_smiles", "SMILES"],
#    "antiviral_potency": ["Molecule Name","Set", "CXSMILES"],
#
# }
#
#
# load_dataset = "antiviral_admet"
## Load the benchmark from polarishub
# smiles, _,_,_ = load_polaris_dataset(
#    dataset_registry[load_dataset],
#    smiles_column=smiles_column[load_dataset],
#    non_task_columns=non_task_columns[load_dataset],datasplit="Train"
# )


dataset_directory = (
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/cmrt"
)

smiles, regression_targets, regression_masks, aux_data, tasks = load_cmrt_data(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/raw_data/cmrt_raw_data.csv",
    single_column_type=False,
)

atom_counts = []
rmsds = []



def process_smile(smi):
    atom_count = 0
    rmsd_list = []
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return atom_count, rmsd_list
    mol = Chem.AddHs(mol)
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=5, randomSeed=42, )
    mol_noH = Chem.RemoveHs(mol)
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            rmsd = AllChem.GetConformerRMS(mol_noH, i, j)
            rmsd_list.append(rmsd)
    atom_count = mol.GetNumAtoms()
    return atom_count, rmsd_list

with ProcessPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_smile, smiles,chunksize=1), total=len(smiles)))

for atom_count, rmsd_list in results:
    atom_counts.append(atom_count)
    rmsds.extend(rmsd_list)


print(np.mean(np.array(atom_counts)))
print(np.std(np.array(atom_counts)))
print(np.max(np.array(atom_counts)))
fig = plt.figure()
plt.hist(atom_counts)
plt.savefig("atom_count_hist.png")


fig = plt.figure()
plt.hist(rmsds, bins=50)
plt.xlabel("RMSD")
plt.ylabel("Frequency")
plt.title("RMSD Distribution")
plt.savefig("rmsd_hist_w_H.png")
