import json

import matplotlib.pyplot as plt
import torch
from mace.calculators import MACECalculator

from threedscriptors.data_handling.preprocessing import get_global_descriptor
from threedscriptors.data_handling.smiles_iterator import FileSmilesIterator
from threedscriptors.model.model import TransformerEncoder
from threedscriptors.utils.analysis import get_PCA

encoder_params = torch.load(
    "/home/steffen/projects/mol_descriptors/transformer_model/transformer_encoder.pth"
)

architecture_parameter = json.load(
    open(
        "/home/steffen/projects/mol_descriptors/transformer_model/architecture_parameters.json"
    )
)

encoder = TransformerEncoder(num_layers=3, **architecture_parameter)
encoder.load_state_dict(encoder_params)

MACE_PATH = "/home/steffen/projects/mol_descriptors/mace_model/2023-12-10-mace-128-L0_energy_epoch-249.model"

mace_calculator = MACECalculator(model_path=MACE_PATH, device="cuda")

smiles_list = []
data_mat = None


smiles_iter = FileSmilesIterator("./data/enols_thiols.smi")

for smiles in smiles_iter:
    smiles_list.append(smiles)
    global_descriptor = get_global_descriptor(
        smiles, encoder, calculator=mace_calculator
    )
    global_descriptor = global_descriptor.squeeze(0)

    if data_mat is None:
        data_mat = global_descriptor
    else:
        data_mat = torch.vstack((data_mat, global_descriptor))


principle_components = get_PCA(data_mat, k=2)


hydroxyl_index = [i for i, smiles in enumerate(smiles_list) if "O" in smiles]
thiol_index = [i for i, smiles in enumerate(smiles_list) if "S" in smiles]


plt.scatter(
    x=principle_components[hydroxyl_index, 0], y=principle_components[hydroxyl_index, 1]
)
plt.scatter(
    x=principle_components[thiol_index, 0], y=principle_components[thiol_index, 1]
)

for idx, smiles in enumerate(smiles_list):
    plt.annotate(smiles, xy=principle_components[idx, :])

plt.title("Hydroxyl vs Thiol comparison")
plt.xlabel("PC 0")
plt.ylabel("PC 1")
plt.savefig("figures/hydroxyl_vs_thiol.png")
