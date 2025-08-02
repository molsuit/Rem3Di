

## Add rattle noise to the dataset and check how this impacts the atomic descriptors

rattled_molecules = []
for atoms in molecules:
    new_atoms = atoms.copy()
    new_atoms.rattle(stdev = 0.05)
    rattled_molecules.append(new_atoms)



rattled_dataset = regression_training_from_structures_pipeline(
    dataset_config, rattled_molecules, structure_ids, regression_targets, regression_masks
).build()



import torch

from threedscriptors.utils.model_utils import (
    get_invariant_indices,
    get_mace_calculator_irrep_signature,
)

inv_indices, _ = get_invariant_indices(get_mace_calculator_irrep_signature(embedding_model_config.mace_calc))

#Distribution of distances of atomic descriptors.
relaxed_embeddings = dataset.embeddings[:,:,inv_indices]
rattled_embeddings = rattled_dataset.embeddings[:,:,inv_indices]


diff = torch.linalg.norm(relaxed_embeddings - rattled_embeddings, dim = (0,1))


import matplotlib.pyplot as plt

dist_fig = plt.figure()
plt.hist(diff[torch.nonzero(diff)])
dist_fig.savefig(f"{dataset_directory}/difference_histogram.png",dpi = 200)




#Change in distribution


std_per_channel_relaxed = torch.std(relaxed_embeddings,dim = (0,1))
std_per_channel_rattled = torch.std(rattled_embeddings,dim = (0,1))

fig_std_bars = plt.figure(figsize=(8,6))
plt.bar(torch.arange(len(std_per_channel_rattled)), std_per_channel_rattled, width = 0.5,align = "edge")
plt.bar(torch.arange(len(std_per_channel_relaxed))+0.5, std_per_channel_relaxed,width= 0.5, align = "edge")
fig_std_bars.savefig(f"{dataset_directory}/atomic_emebddings_std_per_channel.png",dpi = 200)
