import torch 

from threedscriptors.model.preprocessing.atomic_descriptor_preprocessor import RMSLayerNorm, PseudoscalarGenerator

from threedscriptors.configuration.architecture_config import EmbeddingPreprocessConfig

from e3nn.o3 import Irreps
import numpy as np 

data = torch.from_numpy(np.load("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/cmrt_train/embeddings.npy")).float()

masks = torch.from_numpy(np.load("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/cmrt_train/padding_mask.npy"))



print(f"Embedding Shape {data.shape}")


config = EmbeddingPreprocessConfig(input_irreps = "128x0e+128x1o+128x0e", pseudoscalars=True, pseudoscalar_dimension =  128)

irreps = config.input_irreps

print(f"Irreps Dim {irreps.dim}")


pg = PseudoscalarGenerator(config)
out = pg(data, masks)

pseudos = out[:,:,-config.pseudoscalar_dimension:]
scalars = out[:,:, :-config.pseudoscalar_dimension]


pseudos = pseudos.reshape(-1, config.pseudoscalar_dimension)
scalars = pseudos.reshape(-1, config.input_invariant_dimension)

ps_mean = pseudos.mean(dim =  0)
ps_std = pseudos.std(dim = 0)


print(f"Mean {ps_mean}, std {ps_std}, avg_std = {ps_std.mean()}")
print(f"Mean {scalars.mean(dim = 0)}, std = {scalars.std(dim = 0)}")


import matplotlib.pyplot as plt

plt.figure(dpi = 100)
plt.bar(np.arange(config.pseudoscalar_dimension), ps_std.detach().numpy(),width= 1)
plt.savefig("individual_ps_dims")


plt.figure(dpi = 100)
plt.bar(np.arange(config.input_invariant_dimension), scalars.std(dim=0).detach().numpy(),width= 1)
plt.savefig("individual_scalars_dims")




fig = plt.figure()
plt.hist(pseudos.reshape(-1,).detach().numpy())
plt.savefig("ps_distribution")


fig = plt.figure()
plt.hist(scalars.reshape(-1,).detach().numpy())
plt.savefig("scalar_distribution")
