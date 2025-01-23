import torch.utils.data as data
from torch import from_numpy
import numpy as np
from tqdm import tqdm
from preprocessing import get_mace_descriptors
import os
import json

class AtomEmbeddingDataset(data.Dataset):
    
    def __init__(self,data,padding_mask):
        super().__init__()
        self.data = data #So far, this is only the MACE embeddings and no regression target
        self.masks = padding_mask

    @classmethod
    def reload_from_disk(cls, data_path : str, mask_path : str,):
            
        
        data_arr = np.load(data_path)
        mask_arr = np.load(mask_path)

        data = from_numpy(data_arr).float()
        padding_mask = from_numpy(mask_arr).float()

        return cls(data, padding_mask)

    @classmethod
    def construct_from_smiles(cls,smiles_path:str,mace_caluclator,**kwargs):
        
        N_molecules = kwargs.get("N_molecules",250) # Number of molecules to be processed
        BFGS_tol = kwargs.get("BFGS_tol",0.05)
        max_atoms = kwargs.get("max_atoms", 3*13 + 2) # Maximum number of Atoms -> Sequence length in the transformer
        n_descriptor = kwargs.get("embedding_size",256) # Size of the invariant MACE features

        smiles_dict = {}

        data= np.zeros((N_molecules, max_atoms, n_descriptor))
        mask = np.zeros((N_molecules, max_atoms))


        #TODO: Abstract away the file iterator and replace it with an iterator over smiles

        with open(smiles_path, "r") as f:
            for i in tqdm(range(0,N_molecules)):
                smiles = f.readline()
                smiles_dict[i] = smiles[:-1] 
                descriptor = get_mace_descriptors(smiles,mace_caluclator, BFGS_tol)
                data[i,0:descriptor.shape[0],:] = descriptor
                mask[i,0:descriptor.shape[0]] = 1


        # Save data/metadata to disk
        directory = os.path.dirname(
            smiles_path)
        np.save(f"{directory}/descriptor_data.npy", data)
        np.save(f"{directory}/descriptor_mask.npy", mask)
        with open(f"{directory}/smiles_dict.json", "w") as f:
            json.dump(smiles_dict, f)

        with open(f"{directory}/metadata.json", "w") as f:
            json.dump({"max_atoms": max_atoms,
                "dataset_size": N_molecules,
                "BFGS_tolerance": BFGS_tol}, f)

    def __len__(self):
        return self.masks.shape[0]

    def __getitem__(self, index):
        embeddings = self.data[index]
        mask = self.masks[index]

        return embeddings, mask
    
