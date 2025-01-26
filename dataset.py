import torch.utils.data as data
from torch import from_numpy
import numpy as np
from tqdm import tqdm
from preprocessing import get_mace_descriptors, get_ase_atoms
import os
import json
from SmilesIterator import SmilesIterator


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
    def construct_from_smiles(cls,iterator : SmilesIterator,mace_caluclator,**kwargs):
        
        N_molecules = kwargs.get("N_molecules",250) # Number of molecules to be processed
        BFGS_tol = kwargs.get("BFGS_tol",0.05)
        max_atoms = kwargs.get("max_atoms", 3*13 + 2) # Maximum number of Atoms -> Sequence length in the transformer
        n_descriptor = kwargs.get("embedding_size",256) # Size of the invariant MACE features

        smiles_list = []
        
        data= np.zeros((N_molecules, max_atoms, n_descriptor))
        mask = np.zeros((N_molecules, max_atoms))


        #TODO: Abstract away the file iterator and replace it with an iterator over smiles

        i = 0
    
        with tqdm(total=N_molecules) as pbar:
            while(i < N_molecules):
                smiles = next(iterator)
                try:
                    atoms = get_ase_atoms(smiles)
                except ValueError as ve:
                    tqdm.write(f"Error with Smiles {smiles}: {ve}")
                    continue
                else:
                    smiles_list.append(smiles)
                    descriptor = get_mace_descriptors(atoms,mace_caluclator, BFGS_tol)
                    data[i,0:descriptor.shape[0],:] = descriptor
                    mask[i,0:descriptor.shape[0]] = 1
                    i = i+1
                    pbar.update(1)


        # Save data/metadata to disk
        directory = "/home/steffen/projects/mol_descriptors/data"

        np.save(f"{directory}/descriptor_data.npy", data)
        np.save(f"{directory}/descriptor_mask.npy", mask)


        with open(f"{directory}/smiles_list", "w") as f:
            for i in smiles_list:
                f.write(i + "\n")

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
    



class RegressionTask(AtomEmbeddingDataset):
