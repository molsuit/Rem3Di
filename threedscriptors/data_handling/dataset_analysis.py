import numpy as np


def calculate_atomic_descriptor_std_per_dim(atomic_descriptors, padding_masks, invariant_indices):

    # Atomic descriptors are (N_dataset, N_atoms, D_embedding_dim)



    masks = np.where(~np.expand_dims(padding_masks, axis=-1), True, False)
    std_per_dim = np.std(atomic_descriptors[:,:,invariant_indices], axis=(0, 1), where=masks)
    return std_per_dim


def calculate_atomic_descriptor_norms(atomic_descriptors, padding_masks):


    masks = np.where(~np.expand_dims(padding_masks, axis=-1), True, False)
    norms = np.linalg.norm(atomic_descriptors, axis = (0,1), where = masks)
    return norms
