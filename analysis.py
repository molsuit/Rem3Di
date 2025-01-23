import torch



def get_PCA(data_mat,k):

    (U,S,V) = torch.pca_lowrank(data_mat)
    k = 2
    principle_components = torch.matmul(data_mat, V[:,:k]).numpy()

    return principle_components



def get_fingerprint_correlation():
    """
    Might be interesting to see the correlation between our global descriptors and the morgan or ecfp fingerprints
    """
    
    pass