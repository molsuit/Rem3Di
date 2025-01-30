import torch
import matplotlib.pyplot as plt


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


def plot_delta_histogram(delta):
    """
    Plot a histogram of the differences between the predictions and the true values
    """
    fig = plt.figure()
    plt.hist(delta, bins=50)
    plt.xlabel("Difference between prediction and true value")
    plt.ylabel("Frequency")
    return fig