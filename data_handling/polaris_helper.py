import numpy as np



def remove_molecules_with_no_data(regression_targets, regression_masks, smiles):
    datapoints_per_smile = np.sum(regression_masks, axis=1)
    no_data_smiles = np.where(datapoints_per_smile == 0)[0].tolist()

    idx_with_data = [idx for idx, smi in enumerate(smiles) if idx not in no_data_smiles]

    smiles = [smiles[idx] for idx in idx_with_data]
    regression_targets = regression_targets[idx_with_data,:]
    regression_masks = regression_masks[idx_with_data,:]

    return regression_targets, regression_masks, smiles


def pretreat_polaris_dataset(smiles: list[str],regression_targets,metadata: dict):
    # Pretreat polaris benchmark data for regression. Creates the Masks required for multitask training and removes nans. Also removes the smiles from the subset that do not have any data for the required target.
  
    regression_masks = np.where(np.isnan(regression_targets),False, True)

    if regression_masks.ndim == 1:
        regression_masks = regression_masks[:,np.newaxis]
    if regression_targets.ndim == 1:
        regression_targets = regression_targets[:,np.newaxis]

    regression_targets, regression_masks, smiles = remove_molecules_with_no_data(regression_targets, regression_masks, smiles)

    regression_targets = np.where(np.isnan(regression_targets),0, regression_targets)

    return smiles, regression_targets, regression_masks, metadata