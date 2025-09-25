from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.evaluation.regression.cross_validation import (
    CVParams,
    run_kfold_repeated_cross_validation,
)
from threedscriptors.evaluation.regression.featurization import (
    MolfeatDescriptorCalculator,
)
from threedscriptors.evaluation.regression.learner import (
    RandomForestLearner,
    RidgeLearner,
)

dataset_dir = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/molecule_net"
)
dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_dir)
print(dataset.targets_system.shape)
print(len(dataset.smiles))

desc = MolfeatDescriptorCalculator("ecfp")
X = desc.calculate_descriptors(dataset)

y = dataset.targets_system

print(dataset.get_smiles_per_structure())
print(y[:].tolist())

breakpoint()
cv_params = CVParams(scoring="neg_mean_absolute_error")

ridge = RidgeLearner()
ridge_res = run_kfold_repeated_cross_validation(X, y, cv_params=cv_params, learner=ridge)

print(ridge_res)


rf = RandomForestLearner()
rf_res = run_kfold_repeated_cross_validation(X, y, cv_params=cv_params, learner=rf)



print(rf_res)