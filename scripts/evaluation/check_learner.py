from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Crippen

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
    pos_emb_getitem,
)
from threedscriptors.evaluation.evaluation_utils import (
    evaluate_molecular_descriptor_on_dataset,
)
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
    ScalerParams,ScalerType
)
from threedscriptors.model.model_builder import ModelBuilder

dataset_dir = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/molecule_net"
)
model_dir = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/pcqm_with_classic_agg/0-2025_09_26_19_09_22-Train"
)

remedi_model = ModelBuilder.from_directory(model_dir).build_remedi_model()


dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_dir)

ds = TrainingMoleculeDataset(dataset_dir, get_item=pos_emb_getitem)


desc_ecfp = MolfeatDescriptorCalculator("ecfp")


X_remedi = evaluate_molecular_descriptor_on_dataset(remedi_model, ds)

X_ecfp = desc_ecfp.calculate_descriptors(dataset)

import numpy as np

smiles = dataset.get_smiles_per_structure()
y = np.zeros(shape=len(smiles))
for i, smi in enumerate(smiles):
    mol = Chem.MolFromSmiles(smi)
    logp = Crippen.MolLogP(mol)
    y[i] = logp


breakpoint()


#y = np.asarray(dataset.targets_system)
#y = dataset.targets_system
cv_params = CVParams(scoring="neg_mean_absolute_error")

ridge = RidgeLearner(scaler_params = ScalerParams(scaler_type=ScalerType.STANDARD))

ridge_res_remedi = run_kfold_repeated_cross_validation(
    X_remedi, y, cv_params=cv_params, learner=ridge
)

print(ridge_res_remedi)

ridge_res_ecfp = run_kfold_repeated_cross_validation(
    X_ecfp, y, cv_params=cv_params, learner=ridge
)




print(ridge_res_ecfp)


rf = RandomForestLearner()
rf_res_ecfp = run_kfold_repeated_cross_validation(
    X_ecfp, y, cv_params=cv_params, learner=rf
)  #

rf_res_remedi = run_kfold_repeated_cross_validation(
    X_remedi, y, cv_params=cv_params, learner=rf
)
print(ridge_res_ecfp)
print(ridge_res_remedi)



print(rf_res_ecfp)
print(rf_res_remedi)
