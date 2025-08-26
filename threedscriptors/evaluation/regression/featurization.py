from molfeat.trans.fp import FPVecTransformer


def calculate_mol_features(smiles_list, descriptor_name):
    featurizer = FPVecTransformer(kind=descriptor_name)
    descriptors = featurizer(smiles_list)

    return descriptors
