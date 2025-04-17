import numpy as np
from molfeat.trans.fp import FPVecTransformer


def calculate_molfeat_fingerprint(
    smiles_list: list[str], fingerprint_name
) -> np.ndarray:
    featurizer = FPVecTransformer(kind=fingerprint_name)
    fingerprints = featurizer(smiles_list)

    return fingerprints
