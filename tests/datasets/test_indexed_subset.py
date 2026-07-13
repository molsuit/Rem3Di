import numpy as np

from remedi.data_handling.indexed_subset import IndexedSubset
from remedi.data_handling.pipelines import reload_dataset_pipeline


def test_indexed_subset():
    dir = "/path/to/3DMolecularDescriptors/data/qm9_test"
    dataset = reload_dataset_pipeline(dir).build()

    sub = IndexedSubset(dataset, np.arange(50).tolist())

    assert sub.regression_targets.shape[0] == 50

    assert len(sub.molecules) == 50
