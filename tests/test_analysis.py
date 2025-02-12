import numpy as np

from threedscriptors.utils.analysis import compute_class_std


def test_class_std():
    class_ids = np.array([0, 0, 0, 1, 1, 1, 1, 1])

    data_class_1 = np.random.normal(size=(3, 5))
    data_class_2 = np.random.normal(size=(5, 5))

    data = np.vstack((data_class_1, data_class_2))

    mean, std = compute_class_std(data, class_ids)

    assert np.allclose(mean[0, :], np.mean(data_class_1, axis=0))
    assert np.allclose(mean[1, :], np.mean(data_class_2, axis=0))

    assert np.allclose(std[0, :], np.std(data_class_1, axis=0))
    assert np.allclose(std[1, :], np.std(data_class_2, axis=0))
