import torch

from threedscriptors.data_handling.dataset import RegressionDataset
from threedscriptors.data_handling.dataset_io import (
    load_data_from_disk,
    store_data_to_disk,
)
from threedscriptors.data_handling.pipelines import regression_training_pipeline


def test_dataset_io(
    sample_smiles, regression_targets, regression_masks, sample_dataset_config, tmp_path
):
    regression_pipeline = regression_training_pipeline(
        sample_dataset_config, sample_smiles, regression_targets, regression_masks
    )

    dataset = regression_pipeline.build()

    subdir = tmp_path / "dataset_io"
    subdir.mkdir()

    store_data_to_disk(dataset=dataset, directory=subdir)
    new_dataset = load_data_from_disk(subdir, RegressionDataset)

    assert new_dataset.molecules == dataset.molecules
    assert torch.all(new_dataset.regression_targets == dataset.regression_targets)
    assert torch.all(new_dataset.embeddings == dataset.embeddings)
    assert new_dataset.smiles_list == dataset.smiles_list
    assert torch.all(new_dataset.regression_masks == dataset.regression_masks)
