from collections.abc import Sequence

from threedscriptors.data_handling.dataset import (
    BaseDataset,
    RegressionDataset,
    RegressionWithAuxDataset,
)


class DatasetConcatenation:
    def __init__(self, datasets: Sequence[BaseDataset]):
        # takes in a list of datasets
        self.datasets = datasets

        dataset_types = set([type(d) for d in self.datasets])

        assert dataset_types.issubset(set(RegressionDataset, RegressionWithAuxDataset))

        embedding_models = set(
            [d.dataset_config.embedding_model_config.model_name for d in self.datasets]
        )

        assert len(embedding_models) == 1

    def concatenate(self):
        self.concatenate_molecules()

    def concatenate_molecules(self):
        pass
        # Adds mol_ids and smiles

    # Expand the padding mask and the atomic embeddings to the max dimension.

    # Creates the Block matrices of regression targets, and regression masks.

    # Append the dataset configs. assert no tasks have the same name
