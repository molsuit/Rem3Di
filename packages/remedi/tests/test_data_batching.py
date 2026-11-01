"""``sample_collate_fn`` stacks per-molecule samples and their auxiliary data."""

import pytest
import torch

from remedi.data_handling.sample import Sample, sample_collate_fn


@pytest.fixture
def embeddings():
    return torch.rand((5, 5))


@pytest.fixture
def padding_mask():
    return torch.ones((5, 1))


@pytest.fixture
def regression_target():
    return torch.tensor([1.0])


@pytest.fixture
def regression_mask():
    return torch.tensor([1])


def test_collate_fn_regression(
    embeddings, padding_mask, regression_target, regression_mask
):
    Sample0 = Sample(
        embeddings=embeddings,
        padding_mask=padding_mask,
        regression_targets=regression_target,
        regression_masks=regression_mask,
    )
    Sample1 = Sample(
        embeddings=embeddings,
        padding_mask=padding_mask,
        regression_targets=regression_target,
        regression_masks=regression_mask,
    )

    batch = sample_collate_fn([Sample0, Sample1])

    assert batch.embeddings.shape[0] == 2
    assert batch.regression_targets.shape[0] == 2
    assert batch.regression_masks.shape[0] == 2

    assert batch.auxillary_data is None


def test_collate_fn_aux_data(
    embeddings, padding_mask, regression_target, regression_mask
):
    auxillary_data = {"foo": torch.tensor([1.0]), "bar": torch.tensor([2.0])}

    Sample0 = Sample(
        embeddings=embeddings,
        padding_mask=padding_mask,
        regression_targets=regression_target,
        regression_masks=regression_mask,
        auxillary_data=auxillary_data,
    )
    Sample1 = Sample(
        embeddings=embeddings,
        padding_mask=padding_mask,
        regression_targets=regression_target,
        regression_masks=regression_mask,
        auxillary_data=auxillary_data,
    )

    batch = sample_collate_fn([Sample0, Sample1])

    assert batch.embeddings.shape[0] == 2
    assert batch.regression_targets.shape[0] == 2
    assert batch.regression_masks.shape[0] == 2

    for values in batch.auxillary_data.values():
        assert values.shape[0] == 2
