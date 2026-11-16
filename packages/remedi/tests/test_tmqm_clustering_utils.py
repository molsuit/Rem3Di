"""The tmQM metal-centre set and the block colourings use the shared element
blocks of :mod:`remedi_data.chemistry.elements`."""

from __future__ import annotations

import numpy as np

from remedi.latent_evaluation.tmqm_clustering_utils import (
    METAL_CENTRE_ATOMIC_NUMBERS,
    get_block_colors,
    get_tm_colormap,
)


def test_metal_centres_are_the_d_and_f_blocks() -> None:
    assert METAL_CENTRE_ATOMIC_NUMBERS == frozenset(
        [*range(21, 31), *range(39, 49), *range(57, 81), *range(89, 113)]
    )


def test_block_colours_put_lanthanum_with_the_5d_row() -> None:
    assert get_block_colors([26, 46, 57, 77, 60]) == [0, 1, 2, 2, 4]


def test_the_colormap_ramps_the_three_d_rows_and_greys_the_rest() -> None:
    colormap, _ = get_tm_colormap()
    colours = np.asarray(colormap.colors)
    grey = np.full(4, 0.8)
    for atomic_number in (1, 6, 31, 58, 89):
        np.testing.assert_array_equal(colours[atomic_number], grey)
    for atomic_number in (21, 30, 39, 48, 57, 72, 80):
        assert not np.array_equal(colours[atomic_number], grey)
