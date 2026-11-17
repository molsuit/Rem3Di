import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.data.colors import jmol_colors

from remedi_data.chemistry.elements import (
    D_BLOCK_ATOMIC_NUMBERS,
    F_BLOCK_ATOMIC_NUMBERS,
    element_block,
)

#: The metals a tmQM complex is centred on: the d block (groups 3-12) and the
#: f block (lanthanides and actinides).
METAL_CENTRE_ATOMIC_NUMBERS = D_BLOCK_ATOMIC_NUMBERS | F_BLOCK_ATOMIC_NUMBERS

#: The colour index of each block in :func:`get_block_colors`.
_BLOCK_COLOUR_INDEX = {"3d": 0, "4d": 1, "5d": 2, "6d": 3, "4f": 4, "5f": 5}

#: Colour ramp per d-block row in :func:`get_tm_colormap`; other elements grey.
_BLOCK_COLOUR_RAMP = {"3d": plt.cm.Greens, "4d": plt.cm.Blues, "5d": plt.cm.Reds}

#: Atomic numbers the atomic-number colormap covers.
_COLORMAP_MAX_ATOMIC_NUMBER = 100


def get_tm_colormap():
    """A colormap over atomic numbers: one ramp per 3d, 4d and 5d row."""
    palette = np.ones((_COLORMAP_MAX_ATOMIC_NUMBER + 1, 4)) * 0.8
    for block, ramp in _BLOCK_COLOUR_RAMP.items():
        row = [
            number
            for number in range(_COLORMAP_MAX_ATOMIC_NUMBER + 1)
            if element_block(number) == block
        ]
        palette[row] = ramp(np.linspace(0.4, 0.8, len(row)))
    custom_cmap = mcolors.ListedColormap(palette, name="Z_map")
    bounds = np.arange(_COLORMAP_MAX_ATOMIC_NUMBER + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)
    return custom_cmap, norm


def get_block_colors(atomic_nums):
    """The colour index of each metal's d or f block row.

    Raises:
        KeyError: for an element outside the d and f blocks.
    """
    return [_BLOCK_COLOUR_INDEX[element_block(int(number))] for number in atomic_nums]


def get_metal_center_type(molecules: list[Atoms]) -> list[int]:
    centers = []
    for m in molecules:
        nums = m.get_atomic_numbers()
        # find all in the transition-metal set
        metals = [num for num in nums if num in METAL_CENTRE_ATOMIC_NUMBERS]
        if len(metals) != 1:
            raise ValueError(
                f"Expected exactly one TM center in molecule {m}, but found: {metals}"
            )
        centers.append(int(metals[0]))
    return centers


def get_atomic_num_colors(atomic_nums: list[int]):
    colors = [jmol_colors[num] for num in atomic_nums]

    color_atomic_symbols = {
        chemical_symbols[n]: jmol_colors[n] for n in set(atomic_nums)
    }

    handles = []
    for key, val in color_atomic_symbols.items():
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=6,
                markerfacecolor=val,
                markeredgecolor="none",
                label=key,
            )
        )

    return colors, handles
