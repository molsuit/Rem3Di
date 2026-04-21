import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols
from ase.data.colors import jmol_colors
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.io.ase import AseAtomsAdaptor

TM = {
    # transition metals (d‑block)
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Lu",  # sometimes considered d‑block as well
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Lr",  # 103, sometimes placed with d‑block
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Zn",
    "Cd",
    "Hg",
    # lanthanides (f‑block)
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    # actinides (f‑block)
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
}

TM_numbers = set([atomic_numbers[sym] for sym in TM])


def get_coordination_numbers(molecules: list[Atoms]):
    cns = []

    for m in molecules:
        mol = AseAtomsAdaptor.get_molecule(m)
        cn = MinimumDistanceNN(tol=0.20).get_cn(mol, 0)
        cns.append(cn)

    return cns


def get_tm_colormap():
    # 1. Define your “zones”
    Z_green = np.arange(21, 31)  # 21–30
    Z_blue = np.arange(39, 49)  # 29 only
    Z_red = np.r_[57, np.arange(72, 81)]  # 57 and 72–80

    # 2. Sample each gradient from a built‑in cmap
    nG = len(Z_green)
    nB = len(Z_blue)
    nR = len(Z_red)

    # pick a nice slice of each
    greens = plt.cm.Greens(np.linspace(0.4, 0.8, nG))
    blues = plt.cm.Blues(np.linspace(0.4, 0.8, nB))
    reds = plt.cm.Reds(np.linspace(0.4, 0.8, nR))

    # 3. Build a “palette” array indexed by Z up to max you need
    maxZ = 100  # or np.max(your_dataset)
    palette = np.ones((maxZ + 1, 4)) * 0.8  # default light gray for all Z

    # 4. Fill in the three zones
    for i, Z in enumerate(Z_green):
        palette[Z] = greens[i]

    for i, Z in enumerate(Z_blue):
        palette[Z] = blues[i]

    for i, Z in enumerate(Z_red):
        palette[Z] = reds[i]

    custom_cmap = mcolors.ListedColormap(palette, name="Z_map")

    bounds = np.arange(100 + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)

    return custom_cmap, norm


def get_block_colors(atomic_nums):
    blocks = {
        "3d": ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"],
        "4d": ["Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"],
        "5d": ["Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"],
        "f": ["La"],
    }

    back_map = {}

    for i, (_key, val) in enumerate(blocks.items()):
        for v in val:
            back_map[v] = i

    colors = [back_map[chemical_symbols[num]] for num in atomic_nums]

    return colors


def get_metal_center_type(molecules: list[Atoms]) -> list[int]:
    centers = []
    for m in molecules:
        nums = m.get_atomic_numbers()
        # find all in the transition‐metal set
        metals = [num for num in nums if num in TM_numbers]
        if len(metals) != 1:
            raise ValueError(
                f"Expected exactly one TM center in molecule {m}, "
                f"but found: {metals}"
            )
        centers.append(int(metals[0]))
    return centers


def get_atomic_num_colors(atomic_nums: int):
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
