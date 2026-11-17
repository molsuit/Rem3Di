"""Split membership codes: the uint8 code of each split name a table uses."""

from collections.abc import Sequence
from enum import Enum

import numpy as np


class Split(int, Enum):
    """Per-structure split membership as a uint8 code.

    The table's split columns spell the splits out (``train`` / ``valid`` /
    ``test`` / ``unassigned``); the evaluation converts one column to these
    codes to build a partition.
    """

    train = 0
    valid = 1
    test = 2
    unassigned = 255


#: The split names a table's split columns use (``BENCHMARK_DATA_FORMAT.md``
#: §4.2) mapped onto their uint8 codes.
SPLIT_CODE_BY_NAME: dict[str, Split] = {split.name: split for split in Split}


def split_codes_from_names(values: Sequence[str]) -> np.ndarray:
    """Convert table split names to uint8 ``Split`` codes.

    Args:
        values: one split name per structure, each of ``train`` / ``valid`` /
            ``test`` / ``unassigned``.

    Returns:
        A ``uint8`` array of :class:`Split` codes, same length as ``values``.

    Raises:
        ValueError: if any entry is not a known split name; the message names
            the offending values.
    """
    unknown = sorted({str(value) for value in values} - set(SPLIT_CODE_BY_NAME))
    if unknown:
        raise ValueError(
            f"unknown split name(s) {unknown}; expected one of "
            f"{sorted(SPLIT_CODE_BY_NAME)}"
        )
    return np.array(
        [SPLIT_CODE_BY_NAME[str(value)].value for value in values], dtype="u1"
    )
