"""The split names a table uses and the uint8 codes a partition is built from."""

from __future__ import annotations

import numpy as np
import pytest

from remedi_data.format import SPLIT_VALUES
from remedi_data.format.split_codes import (
    SPLIT_CODE_BY_NAME,
    Split,
    split_codes_from_names,
)


def test_the_split_names_are_the_split_members() -> None:
    assert set(SPLIT_CODE_BY_NAME) == SPLIT_VALUES == {split.name for split in Split}
    assert all(SPLIT_CODE_BY_NAME[split.name] is split for split in Split)


def test_split_names_become_codes_and_unknown_names_are_refused() -> None:
    codes = split_codes_from_names(["train", "test", "unassigned", "valid"])
    np.testing.assert_array_equal(codes, np.array([0, 2, 255, 1], dtype="u1"))
    with pytest.raises(ValueError, match="TEST"):
        split_codes_from_names(["train", "TEST"])
