"""The class taxonomy is the one thing every stage agrees on."""

from __future__ import annotations

import pytest

from chiralcat_dataset.taxonomy import (
    CLASS_ORDER,
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    normalize_class,
)


def test_class_order_matches_paper():
    assert CLASS_ORDER == ("achiral", "central", "axial", "helical", "planar")
    assert CLASS_TO_LABEL == {
        "achiral": 0,
        "central": 1,
        "axial": 2,
        "helical": 3,
        "planar": 4,
    }


def test_label_to_class_is_the_inverse_mapping():
    assert {v: k for k, v in CLASS_TO_LABEL.items()} == LABEL_TO_CLASS


def test_normalize_class_maps_synonym():
    assert normalize_class("center") == "central"
    assert normalize_class("axial") == "axial"


def test_normalize_class_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown chiral class"):
        normalize_class("helicoidal")
