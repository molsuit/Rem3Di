"""The class taxonomy is the one thing every stage agrees on."""

from __future__ import annotations

import pytest

from chiralcat_dataset.taxonomy import CLASS_TO_LABEL, normalize_class


def test_labels_match_the_paper():
    assert CLASS_TO_LABEL == {
        "achiral": 0,
        "central": 1,
        "axial": 2,
        "helical": 3,
        "planar": 4,
    }


def test_normalize_class_maps_the_synonym_and_rejects_unknown_classes():
    assert normalize_class("center") == "central"
    assert normalize_class("axial") == "axial"
    with pytest.raises(ValueError, match="Unknown chiral class"):
        normalize_class("helicoidal")
