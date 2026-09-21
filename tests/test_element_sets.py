"""The named element-set presets resolve to the concrete element sets.

Salvaged from ``test_benchmark_build_config.py`` when ``BenchmarkBuildConfig``
was deleted with the five-generator benchmark build: the presets outlived it —
``FilterMoleculeStageConfig``, ``FilterAtomsStageConfig`` and the bundle's
``GeometryLimits`` all name an :class:`ElementSet` rather than a literal set.
"""

from __future__ import annotations

from remedi.data_handling.dataset.tasks import ElementSet
from remedi.data_handling.dataset_creation.generators.utils import (
    MACE_OFF_ELEMENTS,
    MACE_POLAR_ELEMENTS,
    resolve_element_set,
)


def test_resolve_element_set_maps_to_concrete_sets() -> None:
    assert resolve_element_set(ElementSet.mace_off) is MACE_OFF_ELEMENTS
    assert resolve_element_set(ElementSet.mace_polar) is MACE_POLAR_ELEMENTS


def test_mace_off_is_the_organic_drug_subset() -> None:
    assert MACE_OFF_ELEMENTS == {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}


def test_mace_polar_is_a_strict_superset_of_mace_off() -> None:
    assert MACE_OFF_ELEMENTS < MACE_POLAR_ELEMENTS
