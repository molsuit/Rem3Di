"""Selections: subsets of a dataset as sorted ``structure_id`` files (§5.2.3).

:mod:`config` holds the steps and records, :mod:`order` the seeded drawing
orders and the cluster files they read, :mod:`select` makes, writes and reads
selections. Train on a selection with
``MoleculeDataset.open_structure_arrays(path, structure_ids=read_selection(path, name))``.
"""

from remedi_data.selection.config import (
    ConformersPerMolecule,
    CorpusSelections,
    ExcludeOverlap,
    Filter,
    HierarchicalClusters,
    RandomMolecules,
    SelectionBatchConfig,
    SelectionConfig,
    SelectionError,
    SelectionRecord,
    SelectionStep,
)
from remedi_data.selection.order import (
    ClusterRecord,
    hierarchical_order,
    random_order,
    read_clusters,
    write_clusters,
)
from remedi_data.selection.select import (
    dataset_identity,
    pool_datasets,
    read_selection,
    read_selection_record,
    select_batch,
    select_structures,
)

__all__ = [
    "ClusterRecord",
    "ConformersPerMolecule",
    "CorpusSelections",
    "ExcludeOverlap",
    "Filter",
    "HierarchicalClusters",
    "RandomMolecules",
    "SelectionBatchConfig",
    "SelectionConfig",
    "SelectionError",
    "SelectionRecord",
    "SelectionStep",
    "dataset_identity",
    "hierarchical_order",
    "pool_datasets",
    "random_order",
    "read_clusters",
    "read_selection",
    "read_selection_record",
    "select_batch",
    "select_structures",
    "write_clusters",
]
