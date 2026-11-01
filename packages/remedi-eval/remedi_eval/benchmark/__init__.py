"""Descriptor-probe benchmark cells: learners, metrics, pairwise ranking, scoring.

Reads a built dataset (zarr plus the ``DatasetSpec`` beside it), fits each
configured learner on the train split, and scores the test split with the
spec's headline metric. The framework's ``benchmark_panel`` task composes these
cells into a panel.
"""
