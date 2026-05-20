"""Descriptor-probe benchmark: learners, metrics, descriptors, runner.

Reads a prepared zarr + its :class:`BenchmarkManifest` (no registry import),
fits each configured learner on the train split, scores on test per the
manifest's metric. Independent of the legacy ``evaluation/regression/``
framework; the only shared piece is :func:`PydanticResult` for serialization.
"""
