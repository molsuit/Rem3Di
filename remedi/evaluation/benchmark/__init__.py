"""Descriptor-probe benchmark: learners, metrics, descriptors, runner.

Reads a prepared zarr + the :class:`BenchmarkSpec` copied into it by
``ingest_benchmark`` (no registry import), fits each configured learner on the
train split, scores on test per the spec's headline metric. Independent of the
legacy ``evaluation/regression/`` framework; the only shared piece is
:func:`PydanticResult` for serialization.
"""
