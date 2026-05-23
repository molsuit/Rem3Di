"""BenchmarkManifest sidecar + discovery for self-describing zarr benchmarks.

This is the eval-time interface: walk a root and read each zarr's manifest
(no registry import). Pins yaml round-trip, the from-benchmark factory, and
the silent skip of non-benchmark subdirs so the eval root may hold unrelated
zarrs without crashing discovery.
"""

from __future__ import annotations

from pathlib import Path

from threedscriptors.data_handling.benchmarks import (
    BENCHMARK_MANIFEST_FILENAME,
    BenchmarkManifest,
    EvalMetric,
    SplitVariant,
    discover_benchmark_zarrs,
    get_benchmark,
)


def test_from_benchmark_carries_eval_fields() -> None:
    bench = get_benchmark("esol")
    manifest = BenchmarkManifest.from_benchmark(bench)
    assert manifest.dataset_id == "esol"
    assert manifest.metric is EvalMetric.rmse
    assert manifest.split_variant is SplitVariant.scaffold
    assert manifest.source == "moleculenet"

    tdc = BenchmarkManifest.from_benchmark(get_benchmark("LD50_Zhu"))
    assert tdc.source == "tdc"
    assert tdc.split_variant is SplitVariant.tdc_default


def test_yaml_roundtrip_per_zarr_dir(tmp_path: Path) -> None:
    dst = tmp_path / "esol"
    dst.mkdir()
    src = BenchmarkManifest.from_benchmark(get_benchmark("esol"))
    src.to_zarr_dir(dst)
    assert (dst / BENCHMARK_MANIFEST_FILENAME).is_file()
    assert BenchmarkManifest.from_zarr_dir(dst) == src


def test_discover_walks_root_and_skips_non_benchmarks(tmp_path: Path) -> None:
    # Two valid benchmark dirs + one unrelated subdir without a manifest.
    for did in ("esol", "LD50_Zhu"):
        sub = tmp_path / did
        sub.mkdir()
        BenchmarkManifest.from_benchmark(get_benchmark(did)).to_zarr_dir(sub)
    (tmp_path / "unrelated").mkdir()
    (tmp_path / "unrelated" / "some_file.txt").write_text("x")

    found = discover_benchmark_zarrs(tmp_path)
    ids = sorted(m.dataset_id for _, m in found)
    paths = sorted(p.name for p, _ in found)
    assert ids == ["LD50_Zhu", "esol"]
    assert paths == ["LD50_Zhu", "esol"]
    assert "unrelated" not in paths


def test_discover_missing_root_returns_empty(tmp_path: Path) -> None:
    assert discover_benchmark_zarrs(tmp_path / "nope") == []
