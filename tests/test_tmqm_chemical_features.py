"""Chemical interpretability helpers for tmQM descriptor clusters.

Fast, GPU-free: geometry-only feature extraction on synthetic complexes plus
the chemiscope/fingerprint assembly with the (expensive) UMAP→HDBSCAN step
monkeypatched out.
"""

import numpy as np
import pytest
from ase import Atoms

from threedscriptors.evaluation.descriptor_analysis import analysis_tasks
from threedscriptors.evaluation.descriptor_analysis.analysis_tasks import (
    ChemiscopeClusterTask,
    ClusterAxisAnalysisTask,
    ClusterChemicalFingerprintTask,
    ClusterGranularitySweepTask,
    DescriptorStructureBenchmarkTask,
    _fingerprint_cluster,
    _suggest_label,
    compute_hdbscan_labels,
)
from threedscriptors.evaluation.descriptor_analysis.context import (
    DescriptorAnalysisContext,
)
from threedscriptors.evaluation.descriptor_analysis.tmqm_chemical_features import (
    MetalEnvironmentFeatures,
    compute_metal_environment_features,
    metal_block,
)
from threedscriptors.evaluation.results import ChemiscopeResult


def _ring(z: float, radius: float = 1.21, n: int = 5) -> np.ndarray:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack(
        [radius * np.cos(ang), radius * np.sin(ang), np.full(n, z)]
    )


def _ferrocene() -> Atoms:
    pos = np.vstack([[[0, 0, 0]], _ring(1.65), _ring(-1.65)])
    return Atoms(numbers=[26] + [6] * 10, positions=pos, info={"total_charge": 0.0})


def _octahedral(metal_z: int, donor_z: int, d: float = 2.1) -> Atoms:
    pos = np.array(
        [[0, 0, 0], [d, 0, 0], [-d, 0, 0], [0, d, 0], [0, -d, 0], [0, 0, d], [0, 0, -d]]
    )
    return Atoms(
        numbers=[metal_z] + [donor_z] * 6, positions=pos, info={"total_charge": 0.0}
    )


def _phosphine() -> Atoms:
    # Pd with 4 ~tetrahedral P donors.
    t = 1.6
    pos = np.array(
        [[0, 0, 0], [t, t, t], [t, -t, -t], [-t, t, -t], [-t, -t, t]]
    )
    return Atoms(numbers=[46] + [15] * 4, positions=pos, info={"total_charge": 0.0})


def _carborane() -> Atoms:
    rng = np.random.default_rng(0)
    cage = rng.normal(scale=1.0, size=(10, 3)) + np.array([3.0, 0, 0])
    pos = np.vstack([[[0, 0, 0]], [[2.0, 0, 0], [2.0, 1.4, 0]], cage])
    return Atoms(
        numbers=[77] + [6, 6] + [5] * 10, positions=pos, info={"total_charge": 0.0}
    )


def test_metal_block_rows():
    assert metal_block(26) == "3d"  # Fe
    assert metal_block(46) == "4d"  # Pd
    assert metal_block(77) == "5d"  # Ir
    assert metal_block(57) == "5d"  # La (grouped with 5d here)
    assert metal_block(60) == "4f"  # Nd
    assert metal_block(6) == "other"


def test_ferrocene_is_sandwich():
    (f,) = compute_metal_environment_features([_ferrocene()])
    assert f.metal_symbol == "Fe"
    assert f.metal_block == "3d"
    assert f.coordination_number == 10
    assert f.donor_set == "C10"
    assert f.hapticity_max == 5
    assert f.n_pi_groups == 2
    assert f.is_sandwich is True
    assert f.is_carborane is False
    assert f.formula == "C10Fe"


def test_oxygen_and_nitrogen_donor_octahedra():
    aqua, diimine = compute_metal_environment_features(
        [_octahedral(26, 8), _octahedral(26, 7)]
    )
    assert aqua.coordination_number == 6
    assert aqua.donor_set == "O6"
    assert aqua.n_donor_O == 6 and aqua.is_sandwich is False
    assert diimine.donor_set == "N6" and diimine.n_donor_N == 6


def test_phosphine_and_carborane_flags():
    phos, carb = compute_metal_environment_features([_phosphine(), _carborane()])
    assert phos.metal_symbol == "Pd"
    assert phos.n_donor_P == 4 and phos.donor_set == "P4"
    assert carb.metal_symbol == "Ir"
    assert carb.n_B == 10
    assert carb.is_carborane is True


def test_features_robust_without_unique_metal():
    # Pure organic fragment: no TM center → composition still valid.
    organic = Atoms(numbers=[6, 6, 8], positions=[[0, 0, 0], [1.4, 0, 0], [2.6, 0, 0]])
    (f,) = compute_metal_environment_features([organic])
    assert f.metal_symbol == "?"
    assert f.coordination_number == -1
    assert f.donor_set == ""
    assert f.n_C == 2 and f.n_O == 1 and f.n_atoms == 3


def _mef(**kw) -> MetalEnvironmentFeatures:
    base = dict(
        metal_symbol="Fe",
        metal_block="3d",
        coordination_number=6,
        donor_set="N6",
        donor_counts={"N": 6},
        n_donor_C=0,
        n_donor_N=6,
        n_donor_O=0,
        n_donor_P=0,
        n_donor_S=0,
        n_donor_halogen=0,
        hapticity_max=0,
        n_pi_groups=0,
        n_atoms=20,
        formula="C10FeN6",
        n_B=0,
        n_C=10,
        n_N=6,
        n_O=0,
        n_P=0,
        n_S=0,
        n_halogen=0,
        is_sandwich=False,
        is_carborane=False,
    )
    base.update(kw)
    return MetalEnvironmentFeatures(**base)


@pytest.mark.parametrize(
    ("kw", "expected"),
    [
        ({"is_carborane": True, "n_B": 10}, "carborane / borane-cage"),
        (
            {"is_sandwich": True, "n_pi_groups": 2, "hapticity_max": 5,
             "donor_set": "C10"},
            "sandwich (bis-π)",
        ),
        (
            {"donor_set": "P4", "coordination_number": 4},
            "phosphine / P-donor [P4 100%]",
        ),
        ({"donor_set": "O6"}, "oxygen-donor [O6 100%]"),
        ({"donor_set": "N6"}, "N-donor octahedral (diimine?) [N6 100%]"),
    ],
)
def test_suggest_label_rules(kw, expected):
    members = [_mef(**kw) for _ in range(4)]
    fp = _fingerprint_cluster(0, members, top_k=5)
    assert fp.suggested_label == expected
    # Pure cluster ⇒ modal fraction 1.0 and zero donor-set entropy.
    if fp.modal_donor_set:
        assert fp.modal_donor_set_frac == 1.0
        assert fp.donor_set_entropy_bits == 0.0
    assert _suggest_label(
        fp.modal_donor_set,
        fp.modal_donor_set_frac,
        members[0].coordination_number,
        fp.frac_sandwich,
        fp.frac_carborane,
        fp.mean_hapticity_max,
    ) == expected


def test_mixed_cluster_is_not_mislabeled():
    # Heterogeneous donor sets ⇒ no dominant motif ⇒ flagged as a mixture,
    # not collapsed to the most-frequent element (the old bug).
    members = (
        [_mef(donor_set="O6")] * 2
        + [_mef(donor_set="P4")] * 2
        + [_mef(donor_set="N4")] * 2
    )
    fp = _fingerprint_cluster(0, members, top_k=5)
    assert fp.suggested_label.startswith("mixed (")
    assert fp.modal_donor_set_frac < 0.5
    assert fp.donor_set_entropy_bits > 1.0


def _make_ctx(molecules: list[Atoms]) -> DescriptorAnalysisContext:
    class _FakeDataset:
        def get_all_molecules(self):
            return molecules

    n = len(molecules)
    return DescriptorAnalysisContext(
        dataset=_FakeDataset(),
        descriptors=np.random.rand(n, 8),
        descriptors_raw=np.random.rand(n, 8),
        projection=np.random.rand(n, 2),
        cache={},
    )


def test_chemiscope_cluster_task_bundles_clusterings_and_chemistry(monkeypatch):
    mols = [
        _ferrocene(),
        _octahedral(26, 8),
        _octahedral(26, 7),
        _phosphine(),
        _carborane(),
        _octahedral(46, 8),
    ]
    labels = np.array([0, 0, 1, 1, -1, 0])
    monkeypatch.setattr(
        analysis_tasks, "compute_hdbscan_labels", lambda *a, **k: labels
    )

    ctx = _make_ctx(mols)
    task = ChemiscopeClusterTask(
        min_cluster_sizes=[50, 200], default_color_mcs=200
    )
    (result,) = task.run(ctx)

    assert isinstance(result, ChemiscopeResult)
    data = result.data
    assert len(data["structures"]) == len(mols)
    props = data["properties"]
    for key in ("cluster_mcs50", "cluster_mcs200", "metal",
                "is_sandwich", "coordination_number"):
        assert key in props
    # Heavy / high-cardinality columns are dropped by default.
    assert "donor_set" not in props
    assert "formula" not in props
    # Cluster ids are stored as numerics (-1 = noise; chemiscope coerces ints
    # to floats internally) so the viewer treats them as an uncapped color
    # axis instead of a categorical one capped at ~5 unique values.
    cluster_values = props["cluster_mcs200"]["values"]
    assert cluster_values == [0, 0, 1, 1, -1, 0]
    assert all(not isinstance(v, str) for v in cluster_values)
    assert data["settings"]["map"]["color"]["property"] == "cluster_mcs200"
    # Features computed once and cached for downstream tasks.
    assert "metal_env_features" in ctx.cache


def test_chemiscope_heavy_properties_opt_in(monkeypatch):
    mols = [_octahedral(26, 8), _phosphine()]
    monkeypatch.setattr(
        analysis_tasks, "compute_hdbscan_labels",
        lambda *a, **k: np.array([0, 0]),
    )
    ctx = _make_ctx(mols)
    (result,) = ChemiscopeClusterTask(
        min_cluster_sizes=[50], include_heavy_properties=True
    ).run(ctx)
    props = result.data["properties"]
    assert "donor_set" in props and "formula" in props and "n_P" in props


def test_cluster_fingerprint_task_outputs_labels(monkeypatch):
    mols = [_octahedral(26, 8) for _ in range(3)] + [_phosphine() for _ in range(3)]
    labels = np.array([0, 0, 0, 1, 1, 1])
    monkeypatch.setattr(
        analysis_tasks, "compute_hdbscan_labels", lambda *a, **k: labels
    )

    ctx = _make_ctx(mols)
    (result,) = ClusterChemicalFingerprintTask(min_cluster_size=2).run(ctx)
    report = result.obj
    assert report.n_clusters == 2
    assert report.mean_donor_set_purity == 1.0
    assert report.mean_donor_set_entropy_bits == 0.0
    by_id = {c.cluster_id: c for c in report.clusters}
    assert by_id[0].suggested_label.startswith("oxygen-donor [O6")
    assert by_id[1].suggested_label.startswith("phosphine / P-donor [P4")


def test_granularity_sweep_recommends_highest_purity(monkeypatch):
    # mcs=25 → two pure clusters; mcs=50 → one mixed cluster; mcs=100 → all noise.
    mols = [_octahedral(26, 8) for _ in range(4)] + [_phosphine() for _ in range(2)]
    by_mcs = {
        25: np.array([0, 0, 0, 0, 1, 1]),
        50: np.array([0, 0, 0, 0, 0, 0]),
        100: np.array([-1, -1, -1, -1, -1, -1]),
    }
    monkeypatch.setattr(
        analysis_tasks,
        "compute_hdbscan_labels",
        lambda *a, **k: by_mcs[k["min_cluster_size"]],
    )

    ctx = _make_ctx(mols)
    (result,) = ClusterGranularitySweepTask(
        min_cluster_sizes=[25, 50, 100]
    ).run(ctx)
    report = result.obj

    rows = {r.min_cluster_size: r for r in report.rows}
    assert report.reducer == "umap"
    assert report.recommended_min_cluster_size == 25
    assert rows[25].recommended is True
    assert rows[25].mean_donor_set_purity == 1.0
    assert rows[25].mean_donor_set_entropy_bits == 0.0
    # mcs=50 lumps O6 + P4 into one cluster: purity = 4/6.
    assert rows[50].n_clusters == 1
    assert abs(rows[50].mean_donor_set_purity - 4 / 6) < 1e-9
    assert rows[50].mean_donor_set_entropy_bits > 0.0


def test_compute_hdbscan_labels_no_umap_runs_on_raw_descriptors():
    # Two well-separated blobs in the 64-D-like space → reducer="none" must
    # recover ≥2 clusters with little noise, without touching UMAP.
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.05, size=(40, 8))
    b = rng.normal(5.0, 0.05, size=(40, 8))
    desc = np.vstack([a, b])
    ctx = DescriptorAnalysisContext(
        dataset=None,
        descriptors=desc,
        descriptors_raw=desc,
        projection=None,
        cache={},
    )
    labels = compute_hdbscan_labels(
        ctx,
        reducer="none",
        cluster_n_components=15,
        cluster_n_neighbors=30,
        cluster_min_dist=0.0,
        cluster_metric="cosine",
        random_state=0,
        min_cluster_size=5,
        min_samples=None,
        cluster_selection_epsilon=0.0,
    )
    assert labels.shape == (80,)
    assert np.unique(labels[labels >= 0]).size >= 2
    assert np.mean(labels == -1) < 0.2


# --- new chemical-feature tests --------------------------------------------


def test_hexaaqua_is_octahedral_with_radius_and_no_carbonyls():
    (f,) = compute_metal_environment_features([_octahedral(26, 8)])
    assert f.geometry_class == "octahedral_like"
    assert f.n_carbonyl == 0
    assert f.n_chelate_rings == 0  # 6 isolated O donors, no backbone between them
    assert f.radius_of_gyration > 0.0


def test_tetrahedral_phosphine_geometry():
    (f,) = compute_metal_environment_features([_phosphine()])
    assert f.geometry_class == "tetrahedral"
    assert f.n_donor_P == 4
    assert f.n_carbonyl == 0


def _hexacarbonyl_metal(metal_z: int = 24) -> Atoms:
    # Cr(CO)_6: 6 CO ligands along +/- x/y/z. Cr-C ~1.92, C-O ~1.14 (terminal CO).
    mc = 1.92
    co = 1.14
    numbers = [metal_z]
    positions = [[0, 0, 0]]
    for vec in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        numbers.extend([6, 8])
        positions.append([mc * vec[0], mc * vec[1], mc * vec[2]])
        positions.append([(mc + co) * vec[0], (mc + co) * vec[1], (mc + co) * vec[2]])
    return Atoms(numbers=numbers, positions=positions, info={"total_charge": 0.0})


def test_metal_hexacarbonyl_counts_six_terminal_co_ligands():
    (f,) = compute_metal_environment_features([_hexacarbonyl_metal(24)])
    assert f.metal_symbol == "Cr"
    assert f.coordination_number == 6
    assert f.donor_set == "C6"
    assert f.n_carbonyl == 6
    assert f.geometry_class == "octahedral_like"
    assert f.n_chelate_rings == 0  # individual COs are monodentate


def _diaminoethane_complex() -> Atoms:
    # Fe(en) fragment: a 5-membered chelate ring M(Fe)-N-C-C-N back to M.
    # Place atoms so each step is ~1.5 Å, well below the 1.85 heavy-bond cutoff.
    return Atoms(
        numbers=[26, 7, 6, 6, 7],
        positions=[
            [0.0, 0.0, 0.0],          # Fe
            [1.9, 0.0, 0.0],          # N
            [2.45, 1.40, 0.0],        # C
            [1.40, 2.45, 0.0],        # C
            [0.0, 1.9, 0.0],          # N
        ],
        info={"total_charge": 0.0},
    )


def test_diaminoethane_forms_five_membered_chelate_ring():
    (f,) = compute_metal_environment_features([_diaminoethane_complex()])
    assert f.coordination_number == 2
    assert f.n_donor_N == 2
    assert f.n_chelate_rings == 1
    assert f.max_chelate_ring_size == 5
    assert f.chelate_ring_sizes == [5]


def test_descriptor_structure_benchmark_scorecard(monkeypatch):
    # Two clusters that perfectly partition by every chemistry axis: AMI ≈ 1.
    mols = [_ferrocene() for _ in range(4)] + [_phosphine() for _ in range(4)]
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    monkeypatch.setattr(
        analysis_tasks, "compute_hdbscan_labels", lambda *a, **k: labels
    )
    ctx = _make_ctx(mols)
    (result,) = DescriptorStructureBenchmarkTask(
        canonical_min_cluster_size=2,
        sweep_min_cluster_sizes=[2, 4],
        pure_island_min_size=2,
    ).run(ctx)
    report = result.obj

    # Protocol is fully recorded for reproducibility / cross-model comparison.
    assert report.protocol.version != ""
    assert report.protocol.canonical_min_cluster_size == 2
    assert report.protocol.random_state == 0
    assert set(report.protocol.chemistry_axes) == {
        "donor_set",
        "ligand_motif",
        "geometry_class",
        "dominant_donor_element",
    }

    # On a perfect partition every chemistry axis hits AMI ≈ 1.
    canonical = report.canonical
    for axis in ("metal_block", "ligand_motif", "geometry_class", "donor_set"):
        assert canonical.ami_per_axis[axis] > 0.99, (
            f"{axis}: {canonical.ami_per_axis[axis]}"
        )
    assert abs(report.aggregate_chemistry_ami - 1.0) < 1e-6

    # 2 clusters x 4 members at purity 1.0 -> 2 pure islands.
    assert report.n_pure_islands["total"] == 2
    # The sweep is a separate Pareto, not the headline; both mcs values appear.
    assert {r.min_cluster_size for r in report.sweep} == {2, 4}


def test_descriptor_structure_benchmark_records_raw_variant(monkeypatch):
    # The euclidean-on-raw companion benchmark must flag use_raw_descriptors
    # in its protocol so it's never silently compared against the canonical
    # cosine scorecard.
    mols = [_ferrocene() for _ in range(4)] + [_phosphine() for _ in range(4)]
    monkeypatch.setattr(
        analysis_tasks,
        "compute_hdbscan_labels",
        lambda *a, **k: np.array([0, 0, 0, 0, 1, 1, 1, 1]),
    )
    ctx = _make_ctx(mols)
    (result,) = DescriptorStructureBenchmarkTask(
        cluster_metric="euclidean",
        use_raw_descriptors=True,
        canonical_min_cluster_size=2,
        sweep_min_cluster_sizes=[2],
        pure_island_min_size=2,
    ).run(ctx)
    report = result.obj
    assert report.protocol.use_raw_descriptors is True
    assert report.protocol.cluster_metric == "euclidean"


def test_descriptor_structure_benchmark_rejects_unknown_axis(monkeypatch):
    monkeypatch.setattr(
        analysis_tasks, "compute_hdbscan_labels",
        lambda *a, **k: np.array([0, 0, 1, 1]),
    )
    ctx = _make_ctx([_ferrocene(), _ferrocene(), _phosphine(), _phosphine()])
    task = DescriptorStructureBenchmarkTask(
        chemistry_axes=["donor_set", "bogus_axis"],
        canonical_min_cluster_size=2,
        sweep_min_cluster_sizes=[2],
        pure_island_min_size=2,
    )
    with pytest.raises(ValueError, match="bogus_axis"):
        task.run(ctx)


def test_axis_analysis_identifies_best_explaining_axis(monkeypatch):
    # cluster 0: four Fe ferrocenes → pure on metal/ligand_motif (sandwich)
    # cluster 1: two octahedral Fe(OH2)_6 → pure on geometry / donor element
    mols = [_ferrocene() for _ in range(4)] + [_octahedral(26, 8) for _ in range(2)]
    labels = np.array([0, 0, 0, 0, 1, 1])
    monkeypatch.setattr(
        analysis_tasks, "compute_hdbscan_labels", lambda *a, **k: labels
    )
    ctx = _make_ctx(mols)
    (result,) = ClusterAxisAnalysisTask(min_cluster_size=2).run(ctx)
    report = result.obj

    assert report.reducer == "umap"
    assert report.n_clusters == 2
    axes_by_name = {a.name: a for a in report.axes}
    # Top-ranked axis must hit perfect purity on these synthetic groups.
    assert report.axes[0].weighted_mean_purity == 1.0
    # Metal block / motif / geometry should all max out (all members identical
    # on those axes); donor_set must too (C10 for ferrocene, O6 for aqua).
    for name in ("metal_block", "ligand_motif", "geometry_class", "donor_set"):
        assert axes_by_name[name].weighted_mean_purity == 1.0
    # Per-cluster best axis is assigned.
    by_id = {c.cluster_id: c for c in report.clusters}
    assert by_id[0].best_purity == 1.0
    assert by_id[1].best_purity == 1.0
