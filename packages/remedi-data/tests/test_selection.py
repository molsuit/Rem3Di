"""Selections of a dataset (§5.2.3): steps, nesting, records and gathering structures."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from remedi_data.build.write import write_dataset
from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_data.dataset.structures import Structures
from remedi_data.selection import (
    ClusterRecord,
    ConformersPerMolecule,
    ExcludeOverlap,
    Filter,
    HierarchicalClusters,
    RandomMolecules,
    SelectionBatchConfig,
    SelectionConfig,
    SelectionError,
    dataset_identity,
    hierarchical_order,
    random_order,
    read_selection,
    read_selection_record,
    select_batch,
    select_structures,
    write_clusters,
)
from remedi_data.selection.config import ClusterWeighting
from remedi_data.selection.select import selection_paths
from remedi_data.yaml_io import read_model
from testing import (
    SMALL_ZARR_LAYOUT,
    embed_frame,
    make_dataset_spec,
    make_dataset_table,
    make_provenance,
    write_small_dataset,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

CORPUS_SMILES = [
    "c1ccccc1",
    "c1ccccc1O",
    "c1ccncc1",
    "C1CCCCC1",
    "CCO",
    "CCC",
    "CC(=O)O",
    "CCN",
]
CONFORMERS = 3


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Eight molecules with three conformers each, and a conformer_rank column."""
    smiles = [text for text in CORPUS_SMILES for _ in range(CONFORMERS)]
    ranks = [rank for _ in CORPUS_SMILES for rank in range(CONFORMERS)]
    spec = make_dataset_spec(dataset_id="corpus", labels=[], metrics=None)
    table = make_dataset_table(spec, smiles, None, None)
    spec = spec.model_copy(update={"extra_columns": ["conformer_rank", "weight"]})
    table["conformer_rank"] = np.asarray(ranks, dtype=np.int64)
    table["weight"] = [0.6, 0.3, 0.1] * len(CORPUS_SMILES)
    frames = [
        embed_frame(text, seed=rank + 1)
        for text, rank in zip(smiles, ranks, strict=True)
    ]
    directory = tmp_path_factory.mktemp("corpus") / "corpus"
    write_dataset(
        spec,
        table[spec.expected_columns()],
        Structures.from_atoms(frames),
        make_provenance("corpus", source_molecules=len(smiles)),
        directory,
        layout=SMALL_ZARR_LAYOUT,
    )
    return directory


@pytest.fixture(scope="module")
def benchmark_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A root holding one benchmark: toluene (scaffold benzene) and ethanol."""
    root = tmp_path_factory.mktemp("benchmarks")
    write_small_dataset(
        root,
        dataset_id="benchmark",
        labels=[],
        metrics=None,
        targets=None,
        smiles=["Cc1ccccc1", "CCO"],
    )
    return root


def _table(corpus: Path) -> pl.DataFrame:
    return pl.read_parquet(corpus / "table.parquet")


def _selected_rows(corpus: Path, name: str) -> pl.DataFrame:
    structure_ids = read_selection(corpus, name)
    return _table(corpus).filter(pl.col("structure_id").is_in(structure_ids))


def _select(corpus: Path, name: str, *steps) -> pl.DataFrame:
    select_structures(corpus, SelectionConfig(name=name, steps=list(steps)))
    return _selected_rows(corpus, name)


def test_random_draws_of_one_seed_are_nested_and_take_whole_molecules(
    corpus: Path,
) -> None:
    molecule_sets = []
    for count in (2, 4, 6):
        rows = _select(corpus, f"random_{count}", RandomMolecules(count=count, seed=3))
        assert rows["molecule_id"].n_unique() == count
        assert rows.height == count * CONFORMERS
        molecule_sets.append(set(rows["molecule_id"].to_list()))
    assert molecule_sets[0] < molecule_sets[1] < molecule_sets[2]
    half = _select(corpus, "random_half", RandomMolecules(fraction=0.5, seed=3))
    assert set(half["molecule_id"].to_list()) == molecule_sets[1]
    other_seed = _select(corpus, "random_4_seed_4", RandomMolecules(count=4, seed=4))
    assert set(other_seed["molecule_id"].to_list()) != molecule_sets[1]


def test_removing_molecules_does_not_reshuffle_the_random_order() -> None:
    molecule_ids = np.arange(1000)
    full_order = random_order(molecule_ids, seed=0)
    kept = molecule_ids[molecule_ids % 3 != 0]
    np.testing.assert_array_equal(
        random_order(kept, seed=0), full_order[full_order % 3 != 0]
    )


def test_a_draw_larger_than_what_is_left_fails(corpus: Path) -> None:
    with pytest.raises(SelectionError, match="only 8 are left"):
        select_structures(
            corpus,
            SelectionConfig(name="too_many", steps=[RandomMolecules(count=9)]),
        )


def test_conformers_per_molecule_keeps_the_best_ranked(corpus: Path) -> None:
    top = _select(
        corpus, "top1", ConformersPerMolecule(rank_by="conformer_rank", count=1)
    )
    assert top.height == len(CORPUS_SMILES)
    assert set(top["conformer_rank"].to_list()) == {0}
    heaviest = _select(
        corpus,
        "top2_by_weight",
        ConformersPerMolecule(rank_by="weight", descending=True, count=2),
    )
    assert sorted(set(heaviest["weight"].to_list())) == [0.3, 0.6]
    weighty = _select(
        corpus, "weight_at_least", ConformersPerMolecule(rank_by="weight", minimum=0.3)
    )
    assert weighty.height == 2 * len(CORPUS_SMILES)


def test_a_step_on_a_missing_column_fails(corpus: Path) -> None:
    with pytest.raises(SelectionError, match="boltzmann_weight"):
        select_structures(
            corpus,
            SelectionConfig(
                name="missing",
                steps=[ConformersPerMolecule(rank_by="boltzmann_weight", count=1)],
            ),
        )


def test_exclude_overlap_on_smiles_drops_exact_matches(
    corpus: Path, benchmark_root: Path
) -> None:
    rows = _select(corpus, "no_overlap", ExcludeOverlap(datasets=[benchmark_root]))
    smiles = set(rows["nonisomeric_smiles"].to_list())
    assert "CCO" not in smiles
    assert len(smiles) == len(CORPUS_SMILES) - 1
    record = read_selection_record(corpus, "no_overlap")
    assert [entry.path for entry in record.inputs] == [benchmark_root / "benchmark"]


def test_exclude_overlap_on_scaffolds_drops_shared_rings_and_exact_acyclic_matches(
    corpus: Path, benchmark_root: Path
) -> None:
    rows = _select(
        corpus,
        "no_scaffold_overlap",
        ExcludeOverlap(
            datasets=[benchmark_root / "benchmark"], match_on="murcko_scaffold"
        ),
    )
    assert set(rows["nonisomeric_smiles"].to_list()) == {
        "c1ccncc1",
        "C1CCCCC1",
        "CCC",
        "CC(=O)O",
        "CCN",
    }


def test_exclude_overlap_refuses_a_path_without_datasets(
    corpus: Path, tmp_path: Path
) -> None:
    with pytest.raises(SelectionError, match="holds no datasets"):
        select_structures(
            corpus,
            SelectionConfig(
                name="nothing", steps=[ExcludeOverlap(datasets=[tmp_path])]
            ),
        )


def test_filter_can_use_the_atom_count(corpus: Path) -> None:
    rows = _select(corpus, "small", Filter(expression="atom_count <= 9"))
    assert set(rows["nonisomeric_smiles"].to_list()) == {"CCO", "CC(=O)O"}
    with pytest.raises(SelectionError, match="filter"):
        select_structures(
            corpus,
            SelectionConfig(name="broken", steps=[Filter(expression="no_such > 1")]),
        )


def test_an_empty_selection_fails(corpus: Path) -> None:
    with pytest.raises(SelectionError, match="empty"):
        select_structures(
            corpus,
            SelectionConfig(name="empty", steps=[Filter(expression="atom_count > 99")]),
        )


def test_steps_chain_and_their_counts_are_recorded(
    corpus: Path, benchmark_root: Path
) -> None:
    record = select_structures(
        corpus,
        SelectionConfig(
            name="chained",
            steps=[
                ExcludeOverlap(datasets=[benchmark_root]),
                ConformersPerMolecule(rank_by="conformer_rank", count=1),
                RandomMolecules(count=3),
            ],
        ),
    )
    assert [(step.structures_out, step.molecules_out) for step in record.steps] == [
        (21, 7),
        (7, 7),
        (3, 3),
    ]
    assert record.structures == 3
    pointer = np.asarray(MoleculeDataset.open_existing_dataset_from_dir(corpus).ptr[:])
    structure_ids = read_selection(corpus, "chained")
    assert record.atoms == int(
        np.sum(pointer[structure_ids + 1] - pointer[structure_ids])
    )
    identity = dataset_identity(corpus)
    assert record.table_content_sha256 == identity.table_content_sha256


def test_an_unchanged_selection_is_kept_and_a_changed_one_needs_overwrite(
    corpus: Path,
) -> None:
    config = SelectionConfig(name="rerun", steps=[RandomMolecules(count=2)])
    first = select_structures(corpus, config)
    assert select_structures(corpus, config).created_at == first.created_at
    changed = SelectionConfig(name="rerun", steps=[RandomMolecules(count=3)])
    with pytest.raises(SelectionError, match="overwrite"):
        select_structures(corpus, changed)
    assert select_structures(corpus, changed, overwrite=True).molecules == 3


def test_a_selection_of_another_build_or_a_changed_file_is_refused(
    corpus: Path,
) -> None:
    select_structures(
        corpus, SelectionConfig(name="stale", steps=[RandomMolecules(count=2)])
    )
    table_path, record_path = selection_paths(corpus, "stale")
    record = yaml.safe_load(record_path.read_text())
    record["structures_sha256"] = "0" * 64
    record_path.write_text(yaml.safe_dump(record))
    with pytest.raises(SelectionError, match="another build"):
        read_selection(corpus, "stale")

    select_structures(
        corpus,
        SelectionConfig(name="tampered", steps=[RandomMolecules(count=2)]),
    )
    table_path, _ = selection_paths(corpus, "tampered")
    pl.DataFrame({"structure_id": np.array([0], dtype=np.int64)}).write_parquet(
        table_path
    )
    with pytest.raises(SelectionError, match="does not match"):
        read_selection(corpus, "tampered")
    with pytest.raises(SelectionError, match="no complete selection"):
        read_selection(corpus, "never_made")


def _cluster_record(corpus: Path, name: str, table_hash: str | None = None):
    return ClusterRecord(
        name=name,
        dataset_id="corpus",
        table_content_sha256=table_hash
        or dataset_identity(corpus).table_content_sha256,
        fingerprint="test",
        method="hand-made",
        coarse_cutoff=0.65,
        fine_cutoff=0.4,
        molecules=8,
        coarse_clusters=2,
        fine_clusters=4,
        created_at=datetime.now(UTC),
        elapsed_seconds=0.0,
    )


def test_a_hierarchical_draw_reads_the_cluster_file(corpus: Path) -> None:
    molecule_ids = np.arange(8)
    assignments = pl.DataFrame(
        {
            "molecule_id": molecule_ids,
            "coarse_cluster": molecule_ids // 4,
            "fine_cluster": molecule_ids // 2,
        }
    )
    write_clusters(corpus, assignments, _cluster_record(corpus, "rings"))
    rows = _select(
        corpus,
        "diverse_2",
        HierarchicalClusters(clusters="rings", count=2, cluster_weighting="uniform"),
    )
    coarse = set((np.array(rows["molecule_id"].unique().to_list()) // 4).tolist())
    assert coarse == {0, 1}

    write_clusters(corpus, assignments, _cluster_record(corpus, "stale", "f" * 64))
    with pytest.raises(SelectionError, match="recluster"):
        select_structures(
            corpus,
            SelectionConfig(
                name="from_stale",
                steps=[HierarchicalClusters(clusters="stale", count=2)],
            ),
        )


def test_reclustering_under_the_same_name_makes_the_selection_stale(
    corpus: Path,
) -> None:
    molecule_ids = np.arange(8)
    first = pl.DataFrame(
        {
            "molecule_id": molecule_ids,
            "coarse_cluster": molecule_ids // 4,
            "fine_cluster": molecule_ids // 2,
        }
    )
    write_clusters(corpus, first, _cluster_record(corpus, "regrouped"))
    config = SelectionConfig(
        name="from_regrouped",
        steps=[
            HierarchicalClusters(
                clusters="regrouped", count=2, cluster_weighting="uniform"
            )
        ],
    )
    made = select_structures(corpus, config)
    assert select_structures(corpus, config).created_at == made.created_at

    second = first.with_columns(
        coarse_cluster=pl.col("molecule_id") % 2, fine_cluster=pl.col("molecule_id")
    )
    write_clusters(corpus, second, _cluster_record(corpus, "regrouped"))
    with pytest.raises(SelectionError, match="inputs"):
        read_selection(corpus, "from_regrouped")
    with pytest.raises(SelectionError, match="overwrite"):
        select_structures(corpus, config)
    remade = select_structures(corpus, config, overwrite=True)
    assert remade.inputs != made.inputs
    assert len(read_selection(corpus, "from_regrouped")) == remade.structures


def test_a_rebuilt_overlap_pool_makes_the_selection_stale(
    corpus: Path, tmp_path: Path
) -> None:
    def write_pool(smiles: list[str]) -> None:
        write_small_dataset(
            tmp_path,
            dataset_id="pool",
            labels=[],
            metrics=None,
            targets=None,
            smiles=smiles,
        )

    write_pool(["CCO"])
    config = SelectionConfig(
        name="without_pool", steps=[ExcludeOverlap(datasets=[tmp_path / "pool"])]
    )
    made = select_structures(corpus, config)
    assert select_structures(corpus, config).created_at == made.created_at

    write_pool(["CCO", "CCC"])
    with pytest.raises(SelectionError, match="inputs"):
        read_selection(corpus, "without_pool")
    with pytest.raises(SelectionError, match="overwrite"):
        select_structures(corpus, config)
    remade = select_structures(corpus, config, overwrite=True)
    assert remade.molecules == made.molecules - 1


def test_a_cluster_file_with_a_fine_cluster_across_coarse_ones_is_refused(
    corpus: Path,
) -> None:
    assignments = pl.DataFrame(
        {"molecule_id": [0, 1], "coarse_cluster": [0, 1], "fine_cluster": [5, 5]}
    )
    with pytest.raises(SelectionError, match="spans several coarse"):
        write_clusters(corpus, assignments, _cluster_record(corpus, "broken"))


def _hierarchy(coarse_sizes: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coarse = np.repeat(np.arange(len(coarse_sizes)), coarse_sizes)
    molecule_ids = np.arange(len(coarse))
    fine = coarse * 1000 + molecule_ids % 5
    return molecule_ids, coarse, fine


def test_uniform_weighting_visits_every_coarse_cluster_before_any_twice() -> None:
    molecule_ids, coarse, fine = _hierarchy([200, 50, 5, 1, 1])
    order = hierarchical_order(molecule_ids, coarse, fine, seed=0, weighting="uniform")
    assert sorted(coarse[order[:5]].tolist()) == [0, 1, 2, 3, 4]
    first_ten_in_largest = order[coarse[order] == 0][:5]
    assert len(set(fine[first_ten_in_largest].tolist())) == 5


@pytest.mark.parametrize(
    ("weighting", "expected_share"),
    [("uniform", 0.5), ("square_root", 0.75), ("proportional", 0.9)],
)
def test_cluster_weighting_sets_how_often_a_large_cluster_comes_up(
    weighting: ClusterWeighting, expected_share: float
) -> None:
    molecule_ids, coarse, fine = _hierarchy([900, 100])
    order = hierarchical_order(molecule_ids, coarse, fine, seed=1, weighting=weighting)
    share_of_large = float(np.mean(coarse[order[:100]] == 0))
    assert share_of_large == pytest.approx(expected_share, abs=0.02)


def test_the_hierarchical_order_is_a_seeded_permutation() -> None:
    molecule_ids, coarse, fine = _hierarchy([30, 20, 10])
    first = hierarchical_order(
        molecule_ids, coarse, fine, seed=5, weighting="square_root"
    )
    again = hierarchical_order(
        molecule_ids, coarse, fine, seed=5, weighting="square_root"
    )
    other = hierarchical_order(
        molecule_ids, coarse, fine, seed=6, weighting="square_root"
    )
    np.testing.assert_array_equal(first, again)
    assert sorted(first.tolist()) == molecule_ids.tolist()
    assert not np.array_equal(first, other)


def test_gathered_structures_match_the_full_arrays(corpus: Path) -> None:
    dataset = MoleculeDataset.open_existing_dataset_from_dir(corpus)
    full = dataset.read_structures()
    structure_ids = np.array([0, 5, 6, 13, 22, 23])
    gathered = dataset.read_structures(structure_ids)
    pointer = np.asarray(full.ptr)
    for position, structure_id in enumerate(structure_ids):
        start, stop = pointer[structure_id], pointer[structure_id + 1]
        new_start, new_stop = gathered.ptr[position : position + 2]
        np.testing.assert_array_equal(
            gathered.positions[new_start:new_stop], full.positions[start:stop]
        )
        np.testing.assert_array_equal(
            gathered.atomic_numbers[new_start:new_stop], full.atomic_numbers[start:stop]
        )
    np.testing.assert_array_equal(
        gathered.total_charge, np.asarray(full.total_charge)[structure_ids]
    )
    np.testing.assert_array_equal(
        gathered.multiplicity, np.asarray(full.multiplicity)[structure_ids]
    )
    empty = dataset.read_structures(np.zeros(0, dtype=np.int64))
    assert empty.positions.shape == (0, 3)


@pytest.mark.parametrize("structure_ids", [[3, 1], [1, 1], [0, 24]])
def test_unsorted_repeated_or_out_of_range_ids_are_refused(
    corpus: Path, structure_ids: list[int]
) -> None:
    with pytest.raises(ValueError, match="structure_ids"):
        MoleculeDataset.open_existing_dataset_from_dir(corpus).read_structures(
            np.array(structure_ids)
        )


def test_a_batch_reports_every_failure_after_trying_all(corpus: Path) -> None:
    config = SelectionBatchConfig.model_validate(
        {
            "corpora": [
                {
                    "dataset": str(corpus),
                    "selections": [
                        {
                            "name": "batch_ok",
                            "steps": [{"kind": "random_molecules", "count": 1}],
                        },
                        {
                            "name": "batch_bad",
                            "steps": [{"kind": "random_molecules", "count": 99}],
                        },
                    ],
                }
            ]
        }
    )
    with pytest.raises(SelectionError, match="1 selections failed"):
        select_batch(config)
    assert read_selection_record(corpus, "batch_ok").molecules == 1


def test_the_production_selection_config_parses() -> None:
    read_model(
        REPOSITORY_ROOT / "configs/dataset_creation/select_structures.yaml",
        SelectionBatchConfig,
    )


@pytest.mark.parametrize(
    ("fraction", "available", "expected"),
    [(0.29, 100, 29), (0.57, 100, 57), (0.5, 7, 3), (1.0, 8, 8), (0.1, 5, 0)],
)
def test_a_fraction_draws_the_floor_of_the_fraction_as_written(
    fraction: float, available: int, expected: int
) -> None:
    """``0.29 * 100`` is ``28.999...`` in binary; the draw must still take 29."""
    draw = RandomMolecules(fraction=fraction)
    assert draw.molecules_to_draw(available) == expected
