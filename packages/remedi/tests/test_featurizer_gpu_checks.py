"""The CPU-checkable parts of ``scripts/featurizer_gpu_checks.py``: the dense graph
its benchmark compares against is every ordered intra-molecule pair and gives the
neighbour-list features, the smoke configuration validates, and the summary reads
the files the benchmark and a run write."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.build import molecule

from remedi.configuration.mace_config import MaceConfig
from remedi.model.featurizer import pad_molecules
from remedi.training.data.mixture import DatasetComponent
from remedi.training.data.sample import MoleculeCollator, Sample

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "featurizer_gpu_checks.py"
MACE_OFF_MEDIUM = Path.home() / ".cache" / "mace" / "MACE-OFF23_medium.model"
# The smoke template's architecture names MACE-POLAR-1-M's irreps.
MACE_POLAR = Path(
    os.environ.get(
        "MACE_POLAR_MODEL_PATH",
        "/dais/u/wedigs/graph_electrostatics/MACE-POLAR-1-M.model",
    )
)


@pytest.fixture(scope="module")
def checks():
    specification = importlib.util.spec_from_file_location(
        "featurizer_gpu_checks", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules["featurizer_gpu_checks"] = module
    specification.loader.exec_module(module)
    return module


def test_dense_pair_index_lists_every_ordered_pair_within_each_molecule(checks):
    pairs = checks.dense_pair_index(molecules=2, atoms=3, device=torch.device("cpu"))
    expected = {
        (molecule_index * 3 + sender, molecule_index * 3 + receiver)
        for molecule_index in range(2)
        for sender in range(3)
        for receiver in range(3)
        if sender != receiver
    }
    listed = [tuple(pair) for pair in pairs.T.tolist()]
    assert len(listed) == len(expected)
    assert set(listed) == expected


def _sample(name: str) -> Sample:
    atoms = molecule(name)
    return Sample(
        atomic_positions=torch.tensor(atoms.get_positions(), dtype=torch.float64),
        atomic_numbers=torch.tensor(atoms.get_atomic_numbers()),
        total_charge=torch.tensor(0.0),
        multiplicity=torch.tensor(1.0),
    )


def test_dense_graph_gives_the_neighbor_list_features(checks):
    if not MACE_OFF_MEDIUM.exists():
        pytest.skip(f"{MACE_OFF_MEDIUM} is missing")
    featurizer = MaceConfig.model_validate(
        {
            "model_path": MACE_OFF_MEDIUM,
            "device": "cpu",
            "dtype": "float64",
            "enable_cueq": False,
        }
    ).build_featurizer()
    batch = MoleculeCollator(cutoff=featurizer.r_max)(
        [_sample("C60"), _sample("CH3CH2OH")]
    )
    padded = pad_molecules(batch, pad_multiple=4)
    molecules, atoms = padded.atomic_numbers.shape
    with torch.no_grad():
        dense = featurizer.features(
            padded.atomic_numbers,
            padded.positions,
            checks.dense_pair_index(molecules, atoms, padded.positions.device),
        )
    neighbor = featurizer(padded)
    assert padded.edge_index.shape[1] < molecules * atoms * (atoms - 1)
    torch.testing.assert_close(neighbor, dense, rtol=1e-12, atol=1e-12)


def test_summary_tabulates_the_benchmark_and_the_smoke_run(checks, tmp_path, capsys):
    record = {
        "set": "small",
        "gpu": "test GPU",
        "median_ms": 10.0,
        "peak_memory_gib": 1.0,
        "neighbor_pairs": [1000],
        "dense_pairs": [3000],
        "collate_ms_cpu": 5.0,
        "first_call_seconds": 1.0,
        "unique_graphs_after_timing": 1,
    }
    with open(tmp_path / "timings.jsonl", "w") as handle:
        for mode in ("neighbor_eager", "dense_eager"):
            handle.write(json.dumps({**record, "mode": mode}) + "\n")
    np.save(tmp_path / "features_dense_eager_small.npy", np.ones((3, 2)))
    np.save(tmp_path / "features_neighbor_eager_small.npy", np.full((3, 2), 1.5))
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    (run_directory / "throughput.jsonl").write_text(
        json.dumps({"global_step": 50, "step_ms": 60.0, "featurizer_ms": 20.0}) + "\n"
    )
    checks.summary(
        type(
            "Arguments", (), {"benchmark_dir": tmp_path, "run_directory": run_directory}
        )
    )
    printed = capsys.readouterr().out
    assert "5.00e-01 (5.0e-01 rel)" in printed
    assert "neighbor_eager" in printed and "dense_eager" in printed
    assert "test GPU" in printed
    smoke_row = printed.strip().splitlines()[-1].split("|")
    assert [cell.strip() for cell in smoke_row][:2] == ["50", "60"]
    assert smoke_row[5].strip() == "85"  # the dense baseline's first window


def test_smoke_config_validates(checks, tmp_path):
    if not MACE_POLAR.exists():
        pytest.skip(f"{MACE_POLAR} is missing (set MACE_POLAR_MODEL_PATH)")
    from remedi.configuration.training_config import PretrainingConfig
    from remedi_data.yaml_io import read_model

    datasets = {}
    for name in ("pcqm4mv2", "geom_drugs"):
        datasets[name] = tmp_path / name
        datasets[name].mkdir()
    arguments = type(
        "Arguments",
        (),
        {
            "model_path": MACE_POLAR,
            "pcqm4mv2": datasets["pcqm4mv2"],
            "geom_drugs": datasets["geom_drugs"],
            "output_base": tmp_path / "runs",
            "config_dir": tmp_path / "configs",
            "num_workers": 0,
            "epochs": 3,
        },
    )
    checks.smoke_config(arguments)
    training = read_model(
        tmp_path / "configs" / "training_config.yaml", PretrainingConfig
    )
    assert not training.compile.enabled
    assert training.epochs == 3
    components = training.data.components
    assert all(isinstance(component, DatasetComponent) for component in components)
    assert {
        str(component.dataset)
        for component in components
        if isinstance(component, DatasetComponent)
    } == {str(path) for path in datasets.values()}
