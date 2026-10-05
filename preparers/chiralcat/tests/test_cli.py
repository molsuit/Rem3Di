"""The console entry point: exit codes and the summary it prints."""

from __future__ import annotations

from conftest import FIXED_ACHIRAL_SOURCE, write_config, write_pickle

from chiralcat_dataset.cli import main


def test_an_unreadable_config_exits_with_code_2(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "missing.yaml")]) == 2
    assert "could not load config" in capsys.readouterr().err


def test_a_missing_source_exits_with_code_1(tmp_path, capsys):
    config = write_config(tmp_path, sources=[FIXED_ACHIRAL_SOURCE])
    assert main(["--config", str(config)]) == 1
    assert "pipeline failed" in capsys.readouterr().err


def test_a_small_build_writes_every_output(tmp_path, capsys):
    write_pickle(tmp_path / "data" / "fixed.pkl", {"SMILES": ["CCO", "CCC"]})
    config = write_config(tmp_path, sources=[FIXED_ACHIRAL_SOURCE])
    assert main(["--config", str(config)]) == 0
    assert "dataset: 2 structures" in capsys.readouterr().out
    assert (tmp_path / "output" / "dataset.csv").is_file()
