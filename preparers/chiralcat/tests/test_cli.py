"""The console entry point: exit codes and the summary it prints."""

from __future__ import annotations

import yaml
from test_extract import _write_pickle

from chiralcat_dataset.cli import main


def test_an_unreadable_config_exits_with_code_2(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "missing.yaml")]) == 2
    assert "could not load config" in capsys.readouterr().err


def test_a_missing_source_exits_with_code_1(tmp_path, capsys):
    config = tmp_path / "pipeline.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "extraction": {
                    "data_dir": "data",
                    "sources": [
                        {"kind": "fixed_label", "path": "absent.pkl", "label": "achiral"}
                    ],
                },
                "validation": {"corrections_file": None},
            }
        )
    )
    assert main(["--config", str(config)]) == 1
    assert "pipeline failed" in capsys.readouterr().err


def test_a_small_build_writes_every_output(tmp_path, capsys):
    (tmp_path / "data").mkdir()
    _write_pickle(tmp_path / "data" / "fixed.pkl", ["CCO", "CCC"])
    config = tmp_path / "pipeline.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "extraction": {
                    "data_dir": "data",
                    "sources": [
                        {"kind": "fixed_label", "path": "fixed.pkl", "label": "achiral"}
                    ],
                },
                "validation": {"corrections_file": None, "audit_central_stereo": False},
                "organometallic": {"enabled": False},
            }
        )
    )
    assert main(["--config", str(config)]) == 0
    assert "dataset: 2 structures" in capsys.readouterr().out
    assert (tmp_path / "output" / "dataset.csv").is_file()
