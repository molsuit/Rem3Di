"""The generated vulture whitelist covers pydantic fields and enum members, nothing else."""

from __future__ import annotations

import importlib.util
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = WORKSPACE_ROOT / "tools" / "vulture" / "generate_whitelist.py"


def _generator():
    spec = importlib.util.spec_from_file_location("generate_whitelist", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fields_of_models_enums_and_their_subclasses_are_whitelisted(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text(
        "from enum import Enum\n"
        "from pydantic import BaseModel\n\n"
        "class Base(BaseModel):\n"
        "    base_field: int\n\n"
        "class Derived(Base):\n"
        "    derived_field: str = 'x'\n\n"
        "class Colour(str, Enum):\n"
        "    red = 'red'\n\n"
        "class Plain:\n"
        "    plain_attribute = 1\n\n"
        "    def method(self):\n"
        "        local_variable = 2\n"
        "        return local_variable\n"
    )

    fields, classes = _generator().model_field_names([source])

    assert fields == {"base_field", "derived_field", "red"}
    assert classes == {"Base", "Derived", "Colour"}
