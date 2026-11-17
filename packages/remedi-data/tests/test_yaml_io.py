"""Models round-trip through yaml files with their field order kept."""

from __future__ import annotations

import math
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, ValidationError

from remedi_data.yaml_io import (
    YamlFileError,
    read_model,
    read_yaml,
    write_model,
    write_yaml,
)


class Colour(StrEnum):
    red = "red"


class Inner(BaseModel):
    path: Path
    value: float


class Outer(BaseModel):
    zeta: int
    alpha: Colour = Colour.red
    inner: Inner
    optional: str | None = None
    renamed: int = Field(0, alias="renamed-key")


def test_a_model_round_trips_with_its_field_order(tmp_path: Path) -> None:
    model = Outer(zeta=1, inner=Inner(path=Path("a/b"), value=math.nan))
    path = tmp_path / "nested" / "model.yaml"
    write_model(path, model)
    text = path.read_text()
    assert text.index("zeta") < text.index("alpha") < text.index("inner")
    loaded = read_model(path, Outer)
    assert loaded.inner.path == Path("a/b") and math.isnan(loaded.inner.value)
    assert list(tmp_path.joinpath("nested").iterdir()) == [path]


def test_exclude_none_and_aliases(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    write_model(
        path,
        Outer(zeta=1, inner=Inner(path=Path("x"), value=1.0)),
        exclude_none=True,
        by_alias=True,
    )
    content = read_yaml(path)
    assert "optional" not in content and content["renamed-key"] == 0


def test_an_empty_or_non_mapping_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")
    with pytest.raises(YamlFileError, match="NoneType"):
        read_model(path, Outer)
    write_yaml(path, [1, 2])
    with pytest.raises(YamlFileError, match="list"):
        read_model(path, Outer)


def test_an_invalid_mapping_raises_the_validation_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    write_yaml(path, {"zeta": "not a number"})
    with pytest.raises(ValidationError):
        read_model(path, Outer)


class _Circle(BaseModel):
    kind: Literal["circle"] = "circle"
    radius: float


class _Square(BaseModel):
    kind: Literal["square"] = "square"
    side: float


_Shape = Annotated[_Circle | _Square, Field(discriminator="kind")]


def test_a_discriminated_union_is_read_as_the_member_its_kind_names(
    tmp_path: Path,
) -> None:
    # Regression: read_model called ``model_validate`` on the union itself,
    # which is not a model (the architecture config could not be loaded).
    path = tmp_path / "shape.yaml"
    write_model(path, _Square(side=2.0))
    assert read_model(path, _Shape) == _Square(side=2.0)
    write_yaml(path, {"kind": "triangle"})
    with pytest.raises(ValidationError):
        read_model(path, _Shape)
    write_yaml(path, [1, 2])
    with pytest.raises(YamlFileError, match="not a"):
        read_model(path, _Shape)
