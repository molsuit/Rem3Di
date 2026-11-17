"""YAML files for pydantic models: the one reader and writer of configs and records.

A model is written as ``model.model_dump(mode="json")`` through PyYAML's safe
dumper with the fields in declaration order (``sort_keys=False``), so a file
reads like its model; it is read back through a pydantic ``TypeAdapter``. Writes go
to a uniquely named temporary file next to the target and are renamed over it,
so a reader never sees half a file and concurrent writers do not collide.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, overload

import yaml
from pydantic import BaseModel, TypeAdapter


class YamlFileError(ValueError):
    """A yaml file is empty or not a mapping where a model was expected."""


def read_yaml(path: Path | str) -> Any:
    """The parsed content of a yaml file (``None`` for an empty one)."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@overload
def read_model[Model: BaseModel](
    path: Path | str, model_class: type[Model]
) -> Model: ...


@overload
def read_model(path: Path | str, model_class: Any) -> Any: ...


def read_model(path: Path | str, model_class: Any) -> Any:
    """``model_class`` validated from the yaml mapping in ``path``.

    ``model_class`` is a model or an ``Annotated`` discriminated union of models
    (the mapping's ``kind`` then picks the member).

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        YamlFileError: if the file is empty or does not hold a mapping.
        pydantic.ValidationError: if the mapping does not validate.
    """
    content = read_yaml(path)
    if not isinstance(content, dict):
        raise YamlFileError(
            f"{path} holds {type(content).__name__}, not a "
            f"{getattr(model_class, '__name__', 'model')} mapping"
        )
    return TypeAdapter(model_class).validate_python(content)


def yaml_text(content: Any) -> str:
    """Plain data (mappings, lists, scalars) as yaml text, keys in their order."""
    return yaml.safe_dump(
        content, sort_keys=False, allow_unicode=True, default_flow_style=False
    )


def write_yaml(path: Path | str, content: Any) -> None:
    """Write plain data to ``path`` atomically, creating its directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(yaml_text(content), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_model(
    path: Path | str,
    model: BaseModel,
    *,
    exclude_none: bool = False,
    by_alias: bool = False,
) -> None:
    """Write ``model`` to ``path`` atomically, fields in declaration order."""
    write_yaml(
        path,
        model.model_dump(mode="json", exclude_none=exclude_none, by_alias=by_alias),
    )
