"""One structure source per raw corpus; ``StructureSource`` is their kind-discriminated union."""

from typing import Annotated

from pydantic import Field

from remedi_data.structures.sources.geom_drugs import GeomDrugsSource
from remedi_data.structures.sources.pcqm4mv2 import Pcqm4mv2Source
from remedi_data.structures.sources.tmqm import TmqmSource

StructureSource = Annotated[
    GeomDrugsSource | Pcqm4mv2Source | TmqmSource, Field(discriminator="kind")
]

__all__ = ["GeomDrugsSource", "Pcqm4mv2Source", "StructureSource", "TmqmSource"]
