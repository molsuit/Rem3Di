from typing import Annotated

import numpy as np
from e3nn.o3 import Irreps
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PlainSerializer,
    WithJsonSchema,
)


class StrictBaseModel(BaseModel):
    """Config base that rejects unknown keys, so a mistyped yaml key fails loudly."""

    model_config = ConfigDict(extra="forbid")


IrrepType = Annotated[
    Irreps,
    BeforeValidator(lambda v: Irreps(v)),
    PlainSerializer(lambda v: str(v), when_used="json"),
    WithJsonSchema(
        {
            "type": "string",
            "description": "A string representation of irreps.",
            "examples": [
                "10x0o + 5x1e + 3x2o",
            ],
        }
    ),
]

NumpyArrayType = Annotated[
    np.ndarray,
    BeforeValidator(lambda v: np.array(v)),
    PlainSerializer(lambda v: v.tolist()),
]
