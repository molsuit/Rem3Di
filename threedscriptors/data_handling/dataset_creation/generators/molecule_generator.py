from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

from threedscriptors.data_handling.dataset_creation.loading_batch import (
    InputBatch,
)

# Elements supported by your downstream MACE-OFF stack




class MoleculeGenerator(Iterable[InputBatch], ABC):
    """Base class for molecule generators yielding InputBatch instances.
    Subclasses must implement an efficient ``__iter__`` that yields
    ``InputBatch`` objects, ideally streaming to minimize memory usage.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[InputBatch]:  # pragma: no cover - interface only
        """Return an iterator over ``InputBatch`` items."""
        raise NotImplementedError

