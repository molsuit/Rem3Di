"""A dataset's structures: in memory (:class:`Structures`), on disk
(:class:`StructureFile`, ``structures.atp``) and joined with the table
(:class:`Dataset`)."""

from remedi_data.dataset.dataset import Dataset, structures_sha256
from remedi_data.dataset.structure_file import StructureFile, write_structures
from remedi_data.dataset.structures import Structures

__all__ = [
    "Dataset",
    "StructureFile",
    "Structures",
    "structures_sha256",
    "write_structures",
]
