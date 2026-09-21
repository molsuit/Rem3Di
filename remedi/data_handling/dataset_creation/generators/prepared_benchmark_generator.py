"""The one benchmark generator: a ``conformers``-stage bundle -> ``InputBatch``.

``BENCHMARK_DATA_FORMAT.md`` §2.2. This replaces the five source-specific
benchmark generators (MoleculeNet, TDC, Polaris, ChiralCat, chiral docking).
Everything source-specific — how to download, which column holds the SMILES,
how to compute a split — happened upstream in a ``remedi-data`` preparer and in
the ``generate_conformers`` prepare task; what reaches this module is one
directory in one format.

The generator does no filtering, no embedding and no id assignment. In
particular the three identity ids are copied **verbatim** off
``table.parquet``: the preparer assigned them, the split is keyed to them and
``enantiomer_of`` points at them, so re-deriving them here (which is what
``get_mol_ids_for_batch``'s SMILES-store path does for the pretraining corpora)
would be a second, disagreeing source of truth (§2.3).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from remedi.data_handling.bundle import Bundle, read_bundle
from remedi.data_handling.dataset.tasks import TaskSet, split_codes_from_names
from remedi.data_handling.dataset_creation.build_stats import LoadStats
from remedi.data_handling.dataset_creation.generators.molecule_generator import (
    MoleculeGenerator,
)
from remedi.data_handling.dataset_creation.loading_batch import (
    InputBatch,
    RegressionData,
    SmilesData,
)
from remedi.data_handling.dataset_creation.structure_ids import StructureID

logger = logging.getLogger(__name__)

#: Extra columns a 3D bundle may carry that the zarr stores per structure.
TOTAL_CHARGE_COLUMN = "total_charge"
MULTIPLICITY_COLUMN = "multiplicity"

#: A closed-shell neutral molecule, the default for a bundle that declares
#: neither extra column. The writer's own default for ``multiplicity`` is 0.0,
#: which is unphysical, so it is never relied on here.
DEFAULT_TOTAL_CHARGE = 0.0
DEFAULT_MULTIPLICITY = 1.0


class PreparedBenchmarkGenerator(MoleculeGenerator):
    """Stream a prepared ``conformers``-stage bundle as ``InputBatch`` batches.

    One bundle row is one structure, one batch is a contiguous slice of rows,
    and the row order is preserved end to end, so a zarr row ``i`` is bundle
    row ``i`` for a bundle ingested with ``[CopyDataStage]`` alone.
    """

    def __init__(self, bundle: Bundle, *, batch_size: int = 256) -> None:
        """Validate the bundle and precompute the per-structure arrays.

        Args:
            bundle: a ``conformers``-stage bundle, as ``read_bundle`` returns it.
            batch_size: rows per yielded :class:`InputBatch`.

        Raises:
            ValueError: if the bundle is not at the ``conformers`` stage, if it
                carries no structures, if the frame count disagrees with the
                row count, or if a split value is not a known split name.
        """
        if bundle.spec.stage != "conformers":
            raise ValueError(
                f"{bundle.spec.dataset_id} is a {bundle.spec.stage!r}-stage bundle; "
                "only a 'conformers'-stage bundle can be ingested (run the "
                "generate_conformers prepare task first)"
            )
        if bundle.structures is None:
            raise ValueError(
                f"{bundle.spec.dataset_id} is a conformers-stage bundle but carries "
                "no structures"
            )
        if len(bundle.structures) != len(bundle.table):
            raise ValueError(
                f"{bundle.spec.dataset_id} has {len(bundle.structures)} frames for "
                f"{len(bundle.table)} table rows"
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")

        self.bundle = bundle
        self.batch_size = int(batch_size)
        self.structures = bundle.structures

        table = bundle.table
        spec = bundle.spec
        self.n_rows = len(table)

        self.structure_id_values = table["structure_id"].to_numpy(dtype=np.int64)
        self.stereoisomer_id_values = table["stereoisomer_id"].to_numpy(dtype=np.int64)
        self.molecule_id_values = table["molecule_id"].to_numpy(dtype=np.int64)
        self.isomeric_smiles_values = table["isomeric_smiles"].astype(str).tolist()
        self.nonisomeric_smiles_values = (
            table["nonisomeric_smiles"].astype(str).tolist()
        )

        task_names = spec.task_names()
        self.targets = table[task_names].to_numpy(dtype=np.float64)
        # NaN is the format's only missingness encoding (§1.1); the zarr's
        # ``mask_system`` is re-derived from it here and nowhere else.
        self.masks = (~np.isnan(self.targets)).astype(np.uint8)
        self.split_codes = split_codes_from_names(
            table[spec.default_split].astype(str).tolist()
        )
        self.total_charge_values = _optional_column(
            table, TOTAL_CHARGE_COLUMN, DEFAULT_TOTAL_CHARGE, self.n_rows
        )
        self.multiplicity_values = _optional_column(
            table, MULTIPLICITY_COLUMN, DEFAULT_MULTIPLICITY, self.n_rows
        )

        # Nothing is dropped here: the preparer and generate_conformers already
        # did every rejection, and their counts live in ``provenance.yaml``.
        self.load_stats = LoadStats(n_raw_rows=self.n_rows, n_kept=self.n_rows)

    @classmethod
    def from_directory(
        cls, directory: Path, *, batch_size: int = 256
    ) -> PreparedBenchmarkGenerator:
        """Read and fully validate the bundle at ``directory``, then wrap it."""
        return cls(read_bundle(Path(directory)), batch_size=batch_size)

    def task_set(self) -> TaskSet:
        """The zarr-side :class:`TaskSet` this bundle's tasks become."""
        return self.bundle.spec.task_set()

    def __iter__(self) -> Iterator[InputBatch]:
        for start in range(0, self.n_rows, self.batch_size):
            stop = min(start + self.batch_size, self.n_rows)
            yield self._batch(start, stop)

    def _batch(self, start: int, stop: int) -> InputBatch:
        return InputBatch(
            smiles=[
                SmilesData(
                    nonisomeric_smiles=self.nonisomeric_smiles_values[row],
                    isomeric_smiles=self.isomeric_smiles_values[row],
                )
                for row in range(start, stop)
            ],
            molecules=list(self.structures[start:stop]),
            structure_ids=[
                StructureID(
                    structure_id=int(self.structure_id_values[row]),
                    molecule_id=int(self.molecule_id_values[row]),
                    stereoisomer_id=int(self.stereoisomer_id_values[row]),
                )
                for row in range(start, stop)
            ],
            total_charge=self.total_charge_values[start:stop].tolist(),
            multiplicity=self.multiplicity_values[start:stop].tolist(),
            regression_data=RegressionData(
                targets_system=self.targets[start:stop],
                mask_system=self.masks[start:stop],
                split=self.split_codes[start:stop],
            ),
            raw_smiles=None,
        )


def _optional_column(table, column_name: str, default: float, n_rows: int):
    """``table[column_name]`` as float64, or ``default`` repeated ``n_rows`` times."""
    if column_name in table.columns:
        return table[column_name].to_numpy(dtype=np.float64)
    return np.full(n_rows, default, dtype=np.float64)
