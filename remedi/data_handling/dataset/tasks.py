from enum import Enum, StrEnum

from pydantic import BaseModel


class ElementSet(StrEnum):
    """Named element-set preset for ``filter_mol(allowed_elements=...)``.

    ``mace_off`` is the organic drug subset (H,C,N,O,F,P,S,Cl,Br,I).
    ``mace_polar`` extends to atomic numbers 1..83 (the MACE-POLAR coverage).
    Resolve via :func:`generators.utils.resolve_element_set`.
    """

    mace_off = "mace_off"
    mace_polar = "mace_polar"


class Split(int, Enum):
    """Per-structure split membership, materialized into the dataset on disk.

    Stored as a uint8 zarr column. ``unassigned`` is the default for datasets
    ingested without a literature split; eval may still override the split at
    run time regardless of what is stored.
    """

    train = 0
    valid = 1
    test = 2
    unassigned = 255


class TaskType(str, Enum):
    regression = "regression"
    classification = "classification"  # single-label or multi-label (see output_dim)
    # Single-label, multi-class (>2 mutually exclusive classes in one column).
    # The integer class index lives in a single ``targets_system`` column; the
    # class count is derived from the data at eval time (``int(nanmax)+1``).
    multiclass = "multiclass"


class TaskScope(str, Enum):
    system = "system"
    atom = "atom"


class TaskConfig(BaseModel):
    name: str
    task_type: TaskType
    scope: TaskScope
    auxillary_dim: int | None = None


class TaskSet(BaseModel):
    system_cols: list[TaskConfig] = []
    atom_cols: list[TaskConfig] = []

    system_map: dict[str, int] = {}
    atom_map: dict[str, int] = {}

    @classmethod
    def from_list(cls, task_list: list[TaskConfig]):
        sys_cols, atom_cols = [], []
        for t in task_list:
            if t.scope == TaskScope.system:
                sys_cols.append(t)
            else:
                atom_cols.append(t)
        return cls(system_cols=sys_cols, atom_cols=atom_cols).finalize()

    def finalize(self):
        self.system_map = {c.name: i for i, c in enumerate(self.system_cols)}
        self.atom_map = {c.name: i for i, c in enumerate(self.atom_cols)}
        return self
