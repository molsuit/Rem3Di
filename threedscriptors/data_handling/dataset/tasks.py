from enum import Enum

from pydantic import BaseModel


class TaskType(str, Enum):
    regression = "regression"
    classification = "classification"  # single-label or multi-label (see output_dim)


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
    atom_map:   dict[str, int] = {}


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
