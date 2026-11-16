"""The shared task runner's run report (``remedi_data.runs.task_runner``)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel

from remedi_data.runs import task_runner
from remedi_data.runs.results import EvalResult
from remedi_data.runs.task import TaskStatus
from remedi_data.runs.task_runner import RunReport
from remedi_data.yaml_io import read_model


def test_the_run_report_resolves_the_git_sha_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``status.yaml`` is rewritten after every task; git must not run every time."""
    directories: list[Path] = []

    def recording_git_head_sha(directory: Path) -> str:
        directories.append(directory)
        return "f" * 40

    monkeypatch.setattr(task_runner, "git_head_sha", recording_git_head_sha)
    task_runner.repository_git_sha.cache_clear()
    status = TaskStatus(name="0_task", kind="task", ok=True, duration_s=0.0)

    reports = [task_runner.make_run_report(1, [status]) for _ in range(3)]

    task_runner.repository_git_sha.cache_clear()
    assert [report.git_sha for report in reports] == ["f" * 40] * 3
    assert len(directories) == 1
    assert directories[0] == Path(task_runner.__file__).parent


class _Task(BaseModel):
    """A task that yields nothing, or raises."""

    kind: str = "toy"
    fails: bool = False

    def run(self, ctx: object) -> Iterator[EvalResult]:
        if self.fails:
            raise RuntimeError("boom")
        yield from ()


class _Manifest(BaseModel):
    name: str = "toy run"


def _status(directory: Path) -> RunReport:
    return read_model(directory / task_runner.STATUS_FILENAME, RunReport)


def test_the_manifest_is_written_before_the_first_task(tmp_path: Path) -> None:
    seen: list[bool] = []

    class _Peeking(_Task):
        def run(self, ctx: object) -> Iterator[EvalResult]:
            seen.append((tmp_path / task_runner.MANIFEST_FILENAME).is_file())
            yield from ()

    report = task_runner.run_and_record(
        [[_Peeking()]], None, tmp_path, _Manifest(), keep_going=True
    )
    assert seen == [True]
    assert read_model(tmp_path / task_runner.MANIFEST_FILENAME, _Manifest).name == (
        "toy run"
    )
    assert report == _status(tmp_path)


def test_groups_are_taken_one_after_the_other(tmp_path: Path) -> None:
    """The next group is built only once the previous one has run."""
    finished_before_second: list[int] = []

    def groups() -> Iterator[list[_Task]]:
        yield [_Task(), _Task(fails=True)]
        finished_before_second.append(_status(tmp_path).n_tasks)
        yield [_Task()]

    report = task_runner.run_and_record(
        groups(), None, tmp_path, _Manifest(), keep_going=True
    )
    assert finished_before_second == [2]
    assert (report.n_tasks, report.n_failed) == (3, 1)
    assert [status.name for status in report.statuses] == ["0_toy", "1_toy", "2_toy"]


def test_a_stopping_failure_still_records_the_partial_run(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        task_runner.run_and_record(
            [[_Task(), _Task(fails=True), _Task()]],
            None,
            tmp_path,
            _Manifest(),
            keep_going=False,
        )
    status = _status(tmp_path)
    assert (status.n_tasks, status.n_failed) == (3, 1)
    assert [status.ok for status in status.statuses] == [True, False]
