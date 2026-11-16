"""The shared task runner's run report (``remedi_data.runs.task_runner``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from remedi_data.runs import task_runner
from remedi_data.runs.task import TaskStatus


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
