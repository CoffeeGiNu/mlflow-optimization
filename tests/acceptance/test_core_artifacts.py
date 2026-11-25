from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from mlflow.entities import FileInfo, Run
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from mlflow_optimization import MlflowOptimizationWarning, autolog

if TYPE_CHECKING:
    from conftest import Tracking


def _artifact_paths(tracking: Tracking, run: Run, path: str | None = None) -> list[str]:
    infos: list[FileInfo] = tracking.client.list_artifacts(run.info.run_id, path)
    return [info.path for info in infos]


def test_requested_solver_file_and_diagnostics_are_attached_to_the_run(
    tracking: Tracking,
) -> None:
    from fakes import FakeSolver

    autolog(libraries=["fake"], log_solver_log=True)
    solver: FakeSolver = FakeSolver(
        artifact_text="solver output",
        diagnostics={"optimization.artifacts.solver_log": "true"},
    )

    assert solver.solve() == "optimal"

    run: Run = tracking.solve_runs()[0]
    assert run.data.tags["optimization.artifacts.solver_log"] == "true"
    assert _artifact_paths(tracking, run) == ["solver"]
    assert _artifact_paths(tracking, run, "solver") == ["solver/fake.log"]
    downloaded: str = tracking.client.download_artifacts(
        run.info.run_id, "solver/fake.log"
    )
    assert Path(downloaded).read_text(encoding="utf-8") == "solver output"


@pytest.mark.parametrize(
    "autolog_options",
    (
        {"log_artifacts": False, "log_solver_log": True},
        {},
    ),
    ids=("artifacts-disabled", "nothing-requested"),
)
def test_no_workspace_and_no_attachment_unless_requested_and_enabled(
    tracking: Tracking, autolog_options: dict[str, bool]
) -> None:
    from fakes import FakeSolver

    autolog(libraries=["fake"], **autolog_options)
    solver: FakeSolver = FakeSolver(artifact_text="solver output")

    solver.solve()

    assert solver.seen_workspace is None
    run: Run = tracking.solve_runs()[0]
    assert _artifact_paths(tracking, run) == []


def test_workspace_directory_is_removed_after_a_successful_solve(
    tracking: Tracking,
) -> None:
    from fakes import FakeSolver

    autolog(libraries=["fake"], log_solver_model=True)
    solver: FakeSolver = FakeSolver(artifact_text="model")

    solver.solve()

    assert solver.seen_workspace is not None
    assert solver.seen_workspace.options.solver_model
    assert not solver.seen_workspace.options.solver_log
    assert not solver.seen_workspace.directory.exists()


def test_raising_solve_attaches_written_files_cleans_up_and_reraises(
    tracking: Tracking,
) -> None:
    from fakes import FakeSolver

    autolog(libraries=["fake"], log_solver_log=True)
    solve_error: RuntimeError = RuntimeError("solver failed")
    solver: FakeSolver = FakeSolver(artifact_text="partial log", raises=solve_error)

    with pytest.raises(RuntimeError) as raised_error:
        solver.solve()

    assert raised_error.value is solve_error
    assert solver.seen_workspace is not None
    assert not solver.seen_workspace.directory.exists()
    run: Run = tracking.solve_runs()[0]
    assert run.info.status == "FAILED"
    assert _artifact_paths(tracking, run, "solver") == ["solver/fake.log"]


def test_upload_failure_warns_and_keeps_the_solve_result_and_run_data(
    tracking: Tracking, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fakes import FakeSolver

    def failing_log_artifact(
        client: MlflowClient, *arguments: object, **keywords: object
    ) -> None:
        raise MlflowException("artifact store unavailable")

    monkeypatch.setattr(MlflowClient, "log_artifact", failing_log_artifact)
    autolog(libraries=["fake"], log_solver_log=True)
    solver: FakeSolver = FakeSolver(
        artifact_text="solver output",
        diagnostics={"optimization.artifacts.solver_log": "true"},
    )

    with pytest.warns(MlflowOptimizationWarning):
        result: str = solver.solve()

    assert result == "optimal"
    run: Run = tracking.solve_runs()[0]
    assert run.info.status == "FINISHED"
    assert run.data.tags["optimization.termination"] == "optimal"
    assert run.data.tags["optimization.artifacts.solver_log"] == "true"
    assert "objective_value" in run.data.metrics
    assert _artifact_paths(tracking, run) == []


def test_nested_backend_solve_sees_no_workspace(tracking: Tracking) -> None:
    from fakes import FakeSolver
    from fakes_frontend import FakeFrontend

    autolog(libraries=["fake", "fake_frontend"], log_solver_log=True)
    backend: FakeSolver = FakeSolver(artifact_text="backend output")

    FakeFrontend(backend).solve()

    assert backend.seen_workspace is None
    runs: list[Run] = tracking.solve_runs()
    assert len(runs) == 1
    assert runs[0].data.tags["optimization.api"] == "fake_frontend"
    assert _artifact_paths(tracking, runs[0]) == []
