from collections.abc import Callable
from pathlib import Path

import pytest
from mlflow.entities import FileInfo, Run
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from conftest import Tracking
from mlflow_optimization import (
    MlflowOptimizationWarning,
    SolveRun,
    autolog,
    last_solve,
)


def test_disabling_autolog_restores_solver_callables_without_writes(
    tracking: Tracking,
) -> None:
    from fakes import FakeSolver

    original_solve: Callable[..., str] = FakeSolver.solve

    autolog(solvers=["fake"])
    assert FakeSolver.solve is not original_solve

    autolog(disable=True)

    assert FakeSolver.solve is original_solve
    solver: FakeSolver = FakeSolver()
    result: str = solver.solve()

    assert result == "optimal"
    assert tracking.solve_runs() == []


def test_unusable_tracking_store_does_not_change_solve_result_or_hide_warning(
    tracking: Tracking,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fakes import FakeSolver

    def fail_create_run(
        client: MlflowClient, *arguments: object, **keywords: object
    ) -> Run:
        raise MlflowException("Tracking store is unavailable.")

    monkeypatch.setattr(MlflowClient, "create_run", fail_create_run)
    autolog(solvers=["fake"])
    solver: FakeSolver = FakeSolver(status="feasible")

    with pytest.warns(MlflowOptimizationWarning):
        result: str = solver.solve()

    assert result == "feasible"
    assert tracking.solve_runs() == []


def test_solve_runs_have_no_artifacts_by_default(tracking: Tracking) -> None:
    from fakes import FakeSolver

    autolog(solvers=["fake"])
    solver: FakeSolver = FakeSolver()
    solver.solve()

    solve_runs: list[Run] = tracking.solve_runs()

    assert len(solve_runs) == 1
    assert tracking.client.list_artifacts(solve_runs[0].info.run_id) == []


def test_unavailable_solver_values_are_missing_with_reasons_and_other_values_remain(
    tracking: Tracking,
) -> None:
    from fakes import FakeSolver

    autolog(solvers=["fake"])
    solver: FakeSolver = FakeSolver(
        objective=None,
        bound=None,
        statistics={"node_count": 4.0},
    )
    solver.solve()

    solve_run: Run = tracking.solve_runs()[0]
    unavailable_metrics: tuple[str, ...] = (
        "objective_value",
        "dual_bound",
        "absolute_gap",
        "relative_gap",
        "solver_time_seconds",
        "simplex_iterations",
        "barrier_iterations",
    )

    metric_name: str
    for metric_name in unavailable_metrics:
        assert metric_name not in solve_run.data.metrics
        assert (
            solve_run.data.tags[f"optimization.missing.{metric_name}"]
            == "unsupported"
        )

    assert solve_run.data.metrics["node_count"] == 4.0
    assert solve_run.data.metrics["wall_time_seconds"] > 0.0
    assert solve_run.data.tags["optimization.api"] == "fake"
    assert solve_run.data.tags["optimization.termination"] == "optimal"


def test_last_solve_additions_stay_on_their_solve(
    tracking: Tracking,
    tmp_path: Path,
) -> None:
    from fakes import FakeSolver

    autolog(solvers=["fake"])
    first_solver: FakeSolver = FakeSolver()
    first_solver.solve()
    first_solve: SolveRun | None = last_solve()
    assert first_solve is not None

    second_solver: FakeSolver = FakeSolver()
    second_solver.solve()
    artifact_path: Path = tmp_path / "route.txt"
    artifact_path.write_text("selected route", encoding="utf-8")

    first_solve.log_metric(
        "travel_distance",
        12.5,
        unit="km",
        description="Total travel distance",
    )
    first_solve.set_tags({"batch": "first"})
    first_solve.log_artifact(str(artifact_path), artifact_path="results")

    solve_runs: list[Run] = tracking.solve_runs()
    first_run: Run = solve_runs[0]
    second_run: Run = solve_runs[1]

    assert first_run.data.metrics["user.travel_distance"] == 12.5
    assert first_run.data.tags["user.travel_distance.unit"] == "km"
    assert (
        first_run.data.tags["user.travel_distance.description"]
        == "Total travel distance"
    )
    assert first_run.data.tags["batch"] == "first"
    first_artifacts: list[FileInfo] = tracking.client.list_artifacts(
        first_run.info.run_id,
        path="results",
    )
    assert len(first_artifacts) == 1
    assert first_artifacts[0].path == "results/route.txt"
    assert "user.travel_distance" not in second_run.data.metrics
    assert "batch" not in second_run.data.tags
    assert tracking.client.list_artifacts(second_run.info.run_id) == []


def test_disabled_artifact_logging_warns_and_refuses_artifact(
    tracking: Tracking,
    tmp_path: Path,
) -> None:
    from fakes import FakeSolver

    autolog(solvers=["fake"], log_artifacts=False)
    solver: FakeSolver = FakeSolver()
    solver.solve()
    solve_run: SolveRun | None = last_solve()
    assert solve_run is not None
    artifact_path: Path = tmp_path / "route.txt"
    artifact_path.write_text("selected route", encoding="utf-8")

    with pytest.warns(MlflowOptimizationWarning):
        solve_run.log_artifact(str(artifact_path))

    recorded_run: Run = tracking.client.get_run(solve_run.run_id)
    assert tracking.client.list_artifacts(recorded_run.info.run_id) == []


def test_solve_exception_propagates_and_marks_run_failed(tracking: Tracking) -> None:
    from fakes import FakeSolver

    autolog(solvers=["fake"])
    solve_error: RuntimeError = RuntimeError("solver failed")
    solver: FakeSolver = FakeSolver(raises=solve_error)
    raised_error: pytest.ExceptionInfo[RuntimeError]

    with pytest.raises(RuntimeError) as raised_error:
        solver.solve()

    assert raised_error.value is solve_error
    solve_run: Run = tracking.solve_runs()[0]
    assert solve_run.info.status == "FAILED"
    assert solve_run.data.tags["optimization.termination"] == "error"
    assert solve_run.data.tags["optimization.error"] == "RuntimeError: solver failed"
