import warnings
from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread

import pytest
from conftest import Tracking
from mlflow.entities import FileInfo, Run
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from mlflow_optimization import (
    MlflowOptimizationWarning,
    SolveRun,
    autolog,
    context,
    last_solve,
)
from mlflow_optimization.features.autolog.ports import SolveObserver


def test_disabling_autolog_restores_solver_callables_without_writes(
    tracking: Tracking,
) -> None:
    from fakes import FakeSolver

    original_solve: Callable[..., str] = FakeSolver.solve

    autolog(libraries=["fake"])
    assert FakeSolver.solve is not original_solve

    autolog(disable=True)

    assert FakeSolver.solve is original_solve
    solver: FakeSolver = FakeSolver()
    result: str = solver.solve()

    assert result == "optimal"
    assert tracking.solve_runs() == []


def test_repeated_autolog_keeps_selected_adapter_installed(
    tracking: Tracking,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fakes import FakeSolver, FakeSolverAdapter

    installed_adapters: list[FakeSolverAdapter] = []
    uninstalled_adapters: list[FakeSolverAdapter] = []
    original_install: Callable[[FakeSolverAdapter, SolveObserver], None] = (
        FakeSolverAdapter.install
    )
    original_uninstall: Callable[[FakeSolverAdapter], None] = (
        FakeSolverAdapter.uninstall
    )

    def track_install(
        adapter: FakeSolverAdapter,
        observer: SolveObserver,
    ) -> None:
        installed_adapters.append(adapter)
        original_install(adapter, observer)

    def track_uninstall(adapter: FakeSolverAdapter) -> None:
        uninstalled_adapters.append(adapter)
        original_uninstall(adapter)

    monkeypatch.setattr(FakeSolverAdapter, "install", track_install)
    monkeypatch.setattr(FakeSolverAdapter, "uninstall", track_uninstall)

    autolog(libraries=["fake"])
    solver: FakeSolver = FakeSolver()
    solver.solve()
    autolog(libraries=["fake"])
    solver.solve()

    assert len(installed_adapters) == 1
    assert uninstalled_adapters == []

    autolog(libraries=[])
    autolog(libraries=["fake"])
    solver.solve()

    assert len(installed_adapters) == 2
    assert installed_adapters[0] is installed_adapters[1]
    assert len(uninstalled_adapters) == 1
    assert uninstalled_adapters[0] is installed_adapters[0]
    assert len(tracking.solve_runs()) == 3


def test_disabling_autolog_preserves_recording_options(
    tracking: Tracking,
    tmp_path: Path,
) -> None:
    from fakes import FakeSolver

    autolog(
        libraries=["fake"],
        log_solver_parameters=False,
        log_artifacts=False,
    )
    first_solver: FakeSolver = FakeSolver()
    first_solver.solve()
    first_solve: SolveRun | None = last_solve()
    assert first_solve is not None

    solve_started: Event = Event()
    finish_solve: Event = Event()

    def hold_solve() -> None:
        solve_started.set()
        if not finish_solve.wait(timeout=10):
            raise TimeoutError("The test did not release the solve.")

    solver: FakeSolver = FakeSolver(
        parameters={"TimeLimit": 5},
        during_solve=hold_solve,
    )
    solve_errors: list[Exception] = []

    def solve_on_thread() -> None:
        try:
            solver.solve()
        except Exception as error:  # noqa: BLE001
            solve_errors.append(error)

    solve_thread: Thread = Thread(target=solve_on_thread, daemon=True)
    solve_thread.start()
    try:
        assert solve_started.wait(timeout=10)
        autolog(disable=True)
    finally:
        finish_solve.set()
    solve_thread.join(timeout=10)

    assert not solve_thread.is_alive()
    assert solve_errors == []
    solve_runs: list[Run] = tracking.solve_runs()
    assert len(solve_runs) == 2
    assert solve_runs[1].data.params == {}

    artifact_path: Path = tmp_path / "route.txt"
    artifact_path.write_text("selected route", encoding="utf-8")
    with pytest.warns(MlflowOptimizationWarning):
        first_solve.log_artifact(str(artifact_path))
    assert tracking.client.list_artifacts(first_solve.run_id) == []


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
    autolog(libraries=["fake"])
    solver: FakeSolver = FakeSolver(status="feasible")

    with pytest.warns(MlflowOptimizationWarning):
        result: str = solver.solve()

    assert result == "feasible"
    assert tracking.solve_runs() == []


def test_failed_recording_clears_last_solve_and_recovers(
    tracking: Tracking,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fakes import FakeSolver

    autolog(libraries=["fake"])
    first_result: str = FakeSolver(status="optimal").solve()
    first_solve: SolveRun | None = last_solve()

    assert first_result == "optimal"
    assert first_solve is not None
    assert len(tracking.solve_runs()) == 1

    def fail_create_run(
        client: MlflowClient, *arguments: object, **keywords: object
    ) -> Run:
        raise MlflowException("Tracking store is unavailable.")

    with monkeypatch.context() as failure:
        failure.setattr(MlflowClient, "create_run", fail_create_run)
        with pytest.warns(MlflowOptimizationWarning, match="Could not record solve"):
            failed_result: str = FakeSolver(status="feasible").solve()

        assert failed_result == "feasible"
        assert last_solve() is None
        assert len(tracking.solve_runs()) == 1

    first_solve.set_tags({"saved_handle": "still usable"})
    first_run: Run = tracking.client.get_run(first_solve.run_id)
    assert first_run.data.tags["saved_handle"] == "still usable"

    recovery_result: str = FakeSolver(status="optimal").solve()
    recovery_solve: SolveRun | None = last_solve()

    assert recovery_result == "optimal"
    assert recovery_solve is not None
    assert recovery_solve.run_id != first_solve.run_id
    assert len(tracking.solve_runs()) == 2


def test_solve_runs_have_no_artifacts_by_default(tracking: Tracking) -> None:
    from fakes import FakeSolver

    autolog(libraries=["fake"])
    solver: FakeSolver = FakeSolver()
    solver.solve()

    solve_runs: list[Run] = tracking.solve_runs()

    assert len(solve_runs) == 1
    assert tracking.client.list_artifacts(solve_runs[0].info.run_id) == []


def test_unavailable_solver_values_are_missing_with_reasons_and_other_values_remain(
    tracking: Tracking,
) -> None:
    from fakes import FakeSolver

    autolog(libraries=["fake"])
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

    for metric_name in unavailable_metrics:
        assert metric_name not in solve_run.data.metrics
        assert (
            solve_run.data.tags[f"optimization.missing.{metric_name}"] == "unsupported"
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

    autolog(libraries=["fake"])
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

    autolog(libraries=["fake"], log_artifacts=False)
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

    autolog(libraries=["fake"])
    solve_error: RuntimeError = RuntimeError("solver failed")
    solver: FakeSolver = FakeSolver(raises=solve_error)

    with pytest.raises(RuntimeError) as raised_error:
        solver.solve()

    assert raised_error.value is solve_error
    solve_run: Run = tracking.solve_runs()[0]
    assert solve_run.info.status == "FAILED"
    assert solve_run.data.tags["optimization.termination"] == "error"
    assert solve_run.data.tags["optimization.error"] == "RuntimeError"


def test_diagnostic_warning_as_error_does_not_change_or_suppress_solve(
    tracking: Tracking,
) -> None:
    from fakes import FakeSolver

    autolog(libraries=["fake"])
    solver: FakeSolver = FakeSolver(status="optimal", has_solution=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error", MlflowOptimizationWarning)
        result: str = solver.solve()

    assert result == "optimal"
    solve_runs: list[Run] = tracking.solve_runs()
    assert len(solve_runs) == 1
    assert solve_runs[0].data.tags["optimization.termination"] == "optimal"
    assert solve_runs[0].data.tags["optimization.has_solution"] == "false"


def test_failed_solve_payload_contains_only_documented_keys_and_no_error_message(
    tracking: Tracking,
) -> None:
    from fakes import FakeSolver

    autolog(libraries=["fake"])
    solver: FakeSolver = FakeSolver(raises=FileNotFoundError("/input/SECRET/data.csv"))

    with pytest.raises(FileNotFoundError):
        solver.solve()

    solve_run: Run = tracking.solve_runs()[0]
    documented_tag_keys: set[str] = {
        "optimization.termination",
        "optimization.raw_status",
        "optimization.has_solution",
        "optimization.has_objective",
        "optimization.objective_sense",
        "optimization.api",
        "optimization.solver",
        "optimization.api_version",
        "optimization.solver_version",
        "optimization.wall_time_includes_problem_build",
        "optimization.instance",
        "mlflow.note.content",
        "optimization.solve_id",
        "optimization.solve_index",
        "optimization.error",
    }
    documented_metric_keys: set[str] = {
        "objective_value",
        "dual_bound",
        "absolute_gap",
        "relative_gap",
        "wall_time_seconds",
        "solver_time_seconds",
        "node_count",
        "simplex_iterations",
        "barrier_iterations",
    }
    for tag_key in solve_run.data.tags:
        assert tag_key in documented_tag_keys or tag_key.startswith(
            ("optimization.missing.", "user.", "mlflow.")
        )

    for parameter_key in solve_run.data.params:
        assert parameter_key.startswith(
            (
                "problem.",
                "solver_param.",
                "solver_param_default.",
                "solver_param_effective.",
            )
        )

    for metric_key in solve_run.data.metrics:
        assert metric_key in documented_metric_keys or metric_key.startswith(
            ("fake.", "user.")
        )

    for tag_value in solve_run.data.tags.values():
        assert "SECRET" not in tag_value

    for parameter_value in solve_run.data.params.values():
        assert "SECRET" not in parameter_value

    for metric_value in solve_run.data.metrics.values():
        assert "SECRET" not in str(metric_value)

    assert solve_run.data.tags["optimization.error"] == "FileNotFoundError"
    assert tracking.client.list_artifacts(solve_run.info.run_id) == []


def test_wall_time_excludes_a_slow_run_store_save(
    tracking: Tracking,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from time import perf_counter, sleep

    from fakes import FakeSolver

    from mlflow_optimization.features.runs.types import RunReference, SolveRecord
    from mlflow_optimization.integrations.mlflow.runs import MlflowRunStore

    original_save: Callable[
        [MlflowRunStore, SolveRecord, RunReference | None], RunReference
    ] = MlflowRunStore.save
    store_delay_seconds: float = 0.3

    def slow_save(
        store: MlflowRunStore,
        record: SolveRecord,
        parent: RunReference | None,
    ) -> RunReference:
        sleep(store_delay_seconds)
        return original_save(store, record, parent)

    baseline_solver: FakeSolver = FakeSolver()
    solve_start: float = perf_counter()
    baseline_result: str = baseline_solver.solve()
    expected_solve_duration: float = perf_counter() - solve_start
    assert baseline_result == "optimal"

    monkeypatch.setattr(MlflowRunStore, "save", slow_save)
    autolog(libraries=["fake"])
    solver: FakeSolver = FakeSolver()
    result: str = solver.solve()

    assert result == "optimal"
    solve_run: Run = tracking.solve_runs()[0]
    recorded_wall_time: float = solve_run.data.metrics["wall_time_seconds"]
    assert abs(recorded_wall_time - expected_solve_duration) < 0.05
    assert recorded_wall_time < store_delay_seconds / 2


def test_unstoreable_context_tag_warns_without_losing_solve(
    tracking: Tracking,
) -> None:
    from hashlib import shake_256

    from fakes import FakeSolver

    autolog(libraries=["fake"])
    long_tag_key: str = shake_256(b"unstoreable tracker tag").hexdigest(200)
    assert len(long_tag_key) == 400
    solver: FakeSolver = FakeSolver()

    with (
        pytest.warns(MlflowOptimizationWarning, match=long_tag_key),
        context(tags={long_tag_key: "value"}),
    ):
        result: str = solver.solve()

    solve_run: SolveRun | None = last_solve()
    assert result == "optimal"
    assert solve_run is not None
    recorded_run: Run = tracking.client.get_run(solve_run.run_id)
    assert recorded_run.data.tags["optimization.termination"] == "optimal"
    assert long_tag_key not in recorded_run.data.tags
    assert recorded_run.info.run_id == tracking.solve_runs()[0].info.run_id
