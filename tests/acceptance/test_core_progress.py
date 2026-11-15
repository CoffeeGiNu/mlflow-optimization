from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from operator import attrgetter
from threading import Event
from typing import TYPE_CHECKING, cast

import pytest
from mlflow.entities import Metric, Run
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.utils.async_logging.run_operations import RunOperations

from mlflow_optimization import (
    MlflowOptimizationWarning,
    SolveRun,
    autolog,
    last_solve,
)
from mlflow_optimization.features.progress.recorder import ProgressRecorder

if TYPE_CHECKING:
    from conftest import Tracking


def test_progress_history_is_written_to_the_running_solve_and_final_values_are_latest(
    tracking: Tracking,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fakes import FakeSolver

    history_written: Event = Event()
    original_log_batch: Callable[..., RunOperations | None] = MlflowClient.log_batch

    def observe_history_write(
        client: MlflowClient,
        *arguments: object,
        **keywords: object,
    ) -> RunOperations | None:
        result = original_log_batch(client, *arguments, **keywords)
        if any(
            metric.key == "objective_value" and metric.value == 9.0
            for metric in cast(Sequence[Metric], keywords.get("metrics", ()))
        ):
            history_written.set()
        return result

    monkeypatch.setattr(MlflowClient, "log_batch", observe_history_write)
    monkeypatch.setattr(
        "mlflow_optimization.features.runs.record.ProgressRecorder",
        partial(ProgressRecorder, interval_seconds=0.01),
    )
    runs_during_solve: list[list[Run]] = []
    solve_handles_during_solve: list[SolveRun | None] = []

    def inspect_running_solve() -> None:
        runs_during_solve.append(
            tracking.client.search_runs(experiment_ids=[tracking.experiment_id])
        )
        solve_handles_during_solve.append(last_solve())
        assert history_written.wait(timeout=2.0)
        history: list[Metric] = tracking.client.get_metric_history(
            runs_during_solve[-1][0].info.run_id, "objective_value"
        )
        assert any(metric.value == 9.0 for metric in history)

    solver: FakeSolver = FakeSolver(
        objective=8.0,
        bound=3.0,
        progress_samples=((10.0, 4.0), (9.0, 6.0)),
        during_solve=inspect_running_solve,
    )

    autolog(libraries=["fake"])
    result: str = solver.solve()

    assert result == "optimal"
    assert len(runs_during_solve) == 1
    assert len(runs_during_solve[0]) == 1
    assert runs_during_solve[0][0].info.status == "RUNNING"
    assert len(solve_handles_during_solve) == 1
    assert solve_handles_during_solve[0] is None

    solve_run: Run = tracking.solve_runs()[0]
    recorded_solve: SolveRun | None = last_solve()
    assert recorded_solve is not None
    assert recorded_solve.run_id == solve_run.info.run_id
    assert solve_run.info.status == "FINISHED"

    progress_values: dict[str, float] = {
        "objective_value": 9.0,
        "dual_bound": 6.0,
        "absolute_gap": 3.0,
        "relative_gap": 3.0 / 9.0,
    }
    final_values: dict[str, float] = {
        "objective_value": 8.0,
        "dual_bound": 3.0,
        "absolute_gap": 5.0,
        "relative_gap": 5.0 / 8.0,
    }
    for metric_name, progress_value in progress_values.items():
        history: list[Metric] = tracking.client.get_metric_history(
            solve_run.info.run_id,
            metric_name,
        )
        history_steps: list[int] = [metric.step for metric in history]
        assert history_steps == sorted(set(history_steps))
        latest_point: Metric = max(history, key=attrgetter("step", "timestamp"))
        latest_step_points: list[Metric] = [
            metric for metric in history if metric.step == latest_point.step
        ]
        assert len(latest_step_points) == 1
        prior_points: list[Metric] = [
            metric for metric in history if metric.step < latest_point.step
        ]
        assert prior_points
        latest_progress_point: Metric = max(
            prior_points,
            key=attrgetter("step", "timestamp"),
        )
        assert latest_progress_point.value == pytest.approx(progress_value)
        assert latest_point.value == pytest.approx(final_values[metric_name])
        assert latest_point.step > max(metric.step for metric in prior_points)
        assert solve_run.data.metrics[metric_name] == pytest.approx(
            final_values[metric_name]
        )


def test_disabling_progress_keeps_completed_solve_metrics_at_step_zero(
    tracking: Tracking,
) -> None:
    from fakes import FakeSolver

    runs_during_solve: list[list[Run]] = []

    def inspect_during_solve() -> None:
        runs_during_solve.append(
            tracking.client.search_runs(experiment_ids=[tracking.experiment_id])
        )

    solver: FakeSolver = FakeSolver(
        objective=8.0,
        bound=3.0,
        progress_samples=((10.0, 4.0), (9.0, 6.0)),
        during_solve=inspect_during_solve,
    )

    autolog(libraries=["fake"], log_progress=False)
    result: str = solver.solve()

    assert result == "optimal"
    assert runs_during_solve == [[]]
    solve_run: Run = tracking.solve_runs()[0]
    for metric_name in (
        "objective_value",
        "dual_bound",
        "absolute_gap",
        "relative_gap",
    ):
        history: list[Metric] = tracking.client.get_metric_history(
            solve_run.info.run_id,
            metric_name,
        )
        assert len(history) == 1
        assert history[0].step == 0
        assert solve_run.data.metrics[metric_name] == history[0].value


def test_failed_run_start_falls_back_to_a_completed_solve_record(
    tracking: Tracking,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fakes import FakeSolver

    original_create_run: Callable[..., Run] = MlflowClient.create_run
    create_run_attempts: int = 0

    def fail_first_create_run(
        client: MlflowClient,
        *arguments: object,
        **keywords: object,
    ) -> Run:
        nonlocal create_run_attempts
        create_run_attempts += 1
        if create_run_attempts == 1:
            raise MlflowException("The first run creation failed.")
        return original_create_run(client, *arguments, **keywords)

    monkeypatch.setattr(MlflowClient, "create_run", fail_first_create_run)
    solver: FakeSolver = FakeSolver(
        status="feasible",
        progress_samples=((10.0, 4.0),),
    )

    autolog(libraries=["fake"])
    with pytest.warns(MlflowOptimizationWarning):
        result: str = solver.solve()

    assert result == "feasible"
    assert create_run_attempts >= 2
    solve_run: Run = tracking.solve_runs()[0]
    assert solve_run.info.status == "FINISHED"
    assert solve_run.data.metrics["objective_value"] == 1.0
    objective_history: list[Metric] = tracking.client.get_metric_history(
        solve_run.info.run_id,
        "objective_value",
    )
    assert [metric.step for metric in objective_history] == [0]
    recorded_solve: SolveRun | None = last_solve()
    assert recorded_solve is not None
    assert recorded_solve.run_id == solve_run.info.run_id


def test_failed_start_tags_keep_one_run_and_finish_it(
    tracking: Tracking,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fakes import FakeSolver

    original_log_batch: Callable[..., RunOperations | None] = MlflowClient.log_batch
    failed_tags: bool = False

    def fail_first_tag_batch(
        client: MlflowClient,
        *arguments: object,
        **keywords: object,
    ) -> RunOperations | None:
        nonlocal failed_tags
        if keywords.get("tags") and not failed_tags:
            failed_tags = True
            raise MlflowException("The first tag write failed.")
        return original_log_batch(client, *arguments, **keywords)

    monkeypatch.setattr(MlflowClient, "log_batch", fail_first_tag_batch)
    autolog(libraries=["fake"])
    with pytest.warns(MlflowOptimizationWarning):
        result: str = FakeSolver().solve()

    assert result == "optimal"
    assert failed_tags
    solve_runs: list[Run] = tracking.solve_runs()
    assert len(solve_runs) == 1
    assert solve_runs[0].info.status == "FINISHED"
    recorded_solve: SolveRun | None = last_solve()
    assert recorded_solve is not None
    assert recorded_solve.run_id == solve_runs[0].info.run_id


def test_fallback_record_still_publishes_the_finished_solve(
    tracking: Tracking,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fakes import FakeSolver

    def fail_build_record(*arguments: object, **keywords: object) -> None:
        raise RuntimeError("Could not build the final record.")

    monkeypatch.setattr(
        "mlflow_optimization.features.runs.record.RecordSolve._build_record",
        fail_build_record,
    )
    autolog(libraries=["fake"])
    with pytest.warns(MlflowOptimizationWarning, match="Solve recording failed"):
        result: str = FakeSolver().solve()

    assert result == "optimal"
    solve_runs: list[Run] = tracking.solve_runs()
    assert len(solve_runs) == 1
    assert solve_runs[0].info.status == "FINISHED"
    recorded_solve: SolveRun | None = last_solve()
    assert recorded_solve is not None
    assert recorded_solve.run_id == solve_runs[0].info.run_id


def test_history_write_failure_warns_but_preserves_solver_exception_and_final_record(
    tracking: Tracking,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fakes import FakeSolver

    original_log_batch: Callable[..., RunOperations | None] = MlflowClient.log_batch
    history_write_failed: bool = False

    def fail_progress_batch(
        client: MlflowClient,
        *arguments: object,
        **keywords: object,
    ) -> RunOperations | None:
        nonlocal history_write_failed
        metric_batch_value: object = keywords.get(
            "metrics",
            arguments[1] if len(arguments) > 1 else (),
        )
        metric_batch: tuple[Metric, ...] = tuple(
            cast(Sequence[Metric], metric_batch_value)
        )
        if any(
            metric.key == "objective_value" and metric.value == 9.0
            for metric in metric_batch
        ):
            history_write_failed = True
            raise MlflowException("A progress history write failed.")
        return original_log_batch(client, *arguments, **keywords)

    monkeypatch.setattr(MlflowClient, "log_batch", fail_progress_batch)
    solve_error: RuntimeError = RuntimeError("The solver failed after progress.")
    solver: FakeSolver = FakeSolver(
        objective=8.0,
        bound=3.0,
        raises=solve_error,
        progress_samples=((10.0, 4.0), (9.0, 6.0)),
    )

    autolog(libraries=["fake"])
    with (
        pytest.warns(MlflowOptimizationWarning),
        pytest.raises(RuntimeError) as raised_error,
    ):
        solver.solve()

    assert raised_error.value is solve_error
    assert history_write_failed
    solve_run: Run = tracking.solve_runs()[0]
    assert solve_run.info.status == "FAILED"
    assert solve_run.data.tags["optimization.error"] == "RuntimeError"
    recorded_solve: SolveRun | None = last_solve()
    assert recorded_solve is not None
    assert recorded_solve.run_id == solve_run.info.run_id
