from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Barrier
from typing import TYPE_CHECKING

import mlflow
import pytest
from mlflow.entities import Run, RunData
from mlflow.tracking.fluent import ActiveRun

from mlflow_optimization import autolog, context

if TYPE_CHECKING:
    from conftest import Tracking


def test_thirty_solves_keep_their_instance_and_parameters(tracking: Tracking) -> None:
    from fakes import FakeSolver

    autolog(solvers=["fake"])
    parameter_settings: tuple[tuple[int, float], ...] = (
        (1, 0.1),
        (2, 0.2),
        (4, 0.4),
    )
    expected_parameters: dict[tuple[str, str], dict[str, str]] = {}
    instance_index: int
    for instance_index in range(10):
        instance_name: str = f"instance-{instance_index:02}"
        setting_threads: int
        setting_relative_gap: float
        for setting_threads, setting_relative_gap in parameter_settings:
            solver_parameters: dict[str, int | float | str | bool] = {
                "threads": setting_threads,
                "relative_gap": setting_relative_gap,
            }
            with context(instance=instance_name):
                result: str = FakeSolver(parameters=solver_parameters).solve()
            assert result == "optimal"
            expected_parameters[(instance_name, str(setting_threads))] = {
                "solver_param.threads": str(setting_threads),
                "solver_param.relative_gap": str(setting_relative_gap),
            }

    solve_runs: list[Run] = tracking.solve_runs()
    assert len(solve_runs) == 30
    run_ids: list[str] = []
    solve_indexes: list[int] = []
    observed_runs: dict[tuple[str, str], Run] = {}
    run: Run
    for run in solve_runs:
        run_ids.append(run.info.run_id)
        solve_indexes.append(int(run.data.tags["optimization.solve_index"]))
        run_instance: str = run.data.tags["optimization.instance"]
        run_threads: str = run.data.params["solver_param.threads"]
        identity: tuple[str, str] = (run_instance, run_threads)
        assert identity not in observed_runs
        observed_runs[identity] = run
        assert run.data.tags["optimization.api"] == "fake"

    assert len(set(run_ids)) == 30
    first_solve_index: int = solve_indexes[0]
    assert solve_indexes == list(range(first_solve_index, first_solve_index + 30))
    assert set(observed_runs) == set(expected_parameters)
    identity: tuple[str, str]
    for identity, run in observed_runs.items():
        assert run.data.params == expected_parameters[identity]


def test_active_user_run_is_preserved_and_parent_of_solve_run(
    tracking: Tracking,
) -> None:
    from fakes import FakeSolver

    autolog(solvers=["fake"])
    mlflow.start_run(run_name="user-parent")
    parent_run_before: ActiveRun | None = mlflow.active_run()
    assert parent_run_before is not None
    parent_run_id: str = parent_run_before.info.run_id
    parent_experiment_id: str = parent_run_before.info.experiment_id
    mlflow.set_tag("user.owner", "parent")
    mlflow.log_param("owner_parameter", "preserved")
    mlflow.log_metric("owner_metric", 3.0)
    parent_data_before: RunData = tracking.client.get_run(parent_run_id).data
    parent_tags_before: dict[str, str] = parent_data_before.tags.copy()
    parent_parameters_before: dict[str, str] = parent_data_before.params.copy()
    parent_metrics_before: dict[str, float] = parent_data_before.metrics.copy()

    solver: FakeSolver = FakeSolver(parameters={"threads": 2})
    with context(instance="child-instance"):
        result: str = solver.solve()

    assert result == "optimal"
    active_run_after: ActiveRun | None = mlflow.active_run()
    assert active_run_after is not None
    assert active_run_after.info.run_id == parent_run_id
    parent_run_after: Run = tracking.client.get_run(parent_run_id)
    assert parent_run_after.info.status == "RUNNING"
    assert parent_run_after.data.tags == parent_tags_before
    assert parent_run_after.data.params == parent_parameters_before
    assert parent_run_after.data.metrics == parent_metrics_before

    solve_runs: list[Run] = tracking.solve_runs()
    assert len(solve_runs) == 1
    child_run: Run = solve_runs[0]
    assert child_run.info.run_id != parent_run_id
    assert child_run.info.experiment_id == parent_experiment_id
    assert child_run.data.tags["mlflow.parentRunId"] == parent_run_id
    assert child_run.data.tags["optimization.instance"] == "child-instance"
    assert int(child_run.data.tags["optimization.solve_index"]) > 0
    mlflow.end_run()


def test_nested_solver_entry_points_record_only_the_outer_api(
    tracking: Tracking,
) -> None:
    from fakes import FakeSolver
    from fakes_frontend import FakeFrontend

    autolog(solvers=["fake", "fake_frontend"])
    frontend: FakeFrontend = FakeFrontend(FakeSolver())
    result: str = frontend.solve()

    assert result == "optimal"
    solve_runs: list[Run] = tracking.solve_runs()
    assert len(solve_runs) == 1
    solve_run: Run = solve_runs[0]
    assert solve_run.data.tags["optimization.api"] == "fake_frontend"
    assert int(solve_run.data.tags["optimization.solve_index"]) > 0


def test_parallel_thread_solves_are_recorded_separately(tracking: Tracking) -> None:
    from fakes import FakeSolver

    autolog(solvers=["fake"])
    worker_count: int = 4
    start_barrier: Barrier = Barrier(worker_count)

    def solve_in_thread(instance_index: int) -> str:
        instance_name: str = f"thread-instance-{instance_index}"
        solver_parameters: dict[str, int | float | str | bool] = {
            "threads": instance_index + 1,
        }
        solver: FakeSolver = FakeSolver(parameters=solver_parameters)
        with context(instance=instance_name):
            start_barrier.wait()
            return solver.solve()

    futures: list[Future[str]] = []
    expected_parameters: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        instance_index: int
        for instance_index in range(worker_count):
            instance_name: str = f"thread-instance-{instance_index}"
            expected_parameters[instance_name] = {
                "solver_param.threads": str(instance_index + 1),
            }
            future: Future[str] = executor.submit(solve_in_thread, instance_index)
            futures.append(future)

        statuses: list[str] = []
        future: Future[str]
        for future in futures:
            statuses.append(future.result())

    assert statuses == ["optimal"] * worker_count
    solve_runs: list[Run] = tracking.solve_runs()
    assert len(solve_runs) == worker_count
    run_ids: list[str] = []
    solve_indexes: list[int] = []
    observed_parameters: dict[str, dict[str, str]] = {}
    run: Run
    for run in solve_runs:
        run_ids.append(run.info.run_id)
        solve_indexes.append(int(run.data.tags["optimization.solve_index"]))
        instance_name: str = run.data.tags["optimization.instance"]
        assert instance_name not in observed_parameters
        observed_parameters[instance_name] = run.data.params
        assert run.data.tags["optimization.api"] == "fake"

    assert len(set(run_ids)) == worker_count
    first_solve_index: int = solve_indexes[0]
    assert solve_indexes == list(
        range(first_solve_index, first_solve_index + worker_count)
    )
    assert observed_parameters == expected_parameters


def test_inherited_async_task_observes_solves_after_outer_solve_ends(
    tracking: Tracking,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fakes import FakeSolver

    original_solve: Callable[..., str] = FakeSolver.solve

    async def run_inherited_solve() -> tuple[str, str]:
        outer_solve_finished: asyncio.Event = asyncio.Event()
        inherited_task: asyncio.Task[str] | None = None
        schedule_inherited_task: bool = True

        async def solve_after_outer() -> str:
            await outer_solve_finished.wait()
            solver: FakeSolver = FakeSolver(parameters={"threads": 2})
            return solver.solve()

        def schedule_task_during_solve(solver: FakeSolver) -> str:
            nonlocal inherited_task
            nonlocal schedule_inherited_task
            if schedule_inherited_task:
                schedule_inherited_task = False
                inherited_task = asyncio.create_task(solve_after_outer())
            return original_solve(solver)

        monkeypatch.setattr(FakeSolver, "solve", schedule_task_during_solve)
        autolog(solvers=["fake"])
        try:
            outer_result: str = FakeSolver().solve()
            assert inherited_task is not None
            outer_solve_finished.set()
            inherited_result: str = await inherited_task
            return outer_result, inherited_result
        finally:
            outer_solve_finished.set()
            autolog(disable=True)

    solve_results: tuple[str, str] = asyncio.run(run_inherited_solve())
    solve_runs: list[Run] = tracking.solve_runs()

    assert solve_results == ("optimal", "optimal")
    assert len(solve_runs) == 2
    assert solve_runs[0].data.params == {}
    assert solve_runs[1].data.params == {"solver_param.threads": "2"}
    run: Run
    for run in solve_runs:
        assert run.data.tags["optimization.api"] == "fake"
