import math
from pathlib import Path

import pytest
from mlflow.entities import FileInfo, Run
from ortools.constraint_solver import (
    pywrapcp,
    routing_enums_pb2,
    routing_parameters_pb2,
)

from conftest import Tracking
from mlflow_optimization import SolveRun, autolog, last_solve

DISTANCE_MATRIX: tuple[tuple[int, ...], ...] = (
    (0, 7, 9, 8, 6),
    (7, 0, 5, 6, 8),
    (9, 5, 0, 7, 4),
    (8, 6, 7, 0, 3),
    (6, 8, 4, 3, 0),
)


@pytest.mark.parametrize(
    "vehicle_count",
    (1, 2),
    ids=("tsp", "vrp"),
)
def test_routing_solve_records_meaningful_results(
    tracking: Tracking,
    vehicle_count: int,
) -> None:
    autolog(solvers=["routing"])

    assignment: pywrapcp.Assignment | None = _solve_routing_problem(vehicle_count)

    assert assignment is not None
    runs: list[Run] = tracking.solve_runs()
    assert len(runs) == 1
    run: Run = runs[0]
    tags: dict[str, str] = run.data.tags
    assert tags["optimization.api"] == "routing"
    assert tags["optimization.solver"] == "routing"
    assert tags["optimization.has_solution"] == "true"
    assert tags["optimization.has_objective"] == "true"
    assert tags["optimization.objective_sense"] == "minimize"
    raw_status: str = tags["optimization.raw_status"]
    successful_statuses: tuple[str, ...] = (
        "ROUTING_SUCCESS (1)",
        "ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED (2)",
        "ROUTING_OPTIMAL (7)",
    )
    assert raw_status in successful_statuses
    expected_termination: str = (
        "optimal" if raw_status == "ROUTING_OPTIMAL (7)" else "feasible"
    )
    assert tags["optimization.termination"] == expected_termination
    assert "objective_value" in run.data.metrics
    assert math.isfinite(run.data.metrics["objective_value"])
    assert "routing.branches" in run.data.metrics
    assert "routing.failures" in run.data.metrics
    assert "dual_bound" not in run.data.metrics
    assert "absolute_gap" not in run.data.metrics
    assert "relative_gap" not in run.data.metrics
    missing_metrics: tuple[str, ...] = (
        "dual_bound",
        "absolute_gap",
        "relative_gap",
    )
    metric_name: str
    for metric_name in missing_metrics:
        assert (
            tags[f"optimization.missing.{metric_name}"] == "unsupported"
        )
    assert run.data.params["problem.num_locations"] == str(len(DISTANCE_MATRIX))
    assert run.data.params["problem.num_vehicles"] == str(vehicle_count)
    assert tracking.client.list_artifacts(run.info.run_id) == []
    recorded_keys: tuple[str, ...] = (
        *run.data.tags.keys(),
        *run.data.params.keys(),
        *run.data.metrics.keys(),
    )
    recorded_key: str
    for recorded_key in recorded_keys:
        assert "route" not in recorded_key.casefold()


def test_requested_route_artifact_is_attached_only_to_its_solve(
    tracking: Tracking,
    tmp_path: Path,
) -> None:
    autolog(solvers=["routing"])

    first_assignment: pywrapcp.Assignment | None = _solve_routing_problem(1)
    second_assignment: pywrapcp.Assignment | None = _solve_routing_problem(2)

    assert first_assignment is not None
    assert second_assignment is not None
    runs: list[Run] = tracking.solve_runs()
    assert len(runs) == 2
    first_run: Run = runs[0]
    second_run: Run = runs[1]
    assert tracking.client.list_artifacts(first_run.info.run_id) == []
    assert tracking.client.list_artifacts(second_run.info.run_id) == []

    route_file: Path = tmp_path / "route.txt"
    route_file.write_text("0 -> 1 -> 2 -> 3 -> 4 -> 0\n", encoding="utf-8")
    solve_run: SolveRun | None = last_solve()
    assert solve_run is not None
    assert solve_run.run_id == second_run.info.run_id
    solve_run.log_artifact(str(route_file))

    assert tracking.client.list_artifacts(first_run.info.run_id) == []
    artifacts: list[FileInfo] = tracking.client.list_artifacts(second_run.info.run_id)
    artifact_paths: list[str] = []
    artifact: FileInfo
    for artifact in artifacts:
        artifact_paths.append(artifact.path)
    assert artifact_paths == ["route.txt"]


def _solve_routing_problem(vehicle_count: int) -> pywrapcp.Assignment | None:
    routing_index_manager: pywrapcp.RoutingIndexManager = (
        pywrapcp.RoutingIndexManager(len(DISTANCE_MATRIX), vehicle_count, 0)
    )
    routing_model: pywrapcp.RoutingModel = pywrapcp.RoutingModel(routing_index_manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_location: int = routing_index_manager.IndexToNode(from_index)
        to_location: int = routing_index_manager.IndexToNode(to_index)
        return DISTANCE_MATRIX[from_location][to_location]

    transit_callback_index: int = routing_model.RegisterTransitCallback(
        distance_callback
    )
    routing_model.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters: routing_parameters_pb2.RoutingSearchParameters = (
        pywrapcp.DefaultRoutingSearchParameters()
    )
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    return routing_model.SolveWithParameters(search_parameters)
