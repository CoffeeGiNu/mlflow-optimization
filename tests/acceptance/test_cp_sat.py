import math
from typing import Any

from mlflow.entities import Run
from ortools.sat.python import cp_model

from mlflow_optimization import autolog


class SolutionCounter(cp_model.CpSolverSolutionCallback):
    """A counter of CP-SAT solution callbacks."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: int = 0

    def on_solution_callback(self) -> None:
        self.calls += 1


def test_cp_sat_records_an_optimal_optimization(tracking: Any) -> None:
    autolog(solvers=["cp_sat"])
    model: cp_model.CpModel = cp_model.CpModel()
    first: cp_model.IntVar = model.new_int_var(0, 10, "first")
    second: cp_model.IntVar = model.new_int_var(0, 10, "second")
    model.add(first + second >= 4)
    model.minimize(3 * first + 2 * second)
    solver: cp_model.CpSolver = cp_model.CpSolver()
    solver.solve(model)

    runs: list[Run] = tracking.solve_runs()
    assert len(runs) == 1
    run: Run = runs[0]
    tags: dict[str, str] = run.data.tags
    metrics: dict[str, float] = run.data.metrics
    assert tags["optimization.api"] == "cp_sat"
    assert tags["optimization.solver"] == "cp_sat"
    assert tags["optimization.termination"] == "optimal"
    assert tags["optimization.has_solution"] == "true"
    assert tags["optimization.has_objective"] == "true"
    assert "objective_value" in metrics
    assert "dual_bound" in metrics
    assert "absolute_gap" in metrics
    assert metrics["objective_value"] >= metrics["dual_bound"]


def test_cp_sat_records_genuine_zero_objective_and_bound(tracking: Any) -> None:
    autolog(solvers=["cp_sat"])
    model: cp_model.CpModel = cp_model.CpModel()
    value: cp_model.IntVar = model.new_int_var(0, 5, "value")
    model.minimize(value)
    solver: cp_model.CpSolver = cp_model.CpSolver()
    solver.solve(model)

    runs: list[Run] = tracking.solve_runs()
    assert len(runs) == 1
    metrics: dict[str, float] = runs[0].data.metrics
    tags: dict[str, str] = runs[0].data.tags
    assert metrics["objective_value"] == 0
    assert metrics["dual_bound"] == 0
    assert "optimization.missing.objective_value" not in tags
    assert "optimization.missing.dual_bound" not in tags


def test_cp_sat_records_an_infeasible_optimization(tracking: Any) -> None:
    autolog(solvers=["cp_sat"])
    model: cp_model.CpModel = cp_model.CpModel()
    value: cp_model.IntVar = model.new_bool_var("value")
    model.add(value == 0)
    model.add(value == 1)
    model.minimize(value)
    solver: cp_model.CpSolver = cp_model.CpSolver()
    solver.solve(model)

    runs: list[Run] = tracking.solve_runs()
    assert len(runs) == 1
    run: Run = runs[0]
    tags: dict[str, str] = run.data.tags
    metrics: dict[str, float] = run.data.metrics
    assert tags["optimization.api"] == "cp_sat"
    assert tags["optimization.solver"] == "cp_sat"
    assert tags["optimization.termination"] == "infeasible"
    assert tags["optimization.has_solution"] == "false"
    assert "objective_value" not in metrics
    assert "absolute_gap" not in metrics
    assert "relative_gap" not in metrics
    assert tags["optimization.missing.objective_value"] == "no_solution"
    assert tags["optimization.missing.dual_bound"] == "undefined"
    assert tags["optimization.missing.absolute_gap"] == "no_solution"
    assert tags["optimization.missing.relative_gap"] == "no_solution"


def test_cp_sat_deterministic_limit_keeps_zero_incumbent(tracking: Any) -> None:
    autolog(solvers=["cp_sat"])
    model: cp_model.CpModel = cp_model.CpModel()
    variables: list[cp_model.IntVar] = [
        model.new_bool_var(f"item_{index}") for index in range(60)
    ]
    for constraint_index in range(5):
        model.add(
            sum(
                ((index * 17 + constraint_index * 13) % 31 + 1) * variables[index]
                for index in range(60)
            )
            <= 180
        )
    model.maximize(sum((index * 11 % 37 + 1) * variables[index] for index in range(60)))
    for variable in variables:
        model.add_hint(variable, 0)
    solver: cp_model.CpSolver = cp_model.CpSolver()
    solver.parameters.max_deterministic_time = 0.0001
    solver.parameters.num_search_workers = 1
    solver.parameters.cp_model_presolve = False
    status: cp_model.CpSolverStatus = solver.solve(model)
    assert status == cp_model.FEASIBLE

    runs: list[Run] = tracking.solve_runs()
    assert len(runs) == 1
    run: Run = runs[0]
    tags: dict[str, str] = run.data.tags
    metrics: dict[str, float] = run.data.metrics
    assert tags["optimization.api"] == "cp_sat"
    assert tags["optimization.solver"] == "cp_sat"
    assert tags["optimization.termination"] == "feasible"
    assert tags["optimization.raw_status"].startswith("FEASIBLE")
    assert tags["optimization.has_solution"] == "true"
    assert metrics["objective_value"] == 0 == solver.objective_value
    assert math.isfinite(metrics["dual_bound"])
    assert metrics["dual_bound"] == solver.best_objective_bound > 0
    assert metrics["absolute_gap"] == metrics["dual_bound"]


def test_cp_sat_zero_time_limit_has_no_solution(tracking: Any) -> None:
    autolog(solvers=["cp_sat"])
    model: cp_model.CpModel = cp_model.CpModel()
    value: cp_model.IntVar = model.new_int_var(-10, -5, "value")
    model.minimize(value)
    solver: cp_model.CpSolver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 0

    status: cp_model.CpSolverStatus = solver.solve(model)

    assert status == cp_model.UNKNOWN
    runs: list[Run] = tracking.solve_runs()
    assert len(runs) == 1
    assert runs[0].data.tags["optimization.has_solution"] == "false"
    assert runs[0].data.tags["optimization.missing.objective_value"] == "no_solution"
    assert "objective_value" not in runs[0].data.metrics


def test_cp_sat_records_objective_less_sat_and_unsat_models(
    tracking: Any,
) -> None:
    autolog(solvers=["cp_sat"])
    cases: tuple[tuple[bool, str, str], ...] = (
        (True, "feasible", "true"),
        (False, "infeasible", "false"),
    )
    satisfiable: bool
    expected_termination: str
    expected_has_solution: str
    for case_index, case in enumerate(cases):
        satisfiable, expected_termination, expected_has_solution = case
        model: cp_model.CpModel = cp_model.CpModel()
        value: cp_model.IntVar = model.new_bool_var(f"value_{case_index}")
        model.add(value == 1)
        if not satisfiable:
            model.add(value == 0)
        solver: cp_model.CpSolver = cp_model.CpSolver()
        solver.solve(model)

    runs: list[Run] = tracking.solve_runs()
    assert len(runs) == len(cases)
    assert runs[0].info.run_id != runs[1].info.run_id
    for run_index, run in enumerate(runs):
        expected_termination = cases[run_index][1]
        expected_has_solution = cases[run_index][2]
        tags: dict[str, str] = run.data.tags
        metrics: dict[str, float] = run.data.metrics
        assert tags["optimization.api"] == "cp_sat"
        assert tags["optimization.solver"] == "cp_sat"
        assert tags["optimization.termination"] == expected_termination
        assert tags["optimization.has_solution"] == expected_has_solution
        assert tags["optimization.has_objective"] == "false"
        assert "objective_value" not in metrics
        assert "dual_bound" not in metrics
        assert "absolute_gap" not in metrics
        assert "relative_gap" not in metrics
        assert "wall_time_seconds" in metrics
        for metric_name in (
            "objective_value",
            "dual_bound",
            "absolute_gap",
            "relative_gap",
        ):
            assert tags[f"optimization.missing.{metric_name}"] == "undefined"


def test_cp_sat_branches_and_conflicts_use_native_metric_names(
    tracking: Any,
) -> None:
    autolog(solvers=["cp_sat"])
    model: cp_model.CpModel = cp_model.CpModel()
    first: cp_model.IntVar = model.new_bool_var("first")
    second: cp_model.IntVar = model.new_bool_var("second")
    model.add(first + second >= 1)
    model.minimize(first + second)
    solver: cp_model.CpSolver = cp_model.CpSolver()
    solver.solve(model)

    runs: list[Run] = tracking.solve_runs()
    assert len(runs) == 1
    metrics: dict[str, float] = runs[0].data.metrics
    assert "cp_sat.num_branches" in metrics
    assert "cp_sat.num_conflicts" in metrics
    assert "num_branches" not in metrics
    assert "num_conflicts" not in metrics


def test_cp_sat_legacy_solve_alias_records_once_and_calls_user_callback(
    tracking: Any,
) -> None:
    autolog(solvers=["cp_sat"])
    model: cp_model.CpModel = cp_model.CpModel()
    value: cp_model.IntVar = model.new_bool_var("value")
    model.add(value == 1)
    solver: cp_model.CpSolver = cp_model.CpSolver()
    callback: SolutionCounter = SolutionCounter()
    solver.Solve(model, callback)

    runs: list[Run] = tracking.solve_runs()
    assert len(runs) == 1
    run: Run = runs[0]
    tags: dict[str, str] = run.data.tags
    assert tags["optimization.api"] == "cp_sat"
    assert tags["optimization.solver"] == "cp_sat"
    assert tags["optimization.has_solution"] == "true"
    assert callback.calls > 0
