"""Acceptance tests for Pyomo solver observations."""

import math
from typing import Any

import pytest
from mlflow.entities import Run
from mlflow_optimization import autolog
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Objective,
    SolverFactory,
    Var,
    maximize,
    minimize,
)


def _solver_or_skip(solver_name: str) -> Any:
    solver: Any = SolverFactory(solver_name)
    available: bool = solver.available(exception_flag=False)
    if not available:
        pytest.skip(f"The {solver_name} backend is unavailable.")
    return solver


def _bounded_model(objective_sense: Any) -> Any:
    model: Any = ConcreteModel()
    model.value = Var()
    model.bounds = Constraint(expr=(2, model.value, 5))
    model.objective = Objective(expr=model.value, sense=objective_sense)
    return model


def _single_solve_run(tracking: Any) -> Run:
    runs: list[Run] = tracking.solve_runs()
    assert len(runs) == 1
    run: Run = runs[0]
    metric_values: dict[str, float] = run.data.metrics
    tags: dict[str, str] = run.data.tags
    common_metric_names: tuple[str, ...] = (
        "objective_value",
        "dual_bound",
        "absolute_gap",
        "relative_gap",
        "wall_time_seconds",
        "solver_time_seconds",
        "node_count",
        "simplex_iterations",
        "barrier_iterations",
    )
    assert all(math.isfinite(metric_value) for metric_value in metric_values.values())
    metric_name: str
    for metric_name in common_metric_names:
        missing_tag: str = f"optimization.missing.{metric_name}"
        assert (metric_name in metric_values) != (missing_tag in tags)
    return run


@pytest.mark.parametrize(
    ("objective_sense", "expected_sense"),
    ((minimize, "minimize"), (maximize, "maximize")),
)
def test_highs_optimal_run_records_objective_direction(
    tracking: Any,
    objective_sense: Any,
    expected_sense: str,
) -> None:
    autolog(solvers=["pyomo"])
    solver: Any = _solver_or_skip("highs")
    model: Any = _bounded_model(objective_sense)

    solver.solve(model)

    run: Run = _single_solve_run(tracking)
    tags: dict[str, str] = run.data.tags
    metrics: dict[str, float] = run.data.metrics
    assert tags["optimization.api"] == "pyomo"
    assert tags["optimization.solver"] == "highs"
    assert tags["optimization.termination"] == "optimal"
    assert tags["optimization.has_solution"] == "true"
    assert tags["optimization.has_objective"] == "true"
    assert tags["optimization.objective_sense"] == expected_sense
    assert {"objective_value", "dual_bound"}.issubset(metrics)
    assert "optimization.missing.objective_value" not in tags
    assert "optimization.missing.dual_bound" not in tags
    if expected_sense == "minimize":
        assert metrics["dual_bound"] <= metrics["objective_value"]
    else:
        assert metrics["dual_bound"] >= metrics["objective_value"]


def test_highs_infeasible_run_has_no_objective_or_gap(tracking: Any) -> None:
    autolog(solvers=["pyomo"])
    solver: Any = _solver_or_skip("highs")
    model: Any = ConcreteModel()
    model.value = Var()
    model.lower_bound = Constraint(expr=model.value >= 1)
    model.upper_bound = Constraint(expr=model.value <= 0)
    model.objective = Objective(expr=model.value, sense=minimize)

    solver.solve(model, load_solutions=False)

    run: Run = _single_solve_run(tracking)
    tags: dict[str, str] = run.data.tags
    metrics: dict[str, float] = run.data.metrics
    assert tags["optimization.termination"] == "infeasible"
    assert tags["optimization.has_solution"] == "false"
    assert "objective_value" not in metrics
    assert tags["optimization.missing.objective_value"] == "no_solution"
    assert tags["optimization.missing.absolute_gap"] == "no_solution"
    assert tags["optimization.missing.relative_gap"] == "no_solution"


def test_highs_objective_free_run_marks_objective_metrics_undefined(
    tracking: Any,
) -> None:
    autolog(solvers=["pyomo"])
    solver: Any = _solver_or_skip("highs")
    model: Any = ConcreteModel()
    model.value = Var(bounds=(0, 1))
    model.lower_bound = Constraint(expr=model.value >= 0.25)

    solver.solve(model)

    run: Run = _single_solve_run(tracking)
    tags: dict[str, str] = run.data.tags
    metrics: dict[str, float] = run.data.metrics
    objective_metrics: tuple[str, ...] = (
        "objective_value",
        "dual_bound",
        "absolute_gap",
        "relative_gap",
    )
    assert tags["optimization.has_objective"] == "false"
    for metric_name in objective_metrics:
        assert metric_name not in metrics
        assert tags[f"optimization.missing.{metric_name}"] == "undefined"


def test_gurobi_direct_overlap_records_one_pyomo_run(tracking: Any) -> None:
    autolog(solvers=["pyomo", "gurobipy"])
    solver: Any = _solver_or_skip("gurobi_direct")
    model: Any = _bounded_model(minimize)

    solver.solve(model)

    runs: list[Run] = tracking.solve_runs()
    assert len(runs) == 1
    tags: dict[str, str] = runs[0].data.tags
    assert tags["optimization.api"] == "pyomo"
    assert tags["optimization.solver"] == "gurobi"
    assert tags["optimization.termination"] == "optimal"
    assert tags["optimization.has_solution"] == "true"
