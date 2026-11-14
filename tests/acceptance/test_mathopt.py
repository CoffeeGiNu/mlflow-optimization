"""Acceptance tests for the MathOpt integration."""

import pytest
from conftest import Tracking
from mlflow.entities import Run
from ortools.math_opt.python import mathopt
from ortools.math_opt.python.model import Model
from ortools.math_opt.python.parameters import SolverType
from ortools.math_opt.python.variables import Variable

from mlflow_optimization import autolog


def _make_bounded_model(*, maximize: bool = False, integer: bool = False) -> Model:
    """A bounded linear optimization problem."""
    model: Model = Model()
    variable: Variable = model.add_variable(
        lb=0.0,
        ub=10.0,
        is_integer=integer,
        name="quantity",
    )
    model.add_linear_constraint(variable >= 4.0)
    if maximize:
        model.maximize(variable)
    else:
        model.minimize(variable)
    return model


def _make_infeasible_model() -> Model:
    """A linear problem with contradictory bounds."""
    model: Model = Model()
    variable: Variable = model.add_variable(lb=0.0, ub=1.0, name="quantity")
    model.add_linear_constraint(variable >= 2.0)
    model.minimize(variable)
    return model


def _make_unbounded_model() -> Model:
    """A linear problem with an unbounded objective."""
    model: Model = Model()
    variable: Variable = model.add_variable(lb=0.0, name="quantity")
    model.maximize(variable)
    return model


def _solve(model: Model, solver_type: SolverType) -> None:
    """A MathOpt solve with automatic logging enabled by the caller."""
    mathopt.solve(model, solver_type=solver_type)


def _single_solve_run(tracking: Tracking) -> Run:
    """The sole solve run recorded by a test."""
    solve_runs: list[Run] = tracking.solve_runs()
    assert len(solve_runs) == 1
    return solve_runs[0]


def test_optimal_linear_problem_is_recorded(tracking: Tracking) -> None:
    autolog(solvers=["mathopt"])
    _solve(_make_bounded_model(), SolverType.GLOP)

    run: Run = _single_solve_run(tracking)

    assert run.data.tags["optimization.api"] == "mathopt"
    assert run.data.tags["optimization.solver"] == "glop"
    assert run.data.tags["optimization.termination"] == "optimal"
    assert run.data.tags["optimization.has_solution"] == "true"
    assert "objective_value" in run.data.metrics
    assert "dual_bound" in run.data.metrics


def test_optimal_mixed_integer_problem_is_recorded(tracking: Tracking) -> None:
    autolog(solvers=["mathopt"])
    _solve(_make_bounded_model(integer=True), SolverType.GSCIP)

    run: Run = _single_solve_run(tracking)

    assert run.data.tags["optimization.api"] == "mathopt"
    assert run.data.tags["optimization.solver"] == "scip"
    assert run.data.tags["optimization.termination"] == "optimal"
    assert run.data.tags["optimization.has_solution"] == "true"
    assert "objective_value" in run.data.metrics
    assert "dual_bound" in run.data.metrics


def test_infeasible_problem_records_no_objective_value(tracking: Tracking) -> None:
    autolog(solvers=["mathopt"])
    _solve(_make_infeasible_model(), SolverType.GLOP)

    run: Run = _single_solve_run(tracking)

    assert run.data.tags["optimization.termination"] == "infeasible"
    assert run.data.tags["optimization.has_solution"] == "false"
    assert "objective_value" not in run.data.metrics
    assert (
        run.data.tags["optimization.missing.objective_value"] == "no_solution"
    )


def test_unbounded_problem_keeps_feasible_solution_and_omits_infinite_bound(
    tracking: Tracking,
) -> None:
    autolog(solvers=["mathopt"])
    _solve(_make_unbounded_model(), SolverType.GSCIP)

    run: Run = _single_solve_run(tracking)

    assert run.data.tags["optimization.termination"] == "unbounded"
    assert run.data.tags["optimization.has_solution"] == "true"
    assert "objective_value" in run.data.metrics
    assert "dual_bound" not in run.data.metrics
    assert run.data.tags["optimization.missing.dual_bound"] == "undefined"


@pytest.mark.parametrize(
    ("maximize", "expected_sense"),
    ((False, "minimize"), (True, "maximize")),
)
def test_objective_and_bound_follow_objective_direction(
    tracking: Tracking,
    maximize: bool,
    expected_sense: str,
) -> None:
    autolog(solvers=["mathopt"])
    _solve(_make_bounded_model(maximize=maximize), SolverType.GLOP)

    run: Run = _single_solve_run(tracking)
    objective_value: float = run.data.metrics["objective_value"]
    dual_bound: float = run.data.metrics["dual_bound"]

    assert run.data.tags["optimization.objective_sense"] == expected_sense
    if maximize:
        assert objective_value <= dual_bound
    else:
        assert objective_value >= dual_bound


def test_different_mathopt_backends_have_distinct_engine_tags(
    tracking: Tracking,
) -> None:
    autolog(solvers=["mathopt"])
    _solve(_make_bounded_model(), SolverType.GLOP)
    _solve(_make_bounded_model(), SolverType.PDLP)

    solve_runs: list[Run] = tracking.solve_runs()
    engine_tags: list[str] = [
        run.data.tags["optimization.solver"] for run in solve_runs
    ]

    assert [run.data.tags["optimization.api"] for run in solve_runs] == [
        "mathopt",
        "mathopt",
    ]
    assert engine_tags == ["glop", "pdlp"]


def test_backend_without_branch_and_bound_nodes_marks_statistic_missing(
    tracking: Tracking,
) -> None:
    autolog(solvers=["mathopt"])
    _solve(_make_bounded_model(), SolverType.GLOP)

    run: Run = _single_solve_run(tracking)

    assert "node_count" not in run.data.metrics
    assert run.data.tags["optimization.missing.node_count"] == "unsupported"
