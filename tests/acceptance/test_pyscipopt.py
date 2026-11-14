"""PySCIPOpt acceptance tests using the real SCIP solver."""

from __future__ import annotations

from typing import Any

import pytest
from mlflow.entities import Run
from pyscipopt import Model

from conftest import Tracking
from mlflow_optimization import autolog


def _single_solve_run(
    tracking: Tracking, termination: str, has_solution: bool
) -> Run:
    """The only solve run recorded by one acceptance scenario."""
    runs: list[Run] = tracking.solve_runs()
    assert len(runs) == 1
    run: Run = runs[0]
    tags: dict[str, str] = run.data.tags
    assert tags["optimization.api"] == "pyscipopt"
    assert tags["optimization.solver"] == "scip"
    assert tags["optimization.termination"] == termination
    assert tags["optimization.has_solution"] == str(has_solution).lower()
    return run


def _assert_missing_reason(run: Run, metric_name: str, reason: str) -> None:
    """An absent metric and its recorded reason."""
    assert metric_name not in run.data.metrics
    assert run.data.tags[f"optimization.missing.{metric_name}"] == reason


def test_optimal_mip_records_solution_metrics_and_engine(tracking: Tracking) -> None:
    autolog(solvers=["pyscipopt"])
    model: Model = Model()
    variable: Any = model.addVar("decision", vtype="B")
    model.setObjective(variable, "maximize")

    model.optimize()

    run: Run = _single_solve_run(tracking, "optimal", True)
    assert run.data.tags["optimization.has_objective"] == "true"
    assert run.data.tags["optimization.objective_sense"] == "maximize"
    assert "objective_value" in run.data.metrics
    assert "dual_bound" in run.data.metrics
    assert "absolute_gap" in run.data.metrics
    assert "relative_gap" in run.data.metrics
    metric_name: str
    for metric_name in (
        "objective_value",
        "dual_bound",
        "absolute_gap",
        "relative_gap",
    ):
        assert f"optimization.missing.{metric_name}" not in run.data.tags


def test_infeasible_mip_records_no_solution_reasons(tracking: Tracking) -> None:
    autolog(solvers=["pyscipopt"])
    model: Model = Model()
    variable: Any = model.addVar("decision", vtype="B")
    model.setObjective(variable, "maximize")
    model.addCons(variable <= 0)
    model.addCons(variable >= 1)

    model.optimize()

    run: Run = _single_solve_run(tracking, "infeasible", False)
    assert run.data.tags["optimization.has_objective"] == "true"
    metric_name: str
    for metric_name in ("objective_value", "absolute_gap", "relative_gap"):
        _assert_missing_reason(run, metric_name, "no_solution")


def test_unbounded_model_records_unbounded_termination(tracking: Tracking) -> None:
    autolog(solvers=["pyscipopt"])
    model: Model = Model()
    variable: Any = model.addVar("decision", vtype="C")
    model.setObjective(variable, "maximize")
    model.addCons(variable >= 0)

    model.optimize()

    run: Run = _single_solve_run(tracking, "unbounded", True)
    assert run.data.tags["optimization.has_objective"] == "true"
    assert "objective_value" in run.data.metrics


def test_time_limit_without_solution_records_missing_objective(
    tracking: Tracking,
) -> None:
    autolog(solvers=["pyscipopt"])
    model: Model = Model()
    variable: Any = model.addVar("decision", vtype="B")
    model.setObjective(variable, "maximize")
    model.setParam("limits/time", 0.0)

    model.optimize()

    run: Run = _single_solve_run(tracking, "time_limit", False)
    assert run.data.tags["optimization.has_objective"] == "true"
    metric_name: str
    for metric_name in ("objective_value", "absolute_gap", "relative_gap"):
        _assert_missing_reason(run, metric_name, "no_solution")


def test_time_limit_with_solution_keeps_objective(tracking: Tracking) -> None:
    autolog(solvers=["pyscipopt"])
    model: Model = Model()
    variable: Any = model.addVar("decision", vtype="B")
    model.setObjective(variable, "maximize")
    solution: Any = model.createSol()
    model.setSolVal(solution, variable, 1.0)
    assert model.addSol(solution)
    model.setParam("limits/time", 0.0)

    model.optimize()

    run: Run = _single_solve_run(tracking, "time_limit", True)
    assert run.data.tags["optimization.has_objective"] == "true"
    assert "objective_value" in run.data.metrics
    assert "optimization.missing.objective_value" not in run.data.tags


@pytest.mark.parametrize("sense", ("minimize", "maximize"))
def test_objective_bound_direction_matches_sense(
    tracking: Tracking, sense: str
) -> None:
    autolog(solvers=["pyscipopt"])
    model: Model = Model()
    variable: Any = model.addVar("decision", vtype="B")
    model.setObjective(variable, sense)

    model.optimize()

    run: Run = _single_solve_run(tracking, "optimal", True)
    assert run.data.tags["optimization.objective_sense"] == sense
    objective_value: float = run.data.metrics["objective_value"]
    dual_bound: float = run.data.metrics["dual_bound"]
    if sense == "minimize":
        assert objective_value >= dual_bound
    else:
        assert objective_value <= dual_bound


def test_explicit_solver_parameter_is_recorded(tracking: Tracking) -> None:
    autolog(solvers=["pyscipopt"])
    model: Model = Model()
    variable: Any = model.addVar("decision", vtype="B")
    model.setObjective(variable, "maximize")
    model.setParam("limits/time", 5.0)

    model.optimize()

    run: Run = _single_solve_run(tracking, "optimal", True)
    assert float(run.data.params["solver_param.limits/time"]) == 5.0


def test_model_imported_before_autolog_is_observed(tracking: Tracking) -> None:
    autolog(solvers=["pyscipopt"])
    model: Model = Model()
    variable: Any = model.addVar("decision", vtype="B")
    model.setObjective(variable, "maximize")

    model.optimize()

    _single_solve_run(tracking, "optimal", True)


def test_disabled_autolog_does_not_record_solves(tracking: Tracking) -> None:
    autolog(solvers=["pyscipopt"])
    autolog(disable=True)
    model: Model = Model()
    variable: Any = model.addVar("decision", vtype="B")
    model.setObjective(variable, "maximize")

    model.optimize()

    assert tracking.solve_runs() == []
