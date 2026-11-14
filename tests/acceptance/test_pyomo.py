"""Acceptance tests for Pyomo solver observations."""

import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mlflow.entities import Run
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    Objective,
    RangeSet,
    SolverFactory,
    Var,
    maximize,
    minimize,
)
from pyomo.opt import SolverResults, TerminationCondition
from pyomo.opt.results.solution import SolutionStatus

from mlflow_optimization import autolog


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


def _stub_shell_solver(
    solver: Any,
    monkeypatch: pytest.MonkeyPatch,
    result: SolverResults,
    error: Exception | None = None,
) -> None:
    def available(exception_flag: bool = True) -> bool:
        return True

    def presolve(*arguments: Any, **keywords: Any) -> None:
        return None

    def apply_solver() -> Any:
        if error is not None:
            raise error
        return SimpleNamespace(rc=0)

    def postsolve() -> SolverResults:
        return result

    monkeypatch.setattr(solver, "available", available)
    monkeypatch.setattr(solver, "_presolve", presolve)
    monkeypatch.setattr(solver, "_apply_solver", apply_solver)
    monkeypatch.setattr(solver, "_postsolve", postsolve)


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


@pytest.mark.parametrize("maximize_sense", (False, True))
def test_highs_node_limit_records_nonzero_gap(
    tracking: Any, maximize_sense: bool
) -> None:
    autolog(solvers=["pyomo"])
    solver: Any = _solver_or_skip("appsi_highs")
    solver.highs_options["presolve"] = "off"
    solver.highs_options["mip_max_nodes"] = 1
    solver.config.warmstart = True
    model: Any = ConcreteModel()
    model.indices = RangeSet(0, 59)
    model.variables = Var(model.indices, domain=Binary, initialize=0)
    model.constraints = Constraint(
        range(5),
        rule=lambda model, constraint_index: (
            sum(
                ((index * 17 + constraint_index * 13) % 31 + 1) * model.variables[index]
                for index in model.indices
            )
            <= 180
        ),
    )
    direction: int = 1 if maximize_sense else -1
    model.objective = Objective(
        expr=direction
        * sum(
            (index * 11 % 37 + 1) * model.variables[index] for index in model.indices
        ),
        sense=maximize if maximize_sense else minimize,
    )

    result: Any = solver.solve(model, load_solutions=False)

    assert str(result.solver.termination_condition) == "maxIterations"
    run: Run = _single_solve_run(tracking)
    objective: float = run.data.metrics["objective_value"]
    bound: float = run.data.metrics["dual_bound"]
    assert math.isfinite(objective) and math.isfinite(bound)
    assert objective == pytest.approx(
        result.problem.lower_bound if maximize_sense else result.problem.upper_bound
    )
    assert bound == pytest.approx(
        result.problem.upper_bound if maximize_sense else result.problem.lower_bound
    )
    assert direction * (bound - objective) > 0
    assert run.data.metrics["absolute_gap"] == pytest.approx(abs(objective - bound))


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


def test_appsi_gurobi_records_native_solve_time(tracking: Any) -> None:
    autolog(solvers=["pyomo"])
    solver: Any = _solver_or_skip("appsi_gurobi")
    model: Any = _bounded_model(minimize)

    solver.solve(model)

    run: Run = _single_solve_run(tracking)
    assert run.data.metrics["solver_time_seconds"] == solver._solver_model.Runtime


def test_appsi_highs_replaced_native_solver_records_its_own_time(tracking: Any) -> None:
    autolog(solvers=["pyomo"])
    solver: Any = _solver_or_skip("appsi_highs")
    first_model: Any = _bounded_model(minimize)
    second_model: Any = _bounded_model(maximize)

    solver.solve(first_model)
    first_native_solver: Any = solver._solver_model
    solver.solve(second_model)

    assert solver._solver_model is not first_native_solver
    runs: list[Run] = tracking.solve_runs()
    assert len(runs) == 2
    assert (
        runs[1].data.metrics["solver_time_seconds"] == solver._solver_model.getRunTime()
    )


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


def test_contrib_gurobi_persistent_imported_after_autolog_is_observed(
    tracking: Any,
) -> None:
    repository_root: Path = Path(__file__).resolve().parents[2]
    environment: dict[str, str] = os.environ.copy()
    previous_python_path: str = environment.get("PYTHONPATH", "")
    python_path_entries: list[str] = [str(repository_root / "src")]
    if previous_python_path:
        python_path_entries.append(previous_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_path_entries)
    subprocess_script: str = """
import sys
from typing import Any

import mlflow

mlflow.set_tracking_uri(sys.argv[1])
mlflow.set_experiment("late-pyomo-solver-test")
from mlflow_optimization import autolog

autolog(solvers=["pyomo"])

from pyomo.contrib.solver.solvers.gurobi.gurobi_persistent import GurobiPersistent
import pyomo.environ as pyomo

model: Any = pyomo.ConcreteModel()
model.value = pyomo.Var(domain=pyomo.Binary)
model.objective = pyomo.Objective(expr=model.value, sense=pyomo.minimize)
solver: Any = GurobiPersistent()
solver.solve(model)
autolog(disable=True)
"""
    process: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            subprocess_script,
            tracking.client.tracking_uri,
        ],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert process.returncode == 0, process.stdout + process.stderr

    runs: list[Run] = tracking.solve_runs()
    assert len(runs) == 1
    assert runs[0].data.tags["optimization.api"] == "pyomo"


def test_gurobi_direct_solution_limit_records_real_incumbent(tracking: Any) -> None:
    autolog(solvers=["pyomo"])
    solver: Any = _solver_or_skip("gurobi_direct")
    solver.options["SolutionLimit"] = 1
    solver.options["Presolve"] = 0
    solver.options["Cuts"] = 0
    solver.options["OutputFlag"] = 0
    weights: tuple[int, ...] = tuple(
        (37 * index * index + 11 * index) % 97 + 23 for index in range(40)
    )
    values: tuple[int, ...] = tuple(
        (53 * index * index + 7 * index) % 89 + 31 for index in range(40)
    )
    capacity: int = sum(weights) // 3
    model: Any = ConcreteModel()
    model.indices = RangeSet(0, 39)
    model.variables = Var(model.indices, domain=Binary)
    model.capacity = Constraint(
        expr=sum(
            weights[index] * model.variables[index] for index in model.indices
        )
        <= capacity
    )
    model.objective = Objective(
        expr=sum(
            values[index] * model.variables[index] for index in model.indices
        ),
        sense=maximize,
    )

    result: Any = solver.solve(model, load_solutions=False)

    run: Run = _single_solve_run(tracking)
    assert str(result.solution[0].status) == "stoppedByLimit"
    assert run.data.tags["optimization.has_solution"] == "true"
    assert run.data.metrics["objective_value"] == pytest.approx(1134)


def test_shell_gurobi_credentials_are_not_recorded_when_solve_raises(
    tracking: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autolog(solvers=["pyomo"])
    solver: Any = SolverFactory("gurobi")
    wls_secret: str = "dummy-wls-secret"
    server_password: str = "dummy-server-password"
    solver.options["WLSSecret"] = wls_secret
    solver.options["ServerPassword"] = server_password
    solver.options["lIcEnSeId"] = 42
    error: RuntimeError = RuntimeError("forced solver failure")
    _stub_shell_solver(solver, monkeypatch, SolverResults(), error)

    with pytest.raises(RuntimeError, match="forced solver failure"):
        solver.solve()

    run: Run = _single_solve_run(tracking)
    parameters: dict[str, str] = run.data.params
    tags: dict[str, str] = run.data.tags
    parameter_names: set[str] = {name.casefold() for name in parameters}
    credential_names: set[str] = {
        "solver_param.wlssecret",
        "solver_param.serverpassword",
    }
    secret_values: set[str] = {wls_secret, server_password}
    assert tags["optimization.solver"] == "unknown"
    assert tags["optimization.termination"] == "error"
    assert not credential_names.intersection(parameter_names)
    assert "solver_param.licenseid" not in parameter_names
    assert not secret_values.intersection(parameters.values())


@pytest.mark.parametrize(
    (
        "solution_status",
        "termination_condition",
        "expected_has_solution",
        "expected_termination",
    ),
    (
        (
            SolutionStatus.infeasible,
            TerminationCondition.infeasible,
            "false",
            "infeasible",
        ),
        (
            SolutionStatus.error,
            TerminationCondition.error,
            "false",
            "error",
        ),
        (
            SolutionStatus.unknown,
            TerminationCondition.maxIterations,
            "unknown",
            "iteration_limit",
        ),
        (
            SolutionStatus.stoppedByLimit,
            TerminationCondition.maxIterations,
            "unknown",
            "iteration_limit",
        ),
        (
            SolutionStatus.feasible,
            TerminationCondition.maxIterations,
            "true",
            "iteration_limit",
        ),
    ),
)
def test_legacy_solution_entry_status_controls_feasibility(
    tracking: Any,
    monkeypatch: pytest.MonkeyPatch,
    solution_status: Any,
    termination_condition: Any,
    expected_has_solution: str,
    expected_termination: str,
) -> None:
    autolog(solvers=["pyomo"])
    solver: Any = SolverFactory("gurobi")
    result: SolverResults = SolverResults()
    solver_information: Any = result.solver.add()
    solver_information.termination_condition = termination_condition
    solution: Any = result.solution.add()
    solution.status = solution_status
    _stub_shell_solver(solver, monkeypatch, result)

    solver.solve()

    run: Run = _single_solve_run(tracking)
    tags: dict[str, str] = run.data.tags
    assert tags["optimization.termination"] == expected_termination
    assert tags["optimization.has_solution"] == expected_has_solution
