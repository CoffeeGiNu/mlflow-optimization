"""Acceptance tests for the MathOpt integration."""

import math
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from conftest import Tracking
from mlflow.entities import Run
from ortools.math_opt import model_pb2
from ortools.math_opt.python import mathopt
from ortools.math_opt.python.mathopt import solve
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
    autolog(libraries=["mathopt"])
    _solve(_make_bounded_model(), SolverType.GLOP)

    run: Run = _single_solve_run(tracking)

    assert run.data.tags["optimization.api"] == "mathopt"
    assert run.data.tags["optimization.solver"] == "glop"
    assert run.data.tags["optimization.termination"] == "optimal"
    assert run.data.tags["optimization.has_solution"] == "true"
    assert "objective_value" in run.data.metrics
    assert "dual_bound" in run.data.metrics


def test_optimal_mixed_integer_problem_is_recorded(tracking: Tracking) -> None:
    autolog(libraries=["mathopt"])
    _solve(_make_bounded_model(integer=True), SolverType.GSCIP)

    run: Run = _single_solve_run(tracking)

    assert run.data.tags["optimization.api"] == "mathopt"
    assert run.data.tags["optimization.solver"] == "scip"
    assert run.data.tags["optimization.termination"] == "optimal"
    assert run.data.tags["optimization.has_solution"] == "true"
    assert "objective_value" in run.data.metrics
    assert "dual_bound" in run.data.metrics


@pytest.mark.parametrize("solver_type", (SolverType.GSCIP, SolverType.CP_SAT))
def test_integer_feasibility_problem_has_no_objective_history(
    tracking: Tracking,
    solver_type: SolverType,
) -> None:
    model: Model = Model()
    variable: Variable = model.add_binary_variable()
    model.add_linear_constraint(variable >= 1.0)
    autolog(libraries=["mathopt"])

    _solve(model, solver_type)

    run: Run = _single_solve_run(tracking)
    assert run.data.tags["optimization.has_objective"] == "false"
    for metric_name in (
        "objective_value",
        "dual_bound",
        "absolute_gap",
        "relative_gap",
    ):
        assert metric_name not in run.data.metrics
        assert tracking.client.get_metric_history(run.info.run_id, metric_name) == []
        assert run.data.tags[f"optimization.missing.{metric_name}"] == "undefined"


def test_infeasible_problem_records_no_objective_value(tracking: Tracking) -> None:
    autolog(libraries=["mathopt"])
    _solve(_make_infeasible_model(), SolverType.GLOP)

    run: Run = _single_solve_run(tracking)

    assert run.data.tags["optimization.termination"] == "infeasible"
    assert run.data.tags["optimization.has_solution"] == "false"
    assert "objective_value" not in run.data.metrics
    assert run.data.tags["optimization.missing.objective_value"] == "no_solution"


def test_unbounded_problem_records_no_solution_and_unbounded_objective(
    tracking: Tracking,
) -> None:
    autolog(libraries=["mathopt"])
    _solve(_make_unbounded_model(), SolverType.GSCIP)

    run: Run = _single_solve_run(tracking)

    assert run.data.tags["optimization.termination"] == "unbounded"
    assert run.data.tags["optimization.has_solution"] == "false"
    assert "objective_value" not in run.data.metrics
    assert run.data.tags["optimization.missing.objective_value"] == "unbounded"
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
    autolog(libraries=["mathopt"])
    _solve(_make_bounded_model(maximize=maximize), SolverType.GLOP)

    run: Run = _single_solve_run(tracking)
    objective_value: float = run.data.metrics["objective_value"]
    dual_bound: float = run.data.metrics["dual_bound"]

    assert run.data.tags["optimization.objective_sense"] == expected_sense
    if maximize:
        assert objective_value <= dual_bound
    else:
        assert objective_value >= dual_bound


@pytest.mark.parametrize("maximize", (False, True))
def test_node_limit_records_nonzero_gap(tracking: Tracking, maximize: bool) -> None:
    autolog(libraries=["mathopt"])
    model: Model = Model()
    variables: list[Variable] = [
        model.add_binary_variable(name=f"value_{index}") for index in range(60)
    ]
    for constraint_index in range(5):
        model.add_linear_constraint(
            sum(
                ((index * 17 + constraint_index * 13) % 31 + 1) * variables[index]
                for index in range(60)
            )
            <= 180
        )
    expression: mathopt.LinearBase = cast(
        mathopt.LinearBase,
        sum((index * 11 % 37 + 1) * variables[index] for index in range(60)),
    )
    if maximize:
        model.maximize(expression)
    else:
        model.minimize(-expression)
    parameters: mathopt.SolveParameters = mathopt.SolveParameters(
        node_limit=1,
        presolve=mathopt.Emphasis.OFF,
        cuts=mathopt.Emphasis.OFF,
        heuristics=mathopt.Emphasis.OFF,
    )
    model_parameters: mathopt.ModelSolveParameters = mathopt.ModelSolveParameters(
        solution_hints=[
            mathopt.SolutionHint(
                variable_values={variable: 0.0 for variable in variables}
            )
        ]
    )

    result: mathopt.SolveResult = mathopt.solve(
        model, SolverType.GSCIP, params=parameters, model_params=model_parameters
    )

    assert result.termination.reason == mathopt.TerminationReason.FEASIBLE
    assert result.termination.limit == mathopt.Limit.NODE
    run: Run = _single_solve_run(tracking)
    objective: float = run.data.metrics["objective_value"]
    bound: float = run.data.metrics["dual_bound"]
    assert math.isfinite(objective) and math.isfinite(bound)
    assert objective == pytest.approx(result.objective_value())
    assert bound == pytest.approx(result.dual_bound())
    assert (1 if maximize else -1) * (bound - objective) > 0
    assert run.data.metrics["absolute_gap"] == pytest.approx(abs(objective - bound))


def test_different_mathopt_backends_have_distinct_engine_tags(
    tracking: Tracking,
) -> None:
    autolog(libraries=["mathopt"])
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
    autolog(libraries=["mathopt"])
    _solve(_make_bounded_model(), SolverType.GLOP)

    run: Run = _single_solve_run(tracking)

    assert "node_count" not in run.data.metrics
    assert run.data.tags["optimization.missing.node_count"] == "unsupported"


def test_disable_preserves_a_reassigned_solve_name(tracking: Tracking) -> None:
    global solve

    original_solve: Callable[..., object] = solve

    def user_solve() -> None:
        return None

    try:
        autolog(libraries=["mathopt"])
        solve = user_solve

        autolog(disable=True)

        assert solve is user_solve
    finally:
        autolog(disable=True)
        solve = original_solve


def _download(tracking: Tracking, run: Run, artifact_path: str) -> Path:
    """A local copy of one artifact of a run."""
    return Path(tracking.client.download_artifacts(run.info.run_id, artifact_path))


def test_solver_log_is_attached_and_user_message_callback_still_receives_lines(
    tracking: Tracking,
) -> None:
    autolog(libraries=["mathopt"], log_solver_log=True)
    user_lines: list[str] = []

    mathopt.solve(
        _make_bounded_model(integer=True),
        SolverType.GSCIP,
        msg_cb=user_lines.extend,
        params=mathopt.SolveParameters(enable_output=True),
    )

    run: Run = _single_solve_run(tracking)
    assert user_lines
    assert run.data.tags["optimization.artifacts.solver_log"] == "true"
    assert _download(tracking, run, "solver/log.txt").read_text().strip() != ""


def test_incremental_solver_log_is_attached(tracking: Tracking) -> None:
    autolog(libraries=["mathopt"], log_solver_log=True)
    incremental_solver: mathopt.IncrementalSolver = mathopt.IncrementalSolver(
        _make_bounded_model(integer=True), SolverType.GSCIP
    )

    incremental_solver.solve(params=mathopt.SolveParameters(enable_output=True))

    run: Run = _single_solve_run(tracking)
    assert run.data.tags["optimization.artifacts.solver_log"] == "true"
    assert _download(tracking, run, "solver/log.txt").read_text().strip() != ""


def test_solver_model_reloads_and_resolves_to_the_same_optimum(
    tracking: Tracking,
) -> None:
    autolog(libraries=["mathopt"], log_solver_model=True)
    model: Model = _make_bounded_model()
    original: mathopt.SolveResult = mathopt.solve(model, SolverType.GLOP)

    run: Run = _single_solve_run(tracking)
    assert run.data.tags["optimization.artifacts.solver_model"] == "true"
    model_proto: model_pb2.ModelProto = model_pb2.ModelProto()
    model_proto.ParseFromString(_download(tracking, run, "model/model.pb").read_bytes())
    reloaded: mathopt.SolveResult = mathopt.solve(
        Model.from_model_proto(model_proto), SolverType.GLOP
    )
    assert reloaded.objective_value() == pytest.approx(4.0)
    assert reloaded.objective_value() == pytest.approx(original.objective_value())


def test_infeasibility_diagnosis_is_unsupported(tracking: Tracking) -> None:
    autolog(libraries=["mathopt"], diagnose_infeasibility=True)
    _solve(_make_infeasible_model(), SolverType.GLOP)

    run: Run = _single_solve_run(tracking)

    assert run.data.tags["optimization.diagnostics.infeasibility"] == "unsupported"


def test_disabled_artifact_logging_ignores_diagnostic_flags(
    tracking: Tracking,
) -> None:
    autolog(
        libraries=["mathopt"],
        log_artifacts=False,
        log_solver_log=True,
        log_solver_model=True,
        diagnose_infeasibility=True,
    )
    _solve(_make_infeasible_model(), SolverType.GLOP)

    run: Run = _single_solve_run(tracking)

    assert tracking.client.list_artifacts(run.info.run_id) == []
    assert not [
        tag
        for tag in run.data.tags
        if tag.startswith(("optimization.artifacts.", "optimization.diagnostics."))
    ]
