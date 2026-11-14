from pathlib import Path

import gurobipy
import pytest
from mlflow.entities import Run

from conftest import Tracking
from mlflow_optimization import autolog


def _new_model() -> gurobipy.Model:
    model: gurobipy.Model = gurobipy.Model()
    model.Params.OutputFlag = 0
    return model


def _recorded_run(tracking: Tracking) -> Run:
    solve_runs: list[Run] = tracking.solve_runs()
    assert len(solve_runs) == 1
    solve_run: Run = solve_runs[0]
    assert solve_run.data.tags["optimization.api"] == "gurobipy"
    assert solve_run.data.tags["optimization.solver"] == "gurobi"
    return solve_run


def _set_time_limit_after_incumbent(
    model: gurobipy.Model, callback_location: int
) -> None:
    if callback_location == gurobipy.GRB.Callback.MIPSOL:
        model.cbSetParam("TimeLimit", 0.0)


def test_optimal_mip_is_recorded(tracking: Tracking) -> None:
    autolog(libraries=["gurobipy"])
    model: gurobipy.Model = _new_model()
    variable: gurobipy.Var = model.addVar(vtype=gurobipy.GRB.BINARY)
    model.addConstr(variable >= 1)
    model.setObjective(variable, gurobipy.GRB.MAXIMIZE)

    model.optimize()

    solve_run: Run = _recorded_run(tracking)
    assert solve_run.data.tags["optimization.termination"] == "optimal"
    assert solve_run.data.tags["optimization.has_solution"] == "true"
    assert "objective_value" in solve_run.data.metrics
    assert "dual_bound" in solve_run.data.metrics
    assert "optimization.missing.objective_value" not in solve_run.data.tags


def test_failed_call_after_success_does_not_record_stale_results(
    tracking: Tracking,
) -> None:
    autolog(libraries=["gurobipy"])
    model: gurobipy.Model = _new_model()
    variable: gurobipy.Var = model.addVar(vtype=gurobipy.GRB.BINARY)
    model.setObjective(variable, gurobipy.GRB.MAXIMIZE)
    model.optimize()

    with pytest.raises(TypeError):
        model.optimize(unexpected=True)

    runs: list[Run] = tracking.solve_runs()
    assert len(runs) == 2
    assert runs[0].data.metrics["objective_value"] == 1
    for metric_name in (
        "objective_value",
        "dual_bound",
        "solver_time_seconds",
        "node_count",
    ):
        assert metric_name not in runs[1].data.metrics


@pytest.mark.parametrize("sense", (gurobipy.GRB.MINIMIZE, gurobipy.GRB.MAXIMIZE))
def test_limited_mip_bound_has_correct_direction(
    tracking: Tracking, sense: int
) -> None:
    autolog(libraries=["gurobipy"])
    model: gurobipy.Model = _new_model()
    model.Params.Presolve = 0
    model.Params.NodeLimit = 1
    model.Params.Heuristics = 0
    model.Params.Cuts = 0
    variables: gurobipy.tupledict[int, gurobipy.Var] = model.addVars(
        60, vtype=gurobipy.GRB.BINARY
    )
    for constraint_index in range(5):
        model.addConstr(
            sum(
                ((index * 17 + constraint_index * 13) % 31 + 1) * variables[index]
                for index in range(60)
            )
            <= 180
        )
    model.setObjective(
        -sense * sum((index * 11 % 37 + 1) * variables[index] for index in range(60)),
        sense,
    )
    for variable in variables.values():
        variable.Start = 0

    model.optimize()

    assert model.Status == gurobipy.GRB.NODE_LIMIT
    run: Run = _recorded_run(tracking)
    objective: float = run.data.metrics["objective_value"]
    bound: float = run.data.metrics["dual_bound"]
    assert (objective - bound) * sense > 0
    assert run.data.metrics["absolute_gap"] == pytest.approx(abs(objective - bound))


def test_infeasible_mip_has_no_objective_value(tracking: Tracking) -> None:
    autolog(libraries=["gurobipy"])
    model: gurobipy.Model = _new_model()
    variable: gurobipy.Var = model.addVar(vtype=gurobipy.GRB.BINARY)
    model.addConstr(variable >= 1)
    model.addConstr(variable <= 0)
    model.setObjective(variable)

    model.optimize()

    solve_run: Run = _recorded_run(tracking)
    assert solve_run.data.tags["optimization.termination"] == "infeasible"
    assert solve_run.data.tags["optimization.has_solution"] == "false"
    assert "objective_value" not in solve_run.data.metrics
    assert solve_run.data.tags["optimization.missing.objective_value"] == "no_solution"


def test_unbounded_status_is_recorded(tracking: Tracking) -> None:
    autolog(libraries=["gurobipy"])
    model: gurobipy.Model = _new_model()
    variable: gurobipy.Var = model.addVar(lb=-gurobipy.GRB.INFINITY)
    model.setObjective(variable, gurobipy.GRB.MAXIMIZE)

    model.optimize()

    solve_run: Run = _recorded_run(tracking)
    assert model.Status == gurobipy.GRB.UNBOUNDED
    assert solve_run.data.tags["optimization.termination"] == "unbounded"
    assert solve_run.data.tags["optimization.raw_status"].startswith("UNBOUNDED (")
    assert solve_run.data.tags["optimization.has_solution"] == "false"
    assert "objective_value" not in solve_run.data.metrics
    assert solve_run.data.tags["optimization.missing.objective_value"] == "no_solution"


def test_infeasible_or_unbounded_status_is_not_resolved_by_guessing(
    tracking: Tracking,
) -> None:
    autolog(libraries=["gurobipy"])
    model: gurobipy.Model = _new_model()
    variable: gurobipy.Var = model.addVar(lb=-gurobipy.GRB.INFINITY)
    model.addConstr(variable >= 1)
    model.addConstr(variable <= 0)
    model.setObjective(variable, gurobipy.GRB.MAXIMIZE)

    model.optimize()

    if model.Status != gurobipy.GRB.INF_OR_UNBD:
        pytest.skip(
            f"Gurobi returned status {model.Status}; this instance did not produce INF_OR_UNBD."
        )
    solve_run: Run = _recorded_run(tracking)
    assert solve_run.data.tags["optimization.termination"] == "infeasible_or_unbounded"
    assert solve_run.data.tags["optimization.raw_status"].startswith("INF_OR_UNBD (")
    assert solve_run.data.tags["optimization.has_solution"] == "false"
    assert "objective_value" not in solve_run.data.metrics
    assert solve_run.data.tags["optimization.missing.objective_value"] == "no_solution"


def test_time_limit_without_solution_does_not_invent_objective(
    tracking: Tracking,
) -> None:
    autolog(libraries=["gurobipy"])
    model: gurobipy.Model = _new_model()
    model.Params.TimeLimit = 0.0
    variable: gurobipy.Var = model.addVar(vtype=gurobipy.GRB.BINARY)
    model.addConstr(variable >= 1)
    model.setObjective(variable, gurobipy.GRB.MAXIMIZE)

    model.optimize()

    if model.Status != gurobipy.GRB.TIME_LIMIT or model.SolCount != 0:
        pytest.skip(
            f"Gurobi returned status {model.Status} with {model.SolCount} solutions at a zero-second limit."
        )
    solve_run: Run = _recorded_run(tracking)
    assert solve_run.data.tags["optimization.termination"] == "time_limit"
    assert solve_run.data.tags["optimization.has_solution"] == "false"
    assert "objective_value" not in solve_run.data.metrics
    assert solve_run.data.tags["optimization.missing.objective_value"] == "no_solution"


def test_time_limit_with_incumbent_keeps_objective(tracking: Tracking) -> None:
    autolog(libraries=["gurobipy"])
    model: gurobipy.Model = _new_model()
    model.Params.TimeLimit = 10.0
    variables: gurobipy.tupledict[int, gurobipy.Var] = model.addVars(
        4, vtype=gurobipy.GRB.BINARY
    )
    model.setObjective(
        sum((index + 1) * variables[index] for index in range(4)),
        gurobipy.GRB.MAXIMIZE,
    )
    model.addConstr(sum(variables.values()) <= 2)

    model.optimize(_set_time_limit_after_incumbent)

    if model.Status != gurobipy.GRB.TIME_LIMIT or model.SolCount == 0:
        pytest.skip(
            f"Gurobi returned status {model.Status} with {model.SolCount} solutions after the callback limit."
        )
    solve_run: Run = _recorded_run(tracking)
    assert solve_run.data.tags["optimization.termination"] == "time_limit"
    assert solve_run.data.tags["optimization.has_solution"] == "true"
    assert "objective_value" in solve_run.data.metrics
    assert "optimization.missing.objective_value" not in solve_run.data.tags


@pytest.mark.parametrize(
    ("sense", "sense_name"),
    (
        (gurobipy.GRB.MINIMIZE, "minimize"),
        (gurobipy.GRB.MAXIMIZE, "maximize"),
    ),
)
def test_objective_and_bound_follow_sense(
    tracking: Tracking, sense: int, sense_name: str
) -> None:
    autolog(libraries=["gurobipy"])
    model: gurobipy.Model = _new_model()
    first_variable: gurobipy.Var = model.addVar(vtype=gurobipy.GRB.BINARY)
    second_variable: gurobipy.Var = model.addVar(vtype=gurobipy.GRB.BINARY)
    model.addConstr(first_variable + second_variable == 1)
    model.setObjective(first_variable + 2 * second_variable, sense)

    model.optimize()

    solve_run: Run = _recorded_run(tracking)
    objective_value: float = solve_run.data.metrics["objective_value"]
    dual_bound: float = solve_run.data.metrics["dual_bound"]
    assert solve_run.data.tags["optimization.objective_sense"] == sense_name
    assert solve_run.data.tags["optimization.termination"] == "optimal"
    if sense == gurobipy.GRB.MINIMIZE:
        assert objective_value >= dual_bound
    else:
        assert objective_value <= dual_bound


def test_explicit_parameter_is_recorded_without_paths_or_secrets(
    tracking: Tracking, tmp_path: Path
) -> None:
    autolog(libraries=["gurobipy"])
    log_file: Path = tmp_path / "gurobi.log"
    secret: str = "acceptance-test-secret"
    with gurobipy.Env(empty=True) as environment:
        environment.setParam("OutputFlag", 0)
        environment.setParam("CSAPISecret", secret)
        environment.start()
        with gurobipy.Model(env=environment) as model:
            model.Params.MIPGap = 0.2
            model.Params.LogFile = str(log_file)
            variable: gurobipy.Var = model.addVar(vtype=gurobipy.GRB.BINARY)
            model.setObjective(variable, gurobipy.GRB.MAXIMIZE)

            model.optimize()

            solve_run: Run = _recorded_run(tracking)
    assert solve_run.data.params["solver_param.MIPGap"] == "0.2"
    assert "solver_param.LogFile" not in solve_run.data.params
    assert "solver_param.CSAPISecret" not in solve_run.data.params
    assert str(log_file) not in solve_run.data.params.values()
    assert secret not in solve_run.data.params.values()
