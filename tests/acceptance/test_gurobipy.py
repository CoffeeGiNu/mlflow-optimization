from collections.abc import Callable
from pathlib import Path

import gurobipy
import pytest
from conftest import Tracking
from mlflow.entities import Run

from mlflow_optimization import autolog


@pytest.mark.parametrize(
    ("has_user_callback", "user_wheres"),
    ((False, None), (True, None), (True, [gurobipy.GRB.Callback.MIPSOL])),
)
def test_progress_filters_only_its_own_callback(
    tracking: Tracking,
    monkeypatch: pytest.MonkeyPatch,
    has_user_callback: bool,
    user_wheres: list[int] | None,
) -> None:
    original_optimize = gurobipy.Model.optimize
    observed_wheres: list[list[int] | None] = []

    def inspect_optimize(
        model: gurobipy.Model,
        callback: Callable[[gurobipy.Model, int], None] | None = None,
        wheres: list[int] | None = None,
    ) -> None:
        assert callback is not None
        assert callback is not user_callback
        observed_wheres.append(wheres)
        original_optimize(model, callback, wheres=wheres)

    user_events: list[int] = []

    def user_callback(model: gurobipy.Model, where: int) -> None:
        user_events.append(where)

    with monkeypatch.context() as sdk:
        sdk.setattr(gurobipy.Model, "optimize", inspect_optimize)
        autolog(libraries=["gurobipy"])
        try:
            with _new_model() as model:
                variable: gurobipy.Var = model.addVar(vtype=gurobipy.GRB.BINARY)
                model.setObjective(variable, gurobipy.GRB.MAXIMIZE)
                if has_user_callback:
                    model.optimize(user_callback, wheres=user_wheres)
                else:
                    model.optimize()
        finally:
            autolog(disable=True)

    expected_wheres: list[int] | None = (
        user_wheres
        if has_user_callback
        else [gurobipy.GRB.Callback.MIP, gurobipy.GRB.Callback.MIPSOL]
    )
    assert observed_wheres == [expected_wheres]
    if has_user_callback:
        assert user_events
        if user_wheres is not None:
            assert set(user_events).issubset(user_wheres)
    _recorded_run(tracking)


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
        model.optimize(unexpected=True)  # pyright: ignore[reportCallIssue] -- Intentional invalid-call test.

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
        model.addConstr(  # pyright: ignore[reportCallIssue]
            sum(
                ((index * 17 + constraint_index * 13) % 31 + 1) * variables[index]
                for index in range(60)
            )
            <= 180  # pyright: ignore[reportArgumentType]
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
    assert solve_run.data.tags["optimization.missing.dual_bound"] == "undefined"


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
    assert solve_run.data.tags["optimization.missing.objective_value"] == "unbounded"


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
    model.addConstr(  # pyright: ignore[reportCallIssue,reportArgumentType]
        sum(variables.values()) <= 2  # pyright: ignore[reportArgumentType]
    )

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


def _artifact_paths(tracking: Tracking, run: Run, directory: str = "") -> set[str]:
    paths: set[str] = set()
    for artifact in tracking.client.list_artifacts(run.info.run_id, directory or None):
        if artifact.is_dir:
            paths |= _artifact_paths(tracking, run, artifact.path)
        else:
            paths.add(artifact.path)
    return paths


def _artifact_file(tracking: Tracking, run: Run, path: str, tmp_path: Path) -> Path:
    return Path(
        tracking.client.download_artifacts(run.info.run_id, path, str(tmp_path))
    )


def _infeasible_model() -> gurobipy.Model:
    model: gurobipy.Model = _new_model()
    variable: gurobipy.Var = model.addVar(vtype=gurobipy.GRB.BINARY)
    model.addConstr(variable >= 1)
    model.addConstr(variable <= 0)
    model.setObjective(variable)
    return model


def _linear_program() -> gurobipy.Model:
    model: gurobipy.Model = _new_model()
    first_variable: gurobipy.Var = model.addVar(name="x")
    second_variable: gurobipy.Var = model.addVar(name="y")
    model.addConstr(first_variable + second_variable <= 4)
    model.addConstr(first_variable + 3 * second_variable <= 6)
    model.setObjective(3 * first_variable + 2 * second_variable, gurobipy.GRB.MAXIMIZE)
    return model


def test_infeasible_solve_attaches_requested_artifacts(
    tracking: Tracking, tmp_path: Path
) -> None:
    autolog(
        libraries=["gurobipy"],
        log_solver_log=True,
        log_solver_model=True,
        diagnose_infeasibility=True,
    )

    _infeasible_model().optimize()

    run: Run = _recorded_run(tracking)
    assert {"solver/log.txt", "model/model.mps", "diagnostics/iis.ilp"} <= (
        _artifact_paths(tracking, run)
    )
    assert _artifact_file(tracking, run, "solver/log.txt", tmp_path).stat().st_size > 0
    tags: dict[str, str] = run.data.tags
    assert tags["optimization.artifacts.solver_log"] == "true"
    assert tags["optimization.artifacts.solver_model"] == "true"
    assert tags["optimization.diagnostics.infeasibility"] == "iis"
    assert "optimization.diagnostics.iis_minimal" in tags


def test_attached_model_resolves_to_the_same_optimum(
    tracking: Tracking, tmp_path: Path
) -> None:
    autolog(libraries=["gurobipy"], log_solver_model=True)
    model: gurobipy.Model = _linear_program()
    model.optimize()
    autolog(disable=True)

    run: Run = _recorded_run(tracking)
    model_file: Path = _artifact_file(tracking, run, "model/model.mps", tmp_path)
    with gurobipy.read(str(model_file)) as reloaded:
        reloaded.Params.OutputFlag = 0
        reloaded.optimize()
        assert reloaded.Status == gurobipy.GRB.OPTIMAL
        assert reloaded.ObjVal == pytest.approx(model.ObjVal)


def test_user_log_file_keeps_content_and_settings(
    tracking: Tracking, tmp_path: Path
) -> None:
    user_log: Path = tmp_path / "user.log"
    user_log.write_text("previous content\n")
    autolog(libraries=["gurobipy"], log_solver_log=True)
    model: gurobipy.Model = _linear_program()
    model.Params.LogFile = str(user_log)
    model.Params.OutputFlag = 1
    model.Params.LogToConsole = 0

    model.optimize()

    assert model.Params.LogFile == str(user_log)
    assert model.Params.OutputFlag == 1
    assert model.Params.LogToConsole == 0
    user_text: str = user_log.read_text()
    assert user_text.startswith("previous content")
    run: Run = _recorded_run(tracking)
    attached: str = _artifact_file(
        tracking, run, "solver/log.txt", tmp_path
    ).read_text()
    assert attached.strip()
    assert "previous content" not in attached
    assert attached.strip() in user_text


def test_silenced_user_solve_stays_silent_and_is_restored(
    tracking: Tracking, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    autolog(libraries=["gurobipy"], log_solver_log=True)
    model: gurobipy.Model = _linear_program()
    model.Params.OutputFlag = 0
    model.Params.LogToConsole = 1
    model.Params.LogFile = ""

    model.optimize()

    assert capfd.readouterr().out == ""
    assert model.Params.OutputFlag == 0
    assert model.Params.LogToConsole == 1
    assert model.Params.LogFile == ""
    run: Run = _recorded_run(tracking)
    assert _artifact_file(tracking, run, "solver/log.txt", tmp_path).stat().st_size > 0
    assert run.data.tags["optimization.artifacts.solver_log"] == "true"


def test_raising_solve_still_attaches_log(tracking: Tracking, tmp_path: Path) -> None:
    autolog(libraries=["gurobipy"], log_solver_log=True)
    model: gurobipy.Model = _new_model()
    model.Params.NonConvex = 0
    variable: gurobipy.Var = model.addVar(lb=-1, ub=1)
    model.setObjective(variable * variable, gurobipy.GRB.MAXIMIZE)

    with pytest.raises(gurobipy.GurobiError):
        model.optimize()

    run: Run = _recorded_run(tracking)
    assert _artifact_file(tracking, run, "solver/log.txt", tmp_path).stat().st_size > 0
    assert run.data.tags["optimization.artifacts.solver_log"] == "true"


def test_log_artifacts_off_disables_all_automatic_artifacts(
    tracking: Tracking,
) -> None:
    autolog(
        libraries=["gurobipy"],
        log_artifacts=False,
        log_solver_log=True,
        log_solver_model=True,
        diagnose_infeasibility=True,
    )

    _infeasible_model().optimize()

    run: Run = _recorded_run(tracking)
    assert _artifact_paths(tracking, run) == set()
    assert not [
        tag
        for tag in run.data.tags
        if tag.startswith(("optimization.artifacts.", "optimization.diagnostics."))
    ]


def test_optimal_solve_gets_no_infeasibility_diagnosis(tracking: Tracking) -> None:
    autolog(libraries=["gurobipy"], diagnose_infeasibility=True)

    _linear_program().optimize()

    run: Run = _recorded_run(tracking)
    assert run.data.tags["optimization.termination"] == "optimal"
    assert "optimization.diagnostics.infeasibility" not in run.data.tags
