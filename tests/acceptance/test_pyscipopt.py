"""PySCIPOpt acceptance tests using the real SCIP solver."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
from conftest import Tracking
from mlflow.entities import FileInfo, Run
from pyscipopt import (
    SCIP_PARAMSETTING,  # pyright: ignore[reportPrivateImportUsage] -- Runtime re-export used by the tests.
    Model,  # pyright: ignore[reportAssignmentType] -- The test rebinds this export.
    quicksum,
)

from mlflow_optimization import autolog


def _single_solve_run(tracking: Tracking, termination: str, has_solution: bool) -> Run:
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


def _artifact_paths(
    tracking: Tracking, run: Run, directory: str | None = None
) -> set[str]:
    """Every file path attached to a run, found recursively."""
    paths: set[str] = set()
    for info in tracking.client.list_artifacts(run.info.run_id, directory):
        file_info: FileInfo = info
        if file_info.is_dir:
            paths |= _artifact_paths(tracking, run, file_info.path)
        else:
            paths.add(file_info.path)
    return paths


def _download(
    tracking: Tracking, run: Run, artifact_path: str, directory: Path
) -> Path:
    """A local copy of one artifact of a run."""
    return Path(
        tracking.client.download_artifacts(
            run.info.run_id, artifact_path, str(directory)
        )
    )


def _knapsack_model() -> Model:
    """A small integer program whose optimal objective is 11."""
    model: Model = Model()
    first: Any = model.addVar("first", vtype="I", ub=3)
    second: Any = model.addVar("second", vtype="I", ub=10)
    model.addCons(first + second <= 4)
    model.setObjective(3 * first + 2 * second, "maximize")
    return model


def _infeasible_model() -> Model:
    """A binary model with contradictory constraints."""
    model: Model = Model()
    variable: Any = model.addVar("decision", vtype="B")
    model.setObjective(variable, "maximize")
    model.addCons(variable <= 0, name="upper")
    model.addCons(variable >= 1, name="lower")
    return model


def test_optimal_mip_records_solution_metrics_and_engine(tracking: Tracking) -> None:
    autolog(libraries=["pyscipopt"])
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
    for metric_name in (
        "objective_value",
        "dual_bound",
        "absolute_gap",
        "relative_gap",
    ):
        assert f"optimization.missing.{metric_name}" not in run.data.tags


def test_infeasible_mip_records_no_solution_reasons(tracking: Tracking) -> None:
    autolog(libraries=["pyscipopt"])
    model: Model = Model()
    variable: Any = model.addVar("decision", vtype="B")
    model.setObjective(variable, "maximize")
    model.addCons(variable <= 0)
    model.addCons(variable >= 1)

    model.optimize()

    run: Run = _single_solve_run(tracking, "infeasible", False)
    assert run.data.tags["optimization.has_objective"] == "true"
    for metric_name in ("objective_value", "absolute_gap", "relative_gap"):
        _assert_missing_reason(run, metric_name, "no_solution")


def test_unbounded_model_records_unbounded_termination(tracking: Tracking) -> None:
    autolog(libraries=["pyscipopt"])
    model: Model = Model()
    variable: Any = model.addVar("decision", vtype="C")
    model.setObjective(variable, "maximize")
    model.addCons(variable >= 0)

    model.optimize()

    run: Run = _single_solve_run(tracking, "unbounded", False)
    assert run.data.tags["optimization.has_objective"] == "true"
    _assert_missing_reason(run, "objective_value", "unbounded")
    assert "pyscipopt.gap" not in run.data.metrics


def test_time_limit_without_solution_records_missing_objective(
    tracking: Tracking,
) -> None:
    autolog(libraries=["pyscipopt"])
    model: Model = Model()
    variable: Any = model.addVar("decision", vtype="B")
    model.setObjective(variable, "maximize")
    model.setParam("limits/time", 0.0)

    model.optimize()

    run: Run = _single_solve_run(tracking, "time_limit", False)
    assert run.data.tags["optimization.has_objective"] == "true"
    for metric_name in ("objective_value", "absolute_gap", "relative_gap"):
        _assert_missing_reason(run, metric_name, "no_solution")


def test_time_limit_with_solution_keeps_objective(tracking: Tracking) -> None:
    autolog(libraries=["pyscipopt"])
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
    autolog(libraries=["pyscipopt"])
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


@pytest.mark.parametrize("maximize", (False, True))
def test_node_limit_records_nonzero_gap(tracking: Tracking, maximize: bool) -> None:
    autolog(libraries=["pyscipopt"])
    model: Model = Model()
    model.hideOutput()
    model.setPresolve(SCIP_PARAMSETTING.OFF)
    model.setHeuristics(SCIP_PARAMSETTING.OFF)
    model.setSeparating(SCIP_PARAMSETTING.OFF)
    model.setLongintParam("limits/nodes", 1)
    variables: list[Any] = [model.addVar(vtype="B") for index in range(60)]
    for constraint_index in range(5):
        model.addCons(
            quicksum(
                ((index * 17 + constraint_index * 13) % 31 + 1) * variables[index]
                for index in range(60)
            )
            <= 180
        )
    direction: int = 1 if maximize else -1
    model.setObjective(
        direction
        * quicksum((index * 11 % 37 + 1) * variables[index] for index in range(60)),
        "maximize" if maximize else "minimize",
    )
    solution: Any = model.createSol()
    for variable in variables:
        model.setSolVal(solution, variable, 0)
    assert model.addSol(solution)

    model.optimize()

    assert str(model.getStatus()) == "nodelimit"
    run: Run = _single_solve_run(tracking, "node_limit", True)
    objective: float = run.data.metrics["objective_value"]
    bound: float = run.data.metrics["dual_bound"]
    assert math.isfinite(objective) and math.isfinite(bound)
    assert objective == pytest.approx(model.getObjVal())
    assert bound == pytest.approx(model.getDualbound())
    assert direction * (bound - objective) > 0
    assert run.data.metrics["absolute_gap"] == pytest.approx(abs(objective - bound))


def test_explicit_solver_parameter_is_recorded(tracking: Tracking) -> None:
    autolog(libraries=["pyscipopt"])
    model: Model = Model()
    variable: Any = model.addVar("decision", vtype="B")
    model.setObjective(variable, "maximize")
    model.setParam("limits/time", 5.0)

    model.optimize()

    run: Run = _single_solve_run(tracking, "optimal", True)
    assert float(run.data.params["solver_param.limits/time"]) == 5.0


def test_free_transform_starts_new_solver_counters(tracking: Tracking) -> None:
    autolog(libraries=["pyscipopt"])
    model: Model = Model()
    model.hideOutput()
    variable: Any = model.addVar("decision", vtype="B")
    model.setObjective(variable, "maximize")

    model.optimize()
    first_solutions_found: int = model.getNSolsFound()
    model.freeTransform()
    model.optimize()

    runs: list[Run] = tracking.solve_runs()
    assert len(runs) == 2
    assert first_solutions_found > 0
    assert model.getNSolsFound() > 0
    assert runs[1].data.metrics["solver_time_seconds"] == model.getSolvingTime()
    assert runs[1].data.metrics["pyscipopt.solutions_found"] == model.getNSolsFound()
    assert "pyscipopt.total_time_seconds" not in runs[1].data.metrics


def test_unhashable_model_still_solves_and_records(tracking: Tracking) -> None:
    autolog(libraries=["pyscipopt"])

    class UnhashableModel(Model):
        __hash__: None = None

    model: Model = UnhashableModel()
    model.hideOutput()
    variable: Any = model.addVar("decision", vtype="B")
    model.setObjective(variable, "maximize")

    model.optimize()
    model.optimize()

    runs: list[Run] = tracking.solve_runs()
    assert len(runs) == 2
    for run in runs:
        assert run.data.tags["optimization.termination"] == "optimal"
        assert run.data.metrics["objective_value"] == 1
        assert "solver_time_seconds" not in run.data.metrics


def test_model_imported_before_autolog_is_observed(tracking: Tracking) -> None:
    autolog(libraries=["pyscipopt"])
    model: Model = Model()
    variable: Any = model.addVar("decision", vtype="B")
    model.setObjective(variable, "maximize")

    model.optimize()

    _single_solve_run(tracking, "optimal", True)


def test_disabled_autolog_does_not_record_solves(tracking: Tracking) -> None:
    autolog(libraries=["pyscipopt"])
    autolog(disable=True)
    model: Model = Model()
    variable: Any = model.addVar("decision", vtype="B")
    model.setObjective(variable, "maximize")

    model.optimize()

    assert tracking.solve_runs() == []


def test_disable_preserves_a_user_model_subclass_binding(tracking: Tracking) -> None:
    global Model

    original_model: type[Any] = Model
    try:
        autolog(libraries=["pyscipopt"])

        model_base: Any = globals()["Model"]

        class Model(model_base):
            """A user-defined PySCIPOpt model."""

        user_model: type[Any] = Model
        autolog(disable=True)

        model: Any = Model()

        assert isinstance(model, user_model)
    finally:
        autolog(disable=True)
        Model = original_model


def test_infeasible_model_attaches_log_model_and_iis(tracking: Tracking) -> None:
    autolog(
        libraries=["pyscipopt"],
        log_solver_log=True,
        log_solver_model=True,
        diagnose_infeasibility=True,
    )
    model: Model = _infeasible_model()
    model.hideOutput()

    model.optimize()

    run: Run = _single_solve_run(tracking, "infeasible", False)
    assert {"solver/log.txt", "model/model.cip", "diagnostics/iis.cip"} <= (
        _artifact_paths(tracking, run)
    )
    tags: dict[str, str] = run.data.tags
    assert tags["optimization.artifacts.solver_log"] == "true"
    assert tags["optimization.artifacts.solver_model"] == "true"
    assert tags["optimization.diagnostics.infeasibility"] == "iis"
    assert tags["optimization.diagnostics.iis_minimal"] in ("true", "false")


def test_solver_log_artifact_is_not_empty(tracking: Tracking, tmp_path: Path) -> None:
    autolog(libraries=["pyscipopt"], log_solver_log=True)
    model: Model = _knapsack_model()
    model.hideOutput()

    model.optimize()

    run: Run = _single_solve_run(tracking, "optimal", True)
    log_path: Path = _download(tracking, run, "solver/log.txt", tmp_path)
    assert log_path.read_text(errors="replace").strip() != ""


def test_optimal_solve_records_presolved_problem_size(tracking: Tracking) -> None:
    autolog(libraries=["pyscipopt"])
    model: Model = _knapsack_model()
    model.hideOutput()

    model.optimize()

    run: Run = _single_solve_run(tracking, "optimal", True)
    assert int(run.data.params["problem.presolved_num_variables"]) >= 0
    assert int(run.data.params["problem.presolved_num_constraints"]) >= 0


def test_attached_model_reloads_and_resolves_to_the_same_optimum(
    tracking: Tracking, tmp_path: Path
) -> None:
    autolog(libraries=["pyscipopt"], log_solver_model=True)
    model: Model = _knapsack_model()
    model.hideOutput()

    model.optimize()

    run: Run = _single_solve_run(tracking, "optimal", True)
    model_path: Path = _download(tracking, run, "model/model.cip", tmp_path)
    reloaded: Model = Model()
    reloaded.hideOutput()
    reloaded.readProblem(str(model_path))
    reloaded.optimize()
    assert model.getObjVal() == pytest.approx(11)
    assert reloaded.getObjVal() == pytest.approx(model.getObjVal())


def test_user_logfile_keeps_prior_content_and_receives_the_solve_log(
    tracking: Tracking, tmp_path: Path
) -> None:
    autolog(libraries=["pyscipopt"], log_solver_log=True)
    logfile: Path = tmp_path / "user.log"
    logfile.write_text("previous content\n")
    model: Model = _knapsack_model()
    model.setLogfile(str(logfile))

    model.optimize()
    model.setLogfile(None)

    run: Run = _single_solve_run(tracking, "optimal", True)
    user_text: str = logfile.read_text(errors="replace")
    assert user_text.startswith("previous content\n")
    assert len(user_text) > len("previous content\n")
    assert "solver/log.txt" in _artifact_paths(tracking, run)
    assert run.data.tags["optimization.artifacts.solver_log"] == "true"


def test_hidden_output_stays_silent_while_the_log_is_attached(
    tracking: Tracking, capfd: pytest.CaptureFixture[str]
) -> None:
    autolog(libraries=["pyscipopt"], log_solver_log=True)
    model: Model = _knapsack_model()
    model.hideOutput()

    model.optimize()

    run: Run = _single_solve_run(tracking, "optimal", True)
    assert capfd.readouterr().out == ""
    assert "solver/log.txt" in _artifact_paths(tracking, run)


def test_log_artifacts_off_disables_every_automatic_artifact(
    tracking: Tracking,
) -> None:
    autolog(
        libraries=["pyscipopt"],
        log_artifacts=False,
        log_solver_log=True,
        log_solver_model=True,
        diagnose_infeasibility=True,
    )
    model: Model = _infeasible_model()
    model.hideOutput()

    model.optimize()

    run: Run = _single_solve_run(tracking, "infeasible", False)
    assert _artifact_paths(tracking, run) == set()
    for tag in run.data.tags:
        assert not tag.startswith("optimization.artifacts.")
        assert not tag.startswith("optimization.diagnostics.")
