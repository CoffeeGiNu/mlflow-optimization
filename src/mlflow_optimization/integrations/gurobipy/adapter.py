"""Observation of Gurobi solve calls."""

from collections.abc import Callable
from functools import wraps
from math import inf
from typing import Any

import gurobipy

from mlflow_optimization.common.patching import Patch, receiver_of
from mlflow_optimization.features.autolog.types import SolveObservation
from mlflow_optimization.features.autolog.ports import ObservationError, SolveObserver, SolverIntegration
from mlflow_optimization.features.metadata.types import ParameterOrigin, ReportedMetadata, SolverParameter
from mlflow_optimization.features.metrics.types import MissingReason, ObjectiveSense, ReportedMetrics
from mlflow_optimization.features.outcomes.types import ReportedOutcome, Termination


_STATUS_TERMINATIONS: dict[int, tuple[str, Termination]] = {
    gurobipy.GRB.LOADED: ("LOADED", Termination.UNKNOWN),
    gurobipy.GRB.OPTIMAL: ("OPTIMAL", Termination.OPTIMAL),
    gurobipy.GRB.INFEASIBLE: ("INFEASIBLE", Termination.INFEASIBLE),
    gurobipy.GRB.INF_OR_UNBD: ("INF_OR_UNBD", Termination.INFEASIBLE_OR_UNBOUNDED),
    gurobipy.GRB.UNBOUNDED: ("UNBOUNDED", Termination.UNBOUNDED),
    gurobipy.GRB.CUTOFF: ("CUTOFF", Termination.OTHER_LIMIT),
    gurobipy.GRB.ITERATION_LIMIT: ("ITERATION_LIMIT", Termination.ITERATION_LIMIT),
    gurobipy.GRB.NODE_LIMIT: ("NODE_LIMIT", Termination.NODE_LIMIT),
    gurobipy.GRB.TIME_LIMIT: ("TIME_LIMIT", Termination.TIME_LIMIT),
    gurobipy.GRB.SOLUTION_LIMIT: ("SOLUTION_LIMIT", Termination.SOLUTION_LIMIT),
    gurobipy.GRB.INTERRUPTED: ("INTERRUPTED", Termination.INTERRUPTED),
    gurobipy.GRB.NUMERIC: ("NUMERIC", Termination.UNKNOWN),
    gurobipy.GRB.SUBOPTIMAL: ("SUBOPTIMAL", Termination.FEASIBLE),
    gurobipy.GRB.INPROGRESS: ("INPROGRESS", Termination.UNKNOWN),
    gurobipy.GRB.USER_OBJ_LIMIT: ("USER_OBJ_LIMIT", Termination.OTHER_LIMIT),
    gurobipy.GRB.WORK_LIMIT: ("WORK_LIMIT", Termination.OTHER_LIMIT),
    gurobipy.GRB.MEM_LIMIT: ("MEM_LIMIT", Termination.OTHER_LIMIT),
}


class GurobiAdapter(SolverIntegration):
    """The Gurobi solver integration."""

    api: str = "gurobipy"

    def __init__(self) -> None:
        self._patch: Patch = Patch(gurobipy.Model, "optimize")

    def install(self, observer: SolveObserver) -> None:
        def factory(original: Any) -> Callable[..., Any]:
            @wraps(original)
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                solve_observer: SolveObserver = observer
                receiver: Any = receiver_of(original, args, kwargs)
                if receiver is None:
                    return original(*args, **kwargs)

                def call() -> Any:
                    return original(*args, **kwargs)

                def extract(result: Any | None, error: BaseException | None) -> SolveObservation:
                    if not isinstance(receiver, gurobipy.Model):
                        raise ObservationError("Could not find the Gurobi model for this solve.") from error
                    model: gurobipy.Model = receiver
                    try:
                        return _extract(model, error)
                    except Exception as observation_error:
                        raise ObservationError("Could not build a Gurobi solve observation.") from observation_error

                return solve_observer.observe(self.api, call, extract)

            return wrapped

        self._patch.install(factory)

    def uninstall(self) -> None:
        self._patch.uninstall()


def _extract(model: gurobipy.Model, error: BaseException | None) -> SolveObservation:
    metadata: ReportedMetadata = _read_metadata(model)
    if error is not None:
        termination: Termination = (
            Termination.INTERRUPTED if isinstance(error, KeyboardInterrupt) else Termination.ERROR
        )
        raw_status: str = "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "ERROR"
        outcome: ReportedOutcome = ReportedOutcome(raw_status, termination, None)
        metrics: ReportedMetrics = ReportedMetrics()
        return SolveObservation(api="gurobipy", outcome=outcome, metrics=metrics, metadata=metadata)
    raw_status: str
    termination: Termination
    raw_status, termination = _read_status(model)
    solution_count: float | MissingReason = _read_solver_number(model, "SolCount")
    has_solution: bool | None
    if isinstance(solution_count, MissingReason):
        has_solution = None
    else:
        has_solution = solution_count > 0
    has_objective: bool | None = _read_has_objective(model)
    sense: ObjectiveSense | None = _read_sense(model)
    objective_value: float | MissingReason = _read_objective_value(model, has_objective, has_solution)
    dual_bound: float | MissingReason = _read_dual_bound(model, has_objective)
    solver_time_seconds: float | MissingReason = _read_solver_number(model, "Runtime")
    statistics: dict[str, float | MissingReason] = {
        "node_count": _read_solver_number(model, "NodeCount"),
        "simplex_iterations": _read_solver_number(model, "IterCount"),
        "barrier_iterations": _read_solver_number(model, "BarIterCount"),
    }
    native: dict[str, float] = {}
    if not isinstance(solution_count, MissingReason):
        native["solution_count"] = solution_count
    for attribute_name, metric_name in (("MIPGap", "mip_gap"), ("Work", "work")):
        metric_value: float | MissingReason = _read_solver_number(model, attribute_name)
        if not isinstance(metric_value, MissingReason):
            native[metric_name] = metric_value
    outcome: ReportedOutcome = ReportedOutcome(raw_status, termination, has_solution)
    metrics: ReportedMetrics = ReportedMetrics(
        has_objective=has_objective,
        sense=sense,
        objective_value=objective_value,
        dual_bound=dual_bound,
        solver_time_seconds=solver_time_seconds,
        statistics=statistics,
        native=native,
    )
    return SolveObservation(api="gurobipy", outcome=outcome, metrics=metrics, metadata=metadata)


def _read_metadata(model: gurobipy.Model) -> ReportedMetadata:
    problem: dict[str, int] = {}
    for attribute_name, problem_name in (
        ("NumVars", "num_variables"),
        ("NumConstrs", "num_constraints"),
        ("NumIntVars", "num_integer_variables"),
        ("NumQConstrs", "num_quadratic_constraints"),
        ("NumGenConstrs", "num_general_constraints"),
        ("NumSOS", "num_sos_constraints"),
        ("NumNZs", "num_nonzeros"),
    ):
        problem_value: int | None = _read_problem_size(model, attribute_name)
        if problem_value is not None:
            problem[problem_name] = problem_value
    api_version: str | None = _read_api_version()
    solver_version: str | None = _read_solver_version()
    parameters: tuple[SolverParameter, ...] = _read_parameters(model)
    return ReportedMetadata(
        solver="gurobi",
        api_version=api_version,
        solver_version=solver_version,
        problem=problem,
        parameters=parameters,
        wall_time_includes_problem_build=False,
    )


def _read_attribute(model: gurobipy.Model, attribute_name: str) -> tuple[bool, Any]:
    try:
        value: Any = getattr(model, attribute_name)
    except Exception:
        return False, None
    return True, value


def _read_status(model: gurobipy.Model) -> tuple[str, Termination]:
    readable: bool
    raw_value: Any
    readable, raw_value = _read_attribute(model, "Status")
    if not readable:
        return "UNKNOWN", Termination.UNKNOWN
    try:
        status_code: int = int(raw_value)
    except Exception:
        return "UNKNOWN", Termination.UNKNOWN
    status_name: str
    termination: Termination
    status_name, termination = _STATUS_TERMINATIONS.get(status_code, ("UNKNOWN", Termination.UNKNOWN))
    return f"{status_name} ({status_code})", termination


def _read_solver_number(model: gurobipy.Model, attribute_name: str) -> float | MissingReason:
    readable: bool
    raw_value: Any
    readable, raw_value = _read_attribute(model, attribute_name)
    if not readable:
        return MissingReason.FAILED
    try:
        value: float = float(raw_value)
        solver_infinity: float = float(gurobipy.GRB.INFINITY)
    except Exception:
        return MissingReason.FAILED
    if value == solver_infinity:
        return inf
    if value == -solver_infinity:
        return -inf
    return value


def _read_problem_size(model: gurobipy.Model, attribute_name: str) -> int | None:
    readable: bool
    raw_value: Any
    readable, raw_value = _read_attribute(model, attribute_name)
    if not readable:
        return None
    try:
        return int(raw_value)
    except Exception:
        return None


def _read_has_objective(model: gurobipy.Model) -> bool | None:
    objective_count: int | None = _read_problem_size(model, "NumObj")
    if objective_count is None:
        return None
    if objective_count <= 0:
        return False
    objective_indices: tuple[int | None, ...]
    if objective_count == 1:
        objective_indices = (None,)
    else:
        objective_indices = tuple(range(objective_count))
    objective_unavailable: bool = False
    for objective_index in objective_indices:
        try:
            objective: Any
            if objective_index is None:
                objective = model.getObjective()
            else:
                objective = model.getObjective(objective_index)
        except Exception:
            objective_unavailable = True
            continue
        objective_has_content: bool | None = _objective_expression_has_content(objective)
        if objective_has_content is True:
            return True
        if objective_has_content is None:
            objective_unavailable = True
    piecewise_objective: bool | None = _has_piecewise_linear_objective(model)
    if piecewise_objective is True:
        return True
    if piecewise_objective is None:
        objective_unavailable = True
    if objective_unavailable:
        return None
    return False


def _has_piecewise_linear_objective(model: gurobipy.Model) -> bool | None:
    try:
        variables: list[gurobipy.Var] = model.getVars()
    except Exception:
        return None
    objective_unavailable: bool = False
    for variable in variables:
        try:
            points: Any = model.getPWLObj(variable)
        except Exception:
            objective_unavailable = True
            continue
        if points:
            return True
    if objective_unavailable:
        return None
    return False


def _objective_expression_has_content(objective: Any) -> bool | None:
    quadratic_term_count: int | None
    linear_expression: Any
    if isinstance(objective, gurobipy.QuadExpr):
        try:
            quadratic_term_count = int(objective.size())
        except Exception:
            quadratic_term_count = None
        try:
            linear_expression = objective.getLinExpr()
        except Exception:
            linear_expression = None
    elif isinstance(objective, gurobipy.LinExpr):
        quadratic_term_count = 0
        linear_expression = objective
    elif isinstance(objective, gurobipy.NLExpr):
        return True
    else:
        return None
    linear_term_count: int | None
    if linear_expression is None:
        linear_term_count = None
        constant_value: float | None = None
    else:
        try:
            linear_term_count = int(linear_expression.size())
        except Exception:
            linear_term_count = None
        try:
            constant_value = float(linear_expression.getConstant())
        except Exception:
            constant_value = None
    if quadratic_term_count is not None and quadratic_term_count > 0:
        return True
    if linear_term_count is not None and linear_term_count > 0:
        return True
    if constant_value is not None and constant_value != 0.0:
        return True
    if quadratic_term_count is None or linear_term_count is None or constant_value is None:
        return None
    return False


def _read_sense(model: gurobipy.Model) -> ObjectiveSense | None:
    sense_value: int | None = _read_problem_size(model, "ModelSense")
    if sense_value == gurobipy.GRB.MINIMIZE:
        return ObjectiveSense.MINIMIZE
    if sense_value == gurobipy.GRB.MAXIMIZE:
        return ObjectiveSense.MAXIMIZE
    return None


def _read_objective_value(
    model: gurobipy.Model,
    has_objective: bool | None,
    has_solution: bool | None,
) -> float | MissingReason:
    if has_objective is False:
        return MissingReason.UNDEFINED
    if has_solution is False:
        return MissingReason.NO_SOLUTION
    return _read_solver_number(model, "ObjVal")


def _read_dual_bound(model: gurobipy.Model, has_objective: bool | None) -> float | MissingReason:
    if has_objective is False:
        return MissingReason.UNDEFINED
    return _read_solver_number(model, "ObjBound")


def _read_parameters(model: gurobipy.Model) -> tuple[SolverParameter, ...]:
    readable: bool
    parameter_set: Any
    readable, parameter_set = _read_attribute(model, "Params")
    if not readable:
        return ()
    parameter_names: list[str] = []
    try:
        available_parameter_names: list[str] = dir(parameter_set)
    except Exception:
        return ()
    for parameter_name in available_parameter_names:
        if parameter_name and parameter_name[0].isupper():
            parameter_names.append(parameter_name)
    parameters: list[SolverParameter] = []
    for parameter_name in parameter_names:
        try:
            parameter_info: Any = model.getParamInfo(parameter_name)
        except Exception:
            continue
        try:
            parameter_type: Any = parameter_info[1]
        except Exception:
            continue
        if parameter_type is str:
            continue
        try:
            current_parameter_value: Any = parameter_info[2]
        except Exception:
            continue
        try:
            default_parameter_value: Any = parameter_info[5]
        except Exception:
            continue
        current_value: int | float | str | bool | None = _plain_parameter_value(current_parameter_value)
        default_value: int | float | str | bool | None = _plain_parameter_value(default_parameter_value)
        if current_value is None or default_value is None or current_value == default_value:
            continue
        parameters.append(
            SolverParameter(
                name=parameter_name,
                value=current_value,
                origin=ParameterOrigin.EXPLICIT,
            )
        )
    return tuple(parameters)


def _plain_parameter_value(value: Any) -> int | float | str | bool | None:
    if isinstance(value, (int, float, str)):
        return value
    return None


def _read_api_version() -> str | None:
    try:
        version_value: Any = getattr(gurobipy, "__version__", None)
    except Exception:
        return None
    if isinstance(version_value, str):
        return version_value
    return None


def _read_solver_version() -> str | None:
    try:
        solver_module: Any = getattr(gurobipy, "gurobi", None)
    except Exception:
        return None
    if solver_module is None:
        return None
    try:
        version_function: Any = getattr(solver_module, "version", None)
    except Exception:
        return None
    if not callable(version_function):
        return None
    try:
        version: Any = version_function()
    except Exception:
        return None
    if not isinstance(version, tuple):
        return None
    version_parts: list[str] = []
    for version_part in version:
        if not isinstance(version_part, int):
            return None
        version_parts.append(str(version_part))
    return ".".join(version_parts)
