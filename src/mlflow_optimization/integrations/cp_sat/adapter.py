"""Observation support for OR-Tools CP-SAT."""

import math
from collections.abc import Callable
from functools import wraps
from typing import Any

import ortools
from ortools.sat import sat_parameters_pb2
from ortools.sat.python import cp_model

from mlflow_optimization.common.patching import Patch
from mlflow_optimization.features.autolog.models import SolveObservation
from mlflow_optimization.features.autolog.ports import SolveObserver
from mlflow_optimization.features.metadata.models import (
    ParameterOrigin,
    ReportedMetadata,
    SolverParameter,
)
from mlflow_optimization.features.metrics.models import (
    MissingReason,
    ObjectiveSense,
    ReportedMetrics,
)
from mlflow_optimization.features.outcomes.models import ReportedOutcome, Termination


class Integration:
    """The OR-Tools CP-SAT integration."""

    api: str = "cp_sat"
    _entry_points: tuple[str, ...] = (
        "solve",
        "Solve",
        "SolveWithSolutionCallback",
        "SearchForAllSolutions",
    )

    def __init__(self) -> None:
        self._observer: SolveObserver | None = None
        self._patches: list[Patch] = []

    def install(self, observer: SolveObserver) -> None:
        self._observer = observer
        if self._patches:
            return
        entry_point: str
        for entry_point in self._entry_points:
            if not hasattr(cp_model.CpSolver, entry_point):
                continue
            patch: Patch = Patch(cp_model.CpSolver, entry_point)
            patch.install(self._wrap)
            self._patches.append(patch)

    def uninstall(self) -> None:
        patch: Patch
        for patch in reversed(self._patches):
            patch.uninstall()
        self._patches.clear()
        self._observer = None

    def _wrap(self, original: Any) -> Callable[..., Any]:
        @wraps(original)
        def wrapped(
            solver: Any,
            *positional_arguments: Any,
            **keyword_arguments: Any,
        ) -> Any:
            observer: SolveObserver | None = self._observer
            if observer is None:
                return original(solver, *positional_arguments, **keyword_arguments)
            model: Any | None = (
                positional_arguments[0]
                if positional_arguments
                else keyword_arguments.get("model")
            )

            def call() -> Any:
                return original(solver, *positional_arguments, **keyword_arguments)

            def extract(
                result: Any | None,
                error: BaseException | None,
            ) -> SolveObservation:
                return self._extract(solver, model, result, error)

            return observer.observe(self.api, call, extract)

        return wrapped

    def _extract(
        self,
        solver: Any,
        model: Any | None,
        result: Any | None,
        error: BaseException | None,
    ) -> SolveObservation:
        outcome: ReportedOutcome = self._read_outcome(solver, model, result, error)
        metrics: ReportedMetrics = self._read_metrics(
            solver,
            model,
            outcome,
            error is not None,
        )
        metadata: ReportedMetadata = self._read_metadata(solver, model)
        return SolveObservation(self.api, outcome, metrics, metadata)

    def _read_outcome(
        self,
        solver: Any,
        model: Any | None,
        result: Any | None,
        error: BaseException | None,
    ) -> ReportedOutcome:
        if error is not None:
            termination: Termination = (
                Termination.INTERRUPTED
                if isinstance(error, KeyboardInterrupt)
                else Termination.ERROR
            )
            raw_status: str = f"{termination.value} ({type(error).__name__})"
            return ReportedOutcome(raw_status, termination, None)

        raw_status = self._read_raw_status(result)
        if result == cp_model.OPTIMAL:
            termination = Termination.OPTIMAL
            has_solution: bool | None = True
        elif result == cp_model.FEASIBLE:
            termination = Termination.FEASIBLE
            has_solution = True
        elif result == cp_model.INFEASIBLE:
            termination = Termination.INFEASIBLE
            has_solution = False
        elif result == cp_model.MODEL_INVALID:
            termination = Termination.ERROR
            has_solution = False
        elif result == cp_model.UNKNOWN:
            termination = Termination.UNKNOWN
            has_solution = self._read_solution_availability(solver, model)
        else:
            termination = Termination.UNKNOWN
            has_solution = self._read_solution_availability(solver, model)
        return ReportedOutcome(raw_status, termination, has_solution)

    def _read_raw_status(self, result: Any | None) -> str:
        status_names: dict[Any, str] = {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
            cp_model.UNKNOWN: "UNKNOWN",
        }
        status_name: str = status_names.get(result, "unknown")
        error: Exception
        if status_name == "unknown":
            try:
                status_name = str(result.name)
            except Exception:
                status_name = "unknown"
        try:
            status_code: int = int(result)
        except Exception:
            return status_name
        if status_name == "unknown":
            return f"unknown ({status_code})"
        return f"{status_name} ({status_code})"

    def _read_solution_availability(
        self,
        solver: Any,
        model: Any | None,
    ) -> bool | None:
        if model is None:
            return None
        error: Exception
        try:
            variable_count: int = len(model.proto.variables)
        except Exception:
            return None
        try:
            solution_count: int = len(solver.response_proto.solution)
        except Exception:
            return None
        if variable_count == 0:
            return None
        return solution_count == variable_count

    def _read_metrics(
        self,
        solver: Any,
        model: Any | None,
        outcome: ReportedOutcome,
        call_failed: bool,
    ) -> ReportedMetrics:
        has_objective: bool | None = self._read_has_objective(model)
        sense: ObjectiveSense | None = self._read_sense(model, has_objective)
        if has_objective is False:
            objective_value: float | MissingReason = MissingReason.UNSUPPORTED
            dual_bound: float | MissingReason = MissingReason.UNSUPPORTED
        elif call_failed:
            objective_value = MissingReason.FAILED
            dual_bound = MissingReason.FAILED
        elif has_objective is None:
            objective_value = MissingReason.UNDEFINED
            dual_bound = MissingReason.UNDEFINED
        else:
            objective_value = self._read_objective_value(solver, outcome.has_solution)
            dual_bound = self._read_dual_bound(solver, outcome)
        if call_failed:
            solver_time_seconds: float | MissingReason = MissingReason.FAILED
            native: dict[str, float] = {}
        else:
            solver_time_seconds = self._read_solver_time(solver)
            native = self._read_native_metrics(solver)
        return ReportedMetrics(
            has_objective=has_objective,
            sense=sense,
            objective_value=objective_value,
            dual_bound=dual_bound,
            solver_time_seconds=solver_time_seconds,
            stats={},
            native=native,
        )

    def _read_has_objective(self, model: Any | None) -> bool | None:
        if model is None:
            return None
        try:
            model_proto: Any = model.proto
        except Exception:
            return None
        has_floating_point_objective: bool | None = None
        try:
            has_floating_point_objective = bool(
                model_proto.has_floating_point_objective()
            )
        except Exception:
            pass
        if has_floating_point_objective is True:
            return True
        has_integer_objective: bool | None = None
        try:
            has_integer_objective = bool(model_proto.has_objective())
        except Exception:
            pass
        if has_integer_objective is True:
            return True
        if has_floating_point_objective is False and has_integer_objective is False:
            return False
        return None

    def _read_sense(
        self,
        model: Any | None,
        has_objective: bool | None,
    ) -> ObjectiveSense | None:
        if model is None or has_objective is not True:
            return None
        try:
            model_proto: Any = model.proto
        except Exception:
            return None
        has_floating_point_objective: bool | None = None
        try:
            has_floating_point_objective = bool(
                model_proto.has_floating_point_objective()
            )
        except Exception:
            pass
        if has_floating_point_objective is True:
            try:
                maximize: bool = bool(
                    model_proto.floating_point_objective.maximize
                )
            except Exception:
                return None
            return ObjectiveSense.MAXIMIZE if maximize else ObjectiveSense.MINIMIZE
        if has_floating_point_objective is not False:
            return None
        try:
            scaling_factor: float = float(model.proto.objective.scaling_factor)
        except Exception:
            return None
        if scaling_factor < 0:
            return ObjectiveSense.MAXIMIZE
        if scaling_factor > 0:
            return ObjectiveSense.MINIMIZE
        return None

    def _read_objective_value(
        self,
        solver: Any,
        has_solution: bool | None,
    ) -> float | MissingReason:
        if has_solution is False:
            return MissingReason.NO_SOLUTION
        if has_solution is None:
            return MissingReason.UNDEFINED
        return self._read_response_metric(solver, "objective_value")

    def _read_dual_bound(
        self,
        solver: Any,
        outcome: ReportedOutcome,
    ) -> float | MissingReason:
        if outcome.termination in (Termination.INFEASIBLE, Termination.ERROR):
            return MissingReason.UNDEFINED
        return self._read_response_metric(solver, "best_objective_bound")

    def _read_solver_time(self, solver: Any) -> float | MissingReason:
        return self._read_response_metric(solver, "wall_time")

    def _read_response_metric(
        self,
        solver: Any,
        name: str,
    ) -> float | MissingReason:
        error: Exception
        try:
            response: Any = solver.response_proto
            value: float = float(getattr(response, name))
        except Exception:
            return MissingReason.FAILED
        if value >= 2**63 - 1:
            return math.inf
        if value <= -(2**63):
            return -math.inf
        return value

    def _read_native_metrics(self, solver: Any) -> dict[str, float]:
        native_fields: tuple[str, ...] = (
            "user_time",
            "deterministic_time",
            "num_branches",
            "num_conflicts",
            "num_booleans",
            "num_integers",
            "num_fixed_booleans",
            "num_binary_propagations",
            "num_integer_propagations",
            "num_restarts",
            "num_lp_iterations",
            "gap_integral",
        )
        native: dict[str, float] = {}
        name: str
        for name in native_fields:
            try:
                response: Any = solver.response_proto
                value: float = float(getattr(response, name))
            except Exception:
                continue
            native[name] = value
        return native

    def _read_metadata(self, solver: Any, model: Any | None) -> ReportedMetadata:
        problem: dict[str, int] = self._read_problem_size(model)
        parameters: tuple[SolverParameter, ...] = self._read_parameters(solver)
        return ReportedMetadata(
            solver="cp_sat",
            api_version=ortools.__version__,
            solver_version=ortools.__version__,
            problem=problem,
            parameters=parameters,
            wall_time_includes_model_build=False,
        )

    def _read_problem_size(self, model: Any | None) -> dict[str, int]:
        if model is None:
            return {}
        try:
            model_proto: Any = model.proto
        except Exception:
            return {}
        problem: dict[str, int] = {}
        variables: Any | None = None
        error: Exception
        try:
            variables = model_proto.variables
            variable_count: int = len(variables)
            problem["num_variables"] = variable_count
            problem["num_integer_variables"] = variable_count
        except Exception:
            variables = ()
        try:
            problem["num_constraints"] = len(model_proto.constraints)
        except Exception:
            pass
        if variables is not None:
            boolean_variable_count: int = 0
            boolean_variable_count_known: bool = True
            variable: Any
            for variable in variables:
                try:
                    domain: list[int] = list(variable.domain)
                    if domain == [0, 1]:
                        boolean_variable_count += 1
                except Exception:
                    boolean_variable_count_known = False
            if boolean_variable_count_known:
                problem["num_boolean_variables"] = boolean_variable_count
        return problem

    def _read_parameters(self, solver: Any) -> tuple[SolverParameter, ...]:
        try:
            configured_parameters: Any = solver.parameters
            default_parameters: Any = sat_parameters_pb2.SatParameters()
        except Exception:
            return ()
        parameters: list[SolverParameter] = []
        field: Any
        for field in sat_parameters_pb2.SatParameters.DESCRIPTOR.fields:
            try:
                configured_value: Any = getattr(configured_parameters, field.name)
                default_value: Any = getattr(default_parameters, field.name)
                if field.is_repeated:
                    configured_value = list(configured_value)
                    default_value = list(default_value)
                if configured_value == default_value:
                    continue
                value: int | float | str | bool = self._parameter_value(configured_value, field)
                parameters.append(
                    SolverParameter(field.name, value, ParameterOrigin.EXPLICIT)
                )
            except Exception:
                continue
        return tuple(parameters)

    def _parameter_value(self, value: Any, field: Any) -> int | float | str | bool:
        if field.is_repeated:
            if field.enum_type is None:
                return str(list(value))
            enum_values: list[str | int] = []
            number: int
            for number in value:
                enum_value: Any | None = field.enum_type.values_by_number.get(number)
                enum_values.append(
                    str(enum_value.name) if enum_value is not None else number
                )
            return str(enum_values)
        if field.enum_type is not None:
            number: int = int(value)
            enum_value: Any | None = field.enum_type.values_by_number.get(number)
            if enum_value is not None:
                return str(enum_value.name)
            return number
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float | str):
            return value
        return str(value)
