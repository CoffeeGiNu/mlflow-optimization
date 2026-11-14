"""Observation hooks for OR-Tools Routing solves."""

from collections.abc import Callable
from contextlib import suppress
from functools import wraps
from typing import Any

import ortools
from ortools.constraint_solver import pywrapcp

from mlflow_optimization.common.patching import Patch
from mlflow_optimization.features.autolog.models import SolveObservation
from mlflow_optimization.features.autolog.ports import SolveObserver, SolverIntegration
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

_STATUS_NAMES: dict[int, str] = {
    0: "ROUTING_NOT_SOLVED",
    1: "ROUTING_SUCCESS",
    2: "ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED",
    3: "ROUTING_FAIL",
    4: "ROUTING_FAIL_TIMEOUT",
    5: "ROUTING_INVALID",
    6: "ROUTING_INFEASIBLE",
    7: "ROUTING_OPTIMAL",
}
_STATUS_TERMINATIONS: dict[int, Termination] = {
    0: Termination.UNKNOWN,
    1: Termination.FEASIBLE,
    2: Termination.FEASIBLE,
    3: Termination.UNKNOWN,
    4: Termination.TIME_LIMIT,
    5: Termination.UNKNOWN,
    6: Termination.INFEASIBLE,
    7: Termination.OPTIMAL,
}
_ENTRY_POINTS: tuple[str, ...] = (
    "SolveWithParameters",
    "Solve",
    "SolveFromAssignmentWithParameters",
    "SolveFromAssignmentsWithParameters",
)


def _explicit_parameters(
    actual: Any,
    defaults: Any,
    prefix: str = "",
) -> list[SolverParameter]:
    """The scalar parameters that differ from OR-Tools defaults."""
    parameters: list[SolverParameter] = []
    fields: Any = ()
    with suppress(BaseException):
        fields = actual.DESCRIPTOR.fields
    field: Any
    for field in fields:
        parameter_name: str = f"{prefix}.{field.name}" if prefix else field.name
        actual_value_available: bool = False
        actual_value: Any = None
        with suppress(BaseException):
            actual_value = getattr(actual, field.name)
            actual_value_available = True
        default_value_available: bool = False
        default_value: Any = None
        with suppress(BaseException):
            default_value = getattr(defaults, field.name)
            default_value_available = True
        if not actual_value_available or not default_value_available:
            continue
        if field.is_repeated:
            actual_values_available: bool = False
            actual_values: list[Any] = []
            with suppress(BaseException):
                actual_values = list(actual_value)
                actual_values_available = True
            default_values_available: bool = False
            default_values: list[Any] = []
            with suppress(BaseException):
                default_values = list(default_value)
                default_values_available = True
            if not actual_values_available or not default_values_available:
                continue
            if actual_values != default_values:
                parameters.append(
                    SolverParameter(
                        parameter_name,
                        str(actual_values),
                        ParameterOrigin.EXPLICIT,
                    )
                )
        elif field.message_type is not None:
            parameters.extend(_explicit_parameters(actual_value, default_value, parameter_name))
        elif actual_value != default_value:
            if field.enum_type is not None:
                enum_value: Any = None
                with suppress(BaseException):
                    enum_value = field.enum_type.values_by_number.get(actual_value)
                value: int | float | bool | str = (
                    enum_value.name if enum_value is not None else actual_value
                )
            else:
                value = actual_value
            if isinstance(value, (int, float, bool, str)):
                parameters.append(
                    SolverParameter(
                        parameter_name,
                        value,
                        ParameterOrigin.EXPLICIT,
                    )
                )
    return parameters


class Integration(SolverIntegration):
    """The OR-Tools Routing solver integration."""

    api: str = "routing"

    def __init__(self) -> None:
        self._patches: list[Patch] = []

    def install(self, observer: SolveObserver) -> None:
        """Route the available Routing solve entry points through an observer."""
        if self._patches:
            return
        entry_point: str
        for entry_point in _ENTRY_POINTS:
            if hasattr(pywrapcp.RoutingModel, entry_point):
                patch: Patch = Patch(pywrapcp.RoutingModel, entry_point)
                patch.install(self._patch_factory(observer, entry_point))
                self._patches.append(patch)

    def uninstall(self) -> None:
        """Restore the original Routing solve entry points."""
        patch: Patch
        for patch in reversed(self._patches):
            patch.uninstall()
        self._patches.clear()

    def _patch_factory(
        self,
        observer: SolveObserver,
        entry_point: str,
    ) -> Callable[[Any], Any]:
        """The wrapper factory for one Routing solve entry point."""

        def factory(original: Any) -> Any:
            @wraps(original)
            def wrapped(
                routing_model: pywrapcp.RoutingModel,
                *arguments: Any,
                **keyword_arguments: Any,
            ) -> Any:
                search_parameters: Any | None = self._search_parameters(
                    entry_point,
                    arguments,
                    keyword_arguments,
                )

                def call() -> Any:
                    return original(routing_model, *arguments, **keyword_arguments)

                def extract(
                    result: Any | None,
                    error: BaseException | None,
                ) -> SolveObservation:
                    return self._extract(routing_model, result, error, search_parameters)

                return observer.observe(self.api, call, extract)

            return wrapped

        return factory

    def _search_parameters(
        self,
        entry_point: str,
        arguments: tuple[Any, ...],
        keyword_arguments: dict[str, Any],
    ) -> Any | None:
        """The search parameters passed to one solve entry point."""
        if entry_point == "SolveWithParameters":
            parameter_position: int = 0
        elif entry_point in (
            "SolveFromAssignmentWithParameters",
            "SolveFromAssignmentsWithParameters",
        ):
            parameter_position = 1
        else:
            return None
        if "search_parameters" in keyword_arguments:
            return keyword_arguments["search_parameters"]
        if parameter_position < len(arguments):
            return arguments[parameter_position]
        return None

    def _extract(
        self,
        routing_model: pywrapcp.RoutingModel,
        result: Any | None,
        error: BaseException | None,
        search_parameters: Any | None,
    ) -> SolveObservation:
        """The plain observation values reported by one Routing solve."""
        status_code: int | None = None
        raw_status: str = "unknown"
        termination: Termination = Termination.UNKNOWN
        with suppress(BaseException):
            status_code = int(routing_model.status())
        if status_code is not None:
            status_name: str = _STATUS_NAMES.get(status_code, "UNKNOWN")
            raw_status = f"{status_name} ({status_code})"
            termination = _STATUS_TERMINATIONS.get(status_code, Termination.UNKNOWN)
        if error is not None:
            termination = (
                Termination.INTERRUPTED
                if isinstance(error, KeyboardInterrupt)
                else Termination.ERROR
            )
        has_solution: bool | None = result is not None if error is None else None
        objective_value: float | MissingReason
        if result is None:
            objective_value = (
                MissingReason.NO_SOLUTION
                if error is None
                else MissingReason.FAILED
            )
        else:
            objective_value = MissingReason.FAILED
            with suppress(BaseException):
                objective_value = float(result.ObjectiveValue())
        solver_time_seconds: float | MissingReason = MissingReason.FAILED
        with suppress(BaseException):
            solver_time_seconds = float(routing_model.solver().WallTime()) / 1000.0
        stats: dict[str, float | MissingReason] = {}
        native: dict[str, float] = {}
        branches: float | None = None
        with suppress(BaseException):
            branches = float(routing_model.solver().Branches())
        if branches is not None:
            native["branches"] = branches
        failures: float | None = None
        with suppress(BaseException):
            failures = float(routing_model.solver().Failures())
        if failures is not None:
            native["failures"] = failures
        problem: dict[str, int] = {}
        num_locations: int | None = None
        with suppress(BaseException):
            num_locations = int(routing_model.nodes())
        if num_locations is not None:
            problem["num_locations"] = num_locations
        vehicle_count: int | None = None
        with suppress(BaseException):
            vehicle_count = int(routing_model.vehicles())
        if vehicle_count is not None:
            problem["num_vehicles"] = vehicle_count
        if result is not None and vehicle_count is not None:
            vehicles_used: int = 0
            vehicle_reads_succeeded: bool = True
            vehicle_index: int
            for vehicle_index in range(vehicle_count):
                vehicle_read_succeeded: bool = False
                is_vehicle_used: bool = False
                with suppress(BaseException):
                    is_vehicle_used = bool(routing_model.IsVehicleUsed(result, vehicle_index))
                    vehicle_read_succeeded = True
                if not vehicle_read_succeeded:
                    vehicle_reads_succeeded = False
                elif is_vehicle_used:
                    vehicles_used += 1
            if vehicle_reads_succeeded:
                native["vehicles_used"] = float(vehicles_used)
        if search_parameters is None:
            with suppress(BaseException):
                search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        parameters: list[SolverParameter] = []
        if search_parameters is not None:
            defaults: Any | None = None
            with suppress(BaseException):
                defaults = pywrapcp.DefaultRoutingSearchParameters()
            if defaults is not None:
                parameters = _explicit_parameters(search_parameters, defaults)
        observation: SolveObservation = SolveObservation(
            api=self.api,
            outcome=ReportedOutcome(raw_status, termination, has_solution),
            metrics=ReportedMetrics(
                has_objective=True,
                sense=ObjectiveSense.MINIMIZE,
                objective_value=objective_value,
                dual_bound=MissingReason.UNSUPPORTED,
                solver_time_seconds=solver_time_seconds,
                stats=stats,
                native=native,
            ),
            metadata=ReportedMetadata(
                solver="routing",
                api_version=ortools.__version__,
                solver_version=ortools.__version__,
                problem=problem,
                parameters=parameters,
                wall_time_includes_model_build=False,
            ),
        )
        return observation
