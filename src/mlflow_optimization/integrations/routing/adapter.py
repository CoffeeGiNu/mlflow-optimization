"""The OR-Tools Routing solver integration."""

from collections.abc import Callable
from functools import wraps
from inspect import BoundArguments, signature
from typing import Any

import ortools
from ortools.constraint_solver import pywrapcp

from mlflow_optimization.common.patching import Patch, receiver_of
from mlflow_optimization.features.autolog.ports import (
    ObservationError,
    SolveObserver,
    SolverIntegration,
)
from mlflow_optimization.features.autolog.types import SolveObservation
from mlflow_optimization.features.metadata.types import (
    ParameterOrigin,
    ReportedMetadata,
    SolverParameter,
)
from mlflow_optimization.features.metrics.types import (
    MissingReason,
    ObjectiveSense,
    ReportedMetrics,
)
from mlflow_optimization.features.outcomes.types import ReportedOutcome, Termination

_SDK_EXCEPTION: type[Exception] = Exception
_STATUSES: dict[int, tuple[str, Termination]] = {
    0: ("ROUTING_NOT_SOLVED", Termination.UNKNOWN),
    1: ("ROUTING_SUCCESS", Termination.FEASIBLE),
    2: ("ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED", Termination.FEASIBLE),
    3: ("ROUTING_FAIL", Termination.UNKNOWN),
    4: ("ROUTING_FAIL_TIMEOUT", Termination.TIME_LIMIT),
    5: ("ROUTING_INVALID", Termination.ERROR),
    6: ("ROUTING_INFEASIBLE", Termination.INFEASIBLE),
    7: ("ROUTING_OPTIMAL", Termination.OPTIMAL),
}
_FAILED_STATUS_CODES: frozenset[int] = frozenset({3, 4, 5, 6})
_ENTRY_POINTS: tuple[str, ...] = (
    "SolveWithParameters",
    "Solve",
    "SolveFromAssignmentWithParameters",
    "SolveFromAssignmentsWithParameters",
    "SolveWithIteratedLocalSearch",
    "FastSolveFromAssignmentWithParameters",
)


def _explicit_parameters(
    actual: Any,
    defaults: Any,
    prefix: str = "",
) -> list[SolverParameter]:
    """Scalar values that differ from OR-Tools search defaults."""
    parameters: list[SolverParameter] = []
    try:
        fields: Any = actual.DESCRIPTOR.fields
    except _SDK_EXCEPTION:
        return parameters
    for field in fields:
        try:
            field_name: str = field.name
        except _SDK_EXCEPTION:
            continue
        parameter_name: str = f"{prefix}.{field_name}" if prefix else field_name
        try:
            actual_value: Any = getattr(actual, field_name)
        except _SDK_EXCEPTION:
            continue
        try:
            default_value: Any = getattr(defaults, field_name)
        except _SDK_EXCEPTION:
            continue
        try:
            is_repeated: bool = bool(field.is_repeated)
        except _SDK_EXCEPTION:
            continue
        if is_repeated:
            try:
                actual_values: list[Any] = list(actual_value)
            except _SDK_EXCEPTION:
                continue
            try:
                default_values: list[Any] = list(default_value)
            except _SDK_EXCEPTION:
                continue
            if actual_values != default_values:
                try:
                    actual_values_string: str = str(actual_values)
                except _SDK_EXCEPTION:
                    continue
                parameters.append(
                    SolverParameter(
                        parameter_name,
                        actual_values_string,
                        ParameterOrigin.EXPLICIT,
                    )
                )
            continue
        try:
            message_type: Any | None = field.message_type
        except _SDK_EXCEPTION:
            continue
        if message_type is not None:
            parameters.extend(
                _explicit_parameters(actual_value, default_value, parameter_name)
            )
            continue
        try:
            differs_from_default: bool = actual_value != default_value
        except _SDK_EXCEPTION:
            continue
        if differs_from_default:
            try:
                enum_type: Any | None = field.enum_type
            except _SDK_EXCEPTION:
                continue
            value: int | float | bool | str = actual_value
            if enum_type is not None:
                try:
                    enum_value: Any = enum_type.values_by_number.get(actual_value)
                except _SDK_EXCEPTION:
                    continue
                if enum_value is not None:
                    try:
                        value = enum_value.name
                    except _SDK_EXCEPTION:
                        continue
            if isinstance(value, (int, float, bool, str)):
                parameters.append(
                    SolverParameter(parameter_name, value, ParameterOrigin.EXPLICIT)
                )
    return parameters


class RoutingAdapter(SolverIntegration):
    """The OR-Tools Routing solver integration."""

    api: str = "routing"

    def __init__(self) -> None:
        self._patches: list[Patch] = []

    def install(self, observer: SolveObserver) -> None:
        """The solve hooks installed on the Routing model."""
        if self._patches:
            return
        for entry_point in _ENTRY_POINTS:
            if hasattr(pywrapcp.RoutingModel, entry_point):
                patch: Patch = Patch(pywrapcp.RoutingModel, entry_point)
                patch.install(self._patch_factory(observer, entry_point))
                self._patches.append(patch)

    def uninstall(self) -> None:
        """The original solve hooks after uninstall."""
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
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                call_observer: SolveObserver = observer
                routing_model: pywrapcp.RoutingModel = receiver_of(
                    original, args, kwargs
                )
                if routing_model is None:
                    return original(*args, **kwargs)
                counter_snapshot: tuple[float | None, float | None] = (
                    self._counter_snapshot(routing_model)
                )
                search_parameters: Any | None = self._search_parameters(
                    original,
                    args,
                    kwargs,
                )

                def call() -> Any:
                    return original(*args, **kwargs)

                def extract(
                    result: Any | None,
                    error: BaseException | None,
                ) -> SolveObservation:
                    return self._extract(
                        routing_model,
                        result,
                        error,
                        search_parameters,
                        counter_snapshot,
                    )

                return call_observer.observe(self.api, call, extract)

            return wrapped

        return factory

    def _search_parameters(
        self,
        original: Callable[..., Any],
        arguments: tuple[Any, ...],
        keyword_arguments: dict[str, Any],
    ) -> Any | None:
        """Search parameters for one Routing solve call."""
        bound_arguments: BoundArguments
        try:
            bound_arguments = signature(original).bind_partial(
                *arguments, **keyword_arguments
            )
        except (TypeError, ValueError):
            return None
        return bound_arguments.arguments.get("search_parameters")

    def _counter_snapshot(
        self,
        routing_model: pywrapcp.RoutingModel,
    ) -> tuple[float | None, float | None]:
        """The branch and failure counters for one Routing model."""
        branches: float | None = None
        try:
            branches = float(routing_model.solver().Branches())
        except _SDK_EXCEPTION:
            branches = None
        failures: float | None = None
        try:
            failures = float(routing_model.solver().Failures())
        except _SDK_EXCEPTION:
            failures = None
        return branches, failures

    def _extract(
        self,
        routing_model: pywrapcp.RoutingModel,
        result: Any | None,
        error: BaseException | None,
        search_parameters: Any | None,
        counter_snapshot: tuple[float | None, float | None],
    ) -> SolveObservation:
        """The plain observation values reported by one Routing solve."""
        try:
            return self._build_observation(
                routing_model,
                result,
                error,
                search_parameters,
                counter_snapshot,
            )
        except ObservationError:
            raise
        except _SDK_EXCEPTION as extraction_error:
            raise ObservationError(
                "Could not read a Routing observation."
            ) from extraction_error

    def _build_observation(
        self,
        routing_model: pywrapcp.RoutingModel,
        result: Any | None,
        error: BaseException | None,
        search_parameters: Any | None,
        counter_snapshot: tuple[float | None, float | None],
    ) -> SolveObservation:
        """The Routing solve observation."""
        status_code: int | None = None
        raw_status: str = "unknown"
        termination: Termination = Termination.UNKNOWN
        has_solution: bool | None = None
        reported_result: Any | None = result
        if error is not None:
            termination = (
                Termination.INTERRUPTED
                if isinstance(error, KeyboardInterrupt)
                else Termination.ERROR
            )
        else:
            try:
                status_code = int(routing_model.status())
            except _SDK_EXCEPTION:
                status_code = None
            if status_code is not None:
                status_name: str
                termination = Termination.UNKNOWN
                status_name, termination = _STATUSES.get(
                    status_code, ("UNKNOWN", Termination.UNKNOWN)
                )
                raw_status = f"{status_name} ({status_code})"
            has_solution = result is not None
            if status_code in _FAILED_STATUS_CODES:
                has_solution = False
                reported_result = None
            elif result is not None:
                termination = (
                    Termination.OPTIMAL if status_code == 7 else Termination.FEASIBLE
                )
        if error is not None:
            metrics: ReportedMetrics = ReportedMetrics()
        else:
            metrics = self._metrics(
                routing_model,
                reported_result,
                counter_snapshot,
            )
        problem: dict[str, int] = {}
        num_locations: int | None = None
        try:
            num_locations = int(routing_model.nodes())
        except _SDK_EXCEPTION:
            num_locations = None
        if num_locations is not None:
            problem["num_locations"] = num_locations
        vehicle_count: int | None = None
        try:
            vehicle_count = int(routing_model.vehicles())
        except _SDK_EXCEPTION:
            vehicle_count = None
        if vehicle_count is not None:
            problem["num_vehicles"] = vehicle_count
        parameters: list[SolverParameter] = self._reported_parameters(search_parameters)
        api_version: str | None = None
        try:
            api_version = str(ortools.__version__)
        except _SDK_EXCEPTION:
            api_version = None
        observation: SolveObservation = SolveObservation(
            api=self.api,
            outcome=ReportedOutcome(
                raw_status,
                termination,
                has_solution,
            ),
            metrics=metrics,
            metadata=ReportedMetadata(
                solver="routing",
                api_version=api_version,
                solver_version=api_version,
                problem=problem,
                parameters=parameters,
                wall_time_includes_problem_build=False,
            ),
        )
        return observation

    def _metrics(
        self,
        routing_model: pywrapcp.RoutingModel,
        result: Any | None,
        counter_snapshot: tuple[float | None, float | None],
    ) -> ReportedMetrics:
        """The Routing metrics reported for a successful solve call."""
        objective_value: float | MissingReason
        if result is None:
            objective_value = MissingReason.NO_SOLUTION
        else:
            try:
                objective_value = float(result.ObjectiveValue())
            except _SDK_EXCEPTION:
                objective_value = MissingReason.FAILED
        solver_time_seconds: float | MissingReason = MissingReason.UNSUPPORTED
        statistics: dict[str, float | MissingReason] = {}
        native: dict[str, float] = {}
        branches_before: float | None = counter_snapshot[0]
        failures_before: float | None = counter_snapshot[1]
        branches_after: float | None
        failures_after: float | None
        branches_after, failures_after = self._counter_snapshot(routing_model)
        if branches_before is not None and branches_after is not None:
            branches_elapsed: float = branches_after - branches_before
            if branches_elapsed >= 0:
                native["branches"] = branches_elapsed
        if failures_before is not None and failures_after is not None:
            failures_elapsed: float = failures_after - failures_before
            if failures_elapsed >= 0:
                native["failures"] = failures_elapsed
        vehicle_count: int | None = None
        try:
            vehicle_count = int(routing_model.vehicles())
        except _SDK_EXCEPTION:
            vehicle_count = None
        if result is not None and vehicle_count is not None:
            vehicles_used: int = 0
            vehicle_reads_succeeded: bool = True
            for vehicle_index in range(vehicle_count):
                try:
                    is_vehicle_used: bool = bool(
                        routing_model.IsVehicleUsed(result, vehicle_index)
                    )
                except _SDK_EXCEPTION:
                    vehicle_reads_succeeded = False
                    continue
                if is_vehicle_used:
                    vehicles_used += 1
            if vehicle_reads_succeeded:
                native["vehicles_used"] = float(vehicles_used)
        return ReportedMetrics(
            has_objective=True,
            sense=ObjectiveSense.MINIMIZE,
            objective_value=objective_value,
            dual_bound=MissingReason.UNSUPPORTED,
            solver_time_seconds=solver_time_seconds,
            statistics=statistics,
            native=native,
        )

    def _reported_parameters(
        self,
        search_parameters: Any | None,
    ) -> list[SolverParameter]:
        """The explicit Routing search parameters reported for one call."""
        parameters: list[SolverParameter] = []
        if search_parameters is None:
            return parameters
        defaults: Any | None = None
        try:
            defaults = pywrapcp.DefaultRoutingSearchParameters()
        except _SDK_EXCEPTION:
            return parameters
        parameters = _explicit_parameters(search_parameters, defaults)
        return parameters
