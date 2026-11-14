"""The PySCIPOpt solver integration."""

import math
import sys
from collections.abc import Callable
from functools import wraps
from inspect import Signature, signature
from types import ModuleType
from typing import Any

import pyscipopt
import pyscipopt.scip
from mlflow_optimization.common.patching import receiver_of
from mlflow_optimization.features.autolog.ports import ObservationError, SolveObserver
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

_PYSCIPOPT_READ_ERRORS: tuple[type[Exception], ...] = (
    AttributeError,
    MemoryError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
    Warning,
)
_OBJECTIVE_VALUE_ERRORS: tuple[type[Exception], ...] = (
    *_PYSCIPOPT_READ_ERRORS,
    AssertionError,
)
_OBSERVATION_BUILD_ERRORS: tuple[type[Exception], ...] = (
    *_OBJECTIVE_VALUE_ERRORS,
    LookupError,
    OSError,
)
_PARAMETER_READ_ERRORS: tuple[type[Exception], ...] = (
    *_PYSCIPOPT_READ_ERRORS,
    LookupError,
)

_STATUS_TERMINATIONS: dict[str, Termination] = {
    "unknown": Termination.UNKNOWN,
    "optimal": Termination.OPTIMAL,
    "infeasible": Termination.INFEASIBLE,
    "unbounded": Termination.UNBOUNDED,
    "inforunbd": Termination.INFEASIBLE_OR_UNBOUNDED,
    "userinterrupt": Termination.INTERRUPTED,
    "terminate": Termination.INTERRUPTED,
    "nodelimit": Termination.NODE_LIMIT,
    "totalnodelimit": Termination.NODE_LIMIT,
    "stallnodelimit": Termination.NODE_LIMIT,
    "timelimit": Termination.TIME_LIMIT,
    "memlimit": Termination.OTHER_LIMIT,
    "gaplimit": Termination.OTHER_LIMIT,
    "primallimit": Termination.OTHER_LIMIT,
    "duallimit": Termination.OTHER_LIMIT,
    "sollimit": Termination.SOLUTION_LIMIT,
    "bestsollimit": Termination.SOLUTION_LIMIT,
    "restartlimit": Termination.OTHER_LIMIT,
}


class Integration:
    """The PySCIPOpt solver integration."""

    api: str = "pyscipopt"

    def __init__(self) -> None:
        self._installed: bool = False
        self._observer: SolveObserver | None = None
        self._original_model: type[Any] | None = None
        self._logging_model: type[Any] | None = None
        self._rebound_names: list[tuple[ModuleType, str]] = []

    def install(self, observer: SolveObserver) -> None:
        """A patch that replaces model references with an observing subclass."""
        if self._installed:
            self._observer = observer
            return

        original_model: type[Any] = pyscipopt.Model
        integration: Integration = self
        default_parameters: dict[str, bool | int | float | str] | None = (
            _read_default_parameters(original_model)
        )

        class LoggingModel(original_model):
            """A PySCIPOpt model that observes optimization calls."""

        entry_point_names: tuple[str, ...] = (
            "optimize",
            "optimizeNogil",
            "solveConcurrent",
        )
        entry_point_name: str
        for entry_point_name in entry_point_names:
            original_entry_point: Any = getattr(original_model, entry_point_name, None)
            if not callable(original_entry_point):
                continue
            observed_entry_point: Callable[..., Any] = _wrap_solve_entry_point(
                original_entry_point,
                integration,
                default_parameters,
            )
            setattr(LoggingModel, entry_point_name, observed_entry_point)

        logging_model: type[Any] = LoggingModel
        rebound_names: list[tuple[ModuleType, str]] = []
        loaded_module_value: Any
        loaded_module: ModuleType
        attribute_name: str
        attribute_value: Any
        module_values: dict[str, Any]

        for loaded_module_value in tuple(sys.modules.values()):
            if not isinstance(loaded_module_value, ModuleType):
                continue
            loaded_module = loaded_module_value
            module_values = vars(loaded_module)
            for attribute_name, attribute_value in tuple(module_values.items()):
                if attribute_value is not original_model:
                    continue
                module_values[attribute_name] = logging_model
                rebound_names.append((loaded_module, attribute_name))

        self._observer = observer
        self._original_model = original_model
        self._logging_model = logging_model
        self._rebound_names = rebound_names
        self._installed = True

    def uninstall(self) -> None:
        """The restoration of original PySCIPOpt model references."""
        if not self._installed:
            return

        assert self._original_model is not None
        assert self._logging_model is not None
        original_model: type[Any] = self._original_model
        logging_model: type[Any] = self._logging_model
        loaded_module_value: Any
        loaded_module: ModuleType
        attribute_name: str
        attribute_value: Any
        module_values: dict[str, Any]

        for loaded_module_value in tuple(sys.modules.values()):
            if not isinstance(loaded_module_value, ModuleType):
                continue
            loaded_module = loaded_module_value
            module_values = vars(loaded_module)
            for attribute_name, attribute_value in tuple(module_values.items()):
                if attribute_value is logging_model:
                    module_values[attribute_name] = original_model

        for loaded_module, attribute_name in reversed(self._rebound_names):
            vars(loaded_module)[attribute_name] = original_model

        self._installed = False
        self._observer = None
        self._original_model = None
        self._logging_model = None
        self._rebound_names = []


def _wrap_solve_entry_point(
    original_entry_point: Callable[..., Any],
    integration: Integration,
    default_parameters: dict[str, bool | int | float | str] | None,
) -> Callable[..., Any]:
    """An observed wrapper for one original PySCIPOpt solve entry point."""
    original_signature: Signature = signature(original_entry_point)

    @wraps(original_entry_point)
    def observed_entry_point(
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        original_signature.bind(*args, **kwargs)
        model: Any = receiver_of(original_entry_point, args, kwargs)
        if not integration._installed or integration._observer is None:
            return original_entry_point(*args, **kwargs)

        # NOTE: SCIP changes parameters while solving, so capture configured values first.
        parameters_before_solve: tuple[SolverParameter, ...] = _read_parameters(
            model,
            default_parameters,
        )

        def call() -> Any:
            return original_entry_point(*args, **kwargs)

        def extract(
            result: Any | None, error: BaseException | None
        ) -> SolveObservation:
            extraction_error: Exception
            try:
                return _extract_observation(
                    model,
                    result,
                    error,
                    parameters_before_solve,
                )
            except _OBSERVATION_BUILD_ERRORS as extraction_error:
                raise ObservationError(
                    "Could not build a PySCIPOpt solve observation."
                ) from extraction_error

        return integration._observer.observe(integration.api, call, extract)

    return observed_entry_point


def _extract_observation(
    model: Any,
    result: Any | None,
    error: BaseException | None,
    parameters_before_solve: tuple[SolverParameter, ...],
) -> SolveObservation:
    """The SDK-free observation of one PySCIPOpt optimization call."""
    raw_status: str = _read_value(lambda: str(model.getStatus()), "unknown")
    has_solution: bool | None = _read_value(lambda: model.getNSols() > 0, None)
    termination: Termination = _STATUS_TERMINATIONS.get(
        raw_status.lower(), Termination.UNKNOWN
    )

    if isinstance(error, KeyboardInterrupt):
        termination = Termination.INTERRUPTED
    elif error is not None:
        termination = Termination.ERROR

    has_objective: bool | None = _read_has_objective(model)
    sense_name: str | MissingReason = _read_value(
        lambda: str(model.getObjectiveSense()).lower(),
        MissingReason.FAILED,
    )
    sense: ObjectiveSense | None = (
        {
            "minimize": ObjectiveSense.MINIMIZE,
            "maximize": ObjectiveSense.MAXIMIZE,
        }.get(sense_name)
        if isinstance(sense_name, str)
        else None
    )

    if has_objective is False:
        objective_value: float | MissingReason = MissingReason.UNDEFINED
        dual_bound: float | MissingReason = MissingReason.UNDEFINED
    else:
        solver_infinity: float | MissingReason = _read_number(
            lambda: float(model.infinity())
        )
        dual_bound = _read_scip_number(
            lambda: float(model.getDualbound()),
            solver_infinity,
        )
        if has_solution is False:
            objective_value = MissingReason.NO_SOLUTION
        else:
            objective_value = _read_scip_number(
                lambda: float(model.getObjVal()),
                solver_infinity,
                _OBJECTIVE_VALUE_ERRORS,
            )

    solver_time_seconds: float | MissingReason = _read_number(
        lambda: float(model.getSolvingTime())
    )
    node_count: float | MissingReason = _read_number(lambda: float(model.getNNodes()))
    statistics: dict[str, float | MissingReason] = {"node_count": node_count}
    native: dict[str, float] = {}

    _add_native_number(native, "solution_count", lambda: float(model.getNSols()))
    _add_native_number(native, "solutions_found", lambda: float(model.getNSolsFound()))
    _add_native_number(native, "lp_count", lambda: float(model.getNLPs()))
    _add_native_number(native, "lp_iterations", lambda: float(model.getNLPIterations()))
    _add_native_number(
        native, "total_node_count", lambda: float(model.getNTotalNodes())
    )
    _add_native_number(
        native, "presolving_time_seconds", lambda: float(model.getPresolvingTime())
    )
    _add_native_number(
        native, "total_time_seconds", lambda: float(model.getTotalTime())
    )

    if has_solution is not False and has_objective is not False:
        gap: float | MissingReason = _read_scip_number(
            lambda: float(model.getGap()),
            solver_infinity,
        )
        if not isinstance(gap, MissingReason):
            native["gap"] = gap

    metrics: ReportedMetrics = ReportedMetrics(
        has_objective=has_objective,
        sense=sense,
        objective_value=objective_value,
        dual_bound=dual_bound,
        solver_time_seconds=solver_time_seconds,
        statistics=statistics,
        native=native,
    )
    metadata: ReportedMetadata = _read_metadata(model, parameters_before_solve)
    outcome: ReportedOutcome = ReportedOutcome(
        raw_status=raw_status,
        termination=termination,
        has_solution=has_solution,
    )

    return SolveObservation(
        api="pyscipopt", outcome=outcome, metrics=metrics, metadata=metadata
    )


def _read_has_objective(model: Any) -> bool | None:
    """Whether the model has a nonzero objective function."""
    objective_terms_present: bool | MissingReason = _read_value(
        lambda: bool(model.getObjective().terms),
        MissingReason.FAILED,
    )
    objective_offset: float | MissingReason = _read_value(
        lambda: float(model.getObjoffset()),
        MissingReason.FAILED,
    )

    if objective_terms_present is True:
        return True
    if isinstance(objective_offset, float) and objective_offset != 0.0:
        return True
    if objective_terms_present is False and objective_offset == 0.0:
        return False
    return None


def _read_metadata(
    model: Any,
    parameters: tuple[SolverParameter, ...],
) -> ReportedMetadata:
    """The solver and problem metadata reported by PySCIPOpt."""
    api_version: str | None = _read_value(lambda: str(pyscipopt.__version__), None)
    major_version: int | None = _read_value(lambda: int(model.getMajorVersion()), None)
    minor_version: int | None = _read_value(lambda: int(model.getMinorVersion()), None)
    technical_version: int | None = _read_value(
        lambda: int(model.getTechVersion()), None
    )
    solver_version: str | None = None

    if (
        major_version is not None
        and minor_version is not None
        and technical_version is not None
    ):
        solver_version = f"{major_version}.{minor_version}.{technical_version}"

    problem: dict[str, int] = {}
    variable_count: int | None = _read_value(
        lambda: int(model.getNVars(transformed=False)), None
    )
    constraint_count: int | None = _read_value(
        lambda: int(model.getNConss(transformed=False)), None
    )

    if variable_count is not None:
        problem["num_variables"] = variable_count
    if constraint_count is not None:
        problem["num_constraints"] = constraint_count

    variables: list[Any] | MissingReason = _read_value(
        lambda: model.getVars(transformed=False),
        MissingReason.FAILED,
    )
    if not isinstance(variables, MissingReason):
        integer_variable_count: int = 0
        integer_variable_count_available: bool = True
        variable: Any
        variable_type: str | MissingReason
        for variable in variables:
            variable_type = _read_value(
                lambda variable=variable: str(variable.vtype()).upper(),
                MissingReason.FAILED,
            )
            if isinstance(variable_type, MissingReason):
                integer_variable_count_available = False
                break
            if variable_type in {"BINARY", "INTEGER", "IMPLINT"}:
                integer_variable_count += 1
        if integer_variable_count_available:
            problem["num_integer_variables"] = integer_variable_count

    return ReportedMetadata(
        solver="scip",
        api_version=api_version,
        solver_version=solver_version,
        problem=problem,
        parameters=parameters,
        wall_time_includes_problem_build=False,
    )


def _read_default_parameters(
    model_class: type[Any],
) -> dict[str, bool | int | float | str] | None:
    """The PySCIPOpt parameter values of a new model."""
    defaults_model: Any | MissingReason = _read_value(
        lambda: model_class(),
        MissingReason.FAILED,
    )
    if isinstance(defaults_model, MissingReason):
        return None

    default_values: dict[str, Any] | MissingReason
    try:
        default_values = _read_value(
            lambda: defaults_model.getParams(),
            MissingReason.FAILED,
        )
    finally:
        _read_value(lambda: defaults_model.free(), None)

    if isinstance(default_values, MissingReason) or not isinstance(
        default_values, dict
    ):
        return None

    default_parameters: dict[str, bool | int | float | str] = {}
    parameter_name: Any
    parameter_value: Any
    for parameter_name, parameter_value in default_values.items():
        if isinstance(parameter_name, str) and isinstance(
            parameter_value,
            (bool, int, float, str),
        ):
            default_parameters[parameter_name] = parameter_value
    return default_parameters


def _read_parameters(
    model: Any,
    default_parameters: dict[str, bool | int | float | str] | None,
) -> tuple[SolverParameter, ...]:
    """The PySCIPOpt parameters that differ from their defaults."""
    if default_parameters is None:
        return ()

    parameters: list[SolverParameter] = []
    parameter_name: str
    default_value: bool | int | float | str
    for parameter_name, default_value in default_parameters.items():
        parameter_value: Any | MissingReason = _read_value(
            lambda parameter_name=parameter_name: model.getParam(parameter_name),
            MissingReason.FAILED,
            _PARAMETER_READ_ERRORS,
        )
        if isinstance(parameter_value, MissingReason):
            continue
        if not isinstance(parameter_value, (bool, int, float, str)):
            continue
        if parameter_value == default_value:
            continue
        parameters.append(
            SolverParameter(
                name=parameter_name,
                value=parameter_value,
                origin=ParameterOrigin.EXPLICIT,
            )
        )

    return tuple(parameters)


def _add_native_number(
    native: dict[str, float],
    metric_name: str,
    reader: Callable[[], float],
) -> None:
    """One available SCIP-specific metric in native metrics."""
    value: float | MissingReason = _read_number(reader)
    if isinstance(value, MissingReason):
        return
    native[metric_name] = value


def _read_number(
    reader: Callable[[], float],
    read_errors: tuple[type[Exception], ...] = _PYSCIPOPT_READ_ERRORS,
) -> float | MissingReason:
    """A guarded PySCIPOpt numeric value."""
    return _read_value(reader, MissingReason.FAILED, read_errors)


def _read_scip_number(
    reader: Callable[[], float],
    solver_infinity: float | MissingReason,
    read_errors: tuple[type[Exception], ...] = _PYSCIPOPT_READ_ERRORS,
) -> float | MissingReason:
    """A PySCIPOpt numeric value with its infinity sentinel normalized."""
    value: float | MissingReason = _read_number(reader, read_errors)
    if isinstance(value, MissingReason):
        return value
    if math.isinf(value):
        return value
    if isinstance(solver_infinity, MissingReason) or solver_infinity <= 0.0:
        return MissingReason.FAILED
    if value >= solver_infinity:
        return math.inf
    if value <= -solver_infinity:
        return -math.inf
    return value


def _read_value[Value](
    reader: Callable[[], Value],
    unavailable: Value,
    read_errors: tuple[type[Exception], ...] = _PYSCIPOPT_READ_ERRORS,
) -> Value:
    """A PySCIPOpt value or its fallback after a supported read error."""
    try:
        return reader()
    except read_errors:
        return unavailable
