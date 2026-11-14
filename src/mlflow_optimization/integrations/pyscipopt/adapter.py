"""The PySCIPOpt solver integration."""

import ast
import math
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pyscipopt
import pyscipopt.scip
from mlflow_optimization.features.autolog.models import SolveObservation
from mlflow_optimization.features.autolog.ports import ObservationError, SolveObserver
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
        """Install a logging subclass in PySCIPOpt model references."""
        if self._installed:
            self._observer = observer
            return

        original_model: type[Any] = pyscipopt.Model
        integration: Integration = self

        class LoggingModel(original_model):
            """A PySCIPOpt model that observes optimization calls."""

            def optimize(model: Any) -> Any:
                if not integration._installed or integration._observer is None:
                    return super(LoggingModel, model).optimize()

                def call() -> Any:
                    return super(LoggingModel, model).optimize()

                def extract(
                    result: Any | None, error: BaseException | None
                ) -> SolveObservation:
                    extraction_error: Exception
                    try:
                        return _extract_observation(model, result, error)
                    except _OBSERVATION_BUILD_ERRORS as extraction_error:
                        raise ObservationError(
                            "Could not build a PySCIPOpt solve observation."
                        ) from extraction_error

                return integration._observer.observe(integration.api, call, extract)

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
        """Restore the original PySCIPOpt model references."""
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


def _extract_observation(
    model: Any,
    result: Any | None,
    error: BaseException | None,
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
        dual_bound = _read_number(lambda: float(model.getDualbound()))
        if has_solution is False:
            objective_value = MissingReason.NO_SOLUTION
        else:
            objective_value = _read_number(
                lambda: float(model.getObjVal()), _OBJECTIVE_VALUE_ERRORS
            )

    solver_time_seconds: float | MissingReason = _read_number(
        lambda: float(model.getSolvingTime())
    )
    node_count: float | MissingReason = _read_number(lambda: float(model.getNNodes()))
    stats: dict[str, float | MissingReason] = {"node_count": node_count}
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

    if has_solution is not False:
        _add_native_number(native, "gap", lambda: float(model.getGap()))

    metrics: ReportedMetrics = ReportedMetrics(
        has_objective=has_objective,
        sense=sense,
        objective_value=objective_value,
        dual_bound=dual_bound,
        solver_time_seconds=solver_time_seconds,
        stats=stats,
        native=native,
    )
    metadata: ReportedMetadata = _read_metadata(model)
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


def _read_metadata(model: Any) -> ReportedMetadata:
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

    parameters: tuple[SolverParameter, ...] = _read_parameters(model)

    return ReportedMetadata(
        solver="scip",
        api_version=api_version,
        solver_version=solver_version,
        problem=problem,
        parameters=parameters,
        wall_time_includes_model_build=False,
    )


def _read_parameters(model: Any) -> tuple[SolverParameter, ...]:
    """The PySCIPOpt parameters that differ from their defaults."""
    temporary_directory: str
    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            parameter_path: Path = Path(temporary_directory) / "changed_parameters.set"
            # NOTE: PySCIPOpt exposes changed-only parameter values through its parameter writer.
            model.writeParams(
                str(parameter_path), comments=False, onlychanged=True, verbose=False
            )
            parameter_lines: list[str] = parameter_path.read_text(
                encoding="utf-8"
            ).splitlines()
    except (MemoryError, OSError, ValueError):
        return ()

    parameters: list[SolverParameter] = []
    parameter_line: str
    for parameter_line in parameter_lines:
        stripped_line: str = parameter_line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue
        parameter_name: str
        separator: str
        parameter_value_text: str
        parameter_name, separator, parameter_value_text = stripped_line.partition("=")
        if not separator:
            continue
        parameter_name = parameter_name.strip()
        if not parameter_name:
            continue
        parameter_value: int | float | str | bool = _parse_parameter_value(
            parameter_value_text
        )
        if (
            parameter_name == "propagating/genvbounds/timingmask"
            and parameter_value == 0
        ):
            # NOTE: PySCIPOpt applies this model-construction default to every new model.
            continue
        parameters.append(
            SolverParameter(
                name=parameter_name,
                value=parameter_value,
                origin=ParameterOrigin.EXPLICIT,
            )
        )

    return tuple(parameters)


def _parse_parameter_value(parameter_value_text: str) -> int | float | str | bool:
    """A SCIP parameter value parsed from its changed-parameter file."""
    stripped_value: str = parameter_value_text.strip()
    normalized_value: str = stripped_value.lower()
    if normalized_value == "true":
        return True
    if normalized_value == "false":
        return False
    try:
        parsed_value: Any = ast.literal_eval(stripped_value)
    except (SyntaxError, ValueError):
        return stripped_value
    if isinstance(parsed_value, (bool, int, float, str)):
        return parsed_value
    return stripped_value


def _add_native_number(
    native: dict[str, float],
    name: str,
    reader: Callable[[], float],
) -> None:
    """Add one available SCIP-specific number to native metrics."""
    value: float | MissingReason = _read_number(reader)
    if isinstance(value, MissingReason):
        return
    native[name] = value


def _read_number(
    reader: Callable[[], float],
    read_errors: tuple[type[Exception], ...] = _PYSCIPOPT_READ_ERRORS,
) -> float | MissingReason:
    """A SCIP numeric value with its infinity sentinels normalized."""
    value: float | MissingReason = _read_value(
        reader, MissingReason.FAILED, read_errors
    )
    if isinstance(value, MissingReason):
        return value
    if value >= 1e20:
        return math.inf
    if value <= -1e20:
        return -math.inf
    return value


def _read_value[Value](
    reader: Callable[[], Value],
    unavailable: Value,
    read_errors: tuple[type[Exception], ...] = _PYSCIPOPT_READ_ERRORS,
) -> Value:
    """A solver value read without interrupting the remaining extraction."""
    try:
        return reader()
    except read_errors:
        return unavailable
