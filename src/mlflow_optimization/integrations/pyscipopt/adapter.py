"""The PySCIPOpt solver integration."""

import math
from collections.abc import Callable
from dataclasses import replace
from functools import wraps
from inspect import Signature, signature
from pathlib import Path
from types import ModuleType
from typing import Any
from weakref import WeakKeyDictionary

import pyscipopt
import pyscipopt.scip
from mlflow_optimization.common.diagnostics import warn
from mlflow_optimization.common.patching import (
    rebind_module_names,
    receiver_of,
    restore_module_names,
)
from mlflow_optimization.features.artifacts.types import (
    ArtifactRequest,
    ArtifactWorkspace,
)
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
from mlflow_optimization.features.progress.types import ProgressHook
from pyscipopt import Eventhdlr
from pyscipopt.scip import PY_SCIP_EVENTTYPE as SCIP_EVENTTYPE

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


class _ProgressEventHandler(Eventhdlr):
    """A PySCIPOpt event handler for solve progress."""

    def __init__(self) -> None:
        super().__init__()
        self.included: bool = False
        self.hook: ProgressHook | None = None

    def start(self, hook: ProgressHook) -> None:
        """The active hook for one solve."""
        self.hook = hook

    def stop(self) -> None:
        """The inactive state after a solve."""
        self.hook = None

    def eventinit(self) -> None:
        """The subscriptions for one active solve."""
        if self.hook is None:
            return
        try:
            self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
            self.model.catchEvent(SCIP_EVENTTYPE.DUALBOUNDIMPROVED, self)
        except Exception:  # noqa: BLE001
            # NOTE: Event registration failures must not interrupt SCIP.
            self.hook = None

    def eventexec(self, event: Any) -> None:
        """A callback for one PySCIPOpt progress event."""
        progress_hook: ProgressHook | None = self.hook
        if progress_hook is None:
            return
        try:
            model: Any = self.model
            # NOTE: Solutions found during presolve can be fake ones of an unbounded model.
            if int(model.getStage()) <= pyscipopt.scip.PY_SCIP_STAGE.PRESOLVING:
                return
            solver_infinity: float = float(model.infinity())
            incumbent: float | None = None
            solution_count: int = int(model.getNSols())
            if solution_count > 0:
                incumbent = _optional_progress_bound(
                    float(model.getPrimalbound()),
                    solver_infinity,
                )
            dual_bound: float | None = _optional_progress_bound(
                float(model.getDualbound()),
                solver_infinity,
            )
            progress_hook(incumbent, dual_bound)
        # NOTE: Progress callback failures must not interrupt SCIP.
        except Exception:  # noqa: BLE001,S110
            pass


class PyScipOptAdapter:
    """The PySCIPOpt solver integration."""

    api: str = "pyscipopt"

    def __init__(self) -> None:
        self._installed: bool = False
        self._observer: SolveObserver | None = None
        self._original_model: type[Any] | None = None
        self._logging_model: type[Any] | None = None
        self._rebound_names: list[tuple[ModuleType, str]] = []
        self._counter_snapshots: WeakKeyDictionary[
            Any,
            tuple[int | MissingReason, dict[str, float | MissingReason]],
        ] = WeakKeyDictionary()
        self._user_logfiles: WeakKeyDictionary[Any, str | None] = WeakKeyDictionary()

    def install(self, observer: SolveObserver) -> None:
        """A patch that replaces model references with an observing subclass."""
        if self._installed:
            self._observer = observer
            return

        original_model: type[Any] = pyscipopt.Model
        integration: PyScipOptAdapter = self
        default_parameters: dict[str, bool | int | float | str] | None = (
            _read_default_parameters(original_model)
        )

        class LoggingModel(original_model):
            """A PySCIPOpt model that observes optimization calls."""

        original_set_logfile: Callable[..., Any] = original_model.setLogfile

        @wraps(original_set_logfile)
        def remembering_set_logfile(model: Any, path: str | None) -> Any:
            result: Any = original_set_logfile(model, path)
            integration._user_logfiles[model] = path
            return result

        LoggingModel.setLogfile = remembering_set_logfile  # type: ignore[method-assign]

        entry_point_names: tuple[str, ...] = (
            "optimize",
            "optimizeNogil",
            "solveConcurrent",
        )
        for entry_point_name in entry_point_names:
            original_entry_point: Any = getattr(original_model, entry_point_name, None)
            if not callable(original_entry_point):
                continue
            observed_entry_point: Callable[..., Any] = _wrap_solve_entry_point(
                original_entry_point,
                integration,
                default_parameters,
                records_progress=entry_point_name == "optimize",
            )
            setattr(LoggingModel, entry_point_name, observed_entry_point)

        logging_model: type[Any] = LoggingModel
        rebound_names: list[tuple[ModuleType, str]] = rebind_module_names(
            original_model,
            logging_model,
        )

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
        restore_module_names(self._rebound_names, original_model, logging_model)
        rebind_module_names(logging_model, original_model)

        self._installed = False
        self._observer = None
        self._original_model = None
        self._logging_model = None
        self._rebound_names = []
        self._counter_snapshots.clear()
        self._user_logfiles.clear()


def _wrap_solve_entry_point(
    original_entry_point: Callable[..., Any],
    integration: PyScipOptAdapter,
    default_parameters: dict[str, bool | int | float | str] | None,
    records_progress: bool,
) -> Callable[..., Any]:
    """An observed wrapper for one original PySCIPOpt solve entry point."""
    original_signature: Signature = signature(original_entry_point)

    @wraps(original_entry_point)
    def observed_entry_point(
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        observer: SolveObserver | None = integration._observer
        model: Any = receiver_of(original_entry_point, args, kwargs)
        if model is None or observer is None:
            return original_entry_point(*args, **kwargs)
        original_signature.bind(*args, **kwargs)
        counter_snapshots: WeakKeyDictionary[
            Any,
            tuple[int | MissingReason, dict[str, float | MissingReason]],
        ] = integration._counter_snapshots
        stage_before_solve: int | MissingReason = _read_value(
            lambda: int(model.getStage()),
            MissingReason.FAILED,
        )
        counters_before_solve: dict[str, float | MissingReason] | None = (
            _counters_before_solve(counter_snapshots, model, stage_before_solve)
        )

        # NOTE: SCIP changes parameters while solving, so capture configured values first.
        parameters_before_solve: tuple[SolverParameter, ...] = _read_parameters(
            model,
            default_parameters,
        )

        recorded_artifacts: list[ArtifactRequest] = []
        recorded_diagnostics: dict[str, str] = {}

        def call(progress_hook: ProgressHook | None) -> Any:
            workspace: ArtifactWorkspace | None = observer.artifact_workspace()
            if workspace is None:
                return call_solver(progress_hook)
            finish_log: Callable[[], None] | None = _begin_artifacts(
                model,
                workspace,
                integration._user_logfiles.get(model),
                recorded_artifacts,
                recorded_diagnostics,
            )
            try:
                return call_solver(progress_hook)
            finally:
                if finish_log is not None:
                    finish_log()

        def call_solver(progress_hook: ProgressHook | None) -> Any:
            effective_progress_hook: ProgressHook | None = (
                progress_hook if records_progress else None
            )
            if (
                effective_progress_hook is not None
                and _read_has_objective(model) is False
            ):
                effective_progress_hook = None
            if effective_progress_hook is None:
                return original_entry_point(*args, **kwargs)
            progress_handler: _ProgressEventHandler | None = _include_progress_handler(
                model
            )
            if progress_handler is None:
                return original_entry_point(*args, **kwargs)
            progress_handler.start(effective_progress_hook)
            try:
                return original_entry_point(*args, **kwargs)
            finally:
                progress_handler.stop()

        def extract(
            result: Any | None, error: BaseException | None
        ) -> SolveObservation:
            try:
                counters_after_solve: dict[str, float | MissingReason] | None = (
                    _store_counters_after_solve(counter_snapshots, model, error)
                )
                observation: SolveObservation = _extract_observation(
                    model,
                    error,
                    parameters_before_solve,
                    counters_before_solve,
                    counters_after_solve,
                )
                workspace: ArtifactWorkspace | None = observer.artifact_workspace()
                if (
                    workspace is not None
                    and workspace.options.diagnose_infeasibility
                    and error is None
                    and observation.outcome.termination is Termination.INFEASIBLE
                ):
                    _diagnose_infeasibility(
                        model,
                        workspace,
                        recorded_artifacts,
                        recorded_diagnostics,
                    )
                return replace(
                    observation,
                    artifacts=tuple(recorded_artifacts),
                    diagnostics=recorded_diagnostics,
                )
            except Exception as extraction_error:
                raise ObservationError(
                    "Could not build a PySCIPOpt solve observation."
                ) from extraction_error

        return observer.observe(integration.api, call, extract)

    return observed_entry_point


def _counters_before_solve(
    counter_snapshots: WeakKeyDictionary[
        Any,
        tuple[int | MissingReason, dict[str, float | MissingReason]],
    ],
    model: Any,
    stage_before_solve: int | MissingReason,
) -> dict[str, float | MissingReason] | None:
    """The solve counters carried over to a solve, or unavailable ones when they cannot be trusted."""
    counter_snapshot: (
        tuple[
            int | MissingReason,
            dict[str, float | MissingReason],
        ]
        | None
    ) = None
    counter_cache_available: bool = True
    try:
        counter_snapshot = counter_snapshots.get(model)
    except Exception:  # noqa: BLE001
        counter_cache_available = False
    if not counter_cache_available or isinstance(
        stage_before_solve,
        MissingReason,
    ):
        return _unavailable_solve_counters()
    if counter_snapshot is None:
        if stage_before_solve != pyscipopt.scip.PY_SCIP_STAGE.PROBLEM:
            return _unavailable_solve_counters()
        return None
    previous_stage: int | MissingReason = counter_snapshot[0]
    if isinstance(previous_stage, MissingReason):
        return _unavailable_solve_counters()
    if stage_before_solve == previous_stage:
        return counter_snapshot[1]
    if stage_before_solve != pyscipopt.scip.PY_SCIP_STAGE.PROBLEM:
        return _unavailable_solve_counters()
    return None


def _store_counters_after_solve(
    counter_snapshots: WeakKeyDictionary[
        Any,
        tuple[int | MissingReason, dict[str, float | MissingReason]],
    ],
    model: Any,
    error: BaseException | None,
) -> dict[str, float | MissingReason] | None:
    """The solve counters after a solve, stored as the model snapshot."""
    if error is not None:
        try:
            counter_snapshots[model] = (
                MissingReason.FAILED,
                _unavailable_solve_counters(),
            )
        except Exception:  # noqa: BLE001
            counter_snapshots.clear()
        return None
    counters_after_solve: dict[str, float | MissingReason] = _read_solve_counters(model)
    stage_after_solve: int | MissingReason = _read_value(
        lambda: int(model.getStage()),
        MissingReason.FAILED,
    )
    try:
        counter_snapshots[model] = (stage_after_solve, counters_after_solve)
    except Exception:  # noqa: BLE001
        counter_snapshots.clear()
    return counters_after_solve


def _begin_artifacts(
    model: Any,
    workspace: ArtifactWorkspace,
    user_logfile: str | None,
    artifacts: list[ArtifactRequest],
    diagnostics: dict[str, str],
) -> Callable[[], None] | None:
    """The model artifact written now and the callback that finishes the log artifact."""
    if workspace.options.solver_model:
        model_path: Path = workspace.directory / "model.cip"
        try:
            model.writeProblem(str(model_path), trans=False, verbose=False)
            artifacts.append(ArtifactRequest(str(model_path), "model"))
            diagnostics["optimization.artifacts.solver_model"] = "true"
        except Exception as error:  # noqa: BLE001
            diagnostics["optimization.artifacts.solver_model"] = "failed"
            warn(f"Could not attach the PySCIPOpt model: {error}")
    if not workspace.options.solver_log:
        return None

    log_path: Path = workspace.directory / "log.txt"
    offset: int = 0
    try:
        if user_logfile is None:
            pyscipopt.Model.setLogfile(model, str(log_path))
        else:
            offset = Path(user_logfile).stat().st_size
    except Exception as error:  # noqa: BLE001
        diagnostics["optimization.artifacts.solver_log"] = "failed"
        warn(f"Could not attach the PySCIPOpt log: {error}")
        return None

    def finish_log() -> None:
        try:
            if user_logfile is None:
                pyscipopt.Model.setLogfile(model, None)
            else:
                with Path(user_logfile).open("rb") as user_file:
                    user_file.seek(offset)
                    log_path.write_bytes(user_file.read())
            artifacts.append(ArtifactRequest(str(log_path), "solver"))
            diagnostics["optimization.artifacts.solver_log"] = "true"
        except Exception as error:  # noqa: BLE001
            diagnostics["optimization.artifacts.solver_log"] = "failed"
            warn(f"Could not attach the PySCIPOpt log: {error}")

    return finish_log


def _diagnose_infeasibility(
    model: Any,
    workspace: ArtifactWorkspace,
    artifacts: list[ArtifactRequest],
    diagnostics: dict[str, str],
) -> None:
    """The irreducible infeasible subsystem of an infeasible model as an artifact."""
    iis_path: Path = workspace.directory / "iis.cip"
    try:
        iis: Any = model.generateIIS()
        iis.getSubscip().writeProblem(str(iis_path), trans=False, verbose=False)
        diagnostics["optimization.diagnostics.iis_minimal"] = (
            "true" if iis.isSubscipIrreducible() else "false"
        )
        artifacts.append(ArtifactRequest(str(iis_path), "diagnostics"))
        diagnostics["optimization.diagnostics.infeasibility"] = "iis"
    except Exception as error:  # noqa: BLE001
        diagnostics["optimization.diagnostics.infeasibility"] = "failed"
        warn(f"Could not compute the PySCIPOpt IIS: {error}")


def _include_progress_handler(model: Any) -> _ProgressEventHandler | None:
    """The included progress event handler for one model."""
    # NOTE: Progress handler installation must not change the solver call.
    try:
        progress_handler: Any = getattr(
            model,
            "_mlflow_optimization_progress_handler",
            None,
        )
        if not isinstance(progress_handler, _ProgressEventHandler):
            progress_handler = _ProgressEventHandler()
            model._mlflow_optimization_progress_handler = progress_handler
        if not progress_handler.included:
            model.includeEventhdlr(
                progress_handler,
                f"mlflow_optimization_progress_{id(model)}",
                "Progress events for one PySCIPOpt model",
            )
            progress_handler.included = True
    except Exception:  # noqa: BLE001
        return None
    return progress_handler


def _optional_progress_bound(value: float, solver_infinity: float) -> float | None:
    """A finite PySCIPOpt progress bound, if one is available."""
    if not math.isfinite(value) or abs(value) >= solver_infinity:
        return None
    return value


def _extract_observation(
    model: Any,
    error: BaseException | None,
    parameters_before_solve: tuple[SolverParameter, ...],
    counters_before_solve: dict[str, float | MissingReason] | None,
    counters_after_solve: dict[str, float | MissingReason] | None,
) -> SolveObservation:
    """The SDK-free observation of one PySCIPOpt optimization call."""
    if error is not None:
        error_termination: Termination = (
            Termination.INTERRUPTED
            if isinstance(error, KeyboardInterrupt)
            else Termination.ERROR
        )
        error_metrics: ReportedMetrics = ReportedMetrics(
            has_objective=None,
            sense=None,
            objective_value=MissingReason.FAILED,
            dual_bound=MissingReason.FAILED,
            solver_time_seconds=MissingReason.FAILED,
            statistics={},
            native={},
        )
        error_metadata: ReportedMetadata = _read_metadata(
            model, parameters_before_solve
        )
        error_outcome: ReportedOutcome = ReportedOutcome(
            raw_status="unknown",
            termination=error_termination,
            has_solution=None,
        )
        return SolveObservation(
            api="pyscipopt",
            outcome=error_outcome,
            metrics=error_metrics,
            metadata=error_metadata,
        )

    raw_status: str = _read_value(lambda: str(model.getStatus()), "unknown")
    has_solution: bool | None = _read_value(lambda: model.getNSols() > 0, None)
    termination: Termination = _STATUS_TERMINATIONS.get(
        raw_status.lower(), Termination.UNKNOWN
    )

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

    gap: float | MissingReason | None = None
    if termination is Termination.UNBOUNDED:
        # NOTE: SCIP reports a fake finite objective for unbounded models.
        has_solution = False
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
        if termination is Termination.UNBOUNDED:
            objective_value = MissingReason.UNBOUNDED
        elif has_solution is False:
            objective_value = MissingReason.NO_SOLUTION
        else:
            objective_value = _read_scip_number(
                lambda: float(model.getObjVal()),
                solver_infinity,
            )
        if has_solution is not False:
            gap = _read_scip_number(
                lambda: float(model.getGap()),
                solver_infinity,
            )

    assert counters_after_solve is not None
    solver_time_seconds: float | MissingReason = _counter_difference(
        counters_after_solve["solver_time_seconds"],
        counters_before_solve,
        "solver_time_seconds",
    )
    node_count: float | MissingReason = _counter_difference(
        counters_after_solve["node_count"],
        counters_before_solve,
        "node_count",
    )
    statistics: dict[str, float | MissingReason] = {"node_count": node_count}
    native: dict[str, float] = {}

    _add_native_counter(
        native,
        "solution_count",
        counters_after_solve["solution_count"],
        counters_before_solve,
    )
    _add_native_counter(
        native,
        "solutions_found",
        counters_after_solve["solutions_found"],
        counters_before_solve,
    )
    _add_native_counter(
        native,
        "lp_count",
        counters_after_solve["lp_count"],
        counters_before_solve,
    )
    _add_native_counter(
        native,
        "lp_iterations",
        counters_after_solve["lp_iterations"],
        counters_before_solve,
    )
    _add_native_counter(
        native,
        "total_node_count",
        counters_after_solve["total_node_count"],
        counters_before_solve,
    )
    _add_native_counter(
        native,
        "presolving_time_seconds",
        counters_after_solve["presolving_time_seconds"],
        counters_before_solve,
    )
    if gap is not None and not isinstance(gap, MissingReason):
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
    api_version: str | None = _read_value(
        lambda: str(
            pyscipopt.__version__  # pyright: ignore[reportPrivateImportUsage] -- Exposed at runtime.
        ),
        None,
    )
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

    presolved_variable_count: int | None = _read_value(
        lambda: int(model.getNVars(transformed=True)), None
    )
    presolved_constraint_count: int | None = _read_value(
        lambda: int(model.getNConss(transformed=True)), None
    )

    if presolved_variable_count is not None:
        problem["presolved_num_variables"] = presolved_variable_count
    if presolved_constraint_count is not None:
        problem["presolved_num_constraints"] = presolved_constraint_count
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
        for variable in variables:
            variable_type: str | MissingReason = _read_value(
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
    for parameter_name, default_value in default_parameters.items():
        parameter_value: Any | MissingReason = _read_value(
            lambda parameter_name=parameter_name: model.getParam(parameter_name),
            MissingReason.FAILED,
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


def _read_solve_counters(model: Any) -> dict[str, float | MissingReason]:
    """The current values of PySCIPOpt counters and timers."""
    return {
        "solver_time_seconds": _read_number(lambda: float(model.getSolvingTime())),
        "node_count": _read_number(lambda: float(model.getNNodes())),
        "solution_count": _read_number(lambda: float(model.getNSols())),
        "solutions_found": _read_number(lambda: float(model.getNSolsFound())),
        "lp_count": _read_number(lambda: float(model.getNLPs())),
        "lp_iterations": _read_number(lambda: float(model.getNLPIterations())),
        "total_node_count": _read_number(lambda: float(model.getNTotalNodes())),
        "presolving_time_seconds": _read_number(
            lambda: float(model.getPresolvingTime())
        ),
    }


def _unavailable_solve_counters() -> dict[str, float | MissingReason]:
    """Unavailable PySCIPOpt counters for a per-solve baseline."""
    return {
        "solver_time_seconds": MissingReason.FAILED,
        "node_count": MissingReason.FAILED,
        "solution_count": MissingReason.FAILED,
        "solutions_found": MissingReason.FAILED,
        "lp_count": MissingReason.FAILED,
        "lp_iterations": MissingReason.FAILED,
        "total_node_count": MissingReason.FAILED,
        "presolving_time_seconds": MissingReason.FAILED,
    }


def _add_native_counter(
    native: dict[str, float],
    metric_name: str,
    current_value: float | MissingReason,
    counters_before_solve: dict[str, float | MissingReason] | None,
) -> None:
    """One available PySCIPOpt counter difference in native metrics."""
    value: float | MissingReason = _counter_difference(
        current_value,
        counters_before_solve,
        metric_name,
    )
    if isinstance(value, MissingReason):
        return
    native[metric_name] = value


def _read_number(
    reader: Callable[[], float],
) -> float | MissingReason:
    """A guarded PySCIPOpt numeric value."""
    return _read_value(reader, MissingReason.FAILED)


def _counter_difference(
    current_value: float | MissingReason,
    counters_before_solve: dict[str, float | MissingReason] | None,
    counter_name: str,
) -> float | MissingReason:
    """The nonnegative difference between two PySCIPOpt counter values."""
    if isinstance(current_value, MissingReason):
        return current_value
    if counters_before_solve is None:
        if counter_name == "solution_count":
            return MissingReason.UNSUPPORTED
        return current_value
    previous_value: float | MissingReason = counters_before_solve.get(
        counter_name,
        MissingReason.FAILED,
    )
    if isinstance(previous_value, MissingReason):
        return previous_value
    difference: float = current_value - previous_value
    if difference < 0.0:
        return MissingReason.FAILED
    return difference


def _read_scip_number(
    reader: Callable[[], float],
    solver_infinity: float | MissingReason,
) -> float | MissingReason:
    """A PySCIPOpt numeric value with its infinity sentinel normalized."""
    value: float | MissingReason = _read_number(reader)
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
) -> Value:
    """A PySCIPOpt value or its fallback after a supported read error."""
    try:
        return reader()
    except Exception:  # noqa: BLE001
        return unavailable
