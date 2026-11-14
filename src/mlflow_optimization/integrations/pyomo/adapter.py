"""Observation hooks for Pyomo solver calls."""

import importlib
import inspect
import math
import pkgutil
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from functools import wraps
from typing import Any

from mlflow_optimization.common.patching import Patch, receiver_of
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
from pyomo.core.base.block import BlockData
from pyomo.opt.base.solvers import OptSolver

PARAMETER_CONTROL_NAMES: frozenset[str] = frozenset(
    {
        "keepfiles",
        "load_solution",
        "load_solutions",
        "logfile",
        "model",
        "options",
        "raise_exception_on_nonoptimal_result",
        "report_timing",
        "solver_io",
        "solver_options",
        "solnfile",
        "stream_solver",
        "suffixes",
        "symbolic_solver_labels",
        "tee",
        "timer",
        "working_dir",
    }
)


class PyomoAdapter:
    """The adapter for solve calls made through Pyomo."""

    api: str = "pyomo"

    def __init__(self) -> None:
        self._patches: list[Patch] = []
        self._engine_types: dict[type[Any], str] = {}

    def install(self, observer: SolveObserver) -> None:
        """Installation of Pyomo solve hooks."""
        if self._patches:
            return
        self._engine_types = {}
        owner_modules: tuple[tuple[str, tuple[str, ...]], ...] = (
            (
                "pyomo.contrib.appsi.base",
                ("LegacySolverInterface", "PersistentSolver"),
            ),
            (
                "pyomo.contrib.solver.common.base",
                ("LegacySolverWrapper", "SolverBase"),
            ),
            (
                "pyomo.solvers.plugins.solvers.direct_or_persistent_solver",
                ("DirectOrPersistentSolver",),
            ),
        )
        owners: list[type[Any]] = [OptSolver]
        for module_name, class_names in owner_modules:
            try:
                owner_module: Any = importlib.import_module(module_name)
            except (ImportError, NameError):
                continue
            for class_name in class_names:
                owner_class: Any | None = getattr(owner_module, class_name, None)
                if isinstance(owner_class, type):
                    owners.append(owner_class)
        try:
            appsi_solver_package: Any = importlib.import_module(
                "pyomo.contrib.appsi.solvers"
            )
        except Exception:
            appsi_solver_package = None
        if appsi_solver_package is not None:
            appsi_solver_modules: tuple[str, ...] = tuple(
                module_info.name
                for module_info in pkgutil.iter_modules(
                    appsi_solver_package.__path__,
                    f"{appsi_solver_package.__name__}.",
                )
                if module_info.name.rsplit(".", 1)[-1] != "tests"
            )
            for appsi_solver_module in appsi_solver_modules:
                try:
                    importlib.import_module(appsi_solver_module)
                except (ImportError, AttributeError, NameError):
                    continue
        specific_owners: list[type[Any]] = []
        optional_modules: tuple[tuple[str, str, str], ...] = (
            ("pyomo.solvers.plugins.solvers.gurobi_direct", "GurobiDirect", "gurobi"),
            (
                "pyomo.solvers.plugins.solvers.gurobi_persistent",
                "GurobiPersistent",
                "gurobi",
            ),
            ("pyomo.contrib.appsi.solvers.highs", "Highs", "highs"),
            ("pyomo.contrib.appsi.solvers.gurobi", "Gurobi", "gurobi"),
            ("pyomo.contrib.solver.solvers.highs", "Highs", "highs"),
            ("pyomo.contrib.solver.solvers.scip.scip_direct", "ScipDirect", "scip"),
            (
                "pyomo.contrib.solver.solvers.scip.scip_persistent",
                "ScipPersistent",
                "scip",
            ),
        )
        for module_name, class_name, engine_name in optional_modules:
            try:
                solver_module: Any = importlib.import_module(module_name)
                solver_class: Any = getattr(solver_module, class_name)
            except (ImportError, AttributeError, NameError):
                continue
            if isinstance(solver_class, type):
                specific_owners.append(solver_class)
                self._engine_types[solver_class] = engine_name
        native_engine_modules: tuple[tuple[str, str, str], ...] = (
            ("highspy.highs", "Highs", "highs"),
            ("gurobipy", "Model", "gurobi"),
            ("pyscipopt.scip", "Model", "scip"),
        )
        for module_name, class_name, engine_name in native_engine_modules:
            try:
                engine_module: Any = importlib.import_module(module_name)
                engine_type: Any = getattr(engine_module, class_name)
            except (ImportError, AttributeError):
                continue
            if isinstance(engine_type, type):
                self._engine_types[engine_type] = engine_name
        patches: list[Patch] = []
        seen: set[type[Any]] = set()
        solver_owners: list[type[Any]] = []
        for owner in owners:
            solver_owners.append(owner)
            solver_owners.extend(self._solve_overrides(owner))
        solver_owners.extend(specific_owners)
        for owner in specific_owners:
            solver_owners.extend(self._solve_overrides(owner))
        try:
            for owner in solver_owners:
                if owner in seen:
                    continue
                seen.add(owner)
                patch: Patch = Patch(owner, "solve")

                def factory(original: Callable[..., Any]) -> Callable[..., Any]:
                    original_signature: inspect.Signature = inspect.signature(original)

                    @wraps(original)
                    def wrapped(
                        *solve_arguments: Any,
                        **solve_keywords: Any,
                    ) -> Any:
                        solve_observer: SolveObserver = observer
                        solver: Any = receiver_of(
                            original,
                            solve_arguments,
                            solve_keywords,
                        )
                        if solver is None:
                            return original(*solve_arguments, **solve_keywords)
                        try:
                            bound_arguments: inspect.BoundArguments = (
                                original_signature.bind(
                                    *solve_arguments,
                                    **solve_keywords,
                                )
                            )
                        except TypeError:
                            return original(*solve_arguments, **solve_keywords)
                        model: Any | None = self._model_from_call(
                            solver,
                            bound_arguments,
                        )
                        solver_time_before: tuple[Any | None, float | None] | None = (
                            self._solver_time_snapshot(solver)
                        )

                        def extract(
                            result: Any | None,
                            error: BaseException | None,
                        ) -> SolveObservation:
                            return self._extract(
                                solver,
                                model,
                                result,
                                error,
                                solve_keywords,
                                bound_arguments.arguments,
                                solver_time_before,
                            )

                        return solve_observer.observe(
                            self.api,
                            lambda: original(
                                *solve_arguments,
                                **solve_keywords,
                            ),
                            extract,
                        )

                    return wrapped

                patch.install(factory)
                patches.append(patch)
        except (AttributeError, TypeError):
            for patch in reversed(patches):
                patch.uninstall()
            raise
        self._patches = patches

    def _solve_overrides(self, owner: type[Any]) -> list[type[Any]]:
        overrides: list[type[Any]] = []
        pending_owners: list[type[Any]] = list(owner.__subclasses__())
        while pending_owners:
            subclass: type[Any] = pending_owners.pop()
            pending_owners.extend(subclass.__subclasses__())
            if callable(vars(subclass).get("solve")):
                overrides.append(subclass)
        return overrides

    def _model_from_call(
        self,
        solver: Any,
        bound_arguments: inspect.BoundArguments,
    ) -> Any | None:
        for argument_value in bound_arguments.arguments.values():
            if self._is_pyomo_model(argument_value):
                return argument_value
            if isinstance(argument_value, Mapping):
                for nested_value in argument_value.values():
                    if self._is_pyomo_model(nested_value):
                        return nested_value
            if isinstance(argument_value, Sequence) and not isinstance(
                argument_value,
                (str, bytes),
            ):
                for nested_value in argument_value:
                    if self._is_pyomo_model(nested_value):
                        return nested_value
        for attribute_name in ("_pyomo_model", "_model"):
            attached_model: Any | None = self._read(solver, attribute_name)[0]
            if self._is_pyomo_model(attached_model):
                return attached_model
        return None

    def _is_pyomo_model(self, value: Any | None) -> bool:
        return isinstance(value, BlockData)

    def uninstall(self) -> None:
        """Restoration of Pyomo solve hooks."""
        for patch in reversed(self._patches):
            patch.uninstall()
        self._patches = []

    def _extract(
        self,
        solver: Any,
        model: Any | None,
        result: Any | None,
        error: BaseException | None,
        solve_keywords: Mapping[str, Any],
        bound_arguments: Mapping[str, Any],
        solver_time_before: tuple[Any | None, float | None] | None = None,
    ) -> SolveObservation:
        try:
            outcome: ReportedOutcome = self._outcome(solver, result, error)
            metrics: ReportedMetrics = self._metrics(
                solver,
                result,
                model,
                error,
                outcome,
                solver_time_before,
            )
            metadata: ReportedMetadata = self._metadata(
                solver,
                model,
                result,
                solve_keywords,
                bound_arguments,
                error,
            )
            return SolveObservation(self.api, outcome, metrics, metadata)
        except Exception as extraction_error:
            raise ObservationError(
                "Pyomo solve observation extraction failed."
            ) from extraction_error

    def _outcome(
        self,
        solver: Any,
        result: Any | None,
        error: BaseException | None,
    ) -> ReportedOutcome:
        if error is not None:
            termination: Termination = (
                Termination.INTERRUPTED
                if isinstance(error, KeyboardInterrupt)
                else Termination.ERROR
            )
            return ReportedOutcome(type(error).__name__, termination, None)
        status: Any | None = None
        native_termination: Any | None = None
        result_sources: tuple[Any, ...] = self._result_sources(solver, result)
        for result_source in result_sources:
            status_source: Any | None = self._read(result_source, "solver")[0]
            if status_source is not None and status is None:
                status = self._read(status_source, "status")[0]
            if status is None:
                status = self._read(result_source, "status")[0]
            if native_termination is None and status_source is not None:
                native_termination = self._read(status_source, "termination_condition")[
                    0
                ]
            if native_termination is None:
                native_termination = self._read(result_source, "termination_condition")[
                    0
                ]
        if status is None:
            status = self._read(result, "status")[0]
        raw_statuses: list[str] = []
        for native_value in (status, native_termination):
            native_name: str | None = self._native_name(native_value)
            if native_name is not None and native_name not in raw_statuses:
                raw_statuses.append(native_name)
        raw_status: str = " / ".join(raw_statuses) if raw_statuses else "unknown"
        termination = self._map_termination(
            native_termination if native_termination is not None else status
        )
        native_termination_name: str | None = self._native_name(native_termination)
        solution_status_name: str | None = self._native_name(
            self._first_value(result_sources, "solution_status")
        )
        if (
            native_termination_name is not None
            and native_termination_name.rsplit(".", 1)[-1].casefold()
            == "convergencecriteriasatisfied"
            and solution_status_name is not None
        ):
            normalized_solution_status: str = solution_status_name.rsplit(".", 1)[
                -1
            ].casefold()
            if normalized_solution_status == "optimal":
                termination = Termination.OPTIMAL
            elif normalized_solution_status == "feasible":
                termination = Termination.FEASIBLE
        native_solver: Any | None = self._read(solver, "_solver_model")[0]
        has_solution: bool | None = self._has_solution(
            result_sources,
            native_solver,
        )
        if (
            native_termination_name is not None
            and native_termination_name.rsplit(".", 1)[-1].casefold() == "other"
            and has_solution is True
        ):
            termination = Termination.FEASIBLE
        return ReportedOutcome(raw_status, termination, has_solution)

    def _map_termination(self, value: Any | None) -> Termination:
        native_name: str | None = self._native_name(value)
        if native_name is None:
            return Termination.UNKNOWN
        normalized_name: str = native_name.rsplit(".", 1)[-1].casefold()
        terminations: Mapping[str, Termination] = {
            "convergencecriteriasatisfied": Termination.UNKNOWN,
            "optimal": Termination.OPTIMAL,
            "globallyoptimal": Termination.OPTIMAL,
            "locallyoptimal": Termination.FEASIBLE,
            "feasible": Termination.FEASIBLE,
            "other": Termination.UNKNOWN,
            "infeasible": Termination.INFEASIBLE,
            "proveninfeasible": Termination.INFEASIBLE,
            "unbounded": Termination.UNBOUNDED,
            "infeasibleorunbounded": Termination.INFEASIBLE_OR_UNBOUNDED,
            "maxTimeLimit".casefold(): Termination.TIME_LIMIT,
            "timeLimit".casefold(): Termination.TIME_LIMIT,
            "time_limit": Termination.TIME_LIMIT,
            "maxIterations".casefold(): Termination.ITERATION_LIMIT,
            "iterationLimit".casefold(): Termination.ITERATION_LIMIT,
            "iteration_limit": Termination.ITERATION_LIMIT,
            "maxNodes".casefold(): Termination.NODE_LIMIT,
            "nodeLimit".casefold(): Termination.NODE_LIMIT,
            "node_limit": Termination.NODE_LIMIT,
            "maxSolutions".casefold(): Termination.SOLUTION_LIMIT,
            "solutionLimit".casefold(): Termination.SOLUTION_LIMIT,
            "solution_limit": Termination.SOLUTION_LIMIT,
            "maxEvaluations".casefold(): Termination.OTHER_LIMIT,
            "maxSubIterations".casefold(): Termination.OTHER_LIMIT,
            "maxMemory".casefold(): Termination.OTHER_LIMIT,
            "objectiveLimit".casefold(): Termination.OTHER_LIMIT,
            "minFunctionValue".casefold(): Termination.UNKNOWN,
            "minStepLength".casefold(): Termination.UNKNOWN,
            "intermediateNonInteger".casefold(): Termination.UNKNOWN,
            "locallyInfeasible".casefold(): Termination.UNKNOWN,
            "emptyModel".casefold(): Termination.UNKNOWN,
            "userinterrupt": Termination.INTERRUPTED,
            "resourceinterrupt": Termination.INTERRUPTED,
            "interrupted": Termination.INTERRUPTED,
            "solverfailure": Termination.ERROR,
            "internalsolvererror": Termination.ERROR,
            "invalidproblem": Termination.ERROR,
            "licensingproblems": Termination.ERROR,
            "error": Termination.ERROR,
            "unknown": Termination.UNKNOWN,
            "nosolution": Termination.UNKNOWN,
        }
        return terminations.get(normalized_name, Termination.UNKNOWN)

    def _has_solution(
        self,
        result_sources: Sequence[Any],
        native_solver: Any | None,
    ) -> bool | None:
        if not result_sources:
            return None
        for source in result_sources:
            explicit_value: Any | None = self._read(source, "has_solution")[0]
            if isinstance(explicit_value, bool):
                return explicit_value
            solution_status: Any | None = self._read(source, "solution_status")[0]
            solution_status_name: str | None = self._native_name(solution_status)
            if solution_status_name is not None:
                normalized_status: str = solution_status_name.rsplit(".", 1)[
                    -1
                ].casefold()
                if normalized_status == "nosolution":
                    return False
                if normalized_status in ("feasible", "optimal"):
                    return True
            solution_count: float | None = self._solution_count(source)
            if solution_count is not None:
                return solution_count > 0
        for source in result_sources:
            feasible_objective: Any | None
            objective_reason: MissingReason | None
            feasible_objective, objective_reason = self._read(
                source,
                "best_feasible_objective",
            )
            feasible_number: float | None = self._native_number(
                feasible_objective,
                native_solver,
            )
            if feasible_number is not None:
                return math.isfinite(feasible_number)
            if objective_reason is MissingReason.UNDEFINED:
                return False
            incumbent_objective: Any | None
            incumbent_reason: MissingReason | None
            incumbent_objective, incumbent_reason = self._read(
                source, "incumbent_objective"
            )
            incumbent_number: float | None = self._native_number(
                incumbent_objective,
                native_solver,
            )
            if incumbent_number is not None:
                return math.isfinite(incumbent_number)
            if incumbent_reason is MissingReason.UNDEFINED:
                return False
        for source in result_sources:
            solutions: Any | None = self._read(source, "solution")[0]
            if solutions is None:
                continue
            solution_count: int | None = None
            try:
                solution_count = len(solutions)
            except Exception:
                solution_count = None
            if solution_count is not None and solution_count > 0:
                return True
        return None

    def _metrics(
        self,
        solver: Any,
        result: Any | None,
        model: Any | None,
        error: BaseException | None,
        outcome: ReportedOutcome,
        solver_time_before: tuple[Any | None, float | None] | None,
    ) -> ReportedMetrics:
        size: dict[str, int] = self._model_size(model)
        objectives: tuple[Any, ...] | None = self._model_objects(model)
        has_objective: bool | None = (
            len(objectives) > 0 if objectives is not None else None
        )
        sense: ObjectiveSense | None = self._sense(objectives or ())
        result_sources: tuple[Any, ...] = (
            () if error is not None else self._result_sources(solver, result)
        )
        native_solver: Any | None = None
        if error is None and result is not None:
            native_solver = self._read(solver, "_solver_model")[0]
        has_solution: bool | None = outcome.has_solution
        objective_value: float | MissingReason = self._objective_value(
            result_sources,
            native_solver,
        )
        if objective_value is MissingReason.UNSUPPORTED:
            objective_value = self._solution_objective(result_sources, native_solver)
        if objective_value is MissingReason.UNSUPPORTED:
            objective_value = self._problem_objective(
                result_sources,
                sense,
                native_solver,
            )
        if has_solution is False and has_objective is not False:
            objective_value = MissingReason.NO_SOLUTION
        if objective_value is MissingReason.UNSUPPORTED and has_objective is False:
            objective_value = MissingReason.UNDEFINED
        dual_bound: float | MissingReason = self._dual_bound(
            result_sources,
            sense,
            outcome.termination,
            native_solver,
        )
        solver_time_seconds: float | MissingReason = self._solver_time(
            result_sources,
            native_solver,
            None if error is not None else self._engine(solver),
            solver_time_before,
        )
        statistics: dict[str, float | MissingReason] = self._statistics(result_sources)
        native_metrics: dict[str, float] = self._native_metrics(
            result_sources,
            native_solver,
            has_solution,
            size.get("num_integer_variables"),
            outcome.termination,
        )
        return ReportedMetrics(
            has_objective=has_objective,
            sense=sense,
            objective_value=objective_value,
            dual_bound=dual_bound,
            solver_time_seconds=solver_time_seconds,
            statistics=statistics,
            native=native_metrics,
        )

    def _metadata(
        self,
        solver: Any,
        model: Any | None,
        result: Any | None,
        solve_keywords: Mapping[str, Any],
        bound_arguments: Mapping[str, Any],
        error: BaseException | None,
    ) -> ReportedMetadata:
        problem: dict[str, int | float | str | bool] = self._model_size(model)
        if not problem and error is None:
            problem.update(self._result_size(result))
        parameters: tuple[SolverParameter, ...] = self._parameters(
            solver,
            solve_keywords,
            bound_arguments,
        )
        return ReportedMetadata(
            solver=self._engine(solver),
            api_version=self._api_version(),
            solver_version=self._solver_version(
                solver,
                None if error is not None else result,
            ),
            problem=problem,
            parameters=parameters,
            wall_time_includes_problem_build=True,
        )

    def _model_size(self, model: Any | None) -> dict[str, int]:
        if model is None:
            return {}
        try:
            variable_type: Any = importlib.import_module("pyomo.core.base.var").Var
            variables: tuple[Any, ...] = tuple(
                model.component_data_objects(variable_type, active=True)
            )
        except (ImportError, AttributeError):
            variables = ()
            variable_count_available: bool = False
        except Exception:
            variables = ()
            variable_count_available = False
        else:
            variable_count_available = True
        try:
            constraint_type: Any = importlib.import_module(
                "pyomo.core.base.constraint"
            ).Constraint
            constraints: tuple[Any, ...] = tuple(
                model.component_data_objects(constraint_type, active=True)
            )
        except (ImportError, AttributeError):
            constraints = ()
            constraint_count_available: bool = False
        except Exception:
            constraints = ()
            constraint_count_available = False
        else:
            constraint_count_available = True
        integer_count: int = 0
        integer_count_available: bool = variable_count_available
        if variable_count_available:
            for variable in variables:
                try:
                    is_integer: bool = bool(variable.is_integer())
                except (AttributeError, TypeError):
                    integer_count_available = False
                except Exception:
                    integer_count_available = False
                else:
                    integer_count += int(is_integer)
        problem: dict[str, int] = {}
        if variable_count_available:
            problem["num_variables"] = len(variables)
        if constraint_count_available:
            problem["num_constraints"] = len(constraints)
        if integer_count_available:
            problem["num_integer_variables"] = integer_count
        return problem

    def _model_objects(self, model: Any | None) -> tuple[Any, ...] | None:
        if model is None:
            return None
        try:
            component_type: Any = importlib.import_module(
                "pyomo.core.base.objective"
            ).Objective
            objects: tuple[Any, ...] = tuple(
                model.component_data_objects(component_type, active=True)
            )
        except Exception:
            return None
        return objects

    def _sense(self, objectives: Sequence[Any]) -> ObjectiveSense | None:
        if len(objectives) != 1:
            return None
        objective_sense: Any | None = self._read(objectives[0], "sense")[0]
        if objective_sense == 1:
            return ObjectiveSense.MINIMIZE
        if objective_sense == -1:
            return ObjectiveSense.MAXIMIZE
        return None

    def _objective_value(
        self,
        result_sources: Sequence[Any],
        native_solver: Any | None,
    ) -> float | MissingReason:
        for result_source in result_sources:
            for field_name in ("best_feasible_objective", "incumbent_objective"):
                value: Any | None = self._read(result_source, field_name)[0]
                number: float | None = self._native_number(value, native_solver)
                if number is not None and math.isfinite(number):
                    return number
        return MissingReason.UNSUPPORTED

    def _solution_objective(
        self,
        result_sources: Sequence[Any],
        native_solver: Any | None,
    ) -> float | MissingReason:
        for result_source in result_sources:
            solutions: Any | None = self._read(result_source, "solution")[0]
            if solutions is None:
                continue
            first_solution: Any | None = None
            try:
                if len(solutions) > 0:
                    first_solution = solutions[0]
            except Exception:
                first_solution = None
            if first_solution is None:
                continue
            objectives: Any | None = self._read(first_solution, "objective")[0]
            if isinstance(objectives, Mapping):
                if len(objectives) != 1:
                    continue
                objectives = next(iter(objectives.values()))
            number: float | None = self._native_number(objectives, native_solver)
            if number is not None and math.isfinite(number):
                return number
        return MissingReason.UNSUPPORTED

    def _problem_objective(
        self,
        result_sources: Sequence[Any],
        sense: ObjectiveSense | None,
        native_solver: Any | None,
    ) -> float | MissingReason:
        bound_name: str | None = (
            "Upper bound"
            if sense is ObjectiveSense.MINIMIZE
            else "Lower bound"
            if sense is ObjectiveSense.MAXIMIZE
            else None
        )
        if bound_name is None:
            return MissingReason.UNSUPPORTED
        value: Any | None = self._section_value(
            result_sources, "problem", (bound_name,)
        )[0]
        number: float | None = self._native_number(value, native_solver)
        if number is not None and math.isfinite(number):
            return number
        return MissingReason.UNSUPPORTED

    def _dual_bound(
        self,
        result_sources: Sequence[Any],
        sense: ObjectiveSense | None,
        termination: Termination,
        native_solver: Any | None,
    ) -> float | MissingReason:
        if termination in (
            Termination.INFEASIBLE,
            Termination.INFEASIBLE_OR_UNBOUNDED,
        ):
            return MissingReason.UNSUPPORTED
        for result_source in result_sources:
            for field_name in ("best_objective_bound", "objective_bound"):
                value: Any | None = self._read(result_source, field_name)[0]
                number: float | None = self._native_number(value, native_solver)
                if number is not None:
                    if math.isinf(number) and termination is not Termination.UNBOUNDED:
                        continue
                    return number
        bound_name: str | None = (
            "Lower bound"
            if sense is ObjectiveSense.MINIMIZE
            else "Upper bound"
            if sense is ObjectiveSense.MAXIMIZE
            else None
        )
        if bound_name is None:
            return MissingReason.UNSUPPORTED
        bound: Any | None = self._section_value(
            result_sources, "problem", (bound_name,)
        )[0]
        number: float | None = self._native_number(bound, native_solver)
        if (
            number is not None
            and math.isinf(number)
            and termination is not Termination.UNBOUNDED
        ):
            return MissingReason.UNSUPPORTED
        return number if number is not None else MissingReason.UNSUPPORTED

    def _solver_time_snapshot(
        self, solver: Any
    ) -> tuple[Any | None, float | None] | None:
        native_solver: Any | None
        native_reason: MissingReason | None
        native_solver, native_reason = self._read(solver, "_solver_model")
        if native_reason is MissingReason.FAILED:
            return None
        previous_result: Any | None = self._read(solver, "_last_results_object")[0]
        native_solver_time: float | None = self._method_number(
            native_solver,
            "getRunTime",
        )
        if native_solver_time is not None:
            return native_solver, native_solver_time
        if previous_result is None and native_solver is None:
            # NOTE: No native instance or cached result means this solver has no prior run.
            return None, 0.0
        engine: str | None = self._engine(solver)
        if engine != "highs":
            return native_solver, None
        previous_timing_information: Any | None = self._read(
            previous_result,
            "timing_info",
        )[0]
        previous_timing_value: Any | None = self._read(
            previous_timing_information,
            "highs_time",
        )[0]
        if isinstance(previous_timing_value, timedelta):
            return native_solver, float(previous_timing_value.total_seconds())
        previous_solver_time: float | None = self._number(previous_timing_value)
        return native_solver, previous_solver_time

    def _solver_time(
        self,
        result_sources: Sequence[Any],
        native_solver: Any | None,
        engine: str | None,
        solver_time_before: tuple[Any | None, float | None] | None,
    ) -> float | MissingReason:
        if engine == "highs":
            if solver_time_before is None:
                return MissingReason.UNSUPPORTED
            previous_solver: Any | None
            previous_time: float | None
            previous_solver, previous_time = solver_time_before
            if native_solver is not None and native_solver is not previous_solver:
                previous_time = 0.0
            elif native_solver is None and (
                previous_solver is not None or previous_time != 0.0
            ):
                return MissingReason.UNSUPPORTED
            if previous_time is None:
                return MissingReason.UNSUPPORTED
            cumulative_solver_time: float | None = self._method_number(
                native_solver,
                "getRunTime",
            )
            if cumulative_solver_time is None:
                for result_source in result_sources:
                    timing_information: Any | None = self._read(
                        result_source,
                        "timing_info",
                    )[0]
                    timing_value: Any | None = self._read(
                        timing_information,
                        "highs_time",
                    )[0]
                    if isinstance(timing_value, timedelta):
                        cumulative_solver_time = float(timing_value.total_seconds())
                    else:
                        cumulative_solver_time = self._number(timing_value)
                    if cumulative_solver_time is not None:
                        break
            if (
                cumulative_solver_time is None
                or cumulative_solver_time < previous_time
            ):
                return MissingReason.UNSUPPORTED
            return cumulative_solver_time - previous_time
        if engine == "gurobi":
            runtime: float | None = self._number(self._read(native_solver, "Runtime")[0])
            if runtime is not None:
                return runtime
        for result_source in result_sources:
            timing_information: Any | None = self._read(
                result_source,
                "timing_info",
            )[0]
            for owner, name in (
                (timing_information, "gurobi_time"),
                (timing_information, "scip_time"),
            ):
                value: Any | None = self._read(owner, name)[0]
                if value is None:
                    continue
                if isinstance(value, timedelta):
                    return float(value.total_seconds())
                number: float | None = self._number(value)
                if number is not None:
                    return number
        return MissingReason.UNSUPPORTED

    def _statistics(
        self,
        result_sources: Sequence[Any],
    ) -> dict[str, float | MissingReason]:
        common_paths: Mapping[str, tuple[tuple[str, ...], ...]] = {
            "node_count": (
                ("node_count",),
                ("NodeCount",),
                ("mip_node_count",),
                ("statistics", "node_count"),
                ("statistics", "branch_and_bound", "number_of_nodes"),
                ("extra_info", "NNodes"),
            ),
            "simplex_iterations": (
                ("simplex_iterations",),
                ("simplex_iteration_count",),
                ("IterCount",),
                ("statistics", "simplex_iterations"),
                ("statistics", "black_box", "simplex_iterations"),
            ),
            "barrier_iterations": (
                ("barrier_iterations",),
                ("ipm_iteration_count",),
                ("BarIterCount",),
                ("statistics", "barrier_iterations"),
                ("statistics", "black_box", "barrier_iterations"),
            ),
        }
        statistics: dict[str, float | MissingReason] = {}
        owners: list[Any | None] = list(result_sources)
        for result_source in result_sources:
            solver_result: Any | None = self._read(result_source, "solver")[0]
            if solver_result is not None:
                owners.append(solver_result)
        for statistic_name, paths in common_paths.items():
            value: float | None = self._first_number(owners, paths)
            if value is not None and value >= 0:
                statistics[statistic_name] = value
        return statistics

    def _native_metrics(
        self,
        result_sources: Sequence[Any],
        native_solver: Any | None,
        has_solution: bool | None,
        integer_variable_count: int | None,
        termination: Termination,
    ) -> dict[str, float]:
        native_paths: Mapping[str, tuple[tuple[str, ...], ...]] = {
            "reported_gap": (
                ("gap",),
                ("mip_gap",),
                ("MIPGap",),
                ("statistics", "gap"),
            ),
            "solution_count": (
                ("solution_count",),
                ("number_of_solutions",),
                ("SolCount",),
            ),
            "created_subproblems": (
                ("statistics", "branch_and_bound", "number_of_created_subproblems"),
            ),
        }
        native: dict[str, float] = {}
        owners: list[Any | None] = list(result_sources)
        for result_source in result_sources:
            solver_result: Any | None = self._read(result_source, "solver")[0]
            if solver_result is not None:
                owners.append(solver_result)
        for metric_name, paths in native_paths.items():
            if metric_name == "reported_gap" and (
                has_solution is not True
                or integer_variable_count is None
                or integer_variable_count == 0
                or termination
                in (
                    Termination.UNBOUNDED,
                    Termination.INFEASIBLE,
                    Termination.INFEASIBLE_OR_UNBOUNDED,
                )
            ):
                continue
            value: float | None = self._first_number(owners, paths)
            if metric_name == "reported_gap":
                value = self._native_number(value, native_solver)
            if value is not None:
                native[metric_name] = value
        for result_source in result_sources:
            solution_count: float | None = self._solution_count(result_source)
            if solution_count is not None:
                native["solution_count"] = solution_count
                break
        return native

    def _method_number(self, owner: Any | None, name: str) -> float | None:
        method: Any | None = self._read(owner, name)[0]
        if not callable(method):
            return None
        try:
            value: Any = method()
        except AttributeError:
            return None
        except Exception:
            return None
        return self._number(value)

    def _first_number(
        self,
        owners: Sequence[Any | None],
        paths: Sequence[tuple[str, ...]],
    ) -> float | None:
        for owner in owners:
            for path in paths:
                value: Any | None = owner
                for name in path:
                    value = self._read(value, name)[0]
                    if value is None:
                        break
                number: float | None = self._number(value)
                if number is not None:
                    return number
        return None

    def _result_sources(self, solver: Any, result: Any | None) -> tuple[Any, ...]:
        if result is None:
            return ()
        sources: list[Any] = []
        sources.append(result)
        cached_result: Any | None = self._read(solver, "_last_results_object")[0]
        if cached_result is not None and all(
            cached_result is not source for source in sources
        ):
            sources.append(cached_result)
        return tuple(sources)

    def _first_value(self, sources: Sequence[Any], field_name: str) -> Any | None:
        for source in sources:
            value: Any | None = self._read(source, field_name)[0]
            if value is not None:
                return value
        return None

    def _solution_count(self, result: Any) -> float | None:
        count_names: tuple[str, ...] = (
            "Number of solutions",
            "number of solutions",
            "solution_count",
            "number_of_solutions",
        )
        value: Any | None
        value_reason: MissingReason | None
        value, value_reason = self._section_value((result,), "problem", count_names)
        if value_reason is None:
            return self._number(value)
        value, value_reason = self._section_value((result,), "solution", count_names)
        if value_reason is None:
            return self._number(value)
        return None

    def _section_value(
        self,
        result_sources: Sequence[Any],
        section_name: str,
        value_names: Sequence[str],
    ) -> tuple[Any | None, MissingReason | None]:
        for result_source in result_sources:
            section: Any | None = self._read(result_source, section_name)[0]
            if section is None:
                continue
            section_items: tuple[Any, ...] = self._section_items(section)
            for section_item in section_items:
                for value_name in value_names:
                    value: Any | None
                    value_reason: MissingReason | None
                    value, value_reason = self._read_key(section_item, value_name)
                    if value_reason is None or value_reason is MissingReason.UNDEFINED:
                        return value, value_reason
        return None, MissingReason.UNSUPPORTED

    def _section_items(self, section: Any) -> tuple[Any, ...]:
        if isinstance(section, Mapping):
            return (section,)
        item_count: int | None = None
        try:
            item_count = len(section)
        except Exception:
            item_count = None
        if item_count is None:
            return ()
        items: list[Any] = []
        for index in range(item_count):
            item: Any | None = None
            try:
                item = section[index]
            except Exception:
                item = None
            if item is not None:
                items.append(item)
        return tuple(items)

    def _read_key(
        self,
        owner: Any,
        key: str,
    ) -> tuple[Any | None, MissingReason | None]:
        if not isinstance(owner, Mapping):
            return None, MissingReason.UNSUPPORTED
        try:
            if key not in owner:
                return None, MissingReason.UNSUPPORTED
            value: Any = owner[key]
        except (AttributeError, KeyError, TypeError):
            return None, MissingReason.FAILED
        except Exception:
            return None, MissingReason.FAILED
        if value is None:
            return None, MissingReason.UNDEFINED
        return value, None

    def _result_size(self, result: Any | None) -> dict[str, int | float | str | bool]:
        result_keys: Mapping[str, tuple[str, ...]] = {
            "num_variables": ("Number of variables", "number_of_variables"),
            "num_constraints": ("Number of constraints", "number_of_constraints"),
            "num_integer_variables": (
                "Number of integer variables",
                "number_of_integer_variables",
            ),
            "num_binary_variables": (
                "Number of binary variables",
                "number_of_binary_variables",
            ),
        }
        problem: dict[str, int | float | str | bool] = {}
        for common_key, result_names in result_keys.items():
            value: Any | None
            value_reason: MissingReason | None
            value, value_reason = self._section_value(
                (result,), "problem", result_names
            )
            if value_reason is not None:
                continue
            number: float | None = self._number(value)
            if number is not None and math.isfinite(number) and number >= 0:
                problem[common_key] = int(number)
        return problem

    def _parameters(
        self,
        solver: Any,
        solve_keywords: Mapping[str, Any],
        bound_arguments: Mapping[str, Any],
    ) -> tuple[SolverParameter, ...]:
        values: dict[str, int | float | str | bool] = {}
        # NOTE: Gurobi string parameters can contain credentials, hosts, paths, or job labels.
        include_string_values: bool = self._engine(solver) != "gurobi"
        options: Any | None = self._read(solver, "options")[0]
        self._add_parameter_mapping(
            values,
            options,
            include_string_values=include_string_values,
        )
        solver_options: Any | None = self._read(solver, "_solver_options")[0]
        self._add_parameter_mapping(
            values,
            solver_options,
            include_string_values=include_string_values,
        )
        configuration: Any | None = self._read(solver, "config")[0]
        configuration_template: Any | None = self._read(type(solver), "CONFIG")[0]
        if configuration_template is None and configuration is not None:
            try:
                configuration_template = type(configuration)()
            except TypeError:
                configuration_template = None
        user_values_member: Any | None = self._read(configuration, "user_values")[0]
        if callable(user_values_member):
            try:
                configuration_values: Any = tuple(user_values_member())
            except Exception:
                configuration_values = None
        else:
            configuration_values = user_values_member
        self._add_parameter_mapping(
            values,
            configuration_values,
            PARAMETER_CONTROL_NAMES,
            configuration_template,
            include_string_values,
        )
        for keyword_name in ("options", "solver_options"):
            self._add_parameter_mapping(
                values,
                solve_keywords.get(keyword_name),
                include_string_values=include_string_values,
            )
        self._add_parameter_mapping(
            values,
            solve_keywords,
            PARAMETER_CONTROL_NAMES,
            include_string_values=include_string_values,
        )
        for argument_name, argument_value in bound_arguments.items():
            if argument_value is solver or argument_name in (
                "model",
                "args",
                "kwargs",
                "kwds",
            ):
                continue
            if argument_name in ("options", "solver_options"):
                self._add_parameter_mapping(
                    values,
                    argument_value,
                    include_string_values=include_string_values,
                )
            elif argument_name not in PARAMETER_CONTROL_NAMES:
                self._add_parameter_mapping(
                    values,
                    ((argument_name, argument_value),),
                    include_string_values=include_string_values,
                )
        return tuple(
            SolverParameter(name, value, ParameterOrigin.EXPLICIT)
            for name, value in sorted(values.items())
        )

    def _add_parameter_mapping(
        self,
        target: dict[str, int | float | str | bool],
        values: Any | None,
        excluded_names: frozenset[str] = frozenset(),
        configuration_template: Any | None = None,
        include_string_values: bool = True,
    ) -> None:
        if isinstance(values, Mapping):
            parameter_items: Any = values.items()
        elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            parameter_items = values
        else:
            return
        for parameter_item in parameter_items:
            if (
                isinstance(parameter_item, Sequence)
                and not isinstance(
                    parameter_item,
                    (str, bytes),
                )
                and len(parameter_item) == 2
            ):
                name: Any = parameter_item[0]
                value: Any = parameter_item[1]
            else:
                name = self._read(parameter_item, "_name")[0]
                if name is None:
                    name = self._read(parameter_item, "name")[0]
                value = self._read(parameter_item, "value")[0]
                if callable(value):
                    try:
                        value = value(accessValue=False)
                    except Exception:
                        value = None
                default: Any | None = self._read(parameter_item, "_default")[0]
                is_user_set: Any | None = self._read(parameter_item, "_userSet")[0]
                template_default: Any | None
                template_reason: MissingReason | None
                template_default, template_reason = self._read(
                    configuration_template,
                    str(name),
                )
                same_as_template_default: bool = (
                    template_reason in (None, MissingReason.UNDEFINED)
                    and value == template_default
                )
                same_as_config_default: bool = (
                    default is not None and not callable(default) and value == default
                )
                if (
                    is_user_set is False
                    or same_as_template_default
                    or same_as_config_default
                ):
                    continue
            if (
                isinstance(value, (int, float, str, bool))
                and (include_string_values or not isinstance(value, str))
                and name is not None
                and str(name) not in excluded_names
            ):
                target[str(name)] = value

    def _engine(self, solver: Any) -> str | None:
        candidates: list[Any] = []
        for attribute_name in (
            "_solver_model",
            "_solver_instance",
            "_solver",
            "solver",
            "_opt",
        ):
            nested_solver: Any | None = self._read(solver, attribute_name)[0]
            if nested_solver is not None and not isinstance(
                nested_solver, (str, int, float, bool)
            ):
                candidates.append(nested_solver)
        candidates.append(solver)
        for candidate in candidates:
            for candidate_type in type(candidate).__mro__:
                engine: str | None = self._engine_types.get(candidate_type)
                if engine is not None:
                    return engine
        return None

    def _api_version(self) -> str | None:
        try:
            version_module: Any = importlib.import_module("pyomo.version")
        except ImportError:
            return None
        version: Any | None = self._read(version_module, "version")[0]
        return self._string(version)

    def _solver_version(self, solver: Any, result: Any | None) -> str | None:
        native_solver: Any | None = self._read(solver, "_solver_model")[0]
        engine: str | None = self._engine(solver)
        if native_solver is not None and engine == "scip":
            version_parts: tuple[float | None, ...] = tuple(
                self._method_number(native_solver, method_name)
                for method_name in (
                    "getMajorVersion",
                    "getMinorVersion",
                    "getTechVersion",
                )
            )
            if all(version_part is not None for version_part in version_parts):
                return ".".join(
                    str(int(version_part)) for version_part in version_parts
                )
        if native_solver is not None:
            for method_name in ("version", "getVersion"):
                version_member: Any | None = self._read(native_solver, method_name)[0]
                if callable(version_member):
                    native_version: Any | None = None
                    try:
                        native_version = version_member()
                    except Exception:
                        native_version = None
                    if native_version is not None:
                        version: str | None = self._string(native_version)
                        if version is not None:
                            return version
        if engine == "scip":
            return None
        solver_version: Any | None = self._read(solver, "_version")[0]
        if solver_version is not None:
            version: str | None = self._string(solver_version)
            if version is not None:
                return version
        engine_module: str | None = None
        if native_solver is not None:
            engine_module = type(native_solver).__module__.split(".", 1)[0]
        if engine_module is not None:
            try:
                module: Any = importlib.import_module(engine_module)
            except ImportError:
                module = None
            module_version: Any | None = self._read(module, "__version__")[0]
            if module_version is not None:
                version = self._string(module_version)
                if version is not None:
                    return version
        result_sources: tuple[Any, ...] = self._result_sources(solver, result)
        for result_source in result_sources:
            solver_result: Any | None = self._read(result_source, "solver")[0]
            for owner in (solver_result, result_source):
                for name in ("version", "solver_version"):
                    value: Any | None = self._read(owner, name)[0]
                    if value is None or callable(value):
                        continue
                    version = self._string(value)
                    if version is not None:
                        return version
        if engine == "highs":
            version_method: Any | None = self._read(solver, "version")[0]
            if callable(version_method):
                try:
                    version_value: Any = version_method()
                except Exception:
                    return None
                return self._string(version_value)
        return None

    def _read(
        self, owner: Any | None, name: str
    ) -> tuple[Any | None, MissingReason | None]:
        if owner is None:
            return None, MissingReason.UNSUPPORTED
        try:
            value: Any = getattr(owner, name)
        except AttributeError:
            return None, MissingReason.UNSUPPORTED
        except Exception:
            return None, MissingReason.FAILED
        if value is None:
            return None, MissingReason.UNDEFINED
        return value, None

    def _native_name(self, value: Any | None) -> str | None:
        if value is None:
            return None
        native_name: Any | None = self._read(value, "name")[0]
        if isinstance(native_name, str):
            return native_name
        if isinstance(value, str):
            return value
        try:
            return str(value)
        except (TypeError, ValueError):
            return None
        except Exception:
            return None

    def _string(self, value: Any | None) -> str | None:
        if value is None:
            return None
        try:
            return str(value)
        except (TypeError, ValueError):
            return None
        except Exception:
            return None

    def _number(self, value: Any | None) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            number: float = float(value)
        except Exception:
            return None
        return number

    def _native_number(
        self,
        value: Any | None,
        native_solver: Any | None,
    ) -> float | None:
        number: float | None = self._number(value)
        if number is None or math.isinf(number):
            return number
        solver_infinity: float | None = self._solver_infinity(native_solver)
        if (
            solver_infinity is not None
            and math.isfinite(solver_infinity)
            and abs(number) == abs(solver_infinity)
        ):
            return math.copysign(math.inf, number)
        return number

    def _solver_infinity(self, native_solver: Any | None) -> float | None:
        if native_solver is None:
            return None
        engine: str | None = self._engine(native_solver)
        if engine == "highs":
            return self._method_number(native_solver, "getInfinity")
        if engine == "scip":
            return self._method_number(native_solver, "infinity")
        if engine == "gurobi":
            try:
                engine_module: Any = importlib.import_module("gurobipy")
            except ImportError:
                return None
            except Exception:
                return None
            grb_namespace: Any | None = self._read(engine_module, "GRB")[0]
            infinity_value: Any | None = self._read(grb_namespace, "INFINITY")[0]
            return self._number(infinity_value)
        return None
