"""The MathOpt solver integration."""

import dataclasses
from collections.abc import Callable, Mapping
from datetime import timedelta
from enum import Enum
from functools import wraps
from inspect import Signature, signature
from typing import cast

import ortools
from ortools.math_opt.python import mathopt
from ortools.math_opt.python import solve as mathopt_solve
from ortools.math_opt.python.model import Model
from ortools.math_opt.python.parameters import SolveParameters, SolverType
from ortools.math_opt.python.result import SolveResult
from ortools.math_opt.python.solve import StatusNotOk

from mlflow_optimization.common.patching import Patch
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

ParameterValue = int | float | str | bool
_UNAVAILABLE: object = object()


class Integration:
    """The MathOpt integration hook."""

    api: str = "mathopt"

    def __init__(self) -> None:
        self._patches: list[tuple[Patch, str]] = [
            (Patch(mathopt, "solve"), "function"),
            (Patch(mathopt_solve, "solve"), "function"),
            (Patch(mathopt_solve.IncrementalSolver, "solve"), "incremental"),
        ]

    def install(self, observer: SolveObserver) -> None:
        """The installed MathOpt solve hooks."""
        installed_patches: list[Patch] = []
        patch: Patch
        entry_point: str
        try:
            for patch, entry_point in self._patches:

                def create_wrapper(
                    original: object,
                    current_entry_point: str = entry_point,
                ) -> object:
                    return self._wrap(original, observer, current_entry_point)

                patch.install(create_wrapper)
                installed_patches.append(patch)
        except (AttributeError, TypeError):
            patch: Patch
            for patch in reversed(installed_patches):
                patch.uninstall()
            raise

    def uninstall(self) -> None:
        """The restored MathOpt solve hooks."""
        patch: Patch
        for patch, _ in reversed(self._patches):
            patch.uninstall()

    def _wrap(
        self, original: object, observer: SolveObserver, entry_point: str
    ) -> object:
        original_solve: Callable[..., SolveResult] = cast(
            Callable[..., SolveResult], original
        )

        @wraps(original_solve)
        def wrapped_solve(
            *call_arguments: object, **call_keywords: object
        ) -> SolveResult:
            call_signature: Signature = signature(original_solve)
            call_signature.bind(*call_arguments, **call_keywords)

            def call() -> SolveResult:
                return original_solve(*call_arguments, **call_keywords)

            def extract(
                result: SolveResult | None,
                error: BaseException | None,
            ) -> SolveObservation:
                return self._extract(
                    result,
                    error,
                    entry_point,
                    call_arguments,
                    cast(dict[str, object], call_keywords),
                )

            return observer.observe(self.api, call, extract)

        return wrapped_solve

    def _extract(
        self,
        result: SolveResult | None,
        solve_error: BaseException | None,
        entry_point: str,
        call_arguments: tuple[object, ...],
        call_keywords: Mapping[str, object],
    ) -> SolveObservation:
        """The solver-independent snapshot of one MathOpt result."""
        try:
            solver_type: SolverType | None = self._solver_type(
                entry_point,
                call_arguments,
                call_keywords,
            )
            model: Model | None = self._model(
                entry_point, call_arguments, call_keywords
            )
            solve_parameters: SolveParameters | None = self._solve_parameters(
                call_keywords
            )
            outcome: ReportedOutcome = self._outcome(result, solve_error)
            metrics: ReportedMetrics = self._metrics(
                result,
                model,
                solver_type,
                solve_parameters,
                outcome.has_solution,
            )
            metadata: ReportedMetadata = self._metadata(
                solver_type, model, solve_parameters
            )
            return SolveObservation(self.api, outcome, metrics, metadata)
        except (
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            StatusNotOk,
            TypeError,
            ValueError,
        ) as error:
            raise ObservationError("MathOpt observation could not be built") from error

    def _solver_type(
        self,
        entry_point: str,
        call_arguments: tuple[object, ...],
        call_keywords: Mapping[str, object],
    ) -> SolverType | None:
        """The solver type supplied to a MathOpt entry point."""
        if entry_point == "incremental":
            incremental_solver: object = (
                call_arguments[0]
                if call_arguments
                else call_keywords.get("self", _UNAVAILABLE)
            )
            solver_type: object = self._guarded_attribute(
                incremental_solver, "_solver_type"
            )
            return solver_type if isinstance(solver_type, SolverType) else None
        solver_type: object = call_keywords.get("solver_type", _UNAVAILABLE)
        if solver_type is _UNAVAILABLE and len(call_arguments) > 1:
            solver_type = call_arguments[1]
        return solver_type if isinstance(solver_type, SolverType) else None

    def _model(
        self,
        entry_point: str,
        call_arguments: tuple[object, ...],
        call_keywords: Mapping[str, object],
    ) -> Model | None:
        """The model supplied to a MathOpt entry point."""
        if entry_point == "incremental":
            incremental_solver: object = (
                call_arguments[0]
                if call_arguments
                else call_keywords.get("self", _UNAVAILABLE)
            )
            model: object = self._guarded_attribute(incremental_solver, "_model")
            return model if isinstance(model, Model) else None
        model: object = call_keywords.get("opt_model", _UNAVAILABLE)
        if model is _UNAVAILABLE and call_arguments:
            model = call_arguments[0]
        return model if isinstance(model, Model) else None

    def _solve_parameters(
        self, call_keywords: Mapping[str, object]
    ) -> SolveParameters | None:
        """The explicit MathOpt solve parameters."""
        solve_parameters: object = call_keywords.get("params")
        return (
            solve_parameters if isinstance(solve_parameters, SolveParameters) else None
        )

    def _outcome(
        self,
        result: SolveResult | None,
        error: BaseException | None,
    ) -> ReportedOutcome:
        """The MathOpt termination status and primal solution availability."""
        if error is not None:
            termination: Termination = (
                Termination.INTERRUPTED
                if isinstance(error, KeyboardInterrupt)
                else Termination.ERROR
            )
            return ReportedOutcome(type(error).__name__, termination, None)
        if result is None:
            return ReportedOutcome("unknown", Termination.UNKNOWN, None)
        termination_value: object = self._guarded_attribute(result, "termination")
        reason: object = self._guarded_attribute(termination_value, "reason")
        limit: object = self._guarded_attribute(termination_value, "limit")
        reason_name: str | None = self._enum_name(reason)
        limit_name: str | None = self._enum_name(limit)
        raw_status: str = self._raw_status(reason, limit, reason_name, limit_name)
        termination: Termination = self._termination(reason_name, limit_name)
        has_solution: bool | None = self._has_solution(result)
        return ReportedOutcome(raw_status, termination, has_solution)

    def _has_solution(self, result: SolveResult) -> bool | None:
        """Whether a MathOpt result contains a primal feasible solution."""
        try:
            return result.has_primal_feasible_solution()
        except (AttributeError, IndexError, RuntimeError):
            return None

    def _raw_status(
        self,
        reason: object,
        limit: object,
        reason_name: str | None,
        limit_name: str | None,
    ) -> str:
        """The native MathOpt reason and limit names."""
        if reason_name is None:
            return "unknown"
        reason_value: object = self._guarded_attribute(reason, "value")
        raw_status: str = reason_name
        if isinstance(reason_value, int):
            raw_status = f"{raw_status} ({reason_value})"
        if limit_name is not None:
            limit_value: object = self._guarded_attribute(limit, "value")
            raw_status = f"{raw_status}; limit={limit_name}"
            if isinstance(limit_value, int):
                raw_status = f"{raw_status} ({limit_value})"
        return raw_status

    def _termination(
        self, reason_name: str | None, limit_name: str | None
    ) -> Termination:
        """The solver-independent termination of a MathOpt result."""
        limit_terminations: dict[str, Termination] = {
            "UNDETERMINED": Termination.OTHER_LIMIT,
            "ITERATION": Termination.ITERATION_LIMIT,
            "TIME": Termination.TIME_LIMIT,
            "NODE": Termination.NODE_LIMIT,
            "SOLUTION": Termination.SOLUTION_LIMIT,
            "MEMORY": Termination.OTHER_LIMIT,
            "OBJECTIVE": Termination.OTHER_LIMIT,
            "NORM": Termination.OTHER_LIMIT,
            "INTERRUPTED": Termination.INTERRUPTED,
            "SLOW_PROGRESS": Termination.OTHER_LIMIT,
            "OTHER": Termination.OTHER_LIMIT,
        }
        if reason_name in {"FEASIBLE", "NO_SOLUTION_FOUND"} and limit_name is not None:
            return limit_terminations.get(limit_name, Termination.UNKNOWN)
        reason_terminations: dict[str, Termination] = {
            "OPTIMAL": Termination.OPTIMAL,
            "INFEASIBLE": Termination.INFEASIBLE,
            "UNBOUNDED": Termination.UNBOUNDED,
            "INFEASIBLE_OR_UNBOUNDED": Termination.INFEASIBLE_OR_UNBOUNDED,
            "IMPRECISE": Termination.UNKNOWN,
            "FEASIBLE": Termination.FEASIBLE,
            "NO_SOLUTION_FOUND": Termination.UNKNOWN,
            "NUMERICAL_ERROR": Termination.ERROR,
            "OTHER_ERROR": Termination.ERROR,
        }
        return reason_terminations.get(reason_name or "", Termination.UNKNOWN)

    def _metrics(
        self,
        result: SolveResult | None,
        model: Model | None,
        solver_type: SolverType | None,
        solve_parameters: SolveParameters | None,
        has_solution: bool | None,
    ) -> ReportedMetrics:
        """The objective, bound, time, and statistics of a MathOpt result."""
        has_objective: bool | None = self._has_objective(model)
        sense: ObjectiveSense | None = self._objective_sense(model, has_objective)
        if result is None:
            objective_value: float | MissingReason = MissingReason.UNSUPPORTED
            dual_bound: float | MissingReason = MissingReason.UNSUPPORTED
            solver_time_seconds: float | MissingReason = MissingReason.UNSUPPORTED
        else:
            objective_value = self._objective_value(result, has_solution)
            dual_bound = self._numeric_result_value(result, "dual_bound")
            solver_time_seconds = self._solver_time(result)
        statistics: dict[str, float | MissingReason] = self._statistics(
            result,
            model,
            solver_type,
            solve_parameters,
        )
        native: dict[str, float] = self._native_statistics(result, solver_type)
        return ReportedMetrics(
            has_objective=has_objective,
            sense=sense,
            objective_value=objective_value,
            dual_bound=dual_bound,
            solver_time_seconds=solver_time_seconds,
            statistics=statistics,
            native=native,
        )

    def _has_objective(self, model: Model | None) -> bool | None:
        """Whether a MathOpt objective has nonzero terms or an offset."""
        if model is None:
            return None
        objective: object = self._guarded_attribute(model, "objective")
        try:
            linear_terms: object = tuple(objective.linear_terms())
        except (AttributeError, RuntimeError, StatusNotOk, TypeError):
            linear_terms = _UNAVAILABLE
        try:
            quadratic_terms: object = tuple(objective.quadratic_terms())
        except (AttributeError, RuntimeError, StatusNotOk, TypeError):
            quadratic_terms = _UNAVAILABLE
        offset: object = self._guarded_attribute(objective, "offset")
        if not isinstance(linear_terms, tuple) or not isinstance(
            quadratic_terms, tuple
        ):
            return None
        if not isinstance(offset, (int, float)):
            return None
        return bool(linear_terms or quadratic_terms or offset != 0.0)

    def _objective_sense(
        self,
        model: Model | None,
        has_objective: bool | None,
    ) -> ObjectiveSense | None:
        """The direction of a defined MathOpt objective."""
        if model is None or has_objective is not True:
            return None
        objective: object = self._guarded_attribute(model, "objective")
        is_maximize: object = self._guarded_attribute(objective, "is_maximize")
        if not isinstance(is_maximize, bool):
            return None
        return ObjectiveSense.MAXIMIZE if is_maximize else ObjectiveSense.MINIMIZE

    def _objective_value(
        self,
        result: SolveResult,
        has_solution: bool | None,
    ) -> float | MissingReason:
        """The best primal feasible objective value in a MathOpt result."""
        if has_solution is False:
            return MissingReason.NO_SOLUTION
        try:
            value: object = result.objective_value()
        except (AttributeError, AssertionError):
            return MissingReason.UNSUPPORTED
        except ValueError:
            return MissingReason.FAILED
        if isinstance(value, (int, float)):
            return float(value)
        return MissingReason.FAILED

    def _numeric_result_value(
        self,
        result: SolveResult,
        result_method_name: str,
    ) -> float | MissingReason:
        """One numeric value from a MathOpt result."""
        result_method: object = self._guarded_attribute(result, result_method_name)
        if not callable(result_method):
            return MissingReason.UNSUPPORTED
        try:
            value: object = result_method()
        except (AttributeError, RuntimeError):
            return MissingReason.FAILED
        if isinstance(value, (int, float)):
            return float(value)
        return MissingReason.FAILED

    def _solver_time(self, result: SolveResult) -> float | MissingReason:
        """The solve time reported by MathOpt."""
        solve_time_method: object = self._guarded_attribute(result, "solve_time")
        if not callable(solve_time_method):
            return MissingReason.UNSUPPORTED
        try:
            duration: object = solve_time_method()
        except (AttributeError, RuntimeError):
            return MissingReason.FAILED
        if isinstance(duration, timedelta):
            return duration.total_seconds()
        return MissingReason.FAILED

    def _statistics(
        self,
        result: SolveResult | None,
        model: Model | None,
        solver_type: SolverType | None,
        solve_parameters: SolveParameters | None,
    ) -> dict[str, float | MissingReason]:
        """The common search statistics reported by MathOpt."""
        statistics: dict[str, float | MissingReason] = {
            "node_count": MissingReason.UNSUPPORTED,
            "simplex_iterations": MissingReason.UNSUPPORTED,
            "barrier_iterations": MissingReason.UNSUPPORTED,
        }
        if result is None:
            return statistics
        supported_statistics: set[str] = set()
        if solver_type is SolverType.GLOP:
            supported_statistics.add("simplex_iterations")
        elif solver_type is SolverType.GSCIP:
            supported_statistics.add("node_count")
            if self._gscip_iteration_statistics_available(result):
                supported_statistics.update(
                    ("simplex_iterations", "barrier_iterations")
                )
        elif solver_type is SolverType.HIGHS:
            has_integer_variables: bool | None = self._has_integer_variables(model)
            solves_relaxation: bool | None = self._highs_solves_relaxation(
                solve_parameters
            )
            if solves_relaxation is True or (
                solves_relaxation is False and has_integer_variables is False
            ):
                supported_statistics.update(
                    ("simplex_iterations", "barrier_iterations")
                )
            elif solves_relaxation is False and has_integer_variables is True:
                supported_statistics.add("node_count")
        solve_stats: object = self._guarded_attribute(result, "solve_stats")
        statistic_name: str
        for statistic_name in statistics:
            if statistic_name not in supported_statistics:
                continue
            value: object = self._guarded_attribute(solve_stats, statistic_name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                statistics[statistic_name] = float(value)
            elif value is _UNAVAILABLE:
                statistics[statistic_name] = MissingReason.FAILED
        return statistics

    def _gscip_iteration_statistics_available(self, result: SolveResult) -> bool:
        """Whether GSCIP reports post-presolve iteration statistics."""
        termination: object = self._guarded_attribute(result, "termination")
        limit: object = self._guarded_attribute(termination, "limit")
        if limit is _UNAVAILABLE:
            return False
        output: object = self._guarded_attribute(result, "gscip_specific_output")
        solve_stats: object = self._guarded_attribute(output, "stats")
        list_fields_method: object = self._guarded_attribute(solve_stats, "ListFields")
        if not callable(list_fields_method):
            return False
        try:
            fields: object = list_fields_method()
        except (AttributeError, RuntimeError):
            return False
        if not isinstance(fields, list):
            return False
        statistic_names: frozenset[str] = frozenset(
            (
                "node_count",
                "total_lp_iterations",
                "primal_simplex_iterations",
                "dual_simplex_iterations",
                "barrier_iterations",
                "deterministic_time",
            )
        )
        field_descriptor: object
        field_value: object
        for field_descriptor, field_value in cast(list[tuple[object, object]], fields):
            field_name: object = self._guarded_attribute(field_descriptor, "name")
            if (
                isinstance(field_name, str)
                and field_name in statistic_names
                and isinstance(field_value, (int, float))
                and not isinstance(field_value, bool)
                and field_value != 0
            ):
                return True
        return limit is None

    def _has_integer_variables(self, model: Model | None) -> bool | None:
        """Whether a MathOpt model contains integer variables."""
        if model is None:
            return None
        variables_method: object = self._guarded_attribute(model, "variables")
        if not callable(variables_method):
            return None
        try:
            variables: tuple[object, ...] = tuple(variables_method())
        except (AttributeError, RuntimeError, StatusNotOk, TypeError):
            return None
        variable: object
        for variable in variables:
            integer_flag: object = self._guarded_attribute(variable, "integer")
            if not isinstance(integer_flag, bool):
                return None
            if integer_flag:
                return True
        return False

    def _highs_solves_relaxation(
        self, solve_parameters: SolveParameters | None
    ) -> bool | None:
        """Whether HiGHS is configured to solve the relaxation of a MIP."""
        if solve_parameters is None:
            return False
        highs_parameters: object = self._guarded_attribute(
            solve_parameters,
            "highs",
        )
        boolean_options: object = self._guarded_attribute(
            highs_parameters,
            "bool_options",
        )
        if not isinstance(boolean_options, Mapping):
            return None
        try:
            solves_relaxation: object = boolean_options.get("solve_relaxation", False)
        except (KeyError, TypeError):
            return None
        return solves_relaxation if isinstance(solves_relaxation, bool) else None

    def _native_statistics(
        self,
        result: SolveResult | None,
        solver_type: SolverType | None,
    ) -> dict[str, float]:
        """The MathOpt backend-specific numeric statistics."""
        native: dict[str, float] = {}
        if result is None:
            return native
        solve_stats: object = self._guarded_attribute(result, "solve_stats")
        if solver_type is SolverType.PDLP:
            first_order_iterations: object = self._guarded_attribute(
                solve_stats,
                "first_order_iterations",
            )
            if isinstance(first_order_iterations, (int, float)) and not isinstance(
                first_order_iterations,
                bool,
            ):
                native["first_order_iterations"] = float(first_order_iterations)
        output_name: str
        for output_name in (
            "gscip_specific_output",
            "osqp_specific_output",
            "pdlp_specific_output",
        ):
            output: object = self._guarded_attribute(result, output_name)
            if output is not None and output is not _UNAVAILABLE:
                self._collect_native_values(
                    output_name.removesuffix("_specific_output"), output, native
                )
        return native

    def _collect_native_values(
        self,
        prefix: str,
        message: object,
        native: dict[str, float],
    ) -> None:
        """The numeric leaves in a MathOpt backend output message."""
        list_fields_method: object = self._guarded_attribute(message, "ListFields")
        if not callable(list_fields_method):
            return
        try:
            fields: object = list_fields_method()
        except (AttributeError, RuntimeError):
            return
        if not isinstance(fields, list):
            return
        field_descriptor: object
        field_value: object
        for field_descriptor, field_value in fields:
            field_name: object = self._guarded_attribute(field_descriptor, "name")
            enum_descriptor: object = self._guarded_attribute(
                field_descriptor, "enum_type"
            )
            if not isinstance(field_name, str):
                continue
            if enum_descriptor is not None and enum_descriptor is not _UNAVAILABLE:
                continue
            if self._is_message(field_value):
                self._collect_native_values(
                    f"{prefix}.{field_name}", field_value, native
                )
            elif isinstance(field_value, (int, float)) and not isinstance(
                field_value, bool
            ):
                native_name: str = f"{prefix}.{field_name}"
                native[native_name] = float(field_value)

    def _metadata(
        self,
        solver_type: SolverType | None,
        model: Model | None,
        solve_parameters: SolveParameters | None,
    ) -> ReportedMetadata:
        """The engine, version, problem size, and explicit parameters of a MathOpt solve."""
        solver_names: dict[str, str] = {
            "GSCIP": "scip",
            "GUROBI": "gurobi",
            "GLOP": "glop",
            "CP_SAT": "cp_sat",
            "PDLP": "pdlp",
            "GLPK": "glpk",
            "OSQP": "osqp",
            "ECOS": "ecos",
            "SCS": "scs",
            "HIGHS": "highs",
            "SANTORINI": "santorini",
        }
        solver_name: str | None = None
        solver_type_name: str | None = self._enum_name(solver_type)
        if solver_type_name is not None:
            solver_name = solver_names.get(solver_type_name)
        problem: dict[str, int | float | str | bool] = self._problem_size(model)
        parameters: list[SolverParameter] = self._parameters(solve_parameters)
        api_version: object = self._guarded_attribute(ortools, "__version__")
        return ReportedMetadata(
            solver=solver_name,
            api_version=api_version if isinstance(api_version, str) else None,
            solver_version=None,
            problem=problem,
            parameters=parameters,
            wall_time_includes_problem_build=False,
        )

    def _problem_size(self, model: Model | None) -> dict[str, int | float | str | bool]:
        """The dimensions of a MathOpt model."""
        problem: dict[str, int | float | str | bool] = {}
        if model is None:
            return problem
        count_methods: tuple[tuple[str, str], ...] = (
            ("num_variables", "get_num_variables"),
            ("num_linear_constraints", "get_num_linear_constraints"),
            ("num_quadratic_constraints", "get_num_quadratic_constraints"),
            ("num_indicator_constraints", "get_num_indicator_constraints"),
        )
        constraint_counts: list[int] = []
        problem_name: str
        method_name: str
        for problem_name, method_name in count_methods:
            count_method: object = self._guarded_attribute(model, method_name)
            if not callable(count_method):
                continue
            try:
                count: object = count_method()
            except (AttributeError, RuntimeError, StatusNotOk):
                continue
            if isinstance(count, int) and not isinstance(count, bool):
                problem[problem_name] = count
                if problem_name != "num_variables":
                    constraint_counts.append(count)
        if len(constraint_counts) == 3:
            problem["num_constraints"] = sum(constraint_counts)
        variables_method: object = self._guarded_attribute(model, "variables")
        if not callable(variables_method):
            return problem
        try:
            variables: object = tuple(variables_method())
        except (AttributeError, RuntimeError, StatusNotOk, TypeError):
            return problem
        if isinstance(variables, tuple):
            integer_count: int = 0
            all_integer_flags_available: bool = True
            variable: object
            for variable in variables:
                integer_flag: object = self._guarded_attribute(variable, "integer")
                if isinstance(integer_flag, bool):
                    integer_count += int(integer_flag)
                else:
                    all_integer_flags_available = False
            if all_integer_flags_available:
                problem["num_integer_variables"] = integer_count
        return problem

    def _parameters(
        self, solve_parameters: SolveParameters | None
    ) -> list[SolverParameter]:
        """The solve parameters that differ from MathOpt defaults."""
        parameters: list[SolverParameter] = []
        if solve_parameters is None:
            return parameters
        default_parameters: SolveParameters = SolveParameters()
        parameter_field: object
        for parameter_field in dataclasses.fields(SolveParameters):
            parameter_name: str = parameter_field.name
            parameter_value: object = self._guarded_attribute(
                solve_parameters, parameter_name
            )
            default_value: object = self._guarded_attribute(
                default_parameters, parameter_name
            )
            if parameter_value is _UNAVAILABLE or default_value is _UNAVAILABLE:
                continue
            if parameter_value == default_value or parameter_value is None:
                continue
            self._append_parameters(
                parameter_name, parameter_value, default_value, parameters
            )
        return parameters

    def _append_parameters(
        self,
        parameter_name: str,
        parameter_value: object,
        default_value: object,
        parameters: list[SolverParameter],
    ) -> None:
        """The differing nested MathOpt parameter values."""
        if parameter_value is None or parameter_value is _UNAVAILABLE:
            return
        if parameter_value == default_value:
            return
        if isinstance(parameter_value, Mapping):
            default_mapping: Mapping[object, object] = (
                default_value if isinstance(default_value, Mapping) else {}
            )
            nested_name: object
            nested_value: object
            for nested_name, nested_value in parameter_value.items():
                if not isinstance(nested_name, str):
                    continue
                nested_default: object = default_mapping.get(nested_name, _UNAVAILABLE)
                self._append_parameters(
                    f"{parameter_name}.{nested_name}",
                    nested_value,
                    nested_default,
                    parameters,
                )
            return
        if dataclasses.is_dataclass(parameter_value) and not isinstance(
            parameter_value, type
        ):
            default_object: object = default_value
            parameter_field: object
            for parameter_field in dataclasses.fields(parameter_value):
                nested_name: str = parameter_field.name
                nested_value: object = self._guarded_attribute(
                    parameter_value, nested_name
                )
                nested_default: object = self._guarded_attribute(
                    default_object, nested_name
                )
                self._append_parameters(
                    f"{parameter_name}.{nested_name}",
                    nested_value,
                    nested_default,
                    parameters,
                )
            return
        if self._is_message(parameter_value):
            list_fields_method: object = self._guarded_attribute(
                parameter_value, "ListFields"
            )
            if not callable(list_fields_method):
                return
            try:
                fields: object = list_fields_method()
            except (AttributeError, RuntimeError, StatusNotOk):
                return
            if not isinstance(fields, list):
                return
            field_descriptor: object
            field_value: object
            for field_descriptor, field_value in fields:
                field_name: object = self._guarded_attribute(field_descriptor, "name")
                enum_descriptor: object = self._guarded_attribute(
                    field_descriptor, "enum_type"
                )
                if not isinstance(field_name, str):
                    continue
                if enum_descriptor is not None and enum_descriptor is not _UNAVAILABLE:
                    enum_values: object = self._guarded_attribute(
                        enum_descriptor,
                        "values_by_number",
                    )
                    if not isinstance(enum_values, Mapping):
                        continue
                    try:
                        enum_value: object = enum_values[field_value]
                    except KeyError:
                        continue
                    enum_name: object = self._guarded_attribute(enum_value, "name")
                    if isinstance(enum_name, str):
                        parameters.append(
                            SolverParameter(
                                f"{parameter_name}.{field_name}",
                                enum_name,
                                ParameterOrigin.EXPLICIT,
                            )
                        )
                else:
                    self._append_parameters(
                        f"{parameter_name}.{field_name}",
                        field_value,
                        _UNAVAILABLE,
                        parameters,
                    )
            return
        scalar: ParameterValue | None = self._parameter_value(parameter_value)
        if scalar is not None:
            parameters.append(
                SolverParameter(parameter_name, scalar, ParameterOrigin.EXPLICIT)
            )
            return
        serialized_value: str = str(parameter_value)
        if isinstance(serialized_value, str):
            parameters.append(
                SolverParameter(
                    parameter_name, serialized_value, ParameterOrigin.EXPLICIT
                )
            )

    def _parameter_value(self, parameter_value: object) -> ParameterValue | None:
        """The scalar form of a MathOpt parameter."""
        if isinstance(parameter_value, bool):
            return parameter_value
        if isinstance(parameter_value, (int, float, str)):
            return parameter_value
        if isinstance(parameter_value, timedelta):
            return parameter_value.total_seconds()
        if isinstance(parameter_value, Enum):
            enum_name: object = self._guarded_attribute(parameter_value, "name")
            return enum_name if isinstance(enum_name, str) else None
        return None

    def _is_message(self, candidate: object) -> bool:
        """Whether a value is a protobuf message."""
        list_fields_method: object = self._guarded_attribute(candidate, "ListFields")
        return callable(list_fields_method)

    def _enum_name(self, candidate: object) -> str | None:
        """The name of a known MathOpt enum value."""
        enum_name: object = self._guarded_attribute(candidate, "name")
        return enum_name if isinstance(enum_name, str) else None

    def _guarded_attribute(
        self, attribute_owner: object, attribute_name: str
    ) -> object:
        """The optional attribute value when the read succeeds."""
        if attribute_owner is None or attribute_owner is _UNAVAILABLE:
            return _UNAVAILABLE
        try:
            return getattr(attribute_owner, attribute_name)
        except (AttributeError, RuntimeError, StatusNotOk):
            return _UNAVAILABLE
