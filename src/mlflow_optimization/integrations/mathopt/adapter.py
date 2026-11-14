"""The MathOpt solver integration."""

import dataclasses
import math
from collections.abc import Callable, Mapping
from datetime import timedelta
from enum import Enum
from typing import TypeVar, cast

import ortools
from ortools.math_opt.python import mathopt
from ortools.math_opt.python import solve as mathopt_solve
from ortools.math_opt.python.model import Model
from ortools.math_opt.python.parameters import SolveParameters, SolverType
from ortools.math_opt.python.result import SolveResult

from mlflow_optimization.common.patching import Patch
from mlflow_optimization.features.autolog.models import SolveObservation
from mlflow_optimization.features.autolog.ports import SolveObserver
from mlflow_optimization.features.metadata.models import ParameterOrigin, ReportedMetadata, SolverParameter
from mlflow_optimization.features.metrics.models import MissingReason, ObjectiveSense, ReportedMetrics
from mlflow_optimization.features.outcomes.models import ReportedOutcome, Termination


ParameterValue = int | float | str | bool
T = TypeVar("T")
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
        except BaseException:
            patch: Patch
            for patch in reversed(installed_patches):
                patch.uninstall()
            raise

    def uninstall(self) -> None:
        """The restored MathOpt solve hooks."""
        patch: Patch
        entry_point: str
        for patch, _ in reversed(self._patches):
            patch.uninstall()

    def _wrap(self, original: object, observer: SolveObserver, entry_point: str) -> object:
        original_solve: Callable[..., SolveResult] = cast(Callable[..., SolveResult], original)
        if entry_point == "incremental":
            def wrapped_incremental_solve(
                incremental_solver: mathopt_solve.IncrementalSolver,
                *call_arguments: object,
                **call_keywords: object,
            ) -> SolveResult:
                def call() -> SolveResult:
                    return original_solve(incremental_solver, *call_arguments, **call_keywords)

                def extract(
                    result: SolveResult | None,
                    error: BaseException | None,
                ) -> SolveObservation:
                    return self._extract(
                        result,
                        error,
                        entry_point,
                        (incremental_solver, *call_arguments),
                        cast(dict[str, object], call_keywords),
                    )

                return observer.observe(self.api, call, extract)

            return wrapped_incremental_solve

        def wrapped_solve(*call_arguments: object, **call_keywords: object) -> SolveResult:
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
        error: BaseException | None,
        entry_point: str,
        call_arguments: tuple[object, ...],
        call_keywords: Mapping[str, object],
    ) -> SolveObservation:
        """The solver-independent snapshot of one MathOpt result."""
        solver_type: SolverType | None = self._solver_type(entry_point, call_arguments, call_keywords)
        model: Model | None = self._model(entry_point, call_arguments, call_keywords)
        solve_parameters: SolveParameters | None = self._solve_parameters(call_keywords)
        outcome: ReportedOutcome = self._outcome(result, error)
        metrics: ReportedMetrics = self._metrics(result, model, solver_type)
        metadata: ReportedMetadata = self._metadata(solver_type, model, solve_parameters)
        return SolveObservation(self.api, outcome, metrics, metadata)

    def _solver_type(
        self,
        entry_point: str,
        call_arguments: tuple[object, ...],
        call_keywords: Mapping[str, object],
    ) -> SolverType | None:
        """The solver type supplied to a MathOpt entry point."""
        if entry_point == "incremental":
            incremental_solver: object | None = call_arguments[0] if call_arguments else None
            solver_type: object = self._guarded_attribute(incremental_solver, "_solver_type")
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
            incremental_solver: object | None = call_arguments[0] if call_arguments else None
            model: object = self._guarded_attribute(incremental_solver, "_model")
            return model if isinstance(model, Model) else None
        model: object = call_keywords.get("opt_model", _UNAVAILABLE)
        if model is _UNAVAILABLE and call_arguments:
            model = call_arguments[0]
        return model if isinstance(model, Model) else None

    def _solve_parameters(self, call_keywords: Mapping[str, object]) -> SolveParameters | None:
        """The explicit MathOpt solve parameters."""
        solve_parameters: object = call_keywords.get("params")
        return solve_parameters if isinstance(solve_parameters, SolveParameters) else None

    def _outcome(
        self,
        result: SolveResult | None,
        error: BaseException | None,
    ) -> ReportedOutcome:
        """The MathOpt termination status and primal solution availability."""
        if error is not None:
            termination: Termination = (
                Termination.INTERRUPTED if isinstance(error, KeyboardInterrupt) else Termination.ERROR
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
        solution_read: object = self._guarded_call(lambda: result.has_primal_feasible_solution())
        has_solution: bool | None = solution_read if isinstance(solution_read, bool) else None
        return ReportedOutcome(raw_status, termination, has_solution)

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
        if limit_name is not None and limit_name != "UNDETERMINED":
            limit_value: object = self._guarded_attribute(limit, "value")
            raw_status = f"{raw_status}; limit={limit_name}"
            if isinstance(limit_value, int):
                raw_status = f"{raw_status} ({limit_value})"
        return raw_status

    def _termination(self, reason_name: str | None, limit_name: str | None) -> Termination:
        """The solver-independent termination of a MathOpt result."""
        limit_terminations: dict[str, Termination] = {
            "UNDETERMINED": Termination.UNKNOWN,
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
        if (
            reason_name in {"FEASIBLE", "NO_SOLUTION_FOUND"}
            and limit_name is not None
            and limit_name != "UNDETERMINED"
        ):
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
    ) -> ReportedMetrics:
        """The objective, bound, time, and statistics of a MathOpt result."""
        has_objective: bool | None = self._has_objective(model)
        sense: ObjectiveSense | None = self._objective_sense(model, has_objective)
        has_solution: bool | None = None
        if result is not None:
            solution_read: object = self._guarded_call(lambda: result.has_primal_feasible_solution())
            has_solution = solution_read if isinstance(solution_read, bool) else None
        if result is None:
            objective_value: float | MissingReason = MissingReason.UNSUPPORTED
            dual_bound: float | MissingReason = MissingReason.UNSUPPORTED
            solver_time_seconds: float | MissingReason = MissingReason.UNSUPPORTED
        else:
            objective_value = self._objective_value(result, has_solution)
            dual_bound = self._numeric_result_value(result, "dual_bound")
            solver_time_seconds = self._solver_time(result)
        stats: dict[str, float | MissingReason] = self._stats(result)
        native: dict[str, float] = self._native_statistics(result, solver_type)
        return ReportedMetrics(
            has_objective=has_objective,
            sense=sense,
            objective_value=objective_value,
            dual_bound=dual_bound,
            solver_time_seconds=solver_time_seconds,
            stats=stats,
            native=native,
        )

    def _has_objective(self, model: Model | None) -> bool | None:
        """Whether a MathOpt objective has nonzero terms or an offset."""
        if model is None:
            return None
        objective: object = self._guarded_attribute(model, "objective")
        linear_terms: object = self._guarded_call(lambda: tuple(objective.linear_terms()))
        quadratic_terms: object = self._guarded_call(lambda: tuple(objective.quadratic_terms()))
        offset: object = self._guarded_attribute(objective, "offset")
        if not isinstance(linear_terms, tuple) or not isinstance(quadratic_terms, tuple):
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
        value: object = self._guarded_call(lambda: result.objective_value())
        if isinstance(value, (int, float)):
            return self._normalize_infinity(float(value))
        return MissingReason.FAILED

    def _numeric_result_value(self, result: SolveResult, name: str) -> float | MissingReason:
        """One numeric value from a MathOpt result."""
        value: object = self._guarded_call(lambda: getattr(result, name)())
        if isinstance(value, (int, float)):
            return self._normalize_infinity(float(value))
        return MissingReason.FAILED

    def _solver_time(self, result: SolveResult) -> float | MissingReason:
        """The solve time reported by MathOpt."""
        duration: object = self._guarded_call(lambda: result.solve_time())
        if isinstance(duration, timedelta):
            return duration.total_seconds()
        return MissingReason.FAILED

    def _stats(self, result: SolveResult | None) -> dict[str, float | MissingReason]:
        """The common search statistics reported by MathOpt."""
        stats: dict[str, float | MissingReason] = {
            "node_count": MissingReason.UNSUPPORTED,
            "simplex_iterations": MissingReason.UNSUPPORTED,
            "barrier_iterations": MissingReason.UNSUPPORTED,
        }
        if result is None:
            return stats
        solve_stats: object = self._guarded_attribute(result, "solve_stats")
        statistic_name: str
        for statistic_name in stats:
            value: object = self._guarded_attribute(solve_stats, statistic_name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                stats[statistic_name] = float(value)
            elif value is _UNAVAILABLE:
                stats[statistic_name] = MissingReason.FAILED
        return stats

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
                self._collect_native_values(output_name.removesuffix("_specific_output"), output, native)
        return native

    def _collect_native_values(
        self,
        prefix: str,
        message: object,
        native: dict[str, float],
    ) -> None:
        """The numeric leaves in a MathOpt backend output message."""
        fields: object = self._guarded_call(lambda: message.ListFields())
        if not isinstance(fields, list):
            return
        field_descriptor: object
        field_value: object
        for field_descriptor, field_value in fields:
            field_name: object = self._guarded_attribute(field_descriptor, "name")
            enum_descriptor: object = self._guarded_attribute(field_descriptor, "enum_type")
            if not isinstance(field_name, str):
                continue
            if enum_descriptor is not None and enum_descriptor is not _UNAVAILABLE:
                continue
            if self._is_message(field_value):
                self._collect_native_values(f"{prefix}.{field_name}", field_value, native)
            elif isinstance(field_value, (int, float)) and not isinstance(field_value, bool):
                native_name: str = f"{prefix}.{field_name}"
                native[native_name] = self._normalize_infinity(float(field_value))

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
            wall_time_includes_model_build=False,
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
            count: object = self._guarded_call(lambda: getattr(model, method_name)())
            if isinstance(count, int) and not isinstance(count, bool):
                problem[problem_name] = count
                if problem_name != "num_variables":
                    constraint_counts.append(count)
        if len(constraint_counts) == 3:
            problem["num_constraints"] = sum(constraint_counts)
        variables: object = self._guarded_call(lambda: tuple(model.variables()))
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

    def _parameters(self, solve_parameters: SolveParameters | None) -> list[SolverParameter]:
        """The solve parameters that differ from MathOpt defaults."""
        parameters: list[SolverParameter] = []
        if solve_parameters is None:
            return parameters
        default_parameters: SolveParameters = SolveParameters()
        parameter_field: object
        for parameter_field in dataclasses.fields(SolveParameters):
            parameter_name: str = parameter_field.name
            parameter_value: object = self._guarded_attribute(solve_parameters, parameter_name)
            default_value: object = self._guarded_attribute(default_parameters, parameter_name)
            if parameter_value is _UNAVAILABLE or default_value is _UNAVAILABLE:
                continue
            if parameter_value == default_value or parameter_value is None:
                continue
            self._append_parameters(parameter_name, parameter_value, default_value, parameters)
        return parameters

    def _append_parameters(
        self,
        name: str,
        value: object,
        default_value: object,
        parameters: list[SolverParameter],
    ) -> None:
        """The differing nested MathOpt parameter values."""
        if value == default_value:
            return
        if isinstance(value, Mapping):
            default_mapping: Mapping[object, object] = (
                default_value if isinstance(default_value, Mapping) else {}
            )
            nested_name: object
            nested_value: object
            for nested_name, nested_value in value.items():
                if not isinstance(nested_name, str):
                    continue
                nested_default: object = default_mapping.get(nested_name, _UNAVAILABLE)
                self._append_parameters(f"{name}.{nested_name}", nested_value, nested_default, parameters)
            return
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            default_object: object = default_value
            parameter_field: object
            for parameter_field in dataclasses.fields(value):
                nested_name: str = parameter_field.name
                nested_value: object = self._guarded_attribute(value, nested_name)
                nested_default: object = self._guarded_attribute(default_object, nested_name)
                self._append_parameters(f"{name}.{nested_name}", nested_value, nested_default, parameters)
            return
        if self._is_message(value):
            fields: object = self._guarded_call(lambda: value.ListFields())
            if not isinstance(fields, list):
                return
            field_descriptor: object
            field_value: object
            for field_descriptor, field_value in fields:
                field_name: object = self._guarded_attribute(field_descriptor, "name")
                enum_descriptor: object = self._guarded_attribute(field_descriptor, "enum_type")
                if not isinstance(field_name, str):
                    continue
                if enum_descriptor is not None and enum_descriptor is not _UNAVAILABLE:
                    enum_value: object = self._guarded_call(
                        lambda: enum_descriptor.values_by_number[field_value]
                    )
                    enum_name: object = self._guarded_attribute(enum_value, "name")
                    if isinstance(enum_name, str):
                        parameters.append(
                            SolverParameter(
                                f"{name}.{field_name}",
                                enum_name,
                                ParameterOrigin.EXPLICIT,
                            )
                        )
                else:
                    self._append_parameters(
                        f"{name}.{field_name}",
                        field_value,
                        _UNAVAILABLE,
                        parameters,
                    )
            return
        scalar: ParameterValue | None = self._parameter_value(value)
        if scalar is not None:
            parameters.append(SolverParameter(name, scalar, ParameterOrigin.EXPLICIT))
            return
        serialized_value: object = self._guarded_call(lambda: str(value))
        if isinstance(serialized_value, str):
            parameters.append(
                SolverParameter(name, serialized_value, ParameterOrigin.EXPLICIT)
            )

    def _parameter_value(self, value: object) -> ParameterValue | None:
        """The scalar form of a MathOpt parameter."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float, str)):
            return value
        if isinstance(value, timedelta):
            return value.total_seconds()
        if isinstance(value, Enum):
            enum_name: object = self._guarded_attribute(value, "name")
            return enum_name if isinstance(enum_name, str) else None
        return None

    def _is_message(self, value: object) -> bool:
        """Whether a value is a protobuf message."""
        list_fields: object = self._guarded_attribute(value, "ListFields")
        return callable(list_fields)

    def _enum_name(self, value: object) -> str | None:
        """The name of a known MathOpt enum value."""
        name: object = self._guarded_attribute(value, "name")
        return name if isinstance(name, str) else None

    def _guarded_attribute(self, owner: object, name: str) -> object:
        """The optional attribute value when the read succeeds."""
        if owner is None or owner is _UNAVAILABLE:
            return _UNAVAILABLE
        try:
            return getattr(owner, name)
        except Exception:
            return _UNAVAILABLE

    def _guarded_call(self, read: Callable[[], T]) -> T | object:
        """The optional read result when the call succeeds."""
        try:
            return read()
        except Exception:
            return _UNAVAILABLE

    def _normalize_infinity(self, value: float) -> float:
        """The floating value of a MathOpt metric after sentinel normalization."""
        if math.isinf(value):
            return value
        positive_sentinels: tuple[float, ...] = (1e20, 1e98, 1e100, float(2**63 - 1))
        negative_sentinels: tuple[float, ...] = (-1e20, -1e98, -1e100, float(-(2**63)))
        if value in positive_sentinels:
            return math.inf
        if value in negative_sentinels:
            return -math.inf
        return value
