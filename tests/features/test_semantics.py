import math
import warnings

import pytest

from mlflow_optimization.common.diagnostics import MlflowOptimizationWarning
from mlflow_optimization.features.metadata.collect import collect_metadata
from mlflow_optimization.features.metadata.types import (
    CollectedMetadata,
    ParameterOrigin,
    ReportedMetadata,
    SolverParameter,
)
from mlflow_optimization.features.metrics.types import (
    MissingReason,
    ObjectiveSense,
    NormalizedMetrics,
    ReportedMetrics,
)
from mlflow_optimization.features.metrics.normalize import normalize_metrics
from mlflow_optimization.features.outcomes.classify import (
    classify_outcome,
    outcome_tags,
)
from mlflow_optimization.features.outcomes.types import (
    ReportedOutcome,
    SolveOutcome,
    Termination,
)


@pytest.mark.parametrize(
    ("termination", "has_solution", "expected_termination", "should_warn"),
    (
        (Termination.OPTIMAL, False, Termination.OPTIMAL, True),
        (Termination.FEASIBLE, False, Termination.UNKNOWN, True),
        (Termination.INFEASIBLE, True, Termination.INFEASIBLE, True),
        (Termination.OPTIMAL, None, Termination.OPTIMAL, False),
        (Termination.TIME_LIMIT, True, Termination.TIME_LIMIT, False),
        (Termination.TIME_LIMIT, False, Termination.TIME_LIMIT, False),
        (
            Termination.INFEASIBLE_OR_UNBOUNDED,
            True,
            Termination.INFEASIBLE_OR_UNBOUNDED,
            False,
        ),
        (
            Termination.INFEASIBLE_OR_UNBOUNDED,
            False,
            Termination.INFEASIBLE_OR_UNBOUNDED,
            False,
        ),
    ),
)
def test_termination_and_solution_validation(
    termination: Termination,
    has_solution: bool | None,
    expected_termination: Termination,
    should_warn: bool,
) -> None:
    reported: ReportedOutcome = ReportedOutcome(
        raw_status=termination.value,
        termination=termination,
        has_solution=has_solution,
    )
    if should_warn:
        with pytest.warns(MlflowOptimizationWarning):
            outcome: SolveOutcome = classify_outcome(reported, None)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MlflowOptimizationWarning)
            outcome = classify_outcome(reported, None)

    assert outcome.termination is expected_termination
    assert outcome.has_solution is has_solution
    assert outcome.raw_status == termination.value
    assert outcome_tags(outcome) == {
        "optimization.raw_status": termination.value,
        "optimization.termination": expected_termination.value,
        "optimization.has_solution": (
            "unknown" if has_solution is None else str(has_solution).lower()
        ),
    }


@pytest.mark.parametrize(
    ("has_solution", "expected_termination", "should_warn"),
    (
        (True, Termination.FEASIBLE, False),
        (False, Termination.UNKNOWN, True),
        (None, Termination.FEASIBLE, False),
    ),
)
def test_optimal_termination_without_objective_is_normalized_then_validated(
    has_solution: bool | None,
    expected_termination: Termination,
    should_warn: bool,
) -> None:
    reported: ReportedOutcome = ReportedOutcome(
        raw_status="optimal",
        termination=Termination.OPTIMAL,
        has_solution=has_solution,
    )
    if should_warn:
        with pytest.warns(MlflowOptimizationWarning):
            outcome: SolveOutcome = classify_outcome(
                reported, None, has_objective=False
            )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MlflowOptimizationWarning)
            outcome = classify_outcome(reported, None, has_objective=False)

    assert outcome.termination is expected_termination
    assert outcome.has_solution is has_solution
    assert outcome.raw_status == "optimal"


@pytest.mark.parametrize(
    ("reported", "error", "expected_termination", "expected_has_solution"),
    (
        (
            ReportedOutcome("time limit", Termination.TIME_LIMIT, False),
            RuntimeError("solver failure"),
            Termination.ERROR,
            False,
        ),
        (
            ReportedOutcome("time limit", Termination.TIME_LIMIT, True),
            KeyboardInterrupt(),
            Termination.INTERRUPTED,
            True,
        ),
        (None, RuntimeError("solver failure"), Termination.ERROR, None),
        (None, KeyboardInterrupt(), Termination.INTERRUPTED, None),
    ),
)
def test_raised_solve_errors_override_termination(
    reported: ReportedOutcome | None,
    error: BaseException,
    expected_termination: Termination,
    expected_has_solution: bool | None,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", MlflowOptimizationWarning)
        outcome: SolveOutcome = classify_outcome(reported, error)

    assert outcome.termination is expected_termination
    assert outcome.has_solution is expected_has_solution
    if reported is not None:
        assert outcome.raw_status == reported.raw_status


@pytest.mark.parametrize(
    ("reported", "expected_reason"),
    (
        (ReportedMetrics(), MissingReason.UNSUPPORTED),
        (None, MissingReason.FAILED),
    ),
)
def test_absent_common_metrics_get_reason_tags_without_zero_filling(
    reported: ReportedMetrics | None, expected_reason: MissingReason
) -> None:
    normalized: NormalizedMetrics = normalize_metrics(
        reported,
        api="fake",
        has_solution=True,
        wall_time_seconds=2.5,
        include_native=False,
    )
    absent_metrics: tuple[str, ...] = (
        "objective_value",
        "dual_bound",
        "absolute_gap",
        "relative_gap",
        "solver_time_seconds",
        "node_count",
        "simplex_iterations",
        "barrier_iterations",
    )

    assert normalized.values == {"wall_time_seconds": 2.5}
    metric_name: str
    for metric_name in absent_metrics:
        assert (
            normalized.tags[f"optimization.missing.{metric_name}"]
            == expected_reason.value
        )
    assert normalized.tags["optimization.has_objective"] == "unknown"


def test_objective_less_models_make_objective_metrics_undefined() -> None:
    reported: ReportedMetrics = ReportedMetrics(
        has_objective=False,
        sense=ObjectiveSense.MINIMIZE,
        objective_value=15.0,
        dual_bound=12.0,
    )
    normalized: NormalizedMetrics = normalize_metrics(
        reported,
        api="fake",
        has_solution=True,
        wall_time_seconds=1.0,
        include_native=False,
    )
    objective_metrics: tuple[str, ...] = (
        "objective_value",
        "dual_bound",
        "absolute_gap",
        "relative_gap",
    )

    assert normalized.values == {"wall_time_seconds": 1.0}
    assert normalized.tags["optimization.has_objective"] == "false"
    metric_name: str
    for metric_name in objective_metrics:
        assert (
            normalized.tags[f"optimization.missing.{metric_name}"]
            == MissingReason.UNDEFINED.value
        )


def test_no_solution_discards_a_bogus_objective_but_keeps_a_finite_bound() -> None:
    reported: ReportedMetrics = ReportedMetrics(
        has_objective=True,
        sense=ObjectiveSense.MINIMIZE,
        objective_value=999.0,
        dual_bound=900.0,
    )
    normalized: NormalizedMetrics = normalize_metrics(
        reported,
        api="fake",
        has_solution=False,
        wall_time_seconds=1.0,
        include_native=False,
    )

    assert normalized.values == {
        "dual_bound": 900.0,
        "wall_time_seconds": 1.0,
    }
    metric_name: str
    for metric_name in ("objective_value", "absolute_gap", "relative_gap"):
        assert (
            normalized.tags[f"optimization.missing.{metric_name}"]
            == MissingReason.NO_SOLUTION.value
        )
    assert f"optimization.missing.dual_bound" not in normalized.tags


def test_non_finite_common_and_native_metrics_are_not_recorded() -> None:
    reported: ReportedMetrics = ReportedMetrics(
        has_objective=True,
        sense=ObjectiveSense.MINIMIZE,
        objective_value=math.nan,
        dual_bound=math.inf,
        solver_time_seconds=-math.inf,
        statistics={"node_count": math.nan, "barrier_iterations": -math.inf},
        native={"branches": math.inf},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", MlflowOptimizationWarning)
        normalized: NormalizedMetrics = normalize_metrics(
            reported,
            api="fake",
            has_solution=True,
            wall_time_seconds=math.nan,
            include_native=True,
        )
    common_metrics: tuple[str, ...] = (
        "objective_value",
        "dual_bound",
        "absolute_gap",
        "relative_gap",
        "wall_time_seconds",
        "solver_time_seconds",
        "node_count",
        "barrier_iterations",
    )

    assert normalized.values == {}
    metric_name: str
    for metric_name in common_metrics:
        assert (
            normalized.tags[f"optimization.missing.{metric_name}"]
            == MissingReason.UNDEFINED.value
        )


@pytest.mark.parametrize(
    (
        "sense",
        "objective_value",
        "dual_bound",
        "expected_absolute",
        "expected_relative",
    ),
    (
        (ObjectiveSense.MINIMIZE, 10.0, 7.0, 3.0, 0.3),
        (ObjectiveSense.MAXIMIZE, 10.0, 13.0, 3.0, 0.3),
        (ObjectiveSense.MINIMIZE, -10.0, -13.0, 3.0, 0.3),
        (ObjectiveSense.MAXIMIZE, -10.0, -7.0, 3.0, 0.3),
        (ObjectiveSense.MINIMIZE, -10.0, 7.0, 17.0, 1.7),
    ),
)
def test_gap_calculation_handles_sense_and_signed_values(
    sense: ObjectiveSense,
    objective_value: float,
    dual_bound: float,
    expected_absolute: float,
    expected_relative: float,
) -> None:
    reported: ReportedMetrics = ReportedMetrics(
        has_objective=True,
        sense=sense,
        objective_value=objective_value,
        dual_bound=dual_bound,
    )
    normalized: NormalizedMetrics = normalize_metrics(
        reported,
        api="fake",
        has_solution=True,
        wall_time_seconds=1.0,
        include_native=False,
    )

    assert normalized.values["absolute_gap"] == pytest.approx(expected_absolute)
    assert normalized.values["relative_gap"] == pytest.approx(expected_relative)


@pytest.mark.parametrize(
    ("dual_bound", "expected_absolute", "expected_relative"),
    ((0.0, 0.0, 0.0), (2.0, 2.0, None)),
)
def test_zero_objective_has_only_a_defined_relative_gap_when_gap_is_zero(
    dual_bound: float,
    expected_absolute: float,
    expected_relative: float | None,
) -> None:
    reported: ReportedMetrics = ReportedMetrics(
        has_objective=True,
        sense=ObjectiveSense.MINIMIZE,
        objective_value=0.0,
        dual_bound=dual_bound,
    )
    normalized: NormalizedMetrics = normalize_metrics(
        reported,
        api="fake",
        has_solution=True,
        wall_time_seconds=1.0,
        include_native=False,
    )

    assert normalized.values["absolute_gap"] == expected_absolute
    if expected_relative is None:
        assert "relative_gap" not in normalized.values
        assert (
            normalized.tags["optimization.missing.relative_gap"]
            == MissingReason.UNDEFINED.value
        )
    else:
        assert normalized.values["relative_gap"] == expected_relative


@pytest.mark.parametrize("include_native", (True, False))
def test_native_metrics_are_namespaced_and_optional(include_native: bool) -> None:
    reported: ReportedMetrics = ReportedMetrics(
        has_objective=True,
        sense=ObjectiveSense.MINIMIZE,
        objective_value=10.0,
        dual_bound=7.0,
        native={"relative_gap": 0.2, "branches": 18.0},
    )
    normalized: NormalizedMetrics = normalize_metrics(
        reported,
        api="fake",
        has_solution=True,
        wall_time_seconds=1.0,
        include_native=include_native,
    )

    assert normalized.values["relative_gap"] == pytest.approx(0.3)
    if include_native:
        assert normalized.values["fake.relative_gap"] == 0.2
        assert normalized.values["fake.branches"] == 18.0
    else:
        assert "fake.relative_gap" not in normalized.values
        assert "fake.branches" not in normalized.values


@pytest.mark.parametrize(
    ("parameter", "expected_parameter_name"),
    (
        (SolverParameter("threads", 8), "solver_param.threads"),
        (
            SolverParameter("relative_gap", 0.05, ParameterOrigin.DEFAULT),
            "solver_param_default.relative_gap",
        ),
        (
            SolverParameter("seed", 7, ParameterOrigin.EFFECTIVE),
            "solver_param_effective.seed",
        ),
    ),
)
def test_parameter_origins_use_distinct_namespaces(
    parameter: SolverParameter, expected_parameter_name: str | None
) -> None:
    reported: ReportedMetadata = ReportedMetadata(parameters=(parameter,))
    collected: CollectedMetadata = collect_metadata(
        "fake", reported, include_parameters=True
    )

    assert collected.params == {expected_parameter_name: str(parameter.value)}
