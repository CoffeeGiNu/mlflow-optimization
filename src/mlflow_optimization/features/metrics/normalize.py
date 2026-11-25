"""Common rules for metric names, missing values and gaps."""

from math import isfinite, isinf

from .types import (
    COMMON_METRICS,
    COMMON_STATISTICS,
    MissingReason,
    NormalizedMetrics,
    ReportedMetrics,
)


def _record_reported_metric(
    values: dict[str, float],
    missing: dict[str, MissingReason],
    metric_name: str,
    reported_value: float | MissingReason,
) -> None:
    """Record a finite reported metric or its missing reason."""
    if isinstance(reported_value, MissingReason):
        missing[metric_name] = reported_value
        return
    if isfinite(reported_value):
        values[metric_name] = reported_value
        return
    missing[metric_name] = MissingReason.UNDEFINED


def calculate_gaps(
    objective_value: float,
    dual_bound: float,
) -> tuple[float | None, float | None]:
    """Absolute and relative gaps for finite objective and bound values."""
    absolute_difference: float = abs(objective_value - dual_bound)
    absolute_gap: float | None = (
        absolute_difference if isfinite(absolute_difference) else None
    )
    if absolute_gap is None:
        return None, None
    relative_gap: float | None
    if objective_value == 0.0:
        relative_gap = 0.0 if absolute_gap == 0.0 else None
        return absolute_gap, relative_gap
    relative_difference: float = absolute_gap / abs(objective_value)
    relative_gap = relative_difference if isfinite(relative_difference) else None
    return absolute_gap, relative_gap


def _objective_missing_reason(
    reported: ReportedMetrics,
    has_solution: bool | None,
) -> MissingReason | None:
    """Why the objective value is missing regardless of its reported value."""
    if reported.has_objective is False:
        return MissingReason.UNDEFINED
    if reported.objective_value is MissingReason.UNBOUNDED:
        return MissingReason.UNBOUNDED
    if has_solution is False:
        return MissingReason.NO_SOLUTION
    return None


def _record_gaps(
    values: dict[str, float],
    missing: dict[str, MissingReason],
    missing_reason: MissingReason | None,
) -> None:
    """Record the absolute and relative gaps or their missing reason."""
    gap_reason: MissingReason | None = missing_reason
    if gap_reason is None:
        gap_reason = missing.get("objective_value", missing.get("dual_bound"))
    if gap_reason is not None:
        missing["absolute_gap"] = gap_reason
        missing["relative_gap"] = gap_reason
        return
    absolute_gap: float | None
    relative_gap: float | None
    absolute_gap, relative_gap = calculate_gaps(
        values["objective_value"],
        values["dual_bound"],
    )
    if absolute_gap is not None:
        values["absolute_gap"] = absolute_gap
    else:
        missing["absolute_gap"] = MissingReason.UNDEFINED
    if relative_gap is not None:
        values["relative_gap"] = relative_gap
    else:
        missing["relative_gap"] = MissingReason.UNDEFINED


def _record_reported(
    values: dict[str, float],
    missing: dict[str, MissingReason],
    tags: dict[str, str],
    reported: ReportedMetrics,
    has_solution: bool | None,
) -> None:
    """Record the metrics, tags and gaps of a reported solve."""
    if reported.has_objective is None:
        tags["optimization.has_objective"] = "unknown"
    else:
        tags["optimization.has_objective"] = str(reported.has_objective).lower()
    if reported.sense is not None:
        tags["optimization.objective_sense"] = reported.sense.value

    missing_reason: MissingReason | None = _objective_missing_reason(
        reported, has_solution
    )
    if missing_reason is not None:
        missing["objective_value"] = missing_reason
    else:
        _record_reported_metric(
            values, missing, "objective_value", reported.objective_value
        )
    if reported.has_objective is False:
        missing["dual_bound"] = MissingReason.UNDEFINED
    else:
        _record_reported_metric(values, missing, "dual_bound", reported.dual_bound)

    _record_reported_metric(
        values, missing, "solver_time_seconds", reported.solver_time_seconds
    )

    for metric_name, reported_value in reported.statistics.items():
        if metric_name in COMMON_STATISTICS:
            _record_reported_metric(values, missing, metric_name, reported_value)

    _record_gaps(values, missing, missing_reason)


def normalize_metrics(
    reported: ReportedMetrics | None,
    *,
    api: str,
    has_solution: bool | None,
    wall_time_seconds: float,
    include_native: bool,
) -> NormalizedMetrics:
    """The recordable metrics of a solve."""
    values: dict[str, float] = {}
    missing: dict[str, MissingReason] = {}
    tags: dict[str, str] = {}

    _record_reported_metric(values, missing, "wall_time_seconds", wall_time_seconds)

    if reported is None:
        tags["optimization.has_objective"] = "unknown"
        for metric_name in COMMON_METRICS:
            if metric_name != "wall_time_seconds":
                missing[metric_name] = MissingReason.FAILED
    else:
        _record_reported(values, missing, tags, reported, has_solution)

    if (
        reported is not None
        and isinstance(reported.dual_bound, float)
        and isinf(reported.dual_bound)
    ):
        tags["optimization.dual_bound_infinity"] = (
            "+inf" if reported.dual_bound > 0 else "-inf"
        )

    for metric_name in COMMON_METRICS:
        if metric_name not in values and metric_name not in missing:
            missing[metric_name] = MissingReason.UNSUPPORTED
        if metric_name in missing:
            tags[f"optimization.missing.{metric_name}"] = missing[metric_name].value

    if reported is not None and include_native:
        for native_name, native_value in reported.native.items():
            api_metric_name: str = f"{api}.{native_name}"
            if isfinite(native_value):
                values[api_metric_name] = native_value

    return NormalizedMetrics(values=values, tags=tags)
