"""Common rules for metric names, missing values and gaps."""

from math import isfinite

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
        if reported.has_objective is None:
            tags["optimization.has_objective"] = "unknown"
        else:
            tags["optimization.has_objective"] = str(reported.has_objective).lower()
        if reported.sense is not None:
            tags["optimization.objective_sense"] = reported.sense.value

        if reported.has_objective is False:
            missing["objective_value"] = MissingReason.UNDEFINED
            missing["dual_bound"] = MissingReason.UNDEFINED
        elif has_solution is False:
            missing["objective_value"] = MissingReason.NO_SOLUTION
            _record_reported_metric(values, missing, "dual_bound", reported.dual_bound)
        else:
            _record_reported_metric(values, missing, "objective_value", reported.objective_value)
            _record_reported_metric(values, missing, "dual_bound", reported.dual_bound)

        _record_reported_metric(values, missing, "solver_time_seconds", reported.solver_time_seconds)

        for metric_name, reported_value in reported.statistics.items():
            if metric_name in COMMON_STATISTICS:
                _record_reported_metric(values, missing, metric_name, reported_value)

        if reported.has_objective is False:
            missing["absolute_gap"] = MissingReason.UNDEFINED
            missing["relative_gap"] = MissingReason.UNDEFINED
        elif has_solution is False:
            missing["absolute_gap"] = MissingReason.NO_SOLUTION
            missing["relative_gap"] = MissingReason.NO_SOLUTION
        elif "objective_value" in missing:
            missing["absolute_gap"] = missing["objective_value"]
            missing["relative_gap"] = missing["objective_value"]
        elif "dual_bound" in missing:
            missing["absolute_gap"] = missing["dual_bound"]
            missing["relative_gap"] = missing["dual_bound"]
        else:
            absolute_gap: float = abs(values["objective_value"] - values["dual_bound"])
            if isfinite(absolute_gap):
                values["absolute_gap"] = absolute_gap
                if values["objective_value"] == 0.0:
                    if absolute_gap == 0.0:
                        values["relative_gap"] = 0.0
                    else:
                        missing["relative_gap"] = MissingReason.UNDEFINED
                else:
                    relative_gap: float = absolute_gap / abs(values["objective_value"])
                    if isfinite(relative_gap):
                        values["relative_gap"] = relative_gap
                    else:
                        missing["relative_gap"] = MissingReason.UNDEFINED
            else:
                missing["absolute_gap"] = MissingReason.UNDEFINED
                missing["relative_gap"] = MissingReason.UNDEFINED

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
