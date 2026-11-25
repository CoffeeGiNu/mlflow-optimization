"""Objective, bound, time and search statistics of a solve."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class MissingReason(StrEnum):
    """Why a value is absent."""

    UNSUPPORTED = "unsupported"
    UNDEFINED = "undefined"
    NO_SOLUTION = "no_solution"
    FAILED = "failed"
    UNBOUNDED = "unbounded"


class ObjectiveSense(StrEnum):
    """The optimization direction."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


COMMON_STATISTICS = ("node_count", "simplex_iterations", "barrier_iterations")
COMMON_METRICS = (
    "objective_value",
    "dual_bound",
    "absolute_gap",
    "relative_gap",
    "wall_time_seconds",
    "solver_time_seconds",
    *COMMON_STATISTICS,
)


@dataclass(frozen=True)
class ReportedMetrics:
    """The numbers of a solve as an adapter read them from the solver."""

    has_objective: bool | None = None
    sense: ObjectiveSense | None = None
    objective_value: float | MissingReason = MissingReason.UNSUPPORTED
    dual_bound: float | MissingReason = MissingReason.UNSUPPORTED
    solver_time_seconds: float | MissingReason = MissingReason.UNSUPPORTED
    statistics: Mapping[str, float | MissingReason] = field(default_factory=dict)
    native: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedMetrics:
    """The metrics and metric-related tags recorded for a solve, under final key names."""

    values: Mapping[str, float]
    tags: Mapping[str, str]
