"""What is observed about a solve."""

from dataclasses import dataclass, field

from mlflow_optimization.features.metadata.models import ReportedMetadata
from mlflow_optimization.features.metrics.models import ReportedMetrics
from mlflow_optimization.features.outcomes.models import ReportedOutcome


@dataclass(frozen=True)
class SolveObservation:
    """The SDK-free snapshot an adapter takes of one finished solve."""

    api: str
    outcome: ReportedOutcome
    metrics: ReportedMetrics = field(default_factory=ReportedMetrics)
    metadata: ReportedMetadata = field(default_factory=ReportedMetadata)


@dataclass(frozen=True)
class CompletedSolve:
    """One finished outermost solve call as seen by the observer."""

    api: str
    solve_id: str
    solve_index: int
    start_time_ms: int
    end_time_ms: int
    wall_time_seconds: float
    observation: SolveObservation | None
    error: BaseException | None
