"""The stored form of a solve."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class RunReference:
    """The identity of a run in the experiment tracker."""

    run_id: str
    experiment_id: str


class RunStatus(StrEnum):
    """The lifecycle state a solve's run ends in."""

    FINISHED = "FINISHED"
    FAILED = "FAILED"
    KILLED = "KILLED"


@dataclass(frozen=True)
class SolveRecord:
    """Everything stored for one solve, under final key names."""

    name: str
    status: RunStatus
    start_time_ms: int
    end_time_ms: int
    tags: Mapping[str, str]
    metrics: Mapping[str, float]
    params: Mapping[str, str]


@dataclass
class RecordOptions:
    """Which optional parts of a solve are recorded."""

    log_solver_parameters: bool = True
    log_solver_metrics: bool = True
    log_artifacts: bool = True
