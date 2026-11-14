"""How a solve ended."""

from dataclasses import dataclass
from enum import StrEnum


class Termination(StrEnum):
    """The solver-independent reason a solve ended."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    INFEASIBLE_OR_UNBOUNDED = "infeasible_or_unbounded"
    TIME_LIMIT = "time_limit"
    ITERATION_LIMIT = "iteration_limit"
    NODE_LIMIT = "node_limit"
    SOLUTION_LIMIT = "solution_limit"
    OTHER_LIMIT = "other_limit"
    INTERRUPTED = "interrupted"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ReportedOutcome:
    """The end state of a solve as an adapter read it from the solver."""

    raw_status: str
    termination: Termination
    has_solution: bool | None


@dataclass(frozen=True)
class SolveOutcome:
    """The validated end state of a solve."""

    raw_status: str
    termination: Termination
    has_solution: bool | None
