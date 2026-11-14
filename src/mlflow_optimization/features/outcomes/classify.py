"""Common rules for termination and solution availability."""

from mlflow_optimization.common.diagnostics import warn

from .types import ReportedOutcome, SolveOutcome, Termination


def classify_outcome(
    reported: ReportedOutcome | None,
    error: BaseException | None,
    *,
    has_objective: bool | None = None,
) -> SolveOutcome:
    """The validated outcome of a solve."""
    if reported is None:
        if isinstance(error, KeyboardInterrupt):
            return SolveOutcome("unknown", Termination.INTERRUPTED, None)
        if error is not None:
            return SolveOutcome("unknown", Termination.ERROR, None)
        return SolveOutcome("unknown", Termination.UNKNOWN, None)

    termination: Termination = reported.termination
    if isinstance(error, KeyboardInterrupt):
        termination = Termination.INTERRUPTED
    elif error is not None:
        termination = Termination.ERROR
    else:
        if termination is Termination.OPTIMAL and has_objective is False:
            termination = Termination.FEASIBLE
        elif termination is Termination.FEASIBLE and reported.has_solution is False:
            warn("Reported feasible termination conflicts with has_solution=False.")
            termination = Termination.UNKNOWN
        elif termination is Termination.OPTIMAL and reported.has_solution is False:
            warn("Reported optimal termination conflicts with has_solution=False.")
        elif termination is Termination.INFEASIBLE and reported.has_solution is True:
            warn("Reported infeasible termination conflicts with has_solution=True.")

    return SolveOutcome(reported.raw_status, termination, reported.has_solution)


def outcome_tags(outcome: SolveOutcome) -> dict[str, str]:
    """The tags that describe an outcome."""
    has_solution: str = "unknown"
    if outcome.has_solution is True:
        has_solution = "true"
    elif outcome.has_solution is False:
        has_solution = "false"

    return {
        "optimization.termination": outcome.termination.value,
        "optimization.raw_status": outcome.raw_status,
        "optimization.has_solution": has_solution,
    }
