"""A fake solver API for adapter tests."""

from collections.abc import Callable

from mlflow_optimization.common.patching import Patch
from mlflow_optimization.features.autolog.types import SolveObservation
from mlflow_optimization.features.autolog.ports import SolveObserver
from mlflow_optimization.features.metadata.types import ReportedMetadata, SolverParameter
from mlflow_optimization.features.metrics.types import (
    MissingReason,
    ObjectiveSense,
    ReportedMetrics,
)
from mlflow_optimization.features.outcomes.types import ReportedOutcome, Termination


class FakeSolver:
    """A configurable solver for adapter tests."""

    def __init__(
        self,
        *,
        status: str | Termination = "optimal",
        has_solution: bool = True,
        objective: float | None = 1.0,
        bound: float | None = 1.0,
        sense: str | ObjectiveSense = "minimize",
        parameters: dict[str, int | float | str | bool] | None = None,
        statistics: dict[str, float] | None = None,
        raises: BaseException | None = None,
    ) -> None:
        self.status: str | Termination = status
        self.has_solution: bool = has_solution
        self.objective: float | None = objective
        self.bound: float | None = bound
        self.sense: str | ObjectiveSense = sense
        self.parameters: dict[str, int | float | str | bool] = (
            parameters.copy() if parameters is not None else {}
        )
        self.statistics: dict[str, float] = statistics.copy() if statistics is not None else {}
        self.raises: BaseException | None = raises

    def solve(self) -> str:
        """The configured solver termination status."""
        if self.raises is not None:
            raise self.raises
        if isinstance(self.status, Termination):
            return self.status.value
        return self.status


class Integration:
    """The adapter for the fake solver API."""

    api: str = "fake"

    def __init__(self) -> None:
        self._patch: Patch = Patch(FakeSolver, "solve")
        self._observer: SolveObserver | None = None

    def install(self, observer: SolveObserver) -> None:
        """Route fake solver calls through the observer."""
        self._observer = observer

        def factory(original: Callable[[FakeSolver], str]) -> Callable[[FakeSolver], str]:
            def observed(solver: FakeSolver) -> str:
                def call() -> str:
                    return original(solver)

                def extract(
                    result: str | None, error: BaseException | None
                ) -> SolveObservation:
                    reported_status: str = (
                        result
                        if error is None and result is not None
                        else str(solver.status)
                    )
                    outcome: ReportedOutcome = ReportedOutcome(
                        raw_status=reported_status,
                        termination=Termination(reported_status),
                        has_solution=solver.has_solution,
                    )
                    objective_value: float | MissingReason = (
                        solver.objective
                        if solver.objective is not None
                        else MissingReason.UNSUPPORTED
                    )
                    dual_bound: float | MissingReason = (
                        solver.bound if solver.bound is not None else MissingReason.UNSUPPORTED
                    )
                    sense: ObjectiveSense = ObjectiveSense(solver.sense)
                    metrics: ReportedMetrics = ReportedMetrics(
                        has_objective=True,
                        sense=sense,
                        objective_value=objective_value,
                        dual_bound=dual_bound,
                        statistics=solver.statistics,
                    )
                    parameters: tuple[SolverParameter, ...] = tuple(
                        SolverParameter(name=name, value=value)
                        for name, value in solver.parameters.items()
                    )
                    metadata: ReportedMetadata = ReportedMetadata(
                        solver="fake",
                        parameters=parameters,
                    )
                    return SolveObservation(
                        api=self.api,
                        outcome=outcome,
                        metrics=metrics,
                        metadata=metadata,
                    )

                if self._observer is None:
                    return call()
                return self._observer.observe(self.api, call, extract)

            return observed

        self._patch.install(factory)

    def uninstall(self) -> None:
        """Restore the original fake solver call."""
        self._patch.uninstall()
        self._observer = None
