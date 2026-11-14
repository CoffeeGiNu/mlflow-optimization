"""A fake modeling-layer API for adapter tests."""

from collections.abc import Callable

from fakes import FakeSolver
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


class FakeFrontend:
    """A modeling layer that delegates solving to a fake solver."""

    def __init__(self, solver: FakeSolver) -> None:
        self.solver: FakeSolver = solver

    def solve(self) -> str:
        """The status returned by the underlying fake solver."""
        return self.solver.solve()


class Integration:
    """The adapter for the fake modeling-layer API."""

    api: str = "fake_frontend"

    def __init__(self) -> None:
        self._patch: Patch = Patch(FakeFrontend, "solve")
        self._observer: SolveObserver | None = None

    def install(self, observer: SolveObserver) -> None:
        """Route fake modeling-layer calls through the observer."""
        self._observer = observer

        def factory(
            original: Callable[[FakeFrontend], str],
        ) -> Callable[[FakeFrontend], str]:
            def observed(modeling_layer: FakeFrontend) -> str:
                def call() -> str:
                    return original(modeling_layer)

                def extract(
                    result: str | None, error: BaseException | None
                ) -> SolveObservation:
                    solver: FakeSolver = modeling_layer.solver
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
                        wall_time_includes_problem_build=True,
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
        """Restore the original fake modeling-layer call."""
        self._patch.uninstall()
        self._observer = None
