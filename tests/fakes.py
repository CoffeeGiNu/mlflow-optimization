"""A fake solver API for adapter tests."""

from collections.abc import Callable, Sequence
from pathlib import Path

from mlflow_optimization.common.patching import Patch
from mlflow_optimization.features.artifacts.types import (
    ArtifactRequest,
    ArtifactWorkspace,
)
from mlflow_optimization.features.autolog.ports import SolveObserver
from mlflow_optimization.features.autolog.types import SolveObservation
from mlflow_optimization.features.metadata.types import (
    ReportedMetadata,
    SolverParameter,
)
from mlflow_optimization.features.metrics.types import (
    MissingReason,
    ObjectiveSense,
    ReportedMetrics,
)
from mlflow_optimization.features.outcomes.types import ReportedOutcome, Termination
from mlflow_optimization.features.progress.types import ProgressHook


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
        during_solve: Callable[[], None] | None = None,
        progress_samples: Sequence[tuple[float | None, float | None]] = (),
        artifact_text: str | None = None,
        artifact_directory: str = "solver",
        diagnostics: dict[str, str] | None = None,
    ) -> None:
        self.status: str | Termination = status
        self.has_solution: bool = has_solution
        self.objective: float | None = objective
        self.bound: float | None = bound
        self.sense: str | ObjectiveSense = sense
        self.parameters: dict[str, int | float | str | bool] = (
            parameters.copy() if parameters is not None else {}
        )
        self.statistics: dict[str, float] = (
            statistics.copy() if statistics is not None else {}
        )
        self.raises: BaseException | None = raises
        self.during_solve: Callable[[], None] | None = during_solve
        self.progress_samples: tuple[tuple[float | None, float | None], ...] = tuple(
            progress_samples
        )
        self.artifact_text: str | None = artifact_text
        self.artifact_directory: str = artifact_directory
        self.diagnostics: dict[str, str] = (
            diagnostics.copy() if diagnostics is not None else {}
        )
        self.seen_workspace: ArtifactWorkspace | None = None
        self.written_path: Path | None = None

    def solve(self, hook: ProgressHook | None = None) -> str:
        """The configured solver termination status."""
        if hook is not None:
            for incumbent_value, bound_value in self.progress_samples:
                hook(incumbent_value, bound_value)
        if self.during_solve is not None:
            self.during_solve()
        if self.raises is not None:
            raise self.raises
        if isinstance(self.status, Termination):
            return self.status.value
        return self.status


class FakeSolverAdapter:
    """The adapter for the fake solver API."""

    api: str = "fake"

    def __init__(self) -> None:
        self._patch: Patch = Patch(FakeSolver, "solve")
        self._observer: SolveObserver | None = None

    def install(self, observer: SolveObserver) -> None:
        """Route fake solver calls through the observer."""
        self._observer = observer

        def factory(
            original: Callable[..., str],
        ) -> Callable[[FakeSolver], str]:
            def observed(solver: FakeSolver) -> str:
                def call(hook: ProgressHook | None) -> str:
                    workspace: ArtifactWorkspace | None = (
                        self._observer.artifact_workspace()
                        if self._observer is not None
                        else None
                    )
                    solver.seen_workspace = workspace
                    if workspace is not None and solver.artifact_text is not None:
                        written_path: Path = workspace.directory / "fake.log"
                        written_path.write_text(solver.artifact_text, encoding="utf-8")
                        solver.written_path = written_path
                    return original(solver, hook)

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
                        solver.bound
                        if solver.bound is not None
                        else MissingReason.UNSUPPORTED
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
                    artifacts: tuple[ArtifactRequest, ...] = (
                        (
                            ArtifactRequest(
                                local_path=str(solver.written_path),
                                artifact_directory=solver.artifact_directory,
                            ),
                        )
                        if solver.written_path is not None
                        else ()
                    )
                    return SolveObservation(
                        api=self.api,
                        outcome=outcome,
                        metrics=metrics,
                        metadata=metadata,
                        artifacts=artifacts,
                        diagnostics=solver.diagnostics,
                    )

                if self._observer is None:
                    return call(None)
                return self._observer.observe(self.api, call, extract)

            return observed

        self._patch.install(factory)

    def uninstall(self) -> None:
        """Restore the original fake solver call."""
        self._patch.uninstall()
        self._observer = None
