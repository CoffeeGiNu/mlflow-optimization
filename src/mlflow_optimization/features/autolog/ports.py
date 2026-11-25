"""Contracts between solve observation and its surroundings."""

from collections.abc import Callable
from typing import Protocol

from mlflow_optimization.features.artifacts.types import (
    ArtifactOptions,
    ArtifactWorkspace,
)
from mlflow_optimization.features.progress.types import ProgressHook, SampleSink

from .types import CompletedSolve, SolveObservation


class ObservationError(Exception):
    """An error raised when a solve observation cannot be built."""


class SolveObserver(Protocol):
    """The entry through which an adapter routes a solve call."""

    def observe[T](
        self,
        api: str,
        call: Callable[[ProgressHook | None], T],
        extract: Callable[[T | None, BaseException | None], SolveObservation],
    ) -> T:
        """The result of ``call(hook)``, observed as one solve unless nested or disabled."""
        ...

    def artifact_workspace(self) -> ArtifactWorkspace | None:
        """The workspace of the outermost solve observed in this context."""
        ...


class SolverIntegration(Protocol):
    """The hook of one Python solver API."""

    api: str

    def install(self, observer: SolveObserver) -> None:
        """Route the SDK's solve entry points through ``observer``."""
        ...

    def uninstall(self) -> None:
        """Restore the SDK's original solve entry points."""
        ...


class SolveSink(Protocol):
    """The destination of completed solves."""

    def begin(self) -> object:
        """The recording context captured at solve start."""
        ...

    def start(
        self,
        context: object,
        api: str,
        solve_id: str,
        solve_index: int,
        start_time_ms: int,
    ) -> SampleSink | None:
        """The sampler for an observed solve, if progress recording is active."""
        ...

    def record(self, solve: CompletedSolve, context: object) -> None:
        """Store one completed solve."""
        ...

    def artifact_options(self) -> ArtifactOptions:
        """The automatic artifacts requested for the next solve."""
        ...
