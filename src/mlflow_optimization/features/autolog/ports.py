"""Contracts between solve observation and its surroundings."""

from collections.abc import Callable
from typing import Protocol

from .types import CompletedSolve, SolveObservation


class ObservationError(Exception):
    """An error raised when a solve observation cannot be built."""


class SolveObserver(Protocol):
    """The entry through which an adapter routes a solve call."""

    def observe[T](
        self,
        api: str,
        call: Callable[[], T],
        extract: Callable[[T | None, BaseException | None], SolveObservation],
    ) -> T:
        """The result of ``call()``, observed as one solve unless nested or disabled."""
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

    def record(self, solve: CompletedSolve, context: object) -> None:
        """Store one completed solve."""
        ...
