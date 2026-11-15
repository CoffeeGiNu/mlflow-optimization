"""Contract of the run storage."""

from collections.abc import Mapping, Sequence
from typing import Protocol

from .types import HistoryPoint, RunReference, SolveRecord


class RunStoreError(Exception):
    """An error raised when run storage fails."""


class RunStore(Protocol):
    """The storage of solve runs in an experiment tracker."""

    def active_run(self) -> RunReference | None:
        """The user's currently active run."""
        ...

    def start(
        self,
        name: str,
        start_time_ms: int,
        tags: Mapping[str, str],
        parent: RunReference | None,
    ) -> RunReference:
        """A newly started solve run."""
        ...

    def log_history(self, run: RunReference, points: Sequence[HistoryPoint]) -> None:
        """History points of an existing run."""
        ...

    def finish(
        self, run: RunReference, record: SolveRecord, *, final_step: int
    ) -> None:
        """The final record of an existing run."""
        ...

    def save(self, record: SolveRecord, parent: RunReference | None) -> RunReference:
        """Create, fill and terminate the run of one finished solve."""
        ...

    def log_metric(self, run: RunReference, key: str, value: float) -> None:
        """Add one metric to an existing run."""
        ...

    def set_tags(self, run: RunReference, tags: Mapping[str, str]) -> None:
        """Add tags to an existing run."""
        ...
