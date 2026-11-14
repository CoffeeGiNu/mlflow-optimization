"""Contract of the run storage."""

from collections.abc import Mapping
from typing import Protocol

from .models import RunRef, SolveRecord


class RunStoreError(Exception):
    """An error raised when run storage fails."""


class RunStore(Protocol):
    """The storage of solve runs in an experiment tracker."""

    def active_run(self) -> RunRef | None:
        """The user's currently active run."""
        ...

    def save(self, record: SolveRecord, parent: RunRef | None) -> RunRef:
        """Create, fill and terminate the run of one finished solve."""
        ...

    def log_metric(self, run: RunRef, key: str, value: float) -> None:
        """Add one metric to an existing run."""
        ...

    def set_tags(self, run: RunRef, tags: Mapping[str, str]) -> None:
        """Add tags to an existing run."""
        ...
