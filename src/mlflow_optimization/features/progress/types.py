"""Types for recording solve progress."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

ProgressHook = Callable[[float | None, float | None], None]
"""A callback for incumbent and bound values."""

SampleSink = Callable[[float, float | None, float | None], None]
"""A callback for a monotonic origin in seconds, incumbent, and bound."""


@dataclass(frozen=True)
class HistoryPoint:
    """A history point with elapsed and wall-clock milliseconds and metric values."""

    step: int
    timestamp_ms: int
    values: Mapping[str, float]
