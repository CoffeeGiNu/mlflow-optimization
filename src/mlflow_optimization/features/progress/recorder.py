"""Solve progress recording."""

from collections.abc import Callable, Sequence
from math import isfinite
from threading import Event, Lock, Thread
from time import perf_counter, time

from mlflow_optimization.common import diagnostics
from mlflow_optimization.features.metrics.normalize import calculate_gaps
from mlflow_optimization.features.progress.types import HistoryPoint

HISTORY_INTERVAL_SECONDS: float = 3.0
MINIMUM_POINT_INTERVAL_SECONDS: float = 0.1


class ProgressRecorder:
    """Records solve progress history."""

    def __init__(
        self,
        write_history: Callable[[Sequence[HistoryPoint]], None],
        *,
        interval_seconds: float = HISTORY_INTERVAL_SECONDS,
    ) -> None:
        self._write_history: Callable[[Sequence[HistoryPoint]], None] = write_history
        self._interval_seconds: float = interval_seconds
        self._lock: Lock = Lock()
        self._stop_event: Event = Event()
        # shortcut: retries retain history in memory; cap retention for long tracker outages.
        self._buffer: list[HistoryPoint] = []
        self._incumbent: float | None = None
        self._bound: float | None = None
        self._last_recorded_elapsed: float | None = None
        self._pending_sample: tuple[float, int, float | None, float | None] | None = (
            None
        )
        self._greatest_step: int | None = None
        self._closed: bool = False
        self._warned_write_failure: bool = False
        self._worker: Thread = Thread(target=self._run, daemon=True)
        self._worker.start()

    def sample(
        self,
        origin: float,
        incumbent: float | None,
        bound: float | None,
    ) -> None:
        """A sample of solver progress."""
        try:
            with self._lock:
                if self._closed:
                    return
                elapsed: float = perf_counter() - origin
                timestamp_ms: int = int(time() * 1000)
                incumbent_value: float | None = (
                    incumbent if incumbent is not None and isfinite(incumbent) else None
                )
                bound_value: float | None = (
                    bound if bound is not None and isfinite(bound) else None
                )
                merged_incumbent: float | None = (
                    incumbent_value if incumbent_value is not None else self._incumbent
                )
                merged_bound: float | None = (
                    bound_value if bound_value is not None else self._bound
                )
                if (merged_incumbent is None and merged_bound is None) or (
                    merged_incumbent == self._incumbent and merged_bound == self._bound
                ):
                    return
                if (
                    self._last_recorded_elapsed is None
                    or elapsed - self._last_recorded_elapsed
                    >= MINIMUM_POINT_INTERVAL_SECONDS
                ):
                    self._pending_sample = None
                    self._append_point(
                        elapsed,
                        timestamp_ms,
                        merged_incumbent,
                        merged_bound,
                    )
                else:
                    self._pending_sample = (
                        elapsed,
                        timestamp_ms,
                        merged_incumbent,
                        merged_bound,
                    )
                self._incumbent = merged_incumbent
                self._bound = merged_bound
        except Exception:  # noqa: BLE001
            # NOTE: Solver callbacks must not fail because progress recording failed.
            return

    def close(self) -> int | None:
        """The greatest sampled step or None."""
        with self._lock:
            already_closed: bool = self._closed
            self._closed = True
        if not already_closed:
            self._stop_event.set()
            # shortcut: shutdown waits for tracking I/O; add cancellation for bounded shutdown.
            self._worker.join()
            points: tuple[HistoryPoint, ...] = self._take_buffer()
            if points:
                try:
                    self._write_history(points)
                except Exception:  # noqa: BLE001
                    # NOTE: A failed final write must not affect the solve result.
                    self._warn_write_failure()
        with self._lock:
            return self._greatest_step

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            points: tuple[HistoryPoint, ...] = self._take_buffer()
            if not points:
                continue
            try:
                self._write_history(points)
            except Exception:  # noqa: BLE001
                # NOTE: A failed batch must be retried without stopping the worker.
                self._restore_buffer(points)
                self._warn_write_failure()

    def _take_buffer(self) -> tuple[HistoryPoint, ...]:
        with self._lock:
            if self._pending_sample is not None:
                self._append_point(*self._pending_sample)
                self._pending_sample = None
            points: tuple[HistoryPoint, ...] = tuple(self._buffer)
            self._buffer.clear()
            return points

    def _append_point(
        self,
        elapsed: float,
        timestamp_ms: int,
        incumbent: float | None,
        bound: float | None,
    ) -> None:
        step: int = int(elapsed * 1000)
        if self._greatest_step is not None:
            step = max(step, self._greatest_step + 1)
        values: dict[str, float] = {}
        if incumbent is not None:
            values["objective_value"] = incumbent
        if bound is not None:
            values["dual_bound"] = bound
        if incumbent is not None and bound is not None:
            absolute_gap: float | None
            relative_gap: float | None
            absolute_gap, relative_gap = calculate_gaps(incumbent, bound)
            if absolute_gap is not None:
                values["absolute_gap"] = absolute_gap
            if relative_gap is not None:
                values["relative_gap"] = relative_gap
        point: HistoryPoint = HistoryPoint(
            step=step,
            timestamp_ms=timestamp_ms,
            values=values,
        )
        self._buffer.append(point)
        self._greatest_step = step
        self._last_recorded_elapsed = elapsed

    def _restore_buffer(self, points: Sequence[HistoryPoint]) -> None:
        with self._lock:
            self._buffer[0:0] = points

    def _warn_write_failure(self) -> None:
        with self._lock:
            if self._warned_write_failure:
                return
            self._warned_write_failure = True
        diagnostics.warn("Failed to write solve progress history.")
