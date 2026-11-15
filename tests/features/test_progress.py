import itertools
import math
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Barrier, Event, Lock

import pytest

from mlflow_optimization.common.diagnostics import MlflowOptimizationWarning
from mlflow_optimization.features.progress.recorder import ProgressRecorder
from mlflow_optimization.features.progress.types import HistoryPoint


def test_samples_merge_partial_values_and_record_only_changes() -> None:
    points: list[HistoryPoint] = []

    def write_history(history_points: Sequence[HistoryPoint]) -> None:
        points.extend(history_points)

    recorder: ProgressRecorder = ProgressRecorder(write_history, interval_seconds=0.2)
    origin: float = time.perf_counter()
    recorder.sample(origin, incumbent=None, bound=None)
    origin = time.perf_counter()
    recorder.sample(origin, incumbent=10.0, bound=None)
    recorder.sample(origin, incumbent=None, bound=None)
    origin = time.perf_counter() - 0.02
    recorder.sample(origin, incumbent=None, bound=7.0)
    origin = time.perf_counter() - 0.03
    recorder.sample(origin, incumbent=10.0, bound=7.0)
    origin = time.perf_counter() - 0.04
    recorder.sample(origin, incumbent=math.nan, bound=math.inf)
    origin = time.perf_counter() - 0.05
    recorder.sample(origin, incumbent=12.0, bound=None)
    origin = time.perf_counter() - 0.06
    recorder.sample(origin, incumbent=None, bound=8.0)
    greatest_step: int | None = recorder.close()

    assert len(points) == 2
    assert points[0].values == {"objective_value": 10.0}
    assert set(points[1].values) == {
        "objective_value",
        "dual_bound",
        "absolute_gap",
        "relative_gap",
    }
    assert points[1].values["objective_value"] == 12.0
    assert points[1].values["dual_bound"] == 8.0
    assert points[1].values["absolute_gap"] == 4.0
    assert points[1].values["relative_gap"] == pytest.approx(1.0 / 3.0)
    steps: list[int] = [point.step for point in points]
    assert all(
        previous_step < next_step
        for previous_step, next_step in itertools.pairwise(steps)
    )
    assert greatest_step == max(point.step for point in points)


def test_burst_keeps_first_and_latest_values_until_close() -> None:
    points: list[HistoryPoint] = []

    def write_history(history_points: Sequence[HistoryPoint]) -> None:
        points.extend(history_points)

    recorder: ProgressRecorder = ProgressRecorder(write_history, interval_seconds=0.5)
    burst_sample_count: int = 40
    for sample_number in range(burst_sample_count):
        elapsed_seconds: float = sample_number * 0.002
        origin: float = time.perf_counter() - elapsed_seconds
        objective_value: float = float(sample_number)
        recorder.sample(origin, incumbent=objective_value, bound=objective_value - 1.0)
    greatest_step: int | None = recorder.close()

    assert len(points) == 2
    assert points[0].values["objective_value"] == 0.0
    assert points[-1].values["objective_value"] == float(burst_sample_count - 1)
    assert points[-1].values["dual_bound"] == float(burst_sample_count - 2)
    assert greatest_step == points[-1].step


def test_samples_after_interval_replace_pending_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_times = iter((0.0, 0.02, 0.05, 0.15, 0.16))
    monkeypatch.setattr(
        "mlflow_optimization.features.progress.recorder.perf_counter",
        lambda: next(sample_times),
    )
    points: list[HistoryPoint] = []
    recorder: ProgressRecorder = ProgressRecorder(points.extend, interval_seconds=60.0)
    for value in range(5):
        recorder.sample(0.0, incumbent=float(value), bound=float(value) - 1.0)
    greatest_step: int | None = recorder.close()

    assert [point.values["objective_value"] for point in points] == [0.0, 3.0, 4.0]
    assert [point.step for point in points] == [0, 150, 160]
    assert greatest_step == 160


@pytest.mark.parametrize(
    ("bound", "expected_absolute_gap", "expected_relative_gap"),
    ((0.0, 0.0, 0.0), (3.0, 3.0, None)),
)
def test_zero_objective_omits_undefined_relative_gap(
    bound: float,
    expected_absolute_gap: float,
    expected_relative_gap: float | None,
) -> None:
    points: list[HistoryPoint] = []

    def write_history(history_points: Sequence[HistoryPoint]) -> None:
        points.extend(history_points)

    recorder: ProgressRecorder = ProgressRecorder(write_history, interval_seconds=0.2)
    recorder.sample(time.perf_counter(), incumbent=0.0, bound=bound)
    recorder.close()

    assert len(points) == 1
    assert points[0].values["absolute_gap"] == expected_absolute_gap
    if expected_relative_gap is None:
        assert "relative_gap" not in points[0].values
    else:
        assert points[0].values["relative_gap"] == expected_relative_gap


def test_failed_history_write_is_retried_with_one_warning() -> None:
    attempts: list[tuple[HistoryPoint, ...]] = []
    retry_succeeded: Event = Event()
    failure_count: int = 2

    def write_history(history_points: Sequence[HistoryPoint]) -> None:
        batch: tuple[HistoryPoint, ...] = tuple(history_points)
        attempts.append(batch)
        if len(attempts) <= failure_count:
            raise RuntimeError("temporary tracking failure")
        retry_succeeded.set()

    recorder: ProgressRecorder = ProgressRecorder(write_history, interval_seconds=0.05)
    recorder.sample(time.perf_counter(), incumbent=5.0, bound=4.0)
    with pytest.warns(MlflowOptimizationWarning) as warning_records:
        retried: bool = retry_succeeded.wait(timeout=0.4)
    greatest_step: int | None = recorder.close()

    assert retried
    assert greatest_step is not None
    assert len(attempts) >= failure_count + 1
    assert all(batch == attempts[0] for batch in attempts[: failure_count + 1])
    assert len(warning_records) == 1


def test_close_flushes_once_returns_last_step_and_ignores_later_samples() -> None:
    writes: list[tuple[HistoryPoint, ...]] = []

    def write_history(history_points: Sequence[HistoryPoint]) -> None:
        writes.append(tuple(history_points))

    empty_recorder: ProgressRecorder = ProgressRecorder(
        write_history, interval_seconds=0.2
    )
    assert empty_recorder.close() is None
    assert empty_recorder.close() is None
    assert writes == []

    recorder: ProgressRecorder = ProgressRecorder(write_history, interval_seconds=0.2)
    origin: float = time.perf_counter() - 0.05
    wall_time_before_ms: int = time.time_ns() // 1_000_000
    recorder.sample(origin, incumbent=2.0, bound=1.0)
    greatest_step: int | None = recorder.close()
    wall_time_after_ms: int = time.time_ns() // 1_000_000
    write_count: int = len(writes)
    recorder.sample(time.perf_counter(), incumbent=3.0, bound=2.0)
    repeated_step: int | None = recorder.close()
    points: list[HistoryPoint] = [point for batch in writes for point in batch]

    assert greatest_step is not None
    assert repeated_step == greatest_step
    assert len(points) == 1
    assert points[0].step == greatest_step
    assert points[0].step >= 40
    assert wall_time_before_ms <= points[0].timestamp_ms <= wall_time_after_ms
    assert len(writes) == write_count


def test_concurrent_samples_produce_ordered_unique_points() -> None:
    worker_count: int = 4
    samples_per_worker: int = 8
    barrier: Barrier = Barrier(worker_count)
    batches: list[tuple[HistoryPoint, ...]] = []
    batches_lock: Lock = Lock()
    expected_pairs: set[tuple[float, float]] = {
        (float(sample_number), float(sample_number) - 1.0)
        for sample_number in range(worker_count * samples_per_worker)
    }

    def write_history(history_points: Sequence[HistoryPoint]) -> None:
        batch: tuple[HistoryPoint, ...] = tuple(history_points)
        with batches_lock:
            batches.append(batch)

    recorder: ProgressRecorder = ProgressRecorder(write_history, interval_seconds=0.2)
    origin: float = time.perf_counter()

    def sample_values(worker_index: int) -> None:
        barrier.wait()
        for sample_index in range(samples_per_worker):
            sample_number: int = worker_index * samples_per_worker + sample_index
            objective_value: float = float(sample_number)
            dual_bound: float = objective_value - 1.0
            recorder.sample(origin, incumbent=objective_value, bound=dual_bound)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures: list[Future[None]] = [
            executor.submit(sample_values, worker_index)
            for worker_index in range(worker_count)
        ]
        for future in futures:
            future.result()
    greatest_step: int | None = recorder.close()
    points: list[HistoryPoint] = [point for batch in batches for point in batch]
    steps: list[int] = [point.step for point in points]
    actual_pairs: set[tuple[float | None, float | None]] = {
        (point.values.get("objective_value"), point.values.get("dual_bound"))
        for point in points
    }

    assert points
    assert all(
        previous_step < next_step
        for previous_step, next_step in itertools.pairwise(steps)
    )
    assert len(points) == len(actual_pairs)
    assert actual_pairs.issubset(expected_pairs)
    assert greatest_step == max(steps)
