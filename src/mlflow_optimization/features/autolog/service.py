"""Activation of integrations and arbitration of observed solve calls."""

import warnings
from collections.abc import Callable, Iterator, Sequence
from contextvars import ContextVar, Token
from dataclasses import replace
from itertools import count
from threading import Lock
from time import perf_counter, time
from typing import cast
from uuid import uuid4

from mlflow_optimization.common.diagnostics import MlflowOptimizationWarning, warn

from .models import CompletedSolve, SolveObservation
from .ports import ObservationError, SolverIntegration, SolveSink


_solve_indices: Iterator[int] = count(1)
_solve_index_lock: Lock = Lock()
_observing: ContextVar[bool] = ContextVar("autolog_observing", default=False)


class AutologManager:
    """The observer that turns outermost solve calls into completed solves."""

    def __init__(self, sink: SolveSink) -> None:
        self._sink: SolveSink = sink
        self._integrations: list[SolverIntegration] = []
        self._enabled: bool = False

    def enable(self, integrations: Sequence[SolverIntegration]) -> None:
        """Observe exactly the given integrations."""
        integration: SolverIntegration
        for integration in self._integrations[:]:
            if not any(integration is candidate for candidate in integrations):
                integration.uninstall()
                self._integrations.remove(integration)
        for integration in integrations:
            if not any(integration is installed for installed in self._integrations):
                integration.install(self)
                self._integrations.append(integration)
        self._enabled = True

    def disable(self) -> None:
        """Stop observing and uninstall every integration."""
        self._enabled = False
        integration: SolverIntegration
        for integration in self._integrations[:]:
            integration.uninstall()
            self._integrations.remove(integration)

    def observe[T](
        self,
        api: str,
        call: Callable[[], T],
        extract: Callable[[T | None, BaseException | None], SolveObservation],
    ) -> T:
        """The result of ``call()``, observed as one solve unless nested or disabled."""
        if not self._enabled or _observing.get():
            return call()
        token: Token[bool] = _observing.set(True)
        try:
            failures: list[str] = []

            def guard(operation: Callable[[], object]) -> object:
                """The result of a guarded observation operation."""
                try:
                    return operation()
                # NOTE: A defect in logging must never change the result of the user's solve.
                except Exception:
                    failures.append("Solve observation failed")
                    return None

            context: object = guard(self._sink.begin)
            with _solve_index_lock:
                solve_index: int = next(_solve_indices)
            solve_id: str = uuid4().hex
            start_time_ms: int = int(time() * 1000)
            result: T | None = None

            def finish(solve: CompletedSolve) -> None:
                """Extract and record a completed solve."""
                observation: SolveObservation | None = None
                try:
                    observation = extract(result, solve.error)
                except ObservationError:
                    failures.append("Solve extraction failed")
                self._sink.record(replace(solve, observation=observation), context)

            start: float = perf_counter()
            try:
                result = call()
            except BaseException as error:
                end: float = perf_counter()
                end_time_ms: int = int(time() * 1000)
                solve: CompletedSolve = CompletedSolve(
                    api, solve_id, solve_index, start_time_ms, end_time_ms,
                    end - start, None, error,
                )
                guard(lambda: finish(solve))
                with warnings.catch_warnings():
                    warnings.simplefilter("always", MlflowOptimizationWarning)
                    failure: str
                    for failure in failures:
                        warn(failure)
                raise
            end = perf_counter()
            end_time_ms = int(time() * 1000)
            solve = CompletedSolve(
                api, solve_id, solve_index, start_time_ms, end_time_ms,
                end - start, None, None,
            )
            guard(lambda: finish(solve))
            for failure in failures:
                warn(failure)
            return cast(T, result)
        finally:
            _observing.reset(token)
