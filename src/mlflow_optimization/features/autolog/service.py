"""Activation of integrations and arbitration of observed solve calls."""

from collections.abc import Callable, Sequence
from contextvars import ContextVar, Token
from itertools import count
from threading import Lock
from time import perf_counter, time
from uuid import uuid4

from mlflow_optimization.common.diagnostics import warn

from .models import CompletedSolve, SolveObservation
from .ports import SolverIntegration, SolveSink


_solve_indices = count(1)
_solve_index_lock = Lock()
_observing: ContextVar[bool] = ContextVar("autolog_observing", default=False)


class AutologManager:
    """The observer that turns outermost solve calls into completed solves."""

    def __init__(self, sink: SolveSink) -> None:
        self._sink: SolveSink = sink
        self._integrations: list[SolverIntegration] = []
        self._enabled: bool = False

    def enable(self, integrations: Sequence[SolverIntegration]) -> None:
        """Observe exactly the given integrations."""
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
            context: object = None
            try:
                context = self._sink.begin()
            except BaseException:
                failures.append("Solve sink begin failed")
            with _solve_index_lock:
                solve_index: int = next(_solve_indices)
            solve_id: str = uuid4().hex
            start_time_ms: int = int(time() * 1000)
            start: float = perf_counter()
            try:
                result: T = call()
            except BaseException as error:
                end: float = perf_counter()
                end_time_ms: int = int(time() * 1000)
                self._finish(api, solve_id, solve_index, start_time_ms, end_time_ms, end - start, None, error, extract, context, failures)
                for failure in failures:
                    try:
                        warn(failure)
                    except BaseException:
                        pass
                raise
            end = perf_counter()
            end_time_ms = int(time() * 1000)
            self._finish(api, solve_id, solve_index, start_time_ms, end_time_ms, end - start, result, None, extract, context, failures)
            for failure in failures:
                warn(failure)
            return result
        finally:
            _observing.reset(token)

    def _finish[T](
        self, api: str, solve_id: str, solve_index: int, start_time_ms: int,
        end_time_ms: int, wall_time_seconds: float, result: T | None,
        error: BaseException | None,
        extract: Callable[[T | None, BaseException | None], SolveObservation],
        context: object, failures: list[str],
    ) -> None:
        """Extract and record a completed solve."""
        observation: SolveObservation | None = None
        try:
            observation = extract(result, error)
        except BaseException:
            failures.append("Solve extraction failed")
        solve: CompletedSolve = CompletedSolve(api, solve_id, solve_index, start_time_ms, end_time_ms, wall_time_seconds, observation, error)
        try:
            self._sink.record(solve, context)
        except BaseException:
            failures.append("Solve sink record failed")
